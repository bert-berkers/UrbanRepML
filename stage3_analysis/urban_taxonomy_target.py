#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Urban Taxonomy Target Preparation for Linear Probing

Prepares Urban Taxonomy HiMoC (Hierarchical Morphotope Classification) as
categorical targets aligned to H3 hexagons via centroid-based point-in-polygon
assignment. Each hexagon's centroid is matched to the morphotope polygon it
falls within, giving a single unambiguous class at each hierarchy level.

This is ~30-60x faster than area-weighted overlay because morphotopes
(median 52K m²) are much larger than res10 hexagons (15K m²), so centroid
assignment is accurate for 99%+ of hexagons. An ``n_morphotopes`` column
tracks how many morphotopes each hexagon polygon actually intersects, serving
as a boundary confidence indicator (1 = fully interior, 2+ = boundary hex).

Source: https://urbantaxonomy.org/
Data version: v202511 (morphotopes), v202509 (label metadata)

Hierarchy (7 levels, doubling cardinality):
    level 1: 2 classes   (Incoherent / Coherent Fabric)
    level 2: 4 classes
    level 3: 8 classes   (named in label_name.json)
    level 4: 16 classes
    level 5: 25 classes
    level 6: 52 classes
    level 7: 101 classes

Target variables per hexagon:
    type_level{N}       — classification label (int)
    name_level{N}       — human-readable name (levels 1–3 only)
    n_morphotopes       — number of morphotopes the hex polygon intersects
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd

from utils import StudyAreaPaths

logger = logging.getLogger(__name__)

# All hierarchy levels available in the morphotope data
ALL_LEVELS = [1, 2, 3, 4, 5, 6, 7]

# Levels 1–3 have human-readable names in label_name.json
NAMED_LEVELS = [1, 2, 3]


@dataclass
class UrbanTaxonomyConfig:
    """Configuration for Urban Taxonomy target preparation."""

    study_area: str = "netherlands"
    year: int = 2025
    h3_resolutions: List[int] = field(default_factory=lambda: [10])
    levels: List[int] = field(default_factory=lambda: list(ALL_LEVELS))

    # Paths (auto-populated from StudyAreaPaths)
    raw_dir: Optional[str] = None
    output_dir: Optional[str] = None

    def __post_init__(self):
        paths = StudyAreaPaths(self.study_area)
        if self.raw_dir is None:
            self.raw_dir = str(paths.target("urban_taxonomy") / "raw")
        if self.output_dir is None:
            self.output_dir = str(paths.target("urban_taxonomy"))


class UrbanTaxonomyTargetBuilder:
    """
    Builds H3-indexed Urban Taxonomy classification targets via
    centroid-based point-in-polygon assignment from morphotope polygons.

    Pipeline:
        1. Load & concatenate 4 NL morphotope parquets (keep EPSG:3035)
        2. Load pre-computed H3 regions (WGS84), reproject to EPSG:3035
        3. Compute hex centroids in EPSG:3035
        4. sjoin centroids within morphotopes (~1s)
        5. Deduplicate (centroid on boundary → take first match)
        6. Count n_morphotopes per hex (polygon sjoin for boundary indicator)
        7. Attach human-readable names for levels 1–3
        8. Save as parquet indexed by region_id
    """

    PARQUET_FILES = [
        "nl1_morphotopes.parquet",
        "nl2_morphotopes.parquet",
        "nl3_morphotopes.parquet",
        "nl4_morphotopes.parquet",
    ]

    def __init__(self, config: UrbanTaxonomyConfig, project_root: Optional[Path] = None):
        self.config = config
        self.project_root = project_root or Path(__file__).parent.parent
        self.paths = StudyAreaPaths(config.study_area, self.project_root)
        self.morphotopes: Optional[gpd.GeoDataFrame] = None
        self.label_names: Optional[dict] = None

    def load_morphotopes(self) -> gpd.GeoDataFrame:
        """Load and concatenate all 4 NL morphotope parquet files.

        Keeps native EPSG:3035 CRS (no reprojection needed — hexes are
        reprojected to match instead).
        """
        raw_dir = Path(self.config.raw_dir)
        parts = []
        for fname in self.PARQUET_FILES:
            fpath = raw_dir / fname
            logger.info(f"Loading {fpath.name}...")
            gdf = gpd.read_parquet(fpath)
            parts.append(gdf)
            logger.info(f"  {len(gdf):,} morphotopes")

        morphotopes = pd.concat(parts, ignore_index=True)
        morphotopes = gpd.GeoDataFrame(morphotopes, crs=parts[0].crs)
        logger.info(f"Total morphotopes: {len(morphotopes):,}")
        logger.info(f"CRS: {morphotopes.crs.to_epsg()}")

        self.morphotopes = morphotopes
        return morphotopes

    def load_label_names(self) -> dict:
        """Load label_name.json mapping {level: {label: name}}."""
        label_path = Path(self.config.raw_dir) / "label_name.json"
        if label_path.exists():
            with open(label_path) as f:
                self.label_names = json.load(f)
            logger.info(f"Loaded label names for {len(self.label_names)} levels")
        else:
            logger.warning(f"label_name.json not found at {label_path}")
            self.label_names = {}
        return self.label_names

    def _get_h3_regions(self, resolution: int) -> gpd.GeoDataFrame:
        """Load pre-computed H3 regions for a given resolution."""
        regions_path = self.paths.region_file(resolution)
        logger.info(f"Loading H3 res{resolution} regions from {regions_path}")
        regions_gdf = gpd.read_parquet(regions_path)

        # Normalize index name
        if regions_gdf.index.name != "region_id":
            if regions_gdf.index.name in ("hex_id", "h3_index"):
                regions_gdf.index.name = "region_id"
            elif "region_id" in regions_gdf.columns:
                regions_gdf = regions_gdf.set_index("region_id")

        logger.info(f"  {len(regions_gdf):,} hexagons")
        return regions_gdf

    def process_resolution(self, resolution: int) -> pd.DataFrame:
        """Process morphotopes -> H3 targets for a single resolution.

        Uses centroid-based point-in-polygon assignment (fast) instead of
        area-weighted overlay (slow). Morphotopes are much larger than
        hexagons, so centroid assignment is accurate for 99%+ of cases.
        """
        if self.morphotopes is None:
            self.load_morphotopes()

        regions_gdf = self._get_h3_regions(resolution)
        morphotopes = self.morphotopes

        level_cols = [f"level_{l}_label" for l in self.config.levels]

        # --- Reproject hexagons to EPSG:3035 to match morphotopes ---
        logger.info("Reprojecting hexagons EPSG:4326 -> EPSG:3035...")
        regions_3035 = regions_gdf.to_crs(epsg=3035)

        # --- Compute centroids for point-in-polygon assignment ---
        logger.info("Computing hex centroids...")
        centroids = regions_3035.copy()
        centroids["geometry"] = centroids.geometry.centroid

        # --- Point-in-polygon sjoin: centroid within morphotope ---
        logger.info("Spatial join: hex centroids within morphotopes...")
        joined = gpd.sjoin(
            centroids.reset_index()[["region_id", "geometry"]],
            morphotopes[["geometry"] + level_cols],
            how="inner",
            predicate="within",
        )
        logger.info(f"  Raw join matches: {len(joined):,}")

        # Deduplicate: centroid exactly on boundary may match multiple
        # morphotopes — keep the first match
        joined = joined.drop_duplicates(subset="region_id", keep="first")
        joined = joined.set_index("region_id")
        logger.info(f"  After dedup: {len(joined):,} hexagons with assignment")

        # --- Count n_morphotopes per hex (polygon-level intersection) ---
        logger.info("Counting morphotopes per hex polygon (boundary indicator)...")
        poly_join = gpd.sjoin(
            regions_3035.reset_index()[["region_id", "geometry"]],
            morphotopes[["geometry"]],
            how="inner",
            predicate="intersects",
        )
        n_morphotopes = (
            poly_join.groupby("region_id")
            .size()
            .rename("n_morphotopes")
        )
        logger.info(
            f"  n_morphotopes distribution: "
            f"1={int((n_morphotopes == 1).sum()):,}, "
            f"2+={int((n_morphotopes >= 2).sum()):,}"
        )

        # --- Build result DataFrame ---
        result = pd.DataFrame(index=joined.index)
        for col in level_cols:
            level_num = col.replace("level_", "").replace("_label", "")
            result[f"type_level{level_num}"] = joined[col].astype(int)

        # Attach n_morphotopes
        result = result.join(n_morphotopes)
        # Hexagons with centroid match but possibly no polygon overlap count
        # shouldn't happen, but fill with 1 defensively
        result["n_morphotopes"] = result["n_morphotopes"].fillna(1).astype(int)

        # --- Attach human-readable names for levels 1–3 ---
        if self.label_names is None:
            self.load_label_names()

        for level in NAMED_LEVELS:
            if level not in self.config.levels:
                continue
            level_key = str(level)
            if level_key in self.label_names:
                name_map = {
                    int(k): v for k, v in self.label_names[level_key].items()
                }
                result[f"name_level{level}"] = (
                    result[f"type_level{level}"].map(name_map)
                )

        logger.info(f"Result: {len(result):,} hexagons with classifications")

        # Log class distribution for level 1
        if "type_level1" in result.columns:
            dist = result["type_level1"].value_counts()
            logger.info(f"  Level 1 distribution:\n{dist.to_string()}")

        return result

    def save(self, result: pd.DataFrame, resolution: int) -> Path:
        """Save target data as parquet indexed by region_id."""
        out = self.paths.target_file("urban_taxonomy", resolution, self.config.year)
        out.parent.mkdir(parents=True, exist_ok=True)

        result.to_parquet(out)

        logger.info(f"Saved to {out}")
        logger.info(f"  Shape: {result.shape}")
        logger.info(f"  Index: {result.index.name}")
        return out

    def run(self) -> dict:
        """Run the full pipeline for all configured resolutions."""
        logger.info("=== Urban Taxonomy Target Preparation ===")
        logger.info(f"Study area: {self.config.study_area}")
        logger.info(f"Resolutions: {self.config.h3_resolutions}")
        logger.info(f"Levels: {self.config.levels}")

        self.load_morphotopes()
        self.load_label_names()

        outputs = {}
        for res in self.config.h3_resolutions:
            logger.info(f"\n--- Processing H3 resolution {res} ---")
            result = self.process_resolution(res)
            path = self.save(result, res)
            outputs[res] = path

        logger.info("\n=== Target preparation complete ===")
        return outputs


def main():
    """Run Urban Taxonomy target preparation with default config."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = UrbanTaxonomyConfig()
    builder = UrbanTaxonomyTargetBuilder(config)
    builder.run()


if __name__ == "__main__":
    main()
