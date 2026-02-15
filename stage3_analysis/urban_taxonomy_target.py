#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Urban Taxonomy Target Preparation for Linear Probing

Prepares Urban Taxonomy HiMoC (Hierarchical Morphotope Classification) as
categorical targets aligned to H3 hexagons via area-weighted majority vote.
Morphotope polygons are intersected with H3 hexagons; for each hexagon the
dominant class (largest intersection area) is selected at each hierarchy level.

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
    type_level{N}       — dominant classification label (int)
    confidence_level{N} — area fraction of dominant type (0–1)
    name_level{N}       — human-readable name (levels 1–3 only)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import make_valid

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
    h3_resolutions: List[int] = field(default_factory=lambda: [9, 10])
    levels: List[int] = field(default_factory=lambda: list(ALL_LEVELS))
    chunk_size: int = 50_000  # morphotopes per overlay chunk

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
    area-weighted majority vote from morphotope polygons.

    Pipeline:
        1. Load & concatenate 4 NL morphotope parquets
        2. Load pre-computed H3 regions
        3. Spatial pre-filter (sjoin intersects)
        4. Chunked overlay for accurate intersection geometry
        5. Area-weighted majority vote per hexagon per level
        6. Attach human-readable names for levels 1–3
        7. Save as parquet indexed by region_id
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
        """Load and concatenate all 4 NL morphotope parquet files."""
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
        logger.info(f"Source CRS: {morphotopes.crs.to_epsg()}")

        # Reproject to WGS84 for H3 compatibility
        logger.info("Reprojecting EPSG:3035 -> EPSG:4326...")
        morphotopes = morphotopes.to_crs(epsg=4326)

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

    def _majority_vote(
        self, intersection: gpd.GeoDataFrame, level_cols: List[str]
    ) -> pd.DataFrame:
        """
        Area-weighted majority vote: for each hexagon, pick the class
        with the largest total intersection area at each hierarchy level.

        Returns DataFrame indexed by region_id with type_level* and
        confidence_level* columns.
        """
        # Compute intersection areas in the source equal-area CRS
        logger.info("  Computing intersection areas (EPSG:3035)...")
        intersection_proj = intersection.to_crs(epsg=3035)
        intersection["area"] = intersection_proj.geometry.area

        results = {}
        for col in level_cols:
            level_num = col.replace("level_", "").replace("_label", "")
            logger.info(f"  Majority vote for level {level_num}...")

            # Sum area per (region_id, class) pair
            area_by_class = (
                intersection.groupby(["region_id", col])["area"]
                .sum()
                .reset_index()
            )

            # Total area per hexagon (for confidence calculation)
            total_area = area_by_class.groupby("region_id")["area"].sum()

            # Pick class with max area per hexagon
            idx_max = area_by_class.groupby("region_id")["area"].idxmax()
            dominant = area_by_class.loc[idx_max].set_index("region_id")

            results[f"type_level{level_num}"] = dominant[col].astype(int)
            results[f"confidence_level{level_num}"] = (
                dominant["area"] / total_area
            )

        return pd.DataFrame(results)

    def process_resolution(self, resolution: int) -> pd.DataFrame:
        """Process morphotopes -> H3 targets for a single resolution."""
        if self.morphotopes is None:
            self.load_morphotopes()

        regions_gdf = self._get_h3_regions(resolution)
        morphotopes = self.morphotopes

        level_cols = [f"level_{l}_label" for l in self.config.levels]
        keep_cols = ["geometry"] + level_cols

        # --- Spatial pre-filter: find hexagons that intersect morphotopes ---
        logger.info("Finding hexagons that intersect morphotopes (sjoin)...")
        hex_with_data = gpd.sjoin(
            regions_gdf.reset_index()[["region_id", "geometry"]],
            morphotopes[["geometry"]],
            how="inner",
            predicate="intersects",
        )
        relevant_hex_ids = hex_with_data["region_id"].unique()
        regions_relevant = regions_gdf.loc[
            regions_gdf.index.isin(relevant_hex_ids)
        ].copy()
        logger.info(f"  Hexagons with morphotope overlap: {len(regions_relevant):,}")

        # Fix invalid geometries
        morphotopes = morphotopes.copy()
        morphotopes["geometry"] = morphotopes["geometry"].apply(make_valid)
        regions_relevant["geometry"] = regions_relevant["geometry"].apply(make_valid)

        # --- Chunked overlay ---
        chunk_size = self.config.chunk_size
        n_chunks = max(1, (len(morphotopes) + chunk_size - 1) // chunk_size)
        logger.info(f"Processing overlay in {n_chunks} chunks of {chunk_size:,} morphotopes...")

        all_intersections = []
        regions_reset = regions_relevant.reset_index()[["region_id", "geometry"]]

        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(morphotopes))
            chunk = morphotopes.iloc[start:end]

            logger.info(f"  Chunk {i + 1}/{n_chunks}: morphotopes {start:,}-{end:,}...")

            try:
                intersection = gpd.overlay(
                    regions_reset,
                    chunk[keep_cols],
                    how="intersection",
                )
                if len(intersection) > 0:
                    all_intersections.append(intersection)
            except Exception as e:
                logger.warning(f"  Chunk {i + 1} overlay error: {e}")

        if not all_intersections:
            raise ValueError(
                "No intersections found between morphotopes and H3 hexagons."
            )

        intersection = pd.concat(all_intersections, ignore_index=True)
        intersection = gpd.GeoDataFrame(intersection, crs="EPSG:4326")
        logger.info(f"Total intersection pieces: {len(intersection):,}")

        # --- Majority vote aggregation ---
        logger.info("Aggregating area-weighted majority vote per hexagon...")
        result = self._majority_vote(intersection, level_cols)

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
