#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Leefbaarometer Target Preparation for Linear Probing

Prepares Dutch liveability (leefbaarometer) scores as regression targets
aligned to H3 hexagons via area-weighted spatial join. The leefbaarometer
100m grid cells are intersected with H3 res10 hexagons, and scores are
aggregated as area-weighted means per hexagon.

Target variables:
    lbm - overall liveability score
    fys - physical environment (fysieke omgeving)
    onv - safety / nuisance (onveiligheid)
    soc - social cohesion (sociale samenhang)
    vrz - amenities / facilities (voorzieningen)
    won - housing quality (woningen)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from srai.regionalizers import H3Regionalizer
from shapely.validation import make_valid

from utils import StudyAreaPaths
from utils.paths import write_run_info

logger = logging.getLogger(__name__)

# Leefbaarometer target columns
TARGET_COLS = ["lbm", "fys", "onv", "soc", "vrz", "won"]


@dataclass
class LeefbaarometerConfig:
    """Configuration for leefbaarometer target preparation."""

    study_area: str = "netherlands"
    year: int = 2022
    h3_resolution: int = 10
    target_cols: List[str] = field(default_factory=lambda: list(TARGET_COLS))

    # Data paths (relative to project root)
    scores_csv: Optional[str] = None
    grid_gpkg: Optional[str] = None
    output_path: Optional[str] = None
    regions_gdf_path: Optional[str] = None

    # Run-level provenance: when non-empty, output goes to a dated run
    # directory under stage3_analysis/leefbaarometer_target/{run_id}/ and
    # a run_info.json is written.  Empty default keeps current flat-file
    # behavior (target prep is less iterative than analysis probes).
    run_descriptor: str = ""

    def __post_init__(self):
        paths = StudyAreaPaths(self.study_area)
        self.run_id: Optional[str] = None
        if self.scores_csv is None:
            self.scores_csv = str(
                paths.target("leefbaarometer")
                / "open-data-leefbaarometer-meting-2022_2023-11-21_1035"
                / "Leefbaarometer-scores grids 2002-2022.csv"
            )
        if self.grid_gpkg is None:
            self.grid_gpkg = str(
                paths.target("leefbaarometer")
                / "geometrie-lbm3-2024" / "geometrie-lbm3-2024" / "grid 2024.gpkg"
            )
        if self.output_path is None:
            if self.run_descriptor:
                self.run_id = paths.create_run_id(self.run_descriptor)
                run_dir = paths.stage3_run("leefbaarometer_target", self.run_id)
                self.output_path = str(
                    run_dir / f"leefbaarometer_h3res{self.h3_resolution}_{self.year}.parquet"
                )
            else:
                self.output_path = str(
                    paths.target_file("leefbaarometer", self.h3_resolution, self.year)
                )
        if self.regions_gdf_path is None:
            self.regions_gdf_path = str(
                paths.region_file(self.h3_resolution)
            )


class LeefbaarometerTargetBuilder:
    """
    Builds H3-indexed leefbaarometer regression targets via area-weighted
    spatial join from 100m grid cells to H3 hexagons.

    Pipeline:
        1. Load grid CSV, filter to target year
        2. Load grid geometries from GeoPackage
        3. Join scores with geometries on grid_id
        4. Reproject EPSG:28992 -> EPSG:4326
        5. Area-weighted spatial join to H3 hexagons
        6. Aggregate weighted-mean scores per hexagon
        7. Save as parquet indexed by region_id
    """

    def __init__(self, config: LeefbaarometerConfig, project_root: Optional[Path] = None):
        self.config = config
        self.project_root = project_root or Path(__file__).parent.parent
        self.scores_df: Optional[pd.DataFrame] = None
        self.grid_gdf: Optional[gpd.GeoDataFrame] = None
        self.scored_grid_gdf: Optional[gpd.GeoDataFrame] = None
        self.target_gdf: Optional[gpd.GeoDataFrame] = None

    def load_scores(self) -> pd.DataFrame:
        """Load leefbaarometer scores CSV and filter to target year."""
        csv_path = self.project_root / self.config.scores_csv
        logger.info(f"Loading scores from {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"  Total rows: {len(df):,}")
        logger.info(f"  Available years: {sorted(df['jaar'].unique())}")

        df_year = df[df["jaar"] == self.config.year].copy()
        logger.info(f"  Year {self.config.year}: {len(df_year):,} grid cells")

        # Check for nulls in target columns
        null_counts = df_year[self.config.target_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"  Null counts:\n{null_counts[null_counts > 0]}")

        self.scores_df = df_year
        return df_year

    def load_grid_geometries(self) -> gpd.GeoDataFrame:
        """Load grid cell geometries from GeoPackage."""
        gpkg_path = self.project_root / self.config.grid_gpkg
        logger.info(f"Loading grid geometries from {gpkg_path}")

        gdf = gpd.read_file(gpkg_path)
        logger.info(f"  Grid cells: {len(gdf):,}")
        logger.info(f"  CRS: {gdf.crs}")

        self.grid_gdf = gdf
        return gdf

    def join_scores_with_geometries(self) -> gpd.GeoDataFrame:
        """Join scores with grid geometries on grid_id."""
        if self.scores_df is None:
            self.load_scores()
        if self.grid_gdf is None:
            self.load_grid_geometries()

        logger.info("Joining scores with grid geometries on grid_id...")

        # Merge scores with geometries
        scored = self.grid_gdf.merge(
            self.scores_df[["grid_id"] + self.config.target_cols],
            on="grid_id",
            how="inner",
        )
        logger.info(f"  Matched grid cells: {len(scored):,} "
                     f"(of {len(self.scores_df):,} scores, {len(self.grid_gdf):,} geometries)")

        # Reproject to EPSG:4326 for H3 compatibility
        logger.info("  Reprojecting EPSG:28992 -> EPSG:4326...")
        scored = scored.to_crs(epsg=4326)

        self.scored_grid_gdf = scored
        return scored

    def _get_h3_regions(self) -> gpd.GeoDataFrame:
        """Get H3 regions GeoDataFrame, either from disk or by tessellation."""
        regions_path = self.project_root / self.config.regions_gdf_path
        if regions_path.exists():
            logger.info(f"Loading pre-computed H3 regions from {regions_path}")
            regions_gdf = gpd.read_parquet(regions_path)
            # Normalize index name to region_id (may be hex_id, etc.)
            if regions_gdf.index.name != "region_id":
                if regions_gdf.index.name in ("hex_id",):
                    regions_gdf.index.name = "region_id"
                elif "region_id" in regions_gdf.columns:
                    regions_gdf = regions_gdf.set_index("region_id")
            return regions_gdf

        # If no pre-computed regions, tessellate the scored grid extent
        logger.info(f"No pre-computed regions found. Tessellating at res {self.config.h3_resolution}...")
        area_gdf = gpd.GeoDataFrame(
            geometry=[self.scored_grid_gdf.union_all()],
            crs="EPSG:4326"
        )
        regionalizer = H3Regionalizer(resolution=self.config.h3_resolution)
        regions_gdf = regionalizer.transform(area_gdf)
        logger.info(f"  Tessellated {len(regions_gdf):,} hexagons")
        return regions_gdf

    def spatial_join_area_weighted(self, chunk_size: int = 50000) -> gpd.GeoDataFrame:
        """
        Area-weighted spatial join: 100m grid cells -> H3 hexagons.

        For each grid cell that intersects an H3 hexagon, the contribution
        is weighted by the intersection area. The final score per hexagon
        is the area-weighted mean of all contributing grid cells.

        Uses spatial pre-filtering and chunked processing to handle large
        datasets (382K grid cells x 5M+ hexagons).

        Args:
            chunk_size: Number of grid cells to process per chunk.
        """
        if self.scored_grid_gdf is None:
            self.join_scores_with_geometries()

        logger.info("Performing area-weighted spatial join to H3 hexagons...")

        regions_gdf = self._get_h3_regions()
        logger.info(f"  H3 hexagons (total): {len(regions_gdf):,}")

        scored = self.scored_grid_gdf.copy()

        # Spatial pre-filter: only keep hexagons that overlap the grid extent
        grid_bounds = scored.total_bounds  # [minx, miny, maxx, maxy]
        logger.info(f"  Grid extent: {grid_bounds}")

        # Use spatial index to filter hexagons to grid extent (with small buffer)
        buf = 0.01  # ~1km buffer in degrees
        bbox_filter = (
            (regions_gdf.geometry.bounds["minx"] <= grid_bounds[2] + buf) &
            (regions_gdf.geometry.bounds["maxx"] >= grid_bounds[0] - buf) &
            (regions_gdf.geometry.bounds["miny"] <= grid_bounds[3] + buf) &
            (regions_gdf.geometry.bounds["maxy"] >= grid_bounds[1] - buf)
        )
        regions_filtered = regions_gdf[bbox_filter].copy()
        logger.info(f"  H3 hexagons (after bbox filter): {len(regions_filtered):,}")

        # Further filter using sjoin to find only hexagons that actually touch grid cells
        logger.info("  Finding hexagons that intersect grid cells (sjoin)...")
        hex_with_data = gpd.sjoin(
            regions_filtered.reset_index()[["region_id", "geometry"]],
            scored[["geometry"]],
            how="inner",
            predicate="intersects",
        )
        relevant_hex_ids = hex_with_data["region_id"].unique()
        regions_relevant = regions_filtered.loc[
            regions_filtered.index.isin(relevant_hex_ids)
        ]
        logger.info(f"  H3 hexagons (with grid overlap): {len(regions_relevant):,}")

        # Fix any invalid geometries
        scored["geometry"] = scored["geometry"].apply(make_valid)
        regions_relevant["geometry"] = regions_relevant["geometry"].apply(make_valid)

        # Process overlay in chunks of grid cells to manage memory
        n_chunks = max(1, len(scored) // chunk_size + (1 if len(scored) % chunk_size else 0))
        logger.info(f"  Processing overlay in {n_chunks} chunks of {chunk_size:,} grid cells...")

        all_intersections = []
        regions_reset = regions_relevant.reset_index()[["region_id", "geometry"]]

        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(scored))
            chunk = scored.iloc[start:end]

            logger.info(f"    Chunk {i + 1}/{n_chunks}: grid cells {start:,}-{end:,}...")

            try:
                intersection = gpd.overlay(
                    regions_reset,
                    chunk[["grid_id", "geometry"] + self.config.target_cols],
                    how="intersection",
                )
                if len(intersection) > 0:
                    all_intersections.append(intersection)
            except Exception as e:
                logger.warning(f"    Chunk {i + 1} overlay error: {e}")

        if not all_intersections:
            raise ValueError("No intersections found between grid cells and H3 hexagons. "
                             "Check CRS alignment.")

        intersection = pd.concat(all_intersections, ignore_index=True)
        intersection = gpd.GeoDataFrame(intersection, crs="EPSG:4326")
        logger.info(f"  Total intersection pieces: {len(intersection):,}")

        # Calculate intersection areas as weights
        # Project to equal-area CRS for accurate area calculation
        logger.info("  Computing intersection areas...")
        intersection_proj = intersection.to_crs(epsg=3035)  # ETRS89-LAEA Europe
        intersection["weight"] = intersection_proj.geometry.area

        # Area-weighted aggregation per hexagon using vectorized operations
        logger.info("  Aggregating area-weighted means per hexagon...")

        # For each target column, compute weighted sum and weight sum per region_id
        agg_dict = {"weight": "sum", "grid_id": "nunique"}
        for col in self.config.target_cols:
            # Weighted value column
            intersection[f"_wv_{col}"] = intersection[col] * intersection["weight"]
            agg_dict[f"_wv_{col}"] = "sum"

        grouped = intersection.groupby("region_id").agg(agg_dict)
        grouped = grouped.rename(columns={"weight": "weight_sum", "grid_id": "n_grid_cells"})

        # Divide weighted sums by total weight
        for col in self.config.target_cols:
            grouped[col] = grouped[f"_wv_{col}"] / grouped["weight_sum"]
            grouped = grouped.drop(columns=[f"_wv_{col}"])

        # Filter out rows with zero weight
        grouped = grouped[grouped["weight_sum"] > 0]

        logger.info(f"  Hexagons with scores: {len(grouped):,}")

        # Add geometry from regions_gdf
        result_gdf = gpd.GeoDataFrame(
            grouped.join(regions_gdf[["geometry"]]),
            crs="EPSG:4326",
        )

        # Log summary statistics
        for col in self.config.target_cols:
            valid = result_gdf[col].dropna()
            logger.info(f"  {col}: mean={valid.mean():.4f}, std={valid.std():.4f}, "
                        f"n={len(valid):,}")

        self.target_gdf = result_gdf
        return result_gdf

    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save target data as parquet indexed by region_id."""
        if self.target_gdf is None:
            raise ValueError("No target data to save. Run spatial_join_area_weighted() first.")

        out = output_path or (self.project_root / self.config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Save without geometry for lightweight loading
        target_df = self.target_gdf.drop(columns=["geometry"], errors="ignore")
        target_df.to_parquet(out)

        logger.info(f"Saved target data to {out}")
        logger.info(f"  Shape: {target_df.shape}")
        logger.info(f"  Index: {target_df.index.name}")

        # Write run-level provenance when using a run directory
        if self.config.run_id is not None:
            write_run_info(
                out.parent,
                stage="stage3",
                study_area=self.config.study_area,
                config={
                    "year": self.config.year,
                    "h3_resolution": self.config.h3_resolution,
                    "target_cols": self.config.target_cols,
                    "n_hexagons": len(target_df),
                },
            )
            logger.info(f"Saved run_info.json to {out.parent / 'run_info.json'}")

        return out

    def run(self) -> gpd.GeoDataFrame:
        """Run the full target preparation pipeline."""
        logger.info(f"=== Leefbaarometer Target Preparation ===")
        logger.info(f"Study area: {self.config.study_area}")
        logger.info(f"Year: {self.config.year}")
        logger.info(f"H3 resolution: {self.config.h3_resolution}")

        self.load_scores()
        self.load_grid_geometries()
        self.join_scores_with_geometries()
        self.spatial_join_area_weighted()
        self.save()

        logger.info("=== Target preparation complete ===")
        return self.target_gdf


def main():
    """Run leefbaarometer target preparation with default config."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = LeefbaarometerConfig()
    builder = LeefbaarometerTargetBuilder(config)
    builder.run()


if __name__ == "__main__":
    main()
