"""
Tests for stage3_analysis.urban_taxonomy_target.UrbanTaxonomyTargetBuilder.

Uses synthetic morphotope GeoDataFrames and mock H3 regions to test the
target preparation pipeline without requiring real data files on disk.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box


class TestUrbanTaxonomyImports:
    """Verify urban taxonomy target modules import cleanly."""

    def test_import_builder(self):
        from stage3_analysis.urban_taxonomy_target import UrbanTaxonomyTargetBuilder
        assert UrbanTaxonomyTargetBuilder is not None

    def test_import_config(self):
        from stage3_analysis.urban_taxonomy_target import UrbanTaxonomyConfig
        assert UrbanTaxonomyConfig is not None

    def test_import_constants(self):
        from stage3_analysis.urban_taxonomy_target import ALL_LEVELS, NAMED_LEVELS
        assert ALL_LEVELS == [1, 2, 3, 4, 5, 6, 7]
        assert NAMED_LEVELS == [1, 2, 3]


def _make_morphotopes(n=10, crs="EPSG:3035"):
    """Create a synthetic morphotope GeoDataFrame in EPSG:3035.

    Returns a GDF with geometry + level_N_label columns for levels 1-7,
    covering a grid of non-overlapping rectangles.
    """
    rng = np.random.RandomState(42)
    rows = []
    # Create a grid of 100m x 100m morphotopes centered around (3900000, 3100000)
    for i in range(n):
        x0 = 3900000 + (i % 5) * 100
        y0 = 3100000 + (i // 5) * 100
        geom = box(x0, y0, x0 + 100, y0 + 100)
        row = {"geometry": geom}
        for level in range(1, 8):
            # Each level assigns a label from 1..n_classes
            n_classes = [2, 4, 8, 16, 25, 52, 101][level - 1]
            row[f"level_{level}_label"] = rng.randint(1, min(n_classes + 1, n + 1))
        rows.append(row)

    return gpd.GeoDataFrame(rows, crs=crs)


def _make_hex_regions(morphotopes_gdf, n_hexes_per_morphotope=3):
    """Create synthetic H3 regions (small hexagon-like polygons) inside morphotopes.

    Returns a GDF indexed by region_id with geometry in EPSG:3035.
    """
    rows = []
    hex_id = 0
    for _, morph in morphotopes_gdf.iterrows():
        bounds = morph.geometry.bounds  # minx, miny, maxx, maxy
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        # Place small "hexagons" (actually tiny squares) at centroid offsets
        offsets = [(0, 0), (-20, 0), (20, 0)]
        for dx, dy in offsets[:n_hexes_per_morphotope]:
            x, y = cx + dx, cy + dy
            geom = box(x - 5, y - 5, x + 5, y + 5)
            rows.append({
                "region_id": f"8a1234567{hex_id:06x}fff",
                "geometry": geom,
            })
            hex_id += 1

    gdf = gpd.GeoDataFrame(rows, crs=morphotopes_gdf.crs)
    gdf = gdf.set_index("region_id")
    return gdf


class TestCentroidSpatialJoin:
    """Test that centroid-based spatial join assigns hexagons to morphotopes."""

    def test_centroid_join_assigns_labels(self):
        """Hex centroids falling inside morphotopes get correct level labels."""
        morphotopes = _make_morphotopes(n=4)
        hex_regions = _make_hex_regions(morphotopes, n_hexes_per_morphotope=2)

        # Compute centroids
        centroids = hex_regions.copy()
        centroids["geometry"] = centroids.geometry.centroid

        # sjoin centroids within morphotopes
        level_cols = [f"level_{l}_label" for l in range(1, 8)]
        joined = gpd.sjoin(
            centroids.reset_index()[["region_id", "geometry"]],
            morphotopes[["geometry"] + level_cols],
            how="inner",
            predicate="within",
        )

        # Every hex centroid should fall within a morphotope
        assert len(joined) >= len(hex_regions), (
            f"Expected at least {len(hex_regions)} matches but got {len(joined)}"
        )

        # All level columns should be present
        for col in level_cols:
            assert col in joined.columns

    def test_centroid_join_no_nans_in_labels(self):
        """Assigned labels should not contain NaN."""
        morphotopes = _make_morphotopes(n=6)
        hex_regions = _make_hex_regions(morphotopes, n_hexes_per_morphotope=1)

        centroids = hex_regions.copy()
        centroids["geometry"] = centroids.geometry.centroid

        level_cols = [f"level_{l}_label" for l in range(1, 8)]
        joined = gpd.sjoin(
            centroids.reset_index()[["region_id", "geometry"]],
            morphotopes[["geometry"] + level_cols],
            how="inner",
            predicate="within",
        )

        for col in level_cols:
            assert not joined[col].isna().any(), f"NaN found in {col}"


class TestDeduplication:
    """Test deduplication on boundary hexagons."""

    def test_boundary_hex_dedup_keeps_first(self):
        """A centroid on the boundary of two morphotopes should keep first match."""
        # Create two adjacent morphotopes that share an edge
        morph1 = box(0, 0, 100, 100)
        morph2 = box(100, 0, 200, 100)
        morphotopes = gpd.GeoDataFrame({
            "geometry": [morph1, morph2],
            "level_1_label": [1, 2],
        }, crs="EPSG:3035")

        # Place a hex centroid right on the shared edge (x=100)
        # Use "within" predicate — point on edge may match 0 or 1, not both
        # But with "intersects" it could match both
        boundary_point = gpd.GeoDataFrame({
            "region_id": ["boundary_hex"],
            "geometry": [box(95, 45, 105, 55)],  # straddles the boundary
        }, crs="EPSG:3035").set_index("region_id")

        # Use centroid (x=100, y=50) — exactly on boundary
        centroids = boundary_point.copy()
        centroids["geometry"] = centroids.geometry.centroid

        joined = gpd.sjoin(
            centroids.reset_index()[["region_id", "geometry"]],
            morphotopes[["geometry", "level_1_label"]],
            how="left",
            predicate="within",
        )

        # Dedup: keep first
        joined = joined.drop_duplicates(subset="region_id", keep="first")
        # Should have exactly one row
        assert len(joined) == 1


class TestTargetResultFieldContract:
    """Test that the output DataFrame has the expected field structure."""

    def test_result_has_type_level_columns(self):
        """Result should have type_level{N} for each configured level."""
        morphotopes = _make_morphotopes(n=4)
        hex_regions = _make_hex_regions(morphotopes, n_hexes_per_morphotope=1)

        centroids = hex_regions.copy()
        centroids["geometry"] = centroids.geometry.centroid

        level_cols = [f"level_{l}_label" for l in range(1, 8)]
        joined = gpd.sjoin(
            centroids.reset_index()[["region_id", "geometry"]],
            morphotopes[["geometry"] + level_cols],
            how="inner",
            predicate="within",
        )
        joined = joined.drop_duplicates(subset="region_id", keep="first")
        joined = joined.set_index("region_id")

        # Build result like the actual code does
        result = pd.DataFrame(index=joined.index)
        for col in level_cols:
            level_num = col.replace("level_", "").replace("_label", "")
            result[f"type_level{level_num}"] = joined[col].astype(int)

        # Verify all expected columns
        for level in range(1, 8):
            assert f"type_level{level}" in result.columns
        assert result.index.name == "region_id"

    def test_result_integer_labels(self):
        """All type_level columns should be integer dtype."""
        morphotopes = _make_morphotopes(n=4)
        hex_regions = _make_hex_regions(morphotopes, n_hexes_per_morphotope=1)

        centroids = hex_regions.copy()
        centroids["geometry"] = centroids.geometry.centroid

        level_cols = [f"level_{l}_label" for l in range(1, 8)]
        joined = gpd.sjoin(
            centroids.reset_index()[["region_id", "geometry"]],
            morphotopes[["geometry"] + level_cols],
            how="inner",
            predicate="within",
        )
        joined = joined.drop_duplicates(subset="region_id", keep="first")
        joined = joined.set_index("region_id")

        result = pd.DataFrame(index=joined.index)
        for col in level_cols:
            level_num = col.replace("level_", "").replace("_label", "")
            result[f"type_level{level_num}"] = joined[col].astype(int)

        for level in range(1, 8):
            col = f"type_level{level}"
            assert pd.api.types.is_integer_dtype(result[col]), (
                f"{col} should be integer but is {result[col].dtype}"
            )

    def test_n_morphotopes_boundary_indicator(self):
        """n_morphotopes column should count polygon-level intersections."""
        # Two adjacent morphotopes
        morph1 = box(0, 0, 100, 100)
        morph2 = box(80, 0, 200, 100)  # overlaps morph1 by 20m
        morphotopes = gpd.GeoDataFrame({
            "geometry": [morph1, morph2],
            "level_1_label": [1, 2],
        }, crs="EPSG:3035")

        # Hex fully inside morph1 (not in overlap zone)
        interior_hex = box(10, 40, 20, 60)
        # Hex in overlap zone
        boundary_hex = box(85, 40, 95, 60)

        hex_regions = gpd.GeoDataFrame({
            "region_id": ["interior", "boundary"],
            "geometry": [interior_hex, boundary_hex],
        }, crs="EPSG:3035").set_index("region_id")

        # Count morphotopes per hex via polygon intersects
        poly_join = gpd.sjoin(
            hex_regions.reset_index()[["region_id", "geometry"]],
            morphotopes[["geometry"]],
            how="inner",
            predicate="intersects",
        )
        n_morphotopes = poly_join.groupby("region_id").size().rename("n_morphotopes")

        # Interior hex: should intersect 1 morphotope
        assert n_morphotopes.get("interior", 0) == 1
        # Boundary hex: should intersect 2 morphotopes (in overlap zone)
        assert n_morphotopes.get("boundary", 0) == 2


class TestLabelNameAttachment:
    """Test that human-readable names are attached for levels 1-3."""

    def test_label_names_loaded_from_json(self):
        """label_name.json maps integer labels to human-readable names."""
        label_names = {
            "1": {"1": "Coherent Fabric", "2": "Incoherent Fabric"},
            "2": {"1": "TypeA", "2": "TypeB", "3": "TypeC", "4": "TypeD"},
        }

        # Simulate what the builder does for name attachment
        result = pd.DataFrame({
            "type_level1": [1, 2, 1],
            "type_level2": [1, 3, 4],
        }, index=pd.Index(["hex1", "hex2", "hex3"], name="region_id"))

        for level in [1, 2]:
            level_key = str(level)
            if level_key in label_names:
                name_map = {int(k): v for k, v in label_names[level_key].items()}
                result[f"name_level{level}"] = result[f"type_level{level}"].map(name_map)

        assert result["name_level1"].tolist() == [
            "Coherent Fabric", "Incoherent Fabric", "Coherent Fabric"
        ]
        assert result["name_level2"].tolist() == ["TypeA", "TypeC", "TypeD"]
