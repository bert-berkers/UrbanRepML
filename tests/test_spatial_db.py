"""
Tests for utils/spatial_db.py — SpatialDB spatial query engine.

Coverage targets:
- GeoPandas fallback path (sedonadb import mocked out)
- SedonaDB path (sedonadb is installed; tested against synthetic parquet)
- All three existing public methods: centroids, geometry, extent
- Three planned neighbourhood methods: neighbours, neighbours_geometry, k_ring_centroids
- Edge cases: empty input, IDs absent from parquet, single-hex input
- CRS reprojection (4326 -> 28992)

Synthetic data strategy
-----------------------
A small GeoParquet is written to a pytest tmp_path using SRAI H3Regionalizer
over a ~0.05-degree bounding box near The Hague (res 9, ~157 hexagons).
StudyAreaPaths is patched so SpatialDB reads from that file instead of
real study-area data.  The instance cache is cleared before every test so
patch boundaries are respected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from srai.regionalizers import H3Regionalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RES = 9
_STUDY_AREA = "test_area"


def _write_synthetic_parquet(tmp_path: Path) -> tuple[Path, gpd.GeoDataFrame]:
    """Create a synthetic regions GeoParquet and return (path, gdf)."""
    area = gpd.GeoDataFrame(
        geometry=[box(4.3, 52.0, 4.35, 52.03)],
        crs="EPSG:4326",
    )
    regionalizer = H3Regionalizer(resolution=_RES)
    regions = regionalizer.transform(area)
    # regions has index.name == "region_id", CRS == EPSG:4326, column "geometry"

    regions_dir = tmp_path / "data" / "study_areas" / _STUDY_AREA / "regions_gdf"
    regions_dir.mkdir(parents=True)
    parquet_path = regions_dir / f"{_STUDY_AREA}_res{_RES}.parquet"
    regions.to_parquet(parquet_path)
    return parquet_path, regions


def _make_db(tmp_path: Path, backend: str = "geopandas"):
    """Construct a SpatialDB whose StudyAreaPaths points at tmp_path.

    The instance cache is always cleared first so fixture isolation is
    guaranteed.  Pass backend='sedonadb' to let the real sedonadb import
    proceed; backend='geopandas' mocks out the import so the fallback path
    is exercised.
    """
    from utils.spatial_db import SpatialDB

    SpatialDB._instances.clear()

    parquet_path = (
        tmp_path
        / "data"
        / "study_areas"
        / _STUDY_AREA
        / "regions_gdf"
        / f"{_STUDY_AREA}_res{_RES}.parquet"
    )

    mock_paths = MagicMock()
    mock_paths.region_file.return_value = parquet_path

    with patch("utils.spatial_db.StudyAreaPaths", return_value=mock_paths):
        if backend == "geopandas":
            # Hide sedonadb so __init__ falls back to GeoPandas
            with patch.dict(sys.modules, {"sedonadb": None}):
                db = SpatialDB(_STUDY_AREA)
        else:
            db = SpatialDB(_STUDY_AREA)

    # Re-attach the mock so methods called after construction still resolve
    db._paths = mock_paths
    return db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    """Write synthetic parquet once per test-module run; return (tmp_path, gdf)."""
    tmp = tmp_path_factory.mktemp("spatial_db")
    parquet_path, gdf = _write_synthetic_parquet(tmp)
    return tmp, gdf


@pytest.fixture()
def db_gpd(synthetic_data):
    """SpatialDB using the GeoPandas fallback path."""
    tmp, _ = synthetic_data
    return _make_db(tmp, backend="geopandas")


@pytest.fixture()
def db_sedona(synthetic_data):
    """SpatialDB using the SedonaDB backend (real import, synthetic parquet)."""
    tmp, _ = synthetic_data
    return _make_db(tmp, backend="sedonadb")


@pytest.fixture()
def hex_ids(synthetic_data):
    """A slice of 10 valid hex IDs from the synthetic GeoDataFrame."""
    _, gdf = synthetic_data
    return list(gdf.index[:10])


@pytest.fixture()
def all_hex_ids(synthetic_data):
    """All hex IDs from the synthetic GeoDataFrame."""
    _, gdf = synthetic_data
    return list(gdf.index)


# ---------------------------------------------------------------------------
# Contract: GeoDataFrame schema of the synthetic fixture
# ---------------------------------------------------------------------------


class TestSyntheticFixtureSchema:
    """Validate that the fixture itself satisfies the region_id contract."""

    def test_index_name(self, synthetic_data):
        _, gdf = synthetic_data
        assert gdf.index.name == "region_id"

    def test_index_dtype(self, synthetic_data):
        _, gdf = synthetic_data
        # Pandas >=2.0 on Python 3.13+ uses StringDtype for string indexes;
        # older versions use object.  Both are valid for H3 hex strings.
        dtype_name = str(gdf.index.dtype)
        assert dtype_name in ("object", "str") or "string" in dtype_name.lower(), (
            f"Unexpected index dtype: {gdf.index.dtype}"
        )

    def test_no_duplicate_index(self, synthetic_data):
        _, gdf = synthetic_data
        assert not gdf.index.has_duplicates

    def test_crs_wgs84(self, synthetic_data):
        _, gdf = synthetic_data
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_no_empty_geometries(self, synthetic_data):
        _, gdf = synthetic_data
        assert not gdf.geometry.is_empty.any()

    def test_reasonable_hex_count(self, synthetic_data):
        _, gdf = synthetic_data
        # The small bounding box should produce tens to hundreds of hexagons
        assert 10 < len(gdf) < 1000


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


class TestBackendDetection:
    def test_geopandas_backend_when_sedonadb_absent(self, db_gpd):
        assert db_gpd._backend == "geopandas"

    def test_sedonadb_backend_when_sedonadb_present(self, db_sedona):
        assert db_sedona._backend == "sedonadb"


# ---------------------------------------------------------------------------
# centroids() — GeoPandas path
# ---------------------------------------------------------------------------


class TestCentroidsGeoPandas:
    def test_returns_two_arrays(self, db_gpd, hex_ids):
        cx, cy = db_gpd.centroids(hex_ids, resolution=_RES)
        assert isinstance(cx, np.ndarray)
        assert isinstance(cy, np.ndarray)

    def test_length_matches_input(self, db_gpd, hex_ids):
        cx, cy = db_gpd.centroids(hex_ids, resolution=_RES)
        assert len(cx) == len(hex_ids)
        assert len(cy) == len(hex_ids)

    def test_order_matches_input(self, db_gpd, all_hex_ids):
        """Reordering the input IDs must reorder the output."""
        ids_fwd = all_hex_ids[:20]
        ids_rev = list(reversed(ids_fwd))
        cx_fwd, cy_fwd = db_gpd.centroids(ids_fwd, resolution=_RES)
        cx_rev, cy_rev = db_gpd.centroids(ids_rev, resolution=_RES)
        np.testing.assert_array_almost_equal(cx_fwd, cx_rev[::-1])
        np.testing.assert_array_almost_equal(cy_fwd, cy_rev[::-1])

    def test_coordinates_in_wgs84_range(self, db_gpd, hex_ids):
        """Synthetic area is near The Hague; check lat/lon plausibility."""
        cx, cy = db_gpd.centroids(hex_ids, resolution=_RES)
        assert np.all(cx > 4.0) and np.all(cx < 5.0)   # longitude
        assert np.all(cy > 51.5) and np.all(cy < 52.5)  # latitude

    def test_crs_reprojection_28992(self, db_gpd, hex_ids):
        """RD New (EPSG:28992) coordinates should differ from WGS84."""
        cx_4326, cy_4326 = db_gpd.centroids(hex_ids, resolution=_RES, crs=4326)
        cx_rd, cy_rd = db_gpd.centroids(hex_ids, resolution=_RES, crs=28992)
        # RD New uses metres; values are much larger than lat/lon
        assert not np.allclose(cx_4326, cx_rd)
        assert not np.allclose(cy_4326, cy_rd)
        # RD New x for The Hague is roughly 60 000 – 100 000
        assert np.all(cx_rd > 50_000) and np.all(cx_rd < 150_000)

    def test_single_hex_id(self, db_gpd, hex_ids):
        single = hex_ids[:1]
        cx, cy = db_gpd.centroids(single, resolution=_RES)
        assert len(cx) == 1
        assert len(cy) == 1

    def test_empty_input(self, db_gpd):
        cx, cy = db_gpd.centroids([], resolution=_RES)
        assert len(cx) == 0
        assert len(cy) == 0

    def test_missing_ids_produce_nan(self, db_gpd):
        """IDs not in the parquet must yield NaN, not raise."""
        fake_ids = ["8f00000000000000", "8f00000000000001"]
        cx, cy = db_gpd.centroids(fake_ids, resolution=_RES)
        assert np.all(np.isnan(cx))
        assert np.all(np.isnan(cy))

    def test_accepts_pd_index(self, db_gpd, hex_ids):
        idx = pd.Index(hex_ids)
        cx, cy = db_gpd.centroids(idx, resolution=_RES)
        assert len(cx) == len(idx)

    def test_accepts_np_array(self, db_gpd, hex_ids):
        arr = np.array(hex_ids)
        cx, cy = db_gpd.centroids(arr, resolution=_RES)
        assert len(cx) == len(arr)


# ---------------------------------------------------------------------------
# geometry() — GeoPandas path
# ---------------------------------------------------------------------------


class TestGeometryGeoPandas:
    def test_returns_geodataframe(self, db_gpd, hex_ids):
        result = db_gpd.geometry(hex_ids, resolution=_RES)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_index_name_is_region_id(self, db_gpd, hex_ids):
        result = db_gpd.geometry(hex_ids, resolution=_RES)
        assert result.index.name == "region_id"

    def test_length_matches_input(self, db_gpd, hex_ids):
        result = db_gpd.geometry(hex_ids, resolution=_RES)
        assert len(result) == len(hex_ids)

    def test_order_matches_input(self, db_gpd, all_hex_ids):
        ids = all_hex_ids[:20]
        result = db_gpd.geometry(ids, resolution=_RES)
        assert list(result.index) == ids

    def test_crs_default_wgs84(self, db_gpd, hex_ids):
        result = db_gpd.geometry(hex_ids, resolution=_RES)
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

    def test_crs_reprojection_28992(self, db_gpd, hex_ids):
        result = db_gpd.geometry(hex_ids, resolution=_RES, crs=28992)
        assert result.crs.to_epsg() == 28992

    def test_geometry_column_not_empty(self, db_gpd, hex_ids):
        result = db_gpd.geometry(hex_ids, resolution=_RES)
        assert not result.geometry.is_empty.any()

    def test_single_hex_id(self, db_gpd, hex_ids):
        result = db_gpd.geometry(hex_ids[:1], resolution=_RES)
        assert len(result) == 1

    def test_empty_input(self, db_gpd):
        result = db_gpd.geometry([], resolution=_RES)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    def test_missing_ids_yield_empty_geometry(self, db_gpd):
        """IDs not in the parquet must not raise; rows may be missing or NaN."""
        fake_ids = ["8f00000000000000"]
        result = db_gpd.geometry(fake_ids, resolution=_RES)
        # After reindex the row will exist but geometry is NaN/None
        assert len(result) == 1
        assert result.geometry.iloc[0] is None or result.geometry.isna().iloc[0]


# ---------------------------------------------------------------------------
# extent() — GeoPandas path
# ---------------------------------------------------------------------------


class TestExtentGeoPandas:
    def test_returns_four_floats(self, db_gpd, hex_ids):
        result = db_gpd.extent(hex_ids, resolution=_RES)
        assert isinstance(result, tuple)
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)

    def test_order_minx_miny_maxx_maxy(self, db_gpd, all_hex_ids):
        minx, miny, maxx, maxy = db_gpd.extent(all_hex_ids, resolution=_RES)
        assert minx < maxx
        assert miny < maxy

    def test_bbox_covers_synthetic_area(self, db_gpd, all_hex_ids):
        """The bounding box should at least partially overlap [4.3, 52.0, 4.35, 52.03]."""
        minx, miny, maxx, maxy = db_gpd.extent(all_hex_ids, resolution=_RES)
        assert minx < 4.35 and maxx > 4.3
        assert miny < 52.03 and maxy > 52.0

    def test_crs_reprojection_28992(self, db_gpd, hex_ids):
        minx_4326, _, _, _ = db_gpd.extent(hex_ids, resolution=_RES, crs=4326)
        minx_rd, _, _, _ = db_gpd.extent(hex_ids, resolution=_RES, crs=28992)
        # RD New metres vs degrees — values must differ substantially
        assert abs(minx_4326 - minx_rd) > 1.0

    def test_single_hex_gives_non_degenerate_bbox(self, db_gpd, hex_ids):
        """A single hexagon still has non-zero area bounding box."""
        minx, miny, maxx, maxy = db_gpd.extent(hex_ids[:1], resolution=_RES)
        assert maxx > minx
        assert maxy > miny

    def test_subset_bbox_inside_full_bbox(self, db_gpd, all_hex_ids):
        """Extent of a subset must be contained in the full extent."""
        full = db_gpd.extent(all_hex_ids, resolution=_RES)
        sub = db_gpd.extent(all_hex_ids[:5], resolution=_RES)
        assert sub[0] >= full[0]  # minx
        assert sub[1] >= full[1]  # miny
        assert sub[2] <= full[2]  # maxx
        assert sub[3] <= full[3]  # maxy


# ---------------------------------------------------------------------------
# centroids() / geometry() / extent() — SedonaDB path (smoke-level)
# ---------------------------------------------------------------------------


class TestSedonaDBPath:
    """Smoke tests verifying SedonaDB path returns the same shapes/types.

    These run against the real sedonadb package with the synthetic parquet.
    They are not exhaustive — the GeoPandas tests above cover all contract
    details; here we confirm the SedonaDB code path doesn't blow up and
    returns plausible results.
    """

    def test_centroids_shape(self, db_sedona, hex_ids):
        cx, cy = db_sedona.centroids(hex_ids, resolution=_RES)
        assert len(cx) == len(hex_ids)
        assert len(cy) == len(hex_ids)

    def test_geometry_returns_geodataframe(self, db_sedona, hex_ids):
        result = db_sedona.geometry(hex_ids, resolution=_RES)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(hex_ids)

    def test_extent_returns_four_floats(self, db_sedona, hex_ids):
        result = db_sedona.extent(hex_ids, resolution=_RES)
        assert len(result) == 4
        minx, miny, maxx, maxy = result
        assert minx < maxx
        assert miny < maxy

    def test_sedona_centroids_match_geopandas(self, synthetic_data, hex_ids):
        """Both backends must agree on centroid values within 1e-6 degrees."""
        tmp, _ = synthetic_data
        db_gpd = _make_db(tmp, backend="geopandas")
        db_sdb = _make_db(tmp, backend="sedonadb")

        cx_g, cy_g = db_gpd.centroids(hex_ids, resolution=_RES)
        cx_s, cy_s = db_sdb.centroids(hex_ids, resolution=_RES)

        np.testing.assert_allclose(cx_g, cx_s, atol=1e-6)
        np.testing.assert_allclose(cy_g, cy_s, atol=1e-6)


# ---------------------------------------------------------------------------
# Instance caching via for_study_area()
# ---------------------------------------------------------------------------


class TestInstanceCaching:
    def test_for_study_area_returns_same_instance(self, synthetic_data):
        tmp, _ = synthetic_data
        from utils.spatial_db import SpatialDB

        SpatialDB._instances.clear()
        parquet_path = (
            tmp
            / "data"
            / "study_areas"
            / _STUDY_AREA
            / "regions_gdf"
            / f"{_STUDY_AREA}_res{_RES}.parquet"
        )
        mock_paths = MagicMock()
        mock_paths.region_file.return_value = parquet_path

        with patch("utils.spatial_db.StudyAreaPaths", return_value=mock_paths):
            with patch.dict(sys.modules, {"sedonadb": None}):
                db1 = SpatialDB.for_study_area(_STUDY_AREA)
                db2 = SpatialDB.for_study_area(_STUDY_AREA)

        assert db1 is db2

    def test_different_study_areas_give_different_instances(self, synthetic_data):
        tmp, _ = synthetic_data
        from utils.spatial_db import SpatialDB

        SpatialDB._instances.clear()
        parquet_path = (
            tmp
            / "data"
            / "study_areas"
            / _STUDY_AREA
            / "regions_gdf"
            / f"{_STUDY_AREA}_res{_RES}.parquet"
        )
        mock_paths = MagicMock()
        mock_paths.region_file.return_value = parquet_path

        with patch("utils.spatial_db.StudyAreaPaths", return_value=mock_paths):
            with patch.dict(sys.modules, {"sedonadb": None}):
                db_a = SpatialDB.for_study_area("area_a")
                db_b = SpatialDB.for_study_area("area_b")

        assert db_a is not db_b


# ---------------------------------------------------------------------------
# File-not-found error handling
# ---------------------------------------------------------------------------


class TestFileNotFoundHandling:
    def test_centroids_raises_on_missing_parquet(self):
        """FileNotFoundError when the parquet does not exist."""
        from utils.spatial_db import SpatialDB

        SpatialDB._instances.clear()
        mock_paths = MagicMock()
        mock_paths.region_file.return_value = Path("/nonexistent/regions.parquet")

        with patch("utils.spatial_db.StudyAreaPaths", return_value=mock_paths):
            with patch.dict(sys.modules, {"sedonadb": None}):
                db = SpatialDB(_STUDY_AREA)
        db._paths = mock_paths

        with pytest.raises(FileNotFoundError):
            db.centroids(["89196971487ffff"], resolution=_RES)

    def test_geometry_raises_on_missing_parquet(self):
        from utils.spatial_db import SpatialDB

        SpatialDB._instances.clear()
        mock_paths = MagicMock()
        mock_paths.region_file.return_value = Path("/nonexistent/regions.parquet")

        with patch("utils.spatial_db.StudyAreaPaths", return_value=mock_paths):
            with patch.dict(sys.modules, {"sedonadb": None}):
                db = SpatialDB(_STUDY_AREA)
        db._paths = mock_paths

        with pytest.raises(FileNotFoundError):
            db.geometry(["89196971487ffff"], resolution=_RES)

    def test_extent_raises_on_missing_parquet(self):
        from utils.spatial_db import SpatialDB

        SpatialDB._instances.clear()
        mock_paths = MagicMock()
        mock_paths.region_file.return_value = Path("/nonexistent/regions.parquet")

        with patch("utils.spatial_db.StudyAreaPaths", return_value=mock_paths):
            with patch.dict(sys.modules, {"sedonadb": None}):
                db = SpatialDB(_STUDY_AREA)
        db._paths = mock_paths

        with pytest.raises(FileNotFoundError):
            db.extent(["89196971487ffff"], resolution=_RES)


# ---------------------------------------------------------------------------
# Neighbourhood API  (methods being added in parallel)
#
# These tests are written against the expected signatures:
#   neighbours(hex_ids, resolution, k=1)
#     -> dict[str, set[str]]
#   neighbours_geometry(hex_ids, resolution, k=1, crs=4326)
#     -> gpd.GeoDataFrame  (indexed by region_id, includes neighbour hexes)
#   k_ring_centroids(hex_ids, resolution, k=1, crs=4326)
#     -> (ids: np.ndarray, cx: np.ndarray, cy: np.ndarray)
#
# If the methods do not exist yet these tests will xfail automatically.
# ---------------------------------------------------------------------------


def _has_method(db, name: str) -> bool:
    return callable(getattr(db, name, None))


class TestNeighboursMethod:
    """neighbours(hex_ids, resolution, k=1) -> dict[str, set[str]]"""

    def test_method_exists(self, db_gpd):
        if not _has_method(db_gpd, "neighbours"):
            pytest.xfail("neighbours() not yet implemented")

    def test_returns_dict(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours"):
            pytest.xfail("neighbours() not yet implemented")
        result = db_gpd.neighbours(hex_ids, resolution=_RES, k=1)
        assert isinstance(result, dict)

    def test_keys_match_input(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours"):
            pytest.xfail("neighbours() not yet implemented")
        result = db_gpd.neighbours(hex_ids, resolution=_RES, k=1)
        assert set(result.keys()) == set(hex_ids)

    def test_values_are_sets(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours"):
            pytest.xfail("neighbours() not yet implemented")
        result = db_gpd.neighbours(hex_ids, resolution=_RES, k=1)
        for v in result.values():
            assert isinstance(v, (set, frozenset))

    def test_k1_gives_up_to_six_neighbours(self, db_gpd, hex_ids):
        """Interior hexagons should have exactly 6 k=1 neighbours."""
        if not _has_method(db_gpd, "neighbours"):
            pytest.xfail("neighbours() not yet implemented")
        result = db_gpd.neighbours(hex_ids, resolution=_RES, k=1)
        # At least one interior hex should have exactly 6 neighbours
        counts = [len(v) for v in result.values()]
        assert max(counts) == 6, f"Expected max 6 neighbours at k=1, got {max(counts)}"

    def test_k2_gives_more_neighbours_than_k1(self, db_gpd, hex_ids):
        """k=2 ring must contain more hexagons than k=1."""
        if not _has_method(db_gpd, "neighbours"):
            pytest.xfail("neighbours() not yet implemented")
        result_k1 = db_gpd.neighbours(hex_ids[:3], resolution=_RES, k=1)
        result_k2 = db_gpd.neighbours(hex_ids[:3], resolution=_RES, k=2)
        for hid in hex_ids[:3]:
            # k=2 has up to 18 neighbours; always >= k=1
            assert len(result_k2[hid]) >= len(result_k1[hid])

    def test_neighbours_do_not_include_self(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours"):
            pytest.xfail("neighbours() not yet implemented")
        result = db_gpd.neighbours(hex_ids, resolution=_RES, k=1)
        for hid, nbrs in result.items():
            assert hid not in nbrs

    def test_empty_input(self, db_gpd):
        if not _has_method(db_gpd, "neighbours"):
            pytest.xfail("neighbours() not yet implemented")
        result = db_gpd.neighbours([], resolution=_RES, k=1)
        assert result == {} or len(result) == 0

    def test_single_hex(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours"):
            pytest.xfail("neighbours() not yet implemented")
        result = db_gpd.neighbours(hex_ids[:1], resolution=_RES, k=1)
        assert len(result) == 1
        assert len(result[hex_ids[0]]) <= 6


class TestNeighboursGeometryMethod:
    """neighbours_geometry(hex_ids, resolution, k=1, crs=4326) -> GeoDataFrame"""

    def test_method_exists(self, db_gpd):
        if not _has_method(db_gpd, "neighbours_geometry"):
            pytest.xfail("neighbours_geometry() not yet implemented")

    def test_returns_geodataframe(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours_geometry"):
            pytest.xfail("neighbours_geometry() not yet implemented")
        result = db_gpd.neighbours_geometry(hex_ids, resolution=_RES, k=1)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_index_name_region_id(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours_geometry"):
            pytest.xfail("neighbours_geometry() not yet implemented")
        result = db_gpd.neighbours_geometry(hex_ids, resolution=_RES, k=1)
        assert result.index.name == "region_id"

    def test_crs_default_wgs84(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours_geometry"):
            pytest.xfail("neighbours_geometry() not yet implemented")
        result = db_gpd.neighbours_geometry(hex_ids, resolution=_RES, k=1)
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

    def test_crs_reprojection(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours_geometry"):
            pytest.xfail("neighbours_geometry() not yet implemented")
        result = db_gpd.neighbours_geometry(
            hex_ids, resolution=_RES, k=1, crs=28992
        )
        assert result.crs.to_epsg() == 28992

    def test_result_not_empty(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours_geometry"):
            pytest.xfail("neighbours_geometry() not yet implemented")
        result = db_gpd.neighbours_geometry(hex_ids, resolution=_RES, k=1)
        # Result should include neighbour hexes (more rows than input if
        # neighbours are not already in hex_ids)
        assert len(result) > 0

    def test_no_empty_geometries(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "neighbours_geometry"):
            pytest.xfail("neighbours_geometry() not yet implemented")
        result = db_gpd.neighbours_geometry(hex_ids, resolution=_RES, k=1)
        valid = result.geometry.dropna()
        assert not valid.is_empty.any()

    def test_empty_input(self, db_gpd):
        if not _has_method(db_gpd, "neighbours_geometry"):
            pytest.xfail("neighbours_geometry() not yet implemented")
        result = db_gpd.neighbours_geometry([], resolution=_RES, k=1)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0


class TestKRingCentroidsMethod:
    """k_ring_centroids(hex_ids, resolution, k=1, crs=4326) -> (ids, cx, cy)"""

    def test_method_exists(self, db_gpd):
        if not _has_method(db_gpd, "k_ring_centroids"):
            pytest.xfail("k_ring_centroids() not yet implemented")

    def test_returns_three_arrays(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "k_ring_centroids"):
            pytest.xfail("k_ring_centroids() not yet implemented")
        result = db_gpd.k_ring_centroids(hex_ids, resolution=_RES, k=1)
        assert len(result) == 3
        ids, cx, cy = result
        assert isinstance(cx, np.ndarray)
        assert isinstance(cy, np.ndarray)

    def test_arrays_equal_length(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "k_ring_centroids"):
            pytest.xfail("k_ring_centroids() not yet implemented")
        ids, cx, cy = db_gpd.k_ring_centroids(hex_ids, resolution=_RES, k=1)
        assert len(ids) == len(cx) == len(cy)

    def test_result_at_least_as_many_as_input(self, db_gpd, hex_ids):
        """k-ring centroids include input hexes AND their neighbours."""
        if not _has_method(db_gpd, "k_ring_centroids"):
            pytest.xfail("k_ring_centroids() not yet implemented")
        ids, cx, cy = db_gpd.k_ring_centroids(hex_ids, resolution=_RES, k=1)
        assert len(ids) >= len(hex_ids)

    def test_k2_larger_than_k1(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "k_ring_centroids"):
            pytest.xfail("k_ring_centroids() not yet implemented")
        ids_k1, _, _ = db_gpd.k_ring_centroids(hex_ids[:3], resolution=_RES, k=1)
        ids_k2, _, _ = db_gpd.k_ring_centroids(hex_ids[:3], resolution=_RES, k=2)
        assert len(ids_k2) >= len(ids_k1)

    def test_coordinates_in_wgs84_range(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "k_ring_centroids"):
            pytest.xfail("k_ring_centroids() not yet implemented")
        ids, cx, cy = db_gpd.k_ring_centroids(hex_ids, resolution=_RES, k=1)
        # Drop NaN before range check
        cx_valid = cx[~np.isnan(cx)]
        cy_valid = cy[~np.isnan(cy)]
        if len(cx_valid):
            assert np.all(cx_valid > 3.5) and np.all(cx_valid < 5.5)
            assert np.all(cy_valid > 51.0) and np.all(cy_valid < 53.0)

    def test_crs_reprojection(self, db_gpd, hex_ids):
        if not _has_method(db_gpd, "k_ring_centroids"):
            pytest.xfail("k_ring_centroids() not yet implemented")
        ids_4326, cx_4326, _ = db_gpd.k_ring_centroids(
            hex_ids, resolution=_RES, k=1, crs=4326
        )
        ids_rd, cx_rd, _ = db_gpd.k_ring_centroids(
            hex_ids, resolution=_RES, k=1, crs=28992
        )
        assert not np.allclose(cx_4326[~np.isnan(cx_4326)],
                               cx_rd[~np.isnan(cx_rd)],
                               atol=1.0)

    def test_empty_input(self, db_gpd):
        if not _has_method(db_gpd, "k_ring_centroids"):
            pytest.xfail("k_ring_centroids() not yet implemented")
        ids, cx, cy = db_gpd.k_ring_centroids([], resolution=_RES, k=1)
        assert len(ids) == 0
        assert len(cx) == 0
        assert len(cy) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
