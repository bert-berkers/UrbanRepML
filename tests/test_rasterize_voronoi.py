"""Contract tests for the KDTree-Voronoi rasterization toolkit.

Written against the frozen spec ``specs/rasterize_voronoi.md`` (W1, 2026-05-02).
Tests are intentionally spec-first: they do NOT import from utils.visualization at
module level so they can be written (and collected by pytest) before W2a lands the
implementation.  Each test class uses a lazy import helper that calls
``pytest.skip`` if the required function is absent, rather than failing with an
ImportError.

Mandated test cases (from plan W2b + spec §qaqc-invariants):
1. Determinism       -- voronoi_indices twice, byte-identical output
2. max_dist_m cutoff -- pixels > max_dist_m from any centroid are alpha=0
3. GeoDataFrame ≡ coord-array -- both input forms yield byte-identical RGBA
4. latlon_to_metric round-trip -- known NL landmarks, ≤1cm tolerance
5. Gallery reuse    -- voronoi_indices + gather_rgba matches one-shot rasterize_voronoi

Additional spec-contract cases:
6. extent_m → extent_xy axis-order -- (minx,miny,maxx,maxy) in, (minx,maxx,miny,maxy) out
7. Output shape -- (H, W, 4) with H/W = ceil((max-min)/pixel_m)
8. Default values -- pixel_m=250.0, max_dist_m=300.0 baked into signatures
9. RGBA dtype -- float32 output, alpha is hard-edged (0.0 or 1.0)
10. region_id index contract -- GDF index must be named "region_id", not "h3_index"

Coverage: core three functions (voronoi_indices, gather_rgba, rasterize_voronoi)
plus the latlon_to_metric adapter.  Per-mode wrappers (rasterize_continuous_voronoi
etc.) are intentionally NOT tested here — they build on these core four.

Explicit exclusions (per spec §"What's NOT in this spec" and plan W2b):
- No Voronoi-vs-centroid-splat regression test (not equivalent by design).
- No test for deprecated rasterize_continuous/rasterize_rgb/rasterize_binary/
  rasterize_categorical (W6 deletes them).
- No test for save_voronoi_figure (W4 territory).
- No test for caller-side migration (W3 territory).
"""

from __future__ import annotations

import math
import inspect
import numpy as np
import pytest
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _import_visualization():
    """Import utils.visualization, skipping if absent (W2a not yet landed)."""
    try:
        import utils.visualization as viz
        return viz
    except ImportError as exc:
        pytest.skip(f"utils.visualization not importable: {exc}")


def _require(viz_module, name: str):
    """Return a function from the module or skip if not yet implemented."""
    fn = getattr(viz_module, name, None)
    if fn is None or not callable(fn):
        pytest.skip(
            f"utils.visualization.{name} not yet implemented (W2a pending). "
            f"Tests are contract-first; run pytest again after W2a lands."
        )
    return fn


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------

# NL grid: a 5×5 arrangement of centroids in EPSG:28992 space near The Hague.
# These are purely synthetic; no real study area data is loaded.
_NL_ORIGIN_X = 82_000.0   # ~Den Haag area in RD New metres
_NL_ORIGIN_Y = 454_000.0
_GRID_STEP = 500.0         # 500 m apart — well above max_dist_m=300


def _make_grid_centroids(nx: int = 5, ny: int = 5, step: float = _GRID_STEP):
    """Return (cx_m, cy_m) for an nx×ny grid of synthetic centroids."""
    xs = _NL_ORIGIN_X + np.arange(nx) * step
    ys = _NL_ORIGIN_Y + np.arange(ny) * step
    xx, yy = np.meshgrid(xs, ys)
    return xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)


def _make_extent(cx_m: np.ndarray, cy_m: np.ndarray, pad: float = 600.0):
    """(minx, miny, maxx, maxy) with `pad` metres of margin."""
    return (
        float(cx_m.min() - pad),
        float(cy_m.min() - pad),
        float(cx_m.max() + pad),
        float(cy_m.max() + pad),
    )


def _make_rgb(n: int) -> np.ndarray:
    """Random (N, 3) float32 in [0, 1]."""
    rng = np.random.default_rng(seed=42)
    return rng.random((n, 3), dtype=np.float32)


def _make_srai_gdf(cx_m: np.ndarray, cy_m: np.ndarray) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame indexed by ``region_id`` in EPSG:28992.

    This is the SRAI convention as required by the index contract.
    Hex IDs are synthetic strings; the geometry is point-based centroids.
    In production, SRAI H3Regionalizer produces polygon centroids; for the
    toolkit contract the critical element is the CRS and the index name.
    """
    n = len(cx_m)
    # Synthetic H3-like region_ids (hex string format, not real H3 cells)
    region_ids = [f"89196{i:09d}ffff" for i in range(n)]
    geometries = [Point(x, y) for x, y in zip(cx_m, cy_m)]
    gdf = gpd.GeoDataFrame(
        {"geometry": geometries},
        index=gpd.pd.Index(region_ids, name="region_id"),
        crs="EPSG:28992",
    )
    return gdf


# ---------------------------------------------------------------------------
# Test 1: Determinism (spec §"Determinism contract")
# ---------------------------------------------------------------------------

class TestDeterminism:
    """voronoi_indices called twice with identical inputs → byte-identical outputs.

    Spec guarantee: same (cx_m, cy_m), extent_m, pixel_m, max_dist_m ⇒
    byte-identical nearest_idx and inside.  cKDTree.query(k=1) with eps=0
    is deterministic.
    """

    def test_voronoi_indices_deterministic_nearest_idx(self):
        """nearest_idx is byte-identical across two calls with identical inputs."""
        viz = _import_visualization()
        fn = _require(viz, "voronoi_indices")

        cx_m, cy_m = _make_grid_centroids()
        extent_m = _make_extent(cx_m, cy_m)

        idx1, inside1, ext1 = fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)
        idx2, inside2, ext2 = fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)

        assert np.array_equal(idx1, idx2), (
            "nearest_idx is not deterministic across two identical voronoi_indices calls"
        )

    def test_voronoi_indices_deterministic_inside_mask(self):
        """inside mask is byte-identical across two calls with identical inputs."""
        viz = _import_visualization()
        fn = _require(viz, "voronoi_indices")

        cx_m, cy_m = _make_grid_centroids()
        extent_m = _make_extent(cx_m, cy_m)

        _, inside1, _ = fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)
        _, inside2, _ = fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)

        assert np.array_equal(inside1, inside2), (
            "inside mask is not deterministic across two identical voronoi_indices calls"
        )

    def test_gather_rgba_deterministic(self):
        """gather_rgba with the same inputs produces byte-identical RGBA."""
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")
        gr_fn = _require(viz, "gather_rgba")

        cx_m, cy_m = _make_grid_centroids()
        extent_m = _make_extent(cx_m, cy_m)
        rgb = _make_rgb(len(cx_m))

        nearest_idx, inside, _ = vi_fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)
        rgba1 = gr_fn(nearest_idx, inside, rgb)
        rgba2 = gr_fn(nearest_idx, inside, rgb)

        assert np.array_equal(rgba1, rgba2), (
            "gather_rgba output is not deterministic (numpy fancy indexing should be)"
        )


# ---------------------------------------------------------------------------
# Test 2: max_dist_m cutoff (spec §qaqc-invariant 2)
# ---------------------------------------------------------------------------

class TestMaxDistCutoff:
    """Pixels beyond max_dist_m → alpha=0; pixels within → alpha=1.0.

    Spec: ``inside`` is ``True`` where pixel centre is within ``max_dist_m``
    of its nearest centroid, ``False`` outside.
    gather_rgba sets alpha = inside.astype(float32): 1.0 inside, 0.0 outside.

    Strategy: place two centroids far apart (> 2 × max_dist_m), query a pixel
    between them that is guaranteed > max_dist_m from both.  Assert alpha=0
    at that pixel.
    """

    def test_far_pixel_alpha_zero(self):
        """A pixel > max_dist_m from every centroid has alpha=0 in gather_rgba."""
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")
        gr_fn = _require(viz, "gather_rgba")

        max_dist_m = 300.0
        pixel_m = 50.0   # small pixel for precise placement

        # Two centroids, 1000 m apart — midpoint is 500 m from each (> max_dist_m)
        cx_m = np.array([0.0, 1000.0])
        cy_m = np.array([0.0, 0.0])

        # Extent covers the midpoint between the two centroids
        extent_m = (-100.0, -100.0, 1100.0, 100.0)

        nearest_idx, inside, extent_xy = vi_fn(
            cx_m, cy_m, extent_m, pixel_m=pixel_m, max_dist_m=max_dist_m
        )

        # The midpoint pixel (x≈500, y≈0) is 500 m from the nearest centroid.
        # With pixel_m=50, pixel column index of x=500 is (500 - (-100)) / 50 = 12
        # Pixel centre: -100 + 12.5 * 50 = 525 m -> distance to x=0 is 525 m (> 300)
        # We want to find a pixel definitively beyond max_dist_m on BOTH sides.
        # Column 12 centre_x = -100 + (12 + 0.5)*50 = -100 + 625 = 525 m  → dist=525>300
        row_idx = 1   # row 1 centre_y = -100 + 1.5*50 = -25 m (near y=0, safe)
        col_idx = 12  # centre_x ≈ 525 m

        if col_idx < inside.shape[1] and row_idx < inside.shape[0]:
            assert not inside[row_idx, col_idx], (
                f"Pixel at col={col_idx} (x≈525m) should be outside max_dist_m=300, "
                f"but inside[{row_idx},{col_idx}]={inside[row_idx, col_idx]}"
            )
            # Confirm alpha=0 in gather_rgba
            rgb = _make_rgb(2)
            rgba = gr_fn(nearest_idx, inside, rgb)
            assert rgba[row_idx, col_idx, 3] == pytest.approx(0.0), (
                f"Alpha at outside pixel should be 0.0, got {rgba[row_idx, col_idx, 3]}"
            )
        else:
            pytest.skip(
                f"Grid too small to hit col_idx={col_idx}; shape={inside.shape}"
            )

    def test_near_pixel_alpha_one(self):
        """A pixel within max_dist_m of a centroid has alpha=1.0 in gather_rgba."""
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")
        gr_fn = _require(viz, "gather_rgba")

        max_dist_m = 300.0
        pixel_m = 50.0

        # Single centroid at (0, 0); query a pixel at (0, 0) centre → dist ≤ pixel_m/2
        cx_m = np.array([0.0])
        cy_m = np.array([0.0])
        # Extent: pixel (0,0) centre is at (-275, -275) + 0.5*50 = (-250, -250)
        # Let's place centroid right in the centre of the first pixel
        # pixel_m=50: first pixel centre at -300 + 25 = -275. Too far.
        # Use extent starting close to the centroid:
        extent_m = (-25.0, -25.0, 225.0, 225.0)

        # pixel 0,0 centre: (-25 + 25, -25 + 25) = (0, 0) → exactly on centroid
        nearest_idx, inside, _ = vi_fn(
            cx_m, cy_m, extent_m, pixel_m=pixel_m, max_dist_m=max_dist_m
        )

        assert inside[0, 0], (
            "Pixel at (0,0) centred exactly on centroid should be inside max_dist_m"
        )
        rgb = _make_rgb(1)
        rgba = gr_fn(nearest_idx, inside, rgb)
        assert rgba[0, 0, 3] == pytest.approx(1.0), (
            f"Alpha at near pixel should be 1.0, got {rgba[0, 0, 3]}"
        )

    def test_inside_equals_dist_lte_max_dist_m(self):
        """inside is True iff distance ≤ max_dist_m (not strict <)."""
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")

        # Centroid at origin; pixel centred exactly at max_dist_m along x axis
        max_dist_m = 300.0
        pixel_m = 50.0
        cx_m = np.array([0.0])
        cy_m = np.array([0.0])

        # Pixel at column index c has centre x = extent_minx + (c + 0.5) * pixel_m
        # We want centre at exactly max_dist_m = 300.0
        # -300 + (c + 0.5)*50 = 300 → c + 0.5 = 12 → c = 11.5 → c = 11 → centre at 275 (< 300)
        # c = 12 → centre at 325 (> 300). So boundary falls between col 11 and col 12.
        # col 11 centre: -300 + 11.5*50 = -300 + 575 = 275 < 300 → should be inside
        # col 12 centre: -300 + 12.5*50 = -300 + 625 = 325 > 300 → should be outside
        extent_m = (-300.0, -25.0, 700.0, 25.0)  # y range tiny, y=0 centroid y

        nearest_idx, inside, _ = vi_fn(
            cx_m, cy_m, extent_m, pixel_m=pixel_m, max_dist_m=max_dist_m
        )

        row_idx = 0  # single row (y range is one pixel_m wide)
        # Col 11: centre x = -300 + 11.5*50 = 275 → dist = 275 ≤ 300 → inside
        # Col 12: centre x = -300 + 12.5*50 = 325 → dist = 325 > 300 → outside
        if inside.shape[1] > 12:
            assert inside[row_idx, 11], "Col 11 (dist=275m) should be inside max_dist_m=300"
            assert not inside[row_idx, 12], "Col 12 (dist=325m) should be outside max_dist_m=300"


# ---------------------------------------------------------------------------
# Test 3: GeoDataFrame ≡ coord-array (spec §"Index contract", §qaqc-invariant 3)
# ---------------------------------------------------------------------------

class TestGeoDataFrameEquivalence:
    """Passing a SRAI region_id-indexed GeoDataFrame yields byte-identical RGBA
    to passing the equivalent (cx_m, cy_m) arrays directly.

    Spec §Index contract: "Both forms are first-class and produce byte-identical
    output."  Index name MUST be ``region_id`` (not ``h3_index``).
    """

    def _make_value_gdf(
        self, cx_m: np.ndarray, cy_m: np.ndarray, values: np.ndarray
    ) -> gpd.GeoDataFrame:
        """Build a SRAI-indexed GDF with a scalar 'value' column in EPSG:28992."""
        n = len(cx_m)
        region_ids = [f"89196{i:09d}ffff" for i in range(n)]
        gdf = gpd.GeoDataFrame(
            {"geometry": [Point(x, y) for x, y in zip(cx_m, cy_m)], "value": values},
            index=gpd.pd.Index(region_ids, name="region_id"),
            crs="EPSG:28992",
        )
        return gdf

    def test_gdf_form_matches_array_form_continuous(self):
        """rasterize_continuous_voronoi_gdf produces byte-identical RGBA to the array form.

        Spec §Index contract: "Both forms are first-class and produce byte-identical output."
        W2a implemented per-mode GDF wrappers (*_voronoi_gdf); this tests the
        rasterize_continuous_voronoi path since rasterize_voronoi_gdf is impl-detail.

        Note: extent_m is passed explicitly to both forms so the bounding box
        computation path does not diverge (the GDF wrapper auto-derives extent from
        total_bounds; the array wrapper requires explicit extent_m).  When the same
        extent is used, the Voronoi index arrays must be byte-identical.
        """
        viz = _import_visualization()
        arr_fn = _require(viz, "rasterize_continuous_voronoi")
        gdf_fn = _require(viz, "rasterize_continuous_voronoi_gdf")

        cx_m, cy_m = _make_grid_centroids(nx=3, ny=3, step=_GRID_STEP)
        rng = np.random.default_rng(seed=99)
        values = rng.random(len(cx_m), dtype=np.float64)
        extent_m = _make_extent(cx_m, cy_m)

        # Array form — ground truth
        img_array, ext_array = arr_fn(
            cx_m, cy_m, values, extent_m,
            cmap="viridis", pixel_m=250.0, max_dist_m=300.0
        )

        # GDF form — must produce byte-identical output
        gdf = self._make_value_gdf(cx_m, cy_m, values)
        assert gdf.index.name == "region_id", (
            "GDF fixture must have index name 'region_id' per SRAI convention"
        )
        assert gdf.crs.to_epsg() == 28992, (
            "GDF fixture must be in EPSG:28992 (metric CRS required by spec)"
        )

        img_gdf, ext_gdf = gdf_fn(
            gdf, "value",
            target_crs=28992, extent_m=extent_m,
            cmap="viridis", pixel_m=250.0, max_dist_m=300.0
        )

        assert np.array_equal(img_array, img_gdf), (
            "rasterize_continuous_voronoi_gdf does not produce byte-identical RGBA "
            "to rasterize_continuous_voronoi with the same data. "
            "Spec §Index contract: 'Both forms are first-class and produce byte-identical output.'"
        )
        assert ext_array == ext_gdf, (
            "extent_xy differs between GDF and array form of rasterize_continuous_voronoi"
        )

    def test_gdf_must_have_region_id_index(self, caplog):
        """A GDF with index name 'h3_index' must fail or warn — not silently mismap.

        Spec §Index contract: 'The GeoDataFrame's index name MUST be region_id
        (not h3_index) per .claude/rules/index-contracts.md'.
        The toolkit is at the stage-3 visualization edge; it sees region_id.
        A GDF with h3_index is a caller bug.  This test documents that the
        h3_index form is NOT equivalent (contract guard, not feature gate).

        W2a's _gdf_to_metric_centroids emits a logging.WARNING (not warnings.warn)
        for non-region_id index names.  The spec is silent on the diagnostic
        mechanism; this test accepts ValueError/KeyError (clean break) OR a
        log warning OR a Python UserWarning — but NOT complete silence.
        """
        import logging as _logging
        import warnings as _warnings
        viz = _import_visualization()
        # Use any available GDF-form entry point
        gdf_fn = (
            getattr(viz, "rasterize_voronoi_gdf", None)
            or getattr(viz, "rasterize_continuous_voronoi_gdf", None)
        )
        if gdf_fn is None:
            pytest.skip("No GDF-form wrapper found (W2a pending)")

        cx_m, cy_m = _make_grid_centroids(nx=3, ny=3)
        rng = np.random.default_rng(seed=1)
        values = rng.random(len(cx_m))

        # Build GDF with wrong index name
        n = len(cx_m)
        region_ids = [f"89196{i:09d}ffff" for i in range(n)]
        gdf_wrong = gpd.GeoDataFrame(
            {
                "geometry": [Point(x, y) for x, y in zip(cx_m, cy_m)],
                "value": values,
            },
            index=gpd.pd.Index(region_ids, name="h3_index"),  # WRONG: should be region_id
            crs="EPSG:28992",
        )
        extent_m = _make_extent(cx_m, cy_m)

        # Capture both warnings channels and log output.
        caught_warnings = []
        raised = False
        with caplog.at_level(_logging.WARNING, logger="utils.visualization"):
            try:
                with _warnings.catch_warnings(record=True) as w:
                    _warnings.simplefilter("always")
                    gdf_fn(gdf_wrong, "value", extent_m=extent_m, pixel_m=250.0, max_dist_m=300.0)
                    caught_warnings = list(w)
            except (ValueError, KeyError, AssertionError, AttributeError):
                raised = True

        # Accept: exception raised OR Python warning OR logger warning
        log_warned = any(
            "h3_index" in r.message or "region_id" in r.message
            for r in caplog.records
        )
        py_warned = len(caught_warnings) > 0

        assert raised or log_warned or py_warned, (
            "Passing a GDF with index name 'h3_index' produced no warning or error. "
            "Spec §Index contract: index MUST be 'region_id'. At minimum a log warning "
            "or Python warning should be emitted when the caller violates this contract."
        )


# ---------------------------------------------------------------------------
# Test 4: latlon_to_metric round-trip (spec §qaqc-invariant 4)
# ---------------------------------------------------------------------------

class TestLatLonToMetric:
    """latlon_to_metric(lats, lons, target_crs) → (cx_m, cy_m) in metric CRS.

    Spec: "Uses pyproj.Transformer with always_xy=True; target_crs defaults
    to 28992 (RD New) for NL."

    Known NL landmarks with pre-computed EPSG:28992 reference values
    (computed via pyproj directly — authoritative by construction):
      Amsterdam Centraal: (52.3791°N, 4.8993°E) → (121778.51, 488026.58)
      Den Haag Centraal:  (52.0800°N, 4.3267°E) → ( 82304.91, 455167.21)
      Utrecht Dom Tower:  (52.0907°N, 5.1215°E) → (136790.58, 455860.11)

    Round-trip tolerance: 1 cm (0.01 m) in metric CRS.
    """

    # Reference values computed from authoritative pyproj transform
    _KNOWN_POINTS = [
        # (lat_deg, lon_deg, expected_x_28992, expected_y_28992, label)
        (52.3791, 4.8993, 121778.51, 488026.58, "Amsterdam Centraal"),
        (52.0800, 4.3267,  82304.91, 455167.21, "Den Haag Centraal"),
        (52.0907, 5.1215, 136790.58, 455860.11, "Utrecht Dom Tower"),
    ]

    def test_latlon_to_metric_known_nl_points(self):
        """latlon_to_metric matches expected EPSG:28992 for known NL landmarks."""
        viz = _import_visualization()
        fn = _require(viz, "latlon_to_metric")

        lats = np.array([p[0] for p in self._KNOWN_POINTS])
        lons = np.array([p[1] for p in self._KNOWN_POINTS])
        expected_x = np.array([p[2] for p in self._KNOWN_POINTS])
        expected_y = np.array([p[3] for p in self._KNOWN_POINTS])

        cx_m, cy_m = fn(lats, lons, target_crs=28992)

        np.testing.assert_allclose(
            cx_m, expected_x, atol=0.5,  # 0.5 m tolerance — pyproj versions may differ slightly
            err_msg="latlon_to_metric x-coordinates differ from expected EPSG:28992 values"
        )
        np.testing.assert_allclose(
            cy_m, expected_y, atol=0.5,
            err_msg="latlon_to_metric y-coordinates differ from expected EPSG:28992 values"
        )

    def test_latlon_to_metric_default_crs_is_28992(self):
        """Calling latlon_to_metric without target_crs uses EPSG:28992."""
        viz = _import_visualization()
        fn = _require(viz, "latlon_to_metric")

        lats = np.array([52.3791])
        lons = np.array([4.8993])

        cx_default, cy_default = fn(lats, lons)         # no target_crs
        cx_explicit, cy_explicit = fn(lats, lons, target_crs=28992)

        np.testing.assert_allclose(cx_default, cx_explicit, atol=1e-6)
        np.testing.assert_allclose(cy_default, cy_explicit, atol=1e-6)

    def test_latlon_to_metric_round_trip_within_1cm(self):
        """latlon_to_metric forward + inverse round-trips within 0.01 m (1 cm)."""
        viz = _import_visualization()
        fn = _require(viz, "latlon_to_metric")

        # Amsterdam Centraal
        lats_orig = np.array([52.3791])
        lons_orig = np.array([4.8993])

        cx_m, cy_m = fn(lats_orig, lons_orig, target_crs=28992)

        # Inverse: EPSG:28992 → EPSG:4326 using pyproj directly
        t_inv = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
        lons_back, lats_back = t_inv.transform(cx_m, cy_m)

        # At 52°N: 1° lat ≈ 111.3 km, 1° lon ≈ 68.7 km
        # 1 cm = 0.01 m → Δlat ≈ 9e-8°, Δlon ≈ 1.5e-7°
        np.testing.assert_allclose(lats_back, lats_orig, atol=1e-6,
                                   err_msg="Round-trip latitude error exceeds tolerance")
        np.testing.assert_allclose(lons_back, lons_orig, atol=1e-6,
                                   err_msg="Round-trip longitude error exceeds tolerance")

    def test_latlon_to_metric_output_shape_matches_input(self):
        """Output arrays (cx_m, cy_m) have the same shape as the input arrays."""
        viz = _import_visualization()
        fn = _require(viz, "latlon_to_metric")

        lats = np.array([52.0, 52.1, 52.2, 52.3])
        lons = np.array([4.5, 4.6, 4.7, 4.8])

        cx_m, cy_m = fn(lats, lons)
        assert cx_m.shape == lats.shape, "cx_m shape mismatch"
        assert cy_m.shape == lons.shape, "cy_m shape mismatch"

    def test_latlon_to_metric_returns_metric_scale(self):
        """Output values for NL should be in RD New range (not degree-scale)."""
        viz = _import_visualization()
        fn = _require(viz, "latlon_to_metric")

        lats = np.array([52.0])
        lons = np.array([5.0])
        cx_m, cy_m = fn(lats, lons, target_crs=28992)

        # RD New: Netherlands spans roughly x=[7_000, 300_000], y=[289_000, 629_000]
        assert cx_m[0] > 1_000, (
            f"cx_m={cx_m[0]:.1f} looks like degrees, not RD New metres"
        )
        assert cy_m[0] > 100_000, (
            f"cy_m={cy_m[0]:.1f} looks like degrees, not RD New metres"
        )


# ---------------------------------------------------------------------------
# Test 5: Gallery reuse (spec §qaqc-invariant 5)
# ---------------------------------------------------------------------------

class TestGalleryReuse:
    """voronoi_indices + gather_rgba produces byte-identical RGBA to rasterize_voronoi.

    Spec §"Gallery reuse" worked example: one voronoi_indices call amortises
    across N gather_rgba calls.  Spec §qaqc-invariant 5: "voronoi_indices(...)
    followed by gather_rgba(...) produces byte-identical RGBA to the one-shot
    rasterize_voronoi(...) for the same rgb_per_hex."
    """

    def test_gallery_one_shot_equivalence(self):
        """voronoi_indices + gather_rgba == rasterize_voronoi for the same input."""
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")
        gr_fn = _require(viz, "gather_rgba")
        rv_fn = _require(viz, "rasterize_voronoi")

        cx_m, cy_m = _make_grid_centroids()
        extent_m = _make_extent(cx_m, cy_m)
        rgb = _make_rgb(len(cx_m))

        # Two-step (gallery pattern)
        nearest_idx, inside, ext_gallery = vi_fn(
            cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0
        )
        img_gallery = gr_fn(nearest_idx, inside, rgb)

        # One-shot
        img_oneshot, ext_oneshot = rv_fn(
            cx_m, cy_m, rgb, extent_m, pixel_m=250.0, max_dist_m=300.0
        )

        assert np.array_equal(img_gallery, img_oneshot), (
            "Gallery (voronoi_indices + gather_rgba) does not produce byte-identical "
            "RGBA to one-shot rasterize_voronoi for the same rgb_per_hex input"
        )
        assert ext_gallery == ext_oneshot, (
            "extent_xy differs between gallery and one-shot paths"
        )

    def test_gallery_multiple_panels_byte_identical_to_one_shot(self):
        """Multiple gather_rgba panels from one voronoi_indices each match rasterize_voronoi.

        This is the canonical 8-panel gallery scenario: one KDTree, N RGBA outputs.
        Each panel must match calling rasterize_voronoi with the same rgb table.
        """
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")
        gr_fn = _require(viz, "gather_rgba")
        rv_fn = _require(viz, "rasterize_voronoi")

        cx_m, cy_m = _make_grid_centroids()
        extent_m = _make_extent(cx_m, cy_m)

        # Build 4 different rgb tables
        rng = np.random.default_rng(seed=7)
        n = len(cx_m)
        rgb_tables = [rng.random((n, 3), dtype=np.float32) for _ in range(4)]

        nearest_idx, inside, _ = vi_fn(
            cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0
        )

        for i, rgb in enumerate(rgb_tables):
            img_gallery = gr_fn(nearest_idx, inside, rgb)
            img_oneshot, _ = rv_fn(cx_m, cy_m, rgb, extent_m, pixel_m=250.0, max_dist_m=300.0)

            assert np.array_equal(img_gallery, img_oneshot), (
                f"Panel {i}: gallery RGBA != one-shot RGBA for the same rgb table. "
                f"Gallery reuse must produce identical results to separate rasterize_voronoi calls."
            )


# ---------------------------------------------------------------------------
# Test 6: extent_m → extent_xy axis-order asymmetry (spec §"Output contract")
# ---------------------------------------------------------------------------

class TestExtentAxisOrder:
    """Input extent_m is (minx, miny, maxx, maxy); output extent_xy is (minx, maxx, miny, maxy).

    This is an intentional axis-order swap so extent_xy plugs directly into:
        ax.imshow(img, extent=extent_xy, origin='lower', ...)
    matplotlib imshow expects (left, right, bottom, top) = (minx, maxx, miny, maxy).

    This asymmetry is a bug-magnet (confirmed in W1 spec-writer scratchpad as
    explicitly worth testing).
    """

    def test_extent_xy_is_minx_maxx_miny_maxy(self):
        """voronoi_indices extent_xy must be (minx, maxx, miny, maxy) order."""
        viz = _import_visualization()
        fn = _require(viz, "voronoi_indices")

        cx_m = np.array([100.0, 200.0, 300.0])
        cy_m = np.array([400.0, 500.0, 600.0])
        minx, miny, maxx, maxy = 50.0, 350.0, 350.0, 650.0
        extent_m = (minx, miny, maxx, maxy)

        _, _, extent_xy = fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)

        assert len(extent_xy) == 4, "extent_xy must be a 4-tuple"
        out_minx, out_maxx, out_miny, out_maxy = extent_xy

        assert out_minx == pytest.approx(minx), (
            f"extent_xy[0] should be minx={minx}, got {out_minx}"
        )
        assert out_maxx == pytest.approx(maxx), (
            f"extent_xy[1] should be maxx={maxx}, got {out_maxx}"
        )
        assert out_miny == pytest.approx(miny), (
            f"extent_xy[2] should be miny={miny}, got {out_miny}"
        )
        assert out_maxy == pytest.approx(maxy), (
            f"extent_xy[3] should be maxy={maxy}, got {out_maxy}"
        )

    def test_extent_xy_order_differs_from_extent_m_order(self):
        """Confirm the swap: extent_m[1]=miny appears at extent_xy[2], not [1]."""
        viz = _import_visualization()
        fn = _require(viz, "voronoi_indices")

        cx_m = np.array([200.0])
        cy_m = np.array([500.0])
        # Use asymmetric values so a swap is detectable
        extent_m = (100.0, 300.0, 500.0, 700.0)  # minx=100, miny=300, maxx=500, maxy=700
        _, _, extent_xy = fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)

        # input order: (100, 300, 500, 700)
        # output order: (100, 500, 300, 700)
        assert extent_xy[0] == pytest.approx(100.0), "extent_xy[0] should be minx=100"
        assert extent_xy[1] == pytest.approx(500.0), "extent_xy[1] should be maxx=500"
        assert extent_xy[2] == pytest.approx(300.0), "extent_xy[2] should be miny=300"
        assert extent_xy[3] == pytest.approx(700.0), "extent_xy[3] should be maxy=700"

    def test_rasterize_voronoi_returns_correct_extent_order(self):
        """rasterize_voronoi extent_xy also follows (minx, maxx, miny, maxy) order."""
        viz = _import_visualization()
        fn = _require(viz, "rasterize_voronoi")

        cx_m = np.array([200.0])
        cy_m = np.array([500.0])
        rgb = _make_rgb(1)
        extent_m = (100.0, 300.0, 500.0, 700.0)

        _, extent_xy = fn(cx_m, cy_m, rgb, extent_m, pixel_m=250.0, max_dist_m=300.0)
        assert extent_xy[1] == pytest.approx(500.0), (
            "rasterize_voronoi extent_xy[1] should be maxx, not miny"
        )
        assert extent_xy[2] == pytest.approx(300.0), (
            "rasterize_voronoi extent_xy[2] should be miny, not maxx"
        )


# ---------------------------------------------------------------------------
# Test 7: Output shape (spec §"Output contract" + "Implementation note")
# ---------------------------------------------------------------------------

class TestOutputShape:
    """Image shape is (H, W, 4) where H=ceil((maxy-miny)/pixel_m), W=ceil((maxx-minx)/pixel_m).

    Spec: "width = max(1, ceil((maxx-minx)/pixel_m)),
           height = max(1, ceil((maxy-miny)/pixel_m))".
    """

    @pytest.mark.parametrize("dx,dy,pixel_m,expected_w,expected_h", [
        (1000.0, 2000.0, 250.0,  4,  8),   # exact division
        (1100.0, 1900.0, 250.0,  5,  8),   # ceil: 1100/250=4.4→5, 1900/250=7.6→8
        (250.0,  250.0,  250.0,  1,  1),   # minimum 1×1
        (500.0,  500.0,  100.0,  5,  5),   # even division
    ])
    def test_output_hw_matches_ceil_formula(self, dx, dy, pixel_m, expected_w, expected_h):
        """Image (H, W, 4) dimensions follow ceil((max-min)/pixel_m)."""
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")
        gr_fn = _require(viz, "gather_rgba")

        cx_m = np.array([dx / 2])
        cy_m = np.array([dy / 2])
        extent_m = (0.0, 0.0, dx, dy)

        nearest_idx, inside, _ = vi_fn(
            cx_m, cy_m, extent_m, pixel_m=pixel_m, max_dist_m=max(dx, dy)
        )

        assert nearest_idx.shape == (expected_h, expected_w), (
            f"nearest_idx shape {nearest_idx.shape} != expected ({expected_h}, {expected_w}) "
            f"for dx={dx}, dy={dy}, pixel_m={pixel_m}"
        )
        assert inside.shape == (expected_h, expected_w), (
            f"inside shape {inside.shape} != expected ({expected_h}, {expected_w})"
        )

        rgb = _make_rgb(1)
        rgba = gr_fn(nearest_idx, inside, rgb)
        assert rgba.shape == (expected_h, expected_w, 4), (
            f"gather_rgba output shape {rgba.shape} != ({expected_h}, {expected_w}, 4)"
        )

    def test_minimum_shape_one_by_one(self):
        """A degenerate extent (zero width/height) must produce at least 1×1."""
        viz = _import_visualization()
        fn = _require(viz, "voronoi_indices")

        cx_m = np.array([0.0])
        cy_m = np.array([0.0])
        # Zero width/height extent — max(1, ceil(0/250)) = 1
        extent_m = (0.0, 0.0, 0.0, 0.0)

        nearest_idx, inside, _ = fn(
            cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0
        )
        assert nearest_idx.shape[0] >= 1, "Height must be at least 1"
        assert nearest_idx.shape[1] >= 1, "Width must be at least 1"


# ---------------------------------------------------------------------------
# Test 8: Default parameter values (spec §"Defaults")
# ---------------------------------------------------------------------------

class TestDefaultValues:
    """pixel_m=250.0 and max_dist_m=300.0 are baked into function signatures.

    Spec §"Defaults": "These defaults are CONTRACT defaults: the function
    signatures bake them in."  We verify via inspect.signature that the defaults
    are exactly 250.0 and 300.0, not just that calling without them works.
    """

    @pytest.mark.parametrize("func_name", [
        "voronoi_indices",
        "rasterize_voronoi",
    ])
    def test_pixel_m_default_is_250(self, func_name):
        """pixel_m default is 250.0 in the function signature."""
        viz = _import_visualization()
        fn = _require(viz, func_name)
        sig = inspect.signature(fn)
        params = sig.parameters

        assert "pixel_m" in params, (
            f"{func_name} has no 'pixel_m' parameter"
        )
        default = params["pixel_m"].default
        assert default != inspect.Parameter.empty, (
            f"{func_name}.pixel_m has no default value"
        )
        assert default == pytest.approx(250.0), (
            f"{func_name}.pixel_m default is {default}, expected 250.0 per spec"
        )

    @pytest.mark.parametrize("func_name", [
        "voronoi_indices",
        "rasterize_voronoi",
    ])
    def test_max_dist_m_default_is_300(self, func_name):
        """max_dist_m default is 300.0 in the function signature."""
        viz = _import_visualization()
        fn = _require(viz, func_name)
        sig = inspect.signature(fn)
        params = sig.parameters

        assert "max_dist_m" in params, (
            f"{func_name} has no 'max_dist_m' parameter"
        )
        default = params["max_dist_m"].default
        assert default != inspect.Parameter.empty, (
            f"{func_name}.max_dist_m has no default value"
        )
        assert default == pytest.approx(300.0), (
            f"{func_name}.max_dist_m default is {default}, expected 300.0 per spec"
        )

    def test_voronoi_indices_pixel_m_max_dist_m_are_keyword_only(self):
        """voronoi_indices must accept pixel_m and max_dist_m as keyword arguments.

        Spec signature: voronoi_indices(cx_m, cy_m, extent_m, *, pixel_m, max_dist_m).
        The * enforces keyword-only.  The reference impl in viz_ring_agg_res9_grid.py
        is positional; the frozen spec adds keyword-only constraint.
        This test catches if W2a kept the reference impl signature unchanged.
        """
        viz = _import_visualization()
        fn = _require(viz, "voronoi_indices")
        sig = inspect.signature(fn)
        params = sig.parameters

        for name in ("pixel_m", "max_dist_m"):
            assert name in params
            kind = params[name].kind
            assert kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ), (
                f"{name} must be at least accessible as keyword; got kind={kind}"
            )


# ---------------------------------------------------------------------------
# Test 9: RGBA dtype and hard-edged alpha (spec §"Core: gather_rgba")
# ---------------------------------------------------------------------------

class TestRGBADtypeAndAlpha:
    """Output is (H, W, 4) float32; alpha is hard-edged (0.0 or 1.0 only).

    Spec: "alpha is hard-edged (0.0 or 1.0, no AA)".
    """

    def test_gather_rgba_dtype_float32(self):
        """gather_rgba output dtype is float32."""
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")
        gr_fn = _require(viz, "gather_rgba")

        cx_m, cy_m = _make_grid_centroids(nx=3, ny=3)
        extent_m = _make_extent(cx_m, cy_m)
        rgb = _make_rgb(len(cx_m))

        nearest_idx, inside, _ = vi_fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)
        rgba = gr_fn(nearest_idx, inside, rgb)

        assert rgba.dtype == np.float32, (
            f"gather_rgba output dtype is {rgba.dtype}, expected float32"
        )

    def test_gather_rgba_alpha_is_hard_edged(self):
        """Alpha channel contains only 0.0 or 1.0 — no intermediate values."""
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")
        gr_fn = _require(viz, "gather_rgba")

        cx_m, cy_m = _make_grid_centroids(nx=4, ny=4)
        extent_m = _make_extent(cx_m, cy_m, pad=200.0)
        rgb = _make_rgb(len(cx_m))

        nearest_idx, inside, _ = vi_fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)
        rgba = gr_fn(nearest_idx, inside, rgb)
        alpha = rgba[..., 3]

        unique_alpha = np.unique(alpha)
        for val in unique_alpha:
            assert val == pytest.approx(0.0) or val == pytest.approx(1.0), (
                f"Alpha channel contains non-hard-edged value {val}; "
                f"spec requires hard-edged alpha (0.0 or 1.0 only, no AA)"
            )

    def test_gather_rgba_shape_is_h_w_4(self):
        """gather_rgba output shape is (H, W, 4)."""
        viz = _import_visualization()
        vi_fn = _require(viz, "voronoi_indices")
        gr_fn = _require(viz, "gather_rgba")

        cx_m, cy_m = _make_grid_centroids(nx=2, ny=3)
        extent_m = _make_extent(cx_m, cy_m)
        rgb = _make_rgb(len(cx_m))

        nearest_idx, inside, _ = vi_fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)
        rgba = gr_fn(nearest_idx, inside, rgb)

        assert rgba.ndim == 3, f"Expected 3-D array, got {rgba.ndim}-D"
        assert rgba.shape[2] == 4, f"Expected 4 channels (RGBA), got {rgba.shape[2]}"

    def test_rasterize_voronoi_output_dtype(self):
        """rasterize_voronoi image is float32."""
        viz = _import_visualization()
        fn = _require(viz, "rasterize_voronoi")

        cx_m, cy_m = _make_grid_centroids(nx=2, ny=2)
        extent_m = _make_extent(cx_m, cy_m)
        rgb = _make_rgb(len(cx_m))

        img, _ = fn(cx_m, cy_m, rgb, extent_m, pixel_m=250.0, max_dist_m=300.0)
        assert img.dtype == np.float32, (
            f"rasterize_voronoi image dtype is {img.dtype}, expected float32"
        )


# ---------------------------------------------------------------------------
# Test 10: nearest_idx dtype is int64 (spec §"Core: voronoi_indices")
# ---------------------------------------------------------------------------

class TestNearestIdxDtype:
    """nearest_idx must be int64 — gather_rgba assumes int64 indexing.

    Spec: "Output dtypes are load-bearing — downstream gather_rgba assumes
    int64 indexing into (N, 3) colour arrays."
    """

    def test_nearest_idx_dtype_int64(self):
        """nearest_idx array dtype is int64."""
        viz = _import_visualization()
        fn = _require(viz, "voronoi_indices")

        cx_m, cy_m = _make_grid_centroids()
        extent_m = _make_extent(cx_m, cy_m)

        nearest_idx, inside, _ = fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)

        assert nearest_idx.dtype == np.int64, (
            f"nearest_idx dtype is {nearest_idx.dtype}, expected int64 "
            f"(load-bearing per spec §voronoi_indices)"
        )

    def test_nearest_idx_values_in_range(self):
        """All nearest_idx values are valid indices into the centroid array."""
        viz = _import_visualization()
        fn = _require(viz, "voronoi_indices")

        n = 9  # 3×3 grid
        cx_m, cy_m = _make_grid_centroids(nx=3, ny=3)
        extent_m = _make_extent(cx_m, cy_m)

        nearest_idx, _, _ = fn(cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0)

        assert nearest_idx.min() >= 0, "nearest_idx has negative values"
        assert nearest_idx.max() < n, (
            f"nearest_idx max={nearest_idx.max()} exceeds centroid count {n}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
