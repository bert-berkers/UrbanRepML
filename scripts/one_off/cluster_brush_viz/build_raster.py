"""Build label-encoded raster PNGs for hierarchical cluster-brush viz v2 (Wave 3 rewrite).

Purpose: Load netherlands ring_agg res9 embeddings, cluster with MiniBatchKMeans,
aggregate labels upward via H3 hierarchy to res5/6/7, rasterise each level into
a label-encoded PNG (R channel = cluster id 0..9, alpha=0 outside extent), and
emit a small ``labels.json`` sidecar with extent and counts.

Wave 3 rewrite rationale
------------------------
Wave 1/2 used ``utils.visualization.rasterize_categorical``, which stamps a
square block around each hex CENTROID. At res5/6 in the 2000×2400 NL canvas,
centroid spacing (~65–170 px at res6/res5) is much larger than any reasonable
stamp radius; the previous "stamp=90/34/13" heuristic produced overlapping
stamps that looked like stippled confetti, not continuous hex coverage. User
play-test confirmed bug 1: "fine colored STIPPLE with black gaps".

This rewrite uses proper H3 hex POLYGON fills. We pull polygons from
``SpatialDB.geometry(hex_ids, resolution=r, crs=28992)`` and rasterise them
via matplotlib ``PolyCollection`` with ``antialiased=False`` + matching edge
colour so neighbours merge seamlessly. Identity colormap trick is preserved:
face colour = ``(i/255, i/255, i/255)`` → PNG R channel = cluster id for
browser-side palette swap.

Lifetime: temporary
Stage: 3
Expiry: ~2026-05-17 (30 days from creation)

Usage:
    uv run python scripts/one_off/cluster_brush_viz/build_raster.py

Outputs (all written next to this script, never into data/):
    - labels_res5.png, labels_res6.png, labels_res7.png  (RGBA, 2400x2000)
    - labels.json  (extent, raster shape, pixel counts per cluster per resolution)
    - debug_preview_res{5,6,7}.png  (finished-color tab10 sanity previews)
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import h3  # ALLOWED: hierarchy traversal only (cell_to_parent), per .claude/rules/srai-spatial.md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from PIL import Image, PngImagePlugin
from shapely.geometry import MultiPolygon, Polygon

# Project infra
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.paths import StudyAreaPaths  # noqa: E402
from utils.spatial_db import SpatialDB  # noqa: E402
from utils.visualization import (  # noqa: E402
    load_boundary,
    RASTER_H,
    RASTER_W,
)
from stage3_analysis.visualization.clustering_utils import (  # noqa: E402
    apply_pca_reduction,
    perform_minibatch_clustering,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STUDY_AREA = "netherlands"
MODEL = "ring_agg"
SOURCE_RES = 9
YEAR = "20mix"
TARGET_RESOLUTIONS = [5, 6, 7]
K = 10
PCA_COMPONENTS_DEFAULT = 32
PCA_COMPONENTS_FALLBACK = 64
PCA_VARIANCE_FLOOR = 0.9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy_nats(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h


def _aggregate_to_parent_res(
    res9_hexes: np.ndarray,
    res9_labels: np.ndarray,
    target_res: int,
) -> pd.DataFrame:
    """Group res9 labels by parent at target_res; return DataFrame indexed by parent."""
    print(f"  [res{target_res}] computing parents for {len(res9_hexes):,} res9 hexes...")
    t0 = time.time()
    parents = np.empty(len(res9_hexes), dtype=object)
    for i, hid in enumerate(res9_hexes):
        parents[i] = h3.cell_to_parent(hid, target_res)

    df = pd.DataFrame({"parent": parents, "label": res9_labels})
    rows: list[dict] = []
    grouped = df.groupby("parent", sort=False)
    for parent_hex, sub in grouped:
        counts = Counter(sub["label"].tolist())
        majority_label, _ = counts.most_common(1)[0]
        n_children = int(sub.shape[0])
        entropy = _shannon_entropy_nats(list(counts.values()))
        rows.append({
            "region_id": str(parent_hex),
            "cluster": int(majority_label),
            "n_children": n_children,
            "entropy": round(float(entropy), 4),
        })
    out = pd.DataFrame(rows).set_index("region_id")
    dt = time.time() - t0
    print(f"  [res{target_res}] {len(out):,} parent hexes, aggregation {dt:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Polygon-fill rasteriser
# ---------------------------------------------------------------------------

def _polygon_exterior_xy(geom) -> list[np.ndarray]:
    """Return list of exterior-ring xy arrays for a (Multi)Polygon.

    H3 hex polygons are simple convex polygons; we only need exteriors.
    For boundary-clipped hexes (rare edge cases), MultiPolygon parts are
    each emitted as their own filled polygon.
    """
    rings: list[np.ndarray] = []
    if isinstance(geom, Polygon):
        rings.append(np.asarray(geom.exterior.coords))
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            rings.append(np.asarray(part.exterior.coords))
    return rings


def rasterize_hex_polygons(
    geoms: list,
    labels: np.ndarray,
    extent: tuple,
    n_clusters: int,
    cmap: ListedColormap,
    width: int = RASTER_W,
    height: int = RASTER_H,
) -> np.ndarray:
    """Rasterise filled hex polygons to RGBA uint8.

    Uses a headless matplotlib Figure sized exactly ``width × height`` px.
    PolyCollection is drawn with ``antialiased=False`` and matching
    edge/face colours so adjacent hexes tile seamlessly (no seams, no gaps).

    Args:
        geoms: Shapely (Multi)Polygon objects in EPSG:28992.
        labels: Integer cluster labels, one per geom.
        extent: (minx, miny, maxx, maxy) in EPSG:28992.
        n_clusters: Total cluster count (for cmap sampling denominator).
        cmap: Identity ``ListedColormap`` where entry ``i`` = ``(i/255, i/255, i/255)``.
        width, height: Output canvas size in pixels.

    Returns:
        (height, width, 4) uint8 RGBA. Label ``i`` → R channel == ``i``.
        Alpha=0 outside any polygon; alpha=255 inside.
    """
    minx, miny, maxx, maxy = extent

    # Build polygon vertex arrays + face colours
    verts: list[np.ndarray] = []
    face_colors: list[tuple] = []
    for geom, lab in zip(geoms, labels):
        if geom is None or geom.is_empty:
            continue
        # Normalize to same denominator as rasterize_categorical: label / (k-1)
        norm = float(lab) / max(n_clusters - 1, 1)
        rgba = cmap(norm)  # (r, g, b, a), r=g=b=i/255 with identity cmap
        for ring in _polygon_exterior_xy(geom):
            verts.append(ring)
            face_colors.append(rgba)

    if not verts:
        return np.zeros((height, width, 4), dtype=np.uint8)

    # Figure sized to exact pixel count (dpi=100 → width/100 inches)
    dpi = 100
    fig = plt.figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        frameon=False,
    )
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire canvas
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("auto")  # let pixels fill the rectangle exactly
    ax.set_axis_off()
    # Transparent background - matches alpha=0 outside hex coverage
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    pc = PolyCollection(
        verts,
        facecolors=face_colors,
        edgecolors=face_colors,  # match face → seam-free tiling at neighbour boundaries
        linewidths=0.5,  # slight overdraw covers sub-pixel gaps between neighbours
        antialiased=False,  # critical: preserves exact label values in R channel
        closed=True,
    )
    ax.add_collection(pc)

    # Render to RGBA buffer
    canvas = fig.canvas
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    # Buffer may be (h, w, 4) matching figsize*dpi; verify and crop/pad to request
    bh, bw = buf.shape[:2]
    if (bh, bw) != (height, width):
        # Matplotlib sometimes rounds figsize*dpi; resize via PIL without resampling
        # (should never happen with integer width/height and dpi=100, but defensive).
        img = Image.fromarray(buf, mode="RGBA").resize((width, height), Image.NEAREST)
        buf = np.asarray(img, dtype=np.uint8)

    plt.close(fig)
    return buf


# ---------------------------------------------------------------------------
# Identity colormap for label-encoded PNGs
# ---------------------------------------------------------------------------

def _identity_label_cmap(k: int) -> ListedColormap:
    """Colormap where entry i maps to (i/255, i/255, i/255)."""
    entries = [(i / 255.0, i / 255.0, i / 255.0, 1.0) for i in range(k)]
    return ListedColormap(entries, name="label_identity")


def _save_label_png(arr_uint8: np.ndarray, path: Path) -> None:
    """Save label-encoded PNG with no color profile (strip iCCP/sRGB chunks)."""
    img = Image.fromarray(arr_uint8, mode="RGBA")
    # Rasteriser returns image with matplotlib's native top-down orientation
    # (ax.set_ylim(miny, maxy) means row 0 of the buffer is AT maxy, i.e. top).
    # That matches PNG convention directly — NO flip needed for this path.
    pnginfo = PngImagePlugin.PngInfo()  # empty — no iCCP/sRGB/text chunks
    img.save(path, format="PNG", pnginfo=pnginfo, optimize=True)


def _save_preview_png(arr_uint8_labels: np.ndarray, path: Path, k: int) -> None:
    """Render a finished-color tab10 preview of a label-encoded raster."""
    tab10 = plt.get_cmap("tab10")
    palette = (np.array([tab10(i / max(k - 1, 1))[:3] for i in range(k)]) * 255).astype(np.uint8)

    R = arr_uint8_labels[:, :, 0]
    A = arr_uint8_labels[:, :, 3]
    out = np.zeros_like(arr_uint8_labels)
    mask = A > 0
    labels = R[mask]
    labels = np.clip(labels, 0, k - 1)
    out[mask, :3] = palette[labels]
    out[mask, 3] = 255

    img = Image.fromarray(out, mode="RGBA")
    img.save(path, format="PNG", optimize=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = time.time()
    paths = StudyAreaPaths(STUDY_AREA)
    src = paths.fused_embedding_file(MODEL, SOURCE_RES, YEAR)
    print(f"[load] {src}")
    if not src.exists():
        raise FileNotFoundError(
            f"ring_agg res{SOURCE_RES} {YEAR} parquet not found at {src}. "
            "No fallback — refusing to silently substitute another embedding."
        )

    df = pd.read_parquet(src)
    print(f"[load] shape={df.shape} index={df.index.name!r}")

    if df.index.name != "region_id":
        for cand in ("region_id", "h3", "hex"):
            if cand in df.columns:
                df = df.set_index(cand)
                break
        else:
            raise ValueError(
                f"Expected region_id index or an h3-like column; got index={df.index.name!r} "
                f"cols[:5]={list(df.columns[:5])}"
            )
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"[load] dropping non-feature columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    hex_ids = df.index.astype(str).to_numpy()
    X = df.to_numpy(dtype=np.float32, copy=False)
    print(f"[load] {X.shape[0]:,} hexes x {X.shape[1]}D features")

    # PCA
    n_comp = PCA_COMPONENTS_DEFAULT
    X_pca, pca = apply_pca_reduction(X, n_components=n_comp)
    var_retained = float(pca.explained_variance_ratio_.sum())
    if var_retained < PCA_VARIANCE_FLOOR:
        print(
            f"[pca] variance {var_retained:.3f} < {PCA_VARIANCE_FLOOR}; "
            f"bumping to {PCA_COMPONENTS_FALLBACK}"
        )
        n_comp = PCA_COMPONENTS_FALLBACK
        X_pca, pca = apply_pca_reduction(X, n_components=n_comp)
        var_retained = float(pca.explained_variance_ratio_.sum())
    print(f"[pca] n_components={n_comp} variance_retained={var_retained:.3f}")

    # Clustering
    clusters = perform_minibatch_clustering(X_pca, [K], standardize=False)
    labels = clusters[K].astype(np.int64)
    uniq, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(uniq, counts)}
    print(f"[cluster] k={K} distribution={dist}")

    # Shared extent from NL boundary (EPSG:28992)
    boundary = load_boundary(paths, crs=28992)
    if boundary is None:
        raise FileNotFoundError(
            f"Could not load NL boundary via paths.area_gdf_file(); slabs cannot be aligned."
        )
    minx, miny, maxx, maxy = [float(v) for v in boundary.total_bounds]
    extent = (minx, miny, maxx, maxy)
    print(f"[extent] EPSG:28992 minx={minx:.0f} miny={miny:.0f} maxx={maxx:.0f} maxy={maxy:.0f}")

    cmap_identity = _identity_label_cmap(K)
    db = SpatialDB.for_study_area(STUDY_AREA)

    pixel_cluster_counts: dict[str, list[int]] = {}
    per_res_stats: dict[int, dict] = {}

    for r in TARGET_RESOLUTIONS:
        agg_df = _aggregate_to_parent_res(hex_ids, labels, r)
        lbl_counter = Counter(agg_df["cluster"].tolist())
        print(
            f"[agg]  res{r}: {len(agg_df):,} hexes, "
            f"cluster_dist={dict(sorted(lbl_counter.items()))}"
        )

        # Pull polygon geometry for these parent hexes in EPSG:28992
        t0 = time.time()
        geom_gdf = db.geometry(agg_df.index, resolution=r, crs=28992)
        print(f"  [res{r}] geometry query {time.time() - t0:.1f}s")

        # Align labels to geometry row order (db.geometry reindexes to input)
        labels_r = agg_df["cluster"].to_numpy().astype(np.int64)
        geoms = geom_gdf.geometry.tolist()

        # Drop rows with missing geometry (rare — parent hex not in regions parquet)
        valid = [(g, l) for g, l in zip(geoms, labels_r) if g is not None and not g.is_empty]
        if len(valid) < len(geoms):
            print(f"  [res{r}] dropped {len(geoms) - len(valid)} missing-geom rows")
        valid_geoms = [g for g, _ in valid]
        valid_labels = np.array([l for _, l in valid], dtype=np.int64)

        t0 = time.time()
        image_u8 = rasterize_hex_polygons(
            geoms=valid_geoms,
            labels=valid_labels,
            extent=extent,
            n_clusters=K,
            cmap=cmap_identity,
            width=RASTER_W,
            height=RASTER_H,
        )
        print(f"  [res{r}] polygon raster {time.time() - t0:.1f}s")

        # Defensive checks: R values in [0, K)
        alpha_mask = image_u8[:, :, 3] > 0
        if alpha_mask.any():
            uniq_r = np.unique(image_u8[:, :, 0][alpha_mask])
            assert uniq_r.max() < K, f"res{r}: label drift, got R values {uniq_r}"
            assert uniq_r.min() >= 0, f"res{r}: negative labels {uniq_r}"

        # Pixel counts per cluster
        R = image_u8[:, :, 0][alpha_mask]
        counts_r = np.bincount(R, minlength=K).tolist()
        pixel_cluster_counts[str(r)] = counts_r
        total_pix = int(alpha_mask.sum())
        frac = total_pix / (RASTER_H * RASTER_W)
        print(f"  [res{r}] total_data_pixels={total_pix:,} ({frac:.1%}) per_cluster={counts_r}")

        out_png = _SCRIPT_DIR / f"labels_res{r}.png"
        _save_label_png(image_u8, out_png)
        print(f"  [res{r}] wrote {out_png.name} ({out_png.stat().st_size:,} bytes)")

        preview_png = _SCRIPT_DIR / f"debug_preview_res{r}.png"
        _save_preview_png(image_u8, preview_png, K)
        print(f"  [res{r}] wrote {preview_png.name} ({preview_png.stat().st_size:,} bytes)")

        per_res_stats[r] = {
            "parent_hex_count": int(len(agg_df)),
            "rendered_hex_count": int(len(valid_geoms)),
            "total_data_pixels": total_pix,
            "coverage_fraction": round(frac, 4),
            "cluster_counts_by_pixel": counts_r,
            "cluster_counts_by_hex": {int(k_): int(v) for k_, v in sorted(lbl_counter.items())},
        }

    # Sidecar JSON
    sidecar = {
        "k": K,
        "extent": [minx, miny, maxx, maxy],
        "raster_shape": [RASTER_H, RASTER_W],
        "resolutions": TARGET_RESOLUTIONS,
        "pixel_cluster_counts": pixel_cluster_counts,
        "source": {
            "study_area": STUDY_AREA,
            "model": MODEL,
            "resolution": SOURCE_RES,
            "year": YEAR,
            "pca_components": n_comp,
            "pca_variance_retained": round(var_retained, 4),
            "crs": 28992,
            "render_method": "polygon_fill",  # v2 W3: was "centroid_stamp"
        },
    }
    sidecar_path = _SCRIPT_DIR / "labels.json"
    sidecar_path.write_text(json.dumps(sidecar, separators=(",", ":")), encoding="utf-8")
    print(f"[write] {sidecar_path.name} ({sidecar_path.stat().st_size} bytes)")

    print(f"[done] total={time.time() - t_start:.1f}s")
    print("[done] per-resolution stats:")
    for r, s in per_res_stats.items():
        print(f"  res{r}: {s['rendered_hex_count']:,} hexes, "
              f"{s['total_data_pixels']:,} px ({s['coverage_fraction']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
