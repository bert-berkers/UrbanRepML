"""Build label-encoded raster PNGs for hierarchical cluster-brush viz v2 (Wave 1).

Purpose: Load netherlands ring_agg res9 embeddings, cluster with MiniBatchKMeans,
aggregate labels upward via H3 hierarchy to res5/6/7, rasterise each level into
a label-encoded PNG (R channel = cluster id 0..9, alpha=0 outside extent), and
emit a small ``labels.json`` sidecar with extent and counts. The browser
(viz_raster.html, Wave 2) stacks these PNGs on 3D-transformed canvases and
colorizes via palette swap on brush click.

Aligned with ``utils.visualization.rasterize_categorical`` and
``scripts/stage3/plot_cluster_maps.py``. Same raster canvas (2000x2400 @ EPSG:28992),
same extent across all 3 resolutions (computed once from the NL boundary) so the
CSS-stacked slabs align pixel-for-pixel.

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image, PngImagePlugin

# Project infra
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.paths import StudyAreaPaths  # noqa: E402
from utils.spatial_db import SpatialDB  # noqa: E402
from utils.visualization import (  # noqa: E402
    load_boundary,
    rasterize_categorical,
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
# Helpers (reused verbatim from v1 build.py)
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
    """Group res9 labels by parent at target_res; return DataFrame indexed by parent.

    Columns: cluster (majority label), n_children, entropy.
    """
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
# Identity colormap for label-encoded PNGs
# ---------------------------------------------------------------------------

def _identity_label_cmap(k: int) -> ListedColormap:
    """Colormap where entry i maps to (i/255, i/255, i/255).

    ``rasterize_categorical`` normalises labels to ``label / (k-1)`` then samples
    the colormap. Our cmap must therefore return ``(i/255, i/255, i/255)`` when
    sampled at ``i / (k-1)``. A ``ListedColormap`` with k entries does exactly
    that: ``cmap(norm_vals)`` where ``norm_vals = i / (k-1)`` picks the i-th
    list entry. So we set entry i = (i/255, i/255, i/255).
    """
    entries = [(i / 255.0, i / 255.0, i / 255.0, 1.0) for i in range(k)]
    return ListedColormap(entries, name="label_identity")


def _rgba_float_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float32 RGBA (H,W,4) in [0,1] to uint8 (H,W,4) exactly.

    Uses round then clip — no gamma or sRGB transform. Label integrity
    is preserved: entry i/255 becomes exactly uint8 i.
    """
    out = np.rint(image * 255.0).astype(np.uint8)
    return out


def _save_label_png(arr_uint8: np.ndarray, path: Path) -> None:
    """Save a label-encoded PNG with no color profile (strip iCCP/sRGB chunks).

    Browsers sometimes apply sRGB transfer to PNGs with color profiles, which
    would shift R-channel values by +-1 and corrupt labels. Writing with PIL
    and an empty PngInfo omits the iCCP chunk entirely.
    """
    img = Image.fromarray(arr_uint8, mode="RGBA")
    # Vertical flip: rasterize_categorical uses origin="lower" (y-axis up, as
    # in matplotlib); PNG pixel rows run top-to-bottom. Flip so row 0 of the
    # PNG corresponds to the top of the extent (maxy) — conventional image
    # orientation for CSS stacking.
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
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
    # Defensive clamp — labels should already be in [0, k)
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

    # PCA (v1 logic verbatim)
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

    # Clustering (v1 logic verbatim)
    clusters = perform_minibatch_clustering(X_pca, [K], standardize=False)
    labels = clusters[K].astype(np.int64)
    uniq, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(uniq, counts)}
    print(f"[cluster] k={K} distribution={dist}")

    # Compute shared extent ONCE from the NL boundary (EPSG:28992 — metric CRS)
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

    # Stamp sizing: match hex footprint at 2000x2400 NL canvas.
    # v1 (11-r) gave 6/5/4 — way too small, looked like confetti.
    # H3 edge length scales ~sqrt(7) per resolution step. Canvas is ~135m/px,
    # hex footprint at res7/6/5 ~= 24/65/170 px diameter → stamp (radius) ~12/33/85.
    STAMP_BY_RES = {5: 90, 6: 34, 7: 13}
    for r in TARGET_RESOLUTIONS:
        stamp = STAMP_BY_RES.get(r, max(1, 11 - r))
        agg_df = _aggregate_to_parent_res(hex_ids, labels, r)
        lbl_counter = Counter(agg_df["cluster"].tolist())
        print(
            f"[agg]  res{r}: {len(agg_df):,} hexes, "
            f"cluster_dist={dict(sorted(lbl_counter.items()))}"
        )

        # Bulk centroid query in EPSG:28992 (RD New)
        t0 = time.time()
        cx, cy = db.centroids(agg_df.index, resolution=r, crs=28992)
        print(f"  [res{r}] centroids {time.time() - t0:.1f}s, stamp={stamp}")

        labels_r = agg_df["cluster"].to_numpy().astype(np.int64)

        image_f = rasterize_categorical(
            cx=cx, cy=cy, labels=labels_r,
            extent=extent, n_clusters=K,
            width=RASTER_W, height=RASTER_H,
            cmap=cmap_identity, stamp=stamp,
        )
        image_u8 = _rgba_float_to_uint8(image_f)

        # Defensive check: stamped R channel values should be in [0, K)
        alpha_mask = image_u8[:, :, 3] > 0
        uniq_r = np.unique(image_u8[:, :, 0][alpha_mask])
        assert uniq_r.max() < K, f"res{r}: label drift, got R values {uniq_r}"
        assert uniq_r.min() >= 0, f"res{r}: negative labels {uniq_r}"

        # Pixel counts per cluster (alpha > 0 only)
        R = image_u8[:, :, 0][alpha_mask]
        counts_r = np.bincount(R, minlength=K).tolist()
        pixel_cluster_counts[str(r)] = counts_r
        total_pix = int(alpha_mask.sum())
        print(f"  [res{r}] total_data_pixels={total_pix:,} per_cluster={counts_r}")

        out_png = _SCRIPT_DIR / f"labels_res{r}.png"
        _save_label_png(image_u8, out_png)
        print(f"  [res{r}] wrote {out_png.name} ({out_png.stat().st_size:,} bytes)")

        # Sanity preview in tab10 — gitignored, user-inspectable
        preview_png = _SCRIPT_DIR / f"debug_preview_res{r}.png"
        _save_preview_png(image_u8, preview_png, K)
        print(f"  [res{r}] wrote {preview_png.name} ({preview_png.stat().st_size:,} bytes)")

        per_res_stats[r] = {
            "parent_hex_count": int(len(agg_df)),
            "total_data_pixels": total_pix,
            "stamp": stamp,
            "cluster_counts_by_pixel": counts_r,
            "cluster_counts_by_hex": {int(k_): int(v) for k_, v in sorted(lbl_counter.items())},
        }

    # Sidecar JSON (per plan schema)
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
        },
    }
    sidecar_path = _SCRIPT_DIR / "labels.json"
    sidecar_path.write_text(json.dumps(sidecar, separators=(",", ":")), encoding="utf-8")
    print(f"[write] {sidecar_path.name} ({sidecar_path.stat().st_size} bytes)")

    print(f"[done] total={time.time() - t_start:.1f}s")
    print("[done] per-resolution stats:")
    for r, s in per_res_stats.items():
        print(f"  res{r}: {s['parent_hex_count']:,} hexes, "
              f"{s['total_data_pixels']:,} pixels, stamp={s['stamp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
