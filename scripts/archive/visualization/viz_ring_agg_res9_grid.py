"""Cheeky one-off: rasterize res9 NL ring_agg embedding into 2-panel + 8-panel gallery.

Lifetime: temporary (30-day shelf life, expires ~2026-05-24).
Stage: stage3 (visualisation consumption -- no model/probe work).
Inputs:  data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet
Outputs:
  data/study_areas/netherlands/stage3_analysis/2026-04-24/grid_viz_ring_agg_res9.png         (2-panel canonical)
  data/study_areas/netherlands/stage3_analysis/2026-04-24/grid_viz_ring_agg_res9_gallery.png (2x4 palette gallery)

Canonical 2-panel:
  Left  -- k-means cluster map (k=10, tab10 categorical colours)
  Right -- PC1-3 RGB map (rank-normalised per channel, gamma-boosted)

Palette gallery (2x4):
  Top row    -- k=10 KMeans labels rendered through tab10, Set1, Dark2, Paired
  Bottom row -- continuous PC encodings: PC1.PC2.PC3->RGB, PC1->turbo, PC1->cividis, HSV(PC1,PC2,PC3)

Rasterization is KDTree-Voronoi nearest-neighbour in EPSG:28992 (RD New).
For each output pixel we query the nearest hex centroid; if within
~300 m we paint the hex's value, otherwise leave transparent.  This
eliminates the three bugs of centroid-bin + splat: (1) directional
south-east bleed from asymmetric splat offsets, (2) density-dependent
speckle holes from hexagonal-vs-rectangular packing, and (3) lat-lon
aspect distortion (1 deg lon =/= 1 deg lat at 52 N).

PC->RGB uses per-channel rank normalisation so PC2 and PC3 are not
swamped by PC1's larger dynamic range -- the RGB readout shows
vivid green and red structure, not a blue-dominant blob.

Gallery optimisation: the KDTree query is run ONCE producing a
``(H, W)`` int array of nearest-hex indices + alpha mask, then 8
``rgb_per_hex[idx]`` gathers produce all 8 panels.  Per-panel cost is
trivial fancy-indexing; whole gallery dominated by the single Voronoi.

Run with no args for the canonical 2-panel; pass ``--gallery`` for the
2x4 palette grid.  ``--both`` produces both PNGs back-to-back.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB

STUDY_AREA = "netherlands"
RES = 9
K_CLUSTERS = 10
# Pixel size in metres (EPSG:28992).  NL bbox is ~300 km E-W x ~350 km N-S
# so 250 m/px gives roughly 1200 x 1400 pixels.
PIXEL_M = 250.0
# Voronoi cutoff: res9 hex edge-to-centroid is ~87 m and corner-to-centroid
# is ~174 m, so 300 m conservatively covers any in-tessellation pixel while
# keeping the NL silhouette crisp (anything >300 m from a centroid is
# outside the tessellation proper).
MAX_DIST_M = 300.0
GAMMA = 0.75
BG = "#23262e"


def voronoi_indices(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    extent_m: tuple[float, float, float, float],
    pixel_m: float = PIXEL_M,
    max_dist_m: float = MAX_DIST_M,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Compute the nearest-hex index per pixel + alpha mask once.

    The KDTree query is the dominant cost (~5s for NL res9 at 250 m/px);
    splitting it from the colour-mapping step lets a gallery render
    8 panels for the price of one Voronoi by re-using ``nearest_idx``.

    Args:
        cx_m, cy_m: Hex centroid coordinates in a metric CRS (e.g. EPSG:28992).
        extent_m: ``(minx, miny, maxx, maxy)`` in the same metric CRS.
        pixel_m: Output pixel size (metres).
        max_dist_m: Distance threshold for Voronoi cell validity.

    Returns:
        ``(nearest_idx, inside, extent_xy)`` where:
          - ``nearest_idx`` is ``(H, W)`` int64 -- index into the hex array.
          - ``inside`` is ``(H, W)`` bool -- True where pixel is within
            *max_dist_m* of its nearest centroid (alpha mask).
          - ``extent_xy`` is ``(minx, maxx, miny, maxy)`` for ``imshow``.
    """
    minx, miny, maxx, maxy = extent_m
    width = max(1, int(np.ceil((maxx - minx) / pixel_m)))
    height = max(1, int(np.ceil((maxy - miny) / pixel_m)))

    # Pixel-centre coordinates in metric CRS (origin='lower' convention).
    xs = minx + (np.arange(width) + 0.5) * pixel_m
    ys = miny + (np.arange(height) + 0.5) * pixel_m
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    tree = cKDTree(np.column_stack([cx_m, cy_m]))
    dist, idx = tree.query(pts, k=1)

    nearest_idx = idx.reshape(height, width).astype(np.int64)
    inside = (dist <= max_dist_m).reshape(height, width)
    return nearest_idx, inside, (minx, maxx, miny, maxy)


def gather_rgba(
    nearest_idx: np.ndarray,
    inside: np.ndarray,
    rgb_per_hex: np.ndarray,
) -> np.ndarray:
    """Gather a ``(H, W, 4)`` RGBA from precomputed nearest-hex indices.

    Args:
        nearest_idx: ``(H, W)`` int -- per-pixel hex index.
        inside: ``(H, W)`` bool -- per-pixel alpha mask.
        rgb_per_hex: ``(N, 3)`` float in ``[0, 1]``.

    Returns:
        ``(H, W, 4)`` RGBA float32; alpha=0 outside the Voronoi cutoff.
    """
    h, w = nearest_idx.shape
    img = np.zeros((h, w, 4), dtype=np.float32)
    img[..., :3] = rgb_per_hex[nearest_idx]
    img[..., 3] = inside.astype(np.float32)
    return img


def rasterize_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    rgb_per_hex: np.ndarray,
    extent_m: tuple[float, float, float, float],
    pixel_m: float = PIXEL_M,
    max_dist_m: float = MAX_DIST_M,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """One-shot KDTree-nearest-hex Voronoi rasterization in a metric CRS.

    Thin wrapper around ``voronoi_indices`` + ``gather_rgba`` for callers
    that only need a single colour mapping (the canonical 2-panel main).

    Args:
        cx_m, cy_m: Hex centroid coordinates in a metric CRS (e.g. EPSG:28992).
        rgb_per_hex: ``(N, 3)`` float array with R, G, B in ``[0, 1]``.
        extent_m: ``(minx, miny, maxx, maxy)`` in the same metric CRS.
        pixel_m: Output pixel size (metres).
        max_dist_m: Distance threshold for Voronoi cell validity.

    Returns:
        ``(image, extent)``.  *image* is ``(H, W, 4)`` RGBA float32.  *extent*
        is ``(minx, maxx, miny, maxy)`` suitable for matplotlib ``imshow``.
    """
    nearest_idx, inside, extent_xy = voronoi_indices(
        cx_m, cy_m, extent_m, pixel_m=pixel_m, max_dist_m=max_dist_m
    )
    return gather_rgba(nearest_idx, inside, rgb_per_hex), extent_xy


def rank_rgb(pcs: np.ndarray, gamma: float = GAMMA,
             lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """Per-channel rank normalisation -> [0,1] RGB with gamma boost.

    Rank-based (equal-frequency) normalisation prevents PC1 from
    dominating the dynamic range: each channel becomes a uniform
    distribution over [0,1] regardless of PC variance.  We then squash
    ranks to [lo, hi] to avoid pure white / pure black outlier pixels
    (the extreme 5% of each channel gets clipped), and finally apply
    gamma to compress midtones toward 1 so the image reads as vivid.
    """
    n = len(pcs)
    out = np.empty_like(pcs, dtype=np.float32)
    for i in range(pcs.shape[1]):
        # average rank / n -> uniform on (0, 1]
        r = (rankdata(pcs[:, i], method="average") / n).astype(np.float32)
        # Squash to [lo, hi] so outliers don't blow out to pure white/black.
        out[:, i] = lo + (hi - lo) * r
    return np.power(out, gamma)


def rank_unit(values: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Rank-normalise a 1-D vector to a uniform distribution on ``[lo, hi]``.

    Useful for mapping a single PC into a colormap (turbo, cividis, etc.)
    or into one channel of HSV.  Uniformising via rank means the histogram
    is flat across the [lo, hi] range -- the colormap's full dynamic range
    is exercised regardless of the underlying PC's distribution shape.
    """
    n = len(values)
    r = (rankdata(values, method="average") / n).astype(np.float32)
    return lo + (hi - lo) * r


def hsv_pc_rgb(
    pc1: np.ndarray, pc2: np.ndarray, pc3: np.ndarray,
    s_lo: float = 0.2, s_hi: float = 1.0,
    v_lo: float = 0.3, v_hi: float = 1.0,
) -> np.ndarray:
    """HSV colour mapping: PC1->Hue, PC2->Saturation, PC3->Value.

    Saturation floor (``s_lo``) keeps low-PC2 cells from washing out to
    grey; value floor (``v_lo``) keeps low-PC3 cells from washing to
    black.  All channels rank-normalised so the hue wheel is uniformly
    swept across the embedding's PC1 distribution.
    """
    h = rank_unit(pc1, 0.0, 1.0)
    s = rank_unit(pc2, s_lo, s_hi)
    v = rank_unit(pc3, v_lo, v_hi)
    hsv = np.stack([h, s, v], axis=1)
    return hsv_to_rgb(hsv).astype(np.float32)


def main() -> None:
    paths = StudyAreaPaths(STUDY_AREA)
    src = (
        paths.root
        / "stage2_multimodal"
        / "ring_agg"
        / "embeddings"
        / "netherlands_res9_20mix.parquet"
    )
    out_dir = paths.root / "stage3_analysis" / "2026-04-24"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid_viz_ring_agg_res9.png"

    print(f"[load] {src}")
    t0 = time.time()
    df = pd.read_parquet(src)
    hex_ids = df.index.to_numpy()
    X = df.to_numpy(dtype=np.float32)
    print(f"  -> {X.shape} in {time.time()-t0:.1f}s")

    print(f"[centroids] SpatialDB res={RES} crs=28992")
    t0 = time.time()
    db = SpatialDB.for_study_area(STUDY_AREA)
    cx_m, cy_m = db.centroids(list(hex_ids), resolution=RES, crs=28992)
    cx_m = np.asarray(cx_m, dtype=np.float64)
    cy_m = np.asarray(cy_m, dtype=np.float64)
    print(f"  -> centroids in {time.time()-t0:.1f}s "
          f"(x {cx_m.min():.0f}..{cx_m.max():.0f}, "
          f"y {cy_m.min():.0f}..{cy_m.max():.0f})")

    print("[pca] 3 components")
    t0 = time.time()
    pca = PCA(n_components=3, random_state=0)
    pcs = pca.fit_transform(X)
    print(f"  -> explained var: {pca.explained_variance_ratio_.round(3).tolist()}"
          f" (sum={pca.explained_variance_ratio_.sum():.3f}) in {time.time()-t0:.1f}s")
    rgb = rank_rgb(pcs, gamma=GAMMA)

    print(f"[kmeans] k={K_CLUSTERS} (MiniBatch)")
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=K_CLUSTERS,
        random_state=0,
        batch_size=4096,
        n_init=4,
    )
    labels = km.fit_predict(X)
    print(f"  -> labels in {time.time()-t0:.1f}s")
    # tab10: 10 perceptually distinct colours, not tab20 sampled at 10 steps.
    cmap_cat = colormaps["tab10"]
    cluster_rgb = cmap_cat(labels % 10)[:, :3].astype(np.float32)

    # Extent in metres with ~2 km padding around hex centroids.
    pad_m = 2_000.0
    extent_m = (
        float(cx_m.min() - pad_m),
        float(cy_m.min() - pad_m),
        float(cx_m.max() + pad_m),
        float(cy_m.max() + pad_m),
    )

    print("[raster] KDTree-Voronoi cluster + PC-RGB")
    t0 = time.time()
    cluster_img, extent_xy = rasterize_voronoi(cx_m, cy_m, cluster_rgb, extent_m)
    pcrgb_img, _ = rasterize_voronoi(cx_m, cy_m, rgb, extent_m)
    print(f"  -> rasters {cluster_img.shape[1]}x{cluster_img.shape[0]} "
          f"in {time.time()-t0:.1f}s")

    print(f"[plot] -> {out_path}")
    fig, axes = plt.subplots(1, 2, figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor(BG)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_aspect("equal")

    axes[0].imshow(cluster_img, extent=extent_xy, origin="lower",
                   interpolation="nearest")
    axes[0].set_title(
        f"Ring-agg k={K_CLUSTERS} clusters  .  res9 NL  .  {len(hex_ids):,} hex",
        color="white", fontsize=16, pad=10,
    )
    axes[1].imshow(pcrgb_img, extent=extent_xy, origin="lower",
                   interpolation="nearest")
    axes[1].set_title(
        f"PC1.PC2.PC3 -> RGB (rank-norm)  .  "
        f"var={pca.explained_variance_ratio_.sum():.1%}",
        color="white", fontsize=16, pad=10,
    )

    fig.suptitle(
        f"Netherlands . Ring-Aggregation Embedding (k={K_CLUSTERS}) . "
        f"res9 . 20mix . KDTree-Voronoi raster",
        color="white", fontsize=18, y=0.98,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    print(f"[done] {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")


def main_gallery() -> None:
    """Render the 2x4 palette gallery: 4 cluster palettes + 4 continuous encodings.

    Re-uses the canonical pipeline (load -> centroids -> PCA -> KMeans) and
    runs the KDTree-Voronoi query ONCE, then gathers 8 RGBA images via
    fancy-indexing on the precomputed nearest-hex index array.
    """
    paths = StudyAreaPaths(STUDY_AREA)
    src = (
        paths.root
        / "stage2_multimodal"
        / "ring_agg"
        / "embeddings"
        / "netherlands_res9_20mix.parquet"
    )
    out_dir = paths.root / "stage3_analysis" / "2026-04-24"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grid_viz_ring_agg_res9_gallery.png"

    print(f"[load] {src}")
    t0 = time.time()
    df = pd.read_parquet(src)
    hex_ids = df.index.to_numpy()
    X = df.to_numpy(dtype=np.float32)
    print(f"  -> {X.shape} in {time.time()-t0:.1f}s")

    print(f"[centroids] SpatialDB res={RES} crs=28992")
    t0 = time.time()
    db = SpatialDB.for_study_area(STUDY_AREA)
    cx_m, cy_m = db.centroids(list(hex_ids), resolution=RES, crs=28992)
    cx_m = np.asarray(cx_m, dtype=np.float64)
    cy_m = np.asarray(cy_m, dtype=np.float64)
    print(f"  -> centroids in {time.time()-t0:.1f}s")

    print("[pca] 3 components")
    t0 = time.time()
    pca = PCA(n_components=3, random_state=0)
    pcs = pca.fit_transform(X)
    var_pct = pca.explained_variance_ratio_.sum() * 100
    print(f"  -> explained var: {pca.explained_variance_ratio_.round(3).tolist()}"
          f" (sum={var_pct:.1f}%) in {time.time()-t0:.1f}s")

    print(f"[kmeans] k={K_CLUSTERS} (MiniBatch)")
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=K_CLUSTERS,
        random_state=0,
        batch_size=4096,
        n_init=4,
    )
    labels = km.fit_predict(X)
    print(f"  -> labels in {time.time()-t0:.1f}s")

    # Extent in metres with ~2 km padding around hex centroids.
    pad_m = 2_000.0
    extent_m = (
        float(cx_m.min() - pad_m),
        float(cy_m.min() - pad_m),
        float(cx_m.max() + pad_m),
        float(cy_m.max() + pad_m),
    )

    print("[voronoi] KDTree query (once for all 8 panels)")
    t0 = time.time()
    nearest_idx, inside, extent_xy = voronoi_indices(cx_m, cy_m, extent_m)
    print(f"  -> {nearest_idx.shape[1]}x{nearest_idx.shape[0]} indices "
          f"in {time.time()-t0:.1f}s")

    # ----- Build 8 per-hex colour arrays --------------------------------
    print("[colourmaps] building 8 per-hex RGB tables")
    t0 = time.time()
    cluster_palettes = [
        ("tab10", "tab10 . canonical balanced"),
        ("Set1",  "Set1 . bold primary"),
        ("Dark2", "Dark2 . muted earthy"),
        ("Paired","Paired . pastel-pop pairs"),
    ]
    cluster_rgbs = []
    for cmap_name, _ in cluster_palettes:
        cmap = colormaps[cmap_name]
        # Sample at evenly-spaced positions so palettes with native size
        # other than K (e.g. Paired has 12) still hand back K distinct hues.
        if cmap.N >= K_CLUSTERS:
            colours = cmap(np.arange(K_CLUSTERS) % cmap.N)
        else:
            colours = cmap(np.linspace(0, 1, K_CLUSTERS))
        cluster_rgbs.append(colours[labels][:, :3].astype(np.float32))

    pc_rgb = rank_rgb(pcs, gamma=GAMMA)
    pc1_unit = rank_unit(pcs[:, 0], 0.0, 1.0)
    turbo_rgb = colormaps["turbo"](pc1_unit)[:, :3].astype(np.float32)
    cividis_rgb = colormaps["cividis"](pc1_unit)[:, :3].astype(np.float32)
    hsv_rgb = hsv_pc_rgb(pcs[:, 0], pcs[:, 1], pcs[:, 2])
    continuous_rgbs = [
        (pc_rgb,      "PC1.PC2.PC3 -> RGB . rank-norm baseline"),
        (turbo_rgb,   "PC1 -> turbo . dominant-axis rainbow"),
        (cividis_rgb, "PC1 -> cividis . perceptually uniform muted"),
        (hsv_rgb,     "HSV(PC1->H, PC2->S, PC3->V) . full chroma sweep"),
    ]
    print(f"  -> 8 per-hex colour tables in {time.time()-t0:.2f}s")

    # ----- Gather 8 RGBA panels ----------------------------------------
    print("[gather] 8x rgb_per_hex[nearest_idx]")
    t0 = time.time()
    panels: list[tuple[np.ndarray, str]] = []
    for rgb_per_hex, (_, descriptor) in zip(cluster_rgbs, cluster_palettes):
        panels.append((gather_rgba(nearest_idx, inside, rgb_per_hex), descriptor))
    for rgb_per_hex, descriptor in continuous_rgbs:
        panels.append((gather_rgba(nearest_idx, inside, rgb_per_hex), descriptor))
    print(f"  -> 8 panels in {time.time()-t0:.2f}s")

    # ----- Plot 2x4 grid -----------------------------------------------
    print(f"[plot] -> {out_path}")
    fig, axes = plt.subplots(2, 4, figsize=(28, 14), dpi=130)
    fig.patch.set_facecolor(BG)

    for ax, (img, descriptor) in zip(axes.flat, panels):
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_aspect("equal")
        ax.imshow(img, extent=extent_xy, origin="lower",
                  interpolation="nearest")
        ax.set_title(descriptor, color="white", fontsize=13, pad=6)

    # Row banners via figure-level text (one suptitle isn't enough).
    fig.text(0.5, 0.955, f"Clusters (k={K_CLUSTERS})",
             ha="center", color="white", fontsize=17, weight="bold")
    fig.text(0.5, 0.485,
             f"Continuous (PCA 3 comp, {var_pct:.1f}% var)",
             ha="center", color="white", fontsize=17, weight="bold")
    fig.suptitle(
        "Netherlands . Ring-Agg Embedding . res9 . palette gallery",
        color="white", fontsize=20, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(hspace=0.18, top=0.92)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    print(f"[done] {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    flags = {arg.lstrip("-") for arg in sys.argv[1:]}
    if "gallery" in flags:
        main_gallery()
    elif "both" in flags:
        main()
        main_gallery()
    else:
        main()
