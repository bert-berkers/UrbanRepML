#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive EDA plots for stage1 unimodal embeddings.

Generates ~7 exploratory plots per modality:
  1. Dimension grid (4x3, first 12 dims) -- spatial rasterized maps
  2. Summary stats (mean + std across dims) -- per-hex aggregation maps
  3. PCA top-3 components (3-panel) -- spatial maps of PC1, PC2, PC3
  4. PCA RGB composite (PC1->R, PC2->G, PC3->B) -- percentile-normalized
  5. MiniBatchKMeans clusters (k=8, 12, 16) -- cluster maps
  6. Correlation heatmap -- seaborn heatmap (non-spatial)
  7. Coverage map -- binary rasterize of embedding presence

Usage:
    # All modalities
    python scripts/plot_embeddings.py --study-area netherlands

    # Single modality
    python scripts/plot_embeddings.py --study-area netherlands --modality poi

    # Specific sub-embedder
    python scripts/plot_embeddings.py --study-area netherlands --modality poi --sub-embedder hex2vec
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from stage3_analysis.visualization.cluster_viz import (
    apply_pca_reduction,
    perform_minibatch_clustering,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPI = 300
RASTER_W = 2000
RASTER_H = 2400
RESOLUTION = 10
YEAR = 2022

# Modality registry: (modality_name, sub_embedder_or_None, display_name)
MODALITY_REGISTRY = [
    ("poi", None, "POI Count"),
    ("poi", "hex2vec", "POI Hex2Vec"),
    ("poi", "hex2vec_27feat", "POI Hex2Vec (27-feat)"),
    ("poi", "geovex", "POI GeoVeX"),
    ("roads", None, "Roads Highway2Vec"),
    ("alphaearth", None, "AlphaEarth"),
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boundary loading (reused from plot_targets.py pattern)
# ---------------------------------------------------------------------------


def load_boundary(paths: StudyAreaPaths) -> gpd.GeoDataFrame:
    """Load study area boundary, filter to European NL, reproject to 28992."""
    from shapely import get_geometry, get_num_geometries

    boundary_path = paths.area_gdf_file()
    if not boundary_path.exists():
        logger.warning("Boundary file not found: %s", boundary_path)
        return None

    boundary_gdf = gpd.read_file(boundary_path)
    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    boundary_gdf = boundary_gdf.to_crs(epsg=28992)

    geom = boundary_gdf.geometry.iloc[0]
    n_parts = get_num_geometries(geom)
    if n_parts > 1:
        euro_geom = max(
            (get_geometry(geom, i) for i in range(n_parts)),
            key=lambda g: g.area,
        )
        boundary_gdf = gpd.GeoDataFrame(
            geometry=[euro_geom], crs=boundary_gdf.crs
        )

    return boundary_gdf


# ---------------------------------------------------------------------------
# Rasterization helpers (reused from plot_targets.py)
# ---------------------------------------------------------------------------


def rasterize_continuous(
    cx: np.ndarray,
    cy: np.ndarray,
    values: np.ndarray,
    extent: tuple,
    width: int = RASTER_W,
    height: int = RASTER_H,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Rasterize continuous values to an RGBA image.

    Args:
        cx, cy: centroid coordinates in EPSG:28992.
        values: float array of same length as cx/cy.
        extent: (minx, miny, maxx, maxy) in EPSG:28992.
        width, height: output image dimensions.
        cmap: matplotlib colormap name.
        vmin, vmax: value range for colormap normalization.

    Returns:
        (height, width, 4) RGBA float32 array with white background.
    """
    minx, miny, maxx, maxy = extent
    mask = (
        (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
        & np.isfinite(values)
    )
    cx_m, cy_m, val_m = cx[mask], cy[mask], values[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    if vmin is None:
        vmin = float(np.nanpercentile(val_m, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(val_m, 98))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap_obj = plt.get_cmap(cmap)
    rgb = colormap_obj(norm(val_m))[:, :3].astype(np.float32)

    image = np.ones((height, width, 4), dtype=np.float32)  # white bg
    image[py, px, :3] = rgb
    image[py, px, 3] = 1.0
    return image


def rasterize_rgb(
    cx: np.ndarray,
    cy: np.ndarray,
    rgb_array: np.ndarray,
    extent: tuple,
    width: int = RASTER_W,
    height: int = RASTER_H,
) -> np.ndarray:
    """Rasterize pre-computed RGB values to an RGBA image.

    Extracted from LinearProbeVisualizer._rasterize_centroids pattern.

    Args:
        cx, cy: centroid coordinates in EPSG:28992.
        rgb_array: (N, 3) float array with R, G, B in [0, 1].
        extent: (minx, miny, maxx, maxy) in EPSG:28992.
        width, height: output image dimensions.

    Returns:
        (height, width, 4) RGBA float32 array with white background.
    """
    minx, miny, maxx, maxy = extent
    mask = (
        (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    )

    cx_m = cx[mask]
    cy_m = cy[mask]
    rgb_masked = rgb_array[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    image = np.ones((height, width, 4), dtype=np.float32)
    image[py, px, :3] = rgb_masked
    image[py, px, 3] = 1.0
    return image


def rasterize_binary(
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    width: int = RASTER_W,
    height: int = RASTER_H,
    color: tuple = (0.2, 0.5, 0.8),
) -> np.ndarray:
    """Rasterize binary presence to an RGBA image.

    Args:
        cx, cy: centroid coordinates in EPSG:28992.
        extent: (minx, miny, maxx, maxy) in EPSG:28992.
        width, height: output image dimensions.
        color: RGB tuple for presence pixels.

    Returns:
        (height, width, 4) RGBA float32 array with white background.
    """
    minx, miny, maxx, maxy = extent
    mask = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    cx_m, cy_m = cx[mask], cy[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    image = np.ones((height, width, 4), dtype=np.float32)
    image[py, px, 0] = color[0]
    image[py, px, 1] = color[1]
    image[py, px, 2] = color[2]
    image[py, px, 3] = 1.0
    return image


def rasterize_categorical(
    cx: np.ndarray,
    cy: np.ndarray,
    labels: np.ndarray,
    extent: tuple,
    n_clusters: int,
    width: int = RASTER_W,
    height: int = RASTER_H,
    cmap: str = "tab20",
) -> np.ndarray:
    """Rasterize integer cluster labels to an RGBA image.

    Args:
        cx, cy: centroid coordinates in EPSG:28992.
        labels: integer cluster assignment array.
        extent: (minx, miny, maxx, maxy) in EPSG:28992.
        n_clusters: total number of clusters (for colormap scaling).
        width, height: output image dimensions.
        cmap: matplotlib colormap name.

    Returns:
        (height, width, 4) RGBA float32 array with white background.
    """
    minx, miny, maxx, maxy = extent
    mask = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    cx_m, cy_m, lab_m = cx[mask], cy[mask], labels[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    colormap_obj = plt.get_cmap(cmap)
    norm_vals = lab_m.astype(float) / max(n_clusters - 1, 1)
    rgb = colormap_obj(norm_vals)[:, :3].astype(np.float32)

    image = np.ones((height, width, 4), dtype=np.float32)
    image[py, px, :3] = rgb
    image[py, px, 3] = 1.0
    return image


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------


def _clean_map_axes(ax):
    """Remove ticks and labels for a clean map look."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_spatial_map(ax, image, extent, boundary_gdf, title=""):
    """Render a rasterized image on an axes with boundary underlay."""
    if boundary_gdf is not None:
        boundary_gdf.plot(
            ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5,
        )
    minx, miny, maxx, maxy = extent
    ax.imshow(
        image,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        zorder=2,
    )
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    _clean_map_axes(ax)
    if title:
        ax.set_title(title, fontsize=11)


def _add_colorbar(fig, ax, cmap, vmin, vmax, label=""):
    """Add a vertical colorbar to the right of an axes."""
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    return cbar


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def detect_embedding_columns(df: pd.DataFrame) -> list[str]:
    """Detect embedding columns from a DataFrame.

    Tries known prefix patterns (A00, emb_0, etc.), then falls back to
    all numeric non-metadata columns.
    """
    # Pattern 1: A00, A01, ... (AlphaEarth style)
    cols = [c for c in df.columns if len(c) >= 3 and c[0] == "A" and c[1:].isdigit()]
    if cols:
        return sorted(cols, key=lambda c: int(c[1:]))

    # Pattern 2: emb_0, emb_1, ... (SRAI style)
    cols = [c for c in df.columns if c.startswith("emb_")]
    if cols:
        return sorted(cols, key=lambda c: int(c.split("_")[1]))

    # Pattern 3: Single-letter prefix with digits (P00, R00, etc.)
    for prefix in ("P", "R", "S", "G", "D"):
        cols = [c for c in df.columns if c.startswith(prefix) and len(c) >= 2 and c[1:].isdigit()]
        if cols:
            return sorted(cols, key=lambda c: int(c[1:]))

    # Fallback: all numeric columns except known metadata
    exclude = {"pixel_count", "tile_count", "geometry", "region_id", "cluster_id"}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def load_embeddings(paths: StudyAreaPaths, modality: str, sub_embedder: str | None) -> pd.DataFrame | None:
    """Load embedding parquet file. Returns None if file does not exist."""
    emb_path = paths.embedding_file(modality, RESOLUTION, YEAR, sub_embedder=sub_embedder)

    if not emb_path.exists():
        logger.warning("Embedding file not found, skipping: %s", emb_path)
        return None

    logger.info("Loading embeddings: %s", emb_path)
    df = pd.read_parquet(emb_path)

    # Ensure region_id is the index
    if df.index.name != "region_id":
        if "region_id" in df.columns:
            df = df.set_index("region_id")
        df.index.name = "region_id"

    emb_cols = detect_embedding_columns(df)
    if not emb_cols:
        logger.warning("No embedding columns detected in %s", emb_path)
        return None

    logger.info("  %d hexagons, %d embedding dims", len(df), len(emb_cols))
    return df[emb_cols]


def get_plot_dir(paths: StudyAreaPaths, modality: str, sub_embedder: str | None) -> Path:
    """Get the output plot directory next to the embedding data."""
    base = paths.stage1(modality)
    if sub_embedder:
        base = base / sub_embedder
    plot_dir = base / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


# ---------------------------------------------------------------------------
# Plot 1: Dimension grid (first 12 dims, 4x3)
# ---------------------------------------------------------------------------


def plot_dim_grid(
    emb_df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    display_name: str,
):
    """4x3 grid showing the first 12 embedding dimensions as spatial maps."""
    cols = list(emb_df.columns)
    n_show = min(12, len(cols))
    nrows, ncols = 4, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 24), dpi=DPI)
    fig.set_facecolor("white")

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n_show:
            col = cols[i]
            vals = emb_df[col].values.astype(np.float32)
            image = rasterize_continuous(cx, cy, vals, extent, cmap="viridis")
            plot_spatial_map(ax, image, extent, boundary_gdf, title=col)
        else:
            ax.set_visible(False)

    fig.suptitle(
        f"{display_name} -- First {n_show} Embedding Dimensions\n"
        f"H3 res{RESOLUTION} | {len(emb_df):,} hexagons | viridis [p2, p98]",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = out_dir / "dim_grid.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 2: Summary stats (mean + std)
# ---------------------------------------------------------------------------


def plot_summary_stats(
    emb_df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    display_name: str,
):
    """Two-panel map: per-hex mean and std across all embedding dims."""
    vals = emb_df.values.astype(np.float32)
    hex_mean = np.nanmean(vals, axis=1)
    hex_std = np.nanstd(vals, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 12), dpi=DPI)
    fig.set_facecolor("white")

    # Mean map
    img_mean = rasterize_continuous(cx, cy, hex_mean, extent, cmap="viridis")
    plot_spatial_map(axes[0], img_mean, extent, boundary_gdf, title="Mean across dims")
    v2, v98 = np.nanpercentile(hex_mean, [2, 98])
    _add_colorbar(fig, axes[0], "viridis", v2, v98, label="Mean")

    # Std map
    img_std = rasterize_continuous(cx, cy, hex_std, extent, cmap="inferno")
    plot_spatial_map(axes[1], img_std, extent, boundary_gdf, title="Std across dims")
    v2, v98 = np.nanpercentile(hex_std, [2, 98])
    _add_colorbar(fig, axes[1], "inferno", v2, v98, label="Std dev")

    fig.suptitle(
        f"{display_name} -- Summary Statistics (Mean / Std)\n"
        f"H3 res{RESOLUTION} | {len(emb_df):,} hexagons | {emb_df.shape[1]} dims",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "summary_stats.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 3: PCA top-3 components (3-panel)
# ---------------------------------------------------------------------------


def plot_pca_top3(
    emb_df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    display_name: str,
):
    """3-panel spatial map of PC1, PC2, PC3."""
    n_components = min(3, emb_df.shape[1])
    embeddings = emb_df.values.astype(np.float32)
    reduced, pca = apply_pca_reduction(embeddings, n_components=n_components)

    fig, axes = plt.subplots(1, 3, figsize=(24, 10), dpi=DPI)
    fig.set_facecolor("white")

    for i in range(3):
        if i < n_components:
            vals = reduced[:, i]
            image = rasterize_continuous(cx, cy, vals, extent, cmap="RdBu_r")
            var_pct = pca.explained_variance_ratio_[i] * 100
            plot_spatial_map(
                axes[i], image, extent, boundary_gdf,
                title=f"PC{i+1} ({var_pct:.1f}% var)",
            )
            v2, v98 = np.nanpercentile(vals, [2, 98])
            _add_colorbar(fig, axes[i], "RdBu_r", v2, v98, label=f"PC{i+1}")
        else:
            axes[i].set_visible(False)

    total_var = sum(pca.explained_variance_ratio_[:n_components]) * 100
    fig.suptitle(
        f"{display_name} -- PCA Top-3 Components\n"
        f"H3 res{RESOLUTION} | {len(emb_df):,} hexagons | "
        f"Total variance: {total_var:.1f}%",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "pca_top3.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)

    return reduced, pca


# ---------------------------------------------------------------------------
# Plot 4: PCA RGB composite (PC1->R, PC2->G, PC3->B)
# ---------------------------------------------------------------------------


def plot_pca_rgb(
    emb_df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    display_name: str,
    pca_result: tuple | None = None,
):
    """RGB composite map: PC1->R, PC2->G, PC3->B with p2/p98 normalization.

    Uses the percentile normalization approach from
    LinearProbeVisualizer.plot_rgb_top3_map.
    """
    if emb_df.shape[1] < 3:
        logger.warning("PCA RGB requires >= 3 dims, got %d, skipping", emb_df.shape[1])
        return

    if pca_result is not None:
        reduced, pca = pca_result
    else:
        embeddings = emb_df.values.astype(np.float32)
        reduced, pca = apply_pca_reduction(embeddings, n_components=3)

    # Percentile normalization per channel (p2/p98, from linear_probe_viz.py)
    rgb_array = np.zeros((len(emb_df), 3), dtype=np.float64)
    for ch in range(min(3, reduced.shape[1])):
        vals = reduced[:, ch].astype(np.float64)
        p2, p98 = np.percentile(vals, [2, 98])
        if p98 - p2 > 0:
            normalized = np.clip((vals - p2) / (p98 - p2), 0.0, 1.0)
        else:
            normalized = np.full_like(vals, 0.5)
        rgb_array[:, ch] = normalized

    # Rasterize RGB
    image = rasterize_rgb(cx, cy, rgb_array.astype(np.float32), extent)

    fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
    fig.set_facecolor("white")
    plot_spatial_map(ax, image, extent, boundary_gdf)

    # Mini-colorbars for each channel
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.cm as cm

    channel_cmaps = ["Reds", "Greens", "Blues"]
    channel_labels = ["R (PC1)", "G (PC2)", "B (PC3)"]
    for ch in range(min(3, reduced.shape[1])):
        var_pct = pca.explained_variance_ratio_[ch] * 100
        cbar_ax = inset_axes(
            ax, width="2%", height="12%", loc="lower right",
            bbox_to_anchor=(0.0, 0.02 + ch * 0.14, 1.0, 1.0),
            bbox_transform=ax.transAxes, borderpad=1.5,
        )
        sm = cm.ScalarMappable(
            cmap=plt.get_cmap(channel_cmaps[ch]),
            norm=Normalize(vmin=0, vmax=1),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f"{channel_labels[ch]} ({var_pct:.1f}%)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    total_var = sum(pca.explained_variance_ratio_[:3]) * 100
    ax.set_title(
        f"{display_name} -- PCA RGB Composite\n"
        f"PC1->R, PC2->G, PC3->B | p2/p98 normalized\n"
        f"H3 res{RESOLUTION} | {len(emb_df):,} hexagons | "
        f"Total variance: {total_var:.1f}%",
        fontsize=12, fontweight="bold", pad=10,
    )

    # Interpretation box
    ax.text(
        0.02, 0.98,
        "Similar colors = similar embedding representation\n"
        "in the 3 principal components of embedding space",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.85),
    )

    plt.tight_layout()
    path = out_dir / "pca_rgb.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 5: MiniBatchKMeans clusters (k=8, 12, 16)
# ---------------------------------------------------------------------------


def plot_clusters(
    emb_df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    display_name: str,
):
    """Cluster maps for k=8, 12, 16 using MiniBatchKMeans."""
    k_list = [8, 12, 16]
    embeddings = emb_df.values.astype(np.float32)

    # PCA reduce if > 32 dims for clustering efficiency
    if embeddings.shape[1] > 32:
        pca_dim = min(16, embeddings.shape[1])
        reduced, _ = apply_pca_reduction(embeddings, n_components=pca_dim)
    else:
        reduced = embeddings

    cluster_results = perform_minibatch_clustering(
        reduced, k_list, standardize=True,
    )

    for k, labels in cluster_results.items():
        fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
        fig.set_facecolor("white")

        image = rasterize_categorical(cx, cy, labels, extent, n_clusters=k, cmap="tab20")
        plot_spatial_map(ax, image, extent, boundary_gdf)

        ax.set_title(
            f"{display_name} -- MiniBatchKMeans k={k}\n"
            f"H3 res{RESOLUTION} | {len(emb_df):,} hexagons",
            fontsize=12, fontweight="bold", pad=10,
        )

        # Cluster legend
        cmap_obj = plt.get_cmap("tab20")
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cmap_obj(i / max(k - 1, 1)), label=f"Cluster {i}")
            for i in range(k)
        ]
        ncol = 4 if k > 8 else 3
        ax.legend(
            handles=legend_elements, loc="lower left", ncol=ncol,
            fontsize=7, framealpha=0.9, title="Clusters",
        )

        plt.tight_layout()
        path = out_dir / f"clusters_k{k}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 6: Correlation heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    emb_df: pd.DataFrame,
    out_dir: Path,
    display_name: str,
):
    """Seaborn heatmap of pairwise correlation between embedding dimensions."""
    n_dims = emb_df.shape[1]

    # For very high-dimensional embeddings, show a subset
    if n_dims > 64:
        # Show first 32 + last 32
        cols = list(emb_df.columns)
        selected = cols[:32] + cols[-32:]
        selected = list(dict.fromkeys(selected))  # deduplicate, preserve order
        subset = emb_df[selected]
        subset_label = f"first 32 + last 32 of {n_dims}"
    else:
        subset = emb_df
        subset_label = f"{n_dims} dims"

    corr = subset.corr()

    # Figure size scales with dimension count
    dim = len(corr)
    fig_size = max(8, dim * 0.25)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=DPI)
    fig.set_facecolor("white")

    sns.heatmap(
        corr, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True,
        xticklabels=True, yticklabels=True,
        linewidths=0,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )

    ax.tick_params(axis="both", labelsize=max(4, 8 - dim // 20))
    ax.set_title(
        f"{display_name} -- Embedding Correlation Matrix\n"
        f"({subset_label})",
        fontsize=12, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    path = out_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 7: Coverage map
# ---------------------------------------------------------------------------


def plot_coverage(
    emb_df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    display_name: str,
):
    """Binary map showing which hexagons have embedding data."""
    fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
    fig.set_facecolor("white")

    image = rasterize_binary(cx, cy, extent, color=(0.2, 0.5, 0.8))
    plot_spatial_map(ax, image, extent, boundary_gdf)

    ax.set_title(
        f"{display_name} -- Coverage Map\n"
        f"H3 res{RESOLUTION} | {len(emb_df):,} hexagons with embeddings",
        fontsize=12, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    path = out_dir / "coverage.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Main pipeline for a single modality
# ---------------------------------------------------------------------------


def process_modality(
    paths: StudyAreaPaths,
    modality: str,
    sub_embedder: str | None,
    display_name: str,
    boundary_gdf: gpd.GeoDataFrame | None,
):
    """Run all 7 plots for a single modality."""
    logger.info("=" * 60)
    logger.info("Processing: %s", display_name)
    logger.info("=" * 60)

    # Load embeddings
    emb_df = load_embeddings(paths, modality, sub_embedder)
    if emb_df is None:
        return

    # Output directory
    out_dir = get_plot_dir(paths, modality, sub_embedder)
    logger.info("  Output dir: %s", out_dir)

    # Get centroids via SpatialDB (single call, cached)
    db = SpatialDB.for_study_area(paths.study_area)
    cx, cy = db.centroids(emb_df.index, resolution=RESOLUTION, crs=28992)
    logger.info("  Centroids loaded: %d points", len(cx))

    # Compute extent from boundary or centroids
    if boundary_gdf is not None:
        ext = boundary_gdf.total_bounds
    else:
        ext = db.extent(emb_df.index, resolution=RESOLUTION, crs=28992)
    minx, miny, maxx, maxy = ext
    pad = (maxx - minx) * 0.03
    extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    # Plot 1: Dimension grid
    logger.info("[1/7] Dimension grid...")
    plot_dim_grid(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name)

    # Plot 2: Summary stats
    logger.info("[2/7] Summary stats...")
    plot_summary_stats(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name)

    # Plot 3 & 4: PCA (share computation)
    logger.info("[3/7] PCA top-3 components...")
    pca_result = plot_pca_top3(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name)

    logger.info("[4/7] PCA RGB composite...")
    plot_pca_rgb(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name, pca_result=pca_result)

    # Plot 5: Clusters
    logger.info("[5/7] MiniBatchKMeans clusters...")
    plot_clusters(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name)

    # Plot 6: Correlation heatmap
    logger.info("[6/7] Correlation heatmap...")
    plot_correlation_heatmap(emb_df, out_dir, display_name)

    # Plot 7: Coverage map
    logger.info("[7/7] Coverage map...")
    plot_coverage(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name)

    logger.info("Completed %s: 7+ plots saved to %s", display_name, out_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def filter_modalities(
    modality: str | None,
    sub_embedder: str | None,
) -> list[tuple[str, str | None, str]]:
    """Filter MODALITY_REGISTRY based on CLI args."""
    result = []
    for mod, sub, name in MODALITY_REGISTRY:
        if modality and mod != modality:
            continue
        if sub_embedder is not None:
            # If sub_embedder is explicitly given, match exactly
            if sub != sub_embedder:
                continue
        elif modality and sub is not None:
            # If only --modality given (no --sub-embedder), include
            # the base modality AND its sub-embedders
            pass
        result.append((mod, sub, name))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate EDA plots for stage1 unimodal embeddings",
    )
    parser.add_argument(
        "--study-area", default="netherlands",
        help="Study area name (default: netherlands)",
    )
    parser.add_argument(
        "--modality", default=None,
        help="Single modality to process (e.g. poi, roads). Default: all.",
    )
    parser.add_argument(
        "--sub-embedder", default=None,
        help="Sub-embedder name (e.g. hex2vec, geovex). Default: all for the modality.",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = StudyAreaPaths(args.study_area)

    # Filter modalities
    modalities = filter_modalities(args.modality, args.sub_embedder)
    if not modalities:
        logger.error(
            "No modalities matched --modality=%s --sub-embedder=%s",
            args.modality, args.sub_embedder,
        )
        sys.exit(1)

    logger.info(
        "Study area: %s | Modalities to process: %d",
        args.study_area, len(modalities),
    )
    for mod, sub, name in modalities:
        label = f"{mod}/{sub}" if sub else mod
        logger.info("  - %s (%s)", label, name)

    # Load boundary once (shared across all modalities)
    boundary_gdf = load_boundary(paths)

    # Process each modality
    for mod, sub, name in modalities:
        process_modality(paths, mod, sub, name, boundary_gdf)

    logger.info("All done.")


if __name__ == "__main__":
    main()
