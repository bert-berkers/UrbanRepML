#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive EDA plots for stage1 unimodal embeddings.

Generates ~8 exploratory plots per modality:
  1. Dimension grid (4x3, first 12 dims) -- spatial rasterized maps
  2. Summary stats (mean + std across dims) -- per-hex aggregation maps
  3. PCA top-3 components (3-panel) -- spatial maps of PC1, PC2, PC3
  4. PCA RGB composite (PC1->R, PC2->G, PC3->B) -- percentile-normalized
  5. MiniBatchKMeans clusters (k=8, 12, 16) -- cluster maps
  6. Correlation heatmap -- seaborn heatmap (non-spatial)
  7. Coverage map -- binary rasterize of embedding presence
  8. Leefbaarometer target distributions -- KDE+histogram per score (target only)

Lifetime: durable
Stage: 1 (data) + 3 (analysis/visualization)

Usage:
    # All modalities at res10 (default)
    python scripts/plot_embeddings.py --study-area netherlands

    # All modalities at res9
    python scripts/plot_embeddings.py --study-area netherlands --resolution 9

    # Single modality
    python scripts/plot_embeddings.py --study-area netherlands --modality poi

    # Specific sub-embedder
    python scripts/plot_embeddings.py --study-area netherlands --modality poi --sub-embedder hex2vec

    # Leefbaarometer target only
    python scripts/plot_embeddings.py --study-area netherlands --modality leefbaarometer --resolution 9

    # Specify year override
    python scripts/plot_embeddings.py --study-area netherlands --year 2022
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
from utils.visualization import (
    RASTER_H,
    RASTER_W,
    _add_colorbar,
    _add_rd_grid,
    _clean_map_axes,
    detect_embedding_columns,
    filter_empty_hexagons,
    load_boundary,
    rasterize_binary_voronoi,
    rasterize_categorical_voronoi,
    rasterize_continuous_voronoi,
    rasterize_rgb_voronoi,
    voronoi_params_for_resolution,
    plot_spatial_map,
)
from stage3_analysis.visualization.clustering_utils import (
    apply_pca_reduction,
    perform_minibatch_clustering,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPI = 300

# Modality registry: (modality_name, sub_embedder_or_None, display_name, year)
# year: "2022" for AlphaEarth, "latest" for Overpass-sourced modalities
MODALITY_REGISTRY = [
    ("poi", None, "POI Count", "latest"),
    ("poi", "hex2vec", "POI Hex2Vec", "latest"),
    ("poi", "hex2vec_27feat", "POI Hex2Vec (27-feat)", "2022"),
    ("poi", "geovex", "POI GeoVeX", "latest"),
    ("roads", None, "Roads Highway2Vec", "latest"),
    ("gtfs", None, "GTFS Transit", "latest"),
    ("alphaearth", None, "AlphaEarth", "2022"),
]

# Leefbaarometer score display names
TARGET_SCORE_NAMES = {
    "lbm": "Overall Liveability",
    "fys": "Physical Environment",
    "onv": "Safety",
    "soc": "Social Cohesion",
    "vrz": "Amenities",
    "won": "Housing",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_embeddings(
    paths: StudyAreaPaths,
    modality: str,
    resolution: int,
    year: str,
    sub_embedder: str | None,
) -> pd.DataFrame | None:
    """Load embedding parquet file. Returns None if file does not exist."""
    emb_path = paths.embedding_file(modality, resolution, year, sub_embedder=sub_embedder)

    if not emb_path.exists():
        logger.warning("Embedding file not found, skipping: %s", emb_path)
        return None

    logger.info("Loading embeddings: %s", emb_path)
    df = pd.read_parquet(emb_path)

    # Handle AlphaEarth h3_index inconsistency: it uses h3_index instead of region_id
    if df.index.name != "region_id":
        if "h3_index" in df.columns:
            df = df.set_index("h3_index")
            df.index.name = "region_id"
        elif "region_id" in df.columns:
            df = df.set_index("region_id")
        else:
            df.index.name = "region_id"

    emb_cols = detect_embedding_columns(df)
    if not emb_cols:
        logger.warning("No embedding columns detected in %s", emb_path)
        return None

    logger.info("  %d hexagons, %d embedding dims", len(df), len(emb_cols))
    return df[emb_cols]


def get_plot_dir(
    paths: StudyAreaPaths,
    modality: str,
    sub_embedder: str | None,
    resolution: int,
) -> Path:
    """Get the output plot directory next to the embedding data.

    Plots go to ``{modality}/plots/res{resolution}/`` to keep different
    resolutions separated.
    """
    base = paths.stage1(modality)
    if sub_embedder:
        base = base / sub_embedder
    plot_dir = base / "plots" / f"res{resolution}"
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
    resolution: int,
):
    """4x3 grid showing the first 12 embedding dimensions as spatial maps."""
    cols = list(emb_df.columns)
    n_show = min(12, len(cols))
    nrows, ncols = 4, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 24), dpi=DPI)
    fig.set_facecolor("white")

    pixel_m, max_dist_m = voronoi_params_for_resolution(resolution)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n_show:
            col = cols[i]
            vals = emb_df[col].values.astype(np.float32)
            image, _ = rasterize_continuous_voronoi(
                cx, cy, vals, extent, cmap="viridis",
                pixel_m=pixel_m, max_dist_m=max_dist_m,
            )
            plot_spatial_map(ax, image, extent, boundary_gdf, title=col)
        else:
            ax.set_visible(False)

    fig.suptitle(
        f"{display_name} -- First {n_show} Embedding Dimensions\n"
        f"H3 res{resolution} | {len(emb_df):,} hexagons | viridis [p2, p98]",
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
    resolution: int,
):
    """Two-panel map: per-hex mean and std across all embedding dims."""
    vals = emb_df.values.astype(np.float32)
    hex_mean = np.nanmean(vals, axis=1)
    hex_std = np.nanstd(vals, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 12), dpi=DPI)
    fig.set_facecolor("white")

    pixel_m, max_dist_m = voronoi_params_for_resolution(resolution)
    # Mean map
    img_mean, _ = rasterize_continuous_voronoi(
        cx, cy, hex_mean, extent, cmap="viridis",
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )
    plot_spatial_map(axes[0], img_mean, extent, boundary_gdf, title="Mean across dims")
    v2, v98 = np.nanpercentile(hex_mean, [2, 98])
    _add_colorbar(fig, axes[0], "viridis", v2, v98, label="Mean")

    # Std map
    img_std, _ = rasterize_continuous_voronoi(
        cx, cy, hex_std, extent, cmap="inferno",
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )
    plot_spatial_map(axes[1], img_std, extent, boundary_gdf, title="Std across dims")
    v2, v98 = np.nanpercentile(hex_std, [2, 98])
    _add_colorbar(fig, axes[1], "inferno", v2, v98, label="Std dev")

    fig.suptitle(
        f"{display_name} -- Summary Statistics (Mean / Std)\n"
        f"H3 res{resolution} | {len(emb_df):,} hexagons | {emb_df.shape[1]} dims",
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
    resolution: int,
):
    """3-panel spatial map of PC1, PC2, PC3."""
    n_components = min(3, emb_df.shape[1])
    embeddings = emb_df.values.astype(np.float32)
    reduced, pca = apply_pca_reduction(embeddings, n_components=n_components)

    fig, axes = plt.subplots(1, 3, figsize=(24, 10), dpi=DPI)
    fig.set_facecolor("white")

    pixel_m, max_dist_m = voronoi_params_for_resolution(resolution)
    for i in range(3):
        if i < n_components:
            vals = reduced[:, i]
            image, _ = rasterize_continuous_voronoi(
                cx, cy, vals, extent, cmap="RdBu_r",
                pixel_m=pixel_m, max_dist_m=max_dist_m,
            )
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
        f"H3 res{resolution} | {len(emb_df):,} hexagons | "
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
    resolution: int,
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
    pixel_m, max_dist_m = voronoi_params_for_resolution(resolution)
    image, _ = rasterize_rgb_voronoi(
        cx, cy, rgb_array.astype(np.float32), extent,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )

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
        f"H3 res{resolution} | {len(emb_df):,} hexagons | "
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
    resolution: int,
):
    """Cluster maps for k=8, 12, 16 using rasterized centroid rendering."""
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

    pixel_m, max_dist_m = voronoi_params_for_resolution(resolution)
    for k, labels in cluster_results.items():
        fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
        fig.set_facecolor("white")

        image, _ = rasterize_categorical_voronoi(
            cx, cy, labels, extent, n_clusters=k, cmap="tab20",
            pixel_m=pixel_m, max_dist_m=max_dist_m,
        )
        plot_spatial_map(ax, image, extent, boundary_gdf)

        ax.set_title(
            f"{display_name} -- MiniBatchKMeans k={k}\n"
            f"H3 res{resolution} | {len(emb_df):,} hexagons",
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
    resolution: int,
):
    """Binary map showing which hexagons have embedding data."""
    fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
    fig.set_facecolor("white")

    pixel_m, max_dist_m = voronoi_params_for_resolution(resolution)
    image, _ = rasterize_binary_voronoi(
        cx, cy, extent, color=(0.2, 0.5, 0.8),
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )
    plot_spatial_map(ax, image, extent, boundary_gdf)

    ax.set_title(
        f"{display_name} -- Coverage Map\n"
        f"H3 res{resolution} | {len(emb_df):,} hexagons with embeddings",
        fontsize=12, fontweight="bold", pad=10,
    )

    plt.tight_layout()
    path = out_dir / "coverage.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 8: Leefbaarometer target distributions
# ---------------------------------------------------------------------------


def plot_leefbaarometer_distributions(
    paths: StudyAreaPaths,
    resolution: int,
    year: int = 2022,
):
    """Generate 2x3 grid of KDE + histogram for leefbaarometer scores.

    Output goes to the target data directory (not embedding directory)
    because this is TARGET data. Uses a warm color palette to visually
    distinguish from embedding plots.
    """
    target_path = paths.target_file("leefbaarometer", resolution, year)

    if not target_path.exists():
        logger.warning("Leefbaarometer target not found, skipping: %s", target_path)
        return

    logger.info("Loading leefbaarometer target: %s", target_path)
    df = pd.read_parquet(target_path)
    if df.index.name != "region_id" and "region_id" in df.columns:
        df = df.set_index("region_id")

    score_cols = [c for c in TARGET_SCORE_NAMES.keys() if c in df.columns]
    if not score_cols:
        logger.warning("No score columns found in leefbaarometer target. Available: %s", list(df.columns))
        return

    n_hexagons = len(df)

    # 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Distinct warm palette for target data (vs cool blues/greens for embeddings)
    target_colors = ["#d62728", "#ff7f0e", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]

    from scipy.stats import gaussian_kde

    for i, col in enumerate(score_cols):
        ax = axes[i]
        vals = df[col].dropna().values
        n_valid = len(vals)

        ax.hist(vals, bins=80, density=True, alpha=0.5, color=target_colors[i], edgecolor="none")

        # KDE overlay
        try:
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 300)
            ax.plot(x_range, kde(x_range), color=target_colors[i], linewidth=2)
        except Exception:
            pass

        mean_val = np.mean(vals)
        median_val = np.median(vals)
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axvline(median_val, color="black", linestyle=":", linewidth=1.2, alpha=0.8)

        full_name = TARGET_SCORE_NAMES.get(col, col)
        ax.set_title(f"{full_name} ({col})", fontsize=11, fontweight="bold")
        ax.annotate(
            f"N={n_valid:,}\nmean={mean_val:.3f}\nmedian={median_val:.3f}\nstd={np.std(vals):.3f}",
            xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.set_ylabel("Density" if i % 3 == 0 else "")

    # Hide unused axes
    for i in range(len(score_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"Leefbaarometer Target Distributions (Res {resolution}, {year})\n"
        f"{n_hexagons:,} hexagons | -- mean  : median",
        fontsize=14, fontweight="bold", y=1.02,
    )

    output_dir = paths.target("leefbaarometer") / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"res{resolution}_distributions.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline for a single modality
# ---------------------------------------------------------------------------


def process_modality(
    paths: StudyAreaPaths,
    modality: str,
    sub_embedder: str | None,
    display_name: str,
    year: str,
    resolution: int,
    boundary_gdf: gpd.GeoDataFrame | None,
    filter_threshold: float = 0.10,
):
    """Run all 7 plots for a single modality."""
    logger.info("=" * 60)
    logger.info("Processing: %s", display_name)
    logger.info("=" * 60)

    # Load embeddings
    emb_df_full = load_embeddings(paths, modality, resolution, year, sub_embedder)
    if emb_df_full is None:
        return

    # Output directory (resolution-specific subdirectory)
    out_dir = get_plot_dir(paths, modality, sub_embedder, resolution)
    logger.info("  Output dir: %s", out_dir)

    # Get centroids for FULL data (used by coverage plot)
    db = SpatialDB.for_study_area(paths.study_area)
    cx_full, cy_full = db.centroids(emb_df_full.index, resolution=resolution, crs=28992)
    logger.info("  Centroids loaded: %d points", len(cx_full))

    # Filter empty/water hexagons for analysis plots
    emb_df = filter_empty_hexagons(emb_df_full, display_name, constant_threshold=filter_threshold)
    if len(emb_df) < len(emb_df_full):
        cx, cy = db.centroids(emb_df.index, resolution=resolution, crs=28992)
    else:
        cx, cy = cx_full, cy_full

    # Compute extent from boundary or centroids
    if boundary_gdf is not None:
        ext = boundary_gdf.total_bounds
    else:
        ext = db.extent(emb_df.index, resolution=resolution, crs=28992)
    minx, miny, maxx, maxy = ext
    pad = (maxx - minx) * 0.03
    extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    # Voronoi pixel_m / max_dist_m derived per-resolution inside each plot fn.

    # Plot 1: Dimension grid
    logger.info("[1/7] Dimension grid...")
    plot_dim_grid(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name, resolution)

    # Plot 2: Summary stats
    logger.info("[2/7] Summary stats...")
    plot_summary_stats(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name, resolution)

    # Plot 3 & 4: PCA (share computation)
    logger.info("[3/7] PCA top-3 components...")
    pca_result = plot_pca_top3(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name, resolution)

    logger.info("[4/7] PCA RGB composite...")
    plot_pca_rgb(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name, resolution, pca_result=pca_result)

    # Plot 5: Clusters
    logger.info("[5/7] MiniBatchKMeans clusters...")
    plot_clusters(emb_df, cx, cy, extent, boundary_gdf, out_dir, display_name, resolution)

    # Plot 6: Correlation heatmap
    logger.info("[6/7] Correlation heatmap...")
    plot_correlation_heatmap(emb_df, out_dir, display_name)

    # Plot 7: Coverage map (uses FULL unfiltered data)
    logger.info("[7/7] Coverage map...")
    plot_coverage(emb_df_full, cx_full, cy_full, extent, boundary_gdf, out_dir, display_name, resolution)

    logger.info("Completed %s: 7 plots saved to %s", display_name, out_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def filter_modalities(
    modality: str | None,
    sub_embedder: str | None,
) -> list[tuple[str, str | None, str, str]]:
    """Filter MODALITY_REGISTRY based on CLI args.

    Returns list of (modality, sub_embedder, display_name, year) tuples.
    """
    result = []
    for mod, sub, name, year in MODALITY_REGISTRY:
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
        result.append((mod, sub, name, year))
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
        "--resolution", type=int, default=10,
        help="H3 resolution (default: 10)",
    )
    parser.add_argument(
        "--year", default=None,
        help="Data year override (e.g. 2022, latest). Default: auto per modality.",
    )
    parser.add_argument(
        "--modality", default=None,
        help="Single modality to process (e.g. poi, roads, leefbaarometer). Default: all.",
    )
    parser.add_argument(
        "--sub-embedder", default=None,
        help="Sub-embedder name (e.g. hex2vec, geovex). Default: all for the modality.",
    )
    parser.add_argument(
        "--filter-threshold", type=float, default=0.10,
        help="Minimum fraction of low-variance rows to trigger filtering (default: 0.10).",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = StudyAreaPaths(args.study_area)
    resolution = args.resolution

    # Handle leefbaarometer-only mode
    if args.modality == "leefbaarometer":
        lbm_year = int(args.year) if args.year and args.year.isdigit() else 2022
        logger.info("Generating leefbaarometer target distributions (res%d, year=%d)", resolution, lbm_year)
        plot_leefbaarometer_distributions(paths, resolution, lbm_year)
        logger.info("All done.")
        return

    # Filter modalities
    modalities = filter_modalities(args.modality, args.sub_embedder)
    if not modalities:
        logger.error(
            "No modalities matched --modality=%s --sub-embedder=%s",
            args.modality, args.sub_embedder,
        )
        sys.exit(1)

    logger.info(
        "Study area: %s | Resolution: %d | Modalities to process: %d",
        args.study_area, resolution, len(modalities),
    )
    for mod, sub, name, year in modalities:
        effective_year = args.year if args.year else year
        label = f"{mod}/{sub}" if sub else mod
        logger.info("  - %s (%s, year=%s)", label, name, effective_year)

    # Load boundary once (shared across all modalities)
    boundary_gdf = load_boundary(paths)

    # Process each modality
    for mod, sub, name, year in modalities:
        effective_year = args.year if args.year else year
        process_modality(paths, mod, sub, name, effective_year, resolution, boundary_gdf,
                         filter_threshold=args.filter_threshold)

    # Plot 8: Leefbaarometer distributions (when running all modalities)
    if args.modality is None:
        lbm_year = int(args.year) if args.year and args.year.isdigit() else 2022
        logger.info("[8] Leefbaarometer target distributions...")
        plot_leefbaarometer_distributions(paths, resolution, lbm_year)

    logger.info("All done.")


if __name__ == "__main__":
    main()
