#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GTFS embedding interpretability maps: PCA, coverage, and UMAP visualizations.

Generates spatial maps from GTFS2Vec embeddings to reveal transit accessibility
patterns across the Netherlands. Handles the sparse-coverage nature of GTFS data
(~3% of hexagons have actual transit service) with dedicated coverage analysis.

Plots produced:
  1. PCA variance explained bar chart
  2. Top-3 PCA components as spatial maps
  3. Coverage map (transit vs no-transit hexagons)
  4. UMAP 2D scatter colored by KMeans cluster

Reusable for other modalities by changing --modality and --embedding-file args.

Lifetime: durable
Stage: 1 (data) + 3 (analysis/visualization)

Usage:
    # Default: GTFS res9
    python scripts/stage3/plot_gtfs_embeddings.py --study-area netherlands

    # Custom embedding file
    python scripts/stage3/plot_gtfs_embeddings.py --study-area netherlands \\
        --embedding-file gtfs_embeddings_res9.parquet

    # Different modality (reuse)
    python scripts/stage3/plot_gtfs_embeddings.py --study-area netherlands \\
        --modality poi --resolution 9
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import (
    RASTER_H,
    RASTER_W,
    _add_colorbar,
    detect_embedding_columns,
    load_boundary,
    plot_spatial_map,
    rasterize_binary,
    rasterize_continuous,
)

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPI = 300


def load_gtfs_embeddings(
    paths: StudyAreaPaths,
    resolution: int,
    embedding_file: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load GTFS embeddings and return (DataFrame indexed by region_id, embedding_cols).

    If embedding_file is given, looks for that filename in the GTFS stage1 dir.
    Otherwise tries the standard naming convention.
    """
    gtfs_dir = paths.stage1("gtfs")

    if embedding_file:
        emb_path = gtfs_dir / embedding_file
    else:
        # Try standard naming first, then gtfs_embeddings pattern
        emb_path = paths.embedding_file("gtfs", resolution, "latest")
        if not emb_path.exists():
            emb_path = gtfs_dir / f"gtfs_embeddings_res{resolution}.parquet"

    if not emb_path.exists():
        raise FileNotFoundError(f"GTFS embedding file not found: {emb_path}")

    logger.info("Loading GTFS embeddings: %s", emb_path)
    df = pd.read_parquet(emb_path)

    # Ensure region_id is the index
    if df.index.name != "region_id":
        if "region_id" in df.columns:
            df = df.set_index("region_id")
        else:
            df.index.name = "region_id"

    emb_cols = detect_embedding_columns(df)
    if not emb_cols:
        raise ValueError(f"No embedding columns detected in {emb_path}")

    logger.info("  %d hexagons, %d embedding dims", len(df), len(emb_cols))
    return df, emb_cols


def identify_transit_hexagons(
    df: pd.DataFrame,
    emb_cols: list[str],
) -> np.ndarray:
    """Return boolean mask: True for hexagons with actual transit data.

    GTFS2Vec assigns a shared default embedding to hexagons without transit
    service. We detect these by finding the most common embedding vector.
    """
    emb = df[emb_cols].values.astype(np.float32)

    # The default row is the most frequent -- check against first row
    first_row = emb[0]
    is_default = np.all(emb == first_row, axis=1)

    # If > 50% match first row, treat it as the default embedding
    if is_default.sum() > len(emb) * 0.5:
        has_transit = ~is_default
    else:
        # No dominant default -- all rows are "real"
        has_transit = np.ones(len(emb), dtype=bool)

    n_transit = has_transit.sum()
    n_total = len(emb)
    logger.info(
        "  Transit coverage: %d / %d hexagons (%.1f%%)",
        n_transit, n_total, 100.0 * n_transit / n_total,
    )
    return has_transit


# ---------------------------------------------------------------------------
# Plot 1: PCA variance explained bar chart
# ---------------------------------------------------------------------------


def plot_pca_variance(
    emb_df: pd.DataFrame,
    emb_cols: list[str],
    out_dir: Path,
    display_name: str,
    resolution: int,
    transit_only: bool = True,
    transit_mask: np.ndarray | None = None,
):
    """Bar chart showing variance explained by each PCA component."""
    if transit_only and transit_mask is not None:
        data = emb_df.loc[transit_mask, emb_cols].values.astype(np.float32)
        label_suffix = " (transit hexagons only)"
    else:
        data = emb_df[emb_cols].values.astype(np.float32)
        label_suffix = ""

    n_components = min(20, data.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(data)

    var_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)

    # Find 90% threshold
    n_90 = int(np.searchsorted(cumulative, 0.90)) + 1

    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    fig.set_facecolor("white")

    x = np.arange(1, n_components + 1)
    bars = ax.bar(x, var_ratio * 100, color="#4C72B0", alpha=0.8, label="Individual")
    ax.plot(x, cumulative * 100, "o-", color="#C44E52", linewidth=2, markersize=5, label="Cumulative")

    # Mark 90% line
    ax.axhline(90, color="#55A868", linestyle="--", linewidth=1.5, alpha=0.7, label="90% threshold")
    ax.axvline(n_90, color="#55A868", linestyle=":", linewidth=1.5, alpha=0.7)

    # Highlight the 90% component
    if n_90 <= n_components:
        bars[n_90 - 1].set_color("#C44E52")
        bars[n_90 - 1].set_alpha(1.0)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    ax.set_title(
        f"{display_name} -- PCA Variance Explained{label_suffix}\n"
        f"H3 res{resolution} | {len(data):,} hexagons | "
        f"90% at {n_90} components",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.legend(fontsize=10, loc="center right")
    ax.set_xlim(0.5, n_components + 0.5)
    ax.set_ylim(0, max(var_ratio[0] * 100 * 1.15, 100))

    # Add secondary y-axis for cumulative (right side)
    ax2 = ax.twinx()
    ax2.set_ylabel("Cumulative %", fontsize=10, color="#C44E52")
    ax2.set_ylim(0, max(var_ratio[0] * 100 * 1.15, 100))
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    plt.tight_layout()
    path = out_dir / "pca_variance_explained.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)

    return pca, n_90


# ---------------------------------------------------------------------------
# Plot 2: Top-3 PCA components as spatial maps
# ---------------------------------------------------------------------------


def plot_pca_spatial(
    emb_df: pd.DataFrame,
    emb_cols: list[str],
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    display_name: str,
    resolution: int,
    stamp: int = 2,
    transit_mask: np.ndarray | None = None,
):
    """3-panel spatial map of PC1, PC2, PC3 for transit hexagons."""
    # Fit PCA on transit hexagons only, then transform all
    if transit_mask is not None:
        transit_data = emb_df.loc[transit_mask, emb_cols].values.astype(np.float32)
    else:
        transit_data = emb_df[emb_cols].values.astype(np.float32)

    n_components = min(3, transit_data.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(transit_data)

    # Transform ALL hexagons for spatial context, but only show transit ones
    all_data = emb_df[emb_cols].values.astype(np.float32)
    all_reduced = pca.transform(all_data)

    # Use transit-only centroids and values
    if transit_mask is not None:
        cx_t = cx[transit_mask]
        cy_t = cy[transit_mask]
        reduced = all_reduced[transit_mask]
    else:
        cx_t, cy_t = cx, cy
        reduced = all_reduced

    fig, axes = plt.subplots(1, 3, figsize=(24, 10), dpi=DPI)
    fig.set_facecolor("white")

    for i in range(3):
        if i < n_components:
            vals = reduced[:, i]
            image = rasterize_continuous(cx_t, cy_t, vals, extent, cmap="RdBu_r", stamp=stamp)
            var_pct = pca.explained_variance_ratio_[i] * 100
            plot_spatial_map(
                axes[i], image, extent, boundary_gdf,
                title=f"PC{i + 1} ({var_pct:.1f}% var)",
            )
            v2, v98 = np.nanpercentile(vals, [2, 98])
            _add_colorbar(fig, axes[i], "RdBu_r", v2, v98, label=f"PC{i + 1}")
        else:
            axes[i].set_visible(False)

    total_var = sum(pca.explained_variance_ratio_[:n_components]) * 100
    n_shown = len(reduced)
    fig.suptitle(
        f"{display_name} -- PCA Top-3 Components (transit hexagons)\n"
        f"H3 res{resolution} | {n_shown:,} hexagons | "
        f"Total variance: {total_var:.1f}%",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "pca_top3_spatial.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)

    return all_reduced, pca


# ---------------------------------------------------------------------------
# Plot 3: Coverage map
# ---------------------------------------------------------------------------


def plot_coverage_map(
    cx_all: np.ndarray,
    cy_all: np.ndarray,
    cx_transit: np.ndarray,
    cy_transit: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    display_name: str,
    resolution: int,
    n_total: int,
    n_transit: int,
    stamp: int = 2,
):
    """Two-panel map: all hexagons (gray) vs transit hexagons (blue)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 12), dpi=DPI)
    fig.set_facecolor("white")

    # Panel 1: All hexagons
    img_all = rasterize_binary(cx_all, cy_all, extent, color=(0.7, 0.7, 0.7), stamp=stamp)
    # Overlay transit hexagons in blue
    img_transit_on_all = rasterize_binary(
        cx_transit, cy_transit, extent, color=(0.15, 0.45, 0.80), stamp=stamp,
    )
    # Composite: all gray, transit blue on top
    composite = img_all.copy()
    transit_pixels = img_transit_on_all[:, :, 3] > 0
    composite[transit_pixels] = img_transit_on_all[transit_pixels]

    plot_spatial_map(
        axes[0], composite, extent, boundary_gdf,
        title=f"Transit Coverage\n{n_transit:,} transit (blue) / {n_total:,} total (gray)",
    )

    # Panel 2: Transit hexagons only (zoomed to coverage area)
    img_transit = rasterize_binary(
        cx_transit, cy_transit, extent, color=(0.15, 0.45, 0.80), stamp=stamp,
    )
    plot_spatial_map(axes[1], img_transit, extent, boundary_gdf, title="Transit hexagons only")

    coverage_pct = 100.0 * n_transit / n_total if n_total > 0 else 0
    fig.suptitle(
        f"{display_name} -- Coverage Map\n"
        f"H3 res{resolution} | {coverage_pct:.1f}% coverage ({n_transit:,} / {n_total:,})",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "coverage_map.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 4: UMAP 2D scatter colored by cluster
# ---------------------------------------------------------------------------


def plot_umap_clusters(
    emb_df: pd.DataFrame,
    emb_cols: list[str],
    out_dir: Path,
    display_name: str,
    resolution: int,
    transit_mask: np.ndarray | None = None,
    n_clusters: int = 8,
    max_points: int = 50_000,
):
    """UMAP 2D projection colored by KMeans cluster assignment."""
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP plot")
        return

    if transit_mask is not None:
        data = emb_df.loc[transit_mask, emb_cols].values.astype(np.float32)
    else:
        data = emb_df[emb_cols].values.astype(np.float32)

    n_points = len(data)

    # Subsample for speed if needed
    rng = np.random.RandomState(42)
    if n_points > max_points:
        idx = rng.choice(n_points, max_points, replace=False)
        data_sub = data[idx]
        logger.info("  UMAP: subsampled %d -> %d points", n_points, max_points)
    else:
        data_sub = data
        idx = np.arange(n_points)

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_sub)

    # KMeans clustering
    logger.info("  KMeans clustering (k=%d)...", n_clusters)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42,
        batch_size=min(10000, len(data_scaled)),
        max_iter=100, n_init=3, init="k-means++",
    )
    clusters = kmeans.fit_predict(data_scaled)

    # UMAP
    logger.info("  Computing UMAP projection...")
    reducer = umap.UMAP(
        n_components=2, random_state=42, n_neighbors=15, min_dist=0.1,
        metric="euclidean",
    )
    embedding_2d = reducer.fit_transform(data_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=DPI)
    fig.set_facecolor("white")

    cmap = plt.get_cmap("tab10")
    for k in range(n_clusters):
        mask_k = clusters == k
        ax.scatter(
            embedding_2d[mask_k, 0], embedding_2d[mask_k, 1],
            c=[cmap(k / max(n_clusters - 1, 1))], s=3, alpha=0.5,
            label=f"Cluster {k}", rasterized=True,
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        f"{display_name} -- UMAP Projection\n"
        f"H3 res{resolution} | {len(data_sub):,} transit hexagons | "
        f"{n_clusters} KMeans clusters",
        fontsize=13, fontweight="bold",
    )
    ax.legend(
        fontsize=8, markerscale=3, loc="best",
        framealpha=0.9, ncol=2,
    )

    plt.tight_layout()
    path = out_dir / "umap_clusters.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate GTFS embedding interpretability maps",
    )
    parser.add_argument("--study-area", default="netherlands", help="Study area name")
    parser.add_argument("--modality", default="gtfs", help="Modality name (default: gtfs)")
    parser.add_argument("--resolution", type=int, default=9, help="H3 resolution (default: 9)")
    parser.add_argument(
        "--embedding-file", default=None,
        help="Embedding filename (relative to modality dir). "
             "Default: auto-detect from modality dir.",
    )
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of UMAP clusters")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = StudyAreaPaths(args.study_area)
    resolution = args.resolution
    modality = args.modality
    display_name = f"{modality.upper()} ({args.study_area})"

    # Load embeddings
    logger.info("=" * 60)
    logger.info("Processing: %s res%d", display_name, resolution)
    logger.info("=" * 60)

    df, emb_cols = load_gtfs_embeddings(paths, resolution, args.embedding_file)
    logger.info("  Loaded %d hexagons, %d dims", len(df), len(emb_cols))

    # Identify transit vs default hexagons
    transit_mask = identify_transit_hexagons(df, emb_cols)
    n_transit = transit_mask.sum()
    n_total = len(df)

    # Output directory
    out_dir = paths.stage1(modality) / "plots" / f"res{resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Output dir: %s", out_dir)

    # Get centroids via SpatialDB
    db = SpatialDB.for_study_area(args.study_area)
    cx_all, cy_all = db.centroids(df.index, resolution=resolution, crs=28992)
    cx_transit = cx_all[transit_mask]
    cy_transit = cy_all[transit_mask]
    logger.info("  Centroids loaded: %d total, %d transit", len(cx_all), len(cx_transit))

    # Compute extent from boundary or centroids
    boundary_gdf = load_boundary(paths)
    if boundary_gdf is not None:
        ext = boundary_gdf.total_bounds
    else:
        ext = db.extent(df.index, resolution=resolution, crs=28992)
    minx, miny, maxx, maxy = ext
    pad = (maxx - minx) * 0.03
    extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    stamp = max(1, 11 - resolution)  # res9 -> stamp=2

    # Plot 1: PCA variance explained
    logger.info("[1/4] PCA variance explained...")
    pca, n_90 = plot_pca_variance(
        df, emb_cols, out_dir, display_name, resolution,
        transit_only=True, transit_mask=transit_mask,
    )

    # Plot 2: Top-3 PCA spatial maps
    logger.info("[2/4] PCA top-3 spatial maps...")
    plot_pca_spatial(
        df, emb_cols, cx_all, cy_all, extent, boundary_gdf,
        out_dir, display_name, resolution, stamp=stamp,
        transit_mask=transit_mask,
    )

    # Plot 3: Coverage map
    logger.info("[3/4] Coverage map...")
    plot_coverage_map(
        cx_all, cy_all, cx_transit, cy_transit,
        extent, boundary_gdf, out_dir, display_name, resolution,
        n_total, n_transit, stamp=stamp,
    )

    # Plot 4: UMAP clusters
    logger.info("[4/4] UMAP clusters...")
    plot_umap_clusters(
        df, emb_cols, out_dir, display_name, resolution,
        transit_mask=transit_mask, n_clusters=args.n_clusters,
    )

    logger.info("=" * 60)
    logger.info("Done! 4 plots saved to %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
