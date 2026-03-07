#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-modality unimodal embedding diagnostic plots.

Generates standardized diagnostic visualizations for Stage 1 embeddings:
  1. PCA RGB spatial map -- top 3 PCs mapped as RGB channels on Netherlands map
  2. Embedding distributions -- KDE of top PCA components
  3. Cluster maps -- MiniBatchKMeans with k=8 and k=12
  4. Coverage map -- which hexagons have embeddings vs total tessellation
  5. PCA explained variance curve

Also generates leefbaarometer target distribution plots (separate from embeddings).

Lifetime: durable
Stage: 1 (data) + 3 (analysis/visualization)

Usage::

    # All modalities
    python scripts/stage1/plot_unimodal_diagnostics.py --study-area netherlands --resolution 9

    # Single modality
    python scripts/stage1/plot_unimodal_diagnostics.py --study-area netherlands --resolution 9 --modality alphaearth

    # Leefbaarometer target only
    python scripts/stage1/plot_unimodal_diagnostics.py --study-area netherlands --resolution 9 --modality leefbaarometer

    # Specify year
    python scripts/stage1/plot_unimodal_diagnostics.py --study-area netherlands --resolution 9 --year 2022
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import StudyAreaPaths
from utils.spatial_db import SpatialDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modality registry: maps modality name -> (sub_embedder, year_key, col_prefix)
# ---------------------------------------------------------------------------
MODALITY_REGISTRY = {
    "alphaearth": {
        "sub_embedder": None,
        "year": "2022",
        "embedding_col_prefixes": ("A",),
        "index_col": "h3_index",  # known inconsistency -- AlphaEarth uses h3_index
        "exclude_cols": {"pixel_count", "tile_count", "h3_resolution", "h3_index"},
        "display_name": "AlphaEarth (Google EE)",
    },
    "poi/hex2vec": {
        "sub_embedder": "hex2vec",
        "year": "latest",
        "embedding_col_prefixes": ("hex2vec_",),
        "index_col": "region_id",
        "exclude_cols": set(),
        "display_name": "POI (hex2vec)",
    },
    "roads": {
        "sub_embedder": None,
        "year": "latest",
        "embedding_col_prefixes": ("R",),
        "index_col": "region_id",
        "exclude_cols": set(),
        "display_name": "Roads (highway2vec)",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(
    study_area: str,
    modality: str,
    resolution: int,
    year: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load embedding parquet and return (DataFrame indexed by region_id, embedding_cols).

    Handles the AlphaEarth h3_index inconsistency by normalizing to region_id index.
    """
    paths = StudyAreaPaths(study_area)
    config = MODALITY_REGISTRY[modality]

    # Determine base modality and sub-embedder
    if "/" in modality:
        base_mod, sub_emb = modality.split("/", 1)
    else:
        base_mod = modality
        sub_emb = config["sub_embedder"]

    file_year = year or config["year"]
    emb_path = paths.embedding_file(base_mod, resolution, year=file_year, sub_embedder=sub_emb)

    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")

    logger.info(f"Loading {modality} from {emb_path}")
    df = pd.read_parquet(emb_path)

    # Normalize index to region_id
    if df.index.name == "region_id":
        pass  # already correct
    elif "h3_index" in df.columns:
        df = df.set_index("h3_index")
        df.index.name = "region_id"
    elif "region_id" in df.columns:
        df = df.set_index("region_id")
    else:
        raise ValueError(f"Cannot find region_id or h3_index in {emb_path}")

    # Identify embedding columns
    exclude = config["exclude_cols"] | {"geometry"}
    embedding_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    logger.info(f"  {len(df):,} hexagons, {len(embedding_cols)} dimensions")
    return df[embedding_cols], embedding_cols


def load_boundary(study_area: str) -> Optional[gpd.GeoDataFrame]:
    """Load study area boundary for background rendering."""
    paths = StudyAreaPaths(study_area)
    boundary_path = paths.area_gdf_file("geojson")
    if not boundary_path.exists():
        boundary_path = paths.area_gdf_file("parquet")
    if boundary_path.exists():
        gdf = gpd.read_file(boundary_path) if boundary_path.suffix == ".geojson" else gpd.read_parquet(boundary_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf.to_crs(epsg=28992)
    return None


# ---------------------------------------------------------------------------
# Centroid rasterization (extracted from linear_probe_viz._rasterize_centroids)
# ---------------------------------------------------------------------------

def rasterize_centroids(
    hex_ids: pd.Index,
    rgb_array: np.ndarray,
    extent: Tuple[float, float, float, float],
    study_area: str,
    resolution: int,
    width: int = 1200,
    height: int = 1400,
) -> np.ndarray:
    """Rasterize H3 centroids to an RGBA image via SpatialDB.

    This is a standalone extraction of the pattern from
    ``LinearProbeVisualizer._rasterize_centroids``.

    Args:
        hex_ids: Index of H3 hex ID strings.
        rgb_array: (N, 3) float array with R, G, B in [0, 1].
        extent: (minx, miny, maxx, maxy) in EPSG:28992.
        study_area: Study area name for SpatialDB lookup.
        resolution: H3 resolution.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        (height, width, 4) RGBA float32 array with white background.
    """
    db = SpatialDB.for_study_area(study_area)
    all_cx, all_cy = db.centroids(hex_ids, resolution=resolution, crs=28992)

    minx, miny, maxx, maxy = extent
    mask = (
        np.isfinite(all_cx) & np.isfinite(all_cy)
        & (all_cx >= minx) & (all_cx <= maxx)
        & (all_cy >= miny) & (all_cy <= maxy)
    )

    cx = all_cx[mask]
    cy = all_cy[mask]
    rgb_masked = rgb_array[mask]

    # Map to pixel coordinates
    px = ((cx - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    # Write into RGBA image (white background)
    image = np.ones((height, width, 4), dtype=np.float32)
    image[py, px, :3] = rgb_masked
    image[py, px, 3] = 1.0

    return image


def get_render_extent(
    hex_ids: pd.Index,
    study_area: str,
    resolution: int,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Tuple[float, float, float, float]:
    """Compute rendering extent in EPSG:28992, preferring boundary if available."""
    if boundary_gdf is not None:
        b = boundary_gdf.total_bounds
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    db = SpatialDB.for_study_area(study_area)
    return db.extent(hex_ids, resolution=resolution, crs=28992)


# ---------------------------------------------------------------------------
# Plot 1: PCA RGB spatial map
# ---------------------------------------------------------------------------

def plot_pca_rgb_map(
    emb_df: pd.DataFrame,
    embedding_cols: List[str],
    study_area: str,
    resolution: int,
    modality: str,
    output_dir: Path,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Path:
    """Top 3 PCA components mapped as RGB channels on spatial map."""
    display_name = MODALITY_REGISTRY[modality]["display_name"]
    logger.info(f"Generating PCA RGB map for {display_name}...")

    # PCA
    data = emb_df[embedding_cols].values.astype(np.float32)
    n_components = min(3, data.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(data)

    # Normalize each component to [0, 1] via percentile clipping
    rgb = np.zeros((len(pca_result), 3), dtype=np.float32)
    for i in range(n_components):
        col = pca_result[:, i]
        lo, hi = np.percentile(col, [2, 98])
        if hi - lo < 1e-8:
            rgb[:, i] = 0.5
        else:
            rgb[:, i] = np.clip((col - lo) / (hi - lo), 0, 1)

    hex_ids = pd.Index(emb_df.index, name="region_id")
    extent = get_render_extent(hex_ids, study_area, resolution, boundary_gdf)
    raster = rasterize_centroids(hex_ids, rgb, extent, study_area, resolution)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 14), dpi=150)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)

    ax.imshow(
        raster,
        extent=[extent[0], extent[2], extent[1], extent[3]],
        origin="lower", aspect="equal", interpolation="nearest", zorder=2,
    )

    var_pct = pca.explained_variance_ratio_[:3] * 100
    var_labels = " | ".join(f"PC{i+1}: {v:.1f}%" for i, v in enumerate(var_pct))
    ax.set_title(
        f"{display_name} -- PCA RGB Map (Res {resolution})\n"
        f"R=PC1, G=PC2, B=PC3 | {var_labels}\n"
        f"{len(emb_df):,} hexagons, {len(embedding_cols)}D embeddings",
        fontsize=12, fontweight="bold", pad=10,
    )

    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.grid(True, linewidth=0.3, alpha=0.4, color="gray")
    ax.tick_params(labelsize=7)

    output_path = output_dir / "pca_rgb_map.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Plot 2: Embedding distributions (KDE of top PCA components)
# ---------------------------------------------------------------------------

def plot_embedding_distributions(
    emb_df: pd.DataFrame,
    embedding_cols: List[str],
    modality: str,
    resolution: int,
    output_dir: Path,
    n_components: int = 6,
) -> Path:
    """KDE + histogram of top PCA components."""
    display_name = MODALITY_REGISTRY[modality]["display_name"]
    logger.info(f"Generating embedding distributions for {display_name}...")

    data = emb_df[embedding_cols].values.astype(np.float32)
    n_pca = min(n_components, data.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    pca_result = pca.fit_transform(data)

    ncols = 3
    nrows = int(np.ceil(n_pca / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_pca))

    for i in range(n_pca):
        ax = axes[i]
        vals = pca_result[:, i]
        var_pct = pca.explained_variance_ratio_[i] * 100

        ax.hist(vals, bins=100, density=True, alpha=0.6, color=colors[i], edgecolor="none")

        # KDE overlay
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 300)
            ax.plot(x_range, kde(x_range), color="black", linewidth=1.5)
        except Exception:
            pass  # KDE can fail on degenerate distributions

        ax.axvline(np.mean(vals), color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"mean={np.mean(vals):.2f}")
        ax.axvline(np.median(vals), color="blue", linestyle=":", linewidth=1, alpha=0.7, label=f"median={np.median(vals):.2f}")
        ax.set_title(f"PC{i+1} ({var_pct:.1f}% var)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylabel("Density" if i % ncols == 0 else "")

    # Hide unused axes
    for i in range(n_pca, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"{display_name} -- PCA Component Distributions (Res {resolution})\n"
        f"{len(emb_df):,} hexagons, {len(embedding_cols)}D -> {n_pca} PCs",
        fontsize=13, fontweight="bold", y=1.02,
    )

    output_path = output_dir / "embedding_distributions.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Plot 3: Cluster maps (MiniBatchKMeans k=8 and k=12)
# ---------------------------------------------------------------------------

def plot_cluster_maps(
    emb_df: pd.DataFrame,
    embedding_cols: List[str],
    study_area: str,
    resolution: int,
    modality: str,
    output_dir: Path,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    k_values: Tuple[int, ...] = (8, 12),
) -> List[Path]:
    """Generate cluster maps using MiniBatchKMeans, rendered via centroid rasterization."""
    display_name = MODALITY_REGISTRY[modality]["display_name"]
    logger.info(f"Generating cluster maps for {display_name} (k={k_values})...")

    data = emb_df[embedding_cols].values.astype(np.float32)

    # PCA reduction for clustering efficiency
    n_pca = min(16, data.shape[1])
    if data.shape[1] > n_pca:
        pca = PCA(n_components=n_pca, random_state=42)
        data_reduced = pca.fit_transform(data)
    else:
        data_reduced = data

    data_scaled = StandardScaler().fit_transform(data_reduced)
    hex_ids = pd.Index(emb_df.index, name="region_id")
    extent = get_render_extent(hex_ids, study_area, resolution, boundary_gdf)

    output_paths = []
    for k in k_values:
        logger.info(f"  Clustering k={k}...")
        t0 = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=k, random_state=42,
            batch_size=min(10000, len(data_scaled)),
            max_iter=100, n_init=3, init="k-means++", verbose=0,
        )
        labels = kmeans.fit_predict(data_scaled)
        logger.info(f"    k={k} completed in {time.time() - t0:.1f}s")

        # Map cluster labels to distinct colors
        cmap = plt.cm.tab20 if k > 10 else plt.cm.tab10
        norm = Normalize(vmin=0, vmax=max(k - 1, 1))
        rgb = np.array([cmap(norm(l))[:3] for l in labels], dtype=np.float32)

        raster = rasterize_centroids(hex_ids, rgb, extent, study_area, resolution)

        fig, ax = plt.subplots(figsize=(12, 14), dpi=150)
        fig.set_facecolor("white")
        ax.set_facecolor("white")

        if boundary_gdf is not None:
            boundary_gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)

        ax.imshow(
            raster,
            extent=[extent[0], extent[2], extent[1], extent[3]],
            origin="lower", aspect="equal", interpolation="nearest", zorder=2,
        )

        ax.set_title(
            f"{display_name} -- {k} Clusters (MiniBatchKMeans, Res {resolution})\n"
            f"{len(emb_df):,} hexagons, {len(embedding_cols)}D embeddings",
            fontsize=12, fontweight="bold", pad=10,
        )

        # Add legend
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor=cmap(norm(i)), label=f"Cluster {i}")
            for i in range(k)
        ]
        ax.legend(
            handles=legend_patches, loc="lower right",
            fontsize=7, ncol=2, framealpha=0.9,
            title="Clusters", title_fontsize=9,
        )

        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.grid(True, linewidth=0.3, alpha=0.4, color="gray")
        ax.tick_params(labelsize=7)

        out_path = output_dir / f"clusters_k{k}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"    Saved: {out_path}")
        output_paths.append(out_path)

    return output_paths


# ---------------------------------------------------------------------------
# Plot 4: Coverage map
# ---------------------------------------------------------------------------

def plot_coverage_map(
    emb_df: pd.DataFrame,
    study_area: str,
    resolution: int,
    modality: str,
    output_dir: Path,
    boundary_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Path:
    """Coverage map: which hexagons have embeddings vs total tessellation."""
    display_name = MODALITY_REGISTRY[modality]["display_name"]
    logger.info(f"Generating coverage map for {display_name}...")

    paths = StudyAreaPaths(study_area)
    region_path = paths.region_file(resolution)
    if not region_path.exists():
        raise FileNotFoundError(f"Region file not found: {region_path}")

    all_regions = gpd.read_parquet(region_path)
    if all_regions.index.name != "region_id":
        all_regions.index.name = "region_id"

    total_hexagons = len(all_regions)
    covered_ids = set(emb_df.index)
    n_covered = len(covered_ids)
    coverage_pct = 100.0 * n_covered / total_hexagons if total_hexagons > 0 else 0.0

    # Create coverage indicator: 1 = covered, 0 = not covered
    coverage = np.zeros(total_hexagons, dtype=np.float32)
    all_ids = all_regions.index
    covered_mask = all_ids.isin(covered_ids)
    coverage[covered_mask] = 1.0

    # RGB: green for covered, light gray for not covered
    rgb = np.full((total_hexagons, 3), 0.85, dtype=np.float32)  # light gray
    rgb[covered_mask] = [0.2, 0.7, 0.3]  # green

    hex_ids = pd.Index(all_ids, name="region_id")
    extent = get_render_extent(hex_ids, study_area, resolution, boundary_gdf)
    raster = rasterize_centroids(hex_ids, rgb, extent, study_area, resolution)

    fig, ax = plt.subplots(figsize=(12, 14), dpi=150)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor="#f8f8f8", edgecolor="#cccccc", linewidth=0.5)

    ax.imshow(
        raster,
        extent=[extent[0], extent[2], extent[1], extent[3]],
        origin="lower", aspect="equal", interpolation="nearest", zorder=2,
    )

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=[0.2, 0.7, 0.3], label=f"Covered ({n_covered:,})"),
        Patch(facecolor=[0.85, 0.85, 0.85], label=f"Missing ({total_hexagons - n_covered:,})"),
    ]
    ax.legend(
        handles=legend_patches, loc="lower right",
        fontsize=10, framealpha=0.9,
        title=f"Coverage: {coverage_pct:.1f}%", title_fontsize=11,
    )

    ax.set_title(
        f"{display_name} -- Coverage Map (Res {resolution})\n"
        f"{n_covered:,} / {total_hexagons:,} hexagons ({coverage_pct:.1f}%)",
        fontsize=12, fontweight="bold", pad=10,
    )

    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.grid(True, linewidth=0.3, alpha=0.4, color="gray")
    ax.tick_params(labelsize=7)

    output_path = output_dir / "coverage_map.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Plot 5: PCA explained variance curve
# ---------------------------------------------------------------------------

def plot_pca_variance_curve(
    emb_df: pd.DataFrame,
    embedding_cols: List[str],
    modality: str,
    resolution: int,
    output_dir: Path,
    max_components: int = 30,
) -> Path:
    """PCA explained variance curve with cumulative line."""
    display_name = MODALITY_REGISTRY[modality]["display_name"]
    logger.info(f"Generating PCA variance curve for {display_name}...")

    data = emb_df[embedding_cols].values.astype(np.float32)
    n_components = min(max_components, data.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(data)

    var_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for individual variance
    x = np.arange(1, n_components + 1)
    ax1.bar(x, var_ratio * 100, alpha=0.7, color="steelblue", label="Individual")
    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Explained Variance (%)", fontsize=11, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Cumulative line on twin axis
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative * 100, "o-", color="darkorange", linewidth=2, markersize=4, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=11, color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 105)

    # Reference lines
    for threshold in [80, 90, 95]:
        ax2.axhline(threshold, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        # Find component reaching threshold
        idx = np.searchsorted(cumulative * 100, threshold)
        if idx < n_components:
            ax2.annotate(
                f"{threshold}% at PC{idx+1}",
                xy=(idx + 1, threshold), fontsize=8, color="gray",
                ha="left", va="bottom",
            )

    ax1.set_title(
        f"{display_name} -- PCA Explained Variance (Res {resolution})\n"
        f"{len(embedding_cols)}D embeddings, {len(emb_df):,} hexagons",
        fontsize=12, fontweight="bold",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    output_path = output_dir / "pca_variance_curve.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Leefbaarometer target distributions
# ---------------------------------------------------------------------------

TARGET_SCORE_NAMES = {
    "lbm": "Overall Liveability",
    "fys": "Physical Environment",
    "onv": "Safety",
    "soc": "Social Cohesion",
    "vrz": "Amenities",
    "won": "Housing",
}


def plot_leefbaarometer_distributions(
    study_area: str,
    resolution: int,
    year: int = 2022,
) -> Path:
    """Generate 2x3 grid of KDE + histogram plots for leefbaarometer scores.

    Output is placed in the target data directory, not the embedding directory,
    because this is TARGET data -- visually distinct from embedding distributions.
    """
    paths = StudyAreaPaths(study_area)
    target_path = paths.target_file("leefbaarometer", resolution, year)

    if not target_path.exists():
        raise FileNotFoundError(f"Leefbaarometer target not found: {target_path}")

    logger.info(f"Loading leefbaarometer target from {target_path}")
    df = pd.read_parquet(target_path)
    if df.index.name != "region_id" and "region_id" in df.columns:
        df = df.set_index("region_id")

    score_cols = [c for c in TARGET_SCORE_NAMES.keys() if c in df.columns]
    if not score_cols:
        raise ValueError(f"No score columns found. Available: {list(df.columns)}")

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
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_modality_diagnostics(
    study_area: str,
    resolution: int,
    modality: str,
    year: Optional[str] = None,
) -> Dict[str, Path]:
    """Run all 5 diagnostic plots for a single modality.

    Returns dict mapping plot name to output path.
    """
    if modality not in MODALITY_REGISTRY:
        raise ValueError(f"Unknown modality: {modality}. Available: {list(MODALITY_REGISTRY.keys())}")

    emb_df, embedding_cols = load_embeddings(study_area, modality, resolution, year)
    boundary_gdf = load_boundary(study_area)

    # Determine output directory
    paths = StudyAreaPaths(study_area)
    if "/" in modality:
        base_mod, sub_emb = modality.split("/", 1)
        output_dir = paths.stage1(base_mod) / sub_emb / "plots" / f"res{resolution}"
    else:
        output_dir = paths.stage1(modality) / "plots" / f"res{resolution}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    logger.info(f"--- {MODALITY_REGISTRY[modality]['display_name']} (res{resolution}) ---")

    # 1. PCA RGB map
    results["pca_rgb_map"] = plot_pca_rgb_map(
        emb_df, embedding_cols, study_area, resolution, modality, output_dir, boundary_gdf,
    )

    # 2. Embedding distributions
    results["embedding_distributions"] = plot_embedding_distributions(
        emb_df, embedding_cols, modality, resolution, output_dir,
    )

    # 3. Cluster maps
    cluster_paths = plot_cluster_maps(
        emb_df, embedding_cols, study_area, resolution, modality, output_dir, boundary_gdf,
    )
    for p in cluster_paths:
        results[p.stem] = p

    # 4. Coverage map
    results["coverage_map"] = plot_coverage_map(
        emb_df, study_area, resolution, modality, output_dir, boundary_gdf,
    )

    # 5. PCA variance curve
    results["pca_variance_curve"] = plot_pca_variance_curve(
        emb_df, embedding_cols, modality, resolution, output_dir,
    )

    return results


def run_all_diagnostics(
    study_area: str,
    resolution: int,
    modalities: Optional[List[str]] = None,
    year: Optional[str] = None,
    include_leefbaarometer: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """Run diagnostics for all specified modalities + leefbaarometer.

    Returns nested dict: modality -> plot_name -> path.
    """
    if modalities is None:
        modalities = list(MODALITY_REGISTRY.keys())

    all_results = {}
    for mod in modalities:
        try:
            all_results[mod] = run_modality_diagnostics(study_area, resolution, mod, year)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {mod}: {e}")
        except Exception as e:
            logger.error(f"Error processing {mod}: {e}", exc_info=True)

    if include_leefbaarometer:
        try:
            lbm_year = int(year) if year and year.isdigit() else 2022
            lbm_path = plot_leefbaarometer_distributions(study_area, resolution, lbm_year)
            all_results["leefbaarometer"] = {"distributions": lbm_path}
        except FileNotFoundError as e:
            logger.warning(f"Skipping leefbaarometer: {e}")
        except Exception as e:
            logger.error(f"Error processing leefbaarometer: {e}", exc_info=True)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-modality diagnostic plots for Stage 1 embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--study-area", default="netherlands", help="Study area name")
    parser.add_argument("--resolution", type=int, default=9, help="H3 resolution")
    parser.add_argument(
        "--modality", default="all",
        help="Modality to plot (alphaearth, poi/hex2vec, roads, leefbaarometer, or 'all')",
    )
    parser.add_argument("--year", default=None, help="Data year override (e.g. 2022, latest)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()

    if args.modality == "all":
        results = run_all_diagnostics(args.study_area, args.resolution, year=args.year)
    elif args.modality == "leefbaarometer":
        lbm_year = int(args.year) if args.year and args.year.isdigit() else 2022
        path = plot_leefbaarometer_distributions(args.study_area, args.resolution, lbm_year)
        results = {"leefbaarometer": {"distributions": path}}
    else:
        results = {
            args.modality: run_modality_diagnostics(
                args.study_area, args.resolution, args.modality, args.year,
            )
        }

    elapsed = time.time() - t0
    logger.info(f"Completed in {elapsed:.1f}s")
    for mod, plots in results.items():
        logger.info(f"  {mod}:")
        for name, path in plots.items():
            logger.info(f"    {name}: {path}")


if __name__ == "__main__":
    main()
