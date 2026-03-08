#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summary stats and spatial maps for concatenated multi-modality embeddings.

Generates 4 plots for the concat embeddings:
  1. Per-modality variance contribution -- bar chart
  2. PCA top-3 components -- spatial maps (rasterized)
  3. Modality correlation matrix -- heatmap of mean-per-block correlations
  4. Coverage/density map -- which modality blocks are non-zero per hexagon

Reuses rendering infrastructure from scripts/plot_embeddings.py
(rasterize_continuous, plot_spatial_map, etc.).

Lifetime: durable
Stage: 2 visualization / 3 analysis

Usage:
    python scripts/stage3/plot_concat_embeddings.py
    python scripts/stage3/plot_concat_embeddings.py --resolution 10
    python scripts/stage3/plot_concat_embeddings.py --year 2022
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
import seaborn as sns
from sklearn.decomposition import PCA

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import StudyAreaPaths
from utils.spatial_db import SpatialDB
from scripts.plot_embeddings import (
    load_boundary,
    rasterize_continuous,
    plot_spatial_map,
    _add_colorbar,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DPI = 300
RASTER_W = 2000
RASTER_H = 2400

# Modality block definitions: prefix -> (display_name, color)
MODALITY_BLOCKS = {
    "A": ("AlphaEarth", "#2196F3"),
    "P": ("POI", "#4CAF50"),
    "R": ("Roads", "#FF9800"),
    "G": ("GTFS", "#9C27B0"),
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column grouping
# ---------------------------------------------------------------------------


def group_columns_by_modality(columns: list[str]) -> dict[str, list[str]]:
    """Group embedding columns by their single-letter prefix.

    Columns must match the pattern: single uppercase letter followed by digits
    (e.g. A00, P123, R05). Columns not matching are ignored.
    """
    groups: dict[str, list[str]] = {}
    for col in columns:
        if len(col) >= 2 and col[0].isalpha() and col[0].isupper():
            suffix = col[1:]
            if suffix.isdigit() or suffix.replace("_", "").isdigit():
                prefix = col[0]
                groups.setdefault(prefix, []).append(col)
    return groups


# ---------------------------------------------------------------------------
# Plot 1: Per-modality variance contribution
# ---------------------------------------------------------------------------


def plot_variance_contribution(
    emb_df: pd.DataFrame,
    groups: dict[str, list[str]],
    out_dir: Path,
    resolution: int,
):
    """Bar chart showing fraction of total variance each modality contributes."""
    total_var = emb_df.var(axis=0).sum()
    modality_vars = {}
    for prefix, cols in sorted(groups.items()):
        block_var = emb_df[cols].var(axis=0).sum()
        modality_vars[prefix] = block_var

    labels = []
    fractions = []
    abs_vars = []
    colors = []
    for prefix in sorted(modality_vars.keys()):
        info = MODALITY_BLOCKS.get(prefix, (prefix, "#888888"))
        labels.append(f"{info[0]}\n({len(groups[prefix])} dims)")
        fractions.append(modality_vars[prefix] / total_var * 100)
        abs_vars.append(modality_vars[prefix])
        colors.append(info[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.set_facecolor("white")

    # Left: fraction of total variance
    bars1 = axes[0].bar(labels, fractions, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_ylabel("Share of Total Variance (%)", fontsize=11)
    axes[0].set_title("Variance Contribution by Modality", fontsize=13, fontweight="bold")
    for bar, frac in zip(bars1, fractions):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{frac:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    axes[0].set_ylim(0, max(fractions) * 1.15)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: mean variance per dimension
    mean_vars = [modality_vars[p] / len(groups[p]) for p in sorted(modality_vars.keys())]
    bars2 = axes[1].bar(labels, mean_vars, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_ylabel("Mean Variance per Dimension", fontsize=11)
    axes[1].set_title("Mean Per-Dimension Variance", fontsize=13, fontweight="bold")
    for bar, mv in zip(bars2, mean_vars):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mean_vars) * 0.01,
            f"{mv:.2f}", ha="center", va="bottom", fontsize=10,
        )
    axes[1].set_ylim(0, max(mean_vars) * 1.15)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    n_hexagons = len(emb_df)
    n_dims = emb_df.shape[1]
    fig.suptitle(
        f"Concatenated Embeddings -- Variance Breakdown\n"
        f"H3 res{resolution} | {n_hexagons:,} hexagons | {n_dims} total dimensions",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = out_dir / "variance_contribution.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 2: PCA top-3 spatial maps
# ---------------------------------------------------------------------------


def plot_pca_spatial(
    emb_df: pd.DataFrame,
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    resolution: int,
    groups: dict[str, list[str]],
    stamp: int = 2,
):
    """3-panel spatial map of PC1, PC2, PC3 on full concatenated embeddings.

    Also prints per-modality loading analysis for each PC.
    """
    embeddings = emb_df.values.astype(np.float32)
    n_components = 3
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)

    # Log loadings per modality block
    logger.info("  PCA explained variance: %s",
                [f"{v:.3f}" for v in pca.explained_variance_ratio_])
    for pc_idx in range(n_components):
        loadings = np.abs(pca.components_[pc_idx])
        for prefix, cols in sorted(groups.items()):
            col_indices = [emb_df.columns.get_loc(c) for c in cols]
            block_loading = loadings[col_indices].sum() / loadings.sum() * 100
            info = MODALITY_BLOCKS.get(prefix, (prefix, "#888"))
            logger.info("    PC%d %s loading share: %.1f%%", pc_idx + 1, info[0], block_loading)

    fig, axes = plt.subplots(1, 3, figsize=(24, 10), dpi=DPI)
    fig.set_facecolor("white")

    for i in range(3):
        vals = reduced[:, i]
        image = rasterize_continuous(
            cx, cy, vals, extent,
            width=RASTER_W, height=RASTER_H,
            cmap="RdBu_r", stamp=stamp,
        )
        var_pct = pca.explained_variance_ratio_[i] * 100
        plot_spatial_map(
            axes[i], image, extent, boundary_gdf,
            title=f"PC{i+1} ({var_pct:.1f}% var)",
        )
        v2, v98 = np.nanpercentile(vals, [2, 98])
        _add_colorbar(fig, axes[i], "RdBu_r", v2, v98, label=f"PC{i+1}")

    total_var = sum(pca.explained_variance_ratio_[:3]) * 100
    fig.suptitle(
        f"Concatenated Embeddings -- PCA Top-3 Components\n"
        f"H3 res{resolution} | {len(emb_df):,} hexagons | "
        f"Total variance: {total_var:.1f}%",
        fontsize=14, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "pca_top3_spatial.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 3: Modality correlation matrix
# ---------------------------------------------------------------------------


def plot_modality_correlation(
    emb_df: pd.DataFrame,
    groups: dict[str, list[str]],
    out_dir: Path,
    resolution: int,
):
    """Heatmap of inter-modality correlation.

    Computes mean embedding per modality block per hexagon, then
    pairwise Pearson correlation between those block-level means.
    """
    block_means = pd.DataFrame(index=emb_df.index)
    block_labels = []

    for prefix in sorted(groups.keys()):
        cols = groups[prefix]
        info = MODALITY_BLOCKS.get(prefix, (prefix, "#888"))
        label = info[0]
        block_means[label] = emb_df[cols].mean(axis=1)
        block_labels.append(label)

    corr = block_means.corr()

    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI)
    fig.set_facecolor("white")

    sns.heatmap(
        corr, ax=ax,
        annot=True, fmt=".3f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True,
        linewidths=1, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(
        f"Inter-Modality Correlation (mean embedding per block)\n"
        f"H3 res{resolution} | {len(emb_df):,} hexagons",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    path = out_dir / "modality_correlation.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)

    # Also log the correlation values
    for i, row_label in enumerate(block_labels):
        for j, col_label in enumerate(block_labels):
            if j > i:
                logger.info("    %s <-> %s: r=%.4f", row_label, col_label, corr.iloc[i, j])


# ---------------------------------------------------------------------------
# Plot 4: Coverage / density map
# ---------------------------------------------------------------------------


def plot_modality_coverage(
    emb_df: pd.DataFrame,
    groups: dict[str, list[str]],
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
    out_dir: Path,
    resolution: int,
    stamp: int = 2,
):
    """Spatial map showing how many modality blocks have non-zero signal per hexagon.

    A modality block is considered "present" if its mean absolute value > threshold.
    This catches zero-padded blocks (missing modalities).
    """
    threshold = 1e-6
    n_present = np.zeros(len(emb_df), dtype=np.int32)

    for prefix in sorted(groups.keys()):
        cols = groups[prefix]
        block_mean_abs = emb_df[cols].abs().mean(axis=1).values
        present = (block_mean_abs > threshold).astype(np.int32)
        n_present += present

    n_modalities = len(groups)
    logger.info("  Modality coverage distribution:")
    for k in range(n_modalities + 1):
        count = (n_present == k).sum()
        pct = 100 * count / len(emb_df)
        logger.info("    %d/%d modalities: %d hexagons (%.1f%%)", k, n_modalities, count, pct)

    # Spatial map colored by number of modalities present
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=DPI)
    fig.set_facecolor("white")

    # Left: count map
    image = rasterize_continuous(
        cx, cy, n_present.astype(np.float32), extent,
        width=RASTER_W, height=RASTER_H,
        cmap="YlOrRd",
        vmin=0, vmax=n_modalities,
        stamp=stamp,
    )
    plot_spatial_map(axes[0], image, extent, boundary_gdf, title="Modalities Present")
    _add_colorbar(fig, axes[0], "YlOrRd", 0, n_modalities, label="Count")

    # Right: bar chart of coverage distribution
    counts = [int((n_present == k).sum()) for k in range(n_modalities + 1)]
    bar_colors = plt.get_cmap("YlOrRd")(np.linspace(0.2, 0.9, n_modalities + 1))
    axes[1].bar(range(n_modalities + 1), counts, color=bar_colors, edgecolor="white")
    for k, c in enumerate(counts):
        if c > 0:
            axes[1].text(k, c + max(counts) * 0.01, f"{c:,}", ha="center", fontsize=9)
    axes[1].set_xlabel("Number of Modalities Present", fontsize=11)
    axes[1].set_ylabel("Number of Hexagons", fontsize=11)
    axes[1].set_title("Coverage Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xticks(range(n_modalities + 1))
    mod_names = sorted(
        [MODALITY_BLOCKS.get(p, (p,))[0] for p in groups.keys()]
    )
    axes[1].set_xticklabels(
        [f"{k}\n({', '.join(mod_names[:k]) if k <= 3 else '...'})"
         if k < n_modalities else f"{k}\n(all)"
         for k in range(n_modalities + 1)]
    )
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle(
        f"Concatenated Embeddings -- Modality Coverage\n"
        f"H3 res{resolution} | {len(emb_df):,} hexagons | "
        f"{n_modalities} modality blocks ({', '.join(mod_names)})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = out_dir / "modality_coverage.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot summary stats for concatenated multi-modality embeddings."
    )
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument("--resolution", type=int, default=9)
    parser.add_argument("--year", default="2022")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = StudyAreaPaths(args.study_area)

    # Load concat embeddings
    emb_path = (
        paths.model_embeddings("concat")
        / f"{args.study_area}_res{args.resolution}_{args.year}_raw.parquet"
    )
    if not emb_path.exists():
        logger.error("Concat embeddings not found: %s", emb_path)
        sys.exit(1)

    logger.info("Loading: %s", emb_path)
    emb_df = pd.read_parquet(emb_path)
    if emb_df.index.name != "region_id" and "region_id" in emb_df.columns:
        emb_df = emb_df.set_index("region_id")
    logger.info("Shape: %s", emb_df.shape)

    # Group columns by modality prefix
    groups = group_columns_by_modality(list(emb_df.columns))
    for prefix, cols in sorted(groups.items()):
        info = MODALITY_BLOCKS.get(prefix, (prefix,))
        logger.info("  %s (%s): %d columns", prefix, info[0], len(cols))

    if not groups:
        logger.error("No modality blocks detected in columns: %s", list(emb_df.columns[:10]))
        sys.exit(1)

    # Output directory
    out_dir = paths.plots("concat") / f"res{args.resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_dir)

    # Load spatial infrastructure
    db = SpatialDB.for_study_area(args.study_area)
    cx, cy = db.centroids(emb_df.index, resolution=args.resolution, crs=28992)
    logger.info("Centroids loaded: %d points", len(cx))

    boundary_gdf = load_boundary(paths)

    if boundary_gdf is not None:
        ext = boundary_gdf.total_bounds
    else:
        ext = db.extent(emb_df.index, resolution=args.resolution, crs=28992)
    minx, miny, maxx, maxy = ext
    pad = (maxx - minx) * 0.03
    extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    stamp = max(1, 11 - args.resolution)  # res10=1, res9=2, res8=3

    # Plot 1: Variance contribution
    logger.info("[1/4] Per-modality variance contribution...")
    plot_variance_contribution(emb_df, groups, out_dir, args.resolution)

    # Plot 2: PCA spatial maps
    logger.info("[2/4] PCA top-3 spatial maps...")
    plot_pca_spatial(
        emb_df, cx, cy, extent, boundary_gdf,
        out_dir, args.resolution, groups, stamp,
    )

    # Plot 3: Modality correlation matrix
    logger.info("[3/4] Modality correlation matrix...")
    plot_modality_correlation(emb_df, groups, out_dir, args.resolution)

    # Plot 4: Coverage / density map
    logger.info("[4/4] Coverage / density map...")
    plot_modality_coverage(
        emb_df, groups, cx, cy, extent, boundary_gdf,
        out_dir, args.resolution, stamp,
    )

    logger.info("All 4 plots saved to: %s", out_dir)


if __name__ == "__main__":
    main()
