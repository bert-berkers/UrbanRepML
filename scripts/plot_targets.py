#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Comprehensive EDA plots for leefbaarometer and urban taxonomy targets.

Generates ~30 exploratory data analysis plots covering:
  - Section A (plots 1-8): Leefbaarometer spatial maps, distributions, correlations
  - Section B (plots 9-22): Urban Taxonomy spatial maps, class distributions, hierarchy
  - Section C (plots 23-30): Cross-target analysis (inner-join overlap)

Usage:
    python scripts/plot_targets.py --study-area netherlands
"""

import argparse
import colorsys
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPI = 150
LBM_DIMS = ["lbm", "fys", "onv", "soc", "vrz", "won"]
LBM_DIM_NAMES = {
    "lbm": "Overall",
    "fys": "Physical",
    "onv": "Safety",
    "soc": "Social",
    "vrz": "Services",
    "won": "Housing",
}

# Image resolution for rasterized maps (pixels)
RASTER_W = 2000
RASTER_H = 2400


def _shorten_l3(name: str) -> str:
    """Strip trailing ' Fabric' from L3 taxonomy names for compact display."""
    return name.removesuffix(" Fabric")

# ---------------------------------------------------------------------------
# Boundary loading
# ---------------------------------------------------------------------------


def load_boundary(paths: StudyAreaPaths) -> gpd.GeoDataFrame:
    """Load Netherlands boundary, filter to European NL, reproject to 28992."""
    from shapely import get_geometry, get_num_geometries

    boundary_path = paths.area_gdf_file()
    boundary_gdf = gpd.read_file(boundary_path)

    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    boundary_gdf = boundary_gdf.to_crs(epsg=28992)

    # Filter to largest part (European Netherlands) as safety check
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
# Rasterization
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
    colormap = plt.get_cmap(cmap)
    rgb = colormap(norm(val_m))[:, :3].astype(np.float32)

    image = np.ones((height, width, 4), dtype=np.float32)  # white bg
    image[py, px, :3] = rgb
    image[py, px, 3] = 1.0
    return image


def rasterize_categorical(
    cx: np.ndarray,
    cy: np.ndarray,
    labels: np.ndarray,
    extent: tuple,
    color_map: dict,
    width: int = RASTER_W,
    height: int = RASTER_H,
) -> np.ndarray:
    """Rasterize categorical labels to an RGBA image.

    Args:
        cx, cy: centroid coordinates in EPSG:28992.
        labels: integer or string array of class labels.
        extent: (minx, miny, maxx, maxy) in EPSG:28992.
        color_map: dict mapping label -> (r, g, b) tuple in [0, 1].
        width, height: output image dimensions.

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

    # Build RGB for each pixel
    rgb = np.ones((len(lab_m), 3), dtype=np.float32) * 0.8  # fallback gray
    for label, color in color_map.items():
        idx = lab_m == label
        rgb[idx] = color[:3]

    image = np.ones((height, width, 4), dtype=np.float32)
    image[py, px, :3] = rgb
    image[py, px, 3] = 1.0
    return image


def rasterize_labels_to_grid(
    cx: np.ndarray,
    cy: np.ndarray,
    labels: np.ndarray,
    extent: tuple,
    width: int = RASTER_W,
    height: int = RASTER_H,
) -> np.ndarray:
    """Rasterize integer labels to a 2D int array (for edge detection).

    Returns:
        (height, width) int array; -1 means no data.
    """
    minx, miny, maxx, maxy = extent
    mask = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    cx_m, cy_m, lab_m = cx[mask], cy[mask], labels[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    grid = np.full((height, width), -1, dtype=np.int32)
    grid[py, px] = lab_m.astype(np.int32)
    return grid


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------


def _clean_map_axes(ax):
    """Remove ticks and labels for a clean map look."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_spatial_map(
    ax,
    image: np.ndarray,
    extent: tuple,
    boundary_gdf: gpd.GeoDataFrame,
    title: str = "",
):
    """Render a rasterized image on an axes with boundary underlay."""
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
        ax.set_title(title, fontsize=14)


def _add_colorbar(fig, ax, cmap, vmin, vmax, label=""):
    """Add a vertical colorbar to the right of an axes."""
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    return cbar


# ---------------------------------------------------------------------------
# L3-parent-grouped colormap for L6/L7
# ---------------------------------------------------------------------------


def build_l3_grouped_palette(tax: pd.DataFrame, level: int) -> dict:
    """Build a colormap for L6/L7 where subclasses are grouped by L3 parent.

    Each L3 ancestor gets a base hue from tab10. Within each L3 group,
    subclasses vary in lightness so they are distinguishable but grouped.

    Returns:
        dict mapping int class label -> (r, g, b) tuple in [0, 1].
    """
    tab10 = plt.get_cmap("tab10")
    l3_classes = sorted(tax["type_level3"].unique())
    l3_hues = {}
    for i, l3 in enumerate(l3_classes):
        rgb = tab10(i % 10)[:3]
        h, l, s = colorsys.rgb_to_hls(*rgb)
        l3_hues[l3] = (h, s)

    # For each subclass at the target level, find its L3 parent
    sub_to_l3 = (
        tax[["type_level3", f"type_level{level}"]]
        .drop_duplicates()
        .set_index(f"type_level{level}")["type_level3"]
        .to_dict()
    )

    # Group subclasses by L3 parent
    from collections import defaultdict
    groups = defaultdict(list)
    for sub, l3 in sub_to_l3.items():
        groups[l3].append(sub)
    for l3 in groups:
        groups[l3].sort()

    palette = {}
    for l3, subs in groups.items():
        h, s = l3_hues.get(l3, (0.0, 0.5))
        n = len(subs)
        for j, sub in enumerate(subs):
            # Vary lightness from 0.3 to 0.7
            lightness = 0.3 + 0.4 * (j / max(n - 1, 1))
            r, g, b = colorsys.hls_to_rgb(h, lightness, s)
            palette[sub] = (r, g, b)

    return palette


def build_simple_palette(classes, cmap_name="tab10"):
    """Build a simple palette mapping int class -> (r, g, b)."""
    cmap = plt.get_cmap(cmap_name)
    n = max(len(classes), 1)
    return {
        cls: cmap(i / max(n - 1, 1) if n > 1 else 0.0)[:3]
        for i, cls in enumerate(sorted(classes))
    }


# ---------------------------------------------------------------------------
# Section A: Leefbaarometer plots
# ---------------------------------------------------------------------------


def plot_a1_spatial_all_dimensions(lbm, cx_lbm, cy_lbm, extent, boundary, out_dir):
    """2x3 grid of all 6 dims on Netherlands map, viridis, percentile-clipped.

    LBM (Overall) uses its own percentile normalization.  The 5 component
    dimensions (fys, onv, soc, vrz, won) share a common color scale computed
    from pooled percentiles so that gradient strengths are visually comparable.
    """
    # Shared color range across the 5 component dimensions
    component_dims = [d for d in LBM_DIMS if d != "lbm"]
    pooled = np.concatenate([lbm[d].values for d in component_dims])
    comp_v2, comp_v98 = np.nanpercentile(pooled, [2, 98])

    fig, axes = plt.subplots(2, 3, figsize=(18, 22), dpi=DPI)
    fig.set_facecolor("white")
    for i, dim in enumerate(LBM_DIMS):
        ax = axes[i // 3, i % 3]
        vals = lbm[dim].values
        if dim == "lbm":
            v2, v98 = np.nanpercentile(vals, [2, 98])
        else:
            v2, v98 = comp_v2, comp_v98
        img = rasterize_continuous(cx_lbm, cy_lbm, vals, extent,
                                   cmap="viridis", vmin=v2, vmax=v98)
        plot_spatial_map(ax, img, extent, boundary,
                         title=f"{dim.upper()} ({LBM_DIM_NAMES[dim]})")
        _add_colorbar(fig, ax, "viridis", v2, v98)
        # Annotate the normalization range used
        scale_label = "own scale" if dim == "lbm" else "shared scale"
        ax.text(
            0.01, 0.01, f"[{v2:.2f}, {v98:.2f}] ({scale_label})",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    fig.suptitle("Leefbaarometer: All Dimensions", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "spatial_all_dimensions.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_a2_spatial_lbm_detail(lbm, cx_lbm, cy_lbm, extent, boundary, out_dir):
    """Single large map of lbm with RdBu_r diverging colormap centered at median."""
    fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
    fig.set_facecolor("white")
    vals = lbm["lbm"].values
    median = np.nanmedian(vals)
    v2, v98 = np.nanpercentile(vals, [2, 98])
    # Symmetric around median
    max_dev = max(abs(v98 - median), abs(median - v2))
    vmin_sym, vmax_sym = median - max_dev, median + max_dev
    img = rasterize_continuous(cx_lbm, cy_lbm, vals, extent,
                               cmap="RdBu_r", vmin=vmin_sym, vmax=vmax_sym)
    plot_spatial_map(ax, img, extent, boundary, title="LBM Overall Liveability")
    _add_colorbar(fig, ax, "RdBu_r", vmin_sym, vmax_sym, label="LBM score")
    path = out_dir / "spatial_lbm_detail.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_a3_spatial_vrz_vs_soc(lbm, cx_lbm, cy_lbm, extent, boundary, out_dir):
    """1x2 side-by-side: vrz and soc spatial maps with shared color scale."""
    # Shared vmin/vmax so the two component maps are visually comparable
    pooled = np.concatenate([lbm["vrz"].values, lbm["soc"].values])
    v2, v98 = np.nanpercentile(pooled, [2, 98])

    fig, axes = plt.subplots(1, 2, figsize=(20, 12), dpi=DPI)
    fig.set_facecolor("white")
    for i, dim in enumerate(["vrz", "soc"]):
        ax = axes[i]
        vals = lbm[dim].values
        img = rasterize_continuous(cx_lbm, cy_lbm, vals, extent,
                                   cmap="viridis", vmin=v2, vmax=v98)
        plot_spatial_map(ax, img, extent, boundary,
                         title=f"{dim.upper()} ({LBM_DIM_NAMES[dim]})")
        _add_colorbar(fig, ax, "viridis", v2, v98)
        ax.text(
            0.01, 0.01, f"[{v2:.2f}, {v98:.2f}] (shared scale)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    corr = lbm["vrz"].corr(lbm["soc"])
    fig.suptitle(f"Services vs Social (Pearson r = {corr:.3f})", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "spatial_vrz_vs_soc.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_a4_distributions(lbm, out_dir):
    """2x3 faceted histograms with KDE for all 6 dims."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=DPI)
    fig.set_facecolor("white")
    for i, dim in enumerate(LBM_DIMS):
        ax = axes[i // 3, i % 3]
        vals = lbm[dim].dropna().values
        ax.hist(vals, bins=100, density=True, alpha=0.6, color="steelblue",
                edgecolor="none")
        # KDE overlay
        try:
            kde = sp_stats.gaussian_kde(vals)
            x = np.linspace(vals.min(), vals.max(), 300)
            ax.plot(x, kde(x), color="darkred", linewidth=1.5)
        except Exception:
            pass
        ax.set_title(f"{dim.upper()} ({LBM_DIM_NAMES[dim]})", fontsize=14)
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.tick_params(labelsize=10)
        # Add statistics text
        ax.text(0.95, 0.95, f"n={len(vals):,}\nmean={vals.mean():.3f}\nstd={vals.std():.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    fig.suptitle("Leefbaarometer: Score Distributions", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "distributions.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_a5_correlation_matrix(lbm, out_dir):
    """Annotated heatmap of pairwise correlations (6x6 matrix)."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI)
    fig.set_facecolor("white")
    corr = lbm[LBM_DIMS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                mask=mask, square=True, ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1,
                xticklabels=[f"{d} ({LBM_DIM_NAMES[d]})" for d in LBM_DIMS],
                yticklabels=[f"{d} ({LBM_DIM_NAMES[d]})" for d in LBM_DIMS])
    ax.set_title("Leefbaarometer: Pairwise Correlations", fontsize=14)
    ax.tick_params(labelsize=10)
    path = out_dir / "correlation_matrix.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_a6_pairplot(lbm, out_dir):
    """Seaborn pairplot of 5K random hexagons across all 6 dims."""
    sample = lbm[LBM_DIMS].dropna().sample(n=5000, random_state=42)
    g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.15, "s": 3},
                     diag_kws={"linewidth": 1.5})
    g.figure.set_facecolor("white")
    g.figure.suptitle("Leefbaarometer: Pairplot (5K sample)", fontsize=14, y=1.01)
    path = out_dir / "pairplot_subsample.png"
    g.figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(g.figure)
    return path


def plot_a7_boxplots(lbm, out_dir):
    """Boxplots of all 6 dims on shared y-axis."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    fig.set_facecolor("white")
    data = lbm[LBM_DIMS].dropna()
    melted = data.melt(var_name="Dimension", value_name="Score")
    melted["Dimension"] = melted["Dimension"].map(
        lambda d: f"{d.upper()}\n({LBM_DIM_NAMES.get(d, d)})"
    )
    sns.boxplot(data=melted, x="Dimension", y="Score", hue="Dimension",
                ax=ax, palette="Set2", fliersize=0.5, linewidth=0.8,
                legend=False)
    ax.set_title("Leefbaarometer: Dimension Score Ranges", fontsize=14)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.tick_params(labelsize=10)
    path = out_dir / "dimension_profiles_boxplot.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_a8_spatial_extremes(lbm, cx_lbm, cy_lbm, extent, boundary, out_dir):
    """Map: bottom 5% of lbm in red, top 5% in green, rest transparent."""
    fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
    fig.set_facecolor("white")
    vals = lbm["lbm"].values
    p5 = np.nanpercentile(vals, 5)
    p95 = np.nanpercentile(vals, 95)

    minx, miny, maxx, maxy = extent
    finite_mask = np.isfinite(vals)
    in_bounds = (
        (cx_lbm >= minx) & (cx_lbm <= maxx) &
        (cy_lbm >= miny) & (cy_lbm <= maxy) & finite_mask
    )
    cx_m, cy_m, val_m = cx_lbm[in_bounds], cy_lbm[in_bounds], vals[in_bounds]

    px = ((cx_m - minx) / (maxx - minx) * (RASTER_W - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (RASTER_H - 1)).astype(int)
    np.clip(px, 0, RASTER_W - 1, out=px)
    np.clip(py, 0, RASTER_H - 1, out=py)

    # Transparent RGBA image
    image = np.zeros((RASTER_H, RASTER_W, 4), dtype=np.float32)

    # Bottom 5%: red
    bottom = val_m <= p5
    image[py[bottom], px[bottom]] = [0.85, 0.15, 0.15, 1.0]

    # Top 5%: green
    top = val_m >= p95
    image[py[top], px[top]] = [0.15, 0.65, 0.15, 1.0]

    boundary.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)
    ax.imshow(
        image, extent=[minx, maxx, miny, maxy],
        origin="lower", aspect="equal", interpolation="nearest", zorder=2,
    )
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    _clean_map_axes(ax)

    # Legend
    red_patch = mpatches.Patch(color=(0.85, 0.15, 0.15), label=f"Bottom 5% (LBM <= {p5:.3f})")
    green_patch = mpatches.Patch(color=(0.15, 0.65, 0.15), label=f"Top 5% (LBM >= {p95:.3f})")
    ax.legend(handles=[red_patch, green_patch], loc="lower left", fontsize=11,
              frameon=True, facecolor="white", edgecolor="gray")
    ax.set_title("LBM Extremes: Bottom 5% (red) vs Top 5% (green)", fontsize=14)
    path = out_dir / "spatial_extremes.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Section B: Urban Taxonomy plots
# ---------------------------------------------------------------------------


def plot_b9_to_b15_spatial_levels(tax, cx_tax, cy_tax, extent, boundary, out_dir):
    """One spatial map per hierarchy level (7 maps total).

    L1-L5: simple tab10/tab20 categorical palette.
    L6-L7: L3-parent-grouped palette (hue by L3 ancestor, lightness varies).
    L1-L3: include legend with named classes.
    L4+: subtitle with N classes instead of legend.
    """
    paths_out = []
    for level in range(1, 8):
        col = f"type_level{level}"
        classes = sorted(tax[col].unique())
        n_classes = len(classes)

        # Build palette
        if level <= 5:
            cmap_name = "tab10" if n_classes <= 10 else "tab20"
            palette = build_simple_palette(classes, cmap_name)
        else:
            palette = build_l3_grouped_palette(tax, level)

        # Rasterize
        labels = tax[col].values
        img = rasterize_categorical(cx_tax, cy_tax, labels, extent, palette)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
        fig.set_facecolor("white")
        plot_spatial_map(ax, img, extent, boundary)

        if level <= 3:
            name_col = f"name_level{level}"
            if name_col in tax.columns:
                label_map = (
                    tax[[col, name_col]].drop_duplicates()
                    .set_index(col)[name_col].to_dict()
                )
                handles = [
                    mpatches.Patch(color=palette[cls],
                                   label=f"{cls}: {label_map.get(cls, '?')}")
                    for cls in sorted(palette.keys()) if cls in set(classes)
                ]
                ax.legend(handles=handles, loc="lower left", fontsize=10,
                          frameon=True, facecolor="white", edgecolor="gray",
                          title=f"Level {level}", title_fontsize=11)
            ax.set_title(f"Urban Taxonomy: Level {level} ({n_classes} classes)",
                         fontsize=14)
        else:
            ax.set_title(f"Urban Taxonomy: Level {level} ({n_classes} classes)",
                         fontsize=14)

        fname = f"spatial_level{level}.png"
        path = out_dir / fname
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths_out.append(path)

    return paths_out


def plot_b16_named_levels(tax, cx_tax, cy_tax, extent, boundary, out_dir):
    """1x3 grid: L1, L2, L3 side by side with legends."""
    fig, axes = plt.subplots(1, 3, figsize=(30, 12), dpi=DPI)
    fig.set_facecolor("white")

    for i, level in enumerate([1, 2, 3]):
        ax = axes[i]
        col = f"type_level{level}"
        name_col = f"name_level{level}"
        classes = sorted(tax[col].unique())
        palette = build_simple_palette(classes, "tab10")
        labels = tax[col].values
        img = rasterize_categorical(cx_tax, cy_tax, labels, extent, palette)
        plot_spatial_map(ax, img, extent, boundary,
                         title=f"Level {level} ({len(classes)} classes)")

        if name_col in tax.columns:
            label_map = (
                tax[[col, name_col]].drop_duplicates()
                .set_index(col)[name_col].to_dict()
            )
            handles = [
                mpatches.Patch(color=palette[cls],
                               label=f"{cls}: {label_map.get(cls, '?')}")
                for cls in sorted(palette.keys()) if cls in set(classes)
            ]
            ax.legend(handles=handles, loc="lower left", fontsize=9,
                      frameon=True, facecolor="white", edgecolor="gray")

    fig.suptitle("Urban Taxonomy: Named Levels (L1, L2, L3)", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "spatial_named_levels.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_b17_class_distributions(tax, out_dir):
    """7-panel (4x2 grid) bar charts showing class frequency for L1-L7."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), dpi=DPI)
    fig.set_facecolor("white")

    for i, level in enumerate(range(1, 8)):
        ax = axes[i // 2, i % 2]
        col = f"type_level{level}"
        counts = tax[col].value_counts().sort_index()
        ax.bar(range(len(counts)), counts.values, color="steelblue",
               edgecolor="none")
        ax.set_title(f"Level {level} ({len(counts)} classes)", fontsize=14)
        ax.set_xlabel("Class label", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(labelsize=10)
        if level >= 4:
            ax.set_yscale("log")
            ax.set_ylabel("Count (log)", fontsize=12)
        if len(counts) <= 20:
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, fontsize=8, rotation=45)

    # Hide the 8th subplot
    axes[3, 1].set_visible(False)

    fig.suptitle("Urban Taxonomy: Class Frequency per Level", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "class_distributions_all_levels.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_b18_hierarchy_entropy(tax, out_dir):
    """Dual-axis line plot: Shannon entropy (bits) on left, n_classes on right."""
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=DPI)
    fig.set_facecolor("white")

    levels = list(range(1, 8))
    entropies = []
    n_classes_list = []
    for level in levels:
        col = f"type_level{level}"
        counts = tax[col].value_counts()
        n_classes_list.append(len(counts))
        probs = counts.values / counts.values.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        entropies.append(entropy)

    color1 = "steelblue"
    ax1.plot(levels, entropies, "o-", color=color1, linewidth=2, markersize=7,
             label="Shannon entropy")
    ax1.set_xlabel("Hierarchy Level", fontsize=12)
    ax1.set_ylabel("Shannon Entropy (bits)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1, labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.set_xticks(levels)

    ax2 = ax1.twinx()
    color2 = "darkorange"
    ax2.plot(levels, n_classes_list, "s--", color=color2, linewidth=2,
             markersize=7, label="n_classes")
    ax2.set_ylabel("Number of Classes", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=10)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)

    ax1.set_title("Hierarchy Entropy and Class Count", fontsize=14)
    fig.tight_layout()
    path = out_dir / "hierarchy_entropy.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_b19_class_imbalance(tax, out_dir):
    """Bar chart: max_count / min_count per level. Log scale y-axis."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    fig.set_facecolor("white")

    levels = list(range(1, 8))
    ratios = []
    for level in levels:
        col = f"type_level{level}"
        counts = tax[col].value_counts()
        ratios.append(counts.max() / max(counts.min(), 1))

    ax.bar(levels, ratios, color="salmon", edgecolor="darkred", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Hierarchy Level", fontsize=12)
    ax.set_ylabel("Imbalance Ratio (max/min count, log)", fontsize=12)
    ax.set_title("Class Imbalance per Level", fontsize=14)
    ax.set_xticks(levels)
    ax.tick_params(labelsize=10)

    # Annotate values
    for i, (lv, r) in enumerate(zip(levels, ratios)):
        ax.text(lv, r * 1.3, f"{r:.0f}x", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = out_dir / "class_imbalance_ratio.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_b20_hierarchy_tree(tax, out_dir):
    """Tree diagram showing L1->L2->L3 hierarchy with class counts."""
    # Build hierarchy: L1 -> L2 -> L3 with names and counts
    tree_data = (
        tax[["type_level1", "name_level1", "type_level2", "name_level2",
             "type_level3", "name_level3"]]
        .drop_duplicates()
        .sort_values(["type_level1", "type_level2", "type_level3"])
    )

    # Count hexagons per class
    l1_counts = tax["type_level1"].value_counts().to_dict()
    l2_counts = tax["type_level2"].value_counts().to_dict()
    l3_counts = tax["type_level3"].value_counts().to_dict()

    fig, ax = plt.subplots(figsize=(16, 10), dpi=DPI)
    fig.set_facecolor("white")
    ax.set_xlim(-0.5, 3.5)

    # Collect unique nodes at each level
    l1_nodes = sorted(tree_data["type_level1"].unique())
    l2_nodes = sorted(tree_data["type_level2"].unique())
    l3_nodes = sorted(tree_data["type_level3"].unique())

    total_h = max(len(l3_nodes) + 2, 12)
    ax.set_ylim(-1, total_h + 1)

    # Position nodes vertically
    def spread(nodes, total):
        n = len(nodes)
        if n == 1:
            return {nodes[0]: total / 2}
        return {nd: total * i / (n - 1) for i, nd in enumerate(nodes)}

    l1_y = spread(l1_nodes, total_h)
    l2_y = spread(l2_nodes, total_h)
    l3_y = spread(l3_nodes, total_h)

    # Draw nodes and edges
    node_style = dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", lw=0.8)

    # L1 -> L2 edges
    l1_to_l2 = tree_data[["type_level1", "type_level2"]].drop_duplicates()
    for _, row in l1_to_l2.iterrows():
        y1 = l1_y[row["type_level1"]]
        y2 = l2_y[row["type_level2"]]
        ax.plot([0.5, 1.3], [y1, y2], color="gray", linewidth=0.7, zorder=1)

    # L2 -> L3 edges
    l2_to_l3 = tree_data[["type_level2", "type_level3"]].drop_duplicates()
    for _, row in l2_to_l3.iterrows():
        y2 = l2_y[row["type_level2"]]
        y3 = l3_y[row["type_level3"]]
        ax.plot([1.7, 2.5], [y2, y3], color="gray", linewidth=0.7, zorder=1)

    # Draw L1 nodes
    for cls in l1_nodes:
        name_rows = tree_data[tree_data["type_level1"] == cls]
        name = name_rows["name_level1"].iloc[0] if "name_level1" in name_rows else str(cls)
        count = l1_counts.get(cls, 0)
        label = f"{name}\n({count:,})"
        ax.text(0.3, l1_y[cls], label, ha="center", va="center", fontsize=9,
                bbox=node_style, zorder=2)

    # Draw L2 nodes
    for cls in l2_nodes:
        name_rows = tree_data[tree_data["type_level2"] == cls]
        name = name_rows["name_level2"].iloc[0] if "name_level2" in name_rows else str(cls)
        count = l2_counts.get(cls, 0)
        label = f"{name}\n({count:,})"
        ax.text(1.5, l2_y[cls], label, ha="center", va="center", fontsize=8,
                bbox=node_style, zorder=2)

    # Draw L3 nodes
    for cls in l3_nodes:
        name_rows = tree_data[tree_data["type_level3"] == cls]
        name = name_rows["name_level3"].iloc[0] if "name_level3" in name_rows else str(cls)
        count = l3_counts.get(cls, 0)
        label = f"{name}\n({count:,})"
        ax.text(2.7, l3_y[cls], label, ha="center", va="center", fontsize=7,
                bbox=node_style, zorder=2)

    # Level labels at top
    for x, lbl in [(0.3, "Level 1"), (1.5, "Level 2"), (2.7, "Level 3")]:
        ax.text(x, total_h + 0.5, lbl, ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_title("Urban Taxonomy: Hierarchy Tree (L1 -> L2 -> L3)", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    path = out_dir / "hierarchy_tree.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_b21_boundary_confidence(tax, out_dir):
    """Histogram of n_morphotopes values with percentage annotations."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    fig.set_facecolor("white")

    vals = tax["n_morphotopes"].values
    max_val = min(int(vals.max()), 10)
    bins = np.arange(0.5, max_val + 1.5, 1)
    counts, edges, patches = ax.hist(vals.clip(max=max_val), bins=bins,
                                      color="steelblue", edgecolor="white",
                                      linewidth=0.5)
    total = len(vals)
    for count, patch in zip(counts, patches):
        pct = count / total * 100
        ax.text(patch.get_x() + patch.get_width() / 2, count + total * 0.005,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("n_morphotopes (clipped at 10)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Boundary Confidence: Number of Morphotopes per Hexagon",
                 fontsize=14)
    ax.set_xticks(range(1, max_val + 1))
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    path = out_dir / "boundary_confidence.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_b22_spatial_boundary(tax, cx_tax, cy_tax, extent, boundary, out_dir):
    """Spatial map coloring by n_morphotopes (YlOrRd sequential)."""
    fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
    fig.set_facecolor("white")
    vals = tax["n_morphotopes"].values.astype(float)
    img = rasterize_continuous(cx_tax, cy_tax, vals, extent,
                               cmap="YlOrRd", vmin=1, vmax=5)
    plot_spatial_map(ax, img, extent, boundary,
                     title="Boundary Hexagons: n_morphotopes")
    _add_colorbar(fig, ax, "YlOrRd", 1, 5, label="n_morphotopes")
    path = out_dir / "spatial_boundary_hexagons.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Section C: Cross-target plots
# ---------------------------------------------------------------------------


def plot_c23_lbm_by_l1(merged, out_dir):
    """Violin plot: lbm distribution per L1 class (2 violins)."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)
    fig.set_facecolor("white")

    if "name_level1" in merged.columns:
        merged["L1_label"] = merged["name_level1"]
    else:
        merged["L1_label"] = merged["type_level1"].astype(str)

    sns.violinplot(data=merged, x="L1_label", y="lbm", hue="L1_label",
                   ax=ax, palette="Set2", inner="quartile", linewidth=0.8,
                   legend=False)
    ax.set_title("LBM Distribution by Taxonomy Level 1", fontsize=14)
    ax.set_xlabel("Level 1 Class", fontsize=12)
    ax.set_ylabel("LBM Score", fontsize=12)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    path = out_dir / "lbm_by_taxonomy_l1.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_c24_lbm_by_l3(merged, out_dir):
    """Violin plot: lbm per L3 named class (8 violins), sorted by median lbm."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
    fig.set_facecolor("white")

    if "name_level3" in merged.columns:
        merged["L3_label"] = merged["name_level3"].map(_shorten_l3)
    else:
        merged["L3_label"] = merged["type_level3"].astype(str)

    # Sort by median lbm
    order = (
        merged.groupby("L3_label")["lbm"].median()
        .sort_values().index.tolist()
    )

    sns.violinplot(data=merged, x="L3_label", y="lbm", hue="L3_label",
                   ax=ax, palette="Set2", inner="quartile", linewidth=0.8,
                   order=order, legend=False)
    ax.set_title("LBM Distribution by Taxonomy Level 3 (sorted by median)", fontsize=14)
    ax.set_xlabel("Level 3 Class", fontsize=12)
    ax.set_ylabel("LBM Score", fontsize=12)
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = out_dir / "lbm_by_taxonomy_l3.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _row_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize each row of a DataFrame to [-1, 1] independently.

    This makes gradient strength visually comparable across rows with
    very different absolute scales (e.g. lbm ~4.1 vs component dims ~0.05).
    Annotations should still use the original raw values.

    If a row has zero range (all identical values), it is left at 0.
    """
    normed = df.copy()
    for row in normed.index:
        row_min = df.loc[row].min()
        row_max = df.loc[row].max()
        rng = row_max - row_min
        if rng > 0:
            normed.loc[row] = 2.0 * (df.loc[row] - row_min) / rng - 1.0
        else:
            normed.loc[row] = 0.0
    return normed


def plot_c25_dims_by_l3(merged, out_dir):
    """Heatmap: rows=6 lbm dims, cols=8 L3 types. Values=mean, annotated.

    Color gradient is per-row normalized so that lbm (absolute scale ~4.1)
    and the 5 component dims (scale ~0.05-0.2) are visually comparable.
    Annotations show raw mean values.
    """
    fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
    fig.set_facecolor("white")

    if "name_level3" in merged.columns:
        merged["L3_label"] = merged["name_level3"].map(_shorten_l3)
    else:
        merged["L3_label"] = merged["type_level3"].astype(str)

    means = merged.groupby("L3_label")[LBM_DIMS].mean()
    # Transpose so rows=dims, cols=taxonomy classes -- for per-row normalization
    means_T = means.T
    normed_T = _row_normalize(means_T)

    sns.heatmap(normed_T, annot=means_T, fmt=".3f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "row-normalized"},
                yticklabels=[f"{d} ({LBM_DIM_NAMES[d]})" for d in LBM_DIMS])
    ax.set_title(
        "Mean LBM Dimensions by Taxonomy L3 Class\n"
        "(colors: per-row normalized; annotations: raw means)",
        fontsize=13,
    )
    ax.set_xlabel("Level 3 Class", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = out_dir / "all_dims_by_taxonomy_l3.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_c26_dims_by_l1(merged, out_dir):
    """Heatmap: rows=6 lbm dims, cols=L1 classes. Values=mean, annotated.

    Color gradient is per-row normalized so that lbm (absolute scale ~4.1)
    and the 5 component dims (scale ~0.05-0.2) are visually comparable.
    Annotations show raw mean values.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)
    fig.set_facecolor("white")

    if "name_level1" in merged.columns:
        merged["L1_label"] = merged["name_level1"]
    else:
        merged["L1_label"] = merged["type_level1"].astype(str)

    means = merged.groupby("L1_label")[LBM_DIMS].mean()
    means_T = means.T
    normed_T = _row_normalize(means_T)

    sns.heatmap(normed_T, annot=means_T, fmt=".3f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "row-normalized"},
                yticklabels=[f"{d} ({LBM_DIM_NAMES[d]})" for d in LBM_DIMS])
    ax.set_title(
        "Mean LBM Dimensions by Taxonomy L1\n"
        "(colors: per-row normalized; annotations: raw means)",
        fontsize=13,
    )
    ax.set_xlabel("Level 1 Class", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    path = out_dir / "all_dims_by_taxonomy_l1.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_c27_extreme_composition(merged, out_dir):
    """Stacked bar: two bars (bottom 5% and top 5% of lbm),
    segments colored by L3 class. Percentage labels."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    fig.set_facecolor("white")

    p5 = merged["lbm"].quantile(0.05)
    p95 = merged["lbm"].quantile(0.95)

    bottom_5 = merged[merged["lbm"] <= p5]
    top_5 = merged[merged["lbm"] >= p95]

    if "name_level3" in merged.columns:
        class_col = "name_level3"
    else:
        class_col = "type_level3"

    all_classes = sorted(merged[class_col].unique())
    palette = build_simple_palette(range(len(all_classes)), "tab10")
    class_colors = {cls: palette[i] for i, cls in enumerate(all_classes)}

    bar_data = {}
    for label, subset in [("Bottom 5%", bottom_5), ("Top 5%", top_5)]:
        counts = subset[class_col].value_counts()
        total = counts.sum()
        pcts = {cls: counts.get(cls, 0) / total * 100 for cls in all_classes}
        bar_data[label] = pcts

    x = np.arange(2)
    bar_labels = ["Bottom 5%", "Top 5%"]
    bottoms = np.zeros(2)

    for cls in all_classes:
        heights = [bar_data[bl].get(cls, 0) for bl in bar_labels]
        if sum(heights) < 0.1:
            continue
        bars = ax.bar(x, heights, bottom=bottoms, color=class_colors[cls],
                      edgecolor="white", linewidth=0.3,
                      label=_shorten_l3(str(cls)))
        # Add percentage labels for segments > 5%
        for j, h in enumerate(heights):
            if h > 5:
                ax.text(x[j], bottoms[j] + h / 2, f"{h:.0f}%",
                        ha="center", va="center", fontsize=8, fontweight="bold")
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("L3 Class Composition of LBM Extremes", fontsize=14)
    ax.tick_params(labelsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8,
              title="L3 Class", title_fontsize=9)
    fig.tight_layout()
    path = out_dir / "extreme_composition.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_c28_weight_by_taxonomy(merged, out_dir):
    """Boxplot: weight_sum per L3 class."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
    fig.set_facecolor("white")

    if "name_level3" in merged.columns:
        merged["L3_label"] = merged["name_level3"].map(_shorten_l3)
    else:
        merged["L3_label"] = merged["type_level3"].astype(str)

    order = sorted(merged["L3_label"].unique())

    if "weight_sum" in merged.columns:
        sns.boxplot(data=merged, x="L3_label", y="weight_sum", hue="L3_label",
                    ax=ax, palette="Set2", fliersize=0.3, linewidth=0.8,
                    order=order, legend=False)
        if merged["weight_sum"].max() / max(merged["weight_sum"].min(), 1e-9) > 100:
            ax.set_yscale("log")
            ax.set_ylabel("weight_sum (log scale)", fontsize=12)
        else:
            ax.set_ylabel("weight_sum", fontsize=12)
    else:
        ax.text(0.5, 0.5, "weight_sum column not in merged data",
                transform=ax.transAxes, ha="center")

    ax.set_title("weight_sum Distribution by L3 Class", fontsize=14)
    ax.set_xlabel("Level 3 Class", fontsize=12)
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = out_dir / "weight_sum_by_taxonomy.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_c29_lbm_with_taxonomy_overlay(
    merged, cx_merged, cy_merged, extent, boundary, out_dir
):
    """LBM rasterized map (RdBu_r) with L1 boundary contours overlaid."""
    fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
    fig.set_facecolor("white")

    # Base: LBM continuous map
    vals = merged["lbm"].values
    median = np.nanmedian(vals)
    v2, v98 = np.nanpercentile(vals, [2, 98])
    max_dev = max(abs(v98 - median), abs(median - v2))
    vmin_sym, vmax_sym = median - max_dev, median + max_dev

    img = rasterize_continuous(cx_merged, cy_merged, vals, extent,
                               cmap="RdBu_r", vmin=vmin_sym, vmax=vmax_sym)

    # Overlay: L1 edge detection
    l1_labels = merged["type_level1"].values
    l1_grid = rasterize_labels_to_grid(cx_merged, cy_merged, l1_labels, extent)

    # Detect edges: where adjacent pixels differ in L1 class
    edge_mask = np.zeros_like(l1_grid, dtype=bool)
    # Horizontal edges
    h_diff = l1_grid[:, :-1] != l1_grid[:, 1:]
    h_valid = (l1_grid[:, :-1] >= 0) & (l1_grid[:, 1:] >= 0)
    edge_mask[:, :-1] |= h_diff & h_valid
    edge_mask[:, 1:] |= h_diff & h_valid
    # Vertical edges
    v_diff = l1_grid[:-1, :] != l1_grid[1:, :]
    v_valid = (l1_grid[:-1, :] >= 0) & (l1_grid[1:, :] >= 0)
    edge_mask[:-1, :] |= v_diff & v_valid
    edge_mask[1:, :] |= v_diff & v_valid

    # Draw black edge pixels on the image
    img[edge_mask, :3] = 0.0
    img[edge_mask, 3] = 1.0

    plot_spatial_map(ax, img, extent, boundary,
                     title="LBM with Taxonomy L1 Boundaries")
    _add_colorbar(fig, ax, "RdBu_r", vmin_sym, vmax_sym, label="LBM score")
    path = out_dir / "spatial_lbm_with_taxonomy_overlay.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_c30_coverage_bar(lbm, tax, out_dir):
    """Grouped bar chart: leefbaarometer-only, overlap, taxonomy-only counts."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    fig.set_facecolor("white")

    lbm_ids = set(lbm.index)
    tax_ids = set(tax.index)
    overlap = lbm_ids & tax_ids
    lbm_only = lbm_ids - tax_ids
    tax_only = tax_ids - lbm_ids
    total = len(lbm_ids | tax_ids)

    labels = ["Leefbaarometer\nonly", "Overlap", "Urban Taxonomy\nonly"]
    counts = [len(lbm_only), len(overlap), len(tax_only)]
    pcts = [c / total * 100 for c in counts]
    colors = ["#e76f51", "#2a9d8f", "#264653"]

    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)

    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Number of Hexagons", fontsize=12)
    ax.set_title("Region Coverage: Leefbaarometer vs Urban Taxonomy", fontsize=14)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    path = out_dir / "coverage_venn.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate ~30 EDA plots for leefbaarometer and urban taxonomy targets."
    )
    parser.add_argument(
        "--study-area", default="netherlands",
        help="Study area name (default: netherlands)"
    )
    args = parser.parse_args()

    paths = StudyAreaPaths(args.study_area)
    db = SpatialDB.for_study_area(args.study_area)

    # Output directories
    lbm_plot_dir = paths.target("leefbaarometer") / "plots"
    tax_plot_dir = paths.target("urban_taxonomy") / "plots"
    lbm_plot_dir.mkdir(parents=True, exist_ok=True)
    tax_plot_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading leefbaarometer data...")
    lbm = pd.read_parquet(paths.target_file("leefbaarometer", 10, 2022))
    print(f"  Loaded {len(lbm):,} rows, columns: {list(lbm.columns)}")

    print("Loading urban taxonomy data...")
    tax = pd.read_parquet(paths.target_file("urban_taxonomy", 10, 2025))
    print(f"  Loaded {len(tax):,} rows, columns: {list(tax.columns)}")

    # ------------------------------------------------------------------
    # Load boundary and compute extent
    # ------------------------------------------------------------------
    print("Loading boundary...")
    boundary = load_boundary(paths)
    extent_raw = boundary.total_bounds  # minx, miny, maxx, maxy
    pad = (extent_raw[2] - extent_raw[0]) * 0.03
    extent = (
        extent_raw[0] - pad,
        extent_raw[1] - pad,
        extent_raw[2] + pad,
        extent_raw[3] + pad,
    )
    print(f"  Extent (EPSG:28992, padded): {extent}")

    # ------------------------------------------------------------------
    # Cache centroids (one call per dataset)
    # ------------------------------------------------------------------
    print("Computing leefbaarometer centroids...")
    cx_lbm, cy_lbm = db.centroids(lbm.index, resolution=10, crs=28992)
    print(f"  Got {len(cx_lbm):,} centroids")

    print("Computing urban taxonomy centroids...")
    cx_tax, cy_tax = db.centroids(tax.index, resolution=10, crs=28992)
    print(f"  Got {len(cx_tax):,} centroids")

    # ------------------------------------------------------------------
    # Cross-target merge (inner join)
    # ------------------------------------------------------------------
    print("Merging datasets (inner join)...")
    merged = lbm.join(tax, how="inner")
    print(f"  Overlap: {len(merged):,} hexagons")

    # Centroids for merged set
    print("Computing merged centroids...")
    cx_merged, cy_merged = db.centroids(merged.index, resolution=10, crs=28992)
    print(f"  Got {len(cx_merged):,} centroids")

    # ------------------------------------------------------------------
    # Generate all 30 plots
    # ------------------------------------------------------------------
    plot_idx = 0

    # Section A: Leefbaarometer
    print("\n=== Section A: Leefbaarometer ===")

    plot_idx += 1
    p = plot_a1_spatial_all_dimensions(lbm, cx_lbm, cy_lbm, extent, boundary, lbm_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_a2_spatial_lbm_detail(lbm, cx_lbm, cy_lbm, extent, boundary, lbm_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_a3_spatial_vrz_vs_soc(lbm, cx_lbm, cy_lbm, extent, boundary, lbm_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_a4_distributions(lbm, lbm_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_a5_correlation_matrix(lbm, lbm_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_a6_pairplot(lbm, lbm_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_a7_boxplots(lbm, lbm_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_a8_spatial_extremes(lbm, cx_lbm, cy_lbm, extent, boundary, lbm_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    # Section B: Urban Taxonomy
    print("\n=== Section B: Urban Taxonomy ===")

    plot_idx_before = plot_idx
    level_paths = plot_b9_to_b15_spatial_levels(tax, cx_tax, cy_tax, extent, boundary, tax_plot_dir)
    for lp in level_paths:
        plot_idx += 1
        print(f"[{plot_idx}/30] Saved {lp.name}")

    plot_idx += 1
    p = plot_b16_named_levels(tax, cx_tax, cy_tax, extent, boundary, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_b17_class_distributions(tax, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_b18_hierarchy_entropy(tax, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_b19_class_imbalance(tax, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_b20_hierarchy_tree(tax, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_b21_boundary_confidence(tax, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_b22_spatial_boundary(tax, cx_tax, cy_tax, extent, boundary, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    # Section C: Cross-target (save to urban_taxonomy/plots/)
    print("\n=== Section C: Cross-Target ===")

    plot_idx += 1
    p = plot_c23_lbm_by_l1(merged, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_c24_lbm_by_l3(merged, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_c25_dims_by_l3(merged, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_c26_dims_by_l1(merged, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_c27_extreme_composition(merged, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_c28_weight_by_taxonomy(merged, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_c29_lbm_with_taxonomy_overlay(merged, cx_merged, cy_merged, extent, boundary, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    plot_idx += 1
    p = plot_c30_coverage_bar(lbm, tax, tax_plot_dir)
    print(f"[{plot_idx}/30] Saved {p.name}")

    print(f"\nDone! Generated {plot_idx} plots.")
    print(f"  Leefbaarometer: {lbm_plot_dir}")
    print(f"  Urban Taxonomy: {tax_plot_dir}")


if __name__ == "__main__":
    main()
