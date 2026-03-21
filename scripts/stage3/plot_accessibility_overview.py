"""Accessibility graph overview maps and histogram distributions.

Generates:
- 6 overview maps (degree + gravity for walk/bike/drive)
- 4 histogram/bar figures (travel time, gravity, degree distributions + road class bars)

Lifetime: durable
Stage: 3 (analysis/visualization)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.visualization import (
    rasterize_continuous,
    rasterize_binary,
    plot_spatial_map,
    load_boundary,
    _add_colorbar,
    RASTER_W,
    RASTER_H,
)
from utils.spatial_db import SpatialDB
from utils.paths import StudyAreaPaths

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

STUDY_AREA = "netherlands"

# Mode configs: (filename, resolution, display_name, color for histograms)
MODES = [
    ("walk_res9", 9, "Walk (res9)", "#1f77b4"),
    ("bike_res8", 8, "Bike (res8)", "#ff7f0e"),
    ("drive_res7", 7, "Drive (res7)", "#2ca02c"),
]


def load_accessibility(paths: StudyAreaPaths, filename: str) -> pd.DataFrame:
    """Load an accessibility parquet file."""
    fp = paths.root / "accessibility" / f"{filename}.parquet"
    logger.info("Loading %s (%s)", filename, fp)
    return pd.read_parquet(fp)


def compute_hex_metrics(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute per-hex degree and mean incoming gravity."""
    degree = df.groupby("origin_hex").size().rename("degree")
    gravity = df.groupby("dest_hex")["gravity_weight"].mean().rename("mean_gravity")
    return degree, gravity


def render_overview_map(
    metric_series: pd.Series,
    resolution: int,
    paths: StudyAreaPaths,
    db: SpatialDB,
    boundary: gpd.GeoDataFrame,
    all_cx: np.ndarray,
    all_cy: np.ndarray,
    extent: tuple,
    cmap: str,
    title: str,
    label: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render a single overview map with grey background + colored foreground."""
    stamp = max(1, 11 - resolution)

    # Grey background: all hexes
    bg_img = rasterize_binary(all_cx, all_cy, extent, color=(0.85, 0.85, 0.85), stamp=stamp)

    # Foreground: hexes with data
    hex_ids = list(metric_series.index)
    fg_cx, fg_cy = db.centroids(hex_ids, resolution, crs=28992)
    values = metric_series.loc[hex_ids].values.astype(np.float64)

    if vmin is None:
        vmin = float(np.nanpercentile(values, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(values, 98))

    fg_img = rasterize_continuous(
        fg_cx, fg_cy, values, extent, cmap=cmap, stamp=stamp, vmin=vmin, vmax=vmax,
    )

    fig, ax = plt.subplots(figsize=(12, 14))
    plot_spatial_map(ax, bg_img, extent, boundary, title=title)
    ax.imshow(
        fg_img,
        extent=[extent[0], extent[2], extent[1], extent[3]],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        zorder=3,
    )
    _add_colorbar(fig, ax, cmap, vmin, vmax, label=label)

    n_data = len(metric_series)
    n_total = len(all_cx)
    ax.annotate(
        f"{n_data:,} / {n_total:,} hexes with data ({100*n_data/n_total:.1f}%)",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=9, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_overview_maps(paths: StudyAreaPaths, out_dir: Path) -> None:
    """Generate 6 overview maps (degree + gravity for each mode)."""
    db = SpatialDB.for_study_area(STUDY_AREA)
    boundary = load_boundary(paths)

    for filename, resolution, display_name, _ in MODES:
        logger.info("=== Overview maps for %s ===", display_name)

        # Load all hexes at this resolution for background
        region_file = paths.region_file(resolution)
        all_hex_ids = list(gpd.read_parquet(region_file).index)
        all_cx, all_cy = db.centroids(all_hex_ids, resolution, crs=28992)
        extent = (all_cx.min(), all_cy.min(), all_cx.max(), all_cy.max())

        # Load data and compute metrics
        df = load_accessibility(paths, filename)
        degree, gravity = compute_hex_metrics(df)

        # Degree map
        render_overview_map(
            metric_series=degree,
            resolution=resolution,
            paths=paths,
            db=db,
            boundary=boundary,
            all_cx=all_cx,
            all_cy=all_cy,
            extent=extent,
            cmap="viridis",
            title=f"{display_name} -- Degree (edges per hex)",
            label="Degree",
            out_path=out_dir / f"{filename}_degree.png",
        )

        # Gravity map
        render_overview_map(
            metric_series=gravity,
            resolution=resolution,
            paths=paths,
            db=db,
            boundary=boundary,
            all_cx=all_cx,
            all_cy=all_cy,
            extent=extent,
            cmap="magma",
            title=f"{display_name} -- Mean incoming gravity weight",
            label="Mean gravity weight",
            out_path=out_dir / f"{filename}_gravity.png",
        )


def plot_histograms(paths: StudyAreaPaths, out_dir: Path) -> None:
    """Generate 4 histogram/bar figures with all modes overlaid."""
    mode_data = {}
    for filename, resolution, display_name, color in MODES:
        df = load_accessibility(paths, filename)
        degree, gravity = compute_hex_metrics(df)
        mode_data[display_name] = {
            "df": df,
            "degree": degree,
            "gravity": gravity,
            "color": color,
        }

    # --- 1. Travel time distribution ---
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, d in mode_data.items():
        tt = d["df"]["travel_time_s"].values
        median_tt = np.median(tt)
        ax.hist(tt, bins=100, alpha=0.5, label=f"{name} (median={median_tt:.0f}s)", color=d["color"])
    ax.set_xlabel("Travel time (seconds)", fontsize=12)
    ax.set_ylabel("Edge count", fontsize=12)
    ax.set_title("Travel Time Distribution by Mode", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_dir / "hist_travel_times.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved hist_travel_times.png")

    # --- 2. Gravity weight distribution (log scale x-axis) ---
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, d in mode_data.items():
        gw = d["df"]["gravity_weight"].values
        gw = gw[gw > 0]  # skip zeros
        median_gw = np.median(gw)
        ax.hist(gw, bins=np.logspace(np.log10(gw.min()), np.log10(gw.max()), 100),
                alpha=0.5, label=f"{name} (median={median_gw:.4f})", color=d["color"])
    ax.set_xscale("log")
    ax.set_xlabel("Gravity weight (log scale)", fontsize=12)
    ax.set_ylabel("Edge count", fontsize=12)
    ax.set_title("Gravity Weight Distribution by Mode", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_dir / "hist_gravity_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved hist_gravity_weights.png")

    # --- 3. Degree distribution ---
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, d in mode_data.items():
        deg = d["degree"].values
        median_deg = np.median(deg)
        ax.hist(deg, bins=100, alpha=0.5,
                label=f"{name} (median={median_deg:.0f}, N={len(deg):,})", color=d["color"])
    ax.set_xlabel("Degree (edges per hex)", fontsize=12)
    ax.set_ylabel("Hex count", fontsize=12)
    ax.set_title("Degree Distribution by Mode", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_dir / "hist_degree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved hist_degree.png")

    # --- 4. Road class bar chart (top 10) ---
    fig, ax = plt.subplots(figsize=(14, 8))
    # Collect all road classes across modes
    all_classes = set()
    class_counts = {}
    for name, d in mode_data.items():
        counts = d["df"]["fastest_road_class"].value_counts()
        class_counts[name] = counts
        all_classes.update(counts.index)

    # Get top 10 by total count
    totals = pd.Series(0.0, index=list(all_classes))
    for counts in class_counts.values():
        for cls, cnt in counts.items():
            totals[cls] += cnt
    top10 = totals.nlargest(10).index.tolist()

    x = np.arange(len(top10))
    bar_width = 0.25
    for i, (name, d) in enumerate(mode_data.items()):
        counts = class_counts[name]
        vals = [counts.get(cls, 0) for cls in top10]
        ax.bar(x + i * bar_width, vals, bar_width, label=name, color=d["color"], alpha=0.85)

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(top10, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Road class", fontsize=12)
    ax.set_ylabel("Edge count", fontsize=12)
    ax.set_title("Edge Count by Road Class (top 10)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out_dir / "bar_road_classes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bar_road_classes.png")


def main():
    parser = argparse.ArgumentParser(description="Accessibility overview maps and histograms")
    parser.add_argument("--study-area", default=STUDY_AREA)
    parser.add_argument("--maps-only", action="store_true", help="Skip histograms")
    parser.add_argument("--histograms-only", action="store_true", help="Skip maps")
    args = parser.parse_args()

    paths = StudyAreaPaths(args.study_area)
    out_dir = paths.root / "accessibility" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.histograms_only:
        plot_overview_maps(paths, out_dir)
    if not args.maps_only:
        plot_histograms(paths, out_dir)

    logger.info("All done. Output in %s", out_dir)


if __name__ == "__main__":
    main()
