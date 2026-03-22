"""Side-by-side raster cluster comparison maps across approaches.

Reads standardized ClusterResultsWriter parquets (assignments.parquet,
metrics.parquet) from cluster_results/{approach}/ and generates:

1. Side-by-side spatial cluster maps at given k values (full 2000x2400
   raster per panel -- panels are NOT downsized).
2. Silhouette score bar chart comparing approaches across k values.

Cluster labels are sorted by mean leefbaarometer lbm score so that
cluster 0 = lowest liveability, cluster k-1 = highest. Uses viridis
colormap so dark = low lbm, bright = high lbm.

Lifetime: durable
Stage: 3
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import List, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import (
    load_boundary,
    plot_spatial_map,
    rasterize_categorical,
    RASTER_W,
    RASTER_H,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label sorting by target mean
# ---------------------------------------------------------------------------


def sort_labels_by_target(
    labels: np.ndarray,
    region_ids: np.ndarray,
    target_series: pd.Series,
) -> np.ndarray:
    """Relabel clusters so 0 = lowest mean target, k-1 = highest.

    Clusters with zero target coverage (all NaN) are sorted to the end.
    """
    tmp = pd.DataFrame({
        "cluster_label": labels,
        "target": target_series.reindex(region_ids).values,
    })
    cluster_means = tmp.groupby("cluster_label")["target"].mean()
    cluster_means = cluster_means.fillna(np.inf)
    sorted_clusters = cluster_means.sort_values().index.tolist()
    old_to_new = {old: new for new, old in enumerate(sorted_clusters)}
    return np.array([old_to_new[lbl] for lbl in labels], dtype=int)


# ---------------------------------------------------------------------------
# Side-by-side cluster map
# ---------------------------------------------------------------------------


def plot_side_by_side(
    assignments_df: pd.DataFrame,
    k: int,
    approaches: List[str],
    target_series: pd.Series,
    db: SpatialDB,
    boundary_gdf,
    render_extent: tuple,
    output_path: Path,
    h3_resolution: int = 9,
    dpi: int = 300,
    stamp: int = 2,
) -> Path:
    """Render side-by-side cluster maps with full raster resolution per panel.

    Each panel is RASTER_W x RASTER_H (2000x2400) -- total figure width
    scales with the number of panels.
    """
    sns.set_style("white")

    df = assignments_df[assignments_df["k"] == k].copy()
    if df.empty:
        raise ValueError(f"No assignments for k={k}")

    n = len(approaches)

    # Each panel is full raster resolution
    panel_w_inches = RASTER_W / dpi
    panel_h_inches = RASTER_H / dpi
    # Extra width for colorbar
    cbar_width_inches = 0.8
    fig_w = panel_w_inches * n + cbar_width_inches
    fig_h = panel_h_inches

    fig, axes = plt.subplots(
        1, n,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for idx, approach in enumerate(approaches):
        ax = axes[0, idx]
        sub = df[df["approach"] == approach]
        if sub.empty:
            ax.set_visible(False)
            ax.set_title(f"{approach}\n(no data)", fontsize=10)
            continue

        region_ids = sub["region_id"].values
        raw_labels = sub["cluster_label"].values

        # Sort labels by lbm mean
        sorted_labels = sort_labels_by_target(
            raw_labels, region_ids, target_series
        )

        # Get centroids
        hex_ids = pd.Index(region_ids, name="region_id")
        cx, cy = db.centroids(hex_ids, h3_resolution, crs=28992)

        # Rasterize at full resolution
        image = rasterize_categorical(
            cx, cy, sorted_labels,
            render_extent,
            n_clusters=k,
            width=RASTER_W,
            height=RASTER_H,
            cmap="viridis",
            stamp=stamp,
        )

        plot_spatial_map(
            ax, image, render_extent, boundary_gdf,
            title=approach,
            show_rd_grid=False,
            title_fontsize=12,
        )

    # Shared colorbar
    sm = cm.ScalarMappable(
        cmap=plt.get_cmap("viridis"),
        norm=plt.Normalize(vmin=0, vmax=k - 1),
    )
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        shrink=0.5,
        label=f"Cluster (sorted by mean lbm, 0=lowest)",
        pad=0.01,
        aspect=30,
    )
    cbar.set_ticks(range(k))
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Cluster Comparison  k={k}  |  res{h3_resolution}  |  sorted by lbm",
        fontsize=14,
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Metrics bar chart
# ---------------------------------------------------------------------------


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    approaches: List[str],
    k_values: List[int],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Grouped bar chart of silhouette scores across approaches and k values."""
    sns.set_style("whitegrid")

    sil = metrics_df[metrics_df["metric"] == "silhouette"].copy()
    sil = sil[sil["approach"].isin(approaches)]
    sil = sil[sil["k"].isin(k_values)]

    if sil.empty:
        raise ValueError("No silhouette data for requested approaches/k values")

    n_approaches = len(approaches)
    n_k = len(k_values)
    x = np.arange(n_k)
    bar_width = 0.8 / n_approaches

    # Use a colorblind-safe palette
    colors = sns.color_palette("colorblind", n_approaches)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, approach in enumerate(approaches):
        vals = []
        for k_val in k_values:
            row = sil[(sil["approach"] == approach) & (sil["k"] == k_val)]
            vals.append(row["value"].values[0] if len(row) > 0 else 0.0)

        bars = ax.bar(
            x + i * bar_width - (n_approaches - 1) * bar_width / 2,
            vals,
            bar_width * 0.9,
            label=approach,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

        # Value annotations
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_values], fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Cluster Quality: Silhouette Score by Approach", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side raster cluster comparison maps."
    )
    parser.add_argument(
        "--study-area", default="netherlands",
    )
    parser.add_argument(
        "--approaches", nargs="*", default=None,
        help="Approaches to include. Default: all found.",
    )
    parser.add_argument(
        "--k-values", nargs="*", type=int, default=[8, 12],
        help="k values for side-by-side maps. Default: 8 12",
    )
    parser.add_argument(
        "--metric-k-values", nargs="*", type=int, default=[5, 8, 12],
        help="k values for metrics bar chart. Default: 5 8 12",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
    )
    parser.add_argument(
        "--resolution", type=int, default=9,
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
    )
    parser.add_argument(
        "--stamp", type=int, default=2,
        help="Pixel stamp radius (default: 2 for res9).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    paths = StudyAreaPaths(args.study_area)
    db = SpatialDB.for_study_area(args.study_area)

    output_dir = args.output_dir or (
        paths.root / "stage3_analysis" / "comparison"
        / date.today().isoformat() / "clusters"
    )

    # Load all assignments and metrics
    root = paths.cluster_results_root()
    assign_files = sorted(root.glob("*/assignments.parquet"))
    met_files = sorted(root.glob("*/metrics.parquet"))

    assignments = pd.concat(
        [pd.read_parquet(f) for f in assign_files], ignore_index=True
    )
    metrics = pd.concat(
        [pd.read_parquet(f) for f in met_files], ignore_index=True
    )

    if args.approaches:
        approaches = args.approaches
    else:
        approaches = sorted(assignments["approach"].unique())

    assignments = assignments[assignments["approach"].isin(approaches)]
    metrics = metrics[metrics["approach"].isin(approaches)]

    logger.info(
        "Loaded %d approaches: %s", len(approaches), approaches
    )

    # Load lbm target for label sorting
    target_path = paths.target_file("leefbaarometer", args.resolution, 2022)
    target_df = pd.read_parquet(target_path)
    lbm_series = target_df["lbm"].astype(float)
    lbm_series.index.name = "region_id"
    logger.info("Loaded lbm target: %d hexagons", len(lbm_series))

    # Load boundary and compute extent
    boundary_gdf = load_boundary(paths, crs=28992)
    if boundary_gdf is not None:
        extent = tuple(boundary_gdf.total_bounds)
    else:
        # Fallback: compute from first approach's hex ids
        first_sub = assignments[assignments["approach"] == approaches[0]]
        hex_ids = pd.Index(first_sub["region_id"].unique(), name="region_id")
        extent = db.extent(hex_ids, args.resolution, crs=28992)
    minx, miny, maxx, maxy = extent
    pad = (maxx - minx) * 0.03
    render_extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    saved = []

    # Side-by-side cluster maps
    for k in args.k_values:
        out_path = output_dir / f"side_by_side_k{k}.png"
        saved.append(plot_side_by_side(
            assignments, k, approaches, lbm_series,
            db, boundary_gdf, render_extent, out_path,
            h3_resolution=args.resolution,
            dpi=args.dpi,
            stamp=args.stamp,
        ))

    # Metrics comparison bar chart
    met_path = output_dir / "metrics_comparison.png"
    saved.append(plot_metrics_comparison(
        metrics, approaches, args.metric_k_values, met_path,
        dpi=args.dpi,
    ))

    print(f"\nSaved {len(saved)} plots:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
