"""Cross-approach probe comparison visualizations.

Generates scatter grids (pred vs actual) and spatial residual comparison maps
across multiple probe approaches stored in standardized ``dnn_probe`` format.

Approaches are loaded from potentially different date directories under
``stage3_analysis/dnn_probe/{date}/{approach}/``.

Output goes to ``stage3_analysis/comparison/{today}/probes/``.

Lifetime: durable
Stage: 3
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import (
    load_boundary,
    plot_spatial_map,
    rasterize_rgb,
    RASTER_W,
    RASTER_H,
)

logger = logging.getLogger(__name__)

# Colorblind-safe palette -- distinct for 4 approaches
APPROACH_COLORS = {
    "ring_agg_k10": "#1b9e77",       # teal
    "concat_74d": "#d95f02",          # orange
    "unet_supervised": "#7570b3",     # purple
    "unet_supervised_multiscale": "#e7298a",  # pink
}
FALLBACK_COLORS = list(plt.get_cmap("tab10").colors)


def _get_color(approach: str, idx: int) -> str:
    """Return approach color with fallback."""
    return APPROACH_COLORS.get(approach, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def load_approach(
    paths: StudyAreaPaths,
    approach: str,
    date_str: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and metrics for a single approach from a specific date."""
    base = paths.root / "stage3_analysis" / "dnn_probe" / date_str / approach
    pred_path = base / "predictions.parquet"
    met_path = base / "metrics.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions.parquet at {pred_path}")
    predictions = pd.read_parquet(pred_path)
    metrics = pd.read_parquet(met_path)
    logger.info("Loaded %s/%s: %d predictions, %d metrics", date_str, approach,
                len(predictions), len(metrics))
    return predictions, metrics


def load_approaches(
    paths: StudyAreaPaths,
    approach_dates: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine predictions/metrics for multiple approaches.

    Args:
        paths: StudyAreaPaths instance.
        approach_dates: Mapping from approach name to date directory string.

    Returns:
        (combined_predictions, combined_metrics)
    """
    all_preds = []
    all_mets = []
    for approach, date_str in approach_dates.items():
        pred, met = load_approach(paths, approach, date_str)
        all_preds.append(pred)
        all_mets.append(met)
    return pd.concat(all_preds, ignore_index=True), pd.concat(all_mets, ignore_index=True)


# --------------------------------------------------------------------------
# Scatter grid: one target, multiple approaches side by side
# --------------------------------------------------------------------------

def plot_scatter_grid_target(
    predictions_df: pd.DataFrame,
    target: str,
    approaches: List[str],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Side-by-side scatter plots of pred vs actual for a single target.

    One column per approach. Shared axis limits. R2 annotation per panel.
    """
    sns.set_style("whitegrid")

    tdf = predictions_df[predictions_df["target_variable"] == target].copy()
    if tdf.empty:
        raise ValueError(f"No predictions for target '{target}'")

    ncols = len(approaches)
    cell_w = 4.0
    cell_h = 4.0
    fig, axes = plt.subplots(1, ncols, figsize=(cell_w * ncols + 0.5, cell_h + 0.6),
                             squeeze=False)

    # Shared limits across all approaches
    vmin = min(tdf["y_true"].min(), tdf["y_pred"].min())
    vmax = max(tdf["y_true"].max(), tdf["y_pred"].max())
    pad = (vmax - vmin) * 0.05
    lim = (vmin - pad, vmax + pad)

    for i, approach in enumerate(approaches):
        ax = axes[0, i]
        sub = tdf[tdf["approach"] == approach]
        color = _get_color(approach, i)

        if sub.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        else:
            # Subsample for readability
            n_plot = min(len(sub), 8000)
            plot_sub = sub.sample(n_plot, random_state=42) if len(sub) > n_plot else sub

            ax.scatter(
                plot_sub["y_true"], plot_sub["y_pred"],
                s=1.5, alpha=0.25, color=color, rasterized=True,
            )
            ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
            ax.set_xlim(lim)
            ax.set_ylim(lim)

            # R2 annotation
            ss_res = ((sub["y_true"] - sub["y_pred"]) ** 2).sum()
            ss_tot = ((sub["y_true"] - sub["y_true"].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            n_total = len(sub)
            ax.text(
                0.05, 0.95,
                f"R2 = {r2:.4f}\nn = {n_total:,}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#cccccc"),
            )

        # Labels
        label = approach.replace("_", " ")
        ax.set_title(label, fontsize=10, fontweight="bold")
        if i == 0:
            ax.set_ylabel("Predicted", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Actual", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Probe Predictions vs Actual: {target.upper()}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter grid: %s", output_path)
    return output_path


# --------------------------------------------------------------------------
# Spatial residual comparison maps
# --------------------------------------------------------------------------

def plot_residual_comparison(
    predictions_df: pd.DataFrame,
    target: str,
    approaches: List[str],
    paths: StudyAreaPaths,
    output_path: Path,
    h3_resolution: int = 9,
    dpi: int = 300,
) -> Path:
    """Side-by-side spatial residual maps for a single target across approaches.

    Uses full raster resolution (2000x2400) per panel. RdBu_r diverging colormap
    with symmetric TwoSlopeNorm shared across panels.
    """
    df = predictions_df[predictions_df["target_variable"] == target].copy()
    if df.empty:
        raise ValueError(f"No predictions for target '{target}'")

    df["residual"] = df["y_pred"] - df["y_true"]

    n = len(approaches)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    # Load boundary and SpatialDB for centroids
    boundary_gdf = load_boundary(paths, crs=28992)
    db = SpatialDB.for_study_area(paths.study_area)
    stamp = max(1, 11 - h3_resolution)

    # Compute shared symmetric vmax across all approaches for this target
    vmax_res = float(df["residual"].abs().quantile(0.98))
    if vmax_res == 0:
        vmax_res = 1.0
    norm = TwoSlopeNorm(vmin=-vmax_res, vcenter=0, vmax=vmax_res)
    cmap = plt.get_cmap("RdBu_r")

    # Compute shared extent
    if boundary_gdf is not None:
        extent = tuple(boundary_gdf.total_bounds)
    else:
        first_sub = df[df["approach"] == approaches[0]]
        hex_ids = pd.Index(first_sub["region_id"].unique(), name="region_id")
        extent = db.extent(hex_ids, h3_resolution, crs=28992)

    minx, miny, maxx, maxy = extent
    pad_x = (maxx - minx) * 0.03
    pad_y = (maxy - miny) * 0.03
    render_extent = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

    # Each panel gets full raster resolution
    panel_w = RASTER_W
    panel_h = RASTER_H

    # Figure sizing: proportional to raster aspect ratio
    aspect = panel_h / panel_w
    fig_panel_w = 6.0
    fig_panel_h = fig_panel_w * aspect
    fig_w = fig_panel_w * ncols + 2.0  # extra for colorbar
    fig_h = fig_panel_h * nrows + 1.0

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    for idx, approach in enumerate(approaches):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        sub = df[df["approach"] == approach]
        hex_ids = pd.Index(sub["region_id"].values, name="region_id")
        residuals = sub["residual"].values

        # Get centroids via SpatialDB
        cx, cy = db.centroids(hex_ids, h3_resolution, crs=28992)

        # Map residuals through colormap
        rgba = cmap(norm(residuals))
        rgb = rgba[:, :3].astype(np.float32)

        # Rasterize using the shared utility
        image = rasterize_rgb(cx, cy, rgb, render_extent,
                              width=panel_w, height=panel_h, stamp=stamp)

        label = approach.replace("_", " ")
        # Compute R2 for the subtitle
        ss_res = ((sub["y_true"] - sub["y_pred"]) ** 2).sum()
        ss_tot = ((sub["y_true"] - sub["y_true"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        plot_spatial_map(
            ax, image, render_extent,
            boundary_gdf=boundary_gdf,
            title=f"{label}\nR2={r2:.3f}",
            show_rd_grid=False,
            title_fontsize=10,
        )

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        shrink=0.6,
        label="Residual (predicted - actual)",
        pad=0.02,
    )
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Spatial Residuals: {target.upper()}\n"
        f"res{h3_resolution} | EPSG:28992 | blue=underprediction, red=overprediction",
        fontsize=13, fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved residual comparison: %s", output_path)
    return output_path


# --------------------------------------------------------------------------
# R2 comparison bar chart
# --------------------------------------------------------------------------

def plot_r2_comparison(
    metrics_df: pd.DataFrame,
    approaches: List[str],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Grouped bar chart of R2 across approaches and targets."""
    sns.set_style("whitegrid")

    r2 = metrics_df[metrics_df["metric"] == "r2"].copy()
    r2 = r2[r2["approach"].isin(approaches)]

    # Sort targets by canonical order
    canonical = ["lbm", "fys", "onv", "soc", "vrz", "won"]
    all_targets = sorted(r2["target_variable"].unique())
    targets = [t for t in canonical if t in all_targets]
    targets += [t for t in all_targets if t not in targets]

    n_a = len(approaches)
    n_t = len(targets)
    width = 0.8 / n_a
    x = np.arange(n_t)

    fig_w = max(8, n_t * 2.0 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    for i, approach in enumerate(approaches):
        subset = r2[r2["approach"] == approach].set_index("target_variable")
        vals = [subset.loc[t, "value"] if t in subset.index else 0.0 for t in targets]
        offset = (i - (n_a - 1) / 2) * width
        color = _get_color(approach, i)
        label = approach.replace("_", " ")
        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                    rotation=45 if n_a > 3 else 0)

    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in targets], fontsize=10)
    ax.set_ylabel("R2", fontsize=11)
    ax.set_title("Probe R2 Comparison Across Approaches", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0, top=min(1.0, r2["value"].max() * 1.15))

    # Add mean R2 as text annotation
    for i, approach in enumerate(approaches):
        subset = r2[r2["approach"] == approach]
        mean_r2 = subset["value"].mean()
        color = _get_color(approach, i)
        label = approach.replace("_", " ")
        ax.text(0.02, 0.95 - i * 0.05, f"{label}: mean R2={mean_r2:.3f}",
                transform=ax.transAxes, fontsize=8, color=color, fontweight="bold",
                verticalalignment="top")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved R2 comparison: %s", output_path)
    return output_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-approach probe comparison visualizations."
    )
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--resolution", type=int, default=9)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    paths = StudyAreaPaths(args.study_area)
    today = date.today().isoformat()
    output_dir = paths.root / "stage3_analysis" / "comparison" / today / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define which approaches to load and from which date
    approach_dates = {
        "ring_agg_k10": "2026-03-21",
        "concat_74d": "2026-03-21",
        "unet_supervised": "2026-03-22",
        "unet_supervised_multiscale": "2026-03-22",
    }

    logger.info("Loading approaches: %s", list(approach_dates.keys()))
    predictions, metrics = load_approaches(paths, approach_dates)

    approaches = list(approach_dates.keys())

    # 1. R2 comparison bar chart
    r2_path = plot_r2_comparison(metrics, approaches,
                                 output_dir / "r2_comparison.png", dpi=args.dpi)

    # 2. Scatter grids for key targets
    for target in ["lbm", "vrz"]:
        scatter_path = plot_scatter_grid_target(
            predictions, target, approaches,
            output_dir / f"scatter_grid_{target}.png", dpi=args.dpi,
        )

    # 3. Spatial residual comparison maps
    for target in ["lbm", "vrz"]:
        residual_path = plot_residual_comparison(
            predictions, target, approaches, paths,
            output_dir / f"residual_comparison_{target}.png",
            h3_resolution=args.resolution, dpi=args.dpi,
        )

    logger.info("All plots saved to %s", output_dir)


if __name__ == "__main__":
    main()
