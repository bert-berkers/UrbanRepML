"""Generate probe visualization suite for supervised UNet approaches.

Reads prediction parquets from the linear_probe output directory and produces:
- Scatter plots (pred vs actual) for all 6 targets, with density coloring
- Spatial prediction maps (rasterized at 2000x2400) for lbm, vrz, fys, onv, soc, won
- Spatial residual maps (rasterized at 2000x2400) for lbm, vrz, fys, onv, soc, won

Lifetime: durable
Stage: 3 (post-training analysis)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

# -- Project imports --
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import (
    RASTER_W,
    RASTER_H,
    load_boundary,
    rasterize_continuous_voronoi,
    voronoi_params_for_resolution,
    plot_spatial_map,
    _add_colorbar,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STUDY_AREA = "netherlands"
H3_RES = 9
TARGETS = ["lbm", "fys", "onv", "soc", "vrz", "won"]
TARGET_NAMES = {
    "lbm": "Overall Liveability",
    "fys": "Physical Environment",
    "onv": "Safety",
    "soc": "Social Cohesion",
    "vrz": "Amenities",
    "won": "Housing Quality",
}

_LINEAR_PROBE_DIR = StudyAreaPaths(STUDY_AREA).stage3("linear_probe")

APPROACHES = {
    "unet_supervised": str(
        _LINEAR_PROBE_DIR / "2026-03-22" / "2026-03-22_supervised_unet_supervised"
    ),
    "unet_supervised_multiscale": str(
        _LINEAR_PROBE_DIR
        / "2026-03-22"
        / "2026-03-22_supervised_unet_supervised_multiscale"
    ),
}

OUTPUT_BASE = (
    StudyAreaPaths(STUDY_AREA).stage3("comparison") / "2026-03-22" / "probes"
)

DPI = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_predictions(pred_dir: Path, target: str) -> pd.DataFrame:
    """Load a predictions_{target}.parquet file."""
    path = pred_dir / f"predictions_{target}.parquet"
    df = pd.read_parquet(path)
    logger.info("  Loaded %s: %d rows", path.name, len(df))
    return df


def load_metrics(pred_dir: Path) -> pd.DataFrame:
    """Load metrics_summary.csv to get R2 values."""
    path = pred_dir / "metrics_summary.csv"
    return pd.read_csv(path)


def get_r2(metrics_df: pd.DataFrame, target: str) -> float:
    """Extract overall R2 for a target from the metrics summary."""
    row = metrics_df[metrics_df["target"] == target]
    if len(row) == 0:
        return float("nan")
    # The columns may include 'overall_r2' or 'mean_r2' -- check what exists
    for col in ["overall_r2", "mean_r2"]:
        if col in row.columns:
            return float(row[col].iloc[0])
    # Fallback: average the fold R2 columns
    fold_cols = [c for c in row.columns if c.startswith("fold") and c.endswith("_r2")]
    if fold_cols:
        return float(row[fold_cols].iloc[0].mean())
    return float("nan")


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------

def plot_scatter(
    pred_df: pd.DataFrame,
    target: str,
    r2: float,
    approach_name: str,
    output_dir: Path,
) -> Path:
    """Density-colored scatter plot: predicted vs actual."""
    actual = pred_df["actual"].values
    predicted = pred_df["predicted"].values

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    n = len(actual)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Subsample for scatter visibility, compute 2D density for color
    if n > 50_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=50_000, replace=False)
        a_plot, p_plot = actual[idx], predicted[idx]
    else:
        a_plot, p_plot = actual, predicted

    # Compute density via 2D histogram for coloring
    from scipy.stats import gaussian_kde
    try:
        xy = np.vstack([a_plot, p_plot])
        kde = gaussian_kde(xy)
        density = kde(xy)
    except Exception:
        density = np.ones(len(a_plot))

    # Sort by density so high-density points render on top
    sort_idx = np.argsort(density)
    a_sorted = a_plot[sort_idx]
    p_sorted = p_plot[sort_idx]
    d_sorted = density[sort_idx]

    sc = ax.scatter(
        a_sorted, p_sorted,
        c=d_sorted, cmap="viridis", s=2, alpha=0.6, rasterized=True,
    )
    fig.colorbar(sc, ax=ax, label="Density", shrink=0.7)

    # 1:1 line
    lims = [
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max()),
    ]
    ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="1:1 line")

    # Best-fit line
    z = np.polyfit(actual, predicted, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(lims[0], lims[1], 100)
    ax.plot(
        x_fit, p_fit(x_fit), "-", color="white", linewidth=2,
        label=f"Fit: y={z[0]:.3f}x + {z[1]:.3f}",
    )

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    tname = TARGET_NAMES.get(target, target)
    ax.set_xlabel(f"Actual {tname}", fontsize=12)
    ax.set_ylabel(f"Predicted {tname}", fontsize=12)
    ax.set_title(
        f"{approach_name}: {tname} ({target})\n"
        f"R$^2$={r2:.4f}, RMSE={rmse:.4f}, n={n:,}",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    path = output_dir / f"scatter_{target}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved scatter: %s", path)
    return path


# ---------------------------------------------------------------------------
# Spatial maps
# ---------------------------------------------------------------------------

def make_spatial_maps(
    pred_df: pd.DataFrame,
    target: str,
    r2: float,
    approach_name: str,
    output_dir: Path,
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    boundary_gdf,
) -> tuple[Path, Path]:
    """Generate prediction map and residual map for one target."""
    actual = pred_df["actual"].values
    predicted = pred_df["predicted"].values
    residual = pred_df["residual"].values

    tname = TARGET_NAMES.get(target, target)

    # --- Prediction map ---
    vmin_pred = float(np.nanpercentile(predicted, 2))
    vmax_pred = float(np.nanpercentile(predicted, 98))

    pixel_m, max_dist_m = voronoi_params_for_resolution(H3_RES)
    img_pred, _ = rasterize_continuous_voronoi(
        cx, cy, predicted, extent,
        cmap="viridis", vmin=vmin_pred, vmax=vmax_pred,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    plot_spatial_map(
        ax, img_pred, extent, boundary_gdf,
        title=f"{approach_name}: Predicted {tname}\nR$^2$={r2:.4f}",
    )
    _add_colorbar(fig, ax, "viridis", vmin_pred, vmax_pred, label=tname)

    plt.tight_layout()
    pred_path = output_dir / f"prediction_map_{target}.png"
    fig.savefig(pred_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved prediction map: %s", pred_path)

    # --- Residual map ---
    vmax_res = float(np.nanpercentile(np.abs(residual), 98))
    if vmax_res == 0:
        vmax_res = 1.0

    img_res, _ = rasterize_continuous_voronoi(
        cx, cy, residual, extent,
        cmap="RdBu_r", vmin=-vmax_res, vmax=vmax_res,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )

    fig, ax = plt.subplots(figsize=(10, 12))
    plot_spatial_map(
        ax, img_res, extent, boundary_gdf,
        title=f"{approach_name}: Residuals {tname}\nR$^2$={r2:.4f}",
    )
    _add_colorbar(fig, ax, "RdBu_r", -vmax_res, vmax_res, label="Residual (actual - predicted)")

    plt.tight_layout()
    res_path = output_dir / f"residual_map_{target}.png"
    fig.savefig(res_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved residual map: %s", res_path)

    return pred_path, res_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    paths = StudyAreaPaths(STUDY_AREA)
    db = SpatialDB.for_study_area(STUDY_AREA)

    # Load boundary once
    boundary_gdf = load_boundary(paths)
    logger.info("Loaded boundary, CRS=%s", boundary_gdf.crs if boundary_gdf is not None else "N/A")

    # Pre-compute extent from boundary
    if boundary_gdf is not None:
        bounds = boundary_gdf.total_bounds
        pad = (bounds[2] - bounds[0]) * 0.03
        extent = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
    else:
        raise RuntimeError("No boundary GeoDataFrame -- cannot determine extent")

    # Precompute centroids for the first approach's first target to get hex_ids
    # (all approaches share the same hex set at res9)
    first_approach_dir = Path(list(APPROACHES.values())[0])
    sample_df = load_predictions(first_approach_dir, TARGETS[0])
    hex_ids = sample_df.index
    logger.info("Computing centroids for %d hexagons...", len(hex_ids))
    cx, cy = db.centroids(hex_ids, resolution=H3_RES, crs=28992)
    logger.info("Centroids computed. Extent: %.0f,%.0f -> %.0f,%.0f", *extent)

    for approach_name, pred_dir_str in APPROACHES.items():
        pred_dir = Path(pred_dir_str)
        output_dir = OUTPUT_BASE / approach_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Processing %s ===", approach_name)

        metrics_df = load_metrics(pred_dir)

        for target in TARGETS:
            logger.info("  Target: %s", target)
            pred_df = load_predictions(pred_dir, target)
            r2 = get_r2(metrics_df, target)

            # Ensure hex_ids alignment for this target
            # Reindex centroids if hex_ids differ
            if not pred_df.index.equals(hex_ids):
                logger.info("  Re-computing centroids for %d hexagons", len(pred_df))
                cx_t, cy_t = db.centroids(pred_df.index, resolution=H3_RES, crs=28992)
            else:
                cx_t, cy_t = cx, cy

            # Scatter plot for ALL targets
            plot_scatter(pred_df, target, r2, approach_name, output_dir)

            # Spatial maps for ALL targets
            make_spatial_maps(
                pred_df, target, r2, approach_name, output_dir,
                cx_t, cy_t, extent, boundary_gdf,
            )

    logger.info("All done. Output in %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
