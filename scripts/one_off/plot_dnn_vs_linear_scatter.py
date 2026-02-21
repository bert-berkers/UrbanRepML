#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recreate DNN vs Linear Scatter Comparison with Inverted Colormap

This script recreates the 6-panel comparison scatter plot with an inverted
colormap where:
- Darker/more saturated colors = higher |DNN residual| (interesting outliers)
- Lighter/fainter colors = lower |DNN residual| (well-predicted bulk)

Output:
- Saves to: docs/images/dnn_vs_linear_scatter.png
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Target names for display
TARGET_NAMES = {
    "lbm": "Overall Liveability",
    "fys": "Physical Environment",
    "onv": "Safety",
    "soc": "Social Cohesion",
    "vrz": "Amenities",
    "won": "Housing Quality",
}

logger = logging.getLogger(__name__)


def load_dnn_predictions(dnn_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load DNN probe predictions from parquet files.

    Each parquet has columns: [actual, predicted, residual]
    with region_id as index.
    """
    predictions = {}
    for target in TARGET_NAMES.keys():
        parquet_file = dnn_dir / f"predictions_{target}.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            # Reset index to make region_id a column
            df = df.reset_index()
            predictions[target] = df
            logger.info(f"Loaded DNN predictions for {target}: {len(df)} samples")
        else:
            logger.warning(f"DNN predictions file not found: {parquet_file}")
    return predictions


def load_linear_predictions(linear_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load linear probe predictions from parquet files.

    Each parquet has columns: [actual, predicted, residual]
    with region_id as index.
    """
    predictions = {}
    for target in TARGET_NAMES.keys():
        parquet_file = linear_dir / f"predictions_{target}.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            # Reset index to make region_id a column
            df = df.reset_index()
            predictions[target] = df
            logger.info(f"Loaded linear predictions for {target}: {len(df)} samples")
        else:
            logger.warning(f"Linear predictions file not found: {parquet_file}")
    return predictions


def plot_comparison_scatter_inverted(
    dnn_predictions: Dict[str, pd.DataFrame],
    linear_predictions: Dict[str, pd.DataFrame],
    output_path: Path,
    figsize: tuple = (18, 15),
    dpi: int = 150,
) -> None:
    """
    Create 6-panel scatter plot: DNN vs Linear predictions.

    Color by absolute DNN residual with INVERTED colormap:
    - Dark/saturated = high |residual| (interesting outliers)
    - Light/faint = low |residual| (well-predicted)

    Args:
        dnn_predictions: Dict of target -> DNN predictions DataFrame
        linear_predictions: Dict of target -> Linear predictions DataFrame
        output_path: Path to save figure
        figsize: Figure size (width, height)
        dpi: Dots per inch for saved figure
    """

    # Find overlapping targets
    targets = [t for t in dnn_predictions if t in linear_predictions]
    targets.sort()  # Consistent ordering

    if not targets:
        logger.error("No overlapping targets found between DNN and linear predictions")
        return

    logger.info(f"Creating scatter plot for {len(targets)} targets")

    # Create 2x3 grid
    ncols = 3
    nrows = max(1, (len(targets) + ncols - 1) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False
    )

    for idx, target in enumerate(targets):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        dnn_df = dnn_predictions[target]
        lin_df = linear_predictions[target]

        # Align predictions via region_id
        dnn_aligned = dnn_df.set_index("region_id")
        lin_aligned = lin_df.set_index("region_id")

        # Join: keep DNN actual and predicted, add linear predicted
        merged = dnn_aligned[["actual", "predicted"]].join(
            lin_aligned[["predicted"]], how="inner", rsuffix="_lin"
        ).dropna()

        if len(merged) == 0:
            logger.warning(f"No overlapping data for {target}, hiding axis")
            ax.set_visible(False)
            continue

        # Extract columns
        dnn_pred = merged["predicted"].values
        lin_pred = merged["predicted_lin"].values
        actual = merged["actual"].values

        # Compute absolute DNN residual
        abs_residual = np.abs(actual - dnn_pred)

        # Subsample if too many points
        n_points = len(merged)
        if n_points > 50000:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n_points, size=50000, replace=False)
            dnn_pred = dnn_pred[sample_idx]
            lin_pred = lin_pred[sample_idx]
            abs_residual = abs_residual[sample_idx]
            alpha = 0.1
        elif n_points > 10000:
            alpha = 0.2
        else:
            alpha = 0.4

        # Scatter with INVERTED colormap (viridis_r, plasma, inferno, magma)
        # viridis_r: highest values (high residuals) get dark/saturated colors
        sc = ax.scatter(
            dnn_pred, lin_pred, c=abs_residual, s=2,
            alpha=alpha, cmap="viridis_r", rasterized=True,
        )

        # 1:1 reference line
        lims = [
            min(dnn_pred.min(), lin_pred.min()),
            max(dnn_pred.max(), lin_pred.max()),
        ]
        ax.plot(lims, lims, "--", color="red", linewidth=1, alpha=0.7)

        # Labels and title
        target_name = TARGET_NAMES.get(target, target)
        ax.set_title(f"{target_name} ({target})", fontsize=10, fontweight="bold")
        ax.set_xlabel("DNN Predictions", fontsize=9)
        ax.set_ylabel("Linear Predictions", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")

        # Colorbar
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, label="|DNN residual|")
        cbar.ax.tick_params(labelsize=8)

        logger.info(f"  {target}: {n_points:,} samples plotted")

    # Hide unused axes
    for idx in range(len(targets), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "DNN vs Linear Predictions (Inverted Colormap)\n"
        "Darker = Higher |DNN residual|, dashed = 1:1 line",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved comparison scatter (inverted colormap): {output_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Project root
    project_root = Path(__file__).parent.parent.parent

    # Data directories (use same ones as original)
    dnn_dir = (
        project_root / "data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-02-15_default"
    )
    linear_dir = (
        project_root / "data/study_areas/netherlands/stage3_analysis/linear_probe/2026-02-15_default"
    )

    # Output path
    output_path = project_root / "docs/images/dnn_vs_linear_scatter.png"

    logger.info(f"Loading DNN predictions from {dnn_dir}")
    dnn_predictions = load_dnn_predictions(dnn_dir)

    logger.info(f"Loading linear predictions from {linear_dir}")
    linear_predictions = load_linear_predictions(linear_dir)

    logger.info(f"Creating scatter plot and saving to {output_path}")
    plot_comparison_scatter_inverted(
        dnn_predictions,
        linear_predictions,
        output_path,
        figsize=(18, 15),
        dpi=150,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
