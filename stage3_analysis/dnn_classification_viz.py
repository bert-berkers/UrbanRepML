#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Classification Probe Visualization: MLP-specific Analysis Plots

Extends ClassificationVisualizer with DNN-specific training curve plots.
The parent class provides confusion matrices, accuracy/F1 comparison bars,
and fold metric box plots. This subclass adds:

    1. Training curves: val_loss (CrossEntropy) vs epoch per fold,
       with early stopping markers and per-target Acc/F1 in titles
    2. Faceted training curves: all targets in a single grid figure

Coefficient-based visualizations are not applicable (MLP has no
interpretable linear coefficients).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .linear_probe import TargetResult, TARGET_NAMES, TAXONOMY_TARGET_NAMES
from .classification_viz import ClassificationVisualizer

logger = logging.getLogger(__name__)


class DNNClassificationVisualizer(ClassificationVisualizer):
    """
    Visualization for DNN (MLP) classification probe results.

    Inherits from ClassificationVisualizer which provides:
        - plot_confusion_matrix (per-target normalized heatmap)
        - plot_classification_metrics_comparison (accuracy + F1 bars)
        - plot_fold_metrics (box plots of per-fold accuracy/F1)

    Adds DNN-specific visualizations:
        - Training curves (val CrossEntropy loss vs epoch per fold)
        - Faceted training curves for all targets
    """

    def __init__(
        self,
        results: Dict[str, TargetResult],
        output_dir: Path,
        training_curves: Optional[Dict[str, Dict[int, List[float]]]] = None,
        study_area: str = "netherlands",
        figsize_base: Tuple[float, float] = (10, 6),
        dpi: int = 150,
    ):
        """
        Args:
            results: Dictionary mapping target column to TargetResult.
            output_dir: Directory to save figures.
            training_curves: Dict of {target_col: {fold_id: [val_loss_per_epoch]}}.
                This is the structure stored in DNNClassificationProber.training_curves.
            study_area: Study area name for resolving data paths.
            figsize_base: Base figure size (width, height).
            dpi: Dots per inch for saved figures.
        """
        super().__init__(results, output_dir, study_area, figsize_base, dpi)
        self.training_curves = training_curves or {}

    # ------------------------------------------------------------------
    # DNN-specific: training curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self, target_col: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot validation loss (CrossEntropy) vs epoch for each fold.

        If target_col is None, produces a faceted grid of all targets.
        If target_col is specified, produces a single panel.

        Each fold is a separate line. The early stopping epoch (where best
        val_loss was achieved) is marked with a dot. Title includes
        overall Acc and F1 instead of R2.

        Args:
            target_col: Target variable name, or None for all targets.

        Returns:
            Path to saved figure, or None if no training curves available.
        """
        if not self.training_curves:
            logger.warning(
                "No training curves available, skipping plot_training_curves"
            )
            return None

        if target_col is not None:
            return self._plot_training_curves_single(target_col)

        # Faceted grid: all targets
        targets = [t for t in self.results if t in self.training_curves]
        if not targets:
            logger.warning("No targets with training curves, skipping")
            return None

        ncols = 3
        nrows = max(1, (len(targets) + ncols - 1) // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
        )

        fold_colors = sns.color_palette("Set2", n_colors=10)

        for idx, tc in enumerate(targets):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            self._draw_training_curve_panel(ax, tc, fold_colors)

        # Hide unused axes
        for idx in range(len(targets), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle(
            "DNN Classification Probe: Validation Loss per Epoch (Spatial CV Folds)",
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        path = self.output_dir / "training_curves_all.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved training curves (all targets): {path}")
        return path

    def _plot_training_curves_single(self, target_col: str) -> Optional[Path]:
        """Plot training curves for a single target."""
        if target_col not in self.training_curves:
            logger.warning(
                f"No training curves for {target_col}, skipping"
            )
            return None

        fig, ax = plt.subplots(figsize=self.figsize_base)
        fold_colors = sns.color_palette("Set2", n_colors=10)
        self._draw_training_curve_panel(ax, target_col, fold_colors)

        plt.tight_layout()
        path = self.output_dir / f"training_curves_{target_col}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved training curves: {path}")
        return path

    def _draw_training_curve_panel(
        self,
        ax: plt.Axes,
        target_col: str,
        fold_colors: list,
    ) -> None:
        """Draw training curves for one target onto an axes."""
        curves = self.training_curves.get(target_col, {})
        result = self.results.get(target_col)

        best_overall_loss = float("inf")
        best_overall_epoch = 0

        for i, (fold_id, losses) in enumerate(sorted(curves.items())):
            epochs = np.arange(1, len(losses) + 1)
            color = fold_colors[i % len(fold_colors)]
            ax.plot(
                epochs,
                losses,
                color=color,
                linewidth=1.2,
                label=f"Fold {fold_id}",
                alpha=0.85,
                rasterized=True,
            )

            # Mark early stopping epoch (minimum val_loss)
            best_epoch_idx = int(np.argmin(losses))
            best_loss = losses[best_epoch_idx]
            ax.plot(
                best_epoch_idx + 1,
                best_loss,
                "o",
                color=color,
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=5,
            )

            if best_loss < best_overall_loss:
                best_overall_loss = best_loss
                best_overall_epoch = best_epoch_idx + 1

        target_name = TAXONOMY_TARGET_NAMES.get(
            target_col, TARGET_NAMES.get(target_col, target_col)
        )

        # Build metric string: Acc and F1 instead of R2
        metric_str = ""
        if result is not None:
            acc = result.overall_accuracy
            f1 = result.overall_f1_macro
            if acc is not None and f1 is not None:
                metric_str = f"Acc={acc:.4f}, F1={f1:.4f}"

        ax.set_title(
            f"{target_name} ({target_col})\n"
            f"Best val_loss={best_overall_loss:.5f} @ epoch {best_overall_epoch}"
            + (f" | {metric_str}" if metric_str else ""),
            fontsize=10,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss (CrossEntropy)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linewidth=0.5, alpha=0.5)

    # ------------------------------------------------------------------
    # Override plot_all
    # ------------------------------------------------------------------

    def plot_all(
        self,
        skip_spatial: bool = False,
        **kwargs,
    ) -> List[Path]:
        """
        Generate all DNN classification probe visualizations.

        Calls the parent ClassificationVisualizer.plot_all() for base plots
        (confusion matrices, accuracy/F1 bars, fold metrics), then adds
        DNN-specific training curve plots.

        Args:
            skip_spatial: If True, skip spatial maps (passed to parent).
            **kwargs: Additional keyword arguments passed to parent's plot_all.

        Returns:
            List of paths to all saved figures.
        """
        logger.info(
            f"Generating DNN classification visualizations to {self.output_dir}"
        )

        # Get base classification plots from parent
        paths: List[Path] = super().plot_all(skip_spatial=skip_spatial, **kwargs)

        # -- DNN-specific: training curves --
        curves_path = self.plot_training_curves()
        if curves_path is not None:
            paths.append(curves_path)

        # Also individual target curves
        for target_col in self.results:
            tc_path = self.plot_training_curves(target_col)
            if tc_path is not None:
                paths.append(tc_path)

        # Filter out None values
        paths = [p for p in paths if p is not None]

        logger.info(f"Generated {len(paths)} DNN classification visualizations")
        return paths
