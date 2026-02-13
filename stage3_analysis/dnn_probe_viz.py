#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe Visualization: GNN-specific Analysis Plots

Extends LinearProbeVisualizer for GNN probe results. Coefficient-based
visualizations (meaningless for neural networks) are gracefully skipped,
while prediction-based plots are inherited unchanged. Adds DNN-specific
visualizations: training curves, linear-vs-DNN comparison bars,
comparison scatter, and spatial improvement maps.

New visualizations:
    1. Training curves: val_loss vs epoch per fold, with early stopping markers
    2. Comparison bars: side-by-side R2/RMSE for linear vs DNN
    3. Comparison scatter: DNN vs linear predictions coloured by residual
    4. Spatial improvement: map of where DNN beats/loses to linear
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from .linear_probe import TargetResult, TARGET_NAMES
from .linear_probe_viz import LinearProbeVisualizer

logger = logging.getLogger(__name__)


class DNNProbeVisualizer(LinearProbeVisualizer):
    """
    Visualization for DNN (GNN) probe regression results.

    Inherits from LinearProbeVisualizer. Coefficient-based methods are
    overridden to return None with an info log, since GNN probes have
    no meaningful linear coefficients. Prediction-based methods
    (scatter, spatial residuals, fold metrics) are inherited unchanged.

    Adds DNN-specific visualizations:
        - Training curves (val_loss vs epoch per fold)
        - Linear vs DNN comparison bar chart
        - Comparison scatter (DNN vs linear predictions)
        - Spatial improvement map (where DNN beats linear)
    """

    def __init__(
        self,
        results: Dict[str, TargetResult],
        output_dir: Path,
        training_curves: Optional[Dict[str, Dict[int, List[float]]]] = None,
        figsize_base: Tuple[float, float] = (10, 6),
        dpi: int = 150,
    ):
        """
        Args:
            results: Dictionary mapping target column to TargetResult.
            output_dir: Directory to save figures.
            training_curves: Dict of {target_col: {fold_id: [val_loss_per_epoch]}}.
                This is the structure stored in DNNProbeRegressor.training_curves.
            figsize_base: Base figure size (width, height).
            dpi: Dots per inch for saved figures.
        """
        super().__init__(results, output_dir, figsize_base, dpi)
        self.training_curves = training_curves or {}

    # ------------------------------------------------------------------
    # Override coefficient-based methods (not meaningful for GNN)
    # ------------------------------------------------------------------

    def plot_coefficient_bars(self, target_col: str, top_n: int = 15) -> None:
        """Skipped -- coefficients are not meaningful for GNN probe."""
        logger.info(
            "Skipping plot_coefficient_bars -- not meaningful for GNN probe"
        )
        return None

    def plot_coefficient_bars_faceted(self) -> None:
        """Skipped -- coefficients are not meaningful for GNN probe."""
        logger.info(
            "Skipping plot_coefficient_bars_faceted -- not meaningful for GNN probe"
        )
        return None

    def plot_coefficient_heatmap(self) -> None:
        """Skipped -- coefficients are not meaningful for GNN probe."""
        logger.info(
            "Skipping plot_coefficient_heatmap -- not meaningful for GNN probe"
        )
        return None

    def plot_rgb_top3_map(
        self,
        target_col: str,
        embeddings_path: Optional[Path] = None,
        boundary_gdf: Optional[gpd.GeoDataFrame] = None,
        max_hexagons: int = 600_000,
    ) -> None:
        """Skipped -- coefficient-ranked RGB channels are not meaningful for GNN probe."""
        logger.info(
            "Skipping plot_rgb_top3_map -- not meaningful for GNN probe"
        )
        return None

    def plot_cross_target_correlation(self) -> None:
        """Skipped -- coefficient correlation is not meaningful for GNN probe."""
        logger.info(
            "Skipping plot_cross_target_correlation -- not meaningful for GNN probe"
        )
        return None

    # ------------------------------------------------------------------
    # DNN-specific: training curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self, target_col: Optional[str] = None
    ) -> Optional[Path]:
        """
        Plot validation loss vs epoch for each fold.

        If target_col is None, produces a faceted 2x3 grid of all targets.
        If target_col is specified, produces a single panel.

        Each fold is a separate line. The early stopping epoch (where best
        val_loss was achieved) is marked with a dot.

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
            "DNN Probe: Validation Loss per Epoch (Spatial CV Folds)",
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

        target_name = TARGET_NAMES.get(target_col, target_col)
        r2_str = f"R2={result.overall_r2:.4f}" if result else ""
        ax.set_title(
            f"{target_name} ({target_col})\n"
            f"Best val_loss={best_overall_loss:.5f} @ epoch {best_overall_epoch}"
            + (f" | {r2_str}" if r2_str else ""),
            fontsize=10,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss (MSE)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linewidth=0.5, alpha=0.5)

    # ------------------------------------------------------------------
    # DNN-specific: comparison bars (linear vs DNN)
    # ------------------------------------------------------------------

    def plot_comparison_bars(
        self,
        linear_results: Dict[str, TargetResult],
        dnn_results: Optional[Dict[str, TargetResult]] = None,
    ) -> Optional[Path]:
        """
        Side-by-side grouped bar chart comparing Linear vs DNN R2 and RMSE.

        Two subplots: (1) R2 comparison with delta labels, (2) RMSE comparison.

        Args:
            linear_results: Linear probe TargetResult dictionary.
            dnn_results: DNN probe results. Defaults to self.results.

        Returns:
            Path to saved figure, or None if no overlapping targets.
        """
        dnn_results = dnn_results or self.results

        # Find overlapping targets
        targets = [
            t for t in dnn_results if t in linear_results
        ]
        if not targets:
            logger.warning(
                "No overlapping targets between linear and DNN results, "
                "skipping comparison bars"
            )
            return None

        target_labels = [TARGET_NAMES.get(t, t) for t in targets]
        linear_r2 = [linear_results[t].overall_r2 for t in targets]
        dnn_r2 = [dnn_results[t].overall_r2 for t in targets]
        linear_rmse = [linear_results[t].overall_rmse for t in targets]
        dnn_rmse = [dnn_results[t].overall_rmse for t in targets]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(targets))
        width = 0.35

        # -- R2 subplot --
        ax = axes[0]
        bars_lin = ax.bar(
            x - width / 2, linear_r2, width,
            label="Linear", color="steelblue", edgecolor="white"
        )
        bars_dnn = ax.bar(
            x + width / 2, dnn_r2, width,
            label="DNN (GNN)", color="coral", edgecolor="white"
        )

        # Value labels on top of bars
        for bar in bars_lin:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=8,
            )
        for bar in bars_dnn:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

        # Delta labels between bar pairs
        for i in range(len(targets)):
            delta = dnn_r2[i] - linear_r2[i]
            y_pos = max(linear_r2[i], dnn_r2[i]) + 0.025
            sign = "+" if delta >= 0 else ""
            ax.text(
                x[i], y_pos, f"{sign}{delta:.3f}",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold",
                color="darkgreen" if delta >= 0 else "darkred",
            )

        ax.set_ylabel("R-squared (OOF)")
        ax.set_title("R-squared: Linear vs DNN Probe")
        ax.set_xticks(x)
        ax.set_xticklabels(target_labels, rotation=30, ha="right")
        ax.legend()
        ax.axhline(y=0, color="black", linewidth=0.5)

        # -- RMSE subplot --
        ax = axes[1]
        bars_lin = ax.bar(
            x - width / 2, linear_rmse, width,
            label="Linear", color="steelblue", edgecolor="white"
        )
        bars_dnn = ax.bar(
            x + width / 2, dnn_rmse, width,
            label="DNN (GNN)", color="coral", edgecolor="white"
        )

        for bar in bars_lin:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.002,
                f"{h:.4f}",
                ha="center", va="bottom", fontsize=8,
            )
        for bar in bars_dnn:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.002,
                f"{h:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

        # Delta labels for RMSE (negative is better)
        for i in range(len(targets)):
            delta = dnn_rmse[i] - linear_rmse[i]
            y_pos = max(linear_rmse[i], dnn_rmse[i]) + 0.005
            sign = "+" if delta >= 0 else ""
            ax.text(
                x[i], y_pos, f"{sign}{delta:.4f}",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold",
                color="darkred" if delta >= 0 else "darkgreen",
            )

        ax.set_ylabel("RMSE (OOF)")
        ax.set_title("RMSE: Linear vs DNN Probe")
        ax.set_xticks(x)
        ax.set_xticklabels(target_labels, rotation=30, ha="right")
        ax.legend()

        fig.suptitle(
            "Linear vs DNN Probe: AlphaEarth Embeddings -> Leefbaarometer\n"
            "Spatial Block Cross-Validation",
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        path = self.output_dir / "comparison_linear_vs_dnn.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved comparison bars: {path}")
        return path

    # ------------------------------------------------------------------
    # DNN-specific: comparison scatter
    # ------------------------------------------------------------------

    def plot_comparison_scatter(
        self,
        linear_results: Dict[str, TargetResult],
    ) -> Optional[Path]:
        """
        Faceted scatter: DNN predictions (x) vs Linear predictions (y),
        coloured by absolute residual magnitude.

        Shows where the two models agree vs disagree.

        Args:
            linear_results: Linear probe TargetResult dictionary.

        Returns:
            Path to saved figure, or None if no overlapping targets.
        """
        targets = [
            t for t in self.results
            if t in linear_results
            and len(self.results[t].oof_predictions) > 0
            and len(linear_results[t].oof_predictions) > 0
        ]
        if not targets:
            logger.warning(
                "No overlapping targets with predictions, "
                "skipping comparison scatter"
            )
            return None

        ncols = 3
        nrows = max(1, (len(targets) + ncols - 1) // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
        )

        for idx, tc in enumerate(targets):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            dnn_res = self.results[tc]
            lin_res = linear_results[tc]

            # Align predictions via region_id
            dnn_df = pd.DataFrame({
                "region_id": dnn_res.region_ids,
                "dnn_pred": dnn_res.oof_predictions,
                "actual": dnn_res.actual_values,
            }).set_index("region_id")
            lin_df = pd.DataFrame({
                "region_id": lin_res.region_ids,
                "lin_pred": lin_res.oof_predictions,
            }).set_index("region_id")

            merged = dnn_df.join(lin_df, how="inner").dropna()

            if len(merged) == 0:
                ax.set_visible(False)
                continue

            dnn_pred = merged["dnn_pred"].values
            lin_pred = merged["lin_pred"].values
            actual = merged["actual"].values

            # Colour by DNN absolute residual
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

            sc = ax.scatter(
                dnn_pred, lin_pred, c=abs_residual, s=2,
                alpha=alpha, cmap="viridis", rasterized=True,
            )

            # 1:1 line
            lims = [
                min(dnn_pred.min(), lin_pred.min()),
                max(dnn_pred.max(), lin_pred.max()),
            ]
            ax.plot(lims, lims, "--", color="red", linewidth=1, alpha=0.7)

            target_name = TARGET_NAMES.get(tc, tc)
            ax.set_title(f"{target_name} ({tc})", fontsize=10)
            ax.set_xlabel("DNN Predictions")
            ax.set_ylabel("Linear Predictions")
            ax.set_aspect("equal", adjustable="datalim")

            fig.colorbar(sc, ax=ax, shrink=0.7, label="|DNN residual|")

        # Hide unused axes
        for idx in range(len(targets), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle(
            "DNN vs Linear Predictions\n"
            "Colour = |DNN residual|, dashed = 1:1 line",
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        path = self.output_dir / "comparison_scatter_dnn_vs_linear.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved comparison scatter: {path}")
        return path

    # ------------------------------------------------------------------
    # DNN-specific: spatial improvement map
    # ------------------------------------------------------------------

    def plot_spatial_improvement(
        self,
        target_col: str,
        linear_results: Dict[str, TargetResult],
        geometry_source: Optional[gpd.GeoDataFrame] = None,
    ) -> Optional[Path]:
        """
        Map showing where DNN is better (blue) vs worse (red) than linear.

        Metric: |linear_residual| - |dnn_residual| per hexagon.
        Positive = DNN is better (smaller absolute error).
        Uses the same qcut+dissolve+rasterized+EPSG:28992 approach as
        plot_spatial_residuals.

        Args:
            target_col: Target variable name.
            linear_results: Linear probe TargetResult dictionary.
            geometry_source: GeoDataFrame with region_id index and geometry.

        Returns:
            Path to saved figure, or None if insufficient data.
        """
        if target_col not in linear_results:
            logger.warning(
                f"Target {target_col} not in linear results, "
                "skipping spatial improvement"
            )
            return None

        dnn_res = self.results[target_col]
        lin_res = linear_results[target_col]

        # Build per-hexagon comparison dataframe
        dnn_df = pd.DataFrame({
            "region_id": dnn_res.region_ids,
            "dnn_pred": dnn_res.oof_predictions,
            "actual": dnn_res.actual_values,
        }).set_index("region_id")
        dnn_df = dnn_df.dropna()
        dnn_df["dnn_abs_resid"] = np.abs(dnn_df["actual"] - dnn_df["dnn_pred"])

        lin_df = pd.DataFrame({
            "region_id": lin_res.region_ids,
            "lin_pred": lin_res.oof_predictions,
            "lin_actual": lin_res.actual_values,
        }).set_index("region_id")
        lin_df = lin_df.dropna()
        lin_df["lin_abs_resid"] = np.abs(lin_df["lin_actual"] - lin_df["lin_pred"])

        merged = dnn_df[["dnn_abs_resid"]].join(
            lin_df[["lin_abs_resid"]], how="inner"
        )

        if len(merged) == 0:
            logger.warning(
                f"No overlapping hexagons for {target_col}, "
                "skipping spatial improvement"
            )
            return None

        # Improvement = |linear_resid| - |dnn_resid|
        # Positive means DNN is better, negative means linear is better
        merged["improvement"] = merged["lin_abs_resid"] - merged["dnn_abs_resid"]

        # Get geometry
        if geometry_source is not None:
            improve_gdf = gpd.GeoDataFrame(
                merged.join(geometry_source[["geometry"]]),
                crs="EPSG:4326",
            )
        else:
            from srai.h3 import h3_to_geoseries

            geom = h3_to_geoseries(merged.index)
            geom.index = merged.index
            improve_gdf = gpd.GeoDataFrame(
                merged, geometry=geom, crs="EPSG:4326"
            )

        improve_gdf = improve_gdf.dropna(subset=["geometry"])
        n_hexagons = len(improve_gdf)

        if n_hexagons == 0:
            logger.warning(
                f"No hexagons with geometry for {target_col}, "
                "skipping spatial improvement"
            )
            return None

        # Reproject to RD New
        improve_gdf = improve_gdf.to_crs(epsg=28992)

        # Quantile-bin dissolve
        n_bins = 20
        improve_gdf["improve_bin"] = pd.qcut(
            improve_gdf["improvement"],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        dissolved = improve_gdf.dissolve(by="improve_bin", aggfunc="mean")

        logger.info(
            f"  Dissolved {n_hexagons:,} hexagons into "
            f"{len(dissolved)} improvement bins"
        )

        fig, ax = plt.subplots(figsize=(12, 14))
        fig.set_facecolor("white")
        ax.set_facecolor("white")

        # Diverging colormap: blue = DNN better, red = linear better
        vmax = improve_gdf["improvement"].abs().quantile(0.98)
        if vmax == 0:
            vmax = 1.0

        dissolved.plot(
            column="improvement",
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
            ax=ax,
            legend=True,
            legend_kwds={
                "shrink": 0.7,
                "label": "|Linear resid| - |DNN resid|\n(+ = DNN better)",
            },
            edgecolor="none",
            rasterized=True,
        )

        ax.grid(True, linewidth=0.5, alpha=0.5, color="gray")
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

        # Cartographic elements
        self._add_scale_bar(ax, length_km=50)
        self._add_north_arrow(ax)

        # Statistics
        n_dnn_better = int((improve_gdf["improvement"] > 0).sum())
        n_lin_better = int((improve_gdf["improvement"] < 0).sum())
        mean_improvement = float(improve_gdf["improvement"].mean())

        target_name = TARGET_NAMES.get(target_col, target_col)
        title_lines = [
            f"Spatial Improvement: DNN vs Linear -- {target_name} ({target_col})",
            f"Blue = DNN better ({n_dnn_better:,} hex) | "
            f"Red = Linear better ({n_lin_better:,} hex)",
            f"Mean improvement: {mean_improvement:+.5f} | "
            f"n={n_hexagons:,} hexagons ({len(dissolved)} dissolved bins) | "
            f"EPSG:28992",
        ]
        ax.set_title("\n".join(title_lines), fontsize=11, pad=10)

        plt.tight_layout()
        path = self.output_dir / f"spatial_improvement_{target_col}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved spatial improvement map: {path}")
        return path

    # ------------------------------------------------------------------
    # Override plot_all
    # ------------------------------------------------------------------

    def plot_all(
        self,
        geometry_source: Optional[gpd.GeoDataFrame] = None,
        linear_results: Optional[Dict[str, TargetResult]] = None,
        training_curves: Optional[Dict[str, Dict[int, List[float]]]] = None,
        embeddings_path: Optional[Path] = None,
        boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> List[Path]:
        """
        Generate all DNN probe visualizations.

        Runs inherited prediction-based plots, DNN-specific training curves,
        and (if linear_results provided) comparison plots.

        Args:
            geometry_source: GeoDataFrame for spatial maps.
            linear_results: Linear probe results for comparison plots.
            training_curves: Override training curves (uses self.training_curves
                if None).
            embeddings_path: Not used by DNN visualizer (kept for API compat).
            boundary_gdf: Not used by DNN visualizer (kept for API compat).

        Returns:
            List of paths to all saved figures.
        """
        if training_curves is not None:
            self.training_curves = training_curves

        logger.info(f"Generating DNN probe visualizations to {self.output_dir}")
        paths: List[Path] = []

        # -- Per-target plots (inherited, work correctly) --
        for target_col in self.results:
            paths.append(
                self.plot_scatter_predicted_vs_actual(target_col)
            )
            paths.append(
                self.plot_spatial_residuals(target_col, geometry_source)
            )

        # -- Cross-target plots (inherited) --
        fold_path = self.plot_fold_metrics()
        if fold_path is not None:
            paths.append(fold_path)

        paths.append(self.plot_metrics_comparison())

        # -- DNN-specific: training curves --
        curves_path = self.plot_training_curves()
        if curves_path is not None:
            paths.append(curves_path)

        # Also individual target curves
        for target_col in self.results:
            tc_path = self.plot_training_curves(target_col)
            if tc_path is not None:
                paths.append(tc_path)

        # -- Comparison plots (require linear_results) --
        if linear_results is not None:
            comp_bars = self.plot_comparison_bars(linear_results)
            if comp_bars is not None:
                paths.append(comp_bars)

            comp_scatter = self.plot_comparison_scatter(linear_results)
            if comp_scatter is not None:
                paths.append(comp_scatter)

            # Spatial improvement per target
            for target_col in self.results:
                imp_path = self.plot_spatial_improvement(
                    target_col, linear_results, geometry_source
                )
                if imp_path is not None:
                    paths.append(imp_path)

        # Filter out None values
        paths = [p for p in paths if p is not None]

        logger.info(f"Generated {len(paths)} DNN probe visualizations")
        return paths
