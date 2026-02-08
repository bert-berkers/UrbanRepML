#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization and Interpretability for Linear Probe Results

Creates publication-quality visualizations of ElasticNet linear probe
regression results: coefficient analysis, prediction quality, and
spatial residual maps.

Visualizations:
    1. Coefficient bar chart per target (diverging colormap, top-N)
    2. Faceted coefficient comparison across all 6 indicators
    3. Coefficient heatmap: features x targets
    4. 1:1 scatter plots: predicted vs actual per target
    5. Spatial maps: predictions and residuals on hex grid
    6. RGB spatial map of top-3 coefficient dimensions per indicator
    7. Cross-target correlation of coefficient patterns
    8. Performance comparison: full vs PCA-16
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
from matplotlib.colors import TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .linear_probe import LinearProbeRegressor, TargetResult, TARGET_NAMES

logger = logging.getLogger(__name__)


class LinearProbeVisualizer:
    """
    Creates visualizations for linear probe regression results.

    All plots are saved to the output directory as high-resolution PNGs.
    """

    def __init__(
        self,
        results: Dict[str, TargetResult],
        output_dir: Path,
        figsize_base: Tuple[float, float] = (10, 6),
        dpi: int = 150,
    ):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize_base = figsize_base
        self.dpi = dpi

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
        })

    def plot_coefficient_bars(self, target_col: str, top_n: int = 15) -> Path:
        """
        Horizontal bar chart of coefficients for a single target.

        Uses diverging red-white-blue colormap with top-N coefficients
        highlighted, following the GEE_MediumBlog_Logic style.

        Args:
            target_col: Target variable name.
            top_n: Number of top coefficients to label.

        Returns:
            Path to saved figure.
        """
        result = self.results[target_col]
        coefs = result.coefficients
        names = result.feature_names

        # Natural A00→A63 order (ascending by feature name)
        natural_idx = sorted(range(len(names)), key=lambda i: names[i])
        ordered_coefs = coefs[natural_idx]
        ordered_names = [names[i] for i in natural_idx]

        # Diverging colormap
        vmax = max(abs(ordered_coefs.min()), abs(ordered_coefs.max()))
        if vmax == 0:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = cm.RdBu_r
        colors = [cmap(norm(c)) for c in ordered_coefs]

        fig, ax = plt.subplots(figsize=(self.figsize_base[0], max(6, len(names) * 0.25)))

        bars = ax.barh(range(len(ordered_coefs)), ordered_coefs, color=colors, edgecolor="none")

        # Highlight top-N by absolute value
        abs_coefs = np.abs(ordered_coefs)
        top_threshold = np.sort(abs_coefs)[-min(top_n, len(abs_coefs))]
        for i in range(len(ordered_coefs)):
            if abs_coefs[i] >= top_threshold:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(1.0)

        ax.set_yticks(range(len(ordered_names)))
        ax.set_yticklabels(ordered_names, fontsize=8)
        ax.set_xlabel("Coefficient Value")

        # Adapt title based on mode (simple vs elasticnet)
        if result.best_alpha == 0.0 and result.best_l1_ratio == 0.0:
            # Simple mode
            title = (f"Linear Regression Coefficients: {result.target_name} ({target_col})\n"
                    f"R2={result.overall_r2:.4f}")
        else:
            # ElasticNet mode
            title = (f"ElasticNet Coefficients: {result.target_name} ({target_col})\n"
                    f"R2={result.overall_r2:.4f} | alpha={result.best_alpha:.4f} | "
                    f"l1_ratio={result.best_l1_ratio:.3f}")
        ax.set_title(title)
        ax.axvline(x=0, color="black", linewidth=0.5)

        plt.tight_layout()
        path = self.output_dir / f"coefficients_{target_col}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved coefficient bar chart: {path}")
        return path

    def plot_coefficient_bars_faceted(self) -> Optional[Path]:
        """
        Faceted horizontal bar chart comparing ALL coefficients across all targets.

        2x3 subplot grid with shared x-axis range so panels are directly comparable.
        Features displayed in natural A00→A63 order for cross-target comparison.
        Requires more than one target result.

        Returns:
            Path to saved figure.
        """
        if len(self.results) < 2:
            logger.warning("Faceted coefficient plot requires >1 target, skipping")
            return None

        targets = list(self.results.keys())
        n_targets = len(targets)
        ncols = 3
        nrows = (n_targets + ncols - 1) // ncols

        # Global vmax for shared x-axis
        global_vmax = 0.0
        for result in self.results.values():
            vmax = np.max(np.abs(result.coefficients))
            if vmax > global_vmax:
                global_vmax = vmax
        if global_vmax == 0:
            global_vmax = 1.0

        # Taller panels to fit all 64 features
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10 * nrows))
        axes = np.atleast_2d(axes)

        norm = TwoSlopeNorm(vmin=-global_vmax, vcenter=0, vmax=global_vmax)
        cmap = cm.RdBu_r

        for idx, target_col in enumerate(targets):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            result = self.results[target_col]
            coefs = result.coefficients
            names = result.feature_names

            # Natural A00→A63 order
            natural_idx = sorted(range(len(names)), key=lambda i: names[i])
            ordered_coefs = coefs[natural_idx]
            ordered_names = [names[i] for i in natural_idx]

            colors = [cmap(norm(c)) for c in ordered_coefs]
            ax.barh(range(len(ordered_coefs)), ordered_coefs, color=colors, edgecolor="none")

            ax.set_yticks(range(len(ordered_names)))
            ax.set_yticklabels(ordered_names, fontsize=6)
            ax.set_xlim(-global_vmax * 1.05, global_vmax * 1.05)
            ax.axvline(x=0, color="black", linewidth=0.5)
            ax.set_title(f"{result.target_name} ({target_col})\nR²={result.overall_r2:.4f}",
                         fontsize=11)
            if col == 0:
                ax.set_ylabel("Feature")
            if row == nrows - 1:
                ax.set_xlabel("Coefficient")

        # Hide unused axes
        for idx in range(n_targets, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        # Adapt suptitle based on mode
        first_result = self.results[targets[0]]
        if first_result.best_alpha == 0.0 and first_result.best_l1_ratio == 0.0:
            mode_str = "Linear Regression"
        else:
            mode_str = "ElasticNet"

        fig.suptitle(
            f"{mode_str} Coefficients: All Features per Leefbaarometer Indicator\n"
            f"A00→A63 order | Shared x-axis for direct comparison",
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        path = self.output_dir / "coefficients_faceted.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved faceted coefficient bar chart: {path}")
        return path

    def plot_coefficient_heatmap(self) -> Path:
        """
        Heatmap of coefficients: features (rows) x targets (columns).

        Features are sorted by mean absolute coefficient across targets.

        Returns:
            Path to saved figure.
        """
        targets = list(self.results.keys())
        first_result = self.results[targets[0]]
        feature_names = first_result.feature_names
        n_features = len(feature_names)

        # Build coefficient matrix
        coef_matrix = np.zeros((n_features, len(targets)))
        for j, target_col in enumerate(targets):
            coef_matrix[:, j] = self.results[target_col].coefficients

        # Natural A00→A63 order for consistent cross-target comparison
        natural_idx = sorted(range(len(feature_names)), key=lambda i: feature_names[i])
        coef_matrix = coef_matrix[natural_idx]
        sorted_names = [feature_names[i] for i in natural_idx]

        # Target display names
        target_labels = [TARGET_NAMES.get(t, t) for t in targets]

        fig, ax = plt.subplots(figsize=(8, max(6, n_features * 0.25)))

        vmax = np.max(np.abs(coef_matrix))
        if vmax == 0:
            vmax = 1.0

        sns.heatmap(
            coef_matrix,
            xticklabels=target_labels,
            yticklabels=sorted_names,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            annot=n_features <= 20,
            fmt=".3f" if n_features <= 20 else "",
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title("Coefficient Heatmap\n"
                      "Features (A00→A63) x Leefbaarometer Targets")
        ax.set_ylabel("Embedding Feature")
        ax.set_xlabel("Target Variable")

        plt.tight_layout()
        path = self.output_dir / "coefficient_heatmap.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved coefficient heatmap: {path}")
        return path

    def plot_scatter_predicted_vs_actual(self, target_col: str) -> Path:
        """
        1:1 scatter plot: predicted vs actual with best-fit line.

        Args:
            target_col: Target variable name.

        Returns:
            Path to saved figure.
        """
        result = self.results[target_col]
        actual = result.actual_values
        predicted = result.oof_predictions

        # Remove NaN predictions
        valid = ~np.isnan(predicted)
        actual = actual[valid]
        predicted = predicted[valid]

        fig, ax = plt.subplots(figsize=self.figsize_base)

        # Subsample if too many points for visibility
        n_points = len(actual)
        if n_points > 50000:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_points, size=50000, replace=False)
            actual_plot = actual[idx]
            predicted_plot = predicted[idx]
            alpha = 0.05
        elif n_points > 10000:
            actual_plot = actual
            predicted_plot = predicted
            alpha = 0.1
        else:
            actual_plot = actual
            predicted_plot = predicted
            alpha = 0.3

        ax.scatter(actual_plot, predicted_plot, s=2, alpha=alpha, c="steelblue", rasterized=True)

        # 1:1 line
        lims = [min(actual.min(), predicted.min()),
                max(actual.max(), predicted.max())]
        ax.plot(lims, lims, "--", color="red", linewidth=1, label="1:1 line")

        # Best-fit line
        z = np.polyfit(actual, predicted, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, p(x_fit), "-", color="darkgreen", linewidth=1.5,
                label=f"Fit: y={z[0]:.3f}x + {z[1]:.3f}")

        ax.set_xlabel(f"Actual {result.target_name}")
        ax.set_ylabel(f"Predicted {result.target_name}")
        ax.set_title(f"Predicted vs Actual: {result.target_name} ({target_col})\n"
                     f"R2={result.overall_r2:.4f}, RMSE={result.overall_rmse:.4f}, "
                     f"n={n_points:,}")
        ax.legend(loc="upper left")
        ax.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()
        path = self.output_dir / f"scatter_{target_col}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved scatter plot: {path}")
        return path

    def plot_metrics_comparison(
        self,
        results_full: Optional[Dict[str, TargetResult]] = None,
        results_pca: Optional[Dict[str, TargetResult]] = None,
    ) -> Path:
        """
        Bar chart comparing R2 across targets and embedding types.

        If only self.results is available, plots a single set.
        If both full and PCA results are provided, plots side-by-side.

        Returns:
            Path to saved figure.
        """
        results_full = results_full or self.results
        targets = list(results_full.keys())
        target_labels = [TARGET_NAMES.get(t, t) for t in targets]

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(targets))
        width = 0.35

        r2_full = [results_full[t].overall_r2 for t in targets]
        bars1 = ax.bar(x - width / 2 if results_pca else x, r2_full,
                       width if results_pca else width * 1.5,
                       label="Full 64-dim", color="steelblue", edgecolor="white")

        if results_pca:
            r2_pca = [results_pca[t].overall_r2 for t in targets]
            bars2 = ax.bar(x + width / 2, r2_pca, width,
                           label="PCA-16", color="coral", edgecolor="white")

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=9)
        if results_pca:
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                        f"{height:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("R-squared (OOF)")
        ax.set_title("Linear Probe Performance: AlphaEarth -> Leefbaarometer\n"
                      "Spatial Block Cross-Validation")
        ax.set_xticks(x)
        ax.set_xticklabels(target_labels, rotation=30, ha="right")
        ax.legend()
        ax.axhline(y=0, color="black", linewidth=0.5)

        plt.tight_layout()
        path = self.output_dir / "metrics_comparison.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved metrics comparison: {path}")
        return path

    def plot_cross_target_correlation(self) -> Path:
        """
        Correlation matrix of coefficient patterns across targets.

        If two targets have similar coefficient patterns, their embeddings
        are used in a similar way, suggesting shared underlying factors.

        Returns:
            Path to saved figure.
        """
        targets = list(self.results.keys())
        target_labels = [TARGET_NAMES.get(t, t) for t in targets]

        # Build coefficient matrix
        coef_matrix = np.column_stack(
            [self.results[t].coefficients for t in targets]
        )

        # Compute correlation
        corr = np.corrcoef(coef_matrix.T)

        fig, ax = plt.subplots(figsize=(7, 6))

        sns.heatmap(
            corr,
            xticklabels=target_labels,
            yticklabels=target_labels,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title("Cross-Target Coefficient Correlation\n"
                      "How similarly each target uses the embedding features")

        plt.tight_layout()
        path = self.output_dir / "cross_target_correlation.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved cross-target correlation: {path}")
        return path

    def plot_spatial_residuals(
        self,
        target_col: str,
        geometry_source: Optional[gpd.GeoDataFrame] = None,
    ) -> Path:
        """
        Spatial map of predictions and residuals on hex grid.

        Uses quantile-bin dissolve to merge hexagons into ~20 groups for
        efficient rendering while showing all data.

        Args:
            target_col: Target variable name.
            geometry_source: GeoDataFrame with region_id index and geometry.
                             If None, creates geometry from H3 hex boundaries.

        Returns:
            Path to saved figure.
        """
        result = self.results[target_col]

        # Build residual dataframe
        valid = ~np.isnan(result.oof_predictions)
        residual_df = pd.DataFrame({
            "region_id": result.region_ids[valid],
            "actual": result.actual_values[valid],
            "predicted": result.oof_predictions[valid],
            "residual": result.actual_values[valid] - result.oof_predictions[valid],
        }).set_index("region_id")

        # Get geometry
        if geometry_source is not None:
            residual_gdf = gpd.GeoDataFrame(
                residual_df.join(geometry_source[["geometry"]]),
                crs="EPSG:4326",
            )
        else:
            # Create geometry from H3 hex boundaries
            from srai.h3 import h3_to_geoseries
            geom = h3_to_geoseries(residual_df.index)
            geom.index = residual_df.index
            residual_gdf = gpd.GeoDataFrame(
                residual_df, geometry=geom, crs="EPSG:4326"
            )

        residual_gdf = residual_gdf.dropna(subset=["geometry"])
        n_hexagons = len(residual_gdf)

        # Quantile-bin dissolve for efficient rendering
        n_bins = 20
        residual_gdf["pred_bin"] = pd.qcut(
            residual_gdf["predicted"], q=n_bins, labels=False, duplicates="drop"
        )
        residual_gdf["resid_bin"] = pd.qcut(
            residual_gdf["residual"], q=n_bins, labels=False, duplicates="drop"
        )

        # Dissolve by bin — merge hexagons in each quantile group
        pred_dissolved = residual_gdf.dissolve(by="pred_bin", aggfunc="mean")
        resid_dissolved = residual_gdf.dissolve(by="resid_bin", aggfunc="mean")

        logger.info(f"  Dissolved {n_hexagons:,} hexagons into "
                     f"{len(pred_dissolved)} predicted / {len(resid_dissolved)} residual polygons")

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.set_facecolor("white")

        # Predicted values map
        vmin_pred = residual_gdf["predicted"].quantile(0.02)
        vmax_pred = residual_gdf["predicted"].quantile(0.98)
        pred_dissolved.plot(
            column="predicted",
            cmap="viridis",
            vmin=vmin_pred,
            vmax=vmax_pred,
            ax=axes[0],
            legend=True,
            legend_kwds={"shrink": 0.7, "label": result.target_name},
            edgecolor="none",
            rasterized=True,
        )
        axes[0].set_title(f"Predicted {result.target_name}")

        # Residual map
        vmax_res = residual_gdf["residual"].abs().quantile(0.98)
        resid_dissolved.plot(
            column="residual",
            cmap="RdBu_r",
            vmin=-vmax_res,
            vmax=vmax_res,
            ax=axes[1],
            legend=True,
            legend_kwds={"shrink": 0.7, "label": "Residual (actual - predicted)"},
            edgecolor="none",
            rasterized=True,
        )
        axes[1].set_title(f"Residuals: {result.target_name}")

        # White background, lat/lon gridlines, axis ticks
        for ax in axes:
            ax.set_facecolor("white")
            ax.grid(True, linewidth=0.5, alpha=0.5, color="gray")
            ax.tick_params(labelsize=8)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        fig.suptitle(
            f"Spatial Maps: {result.target_name} ({target_col})\n"
            f"R2={result.overall_r2:.4f} | n={n_hexagons:,} hexagons "
            f"({n_bins} quantile bins)",
            fontsize=14,
        )

        plt.tight_layout()
        path = self.output_dir / f"spatial_map_{target_col}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved spatial residual map: {path}")
        return path

    # ------------------------------------------------------------------
    # Cartographic helpers
    # ------------------------------------------------------------------

    def _add_scale_bar(self, ax, length_km: int = 50):
        """
        Add a scale bar to a map axis in EPSG:28992 (RD New) coordinate space.

        Since EPSG:28992 units are meters, length_km * 1000 gives the bar
        length in axes units directly.

        Args:
            ax: Matplotlib axes (with EPSG:28992 data plotted).
            length_km: Scale bar length in kilometers.
        """
        bar_length = length_km * 1000  # meters in RD New

        # Position in lower-left of axes data range
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.04
        tick_height = (ylim[1] - ylim[0]) * 0.012

        # Main bar
        ax.plot([x0, x0 + bar_length], [y0, y0], color="black", linewidth=2,
                solid_capstyle="butt", transform=ax.transData)

        # Tick marks at ends
        ax.plot([x0, x0], [y0 - tick_height, y0 + tick_height],
                color="black", linewidth=1.5, transform=ax.transData)
        ax.plot([x0 + bar_length, x0 + bar_length],
                [y0 - tick_height, y0 + tick_height],
                color="black", linewidth=1.5, transform=ax.transData)

        # Label
        ax.text(x0 + bar_length / 2, y0 + tick_height * 1.8,
                f"{length_km} km", ha="center", va="bottom",
                fontsize=9, fontweight="bold", transform=ax.transData)

    def _add_north_arrow(self, ax):
        """
        Add a north arrow with 'N' label to the upper-right corner.

        Args:
            ax: Matplotlib axes.
        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.06
        y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.06
        arrow_length = (ylim[1] - ylim[0]) * 0.06

        ax.annotate(
            "",
            xy=(x_pos, y_pos),
            xytext=(x_pos, y_pos - arrow_length),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
            transform=ax.transData,
        )
        ax.text(x_pos, y_pos + arrow_length * 0.2, "N",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                transform=ax.transData)

    # ------------------------------------------------------------------
    # RGB top-3 coefficient map
    # ------------------------------------------------------------------

    def plot_rgb_top3_map(
        self,
        target_col: str,
        embeddings_path: Optional[Path] = None,
        boundary_gdf: Optional[gpd.GeoDataFrame] = None,
        max_hexagons: int = 600_000,
    ) -> Optional[Path]:
        """
        Spatial RGB map where R, G, B channels are the top-3 embedding
        dimensions by absolute coefficient magnitude for a given target.

        Each channel is percentile-normalized and sign-corrected so that
        brighter values always mean 'more contribution to prediction',
        regardless of whether the coefficient is positive or negative.

        Args:
            target_col: Target variable name.
            embeddings_path: Path to embeddings parquet with h3_index column.
                If None, tries default AlphaEarth path.
            boundary_gdf: Optional study-area boundary for background context.
            max_hexagons: Maximum hexagons to plot (subsampled if exceeded).

        Returns:
            Path to saved figure, or None if fewer than 3 features.
        """
        result = self.results[target_col]
        coefs = result.coefficients
        names = result.feature_names

        if len(coefs) < 3:
            logger.warning(f"RGB top-3 map requires >=3 features, got {len(coefs)}, skipping {target_col}")
            return None

        # ----------------------------------------------------------
        # 1. Select top-3 features by absolute coefficient magnitude
        # ----------------------------------------------------------
        top3_idx = np.argsort(np.abs(coefs))[-3:]  # ascending, last 3 are largest
        # Order: largest absolute coef = R, second = G, third = B
        top3_idx = top3_idx[::-1]
        channel_names = [names[i] for i in top3_idx]
        channel_coefs = coefs[top3_idx]

        logger.info(f"RGB top-3 for {target_col}: "
                     f"R={channel_names[0]} (coef={channel_coefs[0]:.4f}), "
                     f"G={channel_names[1]} (coef={channel_coefs[1]:.4f}), "
                     f"B={channel_names[2]} (coef={channel_coefs[2]:.4f})")

        # ----------------------------------------------------------
        # 2. Load only needed columns from embeddings parquet
        # ----------------------------------------------------------
        if embeddings_path is None:
            embeddings_path = (
                self.output_dir.parent.parent
                / "embeddings" / "alphaearth"
                / "netherlands_res10_2022.parquet"
            )

        embeddings_path = Path(embeddings_path)
        if not embeddings_path.exists():
            logger.error(f"Embeddings file not found: {embeddings_path}")
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

        columns_to_load = ["h3_index"] + channel_names
        emb_df = pd.read_parquet(embeddings_path, columns=columns_to_load)
        emb_df = emb_df.set_index("h3_index")
        emb_df.index.name = "region_id"

        # ----------------------------------------------------------
        # 3. Filter to prediction hexagons (those with target data)
        # ----------------------------------------------------------
        valid_ids = set(result.region_ids)
        emb_df = emb_df.loc[emb_df.index.isin(valid_ids)]
        logger.info(f"  Filtered to {len(emb_df):,} hexagons with target data")

        if len(emb_df) == 0:
            logger.error("No overlapping hexagons between embeddings and predictions")
            raise ValueError("No overlapping hexagons between embeddings and predictions")

        # Subsample if too large
        if len(emb_df) > max_hexagons:
            emb_df = emb_df.sample(n=max_hexagons, random_state=42)
            logger.info(f"  Subsampled to {max_hexagons:,} hexagons")

        # ----------------------------------------------------------
        # 4. Robust percentile normalization per channel
        # ----------------------------------------------------------
        rgb_array = np.zeros((len(emb_df), 3), dtype=np.float64)

        for ch_idx, col_name in enumerate(channel_names):
            vals = emb_df[col_name].values.astype(np.float64)
            p2, p98 = np.percentile(vals, [2, 98])

            # Clip and scale to [0, 1]
            if p98 - p2 > 0:
                normalized = np.clip((vals - p2) / (p98 - p2), 0.0, 1.0)
            else:
                normalized = np.full_like(vals, 0.5)

            # ----------------------------------------------------------
            # 5. Coefficient sign handling: invert if negative so
            #    brighter = more contribution regardless of direction
            # ----------------------------------------------------------
            if channel_coefs[ch_idx] < 0:
                normalized = 1.0 - normalized

            rgb_array[:, ch_idx] = normalized

        # ----------------------------------------------------------
        # 6. Build geometry from H3 hex IDs
        # ----------------------------------------------------------
        from srai.h3 import h3_to_geoseries

        hex_geom = h3_to_geoseries(pd.Index(emb_df.index, name="region_id"))
        hex_geom.index = emb_df.index
        plot_gdf = gpd.GeoDataFrame(
            emb_df,
            geometry=hex_geom,
            crs="EPSG:4326",
        )

        # Reproject to RD New for cartographic elements
        plot_gdf = plot_gdf.to_crs(epsg=28992)

        # ----------------------------------------------------------
        # 7. Load boundary if not provided
        # ----------------------------------------------------------
        if boundary_gdf is None:
            boundary_path = (
                self.output_dir.parent.parent.parent
                / "boundaries" / "netherlands_boundary.geojson"
            )
            if boundary_path.exists():
                boundary_gdf = gpd.read_file(boundary_path)
                logger.info(f"  Loaded boundary from {boundary_path}")

        if boundary_gdf is not None:
            if boundary_gdf.crs is None:
                boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
            boundary_gdf = boundary_gdf.to_crs(epsg=28992)

        # ----------------------------------------------------------
        # 8. Plot
        # ----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 14), dpi=self.dpi)

        # Background boundary
        if boundary_gdf is not None:
            boundary_gdf.plot(
                ax=ax,
                facecolor="#f0f0f0",
                edgecolor="#cccccc",
                linewidth=0.5,
            )

        # Hex layer with RGB colors
        plot_gdf.plot(
            ax=ax,
            color=rgb_array,
            edgecolor="none",
            linewidth=0,
            rasterized=True,
        )

        ax.set_axis_off()

        # Cartographic elements
        self._add_scale_bar(ax, length_km=50)
        self._add_north_arrow(ax)

        # ----------------------------------------------------------
        # Mini-colorbars for each channel
        # ----------------------------------------------------------
        channel_cmaps = ["Reds", "Greens", "Blues"]
        channel_labels = ["R", "G", "B"]
        for ch_idx in range(3):
            # Inset axes on right side, stacked vertically
            cbar_ax = inset_axes(
                ax,
                width="2%",
                height="12%",
                loc="lower right",
                bbox_to_anchor=(0.0, 0.02 + ch_idx * 0.14, 1.0, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=1.5,
            )
            sm = cm.ScalarMappable(
                cmap=plt.get_cmap(channel_cmaps[ch_idx]),
                norm=Normalize(vmin=0, vmax=1),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            sign_str = "+" if channel_coefs[ch_idx] >= 0 else "-"
            cbar.set_label(
                f"{channel_labels[ch_idx]}: {channel_names[ch_idx]} "
                f"({sign_str}{abs(channel_coefs[ch_idx]):.3f})",
                fontsize=8,
            )
            cbar.ax.tick_params(labelsize=7)

        # ----------------------------------------------------------
        # Interpretation text box
        # ----------------------------------------------------------
        interpretation = (
            f"Similar colors = similar embedding representation\n"
            f"in the 3 dimensions most predictive of "
            f"{result.target_name.lower()}"
        )
        ax.text(
            0.02, 0.98, interpretation,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )

        # ----------------------------------------------------------
        # Title
        # ----------------------------------------------------------
        title_lines = [
            f"RGB Top-3 Coefficient Map: {result.target_name} ({target_col})",
            f"R={channel_names[0]} ({channel_coefs[0]:+.4f})  "
            f"G={channel_names[1]} ({channel_coefs[1]:+.4f})  "
            f"B={channel_names[2]} ({channel_coefs[2]:+.4f})",
            f"Percentile [2, 98] normalization | EPSG:28992 (RD New) | "
            f"n={len(plot_gdf):,} hexagons",
        ]
        ax.set_title("\n".join(title_lines), fontsize=11, pad=10)

        plt.tight_layout()
        path = self.output_dir / f"rgb_top3_{target_col}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved RGB top-3 map: {path}")
        return path

    def plot_fold_metrics(self) -> Optional[Path]:
        """
        Box plot of per-fold R2 across targets.

        Shows the spread of performance across spatial folds.
        Returns None if no fold metrics exist (e.g., simple mode).

        Returns:
            Path to saved figure, or None if no fold metrics.
        """
        rows = []
        for target_col, result in self.results.items():
            for fm in result.fold_metrics:
                rows.append({
                    "target": TARGET_NAMES.get(target_col, target_col),
                    "fold": fm.fold,
                    "R2": fm.r2,
                    "RMSE": fm.rmse,
                })

        if not rows:
            logger.warning("No fold metrics to plot (simple mode?), skipping fold_metrics plot")
            return None

        metrics_df = pd.DataFrame(rows)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # R2 by target
        sns.boxplot(data=metrics_df, x="target", y="R2", ax=axes[0],
                    palette="Set2", hue="target", legend=False)
        sns.stripplot(data=metrics_df, x="target", y="R2", ax=axes[0],
                      color="black", size=5, alpha=0.7)
        axes[0].set_title("R-squared by Spatial Fold")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
        axes[0].axhline(y=0, color="red", linewidth=0.5, linestyle="--")

        # RMSE by target
        sns.boxplot(data=metrics_df, x="target", y="RMSE", ax=axes[1],
                    palette="Set2", hue="target", legend=False)
        sns.stripplot(data=metrics_df, x="target", y="RMSE", ax=axes[1],
                      color="black", size=5, alpha=0.7)
        axes[1].set_title("RMSE by Spatial Fold")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")

        fig.suptitle("Linear Probe: Per-Fold Spatial CV Metrics", fontsize=14)
        plt.tight_layout()

        path = self.output_dir / "fold_metrics.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved fold metrics plot: {path}")
        return path

    def plot_all(
        self,
        geometry_source: Optional[gpd.GeoDataFrame] = None,
        results_pca: Optional[Dict[str, TargetResult]] = None,
        embeddings_path: Optional[Path] = None,
        boundary_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> List[Path]:
        """
        Generate all visualizations.

        Args:
            geometry_source: GeoDataFrame for spatial maps.
            results_pca: Optional PCA results for comparison.
            embeddings_path: Path to embeddings parquet for RGB top-3 maps.
            boundary_gdf: Optional study-area boundary for map backgrounds.

        Returns:
            List of paths to all saved figures.
        """
        logger.info(f"Generating all visualizations to {self.output_dir}")
        paths = []

        # Per-target plots
        for target_col in self.results:
            paths.append(self.plot_coefficient_bars(target_col))
            paths.append(self.plot_scatter_predicted_vs_actual(target_col))
            paths.append(self.plot_spatial_residuals(target_col, geometry_source))
            if embeddings_path is not None:
                paths.append(self.plot_rgb_top3_map(
                    target_col, embeddings_path, boundary_gdf
                ))

        # Cross-target plots
        paths.append(self.plot_coefficient_heatmap())
        paths.append(self.plot_cross_target_correlation())
        fold_metrics_path = self.plot_fold_metrics()
        if fold_metrics_path is not None:
            paths.append(fold_metrics_path)
        paths.append(self.plot_metrics_comparison(
            results_full=self.results, results_pca=results_pca
        ))
        if len(self.results) > 1:
            paths.append(self.plot_coefficient_bars_faceted())

        # Filter out None values (e.g., from skipped plots)
        paths = [p for p in paths if p is not None]

        logger.info(f"Generated {len(paths)} visualizations")
        return paths
