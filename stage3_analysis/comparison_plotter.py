"""Cross-approach probe comparison visualization.

Globs standardized probe result parquets from ``probe_results/{approach}/``
and generates comparison plots: R2 bar charts, scatter facets, and spatial
residual maps.

Lifetime: durable
Stage: 3
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm, Normalize
import numpy as np
import pandas as pd
import seaborn as sns

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import (
    load_boundary,
    plot_spatial_map,
    rasterize_continuous,
    RASTER_W,
    RASTER_H,
)

logger = logging.getLogger(__name__)

# Tab10-derived palette for up to 10 approaches; extend with tab20 if needed.
APPROACH_COLORS = list(plt.get_cmap("tab10").colors)


class ProbeComparisonPlotter:
    """Generate comparison visualizations across multiple probe approaches.

    Reads standardized parquets (``predictions.parquet``, ``metrics.parquet``)
    written by ``ProbeResultsWriter`` under each approach subdirectory.

    Args:
        study_area: Name of the study area (e.g. ``"netherlands"``).
        output_dir: Override for output directory.  Defaults to
            ``probe_results/comparison/``.
        h3_resolution: H3 resolution of the embeddings.
        dpi: Output image DPI.
    """

    def __init__(
        self,
        study_area: str,
        output_dir: Optional[Path] = None,
        h3_resolution: int = 9,
        dpi: int = 150,
    ):
        self.paths = StudyAreaPaths(study_area)
        self.study_area = study_area
        self.db = SpatialDB.for_study_area(study_area)
        self.h3_resolution = h3_resolution
        self.dpi = dpi
        self.output_dir = output_dir or (self.paths.probe_results_root() / "comparison")
        self._stamp = max(1, 11 - self.h3_resolution)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_all(
        self, approaches: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load predictions and metrics from all (or selected) approaches.

        Globs ``probe_results/*/predictions.parquet`` and
        ``probe_results/*/metrics.parquet``, concatenates into single
        DataFrames, and optionally filters to the given *approaches*.

        Returns:
            ``(predictions_df, metrics_df)`` with an ``approach`` column
            identifying the source.
        """
        root = self.paths.probe_results_root()
        pred_files = sorted(root.glob("*/predictions.parquet"))
        met_files = sorted(root.glob("*/metrics.parquet"))

        if not pred_files:
            raise FileNotFoundError(
                f"No predictions.parquet found under {root}/*/"
            )

        predictions = pd.concat(
            [pd.read_parquet(f) for f in pred_files], ignore_index=True
        )
        metrics = pd.concat(
            [pd.read_parquet(f) for f in met_files], ignore_index=True
        )

        if approaches:
            predictions = predictions[predictions["approach"].isin(approaches)]
            metrics = metrics[metrics["approach"].isin(approaches)]

        n_approaches = predictions["approach"].nunique()
        n_targets = predictions["target_variable"].nunique()
        logger.info(
            "Loaded %d approaches x %d targets (%d prediction rows, %d metric rows)",
            n_approaches,
            n_targets,
            len(predictions),
            len(metrics),
        )
        return predictions, metrics

    # ------------------------------------------------------------------
    # Approach color mapping
    # ------------------------------------------------------------------

    def _color_map(self, approaches: List[str]) -> dict:
        """Deterministic approach -> color mapping."""
        return {
            name: APPROACH_COLORS[i % len(APPROACH_COLORS)]
            for i, name in enumerate(sorted(approaches))
        }

    # ------------------------------------------------------------------
    # R2 bar chart
    # ------------------------------------------------------------------

    def plot_r2_bars(self, metrics_df: pd.DataFrame) -> Path:
        """Grouped bar chart of R2 across approaches and targets.

        x-axis: target variables (sorted by best R2 descending).
        Grouped bars: one per approach.  Values annotated on bars.

        Returns:
            Path to saved figure.
        """
        sns.set_style("whitegrid")

        r2 = metrics_df[metrics_df["metric"] == "r2"].copy()
        if r2.empty:
            raise ValueError("No R2 metric rows found in metrics_df")

        approaches = sorted(r2["approach"].unique())
        colors = self._color_map(approaches)

        # Sort targets by best R2 descending
        best_r2 = r2.groupby("target_variable")["value"].max().sort_values(ascending=False)
        targets = list(best_r2.index)

        n_approaches = len(approaches)
        n_targets = len(targets)
        width = 0.8 / n_approaches
        x = np.arange(n_targets)

        fig_w = max(8, n_targets * 1.5 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, 6))

        for i, approach in enumerate(approaches):
            subset = r2[r2["approach"] == approach].set_index("target_variable")
            vals = [subset.loc[t, "value"] if t in subset.index else 0.0 for t in targets]
            offset = (i - (n_approaches - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=approach,
                color=colors[approach],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.005,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45 if n_approaches > 4 else 0,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("R2", fontsize=11)
        ax.set_title("Probe R2 Comparison Across Approaches", fontsize=13)
        ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        path = self._save("r2_bars", fig)
        plt.close(fig)
        logger.info("Saved R2 bar chart: %s", path)
        return path

    # ------------------------------------------------------------------
    # Scatter: y_true vs y_pred faceted by approach
    # ------------------------------------------------------------------

    def plot_scatter(self, predictions_df: pd.DataFrame, target: str) -> Path:
        """Faceted scatter of y_true vs y_pred, one panel per approach.

        Shared axis limits across panels.  1:1 reference line and R2
        annotation per panel.

        Args:
            predictions_df: Combined predictions DataFrame.
            target: Target variable name to plot.

        Returns:
            Path to saved figure.
        """
        sns.set_style("whitegrid")

        df = predictions_df[predictions_df["target_variable"] == target].copy()
        if df.empty:
            raise ValueError(f"No predictions for target '{target}'")

        approaches = sorted(df["approach"].unique())
        colors = self._color_map(approaches)
        n = len(approaches)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False
        )

        # Shared limits across all panels
        vmin = min(df["y_true"].min(), df["y_pred"].min())
        vmax = max(df["y_true"].max(), df["y_pred"].max())
        pad = (vmax - vmin) * 0.05
        lim = (vmin - pad, vmax + pad)

        for idx, approach in enumerate(approaches):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]
            sub = df[df["approach"] == approach]

            # Subsample for readability if large
            if len(sub) > 5000:
                sub = sub.sample(5000, random_state=42)

            ax.scatter(
                sub["y_true"],
                sub["y_pred"],
                s=2,
                alpha=0.3,
                color=colors[approach],
                rasterized=True,
            )
            ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.6, label="1:1")
            ax.set_xlim(lim)
            ax.set_ylim(lim)

            # R2 annotation
            ss_res = ((sub["y_true"] - sub["y_pred"]) ** 2).sum()
            ss_tot = ((sub["y_true"] - sub["y_true"].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            ax.text(
                0.05,
                0.95,
                f"R2 = {r2:.4f}\nn = {len(sub):,}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            ax.set_title(approach, fontsize=10)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")

        # Hide unused axes
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle(f"y_true vs y_pred: {target}", fontsize=13)
        plt.tight_layout()
        path = self._save(f"scatter_{target}", fig)
        plt.close(fig)
        logger.info("Saved scatter plot: %s", path)
        return path

    # ------------------------------------------------------------------
    # Spatial residual maps
    # ------------------------------------------------------------------

    def plot_residual_maps(
        self, predictions_df: pd.DataFrame, target: str
    ) -> Path:
        """Side-by-side spatial residual maps, one panel per approach.

        Residual = y_pred - y_true.  Uses RdBu_r with symmetric
        ``TwoSlopeNorm`` shared across panels for fair comparison.

        Args:
            predictions_df: Combined predictions DataFrame.
            target: Target variable name to plot.

        Returns:
            Path to saved figure.
        """
        df = predictions_df[predictions_df["target_variable"] == target].copy()
        if df.empty:
            raise ValueError(f"No predictions for target '{target}'")

        approaches = sorted(df["approach"].unique())
        n = len(approaches)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

        # Load boundary
        boundary_gdf = load_boundary(self.paths, crs=28992)

        # Compute shared symmetric vmax across all approaches
        df["residual"] = df["y_pred"] - df["y_true"]
        vmax_res = float(df["residual"].abs().quantile(0.98))
        if vmax_res == 0:
            vmax_res = 1.0
        norm = TwoSlopeNorm(vmin=-vmax_res, vcenter=0, vmax=vmax_res)
        cmap = cm.get_cmap("RdBu_r")

        # Compute shared extent from boundary or first approach's hex ids
        if boundary_gdf is not None:
            extent = tuple(boundary_gdf.total_bounds)
        else:
            first_sub = df[df["approach"] == approaches[0]]
            hex_ids = pd.Index(first_sub["region_id"].unique(), name="region_id")
            extent = self.db.extent(hex_ids, self.h3_resolution, crs=28992)

        minx, miny, maxx, maxy = extent
        pad = (maxx - minx) * 0.03
        render_extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

        # Panel dimensions: scale down per panel so figure stays reasonable
        panel_w = max(600, RASTER_W // ncols)
        panel_h = max(720, RASTER_H // ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 7 * nrows),
            squeeze=False,
        )

        for idx, approach in enumerate(approaches):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            sub = df[df["approach"] == approach]
            hex_ids = pd.Index(sub["region_id"].values, name="region_id")
            residuals = sub["residual"].values

            # Get centroids via SpatialDB
            cx, cy = self.db.centroids(hex_ids, self.h3_resolution, crs=28992)

            # Map residuals through colormap manually for rasterization
            rgba = cmap(norm(residuals))
            rgb = rgba[:, :3].astype(np.float32)

            # Rasterize via stamping (same pattern as linear_probe_viz)
            image = np.zeros((panel_h, panel_w, 4), dtype=np.float32)
            rx0, ry0, rx1, ry1 = render_extent
            mask = (
                (cx >= rx0) & (cx <= rx1) & (cy >= ry0) & (cy <= ry1)
                & np.isfinite(residuals)
            )
            cx_m, cy_m, rgb_m = cx[mask], cy[mask], rgb[mask]
            px = ((cx_m - rx0) / (rx1 - rx0) * (panel_w - 1)).astype(int)
            py = ((cy_m - ry0) / (ry1 - ry0) * (panel_h - 1)).astype(int)
            np.clip(px, 0, panel_w - 1, out=px)
            np.clip(py, 0, panel_h - 1, out=py)

            stamp = self._stamp
            if stamp <= 1:
                image[py, px, :3] = rgb_m
                image[py, px, 3] = 1.0
            else:
                for dy in range(-stamp + 1, stamp):
                    for dx in range(-stamp + 1, stamp):
                        sy = np.clip(py + dy, 0, panel_h - 1)
                        sx = np.clip(px + dx, 0, panel_w - 1)
                        image[sy, sx, :3] = rgb_m
                        image[sy, sx, 3] = 1.0

            plot_spatial_map(
                ax,
                image,
                render_extent,
                boundary_gdf,
                title=approach,
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
        fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            shrink=0.6,
            label="Residual (predicted - actual)",
            pad=0.02,
        )

        fig.suptitle(
            f"Spatial Residuals: {target}\n"
            f"res{self.h3_resolution} | EPSG:28992 | RdBu_r symmetric",
            fontsize=13,
        )
        plt.tight_layout(rect=[0, 0, 0.92, 0.95])
        path = self._save(f"residual_maps_{target}", fig)
        plt.close(fig)
        logger.info("Saved residual maps: %s", path)
        return path

    # ------------------------------------------------------------------
    # Convenience: generate all plots
    # ------------------------------------------------------------------

    def plot_all(self, target: Optional[str] = None) -> List[Path]:
        """Generate all comparison plots.

        If *target* is not specified, uses the first target variable found
        in the loaded data.

        Returns:
            List of paths to saved figures.
        """
        predictions, metrics = self.load_all()

        if target is None:
            target = predictions["target_variable"].unique()[0]
            logger.info("Auto-selected target: %s", target)

        paths: List[Path] = []
        paths.append(self.plot_r2_bars(metrics))
        paths.append(self.plot_scatter(predictions, target))
        paths.append(self.plot_residual_maps(predictions, target))
        return paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save(self, name: str, fig) -> Path:
        """Save figure to date-keyed output directory."""
        out_dir = self.output_dir / date.today().isoformat()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        return path


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-approach probe comparison plots."
    )
    parser.add_argument(
        "--study-area",
        default="netherlands",
        help="Study area name (default: netherlands)",
    )
    parser.add_argument(
        "--approaches",
        nargs="*",
        default=None,
        help="Subset of approaches to compare (space-separated). "
        "Default: all found under probe_results/.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target variable to plot. Default: first found.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Override output directory.",
    )
    parser.add_argument(
        "--resolution",
        default=9,
        type=int,
        help="H3 resolution (default: 9).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    plotter = ProbeComparisonPlotter(
        study_area=args.study_area,
        output_dir=args.output_dir,
        h3_resolution=args.resolution,
    )

    if args.approaches:
        predictions, metrics = plotter.load_all(approaches=args.approaches)
    else:
        predictions, metrics = plotter.load_all()

    target = args.target
    if target is None:
        target = predictions["target_variable"].unique()[0]
        logger.info("Auto-selected target: %s", target)

    saved = []
    saved.append(plotter.plot_r2_bars(metrics))
    saved.append(plotter.plot_scatter(predictions, target))
    saved.append(plotter.plot_residual_maps(predictions, target))

    print(f"\nSaved {len(saved)} plots:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
