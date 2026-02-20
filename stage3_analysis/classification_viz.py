#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classification Probe Visualization

Creates publication-quality visualizations for classification probe results:
confusion matrices, spatial class maps, accuracy/F1 comparison bars,
per-fold metrics, and hierarchical accuracy degradation curves.

All methods assume classification results. No regression plots.

Visualizations:
    1. Confusion matrix: row-normalized heatmap per target
    2. Spatial class map: predicted classes + correct/incorrect binary map
    3. Metrics comparison: accuracy + F1-macro bars across hierarchy levels
    4. Fold metrics: per-fold accuracy/F1 box plots
    5. Hierarchical accuracy: accuracy/F1 vs hierarchy level with n_classes
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix

from utils import StudyAreaPaths

from .linear_probe import (
    TARGET_NAMES,
    TAXONOMY_TARGET_NAMES,
    TargetResult,
)

logger = logging.getLogger(__name__)


class ClassificationVisualizer:
    """
    Creates visualizations for classification probe results.

    All plots are saved to the output directory as high-resolution PNGs.
    Every method assumes the results contain classification task_type.
    """

    def __init__(
        self,
        results: Dict[str, TargetResult],
        output_dir: Path,
        study_area: str = "netherlands",
        figsize_base: Tuple[float, float] = (10, 6),
        dpi: int = 150,
        h3_resolution: int = 10,
    ):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = StudyAreaPaths(study_area)
        self.figsize_base = figsize_base
        self.dpi = dpi
        self.h3_resolution = h3_resolution

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
        })

    # ------------------------------------------------------------------
    # 1. Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, target_col: str) -> Path:
        """
        Row-normalized confusion matrix heatmap for a single target.

        Annotations shown only when n_classes <= 20 for readability.

        Args:
            target_col: Target variable name.

        Returns:
            Path to saved figure.
        """
        result = self.results[target_col]

        valid = ~np.isnan(result.oof_predictions)
        y_true = result.actual_values[valid].astype(int)
        y_pred = result.oof_predictions[valid].astype(int)

        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        cm_raw = confusion_matrix(y_true, y_pred, labels=labels)

        # Row-normalize (by true label)
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm_raw.astype(float),
            row_sums,
            out=np.zeros_like(cm_raw, dtype=float),
            where=row_sums != 0,
        )

        n_labels = len(labels)
        figsize = max(6, n_labels * 0.4)
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        sns.heatmap(
            cm_norm,
            xticklabels=labels,
            yticklabels=labels,
            cmap="Blues",
            vmin=0,
            vmax=1,
            annot=False,
            linewidths=0.5,
            linecolor="#cccccc",
            square=True,
            ax=ax,
        )

        target_name = TAXONOMY_TARGET_NAMES.get(
            target_col, TARGET_NAMES.get(target_col, target_col)
        )
        acc = result.overall_accuracy
        f1 = result.overall_f1_macro
        n = result.n_classes

        ax.set_title(
            f"Confusion Matrix (normalized): {target_name}\n"
            f"Acc={acc:.2%}, F1={f1:.3f}, n_classes={n}"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        plt.tight_layout()
        path = self.output_dir / f"confusion_matrix_{target_col}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved confusion matrix: {path}")
        return path

    # ------------------------------------------------------------------
    # 2. Spatial class map
    # ------------------------------------------------------------------

    def plot_spatial_class_map(self, target_col: str) -> Path:
        """
        Two-panel rasterized spatial map: predicted classes and correctness.

        Left panel: predicted class with categorical colormap.
        Right panel: correct/incorrect binary map (green=correct, red=incorrect).

        Args:
            target_col: Target variable name.

        Returns:
            Path to saved figure.
        """
        from shapely import get_geometry, get_num_geometries

        result = self.results[target_col]

        valid = ~np.isnan(result.oof_predictions)
        pred_df = pd.DataFrame(
            {
                "region_id": result.region_ids[valid],
                "actual": result.actual_values[valid].astype(int),
                "predicted": result.oof_predictions[valid].astype(int),
            }
        ).set_index("region_id")
        pred_df["correct"] = (pred_df["actual"] == pred_df["predicted"]).astype(int)

        n_hexagons = len(pred_df)
        hex_ids = pd.Index(pred_df.index, name="region_id")

        # Load boundary
        boundary_gdf = None
        boundary_path = self.paths.area_gdf_file()
        if boundary_path.exists():
            boundary_gdf = gpd.read_file(boundary_path)
            logger.info(f"  Loaded boundary from {boundary_path}")

        if boundary_gdf is not None:
            if boundary_gdf.crs is None:
                boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
            boundary_gdf = boundary_gdf.to_crs(epsg=28992)

        # Filter to European Netherlands (exclude Caribbean)
        if boundary_gdf is not None:
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

        # Compute extent
        if boundary_gdf is not None:
            extent = boundary_gdf.total_bounds
        else:
            from utils.spatial_db import SpatialDB
            extent = SpatialDB.for_study_area(self.paths.study_area).extent(
                hex_ids, resolution=self.h3_resolution, crs=28992
            )
        minx, miny, maxx, maxy = extent
        pad = (maxx - minx) * 0.03
        render_extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

        # --- Left panel: predicted class with categorical colormap ---
        predicted_vals = pred_df["predicted"].values
        unique_classes = sorted(np.unique(predicted_vals))
        n_classes = len(unique_classes)

        # Choose colormap based on number of classes
        if n_classes <= 10:
            cmap_cls = plt.get_cmap("tab10")
        elif n_classes <= 20:
            cmap_cls = plt.get_cmap("tab20")
        else:
            cmap_cls = plt.get_cmap("viridis")

        # Map class IDs to colors
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        norm_cls = Normalize(vmin=0, vmax=max(n_classes - 1, 1))
        class_indices = np.array([class_to_idx[c] for c in predicted_vals])
        rgba_cls = cmap_cls(norm_cls(class_indices))
        rgb_cls = rgba_cls[:, :3].astype(np.float32)

        # --- Right panel: correct/incorrect binary ---
        correct_vals = pred_df["correct"].values
        # Green for correct, red for incorrect
        rgb_correct = np.zeros((len(correct_vals), 3), dtype=np.float32)
        rgb_correct[correct_vals == 1] = [0.2, 0.7, 0.2]  # green
        rgb_correct[correct_vals == 0] = [0.8, 0.2, 0.2]  # red

        # Rasterize both panels
        raster_cls = self._rasterize_centroids(
            hex_ids=hex_ids,
            rgb_array=rgb_cls,
            extent=render_extent,
            width=1200,
            height=1400,
        )
        raster_correct = self._rasterize_centroids(
            hex_ids=hex_ids,
            rgb_array=rgb_correct,
            extent=render_extent,
            width=1200,
            height=1400,
        )

        logger.info(
            f"  Rasterized {n_hexagons:,} hexagons (centroid rasterization)"
        )

        # Plot 2-panel layout
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        fig.set_facecolor("white")

        imshow_kwargs = dict(
            extent=[
                render_extent[0],
                render_extent[2],
                render_extent[1],
                render_extent[3],
            ],
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            zorder=2,
        )

        target_name = TAXONOMY_TARGET_NAMES.get(
            target_col, TARGET_NAMES.get(target_col, target_col)
        )
        acc = result.overall_accuracy

        for ax_idx, (ax, raster, title_text) in enumerate(
            [
                (axes[0], raster_cls, f"Predicted Class: {target_name}"),
                (
                    axes[1],
                    raster_correct,
                    f"Correct/Incorrect: {target_name}",
                ),
            ]
        ):
            ax.set_facecolor("white")

            if boundary_gdf is not None:
                boundary_gdf.plot(
                    ax=ax,
                    facecolor="#f0f0f0",
                    edgecolor="#cccccc",
                    linewidth=0.5,
                )

            ax.imshow(raster, **imshow_kwargs)
            ax.set_xlim(render_extent[0], render_extent[2])
            ax.set_ylim(render_extent[1], render_extent[3])
            ax.set_title(title_text)
            ax.grid(True, linewidth=0.5, alpha=0.5, color="gray")
            ax.tick_params(labelsize=8)
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")

            self._add_scale_bar(ax, length_km=50)
            self._add_north_arrow(ax)

        # Legend for left panel: class colors (only if manageable number)
        if n_classes <= 20:
            from matplotlib.patches import Patch

            legend_patches = [
                Patch(
                    facecolor=cmap_cls(norm_cls(class_to_idx[c])),
                    label=f"Class {c}",
                )
                for c in unique_classes
            ]
            axes[0].legend(
                handles=legend_patches,
                loc="lower left",
                fontsize=7,
                ncol=max(1, n_classes // 10),
                framealpha=0.85,
            )

        # Legend for right panel: correct/incorrect
        from matplotlib.patches import Patch

        axes[1].legend(
            handles=[
                Patch(facecolor=(0.2, 0.7, 0.2), label="Correct"),
                Patch(facecolor=(0.8, 0.2, 0.2), label="Incorrect"),
            ],
            loc="lower left",
            fontsize=9,
            framealpha=0.85,
        )

        fig.suptitle(
            f"Spatial Classification Map: {target_name}\n"
            f"Accuracy={acc:.2%} | n={n_hexagons:,} hexagons",
            fontsize=14,
        )

        plt.tight_layout()
        path = self.output_dir / f"spatial_class_map_{target_col}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved spatial class map: {path}")
        return path

    # ------------------------------------------------------------------
    # 3. Classification metrics comparison
    # ------------------------------------------------------------------

    def plot_classification_metrics_comparison(self) -> Path:
        """
        Accuracy + F1-macro bar chart across all hierarchy levels.

        Two subplots side by side, one for accuracy, one for F1-macro.

        Returns:
            Path to saved figure.
        """
        targets = list(self.results.keys())
        target_labels = [
            TAXONOMY_TARGET_NAMES.get(t, TARGET_NAMES.get(t, t))
            for t in targets
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(targets))
        width = 0.5

        # Accuracy
        acc_vals = [self.results[t].overall_accuracy for t in targets]
        bars = axes[0].bar(
            x, acc_vals, width, color="steelblue", edgecolor="white"
        )
        for bar in bars:
            h = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.005,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[0].set_ylabel("Accuracy (OOF)")
        axes[0].set_title("Classification Accuracy per Level")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(target_labels, rotation=30, ha="right")
        axes[0].set_ylim(0, 1.05)

        # F1 macro
        f1_vals = [self.results[t].overall_f1_macro for t in targets]
        bars = axes[1].bar(
            x, f1_vals, width, color="coral", edgecolor="white"
        )
        for bar in bars:
            h = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.005,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[1].set_ylabel("F1 Macro (OOF)")
        axes[1].set_title("Classification F1 (macro) per Level")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(target_labels, rotation=30, ha="right")
        axes[1].set_ylim(0, 1.05)

        fig.suptitle(
            "Classification Probe: Accuracy & F1 per Hierarchy Level\n"
            "Spatial Block Cross-Validation",
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        path = self.output_dir / "classification_metrics_comparison.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved classification metrics comparison: {path}")
        return path

    # ------------------------------------------------------------------
    # 4. Fold metrics
    # ------------------------------------------------------------------

    def plot_fold_metrics(self) -> Optional[Path]:
        """
        Per-fold accuracy/F1 box plots across targets.

        Two subplots: Accuracy and F1 Macro.

        Returns:
            Path to saved figure, or None if no fold metrics.
        """
        rows = []
        for target_col, result in self.results.items():
            target_label = TAXONOMY_TARGET_NAMES.get(
                target_col, TARGET_NAMES.get(target_col, target_col)
            )
            for fm in result.fold_metrics:
                rows.append(
                    {
                        "target": target_label,
                        "fold": fm.fold,
                        "Accuracy": fm.accuracy,
                        "F1 Macro": fm.f1_macro,
                    }
                )

        if not rows:
            logger.warning(
                "No fold metrics to plot, skipping fold_metrics plot"
            )
            return None

        metrics_df = pd.DataFrame(rows)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(
            data=metrics_df,
            x="target",
            y="Accuracy",
            ax=axes[0],
            palette="Set2",
            hue="target",
            legend=False,
        )
        sns.stripplot(
            data=metrics_df,
            x="target",
            y="Accuracy",
            ax=axes[0],
            color="black",
            size=5,
            alpha=0.7,
        )
        axes[0].set_title("Accuracy by Spatial Fold")
        axes[0].set_xticklabels(
            axes[0].get_xticklabels(), rotation=30, ha="right"
        )

        sns.boxplot(
            data=metrics_df,
            x="target",
            y="F1 Macro",
            ax=axes[1],
            palette="Set2",
            hue="target",
            legend=False,
        )
        sns.stripplot(
            data=metrics_df,
            x="target",
            y="F1 Macro",
            ax=axes[1],
            color="black",
            size=5,
            alpha=0.7,
        )
        axes[1].set_title("F1 (macro) by Spatial Fold")
        axes[1].set_xticklabels(
            axes[1].get_xticklabels(), rotation=30, ha="right"
        )

        fig.suptitle(
            "Classification Probe: Per-Fold Spatial CV Metrics", fontsize=14
        )

        plt.tight_layout()
        path = self.output_dir / "fold_metrics.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved fold metrics plot: {path}")
        return path

    # ------------------------------------------------------------------
    # 5. Hierarchical accuracy
    # ------------------------------------------------------------------

    def plot_hierarchical_accuracy(self) -> Optional[Path]:
        """
        Line plot: accuracy and F1-macro vs hierarchy level with n_classes.

        X-axis: level number (1-7).
        Left y-axis: accuracy + F1-macro lines.
        Right y-axis: n_classes bars.
        Shows how classification degrades as taxonomy granularity increases.

        Returns:
            Path to saved figure, or None if fewer than 2 results.
        """
        if len(self.results) < 2:
            logger.warning(
                "Hierarchical accuracy plot requires >= 2 targets, skipping"
            )
            return None

        # Extract level numbers from target column names
        levels = []
        acc_vals = []
        f1_vals = []
        n_classes_vals = []

        for target_col, result in self.results.items():
            # Parse level number from "type_level{N}"
            if target_col.startswith("type_level"):
                try:
                    level = int(target_col.replace("type_level", ""))
                except ValueError:
                    continue
            else:
                continue

            levels.append(level)
            acc_vals.append(result.overall_accuracy)
            f1_vals.append(result.overall_f1_macro)
            n_classes_vals.append(result.n_classes)

        if len(levels) < 2:
            logger.warning(
                "Hierarchical accuracy plot requires >= 2 type_level targets, "
                "skipping"
            )
            return None

        # Sort by level
        sort_idx = np.argsort(levels)
        levels = [levels[i] for i in sort_idx]
        acc_vals = [acc_vals[i] for i in sort_idx]
        f1_vals = [f1_vals[i] for i in sort_idx]
        n_classes_vals = [n_classes_vals[i] for i in sort_idx]

        fig, ax1 = plt.subplots(figsize=self.figsize_base)

        # Left y-axis: accuracy and F1
        color_acc = "steelblue"
        color_f1 = "coral"

        (line_acc,) = ax1.plot(
            levels,
            acc_vals,
            "o-",
            color=color_acc,
            linewidth=2,
            markersize=8,
            label="Accuracy",
        )
        (line_f1,) = ax1.plot(
            levels,
            f1_vals,
            "s--",
            color=color_f1,
            linewidth=2,
            markersize=8,
            label="F1 Macro",
        )

        ax1.set_xlabel("Hierarchy Level")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1.05)
        ax1.set_xticks(levels)

        # Annotate accuracy values
        for lv, acc, f1 in zip(levels, acc_vals, f1_vals):
            ax1.annotate(
                f"{acc:.3f}",
                (lv, acc),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=color_acc,
            )
            ax1.annotate(
                f"{f1:.3f}",
                (lv, f1),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8,
                color=color_f1,
            )

        # Right y-axis: n_classes
        ax2 = ax1.twinx()
        color_nc = "#888888"
        bars = ax2.bar(
            levels,
            n_classes_vals,
            width=0.3,
            alpha=0.3,
            color=color_nc,
            label="n_classes",
        )
        ax2.set_ylabel("Number of Classes", color=color_nc)
        ax2.tick_params(axis="y", labelcolor=color_nc)

        # Annotate n_classes on bars
        for lv, nc in zip(levels, n_classes_vals):
            ax2.text(
                lv,
                nc + max(n_classes_vals) * 0.02,
                str(nc),
                ha="center",
                fontsize=8,
                color=color_nc,
            )

        # Combined legend
        lines = [line_acc, line_f1]
        labels = [l.get_label() for l in lines]
        from matplotlib.patches import Patch

        lines.append(Patch(facecolor=color_nc, alpha=0.3))
        labels.append("n_classes")
        ax1.legend(lines, labels, loc="upper right")

        ax1.set_title(
            "Hierarchical Accuracy Degradation\n"
            "Classification performance vs taxonomy granularity"
        )
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "hierarchical_accuracy.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved hierarchical accuracy plot: {path}")
        return path

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def plot_all(self, skip_spatial: bool = False) -> List[Path]:
        """
        Generate all classification visualizations.

        Args:
            skip_spatial: If True, skip spatial class maps (faster).

        Returns:
            List of paths to all saved figures.
        """
        logger.info(f"Generating all visualizations to {self.output_dir}")
        paths = []

        # Per-target plots
        for target_col in self.results:
            paths.append(self.plot_confusion_matrix(target_col))
            if not skip_spatial:
                paths.append(self.plot_spatial_class_map(target_col))

        # Cross-target plots
        paths.append(self.plot_classification_metrics_comparison())

        fold_path = self.plot_fold_metrics()
        if fold_path is not None:
            paths.append(fold_path)

        hier_path = self.plot_hierarchical_accuracy()
        if hier_path is not None:
            paths.append(hier_path)

        # Filter None values
        paths = [p for p in paths if p is not None]

        logger.info(f"Generated {len(paths)} visualizations")
        return paths

    # ------------------------------------------------------------------
    # Cartographic helpers (same as LinearProbeVisualizer)
    # ------------------------------------------------------------------

    def _add_scale_bar(self, ax, length_km: int = 50):
        """
        Add a scale bar to a map axis in EPSG:28992 coordinate space.

        Args:
            ax: Matplotlib axes (with EPSG:28992 data plotted).
            length_km: Scale bar length in kilometers.
        """
        bar_length = length_km * 1000  # meters in RD New

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
        y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.04
        tick_height = (ylim[1] - ylim[0]) * 0.012

        # Main bar
        ax.plot(
            [x0, x0 + bar_length],
            [y0, y0],
            color="black",
            linewidth=2,
            solid_capstyle="butt",
            transform=ax.transData,
        )

        # Tick marks at ends
        ax.plot(
            [x0, x0],
            [y0 - tick_height, y0 + tick_height],
            color="black",
            linewidth=1.5,
            transform=ax.transData,
        )
        ax.plot(
            [x0 + bar_length, x0 + bar_length],
            [y0 - tick_height, y0 + tick_height],
            color="black",
            linewidth=1.5,
            transform=ax.transData,
        )

        # Label
        ax.text(
            x0 + bar_length / 2,
            y0 + tick_height * 1.8,
            f"{length_km} km",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            transform=ax.transData,
        )

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
        ax.text(
            x_pos,
            y_pos + arrow_length * 0.2,
            "N",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            transform=ax.transData,
        )

    def _rasterize_centroids(
        self,
        hex_ids: pd.Index,
        rgb_array: np.ndarray,
        extent: tuple,
        width: int = 2000,
        height: int = 2400,
    ) -> np.ndarray:
        """
        Rasterize H3 centroids to an RGB image via SpatialDB.

        Looks up centroids from the pre-computed regions parquet via
        SpatialDB, already in EPSG:28992 (RD New), and writes RGB values
        into a numpy pixel array.

        Args:
            hex_ids: Index of H3 hex ID strings.
            rgb_array: (N, 3) float array with R, G, B in [0, 1].
            extent: (minx, miny, maxx, maxy) in EPSG:28992.
            width: Output image width in pixels.
            height: Output image height in pixels.

        Returns:
            (height, width, 4) RGBA float32 array with white background.
        """
        from utils.spatial_db import SpatialDB

        db = SpatialDB.for_study_area(self.paths.study_area)
        all_cx, all_cy = db.centroids(hex_ids, resolution=self.h3_resolution, crs=28992)

        minx, miny, maxx, maxy = extent
        mask = (
            (all_cx >= minx) & (all_cx <= maxx)
            & (all_cy >= miny) & (all_cy <= maxy)
        )

        cx = all_cx[mask]
        cy = all_cy[mask]
        rgb_masked = rgb_array[mask]

        # Map to pixel coordinates
        px = ((cx - minx) / (maxx - minx) * (width - 1)).astype(int)
        py = ((cy - miny) / (maxy - miny) * (height - 1)).astype(int)

        # Clip to valid pixel range
        np.clip(px, 0, width - 1, out=px)
        np.clip(py, 0, height - 1, out=py)

        # Write into RGBA image (white background)
        image = np.ones((height, width, 4), dtype=np.float32)
        image[py, px, :3] = rgb_masked
        image[py, px, 3] = 1.0

        return image
