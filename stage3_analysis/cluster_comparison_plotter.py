"""Cross-approach cluster comparison visualization with target-sorted labels.

Globs standardized cluster result parquets from ``cluster_results/{approach}/``
and generates comparison plots: side-by-side spatial cluster maps with
semantically sorted labels, and quality metric bar charts.

Sort-at-render-time design: the ``ClusterResultsWriter`` stores raw integer
labels. This plotter loads a sort target (leefbaarometer score or one-hot
morphology level), computes the mean target per cluster, and relabels so that
cluster 0 = lowest mean target and cluster k-1 = highest. Rendered with
``rasterize_categorical(cmap="viridis")`` so dark = low, bright = high.

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


class ClusterComparisonPlotter:
    """Generate comparison visualizations across multiple cluster approaches.

    Reads standardized parquets (``assignments.parquet``, ``metrics.parquet``)
    written by ``ClusterResultsWriter`` under each approach subdirectory.

    Args:
        study_area: Name of the study area (e.g. ``"netherlands"``).
        output_dir: Override for output directory.  Defaults to
            ``cluster_results/comparison/``.
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
        self.output_dir = output_dir or (
            self.paths.cluster_results_root() / "comparison"
        )
        self._stamp = max(1, 11 - self.h3_resolution)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_all(
        self, approaches: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load assignments and metrics from all (or selected) approaches.

        Globs ``cluster_results/*/assignments.parquet`` and
        ``cluster_results/*/metrics.parquet``, concatenates into single
        DataFrames, and optionally filters to the given *approaches*.

        Returns:
            ``(assignments_df, metrics_df)`` with an ``approach`` column
            identifying the source.
        """
        root = self.paths.cluster_results_root()
        assign_files = sorted(root.glob("*/assignments.parquet"))
        met_files = sorted(root.glob("*/metrics.parquet"))

        if not assign_files:
            raise FileNotFoundError(
                f"No assignments.parquet found under {root}/*/"
            )

        assignments = pd.concat(
            [pd.read_parquet(f) for f in assign_files], ignore_index=True
        )
        metrics = pd.concat(
            [pd.read_parquet(f) for f in met_files], ignore_index=True
        )

        if approaches:
            assignments = assignments[assignments["approach"].isin(approaches)]
            metrics = metrics[metrics["approach"].isin(approaches)]

        n_approaches = assignments["approach"].nunique()
        n_k = assignments["k"].nunique()
        logger.info(
            "Loaded %d approaches x %d k values (%d assignment rows, %d metric rows)",
            n_approaches,
            n_k,
            len(assignments),
            len(metrics),
        )
        return assignments, metrics

    # ------------------------------------------------------------------
    # Sort target loading
    # ------------------------------------------------------------------

    def _load_sort_target(self, sort_by: str) -> pd.Series:
        """Load and return a Series indexed by region_id for the sort criterion.

        Args:
            sort_by: One of:
                - ``"lbm"``, ``"fys"``, ``"onv"``, ``"soc"``, ``"vrz"``,
                  ``"won"`` for leefbaarometer columns.
                - ``"level1_0"``, ``"level1_1"``, ``"level2_0"``, ...,
                  ``"level3_7"`` for one-hot morphology columns.

        Returns:
            Float Series indexed by ``region_id``.

        Raises:
            ValueError: If *sort_by* does not match any known target.
        """
        leefbaarometer_cols = {"lbm", "fys", "onv", "soc", "vrz", "won"}

        if sort_by in leefbaarometer_cols:
            target_path = self.paths.target_file(
                "leefbaarometer", self.h3_resolution, 2022
            )
            df = pd.read_parquet(target_path)
            if sort_by not in df.columns:
                raise ValueError(
                    f"Column '{sort_by}' not found in {target_path}. "
                    f"Available: {list(df.columns)}"
                )
            series = df[sort_by].astype(float)
            series.index.name = "region_id"
            return series

        # Morphology one-hot: "level{L}_{C}" -> type_level{L} one-hot encoded
        if sort_by.startswith("level"):
            parts = sort_by.split("_")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid morphology sort_by '{sort_by}'. "
                    f"Expected format: level{{1,2,3}}_{{class_index}}"
                )
            level_key = parts[0]  # e.g. "level1"
            class_idx = int(parts[1])

            target_path = self.paths.target_file(
                "urban_taxonomy", self.h3_resolution, 2025
            )
            df = pd.read_parquet(target_path)
            col_name = f"type_{level_key}"  # e.g. "type_level1"
            if col_name not in df.columns:
                raise ValueError(
                    f"Column '{col_name}' not found in {target_path}. "
                    f"Available: {list(df.columns)}"
                )
            # One-hot encode and extract the requested class column
            dummies = pd.get_dummies(df[col_name], prefix=level_key)
            onehot_col = f"{level_key}_{class_idx}"
            if onehot_col not in dummies.columns:
                raise ValueError(
                    f"One-hot column '{onehot_col}' not found. "
                    f"Available classes: {list(dummies.columns)}"
                )
            series = dummies[onehot_col].astype(float)
            series.index = df.index
            series.index.name = "region_id"
            return series

        raise ValueError(
            f"Unknown sort_by '{sort_by}'. Use one of: "
            f"{sorted(leefbaarometer_cols)} or level{{1,2,3}}_{{class_index}}"
        )

    # ------------------------------------------------------------------
    # Label sorting
    # ------------------------------------------------------------------

    def _sort_labels(
        self,
        labels: np.ndarray,
        region_ids: np.ndarray,
        sort_target: pd.Series,
    ) -> np.ndarray:
        """Relabel clusters so 0 = lowest mean target, k-1 = highest.

        Clusters with zero target coverage (all NaN) are sorted to the
        bottom (assigned the highest label index).

        Args:
            labels: Integer cluster assignment array (length N).
            region_ids: Region ID array (length N), aligned with *labels*.
            sort_target: Float Series indexed by ``region_id``.

        Returns:
            Integer array of relabeled cluster assignments (length N).
        """
        # Build a temporary DataFrame for the groupby
        tmp = pd.DataFrame({
            "cluster_label": labels,
            "target": sort_target.reindex(region_ids).values,
        })

        # Mean target per cluster (NaN rows skipped by default)
        cluster_means = tmp.groupby("cluster_label")["target"].mean()

        # Clusters with all-NaN targets get +inf so they sort to the end
        cluster_means = cluster_means.fillna(np.inf)

        # Rank: lowest mean -> rank 0
        sorted_clusters = cluster_means.sort_values().index.tolist()
        old_to_new = {old: new for new, old in enumerate(sorted_clusters)}

        return np.array([old_to_new[lbl] for lbl in labels], dtype=int)

    # ------------------------------------------------------------------
    # Spatial cluster maps
    # ------------------------------------------------------------------

    def plot_cluster_maps(
        self,
        assignments_df: pd.DataFrame,
        k: int,
        sort_by: str = "lbm",
    ) -> Path:
        """Side-by-side cluster maps, one panel per approach for given k.

        Labels are sorted by the *sort_by* criterion at render time so that
        dark (viridis low) = low mean target and bright = high.

        Args:
            assignments_df: Combined assignments DataFrame with columns
                ``approach``, ``k``, ``region_id``, ``cluster_label``.
            k: Number of clusters to plot.
            sort_by: Sort criterion (see ``_load_sort_target``).

        Returns:
            Path to saved figure.
        """
        sns.set_style("whitegrid")

        df = assignments_df[assignments_df["k"] == k].copy()
        if df.empty:
            raise ValueError(f"No assignments for k={k}")

        approaches = sorted(df["approach"].unique())
        n = len(approaches)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

        # Load sort target and boundary
        sort_target = self._load_sort_target(sort_by)
        boundary_gdf = load_boundary(self.paths, crs=28992)

        # Compute shared extent from boundary or first approach's hex ids
        if boundary_gdf is not None:
            extent = tuple(boundary_gdf.total_bounds)
        else:
            first_sub = df[df["approach"] == approaches[0]]
            hex_ids = pd.Index(
                first_sub["region_id"].unique(), name="region_id"
            )
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
            region_ids = sub["region_id"].values
            raw_labels = sub["cluster_label"].values

            # Sort labels by target mean
            sorted_labels = self._sort_labels(
                raw_labels, region_ids, sort_target
            )

            # Get centroids via SpatialDB
            hex_ids = pd.Index(region_ids, name="region_id")
            cx, cy = self.db.centroids(
                hex_ids, self.h3_resolution, crs=28992
            )

            # Rasterize with viridis (sequential: dark=low, bright=high)
            image = rasterize_categorical(
                cx,
                cy,
                sorted_labels,
                render_extent,
                n_clusters=k,
                width=panel_w,
                height=panel_h,
                cmap="viridis",
                stamp=self._stamp,
            )

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

        # Shared colorbar showing cluster index 0..k-1
        sm = cm.ScalarMappable(
            cmap=plt.get_cmap("viridis"),
            norm=plt.Normalize(vmin=0, vmax=k - 1),
        )
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            shrink=0.6,
            label=f"Cluster (sorted by mean {sort_by})",
            pad=0.02,
        )
        cbar.set_ticks(range(k))

        fig.suptitle(
            f"Cluster Comparison (k={k}, sorted by {sort_by})\n"
            f"res{self.h3_resolution} | EPSG:28992 | viridis",
            fontsize=13,
        )
        plt.tight_layout(rect=[0, 0, 0.92, 0.95])
        path = self._save(f"cluster_maps_k{k}_{sort_by}", fig)
        plt.close(fig)
        logger.info("Saved cluster maps: %s", path)
        return path

    # ------------------------------------------------------------------
    # Quality metric bar chart
    # ------------------------------------------------------------------

    def plot_quality_bars(self, metrics_df: pd.DataFrame, k: int) -> Path:
        """Bar chart of cluster quality metrics per approach for given k.

        Shows silhouette score and Calinski-Harabasz index side by side,
        each on its own y-axis scale (silhouette is bounded [-1, 1],
        Calinski-Harabasz is unbounded positive).

        Args:
            metrics_df: Combined metrics DataFrame with columns
                ``approach``, ``k``, ``metric``, ``value``.
            k: Number of clusters to plot.

        Returns:
            Path to saved figure.
        """
        sns.set_style("whitegrid")

        df = metrics_df[metrics_df["k"] == k].copy()
        if df.empty:
            raise ValueError(f"No metrics for k={k}")

        # Two target metrics
        target_metrics = ["silhouette", "calinski_harabasz"]
        metric_labels = {
            "silhouette": "Silhouette Score",
            "calinski_harabasz": "Calinski-Harabasz Index",
        }

        # Filter to available metrics
        available = [m for m in target_metrics if m in df["metric"].values]
        if not available:
            raise ValueError(
                f"Neither silhouette nor calinski_harabasz found in metrics. "
                f"Available: {list(df['metric'].unique())}"
            )

        approaches = sorted(df["approach"].unique())
        n_approaches = len(approaches)
        n_metrics = len(available)

        fig, axes_list = plt.subplots(
            1, n_metrics, figsize=(5 * n_metrics + 2, 5), squeeze=False
        )

        colors = list(plt.get_cmap("tab10").colors)

        for m_idx, metric_name in enumerate(available):
            ax = axes_list[0, m_idx]
            subset = df[df["metric"] == metric_name]

            vals = []
            for approach in approaches:
                row = subset[subset["approach"] == approach]
                vals.append(row["value"].values[0] if len(row) > 0 else 0.0)

            x = np.arange(n_approaches)
            bars = ax.bar(
                x,
                vals,
                color=[colors[i % len(colors)] for i in range(n_approaches)],
                edgecolor="white",
                linewidth=0.5,
            )

            # Value annotations
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + (max(vals) - min(vals)) * 0.02,
                    f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(approaches, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(metric_labels.get(metric_name, metric_name), fontsize=10)
            ax.set_title(metric_labels.get(metric_name, metric_name), fontsize=11)

        fig.suptitle(
            f"Cluster Quality Metrics (k={k})",
            fontsize=13,
        )
        plt.tight_layout()
        path = self._save(f"quality_bars_k{k}", fig)
        plt.close(fig)
        logger.info("Saved quality bars: %s", path)
        return path

    # ------------------------------------------------------------------
    # Convenience: generate all plots
    # ------------------------------------------------------------------

    def plot_all(self, k: int, sort_by: str = "lbm") -> List[Path]:
        """Generate all cluster comparison plots for the given k.

        Args:
            k: Number of clusters.
            sort_by: Sort criterion for cluster label ordering.

        Returns:
            List of paths to saved figures.
        """
        assignments, metrics = self.load_all()

        saved: List[Path] = []
        saved.append(self.plot_cluster_maps(assignments, k, sort_by=sort_by))
        saved.append(self.plot_quality_bars(metrics, k))
        return saved

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
        description="Generate cross-approach cluster comparison plots."
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
        "Default: all found under cluster_results/.",
    )
    parser.add_argument(
        "--k",
        required=True,
        type=int,
        help="Number of clusters to visualize.",
    )
    parser.add_argument(
        "--sort-by",
        default="lbm",
        help="Sort criterion for cluster labels (default: lbm). "
        "Options: lbm, fys, onv, soc, vrz, won, level1_0, level2_1, etc.",
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

    plotter = ClusterComparisonPlotter(
        study_area=args.study_area,
        output_dir=args.output_dir,
        h3_resolution=args.resolution,
    )

    assignments, metrics = plotter.load_all(approaches=args.approaches)

    saved = []
    saved.append(plotter.plot_cluster_maps(assignments, args.k, sort_by=args.sort_by))
    saved.append(plotter.plot_quality_bars(metrics, args.k))

    print(f"\nSaved {len(saved)} plots:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
