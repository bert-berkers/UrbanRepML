#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate all linear probe visualizations for a completed run.

Loads results (metrics, predictions, coefficients) from a linear probe
output directory and generates all plots via LinearProbeVisualizer.

Usage:
    python scripts/plot_linear_probe.py
    python scripts/plot_linear_probe.py --run-dir path/to/run
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stage3_analysis.dnn_probe import DNNProbeRegressor
from stage3_analysis.linear_probe import FoldMetrics
from stage3_analysis.linear_probe_viz import LinearProbeVisualizer
from utils import StudyAreaPaths

logger = logging.getLogger(__name__)


def load_linear_results_with_coefficients(linear_dir: Path) -> dict:
    """
    Load linear probe results including coefficients and fold metrics.

    DNNProbeRegressor.load_linear_results() leaves coefficients and
    feature_names empty.  This function supplements it by reading
    coefficients.csv and reconstructing fold metrics from
    metrics_summary.csv.

    Args:
        linear_dir: Path to the linear probe run directory.

    Returns:
        Dictionary mapping target column to TargetResult with full data.
    """
    # Base load: predictions + overall metrics
    results = DNNProbeRegressor.load_linear_results(linear_dir)

    # --- Supplement with coefficients ---
    coef_path = linear_dir / "coefficients.csv"
    if coef_path.exists():
        coef_df = pd.read_csv(coef_path)
        for target_col, result in results.items():
            target_coefs = coef_df[coef_df["target"] == target_col]
            if not target_coefs.empty:
                result.feature_names = target_coefs["feature"].tolist()
                result.coefficients = target_coefs["coefficient"].values
                logger.info(
                    f"  Loaded {len(result.coefficients)} coefficients "
                    f"for {target_col}"
                )
    else:
        logger.warning(f"No coefficients.csv in {linear_dir}, "
                       f"coefficient-based plots will be skipped")

    # --- Supplement with fold metrics ---
    metrics_path = linear_dir / "metrics_summary.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        for _, row in metrics_df.iterrows():
            target_col = row["target"]
            if target_col not in results:
                continue
            result = results[target_col]
            # Reconstruct FoldMetrics from fold{k}_r2 / fold{k}_rmse columns
            fold_metrics = []
            for fold_id in range(1, 20):  # scan up to 20 folds
                r2_key = f"fold{fold_id}_r2"
                rmse_key = f"fold{fold_id}_rmse"
                if r2_key in row and not pd.isna(row[r2_key]):
                    fold_metrics.append(FoldMetrics(
                        fold=fold_id,
                        r2=float(row[r2_key]),
                        rmse=float(row[rmse_key]),
                        mae=0.0,  # not stored in summary
                        n_train=0,
                        n_test=0,
                    ))
            if fold_metrics:
                result.fold_metrics = fold_metrics
                logger.info(
                    f"  Loaded {len(fold_metrics)} fold metrics "
                    f"for {target_col}"
                )

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate all linear probe visualizations"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to linear probe run directory "
             "(default: latest run in stage3/linear_probe)",
    )
    parser.add_argument(
        "--study-area",
        type=str,
        default="netherlands",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    paths = StudyAreaPaths(args.study_area)

    # Resolve run directory
    if args.run_dir:
        linear_dir = Path(args.run_dir)
    else:
        linear_dir = paths.latest_run(paths.stage3("linear_probe"))
        if linear_dir is None:
            logger.error("No linear probe runs found")
            sys.exit(1)

    logger.info(f"Loading linear probe results from {linear_dir}")

    plot_dir = linear_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load results with full coefficients and fold metrics
    results = load_linear_results_with_coefficients(linear_dir)
    logger.info(f"Loaded {len(results)} target results")

    # Get embeddings path for RGB top-3 maps
    embeddings_path = paths.embedding_file("alphaearth", 10, 2022)
    if not embeddings_path.exists():
        logger.warning(f"Embeddings not found at {embeddings_path}, "
                       f"RGB top-3 maps will be skipped")
        embeddings_path = None

    # Get boundary GDF for map backgrounds
    boundary_gdf = None
    boundary_path = paths.area_gdf_file()
    if boundary_path.exists():
        import geopandas as gpd
        boundary_gdf = gpd.read_file(boundary_path)
        logger.info(f"Loaded boundary from {boundary_path}")
    else:
        logger.warning(f"Boundary file not found at {boundary_path}")

    # Create visualizer and generate all plots
    viz = LinearProbeVisualizer(results=results, output_dir=plot_dir)
    generated = viz.plot_all(
        embeddings_path=embeddings_path,
        boundary_gdf=boundary_gdf,
    )

    print(f"\nGenerated {len(generated)} plots to {plot_dir}")
    for p in generated:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
