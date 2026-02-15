#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare DNN (MLP) Probe vs Linear Probe Results

Loads both probe results from disk, produces a comparison CSV and
three types of comparison plots:
    1. Side-by-side R2 and RMSE bar chart
    2. DNN vs linear prediction scatter (faceted by target)
    3. Spatial improvement maps per target (where DNN beats linear)

Both probes must have been run with identical spatial blocking
(10km blocks, 5 folds, random_state=42) for a fair comparison.

Usage:
    python scripts/compare_probes.py
    python scripts/compare_probes.py --dnn-dir path/to/dnn --linear-dir path/to/linear
    python scripts/compare_probes.py --skip-spatial   # skip slow spatial maps
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from stage3_analysis.dnn_probe import DNNProbeRegressor, DNNProbeConfig, TARGET_COLS, TARGET_NAMES
from stage3_analysis.dnn_probe_viz import DNNProbeVisualizer
from stage3_analysis.linear_probe import TargetResult

logger = logging.getLogger(__name__)


def load_training_curves(dnn_dir: Path) -> dict:
    """
    Load DNN training curves from JSON files.

    Returns:
        Dict of {target_col: {fold_id: [val_loss_per_epoch]}}.
    """
    curves_dir = dnn_dir / "training_curves"
    curves = {}

    if not curves_dir.exists():
        logger.warning(f"No training_curves directory at {curves_dir}")
        return curves

    for curve_file in sorted(curves_dir.glob("*.json")):
        with open(curve_file) as f:
            data = json.load(f)
        target = data["target"]
        fold = data["fold"]
        val_loss = data["val_loss"]

        if target not in curves:
            curves[target] = {}
        curves[target][fold] = val_loss

    n_files = sum(len(fc) for fc in curves.values())
    logger.info(f"Loaded {n_files} training curve files from {curves_dir}")
    return curves


def main():
    parser = argparse.ArgumentParser(
        description="Compare DNN (MLP) probe vs linear probe results"
    )
    parser.add_argument(
        "--dnn-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-02-15_default"
        ),
        help="Path to DNN probe results directory",
    )
    parser.add_argument(
        "--linear-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data/study_areas/netherlands/stage3_analysis/linear_probe/2026-02-15_default"
        ),
        help="Path to linear probe results directory",
    )
    parser.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Skip spatial improvement maps (saves ~1 min per target)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    dnn_dir = Path(args.dnn_dir)
    linear_dir = Path(args.linear_dir)

    logger.info(f"DNN results dir:    {dnn_dir}")
    logger.info(f"Linear results dir: {linear_dir}")

    # ------------------------------------------------------------------
    # 1. Load results from both probes
    # ------------------------------------------------------------------
    logger.info("\n--- Loading results ---")
    dnn_results = DNNProbeRegressor.load_linear_results(dnn_dir)
    linear_results = DNNProbeRegressor.load_linear_results(linear_dir)

    logger.info(f"DNN targets loaded:    {sorted(dnn_results.keys())}")
    logger.info(f"Linear targets loaded: {sorted(linear_results.keys())}")

    # ------------------------------------------------------------------
    # 2. Build comparison CSV via compare_with_linear()
    # ------------------------------------------------------------------
    logger.info("\n--- Building comparison table ---")

    # Create a minimal DNNProbeConfig. We override output_dir to point at
    # the existing DNN results so no new run directory is created.
    # Note: DNNProbeConfig.__post_init__ creates a new run_id, but we
    # override output_dir afterward and never call run().
    config = DNNProbeConfig(study_area="netherlands")
    config.output_dir = str(dnn_dir)

    regressor = DNNProbeRegressor(config, project_root=PROJECT_ROOT)
    regressor.results = dnn_results

    comparison_df = regressor.compare_with_linear(linear_results)
    comparison_path = dnn_dir / "comparison_linear_vs_dnn.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n=== Comparison Table ===")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved to: {comparison_path}")
    print(f"Rows: {len(comparison_df)} (expected 6)")

    # ------------------------------------------------------------------
    # 3. Load training curves for DNN-specific plots
    # ------------------------------------------------------------------
    training_curves = load_training_curves(dnn_dir)

    # ------------------------------------------------------------------
    # 4. Create visualizer and generate comparison plots
    # ------------------------------------------------------------------
    logger.info("\n--- Generating comparison plots ---")

    plots_dir = dnn_dir / "plots"
    viz = DNNProbeVisualizer(
        results=dnn_results,
        output_dir=plots_dir,
        training_curves=training_curves,
    )

    generated_plots = []

    # (a) Side-by-side R2 and RMSE bars
    bars_path = viz.plot_comparison_bars(linear_results)
    if bars_path:
        generated_plots.append(bars_path)

    # (b) Scatter: DNN vs linear predictions
    scatter_path = viz.plot_comparison_scatter(linear_results)
    if scatter_path:
        generated_plots.append(scatter_path)

    # (c) Spatial improvement maps for each target
    if not args.skip_spatial:
        for target in TARGET_COLS:
            if target in dnn_results and target in linear_results:
                imp_path = viz.plot_spatial_improvement(target, linear_results)
                if imp_path:
                    generated_plots.append(imp_path)
    else:
        logger.info("Skipping spatial improvement maps (--skip-spatial)")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n=== Generated Plot Files ===")
    for p in generated_plots:
        print(f"  {p}")
    print(f"\nTotal plots generated: {len(generated_plots)}")

    # Verify comparison CSV row count
    verify_df = pd.read_csv(comparison_path)
    assert len(verify_df) == 6, f"Expected 6 rows, got {len(verify_df)}"
    print(f"\nComparison CSV verified: {len(verify_df)} rows")
    print(f"\nAll comparison outputs saved to: {dnn_dir}")


if __name__ == "__main__":
    main()
