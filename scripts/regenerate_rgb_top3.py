#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regenerate rgb_top3 maps for all 6 liveability targets.

Validates the fix from 2026-02-13:
- White background (not yellow)
- Dissolve optimization for faster rendering
- 8 color levels per channel (512 max bins)

Loads existing linear probe results (predictions + coefficients)
and regenerates ONLY the rgb_top3 maps.
"""

import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(r"C:\Users\Bert Berkers\PycharmProjects\UrbanRepML")
sys.path.insert(0, str(PROJECT_ROOT))

from stage3_analysis.linear_probe import TargetResult, TARGET_NAMES, TARGET_COLS
from stage3_analysis.linear_probe_viz import LinearProbeVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
ANALYSIS_DIR = PROJECT_ROOT / "data" / "study_areas" / "netherlands" / "analysis" / "linear_probe"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "study_areas" / "netherlands" / "embeddings" / "alphaearth" / "netherlands_res10_2022.parquet"
BOUNDARY_PATH = PROJECT_ROOT / "data" / "study_areas" / "netherlands" / "boundaries" / "netherlands_boundary.parquet"


def load_results_from_disk() -> dict:
    """Reconstruct TargetResult objects from saved predictions + coefficients."""
    # Load coefficients
    coef_df = pd.read_csv(ANALYSIS_DIR / "coefficients.csv")

    # Load config for feature names
    import json
    with open(ANALYSIS_DIR / "config.json") as f:
        config = json.load(f)
    feature_names = config["feature_names"]

    results = {}
    for target_col in TARGET_COLS:
        pred_path = ANALYSIS_DIR / f"predictions_{target_col}.parquet"
        if not pred_path.exists():
            logger.warning(f"Predictions file not found: {pred_path}, skipping {target_col}")
            continue

        pred_df = pd.read_parquet(pred_path)

        # Get coefficients for this target
        target_coefs = coef_df[coef_df["target"] == target_col].set_index("feature")
        coefficients = np.array([target_coefs.loc[f, "coefficient"] for f in feature_names])

        result = TargetResult(
            target=target_col,
            target_name=TARGET_NAMES[target_col],
            best_alpha=0.0,
            best_l1_ratio=0.0,
            fold_metrics=[],
            overall_r2=0.0,  # Not needed for RGB maps
            overall_rmse=0.0,
            overall_mae=0.0,
            coefficients=coefficients,
            intercept=0.0,
            feature_names=feature_names,
            oof_predictions=pred_df["predicted"].values,
            actual_values=pred_df["actual"].values,
            region_ids=pred_df.index.values,
        )
        results[target_col] = result

    return results


def main():
    logger.info("=== Regenerating rgb_top3 maps for all 6 liveability targets ===")
    logger.info(f"Analysis dir: {ANALYSIS_DIR}")
    logger.info(f"Embeddings: {EMBEDDINGS_PATH}")
    logger.info(f"Boundary: {BOUNDARY_PATH}")

    # Load existing results
    results = load_results_from_disk()
    logger.info(f"Loaded results for {len(results)} targets: {list(results.keys())}")

    # Load boundary
    boundary_gdf = gpd.read_parquet(BOUNDARY_PATH)
    logger.info(f"Loaded boundary: {len(boundary_gdf)} features, CRS={boundary_gdf.crs}")

    # Create visualizer
    viz = LinearProbeVisualizer(
        results=results,
        output_dir=ANALYSIS_DIR,
        dpi=150,
    )

    # Generate rgb_top3 maps
    total_start = time.time()
    timings = {}

    for target_col in results:
        logger.info(f"\n--- Generating rgb_top3 for {target_col} ({TARGET_NAMES[target_col]}) ---")
        t0 = time.time()
        path = viz.plot_rgb_top3_map(
            target_col=target_col,
            embeddings_path=EMBEDDINGS_PATH,
            boundary_gdf=boundary_gdf,
        )
        elapsed = time.time() - t0
        timings[target_col] = elapsed
        logger.info(f"  -> {path} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    # Summary report
    logger.info("\n=== RENDER SUMMARY ===")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"{'Target':<6} {'Name':<25} {'Time (s)':>10} {'Output'}")
    logger.info("-" * 90)
    for target_col, elapsed in timings.items():
        out_file = ANALYSIS_DIR / f"rgb_top3_{target_col}.png"
        size_mb = out_file.stat().st_size / (1024 * 1024) if out_file.exists() else 0
        logger.info(f"{target_col:<6} {TARGET_NAMES[target_col]:<25} {elapsed:>10.1f} "
                     f"{out_file.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
