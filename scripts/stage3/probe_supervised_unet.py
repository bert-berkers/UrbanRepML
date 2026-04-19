#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear probe on supervised UNet embeddings -> leefbaarometer.

Runs OLS linear probes with 5-fold spatial block CV on the supervised
FullAreaUNet embeddings (Kendall uncertainty weighting), writing results
via ProbeResultsWriter for standardized cross-approach comparison.

Two approaches:
    1. unet_supervised: res9 embeddings (74D)
    2. unet_supervised_multiscale: multiscale concat (222D)

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/probe_supervised_unet.py
"""

import logging
import sys

from stage3_analysis.linear_probe import LinearProbeConfig, LinearProbeRegressor
from stage3_analysis.probe_results_writer import ProbeResultsWriter
from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

STUDY_AREA = "netherlands"
YEAR = 2022
H3_RESOLUTION = 9

EMBEDDINGS_DIR = str(StudyAreaPaths(STUDY_AREA).model_embeddings("unet"))

APPROACHES = [
    {
        "approach": "unet_supervised",
        "file": f"{EMBEDDINGS_DIR}/netherlands_res9_2022.parquet",
        "description": "Supervised UNet res9 (74D)",
    },
    {
        "approach": "unet_supervised_multiscale",
        "file": f"{EMBEDDINGS_DIR}/netherlands_res9_multiscale_concat_2022.parquet",
        "description": "Supervised UNet multiscale concat (222D)",
    },
]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    for spec in APPROACHES:
        approach = spec["approach"]
        emb_path = spec["file"]
        desc = spec["description"]

        logger.info("=" * 70)
        logger.info("APPROACH: %s", approach)
        logger.info("  %s", desc)
        logger.info("  Embeddings: %s", emb_path)
        logger.info("=" * 70)

        config = LinearProbeConfig(
            study_area=STUDY_AREA,
            year=YEAR,
            h3_resolution=H3_RESOLUTION,
            embeddings_path=emb_path,
            modality="unet",
            run_descriptor=f"supervised_{approach}",
        )

        regressor = LinearProbeRegressor(config)
        results = regressor.run()
        regressor.save_results()

        # Write standardized results for cross-approach comparison
        out = ProbeResultsWriter.write_from_regressor(
            regressor, approach=approach, study_area=STUDY_AREA
        )
        logger.info("Standardized results -> %s", out)

        # Print summary
        r2_values = []
        for target_col, result in results.items():
            r2_values.append(result.overall_r2)
            logger.info(
                "  %s: R2=%.4f  RMSE=%.4f",
                target_col, result.overall_r2, result.overall_rmse,
            )
        mean_r2 = sum(r2_values) / len(r2_values)
        logger.info("  MEAN R2: %.4f", mean_r2)
        logger.info("")


if __name__ == "__main__":
    main()
