#!/usr/bin/env python
"""
Q8 Wave 4: Ridge vs DNN probe parity on shared 74D multiscale UNet embedding.

Purpose: Resolve the Ridge=0.574 vs DNN=0.386 gap from 2026-03-29 report by
running both probes on the identical 74D multiscale_avg embedding with
identical spatial block CV folds (5-fold, 10km blocks, random_state=42).

Lifetime: temporary (one-off for 2026-04-19 Terminal C Wave 4).
Stage: 3 (analysis).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stage3_analysis.linear_probe import LinearProbeConfig, LinearProbeRegressor  # noqa: E402
from stage3_analysis.dnn_probe import DNNProbeConfig, DNNProbeRegressor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("q8_wave4")

EMB_REL = "data/study_areas/netherlands/stage2_multimodal/unet/embeddings/netherlands_res9_multiscale_avg_2022.parquet"
TARGET_REL = "data/study_areas/netherlands/target/leefbaarometer/leefbaarometer_h3res9_2022.parquet"


def run_ridge():
    cfg = LinearProbeConfig(
        study_area="netherlands",
        year=2022,
        h3_resolution=9,
        target_name="leefbaarometer",
        embeddings_path=EMB_REL,
        target_path=TARGET_REL,
        n_folds=5,
        block_width=10_000,
        block_height=10_000,
        random_state=42,
        run_descriptor="q8_74d_multiscale_avg",
    )
    reg = LinearProbeRegressor(cfg)
    reg.run()
    out = reg.save_results()
    logger.info(f"Ridge results saved to {out}")
    return reg, out


def run_dnn():
    cfg = DNNProbeConfig(
        study_area="netherlands",
        year=2022,
        h3_resolution=9,
        target_name="leefbaarometer",
        embeddings_path=EMB_REL,
        target_path=TARGET_REL,
        n_folds=5,
        block_width=10_000,
        block_height=10_000,
        random_state=42,
        hidden_dim=256,
        num_layers=3,
        activation="silu",
        max_epochs=200,
        patience=20,
        run_descriptor="q8_74d_multiscale_avg",
    )
    reg = DNNProbeRegressor(cfg)
    reg.run()
    out = reg.save_results()
    logger.info(f"DNN results saved to {out}")
    return reg, out


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Q8 WAVE 4: Ridge vs DNN on 74D multiscale_avg")
    logger.info("=" * 60)

    logger.info("\n--- RIDGE PROBE ---")
    ridge_reg, ridge_out = run_ridge()

    logger.info("\n--- DNN PROBE ---")
    dnn_reg, dnn_out = run_dnn()

    logger.info("\n=== COMPARISON ===")
    for t in ["lbm", "fys", "onv", "soc", "vrz", "won"]:
        r = ridge_reg.results[t].overall_r2
        d = dnn_reg.results[t].overall_r2
        logger.info(f"  {t}: Ridge={r:.4f}  DNN={d:.4f}  delta={d - r:+.4f}")

    r_mean = sum(ridge_reg.results[t].overall_r2 for t in ridge_reg.results) / len(ridge_reg.results)
    d_mean = sum(dnn_reg.results[t].overall_r2 for t in dnn_reg.results) / len(dnn_reg.results)
    logger.info(f"  MEAN: Ridge={r_mean:.4f}  DNN={d_mean:.4f}  delta={d_mean - r_mean:+.4f}")

    logger.info(f"\nArtifacts:")
    logger.info(f"  Ridge: {ridge_out}")
    logger.info(f"  DNN:   {dnn_out}")
