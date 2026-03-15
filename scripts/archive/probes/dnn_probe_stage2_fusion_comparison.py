#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe: Stage 2 Fusion Comparison (Concat vs Ring Agg vs GCN)

Runs DNN regression probes on all three Stage 2 fusion outputs against
leefbaarometer targets to compare spatial context methods.

Configurations match previous session (2026-03-06):
    hidden_dim=256, lr=1e-4, patience=20, max_epochs=200, activation=silu
    5-fold spatial block CV (10km blocks)

Lifetime: temporary
Stage: stage3 analysis
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stage3_analysis.dnn_probe import DNNProbeConfig, DNNProbeRegressor
from stage3_analysis.linear_probe import TARGET_COLS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Embedding configurations (all 64-dim, res9, 2022)
# ---------------------------------------------------------------------------

EMBEDDINGS = {
    "concat": "data/study_areas/netherlands/stage2_multimodal/concat/embeddings/netherlands_res9_2022.parquet",
    "ring_agg": "data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_2022.parquet",
    "gcn": "data/study_areas/netherlands/stage2_multimodal/gcn/embeddings/netherlands_res9_2022.parquet",
}

TARGET_PATH = "data/study_areas/netherlands/target/leefbaarometer/leefbaarometer_h3res9_2022.parquet"


def run_probe(label: str, embeddings_path: str) -> Dict[str, Any]:
    """Run DNN probe for a given Stage 2 fusion embedding."""
    config = DNNProbeConfig(
        study_area="netherlands",
        year=2022,
        h3_resolution=9,
        modality=label,
        embeddings_path=embeddings_path,
        target_path=TARGET_PATH,
        # Architecture -- match previous session exactly
        hidden_dim=256,
        num_layers=3,
        activation="silu",
        # Training
        learning_rate=1e-4,
        max_epochs=200,
        patience=20,
        initial_batch_size=4096,
        weight_decay=1e-4,
        # CV
        n_folds=5,
        block_width=10_000,
        block_height=10_000,
        # Provenance
        run_descriptor=f"res9_stage2_fusion_{label}",
    )

    regressor = DNNProbeRegressor(config)
    results = regressor.run()
    regressor.save_results()

    # Extract R2 per target
    row = {"embedding": label}
    r2_values = []
    for tc in TARGET_COLS:
        if tc in results:
            r2 = results[tc].overall_r2
            row[f"r2_{tc}"] = r2
            r2_values.append(r2)
    row["mean_r2"] = float(np.mean(r2_values)) if r2_values else float("nan")
    row["output_dir"] = config.output_dir

    return row


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info("=" * 70)
    logger.info("DNN PROBE: STAGE 2 FUSION COMPARISON (CONCAT / RING AGG / GCN)")
    logger.info("=" * 70)

    all_rows = []
    total_start = time.time()

    for label, emb_path in EMBEDDINGS.items():
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Running: {label}")
        logger.info(f"  Path: {emb_path}")
        logger.info(f"{'=' * 70}")
        start = time.time()
        try:
            row = run_probe(label, emb_path)
            elapsed = time.time() - start
            row["duration_s"] = elapsed
            all_rows.append(row)
            logger.info(f"  --> {label}: mean_r2={row['mean_r2']:.4f} ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"  --> {label} FAILED after {elapsed:.0f}s: {e}")
            import traceback
            traceback.print_exc()
            all_rows.append({"embedding": label, "mean_r2": float("nan"), "duration_s": elapsed})

    total_elapsed = time.time() - total_start

    # Save summary CSV
    summary_df = pd.DataFrame(all_rows)
    from utils.paths import StudyAreaPaths
    paths = StudyAreaPaths("netherlands")
    out_dir = paths.stage3("dnn_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "2026-03-07_res9_stage2_fusion_comparison.csv"
    summary_df.to_csv(csv_path, index=False)

    # Print comparison table including previous concat reference
    prev_concat = {
        "lbm": 0.327, "fys": 0.450, "onv": 0.537,
        "soc": 0.669, "vrz": 0.779, "won": 0.501,
    }
    prev_mean = float(np.mean(list(prev_concat.values())))

    logger.info(f"\n{'=' * 90}")
    logger.info("RESULTS: DNN Probe R2 -- Stage 2 Fusion Comparison (res9, 2022)")
    logger.info(f"{'=' * 90}")
    header = f"{'Embedding':25s}" + "".join(f"  {t:>8s}" for t in TARGET_COLS) + f"  {'mean':>8s}"
    logger.info(header)
    logger.info("-" * 90)

    # Previous concat reference
    parts = [f"{'Concat (prev, ref)':25s}"]
    for t in TARGET_COLS:
        parts.append(f"  {prev_concat.get(t, float('nan')):8.4f}")
    parts.append(f"  {prev_mean:8.4f}")
    logger.info("".join(parts))

    # Current runs
    for _, row in summary_df.iterrows():
        parts = [f"{row['embedding']:25s}"]
        for t in TARGET_COLS:
            col = f"r2_{t}"
            if col in row and not pd.isna(row.get(col, float("nan"))):
                parts.append(f"  {row[col]:8.4f}")
            else:
                parts.append(f"  {'N/A':>8s}")
        parts.append(f"  {row['mean_r2']:8.4f}" if not pd.isna(row["mean_r2"]) else f"  {'N/A':>8s}")
        logger.info("".join(parts))

    logger.info(f"\nTotal time: {total_elapsed:.0f}s")
    logger.info(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
