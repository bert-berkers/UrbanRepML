#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe: Res9 2022 Modality Comparison

Runs DNN regression probes on res9 2022 embeddings against leefbaarometer targets
for each modality individually and all three concatenated.

Configurations:
    1. AlphaEarth alone (65 dims)
    2. POI hex2vec alone (50 dims)
    3. Roads highway2vec alone (30 dims)
    4. All three concatenated (145 dims)

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
from stage3_analysis.linear_probe import TARGET_COLS, TARGET_NAMES

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Data paths (all res9, 2022)
# ---------------------------------------------------------------------------

ALPHAEARTH_PATH = "data/study_areas/netherlands/stage1_unimodal/alphaearth/netherlands_res9_2022.parquet"
POI_PATH = "data/study_areas/netherlands/stage1_unimodal/poi/hex2vec/netherlands_res9_2022.parquet"
ROADS_PATH = "data/study_areas/netherlands/stage1_unimodal/roads/netherlands_res9_2022.parquet"
TARGET_PATH = "data/study_areas/netherlands/target/leefbaarometer/leefbaarometer_h3res9_2022.parquet"


def prepare_alphaearth() -> pd.DataFrame:
    """Load AlphaEarth and normalize index/columns."""
    df = pd.read_parquet(PROJECT_ROOT / ALPHAEARTH_PATH)
    # Rename h3_index to region_id for consistency
    if "h3_index" in df.columns:
        df = df.rename(columns={"h3_index": "region_id"})
    if df.index.name != "region_id" and "region_id" in df.columns:
        df = df.set_index("region_id")
    # Keep only embedding columns (A00..A65)
    emb_cols = [c for c in df.columns if len(c) >= 2 and c[0] == "A" and c[1:].isdigit()]
    return df[emb_cols]


def prepare_poi() -> pd.DataFrame:
    """Load POI hex2vec embeddings."""
    df = pd.read_parquet(PROJECT_ROOT / POI_PATH)
    if df.index.name != "region_id" and "region_id" in df.columns:
        df = df.set_index("region_id")
    # Rename hex2vec_X -> P_X for prefix consistency
    rename = {c: f"P{c.replace('hex2vec_', '').zfill(2)}" for c in df.columns if c.startswith("hex2vec_")}
    df = df.rename(columns=rename)
    return df


def prepare_roads() -> pd.DataFrame:
    """Load Roads highway2vec embeddings."""
    df = pd.read_parquet(PROJECT_ROOT / ROADS_PATH)
    if df.index.name != "region_id" and "region_id" in df.columns:
        df = df.set_index("region_id")
    emb_cols = [c for c in df.columns if len(c) >= 2 and c[0] == "R" and c[1:].isdigit()]
    return df[emb_cols]


def prepare_concat(ae: pd.DataFrame, poi: pd.DataFrame, roads: pd.DataFrame) -> pd.DataFrame:
    """Concatenate all modalities, inner-joining on region_id."""
    merged = ae.join(poi, how="inner").join(roads, how="inner")
    logger.info(f"Concat: {ae.shape[1]} + {poi.shape[1]} + {roads.shape[1]} = {merged.shape[1]} dims, {len(merged):,} regions")
    return merged


def save_temp_embeddings(df: pd.DataFrame, label: str) -> Path:
    """Save a temporary parquet for the DNN probe to load."""
    tmp_dir = PROJECT_ROOT / "data" / "study_areas" / "netherlands" / "stage1_unimodal" / "_tmp_probe"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{label}_res9_2022.parquet"
    df.to_parquet(path)
    return path


def run_probe(label: str, embeddings_path: str, hidden_dim: int = 128) -> Dict[str, Any]:
    """Run DNN probe for a given embedding configuration."""
    config = DNNProbeConfig(
        study_area="netherlands",
        year=2022,
        h3_resolution=9,
        modality=label,
        embeddings_path=embeddings_path,
        target_path=TARGET_PATH,
        # Architecture -- scale hidden_dim with input size
        hidden_dim=hidden_dim,
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
        run_descriptor=f"res9_{label}",
    )

    regressor = DNNProbeRegressor(config)
    results = regressor.run()
    regressor.save_results()

    # Extract R2 per target
    row = {"modality": label}
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
    logger.info("DNN PROBE: RES9 2022 MODALITY COMPARISON")
    logger.info("=" * 70)

    # Prepare all embeddings
    logger.info("\n--- Loading embeddings ---")
    ae_df = prepare_alphaearth()
    logger.info(f"AlphaEarth: {ae_df.shape}")
    poi_df = prepare_poi()
    logger.info(f"POI hex2vec: {poi_df.shape}")
    roads_df = prepare_roads()
    logger.info(f"Roads highway2vec: {roads_df.shape}")
    concat_df = prepare_concat(ae_df, poi_df, roads_df)

    # Save temp files for DNN probe loader
    ae_path = save_temp_embeddings(ae_df, "alphaearth")
    poi_path = save_temp_embeddings(poi_df, "poi_hex2vec")
    roads_path = save_temp_embeddings(roads_df, "roads_h2v")
    concat_path = save_temp_embeddings(concat_df, "concat_all")

    # Run probes
    configs = [
        ("alphaearth", str(ae_path.relative_to(PROJECT_ROOT)), 128),
        ("poi_hex2vec", str(poi_path.relative_to(PROJECT_ROOT)), 128),
        ("roads_h2v", str(roads_path.relative_to(PROJECT_ROOT)), 128),
        ("concat_all", str(concat_path.relative_to(PROJECT_ROOT)), 256),
    ]

    all_rows = []
    total_start = time.time()

    for label, emb_path, hdim in configs:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Running: {label} (hidden_dim={hdim})")
        logger.info(f"{'=' * 70}")
        start = time.time()
        try:
            row = run_probe(label, emb_path, hidden_dim=hdim)
            elapsed = time.time() - start
            row["duration_s"] = elapsed
            all_rows.append(row)
            logger.info(f"  --> {label}: mean_r2={row['mean_r2']:.4f} ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"  --> {label} FAILED after {elapsed:.0f}s: {e}")
            import traceback
            traceback.print_exc()
            all_rows.append({"modality": label, "mean_r2": float("nan"), "duration_s": elapsed})

    total_elapsed = time.time() - total_start

    # Summary table
    summary_df = pd.DataFrame(all_rows)
    from utils.paths import StudyAreaPaths
    paths = StudyAreaPaths("netherlands")
    out_dir = paths.stage3("dnn_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "res9_modality_comparison.csv"
    summary_df.to_csv(csv_path, index=False)

    # Print results
    logger.info(f"\n{'=' * 90}")
    logger.info("RESULTS: DNN Probe R2 per target (res9, 2022)")
    logger.info(f"{'=' * 90}")
    header = f"{'Modality':20s}" + "".join(f"  {t:>8s}" for t in TARGET_COLS) + f"  {'mean':>8s}"
    logger.info(header)
    logger.info("-" * 90)

    for _, row in summary_df.iterrows():
        parts = [f"{row['modality']:20s}"]
        for t in TARGET_COLS:
            col = f"r2_{t}"
            if col in row and not pd.isna(row.get(col, float("nan"))):
                parts.append(f"  {row[col]:8.4f}")
            else:
                parts.append(f"  {'N/A':>8s}")
        parts.append(f"  {row['mean_r2']:8.4f}" if not pd.isna(row["mean_r2"]) else f"  {'N/A':>8s}")
        logger.info("".join(parts))

    logger.info(f"\nTotal time: {total_elapsed:.0f}s")
    logger.info(f"Results saved to: {csv_path}")

    # Cleanup temp files
    for p in [ae_path, poi_path, roads_path, concat_path]:
        try:
            p.unlink()
        except Exception:
            pass
    try:
        (PROJECT_ROOT / "data" / "study_areas" / "netherlands" / "stage1_unimodal" / "_tmp_probe").rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
