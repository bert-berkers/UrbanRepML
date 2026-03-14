#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe: Ring-Aggregated Concat (k=10, PCA-64D) vs Concat-PCA-64D

Tests whether simple spatial smoothing (10-hop neighbourhood averaging)
helps or hurts leefbaarometer prediction. Answers: is spatial context
valuable at all, or is leefbaarometer purely local?

Pipeline:
    1. Load 208D concat embeddings (res9, 20mix)
    2. Ring-aggregate with k=10 (exponential weighting)
    3. PCA to 64D (for fair comparison with UNet-64D)
    4. DNN probe with same setup as all other runs

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/probe_ring_agg_comparison.py
"""

import json
import logging
import pickle
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from stage2_fusion.models.ring_aggregation import SimpleRingAggregator
from stage3_analysis.dnn_probe import DNNProbeConfig, DNNProbeRegressor
from stage3_analysis.linear_probe import TARGET_COLS, TARGET_NAMES
from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDY_AREA = "netherlands"
H3_RESOLUTION = 9
YEAR = "20mix"
TARGET_YEAR = 2022
K_RINGS = 10
PCA_DIM = 64

# Shared DNN probe hyperparameters (identical to all other comparison scripts)
SHARED_PARAMS: Dict[str, Any] = {
    "hidden_dim": 256,
    "num_layers": 3,
    "activation": "silu",
    "learning_rate": 1e-4,
    "max_epochs": 200,
    "patience": 20,
    "initial_batch_size": 8192,
    "weight_decay": 1e-4,
    "n_folds": 5,
    "block_width": 10_000,
    "block_height": 10_000,
}


# ---------------------------------------------------------------------------
# Ring aggregation
# ---------------------------------------------------------------------------


def run_ring_aggregation(
    paths: StudyAreaPaths, output_dir: Path
) -> pd.DataFrame:
    """Load concat 208D, apply k=10 ring aggregation, return result."""
    # Load concat embeddings
    concat_path = (
        paths.model_embeddings("concat")
        / f"{paths.study_area}_res{H3_RESOLUTION}_{YEAR}.parquet"
    )
    logger.info("Loading concat embeddings from %s", concat_path)
    concat_df = pd.read_parquet(concat_path)
    logger.info("  Shape: %s", concat_df.shape)

    # Load neighbourhood
    nb_path = paths.neighbourhood_dir() / (
        f"{paths.study_area}_res{H3_RESOLUTION}_neighbourhood.pkl"
    )
    logger.info("Loading neighbourhood from %s", nb_path)
    with open(nb_path, "rb") as f:
        neighbourhood = pickle.load(f)

    # Run ring aggregation
    logger.info("Running ring aggregation with K=%d, weighting=exponential", K_RINGS)
    t0 = time.time()
    aggregator = SimpleRingAggregator(
        neighbourhood=neighbourhood,
        K=K_RINGS,
        weighting="exponential",
    )
    ring_df = aggregator.aggregate(concat_df)
    elapsed = time.time() - t0
    logger.info("  Ring aggregation completed in %.1f seconds", elapsed)
    logger.info("  Output shape: %s", ring_df.shape)

    # Save ring-aggregated embeddings
    ring_path = output_dir / f"ring_agg_k{K_RINGS}_res{H3_RESOLUTION}_{YEAR}.parquet"
    ring_df.to_parquet(ring_path)
    logger.info("  Saved ring-aggregated embeddings to %s", ring_path)

    return ring_df


# ---------------------------------------------------------------------------
# PCA reduction
# ---------------------------------------------------------------------------


def pca_reduce(
    df: pd.DataFrame,
    n_components: int,
    label: str,
    output_dir: Path,
) -> Path:
    """Standardize + PCA reduce, save parquet, return path."""
    logger.info("PCA: %dD -> %dD (%s)", df.shape[1], n_components, label)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_.sum()
    logger.info("  Variance explained: %.4f (%.1f%%)", var_explained, var_explained * 100)

    # Save PCA info
    pca_info = {
        "label": label,
        "n_components": n_components,
        "original_dim": int(df.shape[1]),
        "variance_explained": float(var_explained),
        "per_component_variance": pca.explained_variance_ratio_.tolist(),
    }
    info_path = output_dir / f"pca_info_{label}.json"
    info_path.write_text(json.dumps(pca_info, indent=2))

    # Save parquet
    cols = [f"pca_{i}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, index=df.index, columns=cols)
    pca_df.index.name = "region_id"

    pca_path = output_dir / f"{label}_pca{n_components}_res{H3_RESOLUTION}_{YEAR}.parquet"
    pca_df.to_parquet(pca_path)
    logger.info("  Saved PCA embeddings to %s", pca_path)

    return pca_path


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _build_config(label: str, embeddings_path: str, modality: str) -> DNNProbeConfig:
    """Create DNNProbeConfig for regression probing."""
    return DNNProbeConfig(
        study_area=STUDY_AREA,
        year=TARGET_YEAR,
        h3_resolution=H3_RESOLUTION,
        modality=modality,
        embeddings_path=embeddings_path,
        target_path=str(
            StudyAreaPaths(STUDY_AREA).target_file(
                "leefbaarometer", H3_RESOLUTION, TARGET_YEAR
            )
        ),
        hidden_dim=SHARED_PARAMS["hidden_dim"],
        num_layers=SHARED_PARAMS["num_layers"],
        activation=SHARED_PARAMS["activation"],
        learning_rate=SHARED_PARAMS["learning_rate"],
        max_epochs=SHARED_PARAMS["max_epochs"],
        patience=SHARED_PARAMS["patience"],
        initial_batch_size=SHARED_PARAMS["initial_batch_size"],
        weight_decay=SHARED_PARAMS["weight_decay"],
        n_folds=SHARED_PARAMS["n_folds"],
        block_width=SHARED_PARAMS["block_width"],
        block_height=SHARED_PARAMS["block_height"],
        run_descriptor=f"ring_agg_{label}",
    )


def run_probe(
    sources: List[Dict[str, Any]], output_dir: Path
) -> pd.DataFrame:
    """Run DNN regression probes on given sources."""
    all_rows: List[Dict[str, Any]] = []

    for i, src in enumerate(sources, 1):
        logger.info("=" * 70)
        logger.info("[%d/%d] %s", i, len(sources), src["name"])
        logger.info("  Embeddings: %s", src["path"])
        logger.info("=" * 70)

        run_start = time.time()
        try:
            config = _build_config(src["label"], src["path"], src["modality"])
            regressor = DNNProbeRegressor(config)
            results = regressor.run()
            regressor.save_results()

            row: Dict[str, Any] = {"name": src["name"], "label": src["label"]}
            r2_values = []
            for target_col in TARGET_COLS:
                if target_col in results:
                    r2 = results[target_col].overall_r2
                    row[f"r2_{target_col}"] = r2
                    row[f"r2_std_{target_col}"] = np.std(
                        [f.r2 for f in results[target_col].fold_metrics]
                    )
                    r2_values.append(r2)
            row["mean_r2"] = float(np.mean(r2_values)) if r2_values else float("nan")

            elapsed = time.time() - run_start
            row["duration_s"] = elapsed
            all_rows.append(row)

            logger.info(
                "  %s: mean_r2=%.4f in %s",
                src["name"], row["mean_r2"], _format_duration(elapsed),
            )

        except Exception as e:
            elapsed = time.time() - run_start
            logger.error("  ERROR for %s: %s", src["name"], e, exc_info=True)
            all_rows.append({
                "name": src["name"],
                "label": src["label"],
                "mean_r2": float("nan"),
                "duration_s": elapsed,
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = output_dir / "probe_ring_agg_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved results to %s", csv_path)

    return results_df


def print_comparison_table(results_df: pd.DataFrame) -> None:
    """Print formatted comparison table."""
    logger.info("\n" + "=" * 90)
    logger.info(
        "RING AGGREGATION COMPARISON: Concat-PCA-64D vs RingAgg-k%d-PCA-64D (DNN Probe, %s)",
        K_RINGS, YEAR,
    )
    logger.info("=" * 90)

    target_headers = "  ".join(f"{t:>8s}" for t in TARGET_COLS)
    logger.info(f"{'Source':30s}  {target_headers}  {'mean_r2':>8s}")
    logger.info("-" * 90)

    for _, row in results_df.iterrows():
        r2_strs = []
        for t in TARGET_COLS:
            col = f"r2_{t}"
            if col in row and not pd.isna(row.get(col, float("nan"))):
                r2_strs.append(f"{row[col]:8.4f}")
            else:
                r2_strs.append(f"{'N/A':>8s}")
        r2_line = "  ".join(r2_strs)
        mean_str = f"{row['mean_r2']:.4f}" if not pd.isna(row.get("mean_r2")) else "N/A"
        logger.info(f"{row['name']:30s}  {r2_line}  {mean_str:>8s}")

    logger.info("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    paths = StudyAreaPaths(STUDY_AREA)
    experiment_dir = (
        paths.stage3("dnn_probe")
        / f"{date.today()}_ring_agg_k{K_RINGS}_comparison"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Step 1: Ring aggregation on 208D concat
    logger.info("=" * 70)
    logger.info("STEP 1: Ring aggregation (k=%d) on 208D concat", K_RINGS)
    logger.info("=" * 70)
    ring_df = run_ring_aggregation(paths, experiment_dir)

    # Step 2: PCA both to 64D
    logger.info("=" * 70)
    logger.info("STEP 2: PCA to %dD", PCA_DIM)
    logger.info("=" * 70)

    # PCA the ring-aggregated embeddings
    ring_pca_path = pca_reduce(ring_df, PCA_DIM, "ring_agg", experiment_dir)
    del ring_df  # Free memory

    # Also PCA the raw concat for baseline (re-use if already done today)
    concat_path = (
        paths.model_embeddings("concat")
        / f"{paths.study_area}_res{H3_RESOLUTION}_{YEAR}.parquet"
    )
    concat_df = pd.read_parquet(concat_path)
    concat_pca_path = pca_reduce(concat_df, PCA_DIM, "concat", experiment_dir)
    del concat_df

    # Step 3: DNN probe both
    logger.info("=" * 70)
    logger.info("STEP 3: DNN probe comparison")
    logger.info("=" * 70)

    sources = [
        {
            "name": f"Concat-PCA-{PCA_DIM}D",
            "label": f"concat_pca{PCA_DIM}d",
            "path": str(concat_pca_path),
            "modality": "concat",
        },
        {
            "name": f"RingAgg-k{K_RINGS}-PCA-{PCA_DIM}D",
            "label": f"ring_agg_k{K_RINGS}_pca{PCA_DIM}d",
            "path": str(ring_pca_path),
            "modality": "ring_agg",
        },
    ]

    results_df = run_probe(sources, experiment_dir)
    print_comparison_table(results_df)

    total_elapsed = time.time() - total_start
    logger.info("\nTotal time: %s", _format_duration(total_elapsed))


if __name__ == "__main__":
    main()
