#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fair DNN Probe Comparison: Concat-PCA-64D vs UNet-64D (20mix)

Runs DNN regression probes comparing concat embeddings (PCA-reduced to 64D)
against UNet embeddings (natively 64D) on leefbaarometer targets.

The key constraint: both representations are exactly 64D for a fair comparison.
(UNet outputs 64D, not 128D as previously documented.)

Output:
    - probe_fair_pca_regression.csv: R2 per target x embedding source
    - probe_fair_pca_regression.png: grouped bar chart
    - PCA variance explained summary

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/probe_fair_pca_comparison.py
"""

import argparse
import logging
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

# Shared DNN probe hyperparameters (same as probe_20mix_comparison.py)
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
# PCA reduction
# ---------------------------------------------------------------------------


def pca_reduce_concat(
    paths: StudyAreaPaths,
    n_components: int,
    output_dir: Path,
) -> Path:
    """Load concat 208D, PCA to n_components, save as parquet, return path."""
    concat_path = (
        paths.model_embeddings("concat")
        / f"{paths.study_area}_res{H3_RESOLUTION}_{YEAR}.parquet"
    )
    logger.info(f"Loading concat embeddings from {concat_path}")
    concat_df = pd.read_parquet(concat_path)
    logger.info(f"  Shape: {concat_df.shape}")

    # Standardize before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(concat_df.values)

    # PCA
    logger.info(f"Running PCA: {concat_df.shape[1]}D -> {n_components}D")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"  Variance explained: {variance_explained:.4f} ({variance_explained*100:.1f}%)")
    logger.info(f"  Per-component variance (top 10): {pca.explained_variance_ratio_[:10].round(4)}")

    # Save PCA summary
    pca_info = {
        "n_components": n_components,
        "original_dim": int(concat_df.shape[1]),
        "variance_explained": float(variance_explained),
        "per_component_variance": pca.explained_variance_ratio_.tolist(),
    }
    import json
    pca_info_path = output_dir / "pca_info.json"
    pca_info_path.write_text(json.dumps(pca_info, indent=2))
    logger.info(f"  Saved PCA info to {pca_info_path}")

    # Save as parquet
    pca_columns = [f"pca_{i}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, index=concat_df.index, columns=pca_columns)
    pca_df.index.name = "region_id"

    pca_path = output_dir / f"concat_pca{n_components}_res{H3_RESOLUTION}_{YEAR}.parquet"
    pca_df.to_parquet(pca_path)
    logger.info(f"  Saved PCA embeddings to {pca_path}")

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
    """Create a DNNProbeConfig for regression probing."""
    return DNNProbeConfig(
        study_area=STUDY_AREA,
        year=TARGET_YEAR,
        h3_resolution=H3_RESOLUTION,
        modality=modality,
        embeddings_path=embeddings_path,
        target_path=str(
            StudyAreaPaths(STUDY_AREA).target_file("leefbaarometer", H3_RESOLUTION, TARGET_YEAR)
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
        run_descriptor=f"fair_pca_{label}",
    )


def run_fair_comparison(output_dir: Path) -> pd.DataFrame:
    """Run DNN regression probes on PCA-64D concat and native 64D UNet."""
    paths = StudyAreaPaths(STUDY_AREA)

    # Step 1: Determine UNet dimensionality
    unet_path = (
        paths.model_embeddings("unet")
        / f"{paths.study_area}_res{H3_RESOLUTION}_{YEAR}.parquet"
    )
    unet_df = pd.read_parquet(unet_path)
    unet_dim = unet_df.shape[1]
    logger.info(f"UNet native dimensionality: {unet_dim}D")
    del unet_df

    # Step 2: PCA concat to match UNet dim
    pca_path = pca_reduce_concat(paths, n_components=unet_dim, output_dir=output_dir)

    # Step 3: Define sources
    sources = [
        {
            "name": f"Concat PCA-{unet_dim}D",
            "label": f"concat_pca{unet_dim}d",
            "path": str(pca_path),
            "modality": "concat",
        },
        {
            "name": f"UNet {unet_dim}D",
            "label": f"unet_{unet_dim}d",
            "path": str(unet_path),
            "modality": "unet",
        },
    ]

    logger.info("=" * 70)
    logger.info(f"FAIR DNN PROBE COMPARISON: Both at {unet_dim}D")
    logger.info("=" * 70)

    all_rows: List[Dict[str, Any]] = []

    for i, src in enumerate(sources, 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{i}/{len(sources)}] {src['name']}")
        logger.info(f"  Embeddings: {src['path']}")
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
                f"  {src['name']}: mean_r2={row['mean_r2']:.4f} "
                f"in {_format_duration(elapsed)}"
            )

        except Exception as e:
            elapsed = time.time() - run_start
            logger.error(f"  ERROR for {src['name']}: {e}", exc_info=True)
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
    csv_path = output_dir / "probe_fair_pca_regression.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved regression results to {csv_path}")

    # Print comparison table
    print_comparison_table(results_df, unet_dim)

    return results_df


def print_comparison_table(results_df: pd.DataFrame, dim: int) -> None:
    """Print a formatted comparison table."""
    logger.info("\n" + "=" * 90)
    logger.info(f"FAIR COMPARISON: Concat-PCA-{dim}D vs UNet-{dim}D (DNN Probe, 20mix)")
    logger.info("=" * 90)

    target_headers = "  ".join(f"{t:>8s}" for t in TARGET_COLS)
    logger.info(f"{'Source':25s}  {target_headers}  {'mean_r2':>8s}")
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
        logger.info(f"{row['name']:25s}  {r2_line}  {mean_str:>8s}")

    # Winner per target
    if len(results_df) == 2:
        logger.info("-" * 90)
        winner_strs = []
        wins = {"concat": 0, "unet": 0}
        for t in TARGET_COLS:
            col = f"r2_{t}"
            if col in results_df.columns:
                vals = results_df[col].values
                if not any(pd.isna(vals)):
                    if vals[0] > vals[1]:
                        winner_strs.append(f"{'Concat':>8s}")
                        wins["concat"] += 1
                    elif vals[1] > vals[0]:
                        winner_strs.append(f"{'UNet':>8s}")
                        wins["unet"] += 1
                    else:
                        winner_strs.append(f"{'Tie':>8s}")
                else:
                    winner_strs.append(f"{'N/A':>8s}")
            else:
                winner_strs.append(f"{'N/A':>8s}")

        # Overall winner
        if results_df.iloc[0]["mean_r2"] > results_df.iloc[1]["mean_r2"]:
            overall = "Concat"
        elif results_df.iloc[1]["mean_r2"] > results_df.iloc[0]["mean_r2"]:
            overall = "UNet"
        else:
            overall = "Tie"

        winner_line = "  ".join(winner_strs)
        logger.info(f"{'Winner':25s}  {winner_line}  {overall:>8s}")
        logger.info(f"\nConcat wins: {wins['concat']}/6, UNet wins: {wins['unet']}/6")

    logger.info("=" * 90)


def plot_comparison(results_df: pd.DataFrame, dim: int, output_dir: Path) -> Path:
    """Grouped bar chart: R2 per target for concat-PCA vs UNet."""
    target_cols_present = [c for c in TARGET_COLS if f"r2_{c}" in results_df.columns]
    if not target_cols_present:
        logger.warning("No target columns found for plotting")
        return output_dir / "probe_fair_pca_regression.png"

    n_targets = len(target_cols_present)
    n_models = len(results_df)
    x = np.arange(n_targets)
    width = 0.8 / n_models

    colors = ["#4C72B0", "#DD8452"]  # Blue for concat, orange for UNet

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row.get(f"r2_{t}", 0) for t in target_cols_present]
        stds = [row.get(f"r2_std_{t}", 0) for t in target_cols_present]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            yerr=stds, capsize=3,
            label=row["name"], color=colors[i],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=45,
            )

    ax.set_xlabel("Leefbaarometer Target", fontsize=12)
    ax.set_ylabel("R-squared (5-fold spatial block CV)", fontsize=12)
    ax.set_title(
        f"Fair DNN Regression Probe: Concat-PCA-{dim}D vs UNet-{dim}D (20mix)\n"
        f"(MLP h={SHARED_PARAMS['hidden_dim']}, "
        f"patience={SHARED_PARAMS['patience']}, "
        f"max_epochs={SHARED_PARAMS['max_epochs']})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [TARGET_NAMES.get(t, t) for t in target_cols_present], fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.set_ylim(0, min(1.0, results_df[[f"r2_{t}" for t in target_cols_present]].max().max() * 1.3))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = output_dir / "probe_fair_pca_regression.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison plot to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair DNN Probe: Concat-PCA vs UNet at equal dimensionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip generating comparison plot")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    paths = StudyAreaPaths(STUDY_AREA)
    experiment_dir = (
        paths.stage3("dnn_probe")
        / f"{date.today()}_fair_pca_comparison"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    results_df = run_fair_comparison(experiment_dir)

    # Determine UNet dim for plot title
    unet_path = (
        paths.model_embeddings("unet")
        / f"{paths.study_area}_res{H3_RESOLUTION}_{YEAR}.parquet"
    )
    unet_dim = pd.read_parquet(unet_path, columns=["unet_0"]).shape[0]
    # Re-read dim from results
    unet_df = pd.read_parquet(unet_path)
    unet_dim = unet_df.shape[1]
    del unet_df

    if not args.no_viz:
        plot_comparison(results_df, unet_dim, experiment_dir)

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal time: {_format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
