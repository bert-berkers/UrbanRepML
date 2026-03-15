#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe Comparison: SAGEConv UNet (64D + 192D) vs Concat baselines

Runs DNN regression probes comparing:
  1. SAGEConv-UNet-64D  (res9 only, native 64D)
  2. SAGEConv-UNet-192D (multiscale concat: res9+res8+res7, 3x64D)

Against previously established baselines (hardcoded from earlier runs):
  - Concat-PCA-64D:  mean R2 = 0.5085
  - GCN-UNet-64D:    mean R2 = 0.4793
  - Concat-PCA-192D: mean R2 = 0.5137
  - GCN-UNet-192D:   mean R2 = 0.4386

Probe setup: MLP (h=256, 3 layers, SiLU), 5-fold spatial block CV
(10km blocks), max 200 epochs patience 20.

Output:
    - probe_sageconv_comparison.csv: R2 per target x embedding source
    - probe_sageconv_comparison.png: grouped bar chart (all 6 conditions)
    - Full comparison table printed to console

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/probe_sageconv_comparison.py
"""

import argparse
import json
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

# Shared DNN probe hyperparameters (identical to previous comparisons)
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

# Previous results for the comparison table
PREVIOUS_RESULTS = {
    "Concat-PCA-64D": {
        "lbm": 0.2858, "fys": 0.4118, "onv": 0.5059,
        "soc": 0.6426, "vrz": 0.7382, "won": 0.4668,
        "mean": 0.5085,
    },
    "GCN-UNet-64D": {
        "lbm": 0.2305, "fys": 0.3080, "onv": 0.4934,
        "soc": 0.6470, "vrz": 0.7424, "won": 0.4545,
        "mean": 0.4793,
    },
    "Concat-PCA-192D": {
        "lbm": 0.2933, "fys": 0.4129, "onv": 0.5077,
        "soc": 0.6433, "vrz": 0.7585, "won": 0.4662,
        "mean": 0.5137,
    },
    "GCN-UNet-192D": {
        "lbm": 0.1689, "fys": 0.2018, "onv": 0.4722,
        "soc": 0.6095, "vrz": 0.7499, "won": 0.4291,
        "mean": 0.4386,
    },
}


# ---------------------------------------------------------------------------
# Helpers
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
        run_descriptor=f"sageconv_{label}",
    )


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def run_sageconv_probes(output_dir: Path) -> pd.DataFrame:
    """Run DNN probes on SAGEConv UNet at 64D and 192D."""
    paths = StudyAreaPaths(STUDY_AREA)

    # Locate SAGEConv UNet embedding files
    unet_64d_path = (
        paths.model_embeddings("unet")
        / f"{paths.study_area}_res{H3_RESOLUTION}_{YEAR}.parquet"
    )
    unet_192d_path = (
        paths.model_embeddings("unet")
        / f"{paths.study_area}_res{H3_RESOLUTION}_multiscale_concat_{YEAR}.parquet"
    )

    # Verify files exist
    for p, desc in [(unet_64d_path, "UNet 64D"), (unet_192d_path, "UNet 192D multiscale")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{desc} not found: {p}\n"
                "Run: python scripts/stage2/extract_highway_exits.py "
                "--study-area netherlands --year 20mix"
            )

    # Verify shapes
    df64 = pd.read_parquet(unet_64d_path)
    df192 = pd.read_parquet(unet_192d_path)
    logger.info(f"SAGEConv-UNet-64D:  {df64.shape}")
    logger.info(f"SAGEConv-UNet-192D: {df192.shape}")
    del df64, df192

    # Define probe sources
    sources = [
        {
            "name": "SAGEConv-UNet-64D",
            "label": "sageconv_unet_64d",
            "path": str(unet_64d_path),
            "modality": "unet",
        },
        {
            "name": "SAGEConv-UNet-192D",
            "label": "sageconv_unet_192d",
            "path": str(unet_192d_path),
            "modality": "unet",
        },
    ]

    logger.info("=" * 70)
    logger.info("SAGEConv UNet DNN PROBE COMPARISON")
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
    csv_path = output_dir / "probe_sageconv_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # Print full comparison table
    print_full_comparison_table(results_df)

    return results_df


def print_full_comparison_table(new_results_df: pd.DataFrame) -> None:
    """Print the 6-condition comparison table including all previous results."""
    print("\n" + "=" * 120)
    print("FULL COMPARISON: Concat-PCA vs GCN-UNet vs SAGEConv-UNet (DNN Probe, 20mix)")
    print("=" * 120)

    # Build all conditions
    conditions = []

    # Previous baselines
    for name, vals in PREVIOUS_RESULTS.items():
        row = {"name": name}
        for t in TARGET_COLS:
            row[f"r2_{t}"] = vals.get(t, float("nan"))
        row["mean_r2"] = vals["mean"]
        conditions.append(row)

    # New SAGEConv results
    for _, row in new_results_df.iterrows():
        conditions.append(dict(row))

    # Header
    target_headers = "  ".join(f"{t:>8s}" for t in TARGET_COLS)
    print(f"{'Source':25s}  {target_headers}  {'mean_r2':>8s}  {'Wins':>5s}")
    print("-" * 120)

    # Find per-target winners
    per_target_best = {}
    for t in TARGET_COLS:
        best_val = -999
        best_name = ""
        for c in conditions:
            val = c.get(f"r2_{t}", float("nan"))
            if not pd.isna(val) and val > best_val:
                best_val = val
                best_name = c["name"]
        per_target_best[t] = best_name

    # Print rows
    for c in conditions:
        r2_strs = []
        wins = 0
        for t in TARGET_COLS:
            val = c.get(f"r2_{t}", float("nan"))
            if not pd.isna(val):
                marker = " *" if per_target_best[t] == c["name"] else "  "
                r2_strs.append(f"{val:6.4f}{marker}")
            else:
                r2_strs.append(f"{'N/A':>8s}")
            if per_target_best[t] == c["name"]:
                wins += 1
        r2_line = "  ".join(r2_strs)
        mean_str = f"{c['mean_r2']:.4f}" if not pd.isna(c.get("mean_r2")) else "N/A"
        print(f"{c['name']:25s}  {r2_line}  {mean_str:>8s}  {wins:>4d}/6")

    # Overall winner
    print("-" * 120)
    best_overall = max(conditions, key=lambda c: c.get("mean_r2", -999))
    print(f"Overall winner: {best_overall['name']} (mean R2 = {best_overall['mean_r2']:.4f})")
    print("* = best for that target")
    print("=" * 120)


def plot_comparison(new_results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: R2 per target for all 6 conditions."""
    all_rows = []

    for name, vals in PREVIOUS_RESULTS.items():
        row = {"name": name}
        for t in TARGET_COLS:
            row[f"r2_{t}"] = vals.get(t, float("nan"))
        row["mean_r2"] = vals["mean"]
        all_rows.append(row)

    for _, row in new_results_df.iterrows():
        all_rows.append(dict(row))

    all_df = pd.DataFrame(all_rows)

    target_cols_present = [c for c in TARGET_COLS if f"r2_{c}" in all_df.columns]
    n_targets = len(target_cols_present)
    n_models = len(all_df)
    x = np.arange(n_targets)
    width = 0.8 / n_models

    # 6 distinct colors: concat variants in blues, GCN in oranges, SAGEConv in greens
    colors = [
        "#A0CBE8",  # Concat-PCA-64D (light blue)
        "#DD8452",  # GCN-UNet-64D (orange)
        "#4C72B0",  # Concat-PCA-192D (dark blue)
        "#C44E52",  # GCN-UNet-192D (red)
        "#55A868",  # SAGEConv-UNet-64D (green)
        "#8C8C8C",  # SAGEConv-UNet-192D (gray-green)
    ]

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, (_, row) in enumerate(all_df.iterrows()):
        vals = [row.get(f"r2_{t}", 0) for t in target_cols_present]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=row["name"], color=colors[i % len(colors)],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=60,
                )

    ax.set_xlabel("Leefbaarometer Target", fontsize=12)
    ax.set_ylabel("R-squared (5-fold spatial block CV)", fontsize=12)
    ax.set_title(
        "DNN Probe: Concat-PCA vs GCN-UNet vs SAGEConv-UNet\n"
        f"(MLP h={SHARED_PARAMS['hidden_dim']}, "
        f"{SHARED_PARAMS['num_layers']} layers, SiLU, "
        f"patience={SHARED_PARAMS['patience']}, "
        f"max_epochs={SHARED_PARAMS['max_epochs']})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [TARGET_NAMES.get(t, t) for t in target_cols_present], fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    max_r2 = all_df[[f"r2_{t}" for t in target_cols_present]].max().max()
    ax.set_ylim(0, min(1.0, max_r2 * 1.2))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = output_dir / "probe_sageconv_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison plot to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DNN Probe: SAGEConv UNet at 64D and 192D vs baselines",
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
        / f"{date.today()}_sageconv_comparison"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    results_df = run_sageconv_probes(experiment_dir)

    if not args.no_viz:
        plot_comparison(results_df, experiment_dir)

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal time: {_format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
