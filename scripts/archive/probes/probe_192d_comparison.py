#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fair DNN Probe Comparison: Concat-PCA-192D vs UNet-192D (multiscale)

Runs DNN regression probes comparing:
  1. Concat embeddings (208D) PCA-reduced to 192D
  2. UNet multiscale concat (192D = 3x64D from res9+res8+res7)

Both are exactly 192D for a fair comparison. Uses the same probe setup
as the 64D comparison: MLP (h=256, 3 layers, SiLU), 5-fold spatial
block CV (10km blocks), max 200 epochs patience 20.

Output:
    - probe_192d_comparison.csv: R2 per target x embedding source
    - probe_192d_comparison.png: grouped bar chart
    - Comparison table printed to console

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/probe_192d_comparison.py
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
PCA_DIM = 192

# Shared DNN probe hyperparameters (identical to 64D comparison)
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

# Wave 1 reference results (64D comparison)
WAVE1_RESULTS = {
    "Concat-PCA-64D": {
        "lbm": 0.2858, "fys": 0.4118, "onv": 0.5059,
        "soc": 0.6426, "vrz": 0.7382, "won": 0.4668,
        "mean": 0.5085,
    },
    "UNet-64D": {
        "lbm": 0.2305, "fys": 0.3080, "onv": 0.4934,
        "soc": 0.6470, "vrz": 0.7424, "won": 0.4545,
        "mean": 0.4793,
    },
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
    logger.info(
        f"  Variance explained: {variance_explained:.4f} "
        f"({variance_explained*100:.1f}%)"
    )

    # Save PCA summary
    pca_info = {
        "n_components": n_components,
        "original_dim": int(concat_df.shape[1]),
        "variance_explained": float(variance_explained),
    }
    pca_info_path = output_dir / "pca_info.json"
    pca_info_path.write_text(json.dumps(pca_info, indent=2))

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
        run_descriptor=f"192d_{label}",
    )


def run_comparison(output_dir: Path) -> pd.DataFrame:
    """Run DNN regression probes on PCA-192D concat and UNet 192D multiscale."""
    paths = StudyAreaPaths(STUDY_AREA)

    # Step 1: PCA concat to 192D
    pca_path = pca_reduce_concat(paths, n_components=PCA_DIM, output_dir=output_dir)

    # Step 2: UNet multiscale concat path
    unet_multiscale_path = (
        paths.model_embeddings("unet")
        / f"{paths.study_area}_res{H3_RESOLUTION}_multiscale_concat_{YEAR}.parquet"
    )
    if not unet_multiscale_path.exists():
        raise FileNotFoundError(
            f"UNet multiscale concat not found: {unet_multiscale_path}\n"
            "Run: python scripts/stage2/extract_highway_exits.py --study-area netherlands --year 20mix"
        )

    # Verify dimensions
    unet_df = pd.read_parquet(unet_multiscale_path)
    logger.info(f"UNet multiscale concat: {unet_df.shape}")
    assert unet_df.shape[1] == PCA_DIM, (
        f"Expected {PCA_DIM}D, got {unet_df.shape[1]}D"
    )
    del unet_df

    # Step 3: Define sources
    sources = [
        {
            "name": f"Concat-PCA-{PCA_DIM}D",
            "label": f"concat_pca{PCA_DIM}d",
            "path": str(pca_path),
            "modality": "concat",
        },
        {
            "name": f"UNet-{PCA_DIM}D",
            "label": f"unet_{PCA_DIM}d",
            "path": str(unet_multiscale_path),
            "modality": "unet",
        },
    ]

    logger.info("=" * 70)
    logger.info(f"DNN PROBE COMPARISON: Both at {PCA_DIM}D")
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
    csv_path = output_dir / "probe_192d_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved regression results to {csv_path}")

    # Print comparison table
    print_full_comparison_table(results_df)

    return results_df


def print_full_comparison_table(results_df: pd.DataFrame) -> None:
    """Print the 4-condition comparison table including Wave 1 results."""
    print("\n" + "=" * 110)
    print(f"FULL COMPARISON: 64D (Wave 1) + 192D (Wave 2)")
    print("=" * 110)

    # Build rows for all 4 conditions
    conditions = []

    # Wave 1 reference
    for name, vals in WAVE1_RESULTS.items():
        row = {"name": name}
        for t in TARGET_COLS:
            row[f"r2_{t}"] = vals.get(t, float("nan"))
        row["mean_r2"] = vals["mean"]
        conditions.append(row)

    # Wave 2 new results
    for _, row in results_df.iterrows():
        conditions.append(dict(row))

    # Header
    target_headers = "  ".join(f"{t:>8s}" for t in TARGET_COLS)
    print(f"{'Source':25s}  {target_headers}  {'mean_r2':>8s}  {'Winner':>8s}")
    print("-" * 110)

    # Find per-target winners across all conditions
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
        print(f"{c['name']:25s}  {r2_line}  {mean_str:>8s}  {wins:>5d}/6")

    # Overall winner
    print("-" * 110)
    best_overall = max(conditions, key=lambda c: c.get("mean_r2", -999))
    print(f"Overall winner: {best_overall['name']} (mean R2 = {best_overall['mean_r2']:.4f})")
    print("* = best for that target")
    print("=" * 110)


def plot_comparison(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: R2 per target for all 4 conditions."""
    # Build all-conditions dataframe
    all_rows = []

    for name, vals in WAVE1_RESULTS.items():
        row = {"name": name}
        for t in TARGET_COLS:
            row[f"r2_{t}"] = vals.get(t, float("nan"))
        row["mean_r2"] = vals["mean"]
        all_rows.append(row)

    for _, row in results_df.iterrows():
        all_rows.append(dict(row))

    all_df = pd.DataFrame(all_rows)

    target_cols_present = [c for c in TARGET_COLS if f"r2_{c}" in all_df.columns]
    n_targets = len(target_cols_present)
    n_models = len(all_df)
    x = np.arange(n_targets)
    width = 0.8 / n_models

    colors = ["#A0CBE8", "#4C72B0", "#FFB482", "#DD8452"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (_, row) in enumerate(all_df.iterrows()):
        vals = [row.get(f"r2_{t}", 0) for t in target_cols_present]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=row["name"], color=colors[i],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xlabel("Leefbaarometer Target", fontsize=12)
    ax.set_ylabel("R-squared (5-fold spatial block CV)", fontsize=12)
    ax.set_title(
        f"DNN Probe Comparison: Concat-PCA vs UNet at 64D and 192D\n"
        f"(MLP h={SHARED_PARAMS['hidden_dim']}, "
        f"{SHARED_PARAMS['num_layers']} layers, "
        f"patience={SHARED_PARAMS['patience']}, "
        f"max_epochs={SHARED_PARAMS['max_epochs']})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [TARGET_NAMES.get(t, t) for t in target_cols_present], fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, min(1.0, all_df[[f"r2_{t}" for t in target_cols_present]].max().max() * 1.25))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = output_dir / "probe_192d_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison plot to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair DNN Probe: Concat-PCA-192D vs UNet-192D multiscale",
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
        / f"{date.today()}_192d_comparison"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    results_df = run_comparison(experiment_dir)

    if not args.no_viz:
        plot_comparison(results_df, experiment_dir)

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal time: {_format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
