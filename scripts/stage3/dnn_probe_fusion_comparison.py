#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe Comparison Across All Stage 2 Fusion Levels

Runs identical DNN probes (MLP, hidden_dim=256, patience=20, max_epochs=200)
on all 4 fusion levels to answer: does multi-resolution UNet beat single-res GCN?

Fusion levels:
    1. Concat (PCA-64): baseline concatenation of modality embeddings
    2. Ring Aggregation (64D): spatial smoothing via ring neighbors
    3. Lattice GCN (64D): learned message-passing on H3 lattice
    4. FullAreaUNet (128D): multi-resolution U-Net encoder-decoder

Output:
    - Per-model DNN probe results in stage3_analysis/dnn_probe/ run dirs
    - fusion_comparison_dnn.csv: R2 per target x fusion level
    - fusion_comparison_dnn.png: grouped bar chart

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/dnn_probe_fusion_comparison.py
    python scripts/stage3/dnn_probe_fusion_comparison.py --dry-run
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from stage3_analysis.dnn_probe import DNNProbeConfig, DNNProbeRegressor
from stage3_analysis.linear_probe import TARGET_COLS, TARGET_NAMES
from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDY_AREA = "netherlands"
H3_RESOLUTION = 9
YEAR = 2022

# Fusion levels to compare
FUSION_LEVELS = [
    {"name": "Concat (PCA-64)", "model": "concat", "label": "concat"},
    {"name": "Ring Aggregation", "model": "ring_agg", "label": "ring_agg"},
    {"name": "Lattice GCN", "model": "gcn", "label": "gcn"},
    {"name": "FullAreaUNet", "model": "unet", "label": "unet"},
]

# Shared DNN probe hyperparameters (as specified in task)
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


def _build_config(
    model_name: str,
    label: str,
    embeddings_path: str,
) -> DNNProbeConfig:
    """Create a DNNProbeConfig for the given fusion level."""
    return DNNProbeConfig(
        study_area=STUDY_AREA,
        year=YEAR,
        h3_resolution=H3_RESOLUTION,
        modality=model_name,
        embeddings_path=embeddings_path,
        # MLP architecture
        hidden_dim=SHARED_PARAMS["hidden_dim"],
        num_layers=SHARED_PARAMS["num_layers"],
        activation=SHARED_PARAMS["activation"],
        # Training
        learning_rate=SHARED_PARAMS["learning_rate"],
        max_epochs=SHARED_PARAMS["max_epochs"],
        patience=SHARED_PARAMS["patience"],
        initial_batch_size=SHARED_PARAMS["initial_batch_size"],
        weight_decay=SHARED_PARAMS["weight_decay"],
        # Spatial CV
        n_folds=SHARED_PARAMS["n_folds"],
        block_width=SHARED_PARAMS["block_width"],
        block_height=SHARED_PARAMS["block_height"],
        # Provenance
        run_descriptor=f"fusion_compare_{label}",
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _plot_comparison(
    results_df: pd.DataFrame, output_dir: Path
) -> Path:
    """Grouped bar chart: R2 per target for each fusion level."""
    target_cols_present = [
        c for c in TARGET_COLS if f"r2_{c}" in results_df.columns
    ]
    if not target_cols_present:
        logger.warning("No target columns found for plotting")
        return output_dir / "fusion_comparison_dnn.png"

    n_targets = len(target_cols_present)
    n_models = len(results_df)
    x = np.arange(n_targets)
    width = 0.8 / n_models

    # Use a colorblind-safe palette
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_models))

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row.get(f"r2_{t}", 0) for t in target_cols_present]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=row["name"],
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_xlabel("Leefbaarometer Target", fontsize=12)
    ax.set_ylabel("R-squared (5-fold spatial block CV)", fontsize=12)
    ax.set_title(
        "DNN Probe Comparison: Stage 2 Fusion Progression\n"
        f"(MLP h={SHARED_PARAMS['hidden_dim']}, "
        f"patience={SHARED_PARAMS['patience']}, "
        f"max_epochs={SHARED_PARAMS['max_epochs']})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [TARGET_NAMES.get(t, t) for t in target_cols_present],
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_ylim(0, min(1.0, results_df[[f"r2_{t}" for t in target_cols_present]].max().max() * 1.25))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = output_dir / "fusion_comparison_dnn.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison plot to {out_path}")
    return out_path


def _plot_mean_r2_progression(
    results_df: pd.DataFrame, output_dir: Path
) -> Path:
    """Line plot showing mean R2 progression across fusion levels."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = results_df["name"].values
    mean_r2 = results_df["mean_r2"].values

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))

    ax.plot(range(len(names)), mean_r2, "o-", color="gray", linewidth=1.5, zorder=1)
    for i, (name, val) in enumerate(zip(names, mean_r2)):
        ax.scatter(i, val, s=120, color=colors[i], zorder=2, edgecolors="black", linewidth=0.5)
        ax.annotate(
            f"{val:.4f}",
            (i, val),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Mean R-squared", fontsize=12)
    ax.set_title("Fusion Level Progression: Mean R2 (DNN Probe)", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = output_dir / "fusion_progression_dnn.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved progression plot to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_comparison(args: argparse.Namespace) -> None:
    """Run DNN probes on all fusion levels."""
    paths = StudyAreaPaths(STUDY_AREA)

    logger.info("=" * 70)
    logger.info("DNN PROBE FUSION COMPARISON")
    logger.info("=" * 70)
    logger.info(f"Fusion levels:  {len(FUSION_LEVELS)}")
    logger.info(f"Hidden dim:     {SHARED_PARAMS['hidden_dim']}")
    logger.info(f"Max epochs:     {SHARED_PARAMS['max_epochs']}")
    logger.info(f"Patience:       {SHARED_PARAMS['patience']}")
    logger.info(f"Targets:        {list(TARGET_COLS)}")

    if args.dry_run:
        for fl in FUSION_LEVELS:
            emb_path = paths.fused_embedding_file(fl["model"], H3_RESOLUTION, YEAR)
            emb = pd.read_parquet(emb_path)
            logger.info(
                f"  {fl['name']:20s}  model={fl['model']:10s}  "
                f"shape={emb.shape}  path={emb_path}"
            )
        logger.info("Dry run complete.")
        return

    all_rows: List[Dict[str, Any]] = []
    total_start = time.time()

    for i, fl in enumerate(FUSION_LEVELS, 1):
        model_name = fl["model"]
        display_name = fl["name"]
        label = fl["label"]

        emb_path = str(
            paths.fused_embedding_file(model_name, H3_RESOLUTION, YEAR)
        )

        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{i}/{len(FUSION_LEVELS)}] {display_name}")
        logger.info(f"  Embeddings: {emb_path}")
        logger.info("=" * 70)

        run_start = time.time()
        try:
            config = _build_config(model_name, label, emb_path)
            regressor = DNNProbeRegressor(config)
            results = regressor.run()
            regressor.save_results()

            row: Dict[str, Any] = {
                "name": display_name,
                "model": model_name,
            }
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
                f"  {display_name}: mean_r2={row['mean_r2']:.4f} "
                f"in {_format_duration(elapsed)}"
            )

        except Exception as e:
            elapsed = time.time() - run_start
            logger.error(f"  ERROR for {display_name}: {e}")
            all_rows.append({
                "name": display_name,
                "model": model_name,
                "mean_r2": float("nan"),
                "duration_s": elapsed,
            })

        # Clean GPU memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal time: {_format_duration(total_elapsed)}")

    # ------------------------------------------------------------------
    # Build results DataFrame
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_rows)

    output_dir = paths.stage3("dnn_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "fusion_comparison_dnn.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # Also save to the stage2_multimodal directory for easy access
    csv_path2 = (
        Path("data/study_areas/netherlands/stage2_multimodal")
        / "fusion_comparison_dnn.csv"
    )
    results_df.to_csv(csv_path2, index=False)
    logger.info(f"Saved results to {csv_path2}")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 90)
    logger.info("FUSION COMPARISON RESULTS (DNN Probe, hidden_dim=256)")
    logger.info("=" * 90)

    target_headers = "  ".join(f"{t:>8s}" for t in TARGET_COLS)
    logger.info(
        f"{'Fusion Level':22s}  {target_headers}  {'mean_r2':>8s}  {'time':>8s}"
    )
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

        mean_str = (
            f"{row['mean_r2']:.4f}"
            if not pd.isna(row.get("mean_r2", float("nan")))
            else "N/A"
        )
        dur_str = _format_duration(row.get("duration_s", 0))

        logger.info(f"{row['name']:22s}  {r2_line}  {mean_str:>8s}  {dur_str:>8s}")

    logger.info("=" * 90)

    # Key question
    valid = results_df.dropna(subset=["mean_r2"])
    if not valid.empty:
        best = valid.loc[valid["mean_r2"].idxmax()]
        gcn_row = valid[valid["model"] == "gcn"]
        unet_row = valid[valid["model"] == "unet"]
        if not gcn_row.empty and not unet_row.empty:
            gcn_r2 = gcn_row.iloc[0]["mean_r2"]
            unet_r2 = unet_row.iloc[0]["mean_r2"]
            delta = unet_r2 - gcn_r2
            verdict = "YES" if delta > 0 else "NO"
            logger.info(
                f"\nKEY QUESTION: Does UNet beat GCN?  {verdict}  "
                f"(UNet={unet_r2:.4f}, GCN={gcn_r2:.4f}, delta={delta:+.4f})"
            )

        logger.info(f"Best overall: {best['name']} (mean_r2={best['mean_r2']:.4f})")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    if not args.no_viz:
        _plot_comparison(results_df, output_dir)
        _plot_mean_r2_progression(results_df, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DNN Probe Comparison Across Fusion Levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print embedding info without training",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip generating comparison plots",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_comparison(args)


if __name__ == "__main__":
    main()
