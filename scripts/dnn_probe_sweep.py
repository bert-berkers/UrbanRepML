#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe Hyperparameter Sweep

Runs a grid of MLP configurations over fused (concat) embeddings to find the
best hidden_dim / num_layers / learning_rate combination for leefbaarometer
regression.  Each configuration trains a full 5-fold spatial block CV via
DNNProbeRegressor.

Output:
    - Per-run results in dated run directories under stage3_analysis/dnn_probe/
    - sweep_summary.csv with R2 per target for every configuration
    - sweep_heatmap.png  (mean R2 vs hidden_dim x lr, 3-layer runs only)
    - sweep_bar_chart.png (all configs sorted by mean R2)

Usage:
    python scripts/dnn_probe_sweep.py
    python scripts/dnn_probe_sweep.py --dry-run
    python scripts/dnn_probe_sweep.py --max-epochs 50 --no-viz
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
STAGE2_MODEL = "concat"
H3_RESOLUTION = 10
YEAR = 2022

# Shared parameters across all sweep runs (RTX 3090, 24 GB VRAM)
SHARED_PARAMS: Dict[str, Any] = {
    "activation": "silu",
    "max_epochs": 300,
    "patience": 30,
    "initial_batch_size": 8192,
    "weight_decay": 1e-4,
    "n_folds": 5,
    "block_width": 10_000,
    "block_height": 10_000,
}

# 10-run configuration grid
CONFIGS: List[Dict[str, Any]] = [
    {"lr": 5e-4, "hidden_dim": 32,  "num_layers": 3, "label": "baseline_h32"},
    {"lr": 1e-4, "hidden_dim": 32,  "num_layers": 3, "label": "lr1e4_h32"},
    {"lr": 5e-5, "hidden_dim": 32,  "num_layers": 3, "label": "lr5e5_h32"},
    {"lr": 1e-4, "hidden_dim": 128, "num_layers": 3, "label": "lr1e4_h128"},
    {"lr": 1e-4, "hidden_dim": 256, "num_layers": 3, "label": "lr1e4_h256"},
    {"lr": 1e-4, "hidden_dim": 512, "num_layers": 3, "label": "lr1e4_h512"},
    {"lr": 1e-4, "hidden_dim": 256, "num_layers": 4, "label": "lr1e4_h256_L4"},
    {"lr": 1e-4, "hidden_dim": 256, "num_layers": 5, "label": "lr1e4_h256_L5"},
    {"lr": 5e-5, "hidden_dim": 256, "num_layers": 3, "label": "lr5e5_h256"},
    {"lr": 1e-5, "hidden_dim": 256, "num_layers": 3, "label": "lr1e5_h256"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(
    label: str,
    lr: float,
    hidden_dim: int,
    num_layers: int,
    embeddings_path: str,
) -> DNNProbeConfig:
    """Create a DNNProbeConfig with shared params + per-run overrides."""
    return DNNProbeConfig(
        study_area=STUDY_AREA,
        year=YEAR,
        h3_resolution=H3_RESOLUTION,
        modality=STAGE2_MODEL,
        embeddings_path=embeddings_path,
        # MLP architecture
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=SHARED_PARAMS["activation"],
        # Training
        learning_rate=lr,
        max_epochs=SHARED_PARAMS["max_epochs"],
        patience=SHARED_PARAMS["patience"],
        initial_batch_size=SHARED_PARAMS["initial_batch_size"],
        weight_decay=SHARED_PARAMS["weight_decay"],
        # Spatial CV
        n_folds=SHARED_PARAMS["n_folds"],
        block_width=SHARED_PARAMS["block_width"],
        block_height=SHARED_PARAMS["block_height"],
        # Provenance
        run_descriptor=f"sweep_{label}",
    )


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _plot_heatmap(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Heatmap of mean R2 for 3-layer runs: x=hidden_dim, y=lr.

    Only includes configs with num_layers == 3.
    """
    subset = summary_df[summary_df["num_layers"] == 3].copy()
    if subset.empty:
        logger.warning("No 3-layer configs for heatmap, skipping")
        return output_dir / "sweep_heatmap.png"

    pivot = subset.pivot_table(
        index="lr", columns="hidden_dim", values="mean_r2", aggfunc="first"
    )
    # Sort: lr descending (high lr at top), hidden_dim ascending
    pivot = pivot.sort_index(ascending=False)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        pivot.values,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )

    # Tick labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{lr:.0e}" for lr in pivot.index])

    ax.set_xlabel("Hidden Dim")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Mean R2 across targets (3-layer MLP)")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val < pivot.values[~np.isnan(pivot.values)].mean() else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Mean R2")
    fig.tight_layout()

    out_path = output_dir / "sweep_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved heatmap to {out_path}")
    return out_path


def _plot_bar_chart(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart of all configs sorted by mean R2."""
    df = summary_df.sort_values("mean_r2", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, max(5, 0.5 * len(df))))
    bars = ax.barh(
        range(len(df)),
        df["mean_r2"],
        color=plt.cm.viridis(np.linspace(0.2, 0.9, len(df))),
        edgecolor="none",
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"].values, fontsize=9)
    ax.set_xlabel("Mean R2 (across all targets)")
    ax.set_title("DNN Probe Sweep: All Configurations")

    # Annotate bars with values
    for i, (bar, val) in enumerate(zip(bars, df["mean_r2"])):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8,
        )

    ax.axvline(x=df["mean_r2"].max(), color="red", linestyle="--",
               linewidth=0.8, alpha=0.7, label="Best")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out_path = output_dir / "sweep_bar_chart.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved bar chart to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def run_sweep(args: argparse.Namespace) -> None:
    """Execute the full hyperparameter sweep."""
    paths = StudyAreaPaths(STUDY_AREA)
    embeddings_path = str(
        paths.fused_embedding_file(STAGE2_MODEL, H3_RESOLUTION, YEAR)
    )

    # Apply CLI overrides to shared params
    if args.max_epochs is not None:
        SHARED_PARAMS["max_epochs"] = args.max_epochs

    logger.info("=" * 60)
    logger.info("DNN PROBE HYPERPARAMETER SWEEP")
    logger.info("=" * 60)
    logger.info(f"Study area:      {STUDY_AREA}")
    logger.info(f"Embeddings:      {embeddings_path}")
    logger.info(f"Stage2 model:    {STAGE2_MODEL}")
    logger.info(f"H3 resolution:   {H3_RESOLUTION}")
    logger.info(f"Configurations:  {len(CONFIGS)}")
    logger.info(f"Shared params:   {SHARED_PARAMS}")

    if args.dry_run:
        logger.info("\n--- DRY RUN: printing configs without training ---")
        for i, cfg in enumerate(CONFIGS, 1):
            config = _build_config(
                label=cfg["label"],
                lr=cfg["lr"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                embeddings_path=embeddings_path,
            )
            logger.info(
                f"  [{i:2d}/{len(CONFIGS)}] {cfg['label']:20s}  "
                f"lr={cfg['lr']:.0e}  hidden={cfg['hidden_dim']:4d}  "
                f"layers={cfg['num_layers']}  "
                f"run_id={config.run_id}  "
                f"output_dir={config.output_dir}"
            )
        logger.info("Dry run complete. No training performed.")
        return

    # Collect results across runs
    all_rows: List[Dict[str, Any]] = []
    sweep_start = time.time()

    for i, cfg in enumerate(CONFIGS, 1):
        label = cfg["label"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[{i}/{len(CONFIGS)}] Running config: {label}")
        logger.info(f"  lr={cfg['lr']}, hidden_dim={cfg['hidden_dim']}, "
                     f"num_layers={cfg['num_layers']}")
        logger.info("=" * 60)

        run_start = time.time()
        try:
            config = _build_config(
                label=label,
                lr=cfg["lr"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                embeddings_path=embeddings_path,
            )

            regressor = DNNProbeRegressor(config)
            results = regressor.run()
            regressor.save_results()

            # Extract R2 per target
            row: Dict[str, Any] = {
                "label": label,
                "lr": cfg["lr"],
                "hidden_dim": cfg["hidden_dim"],
                "num_layers": cfg["num_layers"],
            }
            r2_values = []
            for target_col in TARGET_COLS:
                if target_col in results:
                    r2 = results[target_col].overall_r2
                    row[f"r2_{target_col}"] = r2
                    r2_values.append(r2)
            row["mean_r2"] = float(np.mean(r2_values)) if r2_values else float("nan")

            elapsed = time.time() - run_start
            row["duration_s"] = elapsed
            all_rows.append(row)

            logger.info(
                f"  Completed {label} in {_format_duration(elapsed)}: "
                f"mean_r2={row['mean_r2']:.4f}"
            )

        except RuntimeError as e:
            elapsed = time.time() - run_start
            if "out of memory" in str(e).lower():
                logger.error(
                    f"  OOM for {label} after {_format_duration(elapsed)}. "
                    f"Clearing cache and continuing."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                logger.error(f"  RuntimeError for {label}: {e}")
            all_rows.append({
                "label": label,
                "lr": cfg["lr"],
                "hidden_dim": cfg["hidden_dim"],
                "num_layers": cfg["num_layers"],
                "mean_r2": float("nan"),
                "duration_s": elapsed,
            })

        except Exception as e:
            elapsed = time.time() - run_start
            logger.error(
                f"  Error for {label} after {_format_duration(elapsed)}: {e}"
            )
            all_rows.append({
                "label": label,
                "lr": cfg["lr"],
                "hidden_dim": cfg["hidden_dim"],
                "num_layers": cfg["num_layers"],
                "mean_r2": float("nan"),
                "duration_s": elapsed,
            })

        # Always clean up GPU memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - sweep_start
    logger.info(f"\nSweep completed in {_format_duration(total_elapsed)}")

    # ------------------------------------------------------------------
    # Assemble summary
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(all_rows)

    output_dir = paths.stage3("dnn_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "sweep_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved sweep summary to {csv_path}")

    # ------------------------------------------------------------------
    # Print formatted comparison table
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("SWEEP RESULTS (sorted by mean R2)")
    logger.info("=" * 80)

    sorted_df = summary_df.sort_values("mean_r2", ascending=False)

    # Header
    target_headers = "  ".join(f"{t:>8s}" for t in TARGET_COLS)
    logger.info(
        f"{'Label':20s}  {'lr':>8s}  {'hdim':>5s}  {'L':>2s}  "
        f"{target_headers}  {'mean_r2':>8s}  {'time':>8s}"
    )
    logger.info("-" * 80)

    for _, row in sorted_df.iterrows():
        r2_strs = []
        for t in TARGET_COLS:
            col = f"r2_{t}"
            if col in row and not pd.isna(row[col]):
                r2_strs.append(f"{row[col]:8.4f}")
            else:
                r2_strs.append(f"{'N/A':>8s}")
        r2_line = "  ".join(r2_strs)

        mean_str = f"{row['mean_r2']:.4f}" if not pd.isna(row["mean_r2"]) else "N/A"
        dur_str = _format_duration(row.get("duration_s", 0))

        logger.info(
            f"{row['label']:20s}  {row['lr']:8.0e}  "
            f"{int(row['hidden_dim']):5d}  {int(row['num_layers']):2d}  "
            f"{r2_line}  {mean_str:>8s}  {dur_str:>8s}"
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    if not args.no_viz:
        _plot_heatmap(summary_df, output_dir)
        _plot_bar_chart(summary_df, output_dir)

        # Optionally run full viz for the best config
        valid = summary_df.dropna(subset=["mean_r2"])
        if not valid.empty:
            best_row = valid.loc[valid["mean_r2"].idxmax()]
            logger.info(
                f"\nBest config: {best_row['label']} "
                f"(mean_r2={best_row['mean_r2']:.4f})"
            )
    else:
        logger.info("Visualization skipped (--no-viz)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DNN Probe Hyperparameter Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configs without training",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs for all runs (default: 300)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip generating heatmap and bar chart plots",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_sweep(args)


if __name__ == "__main__":
    main()
