#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe: Multi-Scale UNet Embedding Variants

Tests the causal emergence hypothesis: do macro-scale decoder exits carry
more information about livability than micro-scale (res9) alone?

Embedding variants (all from FullAreaUNet checkpoint):
    1. res9-only (baseline): micro-scale embeddings, ~247K x 128D
    2. multiscale-avg: mean of (res9, res8-upsampled, res7-upsampled), 247K x 128D
    3. multiscale-concat: [res9; res8-up; res7-up], 247K x 384D

Core hypothesis: if avg > res9-only, causal emergence is confirmed --
macro-scale decoder exits add information beyond the micro-scale.

Methodology:
    - DNNProbeRegressor (MLP, hidden=256, patience=20, max_epochs=200)
    - 5-fold spatial block CV
    - All 6 leefbaarometer dimensions

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/probe_multiscale_embeddings.py
    python scripts/stage3/probe_multiscale_embeddings.py --dry-run
    python scripts/stage3/probe_multiscale_embeddings.py --no-viz
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

# Base path for UNet embedding variants
EMBEDDINGS_DIR = Path(
    "data/study_areas/netherlands/stage2_multimodal/unet/embeddings"
)

# Embedding variants to probe (Option A: all at res9 geometry)
EMBEDDING_VARIANTS = [
    {
        "name": "res9-only",
        "label": "res9_only",
        "file": "netherlands_res9_2022.parquet",
        "description": "Micro-scale baseline (128D)",
    },
    {
        "name": "multiscale-avg",
        "label": "multiscale_avg",
        "file": "netherlands_res9_multiscale_avg_2022.parquet",
        "description": "Mean of res9+res8+res7 (128D) -- core hypothesis test",
    },
    {
        "name": "multiscale-concat",
        "label": "multiscale_concat",
        "file": "netherlands_res9_multiscale_concat_2022.parquet",
        "description": "Concat [res9;res8;res7] (384D) -- probe picks scale",
    },
]

# Shared DNN probe hyperparameters (matching fusion comparison script)
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


def _build_config(label: str, embeddings_path: str) -> DNNProbeConfig:
    """Create a DNNProbeConfig for the given embedding variant."""
    return DNNProbeConfig(
        study_area=STUDY_AREA,
        year=YEAR,
        h3_resolution=H3_RESOLUTION,
        modality="unet",
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
        run_descriptor=f"multiscale_{label}",
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _plot_comparison(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: R2 per target for each embedding variant."""
    target_cols_present = [
        c for c in TARGET_COLS if f"r2_{c}" in results_df.columns
    ]
    if not target_cols_present:
        logger.warning("No target columns found for plotting")
        return output_dir / "multiscale_comparison.png"

    n_targets = len(target_cols_present)
    n_models = len(results_df)
    x = np.arange(n_targets)
    width = 0.8 / n_models

    # Colorblind-safe palette
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
                fontsize=8,
                rotation=45,
            )

    ax.set_xlabel("Leefbaarometer Target", fontsize=12)
    ax.set_ylabel("R-squared (5-fold spatial block CV)", fontsize=12)
    ax.set_title(
        "Multi-Scale UNet Embedding Probe: Causal Emergence Test\n"
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
    ax.set_ylim(
        0,
        min(
            1.0,
            results_df[[f"r2_{t}" for t in target_cols_present]].max().max() * 1.25,
        ),
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = output_dir / "multiscale_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison plot to {out_path}")
    return out_path


def _plot_delta_from_baseline(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart showing R2 delta from res9-only baseline per target."""
    target_cols_present = [
        c for c in TARGET_COLS if f"r2_{c}" in results_df.columns
    ]
    if not target_cols_present:
        return output_dir / "multiscale_delta.png"

    # Get baseline (res9-only)
    baseline_row = results_df[results_df["label"] == "res9_only"]
    if baseline_row.empty:
        logger.warning("No res9-only baseline found for delta plot")
        return output_dir / "multiscale_delta.png"

    baseline = baseline_row.iloc[0]
    other_rows = results_df[results_df["label"] != "res9_only"]

    if other_rows.empty:
        return output_dir / "multiscale_delta.png"

    n_targets = len(target_cols_present)
    n_variants = len(other_rows)
    x = np.arange(n_targets)
    width = 0.8 / n_variants

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_variants))

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (_, row) in enumerate(other_rows.iterrows()):
        deltas = [
            row.get(f"r2_{t}", 0) - baseline.get(f"r2_{t}", 0)
            for t in target_cols_present
        ]
        offset = (i - n_variants / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            deltas,
            width,
            label=row["name"],
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, deltas):
            y_pos = bar.get_height()
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos + (0.001 if val >= 0 else -0.001),
                f"{val:+.4f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Leefbaarometer Target", fontsize=12)
    ax.set_ylabel("Delta R2 (vs res9-only baseline)", fontsize=12)
    ax.set_title(
        "Multi-Scale Improvement Over Micro-Scale Baseline\n"
        "(positive = causal emergence, macro adds information)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [TARGET_NAMES.get(t, t) for t in target_cols_present],
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = output_dir / "multiscale_delta.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved delta plot to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_probes(args: argparse.Namespace) -> None:
    """Run DNN probes on all multi-scale embedding variants."""
    paths = StudyAreaPaths(STUDY_AREA)

    logger.info("=" * 70)
    logger.info("MULTI-SCALE UNET EMBEDDING PROBE: CAUSAL EMERGENCE TEST")
    logger.info("=" * 70)
    logger.info(f"Variants:       {len(EMBEDDING_VARIANTS)}")
    logger.info(f"Hidden dim:     {SHARED_PARAMS['hidden_dim']}")
    logger.info(f"Max epochs:     {SHARED_PARAMS['max_epochs']}")
    logger.info(f"Patience:       {SHARED_PARAMS['patience']}")
    logger.info(f"Targets:        {list(TARGET_COLS)}")
    logger.info("")

    # Verify all embedding files exist
    for variant in EMBEDDING_VARIANTS:
        emb_path = EMBEDDINGS_DIR / variant["file"]
        if not emb_path.exists():
            logger.error(f"Missing: {emb_path}")
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")
        emb = pd.read_parquet(emb_path)
        logger.info(
            f"  {variant['name']:20s}  shape={str(emb.shape):15s}  "
            f"{variant['description']}"
        )

    if args.dry_run:
        logger.info("\nDry run complete.")
        return

    all_rows: List[Dict[str, Any]] = []
    total_start = time.time()

    for i, variant in enumerate(EMBEDDING_VARIANTS, 1):
        name = variant["name"]
        label = variant["label"]
        emb_path = str(EMBEDDINGS_DIR / variant["file"])

        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{i}/{len(EMBEDDING_VARIANTS)}] {name}")
        logger.info(f"  {variant['description']}")
        logger.info(f"  Embeddings: {emb_path}")
        logger.info("=" * 70)

        run_start = time.time()
        try:
            config = _build_config(label, emb_path)
            regressor = DNNProbeRegressor(config)
            results = regressor.run()
            regressor.save_results()

            row: Dict[str, Any] = {
                "name": name,
                "label": label,
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
            row["mean_r2"] = (
                float(np.mean(r2_values)) if r2_values else float("nan")
            )

            elapsed = time.time() - run_start
            row["duration_s"] = elapsed
            all_rows.append(row)

            logger.info(
                f"  {name}: mean_r2={row['mean_r2']:.4f} "
                f"in {_format_duration(elapsed)}"
            )

        except Exception as e:
            elapsed = time.time() - run_start
            logger.error(f"  ERROR for {name}: {e}", exc_info=True)
            all_rows.append({
                "name": name,
                "label": label,
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

    # Save results CSV
    run_id = paths.create_run_id("multiscale_comparison")
    output_dir = paths.stage3_run("dnn_probe", run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "multiscale_probe_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 100)
    logger.info("MULTI-SCALE PROBE RESULTS (DNN, hidden_dim=256)")
    logger.info("=" * 100)

    target_headers = "  ".join(f"{t:>8s}" for t in TARGET_COLS)
    logger.info(
        f"{'Variant':22s}  {target_headers}  {'mean_r2':>8s}  {'time':>8s}"
    )
    logger.info("-" * 100)

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

        logger.info(
            f"{row['name']:22s}  {r2_line}  {mean_str:>8s}  {dur_str:>8s}"
        )

    logger.info("=" * 100)

    # ------------------------------------------------------------------
    # Causal emergence verdict
    # ------------------------------------------------------------------
    valid = results_df.dropna(subset=["mean_r2"])
    if not valid.empty:
        res9_row = valid[valid["label"] == "res9_only"]
        avg_row = valid[valid["label"] == "multiscale_avg"]
        concat_row = valid[valid["label"] == "multiscale_concat"]

        if not res9_row.empty and not avg_row.empty:
            res9_r2 = res9_row.iloc[0]["mean_r2"]
            avg_r2 = avg_row.iloc[0]["mean_r2"]
            delta = avg_r2 - res9_r2
            verdict = "CONFIRMED" if delta > 0 else "NOT CONFIRMED"
            logger.info("")
            logger.info(
                f"CAUSAL EMERGENCE: {verdict}  "
                f"(avg={avg_r2:.4f}, res9={res9_r2:.4f}, delta={delta:+.4f})"
            )

            # Per-target deltas
            logger.info("Per-target deltas (avg - res9):")
            for t in TARGET_COLS:
                col = f"r2_{t}"
                if col in avg_row.iloc[0] and col in res9_row.iloc[0]:
                    d = avg_row.iloc[0][col] - res9_row.iloc[0][col]
                    direction = "+" if d >= 0 else ""
                    logger.info(
                        f"  {TARGET_NAMES.get(t, t):25s}: {direction}{d:.4f}"
                    )

        if not res9_row.empty and not concat_row.empty:
            res9_r2 = res9_row.iloc[0]["mean_r2"]
            concat_r2 = concat_row.iloc[0]["mean_r2"]
            delta_c = concat_r2 - res9_r2
            logger.info(
                f"CONCAT vs RES9:  "
                f"(concat={concat_r2:.4f}, res9={res9_r2:.4f}, delta={delta_c:+.4f})"
            )

        best = valid.loc[valid["mean_r2"].idxmax()]
        logger.info(f"\nBest overall: {best['name']} (mean_r2={best['mean_r2']:.4f})")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    if not args.no_viz:
        plot_dir = output_dir
        _plot_comparison(results_df, plot_dir)
        _plot_delta_from_baseline(results_df, plot_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Scale UNet Embedding Probe: Causal Emergence Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify embedding files without training",
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

    run_probes(args)


if __name__ == "__main__":
    main()
