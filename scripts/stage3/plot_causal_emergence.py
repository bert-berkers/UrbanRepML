#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Causal Emergence Comparison Plots: Radar, Heatmap, and Fusion Progression.

Creates three publication-quality plots that visualize how multi-scale
UNet embeddings unlock causal emergence -- gains in predictive power that
only appear when information flows across H3 resolutions.

Plots:
  1. Radar/Spider chart -- "Scale Fingerprint" showing R2 shape across LBM dims
  2. Heatmap -- "Causal Scale Matrix" of R2 by embedding variant x LBM dim
  3. Slope chart -- "Fusion Progression" showing R2 along the fusion pipeline

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/plot_causal_emergence.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STUDY_AREA = "netherlands"

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "causal-emergence"

COLORS = {
    "lbm": "#808080",  # Dark Grey
    "vrz": "#FF4500",  # Orange Red
    "fys": "#32CD32",  # Lime Green
    "soc": "#8A2BE2",  # Blue Violet
    "onv": "#1E90FF",  # Dodger Blue
    "won": "#FFA500",  # Orange
}
TARGET_ORDER = ["lbm", "vrz", "fys", "soc", "onv", "won"]
TARGET_NAMES = {
    "lbm": "Overall Liveability",
    "vrz": "Amenities",
    "fys": "Physical Environment",
    "soc": "Social Cohesion",
    "onv": "Safety",
    "won": "Housing Stock",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_probe_r2(paths: StudyAreaPaths) -> dict[str, dict[str, float]]:
    """Load R2 values from all probe runs into a nested dict.

    Returns: {variant_label: {target: r2_value}}
    """
    base = paths.root / "stage3_analysis" / "dnn_probe"
    results = {}

    # Mapping of directory suffixes to human-readable labels
    run_map = {
        # Fusion comparison (all res9, different fusion methods)
        "2026-03-07_fusion_compare_concat": "concat",
        "2026-03-07_fusion_compare_ring_agg": "ring_agg",
        "2026-03-07_fusion_compare_gcn": "GCN",
        "2026-03-07_fusion_compare_unet": "UNet-res9",
        # Multiscale UNet variants
        "2026-03-07_multiscale_res9_only": "UNet-res9",
        "2026-03-07_multiscale_multiscale_avg": "UNet-avg",
        "2026-03-07_multiscale_multiscale_concat": "UNet-concat",
        # Native resolution baselines
        "2026-03-07_native_res7": "native-res7",
        "2026-03-07_native_res8": "native-res8",
    }

    for run_dir_name, label in run_map.items():
        csv_path = base / run_dir_name / "metrics_summary.csv"
        if not csv_path.exists():
            logger.warning("Missing probe results: %s", csv_path)
            continue
        df = pd.read_csv(csv_path)
        r2_dict = dict(zip(df["target"], df["overall_r2"]))
        # UNet-res9 appears in both fusion_compare and multiscale runs;
        # the multiscale run is the canonical one (same model, cleaner config).
        if label in results and "multiscale" in run_dir_name:
            results[label] = r2_dict
        elif label not in results:
            results[label] = r2_dict

    return results


# ---------------------------------------------------------------------------
# Plot 1: Radar / Spider Chart -- "Scale Fingerprint"
# ---------------------------------------------------------------------------


def plot_radar(data: dict[str, dict[str, float]]) -> None:
    """Radar chart comparing res9-only, multiscale avg, multiscale concat."""
    variants = ["UNet-res9", "UNet-avg", "UNet-concat"]
    labels = [TARGET_NAMES[t] for t in TARGET_ORDER]

    n = len(TARGET_ORDER)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True),
                           facecolor="white")

    styles = {
        "UNet-res9": dict(linestyle="-", linewidth=1.8, alpha=0.25),
        "UNet-avg": dict(linestyle="--", linewidth=2.0, alpha=0.25),
        "UNet-concat": dict(linestyle="-", linewidth=2.8, alpha=0.30),
    }
    line_colors = {
        "UNet-res9": "#999999",
        "UNet-avg": "#2196F3",
        "UNet-concat": "#E91E63",
    }

    for variant in variants:
        if variant not in data:
            logger.warning("Variant %s not found in data, skipping.", variant)
            continue
        values = [data[variant].get(t, 0.0) for t in TARGET_ORDER]
        values += values[:1]  # close

        color = line_colors[variant]
        style = styles[variant]
        ax.plot(angles, values, color=color, label=variant,
                linestyle=style["linestyle"], linewidth=style["linewidth"])
        ax.fill(angles, values, color=color, alpha=style["alpha"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="medium")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=9, color="#555")
    ax.set_title("Scale Fingerprint: R$^2$ by Liveability Dimension",
                 fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=11,
              frameon=True, facecolor="white", edgecolor="#ccc")

    ax.spines["polar"].set_color("#ccc")
    ax.grid(color="#ddd", linewidth=0.5)

    _save(fig, "scale_fingerprint_radar")


# ---------------------------------------------------------------------------
# Plot 2: Heatmap -- "Causal Scale Matrix"
# ---------------------------------------------------------------------------


def plot_heatmap(data: dict[str, dict[str, float]]) -> None:
    """Heatmap of R2 values: rows = embedding variants, cols = LBM dimensions."""
    # Row order: native resolutions first, then fusion progression
    row_order = [
        "native-res7", "native-res8",
        "UNet-res9", "UNet-avg", "UNet-concat",
    ]
    row_labels = {
        "native-res7": "Native res7",
        "native-res8": "Native res8",
        "UNet-res9": "UNet res9-only",
        "UNet-avg": "UNet multi-scale avg",
        "UNet-concat": "UNet multi-scale concat",
    }
    # Only include rows that exist in the data
    rows = [r for r in row_order if r in data]

    # Sort columns by delta (UNet-concat - UNet-res9), biggest gain first
    if "UNet-concat" in data and "UNet-res9" in data:
        deltas = {
            t: data["UNet-concat"].get(t, 0) - data["UNet-res9"].get(t, 0)
            for t in TARGET_ORDER
        }
        col_order = sorted(TARGET_ORDER, key=lambda t: deltas.get(t, 0),
                           reverse=True)
    else:
        col_order = TARGET_ORDER

    matrix = np.array([
        [data[r].get(t, np.nan) for t in col_order]
        for r in rows
    ])

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.15, vmax=0.90)

    # Annotate cells
    for i in range(len(rows)):
        for j in range(len(col_order)):
            val = matrix[i, j]
            # Use dark text on light cells, light text on dark cells
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="medium", color=text_color)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([TARGET_NAMES[t] for t in col_order], fontsize=11,
                       rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row_labels.get(r, r) for r in rows], fontsize=11)

    ax.set_title("Causal Scale Matrix: R$^2$ by Embedding and Dimension",
                 fontsize=14, fontweight="bold", pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("R$^2$", fontsize=12)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    fig.tight_layout()
    _save(fig, "causal_scale_matrix")


# ---------------------------------------------------------------------------
# Plot 3: Slope Chart -- "Fusion Progression"
# ---------------------------------------------------------------------------


def plot_slope(data: dict[str, dict[str, float]]) -> None:
    """Slope/bump chart: one line per LBM dim across fusion methods."""
    fusion_order = [
        "concat", "ring_agg", "GCN",
        "UNet-res9", "UNet-avg", "UNet-concat",
    ]
    fusion_labels = [
        "Concat", "Ring Agg", "GCN",
        "UNet\nres9-only", "UNet\navg", "UNet\nconcat",
    ]

    # Filter to available methods
    available = [f for f in fusion_order if f in data]
    labels = [fusion_labels[fusion_order.index(f)] for f in available]
    x = np.arange(len(available))

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")

    for target in TARGET_ORDER:
        values = [data[f].get(target, np.nan) for f in available]
        color = COLORS[target]
        ax.plot(x, values, "-o", color=color, linewidth=2.2, markersize=7,
                label=TARGET_NAMES[target], zorder=3)

        # Label endpoint
        last_val = values[-1]
        if not np.isnan(last_val):
            ax.annotate(f"{last_val:.3f}", (x[-1], last_val),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=9, color=color, va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("R$^2$", fontsize=13)
    ax.set_title("Fusion Progression: R$^2$ per Liveability Dimension",
                 fontsize=14, fontweight="bold", pad=15)

    ax.legend(loc="upper left", fontsize=10, frameon=True,
              facecolor="white", edgecolor="#ccc", ncol=2)
    ax.grid(axis="y", color="#eee", linewidth=0.8)
    ax.set_xlim(-0.3, len(available) - 1 + 0.8)
    ax.set_ylim(0.15, 0.95)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, "fusion_progression")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG and PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURE_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    paths = StudyAreaPaths(STUDY_AREA, project_root=PROJECT_ROOT)
    data = load_probe_r2(paths)

    logger.info("Loaded R2 data for variants: %s", list(data.keys()))

    plot_radar(data)
    plot_heatmap(data)
    plot_slope(data)

    logger.info("All plots saved to %s", FIGURE_DIR)


if __name__ == "__main__":
    main()
