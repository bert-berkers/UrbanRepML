#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Causal Emergence Diamond Profiles: EI-based analysis across H3 scales.

Computes Effective Information (EI) from embedding->target Transition
Probability Matrices at each H3 resolution, then derives Causal Emergence
(CE = EI_coarser - EI_finest) and plots 6 diamond/kite profile subplots.

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/plot_causal_emergence_lollipop.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from scipy import stats

from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDY_AREA = "netherlands"
YEAR = 2022
RESOLUTIONS = [7, 8, 9]
TOP_K = 10  # Number of embedding dimensions to use for TPM
N_BINS_DEFAULT = 8  # Quantile bins for discretization

TARGET_COLS = ["lbm", "vrz", "fys", "soc", "onv", "won"]
TARGET_ORDER = ["lbm", "vrz", "fys", "soc", "onv", "won"]

COLORS = {
    "lbm": "#808080",  # Dark Grey
    "vrz": "#FF4500",  # Orange Red
    "fys": "#32CD32",  # Lime Green
    "soc": "#8A2BE2",  # Blue Violet
    "onv": "#1E90FF",  # Dodger Blue
    "won": "#FFA500",  # Orange
}

TARGET_NAMES = {
    "lbm": "Overall Liveability",
    "vrz": "Amenities",
    "fys": "Physical Environment",
    "soc": "Social Cohesion",
    "onv": "Safety",
    "won": "Housing Stock",
}

RESOLUTION_LABELS = {
    7: "res7\nDistrict\n(~1.2 km)",
    8: "res8\nNeighbourhood\n(~450 m)",
    9: "res9\nHexagon\n(~174 m)",
}


# ---------------------------------------------------------------------------
# Effective Information computation
# ---------------------------------------------------------------------------


def select_top_k_dims(
    embeddings: pd.DataFrame,
    target_series: pd.Series,
    k: int = TOP_K,
) -> List[str]:
    """Select the K embedding dimensions most correlated with the target."""
    common_idx = embeddings.index.intersection(target_series.index)
    emb_aligned = embeddings.loc[common_idx]
    tgt_aligned = target_series.loc[common_idx]

    correlations = {}
    for col in emb_aligned.columns:
        corr, _ = stats.spearmanr(emb_aligned[col].values, tgt_aligned.values)
        correlations[col] = abs(corr) if not np.isnan(corr) else 0.0

    sorted_dims = sorted(correlations, key=correlations.get, reverse=True)
    return sorted_dims[:k]


def build_tpm(
    embeddings: pd.DataFrame,
    target_series: pd.Series,
    top_dims: List[str],
    n_bins: int = N_BINS_DEFAULT,
) -> Tuple[np.ndarray, int]:
    """
    Build a Transition Probability Matrix P(target_bin | embedding_state).

    Returns (tpm, n_states) where tpm has shape (n_states, n_target_bins).
    Rows with zero observations are excluded.
    """
    common_idx = embeddings.index.intersection(target_series.index)
    emb_aligned = embeddings.loc[common_idx, top_dims].copy()
    tgt_aligned = target_series.loc[common_idx].copy()

    # Adaptive bin count: reduce if too few unique values
    actual_bins = n_bins
    min_unique = min(emb_aligned[col].nunique() for col in top_dims)
    min_unique = min(min_unique, tgt_aligned.nunique())
    if min_unique < n_bins:
        actual_bins = max(3, min_unique)
        logger.info(f"  Reduced bins from {n_bins} to {actual_bins} (min_unique={min_unique})")

    # Discretize each embedding dimension
    binned_dims = []
    for col in top_dims:
        try:
            binned = pd.qcut(emb_aligned[col], q=actual_bins, labels=False, duplicates="drop")
        except ValueError:
            # Fall back to equal-width bins if quantiles fail
            binned = pd.cut(emb_aligned[col], bins=actual_bins, labels=False)
        binned = binned.fillna(0).astype(int)
        binned_dims.append(binned)

    # Create composite embedding state as tuple of bin indices
    state_df = pd.DataFrame(dict(enumerate(binned_dims)), index=common_idx)
    state_labels = state_df.apply(tuple, axis=1)

    # Discretize target
    try:
        target_binned = pd.qcut(tgt_aligned, q=actual_bins, labels=False, duplicates="drop")
    except ValueError:
        target_binned = pd.cut(tgt_aligned, bins=actual_bins, labels=False)
    target_binned = target_binned.fillna(0).astype(int)
    n_target_bins = int(target_binned.max()) + 1

    # Build contingency: count(state, target_bin)
    combined = pd.DataFrame({"state": state_labels, "target_bin": target_binned})
    contingency = combined.groupby(["state", "target_bin"]).size().unstack(fill_value=0)

    # Ensure all target bins are represented
    for b in range(n_target_bins):
        if b not in contingency.columns:
            contingency[b] = 0
    contingency = contingency[sorted(contingency.columns)]

    # Normalize rows to get P(target_bin | state)
    row_sums = contingency.sum(axis=1)
    # Drop states with 0 observations
    contingency = contingency[row_sums > 0]
    row_sums = contingency.sum(axis=1)
    tpm = contingency.div(row_sums, axis=0).values

    n_states = tpm.shape[0]
    return tpm, n_states


def compute_effective_information(tpm: np.ndarray) -> float:
    """
    Compute Effective Information from a TPM.

    EI = log2(N) - (1/N) * sum_i H(row_i)

    where N = number of source states, H(row_i) = Shannon entropy of row i.
    """
    n_states = tpm.shape[0]
    if n_states <= 1:
        return 0.0

    max_entropy = np.log2(n_states)

    # Average row entropy
    row_entropies = []
    for row in tpm:
        # Remove zeros to avoid log(0)
        row_nonzero = row[row > 0]
        if len(row_nonzero) == 0:
            row_entropies.append(0.0)
        else:
            h = -np.sum(row_nonzero * np.log2(row_nonzero))
            row_entropies.append(h)

    avg_entropy = np.mean(row_entropies)
    ei = max_entropy - avg_entropy

    return ei


# ---------------------------------------------------------------------------
# Main computation pipeline
# ---------------------------------------------------------------------------


def compute_all_metrics(paths: StudyAreaPaths) -> pd.DataFrame:
    """
    Compute EI and CE for all (target, resolution) pairs.

    Returns DataFrame with columns: target, resolution, EI, CE.
    """
    rows = []

    # Load all embeddings and targets
    data = {}
    for res in RESOLUTIONS:
        emb_path = paths.fused_embedding_file("unet", res, YEAR)
        tgt_path = paths.target_file("leefbaarometer", res, YEAR)

        logger.info(f"Loading res{res} embeddings from {emb_path}")
        emb = pd.read_parquet(emb_path)
        logger.info(f"  Embeddings: {emb.shape}")

        logger.info(f"Loading res{res} targets from {tgt_path}")
        tgt = pd.read_parquet(tgt_path)
        logger.info(f"  Targets: {tgt.shape}")

        data[res] = {"emb": emb, "tgt": tgt}

    # Compute EI for each (target, resolution)
    ei_values = {}  # (target, res) -> EI
    for target_col in TARGET_ORDER:
        for res in RESOLUTIONS:
            emb = data[res]["emb"]
            tgt = data[res]["tgt"]

            if target_col not in tgt.columns:
                logger.warning(f"  Target {target_col} not in res{res} targets, skipping")
                continue

            target_series = tgt[target_col].dropna()
            logger.info(f"Computing EI for {target_col} @ res{res} ({len(target_series):,} targets)")

            # Select top-K dimensions
            top_dims = select_top_k_dims(emb, target_series, k=TOP_K)
            logger.info(f"  Top-{TOP_K} dims: {top_dims[:3]}...")

            # Build TPM
            tpm, n_states = build_tpm(emb, target_series, top_dims)
            logger.info(f"  TPM: {tpm.shape} ({n_states} states)")

            # Compute EI
            ei = compute_effective_information(tpm)
            logger.info(f"  EI = {ei:.4f}")

            ei_values[(target_col, res)] = ei

    # Compute CE relative to res9 (finest resolution)
    for target_col in TARGET_ORDER:
        ei_res9 = ei_values.get((target_col, 9), 0.0)
        for res in RESOLUTIONS:
            ei = ei_values.get((target_col, res), 0.0)
            ce = ei - ei_res9  # CE > 0 means causal emergence at coarser scale
            rows.append({
                "target": target_col,
                "resolution": res,
                "EI": ei,
                "CE": ce,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diamond Profile Plot
# ---------------------------------------------------------------------------


def plot_diamond_profiles(
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Create a 2x3 grid of diamond/kite profile subplots.

    Each subplot shows EI at 3 resolutions with a grey filled envelope
    and colored border for the target dimension.
    """
    plt.style.use("default")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    axes_flat = axes.flatten()

    y_positions = {7: 2, 8: 1, 9: 0}  # res7 at top, res9 at bottom

    # Compute global EI range for consistent x-axes
    all_ei = metrics_df["EI"].values
    ei_min = min(all_ei) * 0.9
    ei_max = max(all_ei) * 1.1
    # Add some padding
    ei_range = ei_max - ei_min
    x_lo = ei_min - 0.05 * ei_range
    x_hi = ei_max + 0.05 * ei_range

    for idx, target_col in enumerate(TARGET_ORDER):
        ax = axes_flat[idx]
        ax.set_facecolor("white")
        color = COLORS[target_col]

        subset = metrics_df[metrics_df["target"] == target_col].sort_values("resolution")

        # Get EI values per resolution
        ei_by_res = {}
        for _, row in subset.iterrows():
            ei_by_res[int(row["resolution"])] = row["EI"]

        # Build diamond envelope polygon
        # Go down the left side (min of EI at each level creates kite shape)
        # Since we have one value per level, the diamond is formed by the shape
        # of the EI values themselves
        points_x = []
        points_y = []

        # Collect points in order: top -> middle -> bottom -> back up
        # The diamond shape comes from plotting EI at each resolution
        for res in [7, 8, 9]:
            if res in ei_by_res:
                points_x.append(ei_by_res[res])
                points_y.append(y_positions[res])

        if len(points_x) >= 3:
            # Create a filled polygon by making a symmetric kite shape
            # Left boundary: slightly left of each point
            # Right boundary: slightly right of each point
            # The natural shape emerges from the EI values
            center_x = np.mean(points_x)

            # Build polygon vertices going clockwise from top
            poly_x = []
            poly_y = []
            for i, (px, py) in enumerate(zip(points_x, points_y)):
                poly_x.append(px)
                poly_y.append(py)

            # Close the polygon by going back (it's already a line, make it a thin diamond)
            # Add slight width to create visible shape
            width_factor = 0.08 * (x_hi - x_lo)

            # Left side going down
            left_x = [px - width_factor * 0.3 for px in points_x]
            right_x = [px + width_factor * 0.3 for px in points_x]
            left_y = [y_positions[r] for r in [7, 8, 9]]
            right_y = list(reversed(left_y))

            # Full polygon: down the left, up the right
            envelope_x = left_x + list(reversed(right_x))
            envelope_y = left_y + right_y

            poly = Polygon(
                list(zip(envelope_x, envelope_y)),
                closed=True,
                facecolor="#E8E8E8",
                edgecolor=color,
                linewidth=2.0,
                alpha=0.6,
                zorder=1,
            )
            ax.add_patch(poly)

        # Plot dots at each resolution
        for res in RESOLUTIONS:
            if res in ei_by_res:
                ei = ei_by_res[res]
                y = y_positions[res]

                # Dot size proportional to EI
                dot_size = 60 + ei * 80
                ax.scatter(
                    ei, y,
                    s=dot_size,
                    c=color,
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=3,
                    alpha=0.95,
                )

                # Connecting line between dots
                ax.plot(
                    [ei, ei], [y - 0.02, y + 0.02],
                    color=color, linewidth=0, zorder=2,
                )

        # Connect dots with colored line
        res_order = [7, 8, 9]
        line_x = [ei_by_res.get(r, 0) for r in res_order]
        line_y = [y_positions[r] for r in res_order]
        ax.plot(line_x, line_y, color=color, linewidth=1.5, alpha=0.7, zorder=2)

        # EI value labels
        for res in RESOLUTIONS:
            if res in ei_by_res:
                ei = ei_by_res[res]
                y = y_positions[res]
                ax.annotate(
                    f"{ei:.2f}",
                    xy=(ei, y),
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    va="center",
                    ha="left",
                )

        # CE annotation (coarser vs finest)
        ei9 = ei_by_res.get(9, 0)
        ei7 = ei_by_res.get(7, 0)
        ce_7 = ei7 - ei9
        ce_sign = "+" if ce_7 >= 0 else ""
        ax.text(
            0.97, 0.03,
            f"CE(7-9) = {ce_sign}{ce_7:.2f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right", va="bottom",
            color="#555555",
            fontstyle="italic",
        )

        # Subplot title
        ax.set_title(
            TARGET_NAMES[target_col],
            fontsize=12,
            fontweight="bold",
            color=color,
            pad=8,
        )

        # Y-axis: resolution labels
        ax.set_yticks([y_positions[r] for r in RESOLUTIONS])
        ax.set_yticklabels(
            [RESOLUTION_LABELS[r] for r in RESOLUTIONS],
            fontsize=9,
        )
        ax.set_ylim(-0.6, 2.6)

        # X-axis
        ax.set_xlim(x_lo, x_hi)
        if idx >= 3:  # Bottom row only
            ax.set_xlabel("Effective Information (bits)", fontsize=10)

        # Grid
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Suptitle
    fig.suptitle(
        "Causal Emergence Diamond Profiles: Effective Information by Scale",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "causal_emergence_diamonds.png"
    pdf_path = output_dir / "causal_emergence_diamonds.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved diamond plot to {png_path}")
    logger.info(f"Saved diamond PDF to {pdf_path}")

    return png_path, pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    paths = StudyAreaPaths(STUDY_AREA)
    figure_dir = paths.project_root / "reports" / "figures" / "causal-emergence"

    # Compute EI and CE metrics
    logger.info("Computing Effective Information and Causal Emergence metrics...")
    metrics_df = compute_all_metrics(paths)

    # Save CSV
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / "causal_emergence_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")

    # Print summary
    logger.info("\n=== Effective Information Summary ===")
    for target_col in TARGET_ORDER:
        subset = metrics_df[metrics_df["target"] == target_col]
        ei_strs = [
            f"res{int(row['resolution'])}={row['EI']:.3f}"
            for _, row in subset.iterrows()
        ]
        ce_7 = subset[subset["resolution"] == 7]["CE"].iloc[0]
        logger.info(f"  {target_col}: {', '.join(ei_strs)} | CE(7-9)={ce_7:+.3f}")

    # Plot diamond profiles
    png_path, pdf_path = plot_diamond_profiles(metrics_df, figure_dir)
    logger.info(f"\nOutput PNG: {png_path}")
    logger.info(f"Output PDF: {pdf_path}")


if __name__ == "__main__":
    main()
