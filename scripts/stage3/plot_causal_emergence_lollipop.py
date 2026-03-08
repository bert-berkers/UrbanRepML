#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scale-Dependent Predictive Power: Diamond profiles across H3 resolutions.

Inspired by causal emergence visualizations (Hoel 2013, Rosas et al. 2020),
these diamond profiles encode two complementary metrics at each spatial scale:

  - Diamond envelope width: R2 from DNN probes, measuring total predictive
    power of the embedding for each liveability dimension at each H3 resolution.
  - Dot size: Signal concentration (Gini coefficient of per-PCA-component R2),
    measuring how few embedding dimensions carry the predictive signal.
    High concentration = clean, low-dimensional mapping (deterministic).
    Low concentration = signal diffused across many dimensions.

When the envelope bulges at a coarser resolution (e.g., res8 for Amenities),
the neighbourhood-scale representation captures more predictive signal than
the fine-grained hexagon scale. When the dot is also large at that scale,
the signal is concentrated -- a hallmark of emergent macro-scale structure.

Lifetime: durable
Stage: 3
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDY_AREA = "netherlands"
YEAR = 2022
RESOLUTIONS = [7, 8, 9]

TARGET_COLS = ["lbm", "vrz", "fys", "soc", "onv", "won"]
TARGET_ORDER = ["lbm", "vrz", "fys", "soc", "onv", "won"]

COLORS = {
    "lbm": "#808080",
    "vrz": "#FF4500",
    "fys": "#32CD32",
    "soc": "#8A2BE2",
    "onv": "#1E90FF",
    "won": "#FFA500",
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
# Signal concentration metric
# ---------------------------------------------------------------------------


def gini(values: np.ndarray) -> float:
    """Gini coefficient: 0 = perfectly equal, 1 = maximally concentrated."""
    sorted_vals = np.sort(np.abs(values))
    n = len(sorted_vals)
    if n == 0 or np.sum(sorted_vals) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n


def compute_signal_concentration(
    embeddings_df: pd.DataFrame,
    target_series: pd.Series,
    n_components: int = 50,
) -> float:
    """
    Compute signal concentration as the Gini coefficient of per-PC R2.

    Steps:
    1. Align embeddings and target on common index, drop NaNs.
    2. Run PCA on the embeddings (capped at n_components).
    3. For each PC, compute univariate R2 = correlation(PC_i, target)^2.
    4. Return Gini coefficient of the R2 distribution.

    High Gini = signal concentrated in few PCs (deterministic mapping).
    Low Gini = signal diffused across many PCs.
    """
    common_idx = embeddings_df.index.intersection(target_series.dropna().index)
    emb_aligned = embeddings_df.loc[common_idx].values
    tgt_aligned = target_series.loc[common_idx].values

    # Drop any rows with NaN in embeddings
    valid_mask = ~np.any(np.isnan(emb_aligned), axis=1) & ~np.isnan(tgt_aligned)
    emb_aligned = emb_aligned[valid_mask]
    tgt_aligned = tgt_aligned[valid_mask]

    if len(emb_aligned) < 10:
        logger.warning("  Too few valid samples for PCA, returning 0.0")
        return 0.0

    # PCA
    n_comp = min(n_components, emb_aligned.shape[1], emb_aligned.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    pcs = pca.fit_transform(emb_aligned)

    # Per-component R2: r2_i = corr(PC_i, target)^2
    r2_per_pc = np.zeros(n_comp)
    for i in range(n_comp):
        corr = np.corrcoef(pcs[:, i], tgt_aligned)[0, 1]
        r2_per_pc[i] = corr ** 2 if not np.isnan(corr) else 0.0

    return gini(r2_per_pc)


# ---------------------------------------------------------------------------
# R2 data loading
# ---------------------------------------------------------------------------


def load_r2_data(paths: StudyAreaPaths) -> Dict[Tuple[str, int], float]:
    """
    Load native resolution probe R2 results.

    Returns dict mapping (target_col, resolution) -> R2 value.
    """
    csv_path = (
        paths.project_root
        / "data"
        / "study_areas"
        / STUDY_AREA
        / "stage3_analysis"
        / "native_resolution_probe_results.csv"
    )
    logger.info(f"Loading R2 data from {csv_path}")
    df = pd.read_csv(csv_path)

    r2_lookup: Dict[Tuple[str, int], float] = {}
    for _, row in df.iterrows():
        res = int(row["resolution"])
        for target_col in TARGET_ORDER:
            if target_col in row.index and pd.notna(row[target_col]):
                r2_lookup[(target_col, res)] = float(row[target_col])

    logger.info(f"  Loaded {len(r2_lookup)} R2 values")
    return r2_lookup


# ---------------------------------------------------------------------------
# Main computation pipeline
# ---------------------------------------------------------------------------


def compute_all_metrics(
    paths: StudyAreaPaths,
) -> pd.DataFrame:
    """
    Compute R2 (from CSV) and signal concentration (Gini from PCA) for all
    (target, resolution) pairs.

    Returns DataFrame with columns: target, resolution, R2, gini, delta_R2.
    """
    # Load R2 from CSV
    r2_lookup = load_r2_data(paths)

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

    # Compute Gini for each (target, resolution)
    for target_col in TARGET_ORDER:
        # Get res9 R2 for delta computation
        r2_res9 = r2_lookup.get((target_col, 9), 0.0)

        for res in RESOLUTIONS:
            emb = data[res]["emb"]
            tgt = data[res]["tgt"]

            if target_col not in tgt.columns:
                logger.warning(f"  Target {target_col} not in res{res} targets, skipping")
                continue

            target_series = tgt[target_col].dropna()
            logger.info(
                f"Computing signal concentration for {target_col} @ res{res} "
                f"({len(target_series):,} targets)"
            )

            gini_val = compute_signal_concentration(emb, target_series)
            r2_val = r2_lookup.get((target_col, res), np.nan)
            delta_r2 = r2_val - r2_res9 if not np.isnan(r2_val) else np.nan

            logger.info(f"  R2={r2_val:.4f}, Gini={gini_val:.4f}, dR2={delta_r2:+.4f}")

            rows.append({
                "target": target_col,
                "resolution": res,
                "R2": r2_val,
                "gini": gini_val,
                "delta_R2": delta_r2,
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
    Create a 2x3 grid of diamond profile subplots.

    Each subplot encodes two metrics:
    - Diamond envelope width at each resolution: proportional to R2.
    - Dot size (centered on vertical midline): proportional to signal
      concentration (Gini coefficient of per-PCA-component R2).
    """
    plt.style.use("default")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor="white")
    axes_flat = axes.flatten()

    y_positions = {7: 2, 8: 1, 9: 0}  # res7 at top, res9 at bottom

    # Normalize R2 to visual half-width: map R2 range to [0.3, 1.0]
    all_r2 = metrics_df["R2"].dropna().values
    r2_global_min = all_r2.min()
    r2_global_max = all_r2.max()
    r2_span = r2_global_max - r2_global_min if r2_global_max > r2_global_min else 1.0

    def r2_to_half_width(r2: float) -> float:
        """Map R2 to half-width of diamond at that level."""
        normalized = (r2 - r2_global_min) / r2_span
        return 0.25 + normalized * 0.75  # 0.25 to 1.0

    # Gini range for dot sizing
    all_gini = metrics_df["gini"].values
    gini_min = all_gini.min()
    gini_max = all_gini.max()
    gini_span = gini_max - gini_min if gini_max > gini_min else 1.0

    for idx, target_col in enumerate(TARGET_ORDER):
        ax = axes_flat[idx]
        ax.set_facecolor("white")
        color = COLORS[target_col]

        subset = metrics_df[metrics_df["target"] == target_col].sort_values("resolution")

        # Get R2 and Gini per resolution
        r2_by_res = {}
        gini_by_res = {}
        for _, row in subset.iterrows():
            res = int(row["resolution"])
            r2_by_res[res] = row["R2"]
            gini_by_res[res] = row["gini"]

        # Build diamond polygon: centered at x=0, width proportional to R2
        if len(r2_by_res) >= 3 and all(not np.isnan(v) for v in r2_by_res.values()):
            hw7 = r2_to_half_width(r2_by_res[7])
            hw8 = r2_to_half_width(r2_by_res[8])
            hw9 = r2_to_half_width(r2_by_res[9])

            y7, y8, y9 = y_positions[7], y_positions[8], y_positions[9]

            # Clockwise from top tip -> right edges -> bottom tip -> left edges
            poly_verts = [
                (0, y7 + 0.15),
                (hw7, y7),
                (hw8, y8),
                (hw9, y9),
                (0, y9 - 0.15),
                (-hw9, y9),
                (-hw8, y8),
                (-hw7, y7),
            ]

            poly = Polygon(
                poly_verts,
                closed=True,
                facecolor="#E8E8E8",
                edgecolor=color,
                linewidth=2.5,
                alpha=0.55,
                zorder=1,
            )
            ax.add_patch(poly)

        # Plot centered dots sized by Gini (signal concentration)
        for res in RESOLUTIONS:
            g = gini_by_res.get(res, 0.0)
            y = y_positions[res]

            # Dot size proportional to Gini
            dot_size = 80 + (g - gini_min) / gini_span * 250

            ax.scatter(
                0, y,
                s=dot_size,
                c=color,
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
                alpha=0.95,
            )

        # Annotations
        for res in RESOLUTIONS:
            g = gini_by_res.get(res, 0.0)
            r2 = r2_by_res.get(res, np.nan)
            y = y_positions[res]

            # R2 label on right edge of envelope
            if not np.isnan(r2):
                hw = r2_to_half_width(r2)
                ax.annotate(
                    f"R\u00b2={r2:.2f}",
                    xy=(hw, y),
                    xytext=(8, 0),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=color,
                    fontweight="bold",
                    va="center",
                    ha="left",
                )

            # Gini label next to dot
            ax.annotate(
                f"G={g:.2f}",
                xy=(0, y),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=7.5,
                color="#555555",
                va="center",
                ha="left",
            )

        # Delta R2 annotation (res8 vs res9)
        r2_9 = r2_by_res.get(9, 0.0)
        r2_8 = r2_by_res.get(8, 0.0)
        delta = r2_8 - r2_9
        delta_sign = "+" if delta >= 0 else ""
        ax.text(
            0.97, 0.03,
            f"\u0394R\u00b2(res8\u2212res9) = {delta_sign}{delta:.3f}",
            transform=ax.transAxes,
            fontsize=10,
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
        ax.set_ylim(-0.5, 2.5)

        # X-axis: hide numeric labels
        ax.set_xlim(-1.6, 1.6)
        ax.set_xticks([])
        ax.set_xlabel("")

        # Bottom row label
        if idx >= 3:
            ax.set_xlabel(
                "width ~ R\u00b2  |  dot ~ signal concentration",
                fontsize=9,
                color="#888888",
            )

        # Vertical center line
        ax.axvline(x=0, color="#D0D0D0", linewidth=0.8, linestyle="-", alpha=0.5, zorder=0)

        # Grid
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # Suptitle
    fig.suptitle(
        "Scale-Dependent Predictive Power: R\u00b2 and Signal Concentration by Resolution",
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

    # Compute all metrics (R2 from CSV, Gini from PCA)
    logger.info("Computing R2 and signal concentration metrics...")
    metrics_df = compute_all_metrics(paths)

    # Save CSV
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figure_dir / "causal_emergence_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")

    # Print summary
    logger.info("\n=== Metrics Summary ===")
    for target_col in TARGET_ORDER:
        subset = metrics_df[metrics_df["target"] == target_col]
        parts = []
        for _, row in subset.iterrows():
            parts.append(
                f"res{int(row['resolution'])}: R2={row['R2']:.3f} "
                f"G={row['gini']:.3f}"
            )
        delta_8 = subset[subset["resolution"] == 8]["delta_R2"].iloc[0]
        logger.info(f"  {target_col}: {' | '.join(parts)} | dR2(8-9)={delta_8:+.3f}")

    # Plot diamond profiles
    png_path, pdf_path = plot_diamond_profiles(metrics_df, figure_dir)
    logger.info(f"\nOutput PNG: {png_path}")
    logger.info(f"Output PDF: {pdf_path}")


if __name__ == "__main__":
    main()
