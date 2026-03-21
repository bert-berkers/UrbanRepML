#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parametric DNN Probe Comparison Runner

Consolidates all DNN probe comparison scripts into a single configurable entry
point. Runs identical DNN probes (MLP, hidden_dim=256, patience=20, max_epochs=200,
5-fold spatial block CV with 10km blocks) on user-specified embedding sources,
producing bar-chart comparisons and CSV result tables.

Built-in named configs reproduce the experiments from the original scripts:
    - fusion_2022:    concat, ring_agg, gcn, unet (YEAR=2022)
    - concat_vs_unet: concat 208D vs UNet 128D (YEAR=20mix)
    - fair_pca_64d:   concat PCA-64D vs UNet native 64D (YEAR=20mix)
    - pca_192d:       concat PCA-192D vs UNet multiscale 192D (YEAR=20mix)
    - sageconv:       SAGEConv UNet 64D + 192D vs prior baselines (YEAR=20mix)
    - ring_agg:       concat PCA-64D vs ring-aggregated PCA-64D (YEAR=20mix)

Custom comparisons via --embeddings:
    python scripts/stage3/run_probe_comparison.py \\
        --embeddings "My Model:path/to/embeddings.parquet" \\
                     "Baseline:path/to/baseline.parquet"

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/run_probe_comparison.py --config fusion_2022
    python scripts/stage3/run_probe_comparison.py --config ring_agg
    python scripts/stage3/run_probe_comparison.py --config fusion_2022 --dry-run
    python scripts/stage3/run_probe_comparison.py \\
        --embeddings "Concat:data/.../concat.parquet" "UNet:data/.../unet.parquet"
    python scripts/stage3/run_probe_comparison.py --list-configs
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from stage3_analysis.dnn_probe import DNNProbeConfig, DNNProbeRegressor
from stage3_analysis.linear_probe import TARGET_COLS, TARGET_NAMES
from stage3_analysis.probe_results_writer import ProbeResultsWriter
from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared DNN probe hyperparameters (identical across all comparisons)
# ---------------------------------------------------------------------------

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

STUDY_AREA = "netherlands"
H3_RESOLUTION = 9


# ---------------------------------------------------------------------------
# Embedding source specification
# ---------------------------------------------------------------------------


class EmbeddingSource:
    """Describes one embedding to probe."""

    def __init__(
        self,
        name: str,
        label: str,
        path: str,
        modality: str = "unknown",
    ):
        self.name = name
        self.label = label
        self.path = path
        self.modality = modality

    def __repr__(self) -> str:
        return f"EmbeddingSource({self.name!r}, path={self.path!r})"


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def pca_reduce(
    df: pd.DataFrame,
    n_components: int,
    label: str,
    output_dir: Path,
) -> Path:
    """Standardize + PCA reduce a DataFrame, save parquet, return path."""
    logger.info("PCA: %dD -> %dD (%s)", df.shape[1], n_components, label)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_.sum()
    logger.info(
        "  Variance explained: %.4f (%.1f%%)", var_explained, var_explained * 100
    )

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

    pca_path = output_dir / f"{label}_pca{n_components}.parquet"
    pca_df.to_parquet(pca_path)
    logger.info("  Saved PCA embeddings to %s", pca_path)

    return pca_path


def run_ring_aggregation(
    paths: StudyAreaPaths,
    concat_df: pd.DataFrame,
    k_rings: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Apply k-ring exponential aggregation to embeddings, return result."""
    from stage2_fusion.models.ring_aggregation import SimpleRingAggregator

    # Load neighbourhood
    nb_path = paths.neighbourhood_dir() / (
        f"{paths.study_area}_res{H3_RESOLUTION}_neighbourhood.pkl"
    )
    logger.info("Loading neighbourhood from %s", nb_path)
    with open(nb_path, "rb") as f:
        neighbourhood = pickle.load(f)

    logger.info(
        "Running ring aggregation with K=%d, weighting=exponential", k_rings
    )
    t0 = time.time()
    aggregator = SimpleRingAggregator(
        neighbourhood=neighbourhood,
        K=k_rings,
        weighting="exponential",
    )
    ring_df = aggregator.aggregate(concat_df)
    elapsed = time.time() - t0
    logger.info("  Ring aggregation completed in %.1f seconds", elapsed)
    logger.info("  Output shape: %s", ring_df.shape)

    ring_path = output_dir / f"ring_agg_k{k_rings}.parquet"
    ring_df.to_parquet(ring_path)
    logger.info("  Saved ring-aggregated embeddings to %s", ring_path)

    return ring_df


# ---------------------------------------------------------------------------
# Named config builders
# ---------------------------------------------------------------------------


def _concat_path(paths: StudyAreaPaths, year: str) -> Path:
    """Resolve path to concat embeddings parquet."""
    return (
        paths.model_embeddings("concat")
        / f"{paths.study_area}_res{H3_RESOLUTION}_{year}.parquet"
    )


def _unet_path(paths: StudyAreaPaths, year: str) -> Path:
    """Resolve path to UNet embeddings parquet."""
    return (
        paths.model_embeddings("unet")
        / f"{paths.study_area}_res{H3_RESOLUTION}_{year}.parquet"
    )


def _unet_multiscale_path(paths: StudyAreaPaths, year: str) -> Path:
    """Resolve path to UNet multiscale concat parquet."""
    return (
        paths.model_embeddings("unet")
        / f"{paths.study_area}_res{H3_RESOLUTION}_multiscale_concat_{year}.parquet"
    )


def build_fusion_2022(paths: StudyAreaPaths, output_dir: Path) -> List[EmbeddingSource]:
    """Compare concat, ring_agg, gcn, unet (YEAR=2022).

    Reproduces: dnn_probe_fusion_comparison.py
    """
    year = 2022
    return [
        EmbeddingSource(
            name="Concat (PCA-64)",
            label="concat",
            path=str(paths.fused_embedding_file("concat", H3_RESOLUTION, year)),
            modality="concat",
        ),
        EmbeddingSource(
            name="Ring Aggregation",
            label="ring_agg",
            path=str(paths.fused_embedding_file("ring_agg", H3_RESOLUTION, year)),
            modality="ring_agg",
        ),
        EmbeddingSource(
            name="Lattice GCN",
            label="gcn",
            path=str(paths.fused_embedding_file("gcn", H3_RESOLUTION, year)),
            modality="gcn",
        ),
        EmbeddingSource(
            name="FullAreaUNet",
            label="unet",
            path=str(paths.fused_embedding_file("unet", H3_RESOLUTION, year)),
            modality="unet",
        ),
    ]


def build_concat_vs_unet(
    paths: StudyAreaPaths, output_dir: Path
) -> List[EmbeddingSource]:
    """Compare concat 208D vs UNet 128D (YEAR=20mix).

    Reproduces: probe_20mix_comparison.py (regression part only).
    """
    year = "20mix"
    return [
        EmbeddingSource(
            name="Concat 208D",
            label="concat_208d",
            path=str(_concat_path(paths, year)),
            modality="concat",
        ),
        EmbeddingSource(
            name="UNet 128D",
            label="unet_128d",
            path=str(_unet_path(paths, year)),
            modality="unet",
        ),
    ]


def build_fair_pca_64d(
    paths: StudyAreaPaths, output_dir: Path
) -> List[EmbeddingSource]:
    """Compare concat PCA-64D vs UNet native 64D (YEAR=20mix).

    Reproduces: probe_fair_pca_comparison.py
    Requires PCA preprocessing on concat embeddings.
    """
    year = "20mix"
    concat_p = _concat_path(paths, year)
    unet_p = _unet_path(paths, year)

    # Determine UNet native dimensionality
    unet_df = pd.read_parquet(unet_p)
    unet_dim = unet_df.shape[1]
    del unet_df
    logger.info("UNet native dimensionality: %dD", unet_dim)

    # PCA concat to match UNet dim
    concat_df = pd.read_parquet(concat_p)
    pca_path = pca_reduce(concat_df, unet_dim, "concat", output_dir)
    del concat_df

    return [
        EmbeddingSource(
            name=f"Concat PCA-{unet_dim}D",
            label=f"concat_pca{unet_dim}d",
            path=str(pca_path),
            modality="concat",
        ),
        EmbeddingSource(
            name=f"UNet {unet_dim}D",
            label=f"unet_{unet_dim}d",
            path=str(unet_p),
            modality="unet",
        ),
    ]


def build_pca_192d(
    paths: StudyAreaPaths, output_dir: Path
) -> List[EmbeddingSource]:
    """Compare concat PCA-192D vs UNet multiscale 192D (YEAR=20mix).

    Reproduces: probe_192d_comparison.py
    """
    year = "20mix"
    pca_dim = 192

    # PCA concat to 192D
    concat_df = pd.read_parquet(_concat_path(paths, year))
    pca_path = pca_reduce(concat_df, pca_dim, "concat", output_dir)
    del concat_df

    # UNet multiscale concat
    multiscale_p = _unet_multiscale_path(paths, year)
    if not multiscale_p.exists():
        raise FileNotFoundError(
            f"UNet multiscale concat not found: {multiscale_p}\n"
            "Run: python scripts/stage2/extract_highway_exits.py "
            "--study-area netherlands --year 20mix"
        )

    return [
        EmbeddingSource(
            name=f"Concat-PCA-{pca_dim}D",
            label=f"concat_pca{pca_dim}d",
            path=str(pca_path),
            modality="concat",
        ),
        EmbeddingSource(
            name=f"UNet-{pca_dim}D",
            label=f"unet_{pca_dim}d",
            path=str(multiscale_p),
            modality="unet",
        ),
    ]


def build_sageconv(
    paths: StudyAreaPaths, output_dir: Path
) -> List[EmbeddingSource]:
    """Compare SAGEConv UNet 64D + 192D (YEAR=20mix).

    Reproduces: probe_sageconv_comparison.py (new probes only; hardcoded
    baselines are printed as reference in the comparison table).
    """
    year = "20mix"
    unet_64d_p = _unet_path(paths, year)
    unet_192d_p = _unet_multiscale_path(paths, year)

    for p, desc in [
        (unet_64d_p, "UNet 64D"),
        (unet_192d_p, "UNet 192D multiscale"),
    ]:
        if not p.exists():
            raise FileNotFoundError(
                f"{desc} not found: {p}\n"
                "Run: python scripts/stage2/extract_highway_exits.py "
                "--study-area netherlands --year 20mix"
            )

    return [
        EmbeddingSource(
            name="SAGEConv-UNet-64D",
            label="sageconv_unet_64d",
            path=str(unet_64d_p),
            modality="unet",
        ),
        EmbeddingSource(
            name="SAGEConv-UNet-192D",
            label="sageconv_unet_192d",
            path=str(unet_192d_p),
            modality="unet",
        ),
    ]


def build_ring_agg(
    paths: StudyAreaPaths, output_dir: Path
) -> List[EmbeddingSource]:
    """Compare concat PCA-64D vs ring-aggregated PCA-64D (YEAR=20mix).

    Reproduces: probe_ring_agg_comparison.py
    Requires ring aggregation + PCA preprocessing.
    """
    year = "20mix"
    k_rings = 10
    pca_dim = 64

    concat_df = pd.read_parquet(_concat_path(paths, year))

    # Ring aggregate
    ring_df = run_ring_aggregation(paths, concat_df, k_rings, output_dir)

    # PCA both to 64D
    ring_pca_path = pca_reduce(ring_df, pca_dim, "ring_agg", output_dir)
    del ring_df

    concat_pca_path = pca_reduce(concat_df, pca_dim, "concat", output_dir)
    del concat_df

    return [
        EmbeddingSource(
            name=f"Concat-PCA-{pca_dim}D",
            label=f"concat_pca{pca_dim}d",
            path=str(concat_pca_path),
            modality="concat",
        ),
        EmbeddingSource(
            name=f"RingAgg-k{k_rings}-PCA-{pca_dim}D",
            label=f"ring_agg_k{k_rings}_pca{pca_dim}d",
            path=str(ring_pca_path),
            modality="ring_agg",
        ),
    ]


# Registry of named configs
NAMED_CONFIGS = {
    "fusion_2022": {
        "builder": build_fusion_2022,
        "target_year": 2022,
        "description": "Concat, Ring Agg, GCN, UNet (year=2022)",
        "original_script": "dnn_probe_fusion_comparison.py",
    },
    "concat_vs_unet": {
        "builder": build_concat_vs_unet,
        "target_year": 2022,
        "description": "Concat 208D vs UNet 128D (year=20mix)",
        "original_script": "probe_20mix_comparison.py",
    },
    "fair_pca_64d": {
        "builder": build_fair_pca_64d,
        "target_year": 2022,
        "description": "Concat PCA-64D vs UNet native 64D (year=20mix)",
        "original_script": "probe_fair_pca_comparison.py",
    },
    "pca_192d": {
        "builder": build_pca_192d,
        "target_year": 2022,
        "description": "Concat PCA-192D vs UNet multiscale 192D (year=20mix)",
        "original_script": "probe_192d_comparison.py",
    },
    "sageconv": {
        "builder": build_sageconv,
        "target_year": 2022,
        "description": "SAGEConv UNet 64D + 192D vs baselines (year=20mix)",
        "original_script": "probe_sageconv_comparison.py",
    },
    "ring_agg": {
        "builder": build_ring_agg,
        "target_year": 2022,
        "description": "Concat PCA-64D vs Ring-Aggregated PCA-64D (year=20mix)",
        "original_script": "probe_ring_agg_comparison.py",
    },
}


# ---------------------------------------------------------------------------
# Core probe runner
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds as human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _build_probe_config(
    source: EmbeddingSource,
    target_year: int,
    run_prefix: str,
) -> DNNProbeConfig:
    """Create a DNNProbeConfig for one embedding source."""
    return DNNProbeConfig(
        study_area=STUDY_AREA,
        year=target_year,
        h3_resolution=H3_RESOLUTION,
        modality=source.modality,
        embeddings_path=source.path,
        target_path=str(
            StudyAreaPaths(STUDY_AREA).target_file(
                "leefbaarometer", H3_RESOLUTION, target_year
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
        run_descriptor=f"{run_prefix}_{source.label}",
    )


def run_probes(
    sources: List[EmbeddingSource],
    target_year: int,
    run_prefix: str,
    dry_run: bool = False,
    write_standardized: bool = False,
) -> pd.DataFrame:
    """Run DNN regression probes on all given embedding sources.

    Returns a DataFrame with one row per source, columns for per-target R2,
    R2 std, mean R2, and duration.
    """
    logger.info("=" * 70)
    logger.info("DNN PROBE COMPARISON: %d sources", len(sources))
    logger.info("=" * 70)
    logger.info("Hidden dim:  %d", SHARED_PARAMS["hidden_dim"])
    logger.info("Max epochs:  %d", SHARED_PARAMS["max_epochs"])
    logger.info("Patience:    %d", SHARED_PARAMS["patience"])
    logger.info("Targets:     %s", list(TARGET_COLS))

    if dry_run:
        for src in sources:
            emb = pd.read_parquet(src.path)
            logger.info(
                "  %-25s  shape=%-15s  path=%s",
                src.name,
                str(emb.shape),
                src.path,
            )
        logger.info("Dry run complete.")
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    total_start = time.time()

    for i, src in enumerate(sources, 1):
        logger.info("\n%s", "=" * 70)
        logger.info("[%d/%d] %s", i, len(sources), src.name)
        logger.info("  Embeddings: %s", src.path)
        logger.info("=" * 70)

        run_start = time.time()
        try:
            config = _build_probe_config(src, target_year, run_prefix)
            regressor = DNNProbeRegressor(config)
            results = regressor.run()
            regressor.save_results()

            # Write standardized probe results for cross-approach comparison
            if write_standardized:
                approach_slug = src.label
                out = ProbeResultsWriter.write_from_regressor(
                    regressor, approach=approach_slug, study_area=STUDY_AREA
                )
                logger.info("  Standardized results -> %s", out)

            row: Dict[str, Any] = {"name": src.name, "model": src.modality}
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
                "  %s: mean_r2=%.4f in %s",
                src.name,
                row["mean_r2"],
                _format_duration(elapsed),
            )

        except Exception as e:
            elapsed = time.time() - run_start
            logger.error("  ERROR for %s: %s", src.name, e, exc_info=True)
            all_rows.append(
                {
                    "name": src.name,
                    "model": src.modality,
                    "mean_r2": float("nan"),
                    "duration_s": elapsed,
                }
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start
    logger.info("\nTotal probe time: %s", _format_duration(total_elapsed))

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Output: comparison table + bar chart
# ---------------------------------------------------------------------------


def print_comparison_table(
    results_df: pd.DataFrame,
    reference_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Print formatted comparison table, optionally including reference baselines."""
    conditions: List[Dict[str, Any]] = []

    # Add reference baselines first (if any)
    if reference_results:
        for name, vals in reference_results.items():
            row: Dict[str, Any] = {"name": name}
            for t in TARGET_COLS:
                row[f"r2_{t}"] = vals.get(t, float("nan"))
            row["mean_r2"] = vals.get("mean", float("nan"))
            conditions.append(row)

    # Add new results
    for _, row in results_df.iterrows():
        conditions.append(dict(row))

    # Header
    logger.info("\n%s", "=" * 100)
    logger.info("COMPARISON RESULTS (DNN Probe)")
    logger.info("=" * 100)

    target_headers = "  ".join(f"{t:>8s}" for t in TARGET_COLS)
    logger.info("%-25s  %s  %8s", "Source", target_headers, "mean_r2")
    logger.info("-" * 100)

    # Find per-target winners
    per_target_best: Dict[str, str] = {}
    for t in TARGET_COLS:
        best_val = -999.0
        best_name = ""
        for c in conditions:
            val = c.get(f"r2_{t}", float("nan"))
            if not pd.isna(val) and val > best_val:
                best_val = val
                best_name = c["name"]
        per_target_best[t] = best_name

    for c in conditions:
        r2_strs = []
        for t in TARGET_COLS:
            val = c.get(f"r2_{t}", float("nan"))
            if not pd.isna(val):
                marker = " *" if per_target_best[t] == c["name"] else "  "
                r2_strs.append(f"{val:6.4f}{marker}")
            else:
                r2_strs.append(f"{'N/A':>8s}")
        r2_line = "  ".join(r2_strs)
        mean_str = (
            f"{c['mean_r2']:.4f}"
            if not pd.isna(c.get("mean_r2", float("nan")))
            else "N/A"
        )
        logger.info("%-25s  %s  %8s", c["name"], r2_line, mean_str)

    # Overall winner
    logger.info("-" * 100)
    valid = [c for c in conditions if not pd.isna(c.get("mean_r2", float("nan")))]
    if valid:
        best = max(valid, key=lambda c: c["mean_r2"])
        logger.info(
            "Best overall: %s (mean R2 = %.4f)", best["name"], best["mean_r2"]
        )
    logger.info("* = best for that target")
    logger.info("=" * 100)


def plot_comparison(
    results_df: pd.DataFrame,
    output_dir: Path,
    title: str = "DNN Probe Comparison",
    filename: str = "probe_comparison.png",
    reference_results: Optional[Dict[str, Dict[str, float]]] = None,
) -> Path:
    """Grouped bar chart: R2 per target for each embedding source."""
    # Build combined DataFrame
    all_rows = []
    if reference_results:
        for name, vals in reference_results.items():
            row: Dict[str, Any] = {"name": name}
            for t in TARGET_COLS:
                row[f"r2_{t}"] = vals.get(t, float("nan"))
            row["mean_r2"] = vals.get("mean", float("nan"))
            all_rows.append(row)

    for _, row in results_df.iterrows():
        all_rows.append(dict(row))

    all_df = pd.DataFrame(all_rows)

    target_cols_present = [
        c for c in TARGET_COLS if f"r2_{c}" in all_df.columns
    ]
    if not target_cols_present:
        logger.warning("No target columns found for plotting")
        return output_dir / filename

    n_targets = len(target_cols_present)
    n_models = len(all_df)
    x = np.arange(n_targets)
    width = 0.8 / n_models

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_models))

    fig, ax = plt.subplots(figsize=(max(12, 2 * n_models + 6), 7))

    for i, (_, row) in enumerate(all_df.iterrows()):
        vals = [row.get(f"r2_{t}", 0) for t in target_cols_present]
        stds = [row.get(f"r2_std_{t}", 0) for t in target_cols_present]
        offset = (i - n_models / 2 + 0.5) * width

        bar_kwargs: Dict[str, Any] = {
            "label": row["name"],
            "color": colors[i],
            "edgecolor": "white",
            "linewidth": 0.5,
        }
        # Include error bars only if std data exists
        if any(s > 0 for s in stds):
            bar_kwargs["yerr"] = stds
            bar_kwargs["capsize"] = 3

        bars = ax.bar(x + offset, vals, width, **bar_kwargs)

        fontsize = max(6, 9 - n_models)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    rotation=45,
                )

    ax.set_xlabel("Leefbaarometer Target", fontsize=12)
    ax.set_ylabel("R-squared (5-fold spatial block CV)", fontsize=12)
    ax.set_title(
        f"{title}\n"
        f"(MLP h={SHARED_PARAMS['hidden_dim']}, "
        f"{SHARED_PARAMS['num_layers']} layers, "
        f"patience={SHARED_PARAMS['patience']}, "
        f"max_epochs={SHARED_PARAMS['max_epochs']})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [TARGET_NAMES.get(t, t) for t in target_cols_present], fontsize=10
    )
    ncol = 1 if n_models <= 4 else 2
    ax.legend(loc="upper left", fontsize=max(7, 10 - n_models), framealpha=0.9, ncol=ncol)
    max_r2 = all_df[[f"r2_{t}" for t in target_cols_present]].max().max()
    ax.set_ylim(0, min(1.0, max_r2 * 1.25))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot to %s", out_path)
    return out_path


def plot_mean_r2_progression(
    results_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "probe_progression.png",
) -> Path:
    """Line plot showing mean R2 across embedding sources (for ordered comparisons)."""
    valid = results_df.dropna(subset=["mean_r2"])
    if valid.empty:
        return output_dir / filename

    fig, ax = plt.subplots(figsize=(8, 5))

    names = valid["name"].values
    mean_r2 = valid["mean_r2"].values
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))

    ax.plot(range(len(names)), mean_r2, "o-", color="gray", linewidth=1.5, zorder=1)
    for i, (name, val) in enumerate(zip(names, mean_r2)):
        ax.scatter(
            i, val, s=120, color=colors[i], zorder=2,
            edgecolors="black", linewidth=0.5,
        )
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
    ax.set_title("Mean R2 Progression (DNN Probe)", fontsize=13)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved progression plot to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Custom embedding parsing
# ---------------------------------------------------------------------------


def parse_custom_embeddings(specs: List[str]) -> List[EmbeddingSource]:
    """Parse 'Name:path' strings from --embeddings into EmbeddingSource objects."""
    sources = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid embedding spec: {spec!r}\n"
                "Expected format: 'DisplayName:path/to/embeddings.parquet'"
            )
        name, path_str = spec.split(":", 1)
        name = name.strip()
        path_str = path_str.strip()

        if not Path(path_str).exists():
            raise FileNotFoundError(f"Embedding file not found: {path_str}")

        label = name.lower().replace(" ", "_").replace("-", "_")
        sources.append(
            EmbeddingSource(
                name=name,
                label=label,
                path=path_str,
                modality=label,
            )
        )
    return sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parametric DNN Probe Comparison Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Named configs:\n"
            + "\n".join(
                f"  {name:20s} {info['description']}"
                for name, info in NAMED_CONFIGS.items()
            )
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(NAMED_CONFIGS.keys()),
        help="Named comparison config to run",
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        type=str,
        default=None,
        help="Custom embeddings as 'Name:path.parquet' pairs",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available named configs and exit",
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
    parser.add_argument(
        "--no-progression",
        action="store_true",
        help="Skip mean R2 progression plot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: auto-generated under stage3/dnn_probe)",
    )
    parser.add_argument(
        "--write-standardized",
        action="store_true",
        help="Write standardized probe results (predictions + metrics parquets) "
        "via ProbeResultsWriter for cross-approach comparison",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.list_configs:
        print("\nAvailable named configs:\n")
        for name, info in NAMED_CONFIGS.items():
            print(f"  {name:20s}  {info['description']}")
            print(f"  {'':20s}  (original: {info['original_script']})")
        return

    if args.config is None and args.embeddings is None:
        parser.error("Specify --config or --embeddings (or --list-configs)")

    paths = StudyAreaPaths(STUDY_AREA)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        config_label = args.config or "custom"
        run_id = paths.create_run_id(f"{config_label}_comparison")
        output_dir = paths.stage3_run("dnn_probe", run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build embedding sources
    target_year = 2022  # default
    reference_results = None
    title = "DNN Probe Comparison"

    if args.config:
        config_info = NAMED_CONFIGS[args.config]
        target_year = config_info["target_year"]
        sources = config_info["builder"](paths, output_dir)
        title = f"DNN Probe: {config_info['description']}"
        logger.info("Config: %s (%s)", args.config, config_info["description"])
    else:
        sources = parse_custom_embeddings(args.embeddings)
        title = "DNN Probe: Custom Comparison"
        logger.info("Custom comparison with %d sources", len(sources))

    # Run probes
    results_df = run_probes(
        sources,
        target_year=target_year,
        run_prefix=args.config or "custom",
        dry_run=args.dry_run,
        write_standardized=args.write_standardized,
    )

    if args.dry_run or results_df.empty:
        return

    # Save CSV
    csv_path = output_dir / "probe_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved results to %s", csv_path)

    # Print comparison table
    print_comparison_table(results_df, reference_results)

    # Visualization
    if not args.no_viz:
        plot_comparison(
            results_df,
            output_dir,
            title=title,
            reference_results=reference_results,
        )
        if not args.no_progression and len(results_df) > 2:
            plot_mean_r2_progression(results_df, output_dir)

    logger.info("\nAll outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
