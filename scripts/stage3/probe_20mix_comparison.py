#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe Comparison: 4-Modality 208D Concat vs 128D UNet (20mix)

Runs DNN regression probes (leefbaarometer) and DNN clustering probes
(KMeans cluster label prediction) comparing the new 4-modality concat
embeddings against the UNet-fused embeddings, both with year label 20mix.

Regression probes:
    - Concat 208D -> leefbaarometer (6 targets)
    - UNet 128D -> leefbaarometer (6 targets)

Clustering probes:
    - KMeans(k=10) on Concat 208D -> cluster labels
    - DNN classification: Concat 208D -> cluster labels (self-prediction baseline)
    - DNN classification: UNet 128D -> cluster labels (cross-modal transfer)

Output:
    - probe_20mix_regression.csv: R2 per target x embedding source
    - probe_20mix_clustering.csv: accuracy/F1 per configuration
    - probe_20mix_regression.png: grouped bar chart
    - probe_20mix_clustering.png: clustering probe bar chart

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/probe_20mix_comparison.py
    python scripts/stage3/probe_20mix_comparison.py --regression-only
    python scripts/stage3/probe_20mix_comparison.py --clustering-only
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score
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
N_CLUSTERS = 10

# Embedding sources to compare
EMBEDDING_SOURCES = [
    {
        "name": "Concat 208D",
        "model": "concat",
        "label": "concat_208d",
    },
    {
        "name": "UNet 128D",
        "model": "unet",
        "label": "unet_128d",
    },
]

# Shared DNN probe hyperparameters
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


def _get_embedding_path(paths: StudyAreaPaths, model_name: str) -> Path:
    """Resolve embedding parquet path for a given model and 20mix year."""
    return (
        paths.model_embeddings(model_name)
        / f"{paths.study_area}_res{H3_RESOLUTION}_{YEAR}.parquet"
    )


def _build_regression_config(
    model_name: str, label: str, embeddings_path: str
) -> DNNProbeConfig:
    """Create a DNNProbeConfig for regression probing."""
    return DNNProbeConfig(
        study_area=STUDY_AREA,
        year=2022,  # target year for leefbaarometer
        h3_resolution=H3_RESOLUTION,
        modality=model_name,
        embeddings_path=embeddings_path,
        # Use res9 target
        target_path=str(
            StudyAreaPaths(STUDY_AREA).target_file("leefbaarometer", H3_RESOLUTION, 2022)
        ),
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
        run_descriptor=f"probe_20mix_{label}",
    )


# ---------------------------------------------------------------------------
# Regression probes
# ---------------------------------------------------------------------------


def run_regression_probes(output_dir: Path) -> pd.DataFrame:
    """Run DNN regression probes on all embedding sources."""
    paths = StudyAreaPaths(STUDY_AREA)

    logger.info("=" * 70)
    logger.info("DNN REGRESSION PROBES: 20mix Comparison")
    logger.info("=" * 70)

    all_rows: List[Dict[str, Any]] = []

    for i, src in enumerate(EMBEDDING_SOURCES, 1):
        model_name = src["model"]
        display_name = src["name"]
        label = src["label"]

        emb_path = str(_get_embedding_path(paths, model_name))

        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{i}/{len(EMBEDDING_SOURCES)}] {display_name}")
        logger.info(f"  Embeddings: {emb_path}")
        logger.info("=" * 70)

        run_start = time.time()
        try:
            config = _build_regression_config(model_name, label, emb_path)
            regressor = DNNProbeRegressor(config)
            results = regressor.run()
            regressor.save_results()

            row: Dict[str, Any] = {"name": display_name, "model": model_name}
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
            logger.error(f"  ERROR for {display_name}: {e}", exc_info=True)
            all_rows.append({
                "name": display_name,
                "model": model_name,
                "mean_r2": float("nan"),
                "duration_s": elapsed,
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_rows)

    # Save CSV
    csv_path = output_dir / "probe_20mix_regression.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved regression results to {csv_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("REGRESSION RESULTS (DNN Probe, 20mix)")
    logger.info("=" * 80)
    target_headers = "  ".join(f"{t:>8s}" for t in TARGET_COLS)
    logger.info(f"{'Source':22s}  {target_headers}  {'mean_r2':>8s}")
    logger.info("-" * 80)
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
        logger.info(f"{row['name']:22s}  {r2_line}  {mean_str:>8s}")
    logger.info("=" * 80)

    return results_df


def plot_regression_comparison(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar chart: R2 per target for concat vs UNet."""
    target_cols_present = [c for c in TARGET_COLS if f"r2_{c}" in results_df.columns]
    if not target_cols_present:
        logger.warning("No target columns found for plotting")
        return output_dir / "probe_20mix_regression.png"

    n_targets = len(target_cols_present)
    n_models = len(results_df)
    x = np.arange(n_targets)
    width = 0.8 / n_models

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_models))

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
        "DNN Regression Probe: Concat 208D vs UNet 128D (20mix)\n"
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
    out_path = output_dir / "probe_20mix_regression.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved regression plot to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Clustering probes
# ---------------------------------------------------------------------------


def run_clustering_probes(output_dir: Path) -> pd.DataFrame:
    """
    Run DNN clustering probes.

    1. KMeans(k=10) on Concat 208D -> cluster labels
    2. DNN classify: Concat 208D -> cluster labels (self-prediction sanity)
    3. DNN classify: UNet 128D -> cluster labels (cross-modal transfer)
    """
    from stage3_analysis.dnn_probe import MLPProbeModel, _make_activation

    paths = StudyAreaPaths(STUDY_AREA)
    project_root = Path(__file__).resolve().parent.parent.parent

    logger.info("=" * 70)
    logger.info("DNN CLUSTERING PROBES: 20mix Comparison")
    logger.info("=" * 70)

    # Load embeddings
    concat_path = _get_embedding_path(paths, "concat")
    unet_path = _get_embedding_path(paths, "unet")

    concat_df = pd.read_parquet(concat_path)
    unet_df = pd.read_parquet(unet_path)
    logger.info(f"Concat embeddings: {concat_df.shape}")
    logger.info(f"UNet embeddings: {unet_df.shape}")

    # Load target for spatial blocking (need geometry)
    target_path = paths.target_file("leefbaarometer", H3_RESOLUTION, 2022)
    target_df = pd.read_parquet(target_path)
    if target_df.index.name != "region_id" and "region_id" in target_df.columns:
        target_df = target_df.set_index("region_id")

    # Use only hexagons that have targets (to use same spatial blocks)
    common_idx = concat_df.index.intersection(unet_df.index).intersection(target_df.index)
    logger.info(f"Common hexagons (with targets): {len(common_idx):,}")

    concat_common = concat_df.loc[common_idx]
    unet_common = unet_df.loc[common_idx]

    # Step 1: KMeans on Concat
    logger.info(f"\nRunning KMeans(k={N_CLUSTERS}) on Concat 208D...")
    scaler_concat = StandardScaler()
    concat_scaled = scaler_concat.fit_transform(concat_common.values)

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, random_state=42, batch_size=10000, n_init=3,
    )
    cluster_labels = kmeans.fit_predict(concat_scaled)
    logger.info(f"  Cluster distribution: {np.bincount(cluster_labels)}")

    # Step 2: Spatial blocks for CV
    import geopandas as gpd
    from spatialkfold.blocks import spatial_blocks

    region_path = paths.region_file(H3_RESOLUTION)
    regions_gdf = gpd.read_parquet(project_root / region_path)
    if regions_gdf.index.name != "region_id":
        regions_gdf.index.name = "region_id"

    # Build a GeoDataFrame with centroids for spatial blocking
    geom_series = regions_gdf.loc[common_idx, "geometry"]
    points_gdf = gpd.GeoDataFrame(
        {"cluster": cluster_labels},
        geometry=geom_series.centroid.values,
        index=common_idx,
        crs="EPSG:4326",
    )
    points_proj = points_gdf.to_crs(epsg=28992)

    blocks_gdf = spatial_blocks(
        gdf=points_proj,
        nfolds=SHARED_PARAMS["n_folds"],
        width=SHARED_PARAMS["block_width"],
        height=SHARED_PARAMS["block_height"],
        method="random",
        orientation="tb-lr",
        grid_type="rect",
        random_state=42,
    )

    folds_joined = gpd.sjoin(
        points_proj[["geometry"]],
        blocks_gdf[["geometry", "folds"]],
        how="left",
        predicate="within",
    )
    missing = folds_joined["folds"].isna()
    if missing.any():
        folds_joined.loc[missing, "folds"] = 1
    folds_joined = folds_joined[~folds_joined.index.duplicated(keep="first")]
    fold_assignments = folds_joined.loc[common_idx, "folds"].values.astype(int)

    # Step 3: DNN classification on both embedding sets
    results_rows = []

    for emb_name, emb_values in [
        ("Concat 208D (self)", concat_common.values),
        ("UNet 128D (cross)", unet_common.values),
    ]:
        logger.info(f"\n--- Clustering DNN Probe: {emb_name} ---")
        run_start = time.time()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(emb_values)
        y = cluster_labels

        unique_folds = np.unique(fold_assignments)
        oof_preds = np.full(len(y), -1, dtype=int)
        fold_accs = []
        fold_f1s = []

        for fold_id in unique_folds:
            test_mask = fold_assignments == fold_id
            train_mask = ~test_mask

            X_train = torch.tensor(X_scaled[train_mask], dtype=torch.float32)
            y_train = torch.tensor(y[train_mask], dtype=torch.long)
            X_test = torch.tensor(X_scaled[test_mask], dtype=torch.float32)
            y_test_np = y[test_mask]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_dim = X_train.shape[1]

            model = MLPProbeModel(
                input_dim=input_dim,
                hidden_dim=SHARED_PARAMS["hidden_dim"],
                num_layers=SHARED_PARAMS["num_layers"],
                output_dim=N_CLUSTERS,
                activation=_make_activation(SHARED_PARAMS["activation"]),
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=SHARED_PARAMS["learning_rate"],
                weight_decay=SHARED_PARAMS["weight_decay"],
            )
            criterion = torch.nn.CrossEntropyLoss()

            # Training loop with early stopping
            best_loss = float("inf")
            patience_counter = 0
            best_state = None
            batch_size = SHARED_PARAMS["initial_batch_size"]

            X_train_dev = X_train.to(device)
            y_train_dev = y_train.to(device)

            for epoch in range(SHARED_PARAMS["max_epochs"]):
                model.train()
                perm = torch.randperm(len(X_train_dev))
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, len(X_train_dev), batch_size):
                    idx = perm[start:start + batch_size]
                    xb = X_train_dev[idx]
                    yb = y_train_dev[idx]

                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                if avg_loss < best_loss - SHARED_PARAMS.get("min_delta", 1e-5):
                    best_loss = avg_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= SHARED_PARAMS["patience"]:
                        break

            # Restore best model
            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(device)

            # Predict
            model.eval()
            with torch.no_grad():
                logits = model(X_test.to(device))
                preds = logits.argmax(dim=1).cpu().numpy()

            oof_preds[test_mask] = preds
            fold_acc = accuracy_score(y_test_np, preds)
            fold_f1 = f1_score(y_test_np, preds, average="macro")
            fold_accs.append(fold_acc)
            fold_f1s.append(fold_f1)

            logger.info(
                f"  Fold {fold_id}: acc={fold_acc:.4f}, f1_macro={fold_f1:.4f} "
                f"(train={train_mask.sum():,}, test={test_mask.sum():,})"
            )

        overall_acc = accuracy_score(y, oof_preds)
        overall_f1 = f1_score(y, oof_preds, average="macro")
        elapsed = time.time() - run_start

        logger.info(
            f"  Overall: acc={overall_acc:.4f}, f1_macro={overall_f1:.4f} "
            f"in {_format_duration(elapsed)}"
        )

        results_rows.append({
            "name": emb_name,
            "accuracy": overall_acc,
            "f1_macro": overall_f1,
            "acc_std": np.std(fold_accs),
            "f1_std": np.std(fold_f1s),
            "duration_s": elapsed,
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cluster_df = pd.DataFrame(results_rows)
    csv_path = output_dir / "probe_20mix_clustering.csv"
    cluster_df.to_csv(csv_path, index=False)
    logger.info(f"Saved clustering results to {csv_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("CLUSTERING PROBE RESULTS (20mix)")
    logger.info("=" * 70)
    logger.info(f"{'Source':30s}  {'Accuracy':>10s}  {'F1 Macro':>10s}")
    logger.info("-" * 55)
    for _, row in cluster_df.iterrows():
        logger.info(
            f"{row['name']:30s}  {row['accuracy']:10.4f}  {row['f1_macro']:10.4f}"
        )
    logger.info("=" * 70)

    return cluster_df


def plot_clustering_comparison(cluster_df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart of clustering probe accuracy and F1."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cluster_df)))
    x = np.arange(len(cluster_df))

    # Accuracy
    ax = axes[0]
    bars = ax.bar(
        x, cluster_df["accuracy"], yerr=cluster_df["acc_std"],
        capsize=4, color=colors, edgecolor="white", linewidth=0.5,
    )
    for bar, val in zip(bars, cluster_df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_df["name"], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Cluster Label Prediction: Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # F1
    ax = axes[1]
    bars = ax.bar(
        x, cluster_df["f1_macro"], yerr=cluster_df["f1_std"],
        capsize=4, color=colors, edgecolor="white", linewidth=0.5,
    )
    for bar, val in zip(bars, cluster_df["f1_macro"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_df["name"], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("F1 Macro", fontsize=11)
    ax.set_title("Cluster Label Prediction: F1 Macro", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        f"DNN Clustering Probes: KMeans(k={N_CLUSTERS}) on Concat 208D (20mix)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out_path = output_dir / "probe_20mix_clustering.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved clustering plot to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DNN Probe Comparison: Concat 208D vs UNet 128D (20mix)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--regression-only", action="store_true",
                        help="Run only regression probes")
    parser.add_argument("--clustering-only", action="store_true",
                        help="Run only clustering probes")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip generating comparison plots")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from datetime import date

    paths = StudyAreaPaths(STUDY_AREA)
    experiment_dir = paths.stage3("dnn_probe") / f"{date.today()}_probe_20mix_comparison"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_regression = not args.clustering_only
    run_clustering = not args.regression_only

    total_start = time.time()

    if run_regression:
        reg_df = run_regression_probes(experiment_dir)
        if not args.no_viz:
            plot_regression_comparison(reg_df, experiment_dir)

    if run_clustering:
        clust_df = run_clustering_probes(experiment_dir)
        if not args.no_viz:
            plot_clustering_comparison(clust_df, experiment_dir)

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal time: {_format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
