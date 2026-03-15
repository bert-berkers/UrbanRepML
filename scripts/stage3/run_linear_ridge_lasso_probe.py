#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear, Ridge, and Lasso probes on urban embeddings -> leefbaarometer.

Runs OLS, Ridge, and Lasso regression with the same 5-fold spatial block
cross-validation used by the DNN probes, enabling direct R-squared comparison
between linear and non-linear probes.

For each embedding source:
    - OLS (no regularization)
    - Ridge (alpha from {0.01, 0.1, 1.0, 10.0, 100.0}, inner CV)
    - Lasso (alpha from {1e-4, 1e-3, 1e-2, 0.1, 1.0}, inner CV)

Output: JSON results + comparison CSV + scratchpad-ready summary table.

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/run_linear_ridge_lasso_probe.py
    python scripts/stage3/run_linear_ridge_lasso_probe.py --embeddings-only ring_agg
    python scripts/stage3/run_linear_ridge_lasso_probe.py --embeddings-only concat
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from spatialkfold.blocks import spatial_blocks

from stage3_analysis.linear_probe import TARGET_COLS, TARGET_NAMES
from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

STUDY_AREA = "netherlands"
H3_RESOLUTION = 9
TARGET_YEAR = 2022
N_FOLDS = 5
BLOCK_SIZE = 10_000  # meters
RANDOM_STATE = 42

# Ridge / Lasso alpha grids
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
LASSO_ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]


def load_embeddings_and_targets(
    emb_path: Path,
    target_path: Path,
    regions_path: Path,
) -> gpd.GeoDataFrame:
    """Load embeddings, targets, regions; inner-join; return GeoDataFrame."""
    logger.info("Loading embeddings from %s", emb_path)
    emb_df = pd.read_parquet(emb_path)
    if emb_df.index.name != "region_id" and "region_id" in emb_df.columns:
        emb_df = emb_df.set_index("region_id")
    logger.info("  Shape: %s", emb_df.shape)

    # Identify feature columns (all numeric, exclude metadata)
    exclude = {"pixel_count", "tile_count", "geometry"}
    feature_cols = [
        c for c in emb_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(emb_df[c])
    ]
    logger.info("  Features: %d (%s .. %s)", len(feature_cols), feature_cols[0], feature_cols[-1])
    emb_df = emb_df[feature_cols]

    logger.info("Loading targets from %s", target_path)
    target_df = pd.read_parquet(target_path)
    if target_df.index.name != "region_id" and "region_id" in target_df.columns:
        target_df = target_df.set_index("region_id")

    joined = emb_df.join(target_df[list(TARGET_COLS)], how="inner")
    before = len(joined)
    joined = joined.dropna(subset=feature_cols + list(TARGET_COLS))
    logger.info("  Joined: %d rows (dropped %d NaN)", len(joined), before - len(joined))

    logger.info("Loading regions from %s", regions_path)
    regions_gdf = gpd.read_parquet(regions_path)
    if regions_gdf.index.name != "region_id":
        regions_gdf.index.name = "region_id"

    joined = joined.join(regions_gdf[["geometry"]], how="inner")
    joined = gpd.GeoDataFrame(joined, crs="EPSG:4326")
    joined["geometry"] = joined.geometry.centroid
    logger.info("  Final: %d rows, %d features", len(joined), len(feature_cols))

    return joined, feature_cols


def create_spatial_folds(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Create spatial block fold assignments matching DNN probe setup."""
    logger.info("Creating spatial blocks: %d folds, %dm blocks", N_FOLDS, BLOCK_SIZE)
    gdf_proj = gdf.to_crs(epsg=28992)

    blocks_gdf = spatial_blocks(
        gdf=gdf_proj,
        nfolds=N_FOLDS,
        width=BLOCK_SIZE,
        height=BLOCK_SIZE,
        method="random",
        orientation="tb-lr",
        grid_type="rect",
        random_state=RANDOM_STATE,
    )

    points_with_folds = gpd.sjoin(
        gdf_proj[["geometry"]],
        blocks_gdf[["geometry", "folds"]],
        how="left",
        predicate="within",
    )

    missing = points_with_folds["folds"].isna()
    if missing.any():
        logger.warning("  %d points outside blocks, assigning to fold 1", missing.sum())
        points_with_folds.loc[missing, "folds"] = 1

    points_with_folds = points_with_folds[~points_with_folds.index.duplicated(keep="first")]
    folds = points_with_folds.loc[gdf.index, "folds"].values.astype(int)

    fold_counts = {int(f): int((folds == f).sum()) for f in np.unique(folds)}
    logger.info("  Fold sizes: %s", fold_counts)
    return folds


def run_probe(
    model_class,
    model_kwargs: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Run one probe type with spatial CV. Returns per-fold + overall metrics."""
    unique_folds = np.unique(folds)
    oof_preds = np.full(len(y), np.nan)
    fold_metrics = []
    best_alphas = []

    for fold_id in unique_folds:
        test_mask = folds == fold_id
        train_mask = ~test_mask

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        y_train, y_test = y[train_mask], y[test_mask]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        oof_preds[test_mask] = y_pred

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        fold_metrics.append({
            "fold": int(fold_id),
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })

        # Track best alpha for CV models
        if hasattr(model, "alpha_"):
            best_alphas.append(model.alpha_)

    valid = ~np.isnan(oof_preds)
    overall_r2 = float(r2_score(y[valid], oof_preds[valid]))
    overall_rmse = float(np.sqrt(mean_squared_error(y[valid], oof_preds[valid])))
    overall_mae = float(mean_absolute_error(y[valid], oof_preds[valid]))

    result = {
        "model": model_name,
        "overall_r2": overall_r2,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "fold_metrics": fold_metrics,
        "fold_r2_std": float(np.std([f["r2"] for f in fold_metrics])),
    }
    if best_alphas:
        result["best_alpha_per_fold"] = [float(a) for a in best_alphas]
        result["median_alpha"] = float(np.median(best_alphas))

    return result


def run_all_probes(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    target_col: str,
) -> List[Dict[str, Any]]:
    """Run OLS, Ridge, and Lasso probes for one target."""
    results = []

    # OLS
    logger.info("    OLS ...")
    r = run_probe(LinearRegression, {}, X, y, folds, "OLS")
    results.append(r)
    logger.info("      R2=%.4f (std=%.4f)", r["overall_r2"], r["fold_r2_std"])

    # RidgeCV (efficient closed-form LOO-CV for alpha selection)
    logger.info("    Ridge (CV alpha from %s) ...", RIDGE_ALPHAS)
    r = run_probe(
        RidgeCV,
        {"alphas": RIDGE_ALPHAS, "scoring": "r2"},
        X, y, folds, "Ridge",
    )
    results.append(r)
    logger.info("      R2=%.4f (std=%.4f, median_alpha=%.4f)",
                r["overall_r2"], r["fold_r2_std"], r.get("median_alpha", 0))

    # Lasso with fixed alpha (LassoCV too slow for 208D x 130K)
    lasso_alpha = 0.001
    logger.info("    Lasso (alpha=%.4f, max_iter=5000) ...", lasso_alpha)
    r = run_probe(
        Lasso,
        {"alpha": lasso_alpha, "max_iter": 5000, "random_state": RANDOM_STATE},
        X, y, folds, "Lasso",
    )
    results.append(r)
    logger.info("      R2=%.4f (std=%.4f)",
                r["overall_r2"], r["fold_r2_std"])

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear/Ridge/Lasso probe comparison")
    parser.add_argument(
        "--embeddings-only",
        type=str,
        default=None,
        choices=["concat", "ring_agg"],
        help="Run only one embedding type (default: both)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    paths = StudyAreaPaths(STUDY_AREA)
    target_path = paths.target_file("leefbaarometer", H3_RESOLUTION, TARGET_YEAR)
    regions_path = paths.region_file(H3_RESOLUTION)

    # Embedding sources
    sources = {}

    concat_path = (
        paths.model_embeddings("concat")
        / f"{STUDY_AREA}_res{H3_RESOLUTION}_20mix.parquet"
    )
    ring_agg_path = Path(
        "data/study_areas/netherlands/stage3_analysis/dnn_probe/"
        "2026-03-14/2026-03-14_ring_agg_k10_comparison/ring_agg_k10_res9_20mix.parquet"
    )

    if args.embeddings_only == "concat":
        sources["Concat-208D"] = concat_path
    elif args.embeddings_only == "ring_agg":
        sources["RingAgg-k10-208D"] = ring_agg_path
    else:
        sources["Concat-208D"] = concat_path
        sources["RingAgg-k10-208D"] = ring_agg_path

    # Output directory
    today = "2026-03-15"
    output_dir = paths.stage3("linear_probe") / f"{today}_linear_ridge_lasso"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    comparison_rows = []

    for emb_name, emb_path in sources.items():
        logger.info("\n" + "=" * 70)
        logger.info("EMBEDDING: %s", emb_name)
        logger.info("  Path: %s", emb_path)
        logger.info("=" * 70)

        gdf, feature_cols = load_embeddings_and_targets(
            emb_path, target_path, regions_path,
        )
        folds = create_spatial_folds(gdf)
        X = gdf[feature_cols].values

        emb_results = {}
        for target_col in TARGET_COLS:
            target_name = TARGET_NAMES.get(target_col, target_col)
            logger.info("\n  --- %s (%s) ---", target_col, target_name)

            y = gdf[target_col].values
            logger.info("    y range: [%.4f, %.4f], mean=%.4f", y.min(), y.max(), y.mean())

            probe_results = run_all_probes(X, y, folds, target_col)
            emb_results[target_col] = probe_results

            # Build comparison rows
            for pr in probe_results:
                comparison_rows.append({
                    "embedding": emb_name,
                    "target": target_col,
                    "target_name": target_name,
                    "model": pr["model"],
                    "r2": pr["overall_r2"],
                    "r2_std": pr["fold_r2_std"],
                    "rmse": pr["overall_rmse"],
                    "mae": pr["overall_mae"],
                    "median_alpha": pr.get("median_alpha", None),
                })

        all_results[emb_name] = emb_results

    # Save full results JSON
    json_path = output_dir / "linear_probe_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nSaved full results to %s", json_path)

    # Save comparison CSV
    comp_df = pd.DataFrame(comparison_rows)
    csv_path = output_dir / "probe_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    logger.info("Saved comparison CSV to %s", csv_path)

    # Print summary table
    logger.info("\n" + "=" * 120)
    logger.info("SUMMARY: Linear vs Ridge vs Lasso (5-fold spatial block CV, 10km blocks)")
    logger.info("=" * 120)

    for emb_name in sources:
        logger.info("\n  %s:", emb_name)
        header = "  {:>8s}  ".format("Model") + "  ".join(f"{t:>8s}" for t in TARGET_COLS) + "  {:>8s}".format("Mean")
        logger.info(header)
        logger.info("  " + "-" * len(header))

        for model_name in ["OLS", "Ridge", "Lasso"]:
            r2_vals = []
            parts = []
            for t in TARGET_COLS:
                row = comp_df[
                    (comp_df["embedding"] == emb_name) &
                    (comp_df["target"] == t) &
                    (comp_df["model"] == model_name)
                ]
                if not row.empty:
                    val = row.iloc[0]["r2"]
                    r2_vals.append(val)
                    parts.append(f"{val:8.4f}")
                else:
                    parts.append(f"{'N/A':>8s}")

            mean_r2 = float(np.mean(r2_vals)) if r2_vals else float("nan")
            line = "  {:>8s}  ".format(model_name) + "  ".join(parts) + f"  {mean_r2:8.4f}"
            logger.info(line)

    # Print DNN reference values from existing probe results
    logger.info("\n  DNN reference (from 2026-03-14 ring_agg_k10_comparison):")
    dnn_ref = {
        "Concat-PCA-64D": {
            "lbm": 0.2858, "fys": 0.4133, "onv": 0.5025,
            "soc": 0.6435, "vrz": 0.7375, "won": 0.4680, "mean": 0.5084,
        },
        "RingAgg-k10-PCA-64D": {
            "lbm": 0.3035, "fys": 0.4447, "onv": 0.5230,
            "soc": 0.6600, "vrz": 0.7738, "won": 0.4856, "mean": 0.5318,
        },
    }
    for name, vals in dnn_ref.items():
        parts = [f"{vals[t]:8.4f}" for t in TARGET_COLS]
        line = "  {:>8s}  ".format(f"DNN({name[:10]})") + "  ".join(parts) + f"  {vals['mean']:8.4f}"
        logger.info(line)

    logger.info("\nNote: DNN used PCA-64D; linear probes here use full 208D.")
    logger.info("Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
