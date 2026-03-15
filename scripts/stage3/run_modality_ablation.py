#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-target modality ablation study using Ridge regression probes.

For each leefbaarometer target, tests:
  - Single modality (4 runs): AE-only, POI-only, Roads-only, GTFS-only
  - Drop-one (4 runs): all-but-AE, all-but-POI, all-but-Roads, all-but-GTFS
  - Full 4-modality baseline

Uses 5-fold spatial block CV (10km blocks) with RidgeCV, same setup as
run_linear_ridge_lasso_probe.py.

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/run_modality_ablation.py
    python scripts/stage3/run_modality_ablation.py --use-ring-agg
"""

import argparse
import json
import logging
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
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
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# Modality column groups (prefix -> label)
MODALITY_GROUPS = {
    "AE": {"prefix": "A", "label": "AlphaEarth (64D)"},
    "POI": {"prefix": "hex2vec_", "label": "Hex2Vec POI (50D)"},
    "Roads": {"prefix": "R", "label": "Roads H2V (30D)"},
    "GTFS": {"prefix": "gtfs2vec_", "label": "GTFS2Vec (64D)"},
}


def get_modality_columns(all_columns: List[str]) -> Dict[str, List[str]]:
    """Map each modality to its columns from the embedding dataframe."""
    groups = {}
    for mod_key, info in MODALITY_GROUPS.items():
        prefix = info["prefix"]
        # For single-char prefixes (A, R), match A00..A63 but not hex2vec_*
        if len(prefix) == 1:
            cols = [c for c in all_columns if c.startswith(prefix) and len(c) <= 3]
        else:
            cols = [c for c in all_columns if c.startswith(prefix)]
        groups[mod_key] = sorted(cols)
        logger.info("  %s: %d columns (%s .. %s)", mod_key, len(cols),
                     cols[0] if cols else "?", cols[-1] if cols else "?")
    return groups


def run_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
) -> Dict[str, float]:
    """Run RidgeCV with spatial folds, return overall metrics."""
    unique_folds = np.unique(folds)
    oof_preds = np.full(len(y), np.nan)

    for fold_id in unique_folds:
        test_mask = folds == fold_id
        train_mask = ~test_mask

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        model = RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2")
        model.fit(X_train, y[train_mask])
        oof_preds[test_mask] = model.predict(X_test)

    valid = ~np.isnan(oof_preds)
    return {
        "r2": float(r2_score(y[valid], oof_preds[valid])),
        "rmse": float(np.sqrt(mean_squared_error(y[valid], oof_preds[valid]))),
        "mae": float(mean_absolute_error(y[valid], oof_preds[valid])),
    }


def create_spatial_folds(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Create spatial block fold assignments."""
    gdf_proj = gdf.to_crs(epsg=28992)
    blocks_gdf = spatial_blocks(
        gdf=gdf_proj, nfolds=N_FOLDS, width=BLOCK_SIZE, height=BLOCK_SIZE,
        method="random", orientation="tb-lr", grid_type="rect",
        random_state=RANDOM_STATE,
    )
    points_with_folds = gpd.sjoin(
        gdf_proj[["geometry"]], blocks_gdf[["geometry", "folds"]],
        how="left", predicate="within",
    )
    missing = points_with_folds["folds"].isna()
    if missing.any():
        points_with_folds.loc[missing, "folds"] = 1
    points_with_folds = points_with_folds[~points_with_folds.index.duplicated(keep="first")]
    return points_with_folds.loc[gdf.index, "folds"].values.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modality ablation study")
    parser.add_argument("--use-ring-agg", action="store_true",
                        help="Use ring_agg k=10 embedding instead of raw concat")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    paths = StudyAreaPaths(STUDY_AREA)

    # Choose embedding source
    if args.use_ring_agg:
        emb_path = paths.fused_embedding_file("ring_agg", H3_RESOLUTION, "20mix")
        emb_label = "RingAgg-k10-208D"
    else:
        emb_path = paths.fused_embedding_file("concat", H3_RESOLUTION, "20mix")
        emb_label = "Concat-208D"

    logger.info("Embedding: %s -> %s", emb_label, emb_path)

    # Load data
    emb_df = pd.read_parquet(emb_path)
    if emb_df.index.name != "region_id" and "region_id" in emb_df.columns:
        emb_df = emb_df.set_index("region_id")

    all_feature_cols = [
        c for c in emb_df.columns
        if pd.api.types.is_numeric_dtype(emb_df[c])
        and c not in {"pixel_count", "tile_count", "geometry"}
    ]
    logger.info("Total features: %d", len(all_feature_cols))

    # Map modality groups
    mod_cols = get_modality_columns(all_feature_cols)
    total_mapped = sum(len(v) for v in mod_cols.values())
    logger.info("Mapped %d / %d columns to modalities", total_mapped, len(all_feature_cols))

    # Load targets
    target_path = paths.target_file("leefbaarometer", H3_RESOLUTION, TARGET_YEAR)
    target_df = pd.read_parquet(target_path)
    if target_df.index.name != "region_id" and "region_id" in target_df.columns:
        target_df = target_df.set_index("region_id")

    # Join
    joined = emb_df[all_feature_cols].join(target_df[list(TARGET_COLS)], how="inner")
    joined = joined.dropna(subset=all_feature_cols + list(TARGET_COLS))
    logger.info("Joined rows: %d", len(joined))

    # Load regions for spatial folds
    regions_gdf = gpd.read_parquet(paths.region_file(H3_RESOLUTION))
    if regions_gdf.index.name != "region_id":
        regions_gdf.index.name = "region_id"
    joined = joined.join(regions_gdf[["geometry"]], how="inner")
    joined = gpd.GeoDataFrame(joined, crs="EPSG:4326")
    joined["geometry"] = joined.geometry.centroid
    logger.info("Final rows: %d", len(joined))

    # Create folds once
    folds = create_spatial_folds(joined)
    fold_counts = {int(f): int((folds == f).sum()) for f in np.unique(folds)}
    logger.info("Fold sizes: %s", fold_counts)

    # Define ablation configurations
    mod_keys = list(MODALITY_GROUPS.keys())  # AE, POI, Roads, GTFS

    configs = []
    # Single modality
    for mk in mod_keys:
        configs.append((f"{mk}-only", [mk]))
    # Drop-one (3-modality combos)
    for mk in mod_keys:
        kept = [m for m in mod_keys if m != mk]
        configs.append((f"drop-{mk}", kept))
    # Full
    configs.append(("Full", mod_keys))

    # Run all ablations
    results = []
    t0 = time.time()

    for config_name, active_mods in configs:
        # Gather columns for active modalities
        active_cols = []
        for mk in active_mods:
            active_cols.extend(mod_cols[mk])
        n_dims = len(active_cols)

        logger.info("\n--- Config: %s (%d dims from %s) ---", config_name, n_dims, active_mods)

        X = joined[active_cols].values

        for target_col in TARGET_COLS:
            y = joined[target_col].values
            metrics = run_ridge_cv(X, y, folds)

            results.append({
                "config": config_name,
                "active_modalities": "+".join(active_mods),
                "n_dims": n_dims,
                "target": target_col,
                "target_name": TARGET_NAMES.get(target_col, target_col),
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
            })

            logger.info("  %s: R2=%.4f  RMSE=%.4f", target_col, metrics["r2"], metrics["rmse"])

    elapsed = time.time() - t0
    logger.info("\nTotal time: %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    # Build results dataframe
    results_df = pd.DataFrame(results)

    # Output directory
    today = "2026-03-15"
    suffix = "_ring_agg" if args.use_ring_agg else ""
    output_dir = paths.stage3("linear_probe") / f"{today}_feature_ablation{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)
    logger.info("Saved: %s", output_dir / "ablation_results.csv")

    # Save as JSON too
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Build heatmap-style pivot table
    pivot = results_df.pivot_table(
        index="config", columns="target", values="r2", aggfunc="first"
    )
    # Reorder columns
    pivot = pivot[list(TARGET_COLS)]
    pivot["Mean"] = pivot.mean(axis=1)

    # Reorder rows logically
    row_order = [f"{mk}-only" for mk in mod_keys] + [f"drop-{mk}" for mk in mod_keys] + ["Full"]
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])

    # Save pivot
    pivot.to_csv(output_dir / "ablation_heatmap.csv")
    logger.info("Saved: %s", output_dir / "ablation_heatmap.csv")

    # Print summary table
    logger.info("\n" + "=" * 100)
    logger.info("MODALITY ABLATION RESULTS -- %s (Ridge, 5-fold spatial CV)", emb_label)
    logger.info("=" * 100)

    header = f"{'Config':>14s}  " + "  ".join(f"{t:>8s}" for t in TARGET_COLS) + f"  {'Mean':>8s}"
    logger.info(header)
    logger.info("-" * len(header))
    for config_name in row_order:
        if config_name in pivot.index:
            parts = [f"{pivot.loc[config_name, t]:8.4f}" for t in TARGET_COLS]
            mean_val = pivot.loc[config_name, "Mean"]
            line = f"{config_name:>14s}  " + "  ".join(parts) + f"  {mean_val:8.4f}"
            logger.info(line)
        if config_name == f"{mod_keys[-1]}-only":
            logger.info("-" * len(header))

    # Compute delta from full
    logger.info("\n--- Delta from Full (negative = dropping hurts) ---")
    full_row = pivot.loc["Full"]
    for config_name in row_order:
        if config_name in pivot.index and config_name != "Full":
            delta = pivot.loc[config_name] - full_row
            parts = [f"{delta[t]:+8.4f}" for t in TARGET_COLS]
            mean_delta = delta["Mean"]
            line = f"{config_name:>14s}  " + "  ".join(parts) + f"  {mean_delta:+8.4f}"
            logger.info(line)

    # Key findings
    logger.info("\n--- Key Findings ---")
    for target_col in TARGET_COLS:
        # Best single modality
        single_r2 = {mk: pivot.loc[f"{mk}-only", target_col] for mk in mod_keys}
        best_single = max(single_r2, key=single_r2.get)
        # Most important (biggest drop when removed)
        drop_delta = {mk: full_row[target_col] - pivot.loc[f"drop-{mk}", target_col] for mk in mod_keys}
        most_important = max(drop_delta, key=drop_delta.get)
        # Noise check: does dropping improve?
        noise_mods = [mk for mk in mod_keys if drop_delta[mk] < -0.001]

        logger.info(
            "  %s (%s): best_single=%s(%.3f), biggest_drop=%s(%.4f)%s",
            target_col, TARGET_NAMES.get(target_col, ""),
            best_single, single_r2[best_single],
            most_important, drop_delta[most_important],
            f", NOISE={noise_mods}" if noise_mods else "",
        )

    logger.info("\nOutput: %s", output_dir)


if __name__ == "__main__":
    main()
