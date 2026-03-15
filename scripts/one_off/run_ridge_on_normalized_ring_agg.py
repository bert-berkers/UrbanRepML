"""
Ridge + OLS probes on normalized ring_agg k=10 embeddings (2026-03-15).

Fair comparison with DNN probe results from the same date on the same embedding.
Uses 5-fold spatial block CV (10km blocks) identical to DNN setup.

Lifetime: temporary (expires 2026-04-15)
Stage: 3
"""

import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from spatialkfold.blocks import spatial_blocks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from stage3_analysis.linear_probe import TARGET_COLS, TARGET_NAMES
from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

STUDY_AREA = "netherlands"
H3_RESOLUTION = 9
TARGET_YEAR = 2022
N_FOLDS = 5
BLOCK_SIZE = 10_000
RANDOM_STATE = 42
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    paths = StudyAreaPaths(STUDY_AREA)

    # The NEW normalized ring_agg k=10 embedding (created 2026-03-15 by hollow-walking-leaf)
    emb_path = paths.fused_embedding_file("ring_agg", H3_RESOLUTION, "20mix")
    target_path = paths.target_file("leefbaarometer", H3_RESOLUTION, TARGET_YEAR)
    regions_path = paths.region_file(H3_RESOLUTION)

    logger.info("Embedding: %s", emb_path)
    logger.info("Target: %s", target_path)
    logger.info("Regions: %s", regions_path)

    # Load embeddings
    emb_df = pd.read_parquet(emb_path)
    if emb_df.index.name != "region_id" and "region_id" in emb_df.columns:
        emb_df = emb_df.set_index("region_id")
    logger.info("Embeddings shape: %s", emb_df.shape)

    # Feature columns
    exclude = {"pixel_count", "tile_count", "geometry"}
    feature_cols = [
        c for c in emb_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(emb_df[c])
    ]
    logger.info("Features: %d (%s .. %s)", len(feature_cols), feature_cols[0], feature_cols[-1])
    emb_df = emb_df[feature_cols]

    # Load targets
    target_df = pd.read_parquet(target_path)
    if target_df.index.name != "region_id" and "region_id" in target_df.columns:
        target_df = target_df.set_index("region_id")

    joined = emb_df.join(target_df[list(TARGET_COLS)], how="inner")
    before = len(joined)
    joined = joined.dropna(subset=feature_cols + list(TARGET_COLS))
    logger.info("Joined: %d rows (dropped %d NaN)", len(joined), before - len(joined))

    # Load regions for geometry
    regions_gdf = gpd.read_parquet(regions_path)
    if regions_gdf.index.name != "region_id":
        regions_gdf.index.name = "region_id"

    joined = joined.join(regions_gdf[["geometry"]], how="inner")
    joined = gpd.GeoDataFrame(joined, crs="EPSG:4326")
    joined["geometry"] = joined.geometry.centroid
    logger.info("Final: %d rows, %d features", len(joined), len(feature_cols))

    # Spatial folds
    logger.info("Creating spatial blocks: %d folds, %dm blocks", N_FOLDS, BLOCK_SIZE)
    gdf_proj = joined.to_crs(epsg=28992)
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
        logger.warning("%d points outside blocks, assigning to fold 1", missing.sum())
        points_with_folds.loc[missing, "folds"] = 1
    points_with_folds = points_with_folds[~points_with_folds.index.duplicated(keep="first")]
    folds = points_with_folds.loc[joined.index, "folds"].values.astype(int)
    fold_counts = {int(f): int((folds == f).sum()) for f in np.unique(folds)}
    logger.info("Fold sizes: %s", fold_counts)

    X = joined[feature_cols].values

    # Output directory
    output_dir = paths.stage3("linear_probe") / "2026-03-15_normalized_ring_agg"
    output_dir.mkdir(parents=True, exist_ok=True)

    # DNN reference values (from metrics_summary.csv, same embedding, same date)
    dnn_r2 = {
        "lbm": 0.3414,
        "fys": 0.4736,
        "onv": 0.5392,
        "soc": 0.6725,
        "vrz": 0.8015,
        "won": 0.5065,
    }

    all_results = {}
    comparison_rows = []

    for target_col in TARGET_COLS:
        target_name = TARGET_NAMES.get(target_col, target_col)
        logger.info("\n--- %s (%s) ---", target_col, target_name)

        y = joined[target_col].values
        logger.info("  y range: [%.4f, %.4f], mean=%.4f", y.min(), y.max(), y.mean())

        unique_folds = np.unique(folds)

        for model_name, model_class, model_kwargs in [
            ("OLS", LinearRegression, {}),
            ("Ridge", RidgeCV, {"alphas": RIDGE_ALPHAS, "scoring": "r2"}),
        ]:
            logger.info("  %s ...", model_name)
            t0 = time.time()

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

                if hasattr(model, "alpha_"):
                    best_alphas.append(model.alpha_)

            valid = ~np.isnan(oof_preds)
            overall_r2 = float(r2_score(y[valid], oof_preds[valid]))
            overall_rmse = float(np.sqrt(mean_squared_error(y[valid], oof_preds[valid])))
            overall_mae = float(mean_absolute_error(y[valid], oof_preds[valid]))

            elapsed = time.time() - t0
            logger.info("    R2=%.4f, RMSE=%.4f, MAE=%.4f (%.1fs)", overall_r2, overall_rmse, overall_mae, elapsed)

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
                logger.info("    Best alphas: %s", best_alphas)

            all_results.setdefault(target_col, []).append(result)

            comparison_rows.append({
                "target": target_col,
                "target_name": target_name,
                "model": model_name,
                "r2": overall_r2,
                "r2_std": result["fold_r2_std"],
                "rmse": overall_rmse,
                "mae": overall_mae,
                "median_alpha": result.get("median_alpha"),
            })

    # Save results
    json_path = output_dir / "linear_probe_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nSaved results to %s", json_path)

    comp_df = pd.DataFrame(comparison_rows)
    csv_path = output_dir / "probe_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    logger.info("Saved comparison CSV to %s", csv_path)

    # Print comparison table
    logger.info("\n" + "=" * 100)
    logger.info("COMPARISON: Ridge vs DNN on Normalized RingAgg k=10 (208D, 5-fold spatial block CV)")
    logger.info("=" * 100)
    header = f"  {'Target':>6s}  {'OLS R2':>8s}  {'Ridge R2':>8s}  {'DNN R2':>8s}  {'Ridge-DNN':>10s}  {'Ridge/DNN':>10s}"
    logger.info(header)
    logger.info("  " + "-" * 68)

    ridge_vals = []
    dnn_vals = []
    ols_vals = []

    for t in TARGET_COLS:
        ols_row = comp_df[(comp_df["target"] == t) & (comp_df["model"] == "OLS")]
        ridge_row = comp_df[(comp_df["target"] == t) & (comp_df["model"] == "Ridge")]
        ols_r2 = ols_row.iloc[0]["r2"] if not ols_row.empty else float("nan")
        ridge_r2 = ridge_row.iloc[0]["r2"] if not ridge_row.empty else float("nan")
        dnn_r2_val = dnn_r2.get(t, float("nan"))
        gap = ridge_r2 - dnn_r2_val
        ratio = ridge_r2 / dnn_r2_val if dnn_r2_val != 0 else float("nan")

        ols_vals.append(ols_r2)
        ridge_vals.append(ridge_r2)
        dnn_vals.append(dnn_r2_val)

        logger.info(f"  {t:>6s}  {ols_r2:8.4f}  {ridge_r2:8.4f}  {dnn_r2_val:8.4f}  {gap:+10.4f}  {ratio:10.2%}")

    mean_ols = np.mean(ols_vals)
    mean_ridge = np.mean(ridge_vals)
    mean_dnn = np.mean(dnn_vals)
    mean_gap = mean_ridge - mean_dnn
    mean_ratio = mean_ridge / mean_dnn

    logger.info("  " + "-" * 68)
    logger.info(f"  {'Mean':>6s}  {mean_ols:8.4f}  {mean_ridge:8.4f}  {mean_dnn:8.4f}  {mean_gap:+10.4f}  {mean_ratio:10.2%}")

    logger.info("\nOutput dir: %s", output_dir)


if __name__ == "__main__":
    main()
