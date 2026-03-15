#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ARCHIVED 2026-03-15: Replaced by scripts/stage3/run_probe_comparison.py
# Original purpose: Concat 208D vs UNet Multiscale-Concat 192D comparison
# Archived because: 8 hardcoded paths, duplicated training loop, duplicated viz helpers
"""
DNN Probe Comparison: Concat 208D vs UNet Multiscale-Concat 192D

Builds UNet multiscale-concat (192D) by upsampling res8 and res7 embeddings
to res9 via H3 parent-child hierarchy, then concatenating with res9 embeddings.
Runs DNN regression probes on both representations against 6 leefbaarometer targets.

Outputs:
    - regression_r2.csv: R2 per target x model
    - regression_r2_bar.png: grouped bar chart
    - spatial prediction/residual maps (per target)
    - spatial_improvement.png: where UNet multiscale beats concat

Lifetime: durable
Stage: 3

Usage:
    python scripts/stage3/probe_20mix_multiscale.py
    python scripts/stage3/probe_20mix_multiscale.py --skip-spatial
    python scripts/stage3/probe_20mix_multiscale.py --skip-clustering-probe
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import h3
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stage3_analysis.dnn_probe import DNNProbeConfig, DNNProbeRegressor
from stage3_analysis.linear_probe import TARGET_COLS, TARGET_NAMES
from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDY_AREA = "netherlands"
H3_RESOLUTION = 9
YEAR = "20mix"

DATA_ROOT = Path("data/study_areas/netherlands")

CONCAT_PATH = DATA_ROOT / "stage2_multimodal/concat/embeddings/netherlands_res9_20mix.parquet"
UNET9_PATH = DATA_ROOT / "stage2_multimodal/unet/embeddings/netherlands_res9_20mix.parquet"
UNET8_PATH = DATA_ROOT / "stage2_multimodal/unet/embeddings/netherlands_res8_20mix.parquet"
UNET7_PATH = DATA_ROOT / "stage2_multimodal/unet/embeddings/netherlands_res7_20mix.parquet"

TARGET_PATH = DATA_ROOT / "target/leefbaarometer/leefbaarometer_h3res9_2022.parquet"
REGIONS_PATH = DATA_ROOT / "regions_gdf/netherlands_res9.parquet"

OUTPUT_BASE = DATA_ROOT / "stage3_analysis/dnn_probe/2026-03-08_probe_multiscale_comparison"
CLUSTER_OUTPUT = DATA_ROOT / "stage2_multimodal/clustering/2026-03-08_clustering_probe"

# DNN probe hyperparameters
PROBE_PARAMS: Dict[str, Any] = {
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


def upsample_to_res9(parent_df: pd.DataFrame, parent_res: int) -> pd.DataFrame:
    """Broadcast parent embeddings to all res9 children via H3 hierarchy.

    Args:
        parent_df: DataFrame indexed by region_id (parent hex), columns are embeddings.
        parent_res: H3 resolution of the parent (7 or 8).

    Returns:
        DataFrame indexed by res9 region_id with parent embeddings replicated.
    """
    rows = []
    for parent_id in parent_df.index:
        children = h3.cell_to_children(parent_id, 9)
        for child in children:
            rows.append({"region_id": child, "parent_id": parent_id})
    mapping = pd.DataFrame(rows).set_index("region_id")
    result = mapping.join(parent_df, on="parent_id").drop(columns="parent_id")
    logger.info(
        f"  Upsampled res{parent_res} ({len(parent_df):,}) -> "
        f"res9 ({len(result):,} children)"
    )
    return result


def build_unet_multiscale() -> pd.DataFrame:
    """Build 192D UNet multiscale-concat embeddings at res9.

    Loads res9/8/7 UNet embeddings, renames columns to avoid collision,
    upsamples res8 and res7 to res9, then inner-joins all three.
    """
    logger.info("Loading UNet embeddings...")
    unet9 = pd.read_parquet(project_root / UNET9_PATH)
    unet8 = pd.read_parquet(project_root / UNET8_PATH)
    unet7 = pd.read_parquet(project_root / UNET7_PATH)

    logger.info(f"  res9: {unet9.shape}, res8: {unet8.shape}, res7: {unet7.shape}")

    # Rename columns: unet_X -> unet9_X, unet8_X, unet7_X
    unet9.columns = [f"unet9_{c.split('_', 1)[1]}" for c in unet9.columns]
    unet8.columns = [f"unet8_{c.split('_', 1)[1]}" for c in unet8.columns]
    unet7.columns = [f"unet7_{c.split('_', 1)[1]}" for c in unet7.columns]

    logger.info("Upsampling res8 -> res9...")
    unet8_up = upsample_to_res9(unet8, 8)

    logger.info("Upsampling res7 -> res9...")
    unet7_up = upsample_to_res9(unet7, 7)

    # Inner join on res9 index
    unet_ms = unet9.join(unet8_up, how="inner").join(unet7_up, how="inner")
    logger.info(
        f"UNet multiscale-concat: {unet_ms.shape} "
        f"(inner join survived: {len(unet_ms):,} / {len(unet9):,} res9 hexes)"
    )
    return unet_ms


def load_target_and_join(
    embeddings: pd.DataFrame, label: str
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load leefbaarometer targets, join with embeddings, create spatial blocks.

    Returns:
        (joined_df, X, folds, region_ids)
    """
    target_df = pd.read_parquet(project_root / TARGET_PATH)
    if target_df.index.name != "region_id" and "region_id" in target_df.columns:
        target_df = target_df.set_index("region_id")

    joined = embeddings.join(target_df[TARGET_COLS], how="inner").dropna()
    logger.info(f"[{label}] Joined with targets: {len(joined):,} hexagons")

    # Need geometry for spatial blocking
    regions_gdf = gpd.read_parquet(project_root / REGIONS_PATH)
    if regions_gdf.index.name != "region_id":
        regions_gdf.index.name = "region_id"

    joined = joined.join(regions_gdf[["geometry"]], how="inner")
    joined = gpd.GeoDataFrame(joined, crs="EPSG:4326")
    joined["geometry"] = joined.geometry.centroid

    feature_cols = [c for c in embeddings.columns if c in joined.columns]
    X = joined[feature_cols].values
    region_ids = joined.index.values

    # Spatial block CV
    from spatialkfold.blocks import spatial_blocks

    gdf_proj = joined.to_crs(epsg=28992)
    blocks_gdf = spatial_blocks(
        gdf=gdf_proj,
        nfolds=PROBE_PARAMS["n_folds"],
        width=PROBE_PARAMS["block_width"],
        height=PROBE_PARAMS["block_height"],
        method="random",
        orientation="tb-lr",
        grid_type="rect",
        random_state=42,
    )
    points_with_folds = gpd.sjoin(
        gdf_proj[["geometry"]],
        blocks_gdf[["geometry", "folds"]],
        how="left",
        predicate="within",
    )
    missing = points_with_folds["folds"].isna()
    if missing.any():
        points_with_folds.loc[missing, "folds"] = 1
    points_with_folds = points_with_folds[
        ~points_with_folds.index.duplicated(keep="first")
    ]
    folds = points_with_folds.loc[joined.index, "folds"].values.astype(int)

    return joined, X, folds, region_ids


def run_dnn_probe(
    embeddings: pd.DataFrame,
    label: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run DNN probe for all 6 targets on the given embeddings.

    Returns dict with {target_col: TargetResult} and metadata.
    """
    import torch
    from stage3_analysis.dnn_probe import (
        MLPProbeModel,
        _make_activation,
    )
    from stage3_analysis.linear_probe import FoldMetrics, TargetResult
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    output_dir.mkdir(parents=True, exist_ok=True)

    joined, X, folds, region_ids = load_target_and_join(embeddings, label)
    feature_cols = [c for c in embeddings.columns if c in joined.columns]
    y_all = joined[TARGET_COLS]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{label}] Device: {device}, Features: {X.shape[1]}D, Samples: {X.shape[0]:,}")

    results: Dict[str, TargetResult] = {}
    training_curves: Dict[str, Dict[int, list]] = {}

    for target_col in TARGET_COLS:
        y = y_all[target_col].values
        unique_folds = np.unique(folds)
        target_name = TARGET_NAMES.get(target_col, target_col)

        logger.info(f"\n[{label}] --- {target_col} ({target_name}) ---")

        oof_predictions = np.full(len(y), np.nan)
        fold_metrics_list = []
        tc_curves: Dict[int, list] = {}

        for fold_id in unique_folds:
            val_mask = folds == fold_id
            train_mask = ~val_mask

            # Per-fold standardization
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask]).astype(np.float32)
            X_val = scaler.transform(X[val_mask]).astype(np.float32)

            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y[train_mask].reshape(-1, 1)).ravel().astype(np.float32)
            y_val_std = y_scaler.transform(y[val_mask].reshape(-1, 1)).ravel().astype(np.float32)

            x_train_t = torch.tensor(X_train, device=device)
            x_val_t = torch.tensor(X_val, device=device)
            y_train_t = torch.tensor(y_train, device=device)
            y_val_t = torch.tensor(y_val_std, device=device)

            model = MLPProbeModel(
                input_dim=X.shape[1],
                hidden_dim=PROBE_PARAMS["hidden_dim"],
                num_layers=PROBE_PARAMS["num_layers"],
                use_layer_norm=True,
                activation=_make_activation(PROBE_PARAMS["activation"]),
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=PROBE_PARAMS["learning_rate"],
                weight_decay=PROBE_PARAMS["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=PROBE_PARAMS["max_epochs"]
            )
            criterion = torch.nn.MSELoss()

            n_train = x_train_t.shape[0]
            bs = PROBE_PARAMS["initial_batch_size"]
            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None
            val_loss_history = []

            import math
            import copy

            for epoch in range(PROBE_PARAMS["max_epochs"]):
                # Progressive batch size
                progress = epoch / max(PROBE_PARAMS["max_epochs"] - 1, 1)
                log_bs = math.log(bs) + progress * (math.log(n_train) - math.log(bs))
                current_bs = min(max(int(round(math.exp(log_bs))), 1), n_train)

                model.train()
                perm = torch.randperm(n_train, device=device)
                for start in range(0, n_train, current_bs):
                    idx = perm[start:start + current_bs]
                    optimizer.zero_grad()
                    out = model(x_train_t[idx])
                    loss = criterion(out.squeeze(-1), y_train_t[idx])
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_out = model(x_val_t)
                    val_loss = criterion(val_out.squeeze(-1), y_val_t).item()
                val_loss_history.append(val_loss)

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= PROBE_PARAMS["patience"]:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            model = model.cpu().eval()
            tc_curves[int(fold_id)] = val_loss_history

            with torch.no_grad():
                val_pred_std = model(torch.tensor(X_val)).squeeze(-1).numpy()
            val_pred = y_scaler.inverse_transform(val_pred_std.reshape(-1, 1)).ravel()
            oof_predictions[val_mask] = val_pred

            y_val = y[val_mask]
            rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            mae = float(mean_absolute_error(y_val, val_pred))
            r2 = float(r2_score(y_val, val_pred))
            fold_metrics_list.append(
                FoldMetrics(fold=int(fold_id), rmse=rmse, mae=mae, r2=r2,
                            n_train=int(train_mask.sum()), n_test=int(val_mask.sum()))
            )
            logger.info(
                f"    Fold {fold_id}: R2={r2:.4f}, RMSE={rmse:.4f} "
                f"(epochs={len(val_loss_history)})"
            )

            del x_train_t, x_val_t, y_train_t, y_val_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        training_curves[target_col] = tc_curves

        valid = ~np.isnan(oof_predictions)
        overall_r2 = float(r2_score(y[valid], oof_predictions[valid]))
        overall_rmse = float(np.sqrt(mean_squared_error(y[valid], oof_predictions[valid])))
        overall_mae = float(mean_absolute_error(y[valid], oof_predictions[valid]))

        logger.info(f"  [{label}] Overall OOF: R2={overall_r2:.4f}, RMSE={overall_rmse:.4f}")

        result = TargetResult(
            target=target_col,
            target_name=target_name,
            best_alpha=0.0,
            best_l1_ratio=0.0,
            fold_metrics=fold_metrics_list,
            overall_r2=overall_r2,
            overall_rmse=overall_rmse,
            overall_mae=overall_mae,
            coefficients=np.zeros(len(feature_cols)),
            intercept=0.0,
            feature_names=feature_cols,
            oof_predictions=oof_predictions,
            actual_values=y,
            region_ids=region_ids,
        )
        results[target_col] = result

        # Save predictions
        pred_df = pd.DataFrame({
            "region_id": region_ids,
            "actual": y,
            "predicted": oof_predictions,
            "residual": y - oof_predictions,
        }).set_index("region_id")
        pred_df.to_parquet(output_dir / f"predictions_{target_col}.parquet")

    # Save metrics summary
    rows = []
    for tc, res in results.items():
        row = {"target": tc, "target_name": res.target_name,
               "overall_r2": res.overall_r2, "overall_rmse": res.overall_rmse,
               "overall_mae": res.overall_mae, "n_features": len(feature_cols)}
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)

    return {"results": results, "training_curves": training_curves, "region_ids": region_ids}


# ---------------------------------------------------------------------------
# Comparison Visualization
# ---------------------------------------------------------------------------


def plot_r2_comparison(
    concat_results: Dict[str, Any],
    unet_results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Grouped bar chart: R2 per target for concat vs UNet multiscale."""
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = TARGET_COLS
    labels = [TARGET_NAMES.get(t, t) for t in targets]
    concat_r2 = [concat_results["results"][t].overall_r2 for t in targets]
    unet_r2 = [unet_results["results"][t].overall_r2 for t in targets]

    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_c = ax.bar(x - width / 2, concat_r2, width, label="Concat 208D",
                    color="steelblue", edgecolor="white")
    bars_u = ax.bar(x + width / 2, unet_r2, width, label="UNet Multiscale 192D",
                    color="coral", edgecolor="white")

    for bar in bars_c:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)
    for bar in bars_u:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

    # Delta labels
    for i in range(len(targets)):
        delta = unet_r2[i] - concat_r2[i]
        y_pos = max(concat_r2[i], unet_r2[i]) + 0.025
        sign = "+" if delta >= 0 else ""
        ax.text(x[i], y_pos, f"{sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="darkgreen" if delta >= 0 else "darkred")

    ax.set_ylabel("R-squared (5-fold spatial block CV)", fontsize=12)
    ax.set_title(
        "DNN Probe Comparison: Concat 208D vs UNet Multiscale-Concat 192D\n"
        f"MLP h={PROBE_PARAMS['hidden_dim']}, patience={PROBE_PARAMS['patience']}, "
        f"max_epochs={PROBE_PARAMS['max_epochs']} | Leefbaarometer 2022 targets at res9",
        fontsize=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11, loc="upper right")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = output_dir / "regression_r2_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved R2 comparison bar chart: {path}")
    return path


def save_r2_table(
    concat_results: Dict[str, Any],
    unet_results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Save R2 CSV table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in TARGET_COLS:
        rows.append({
            "target": t,
            "target_name": TARGET_NAMES.get(t, t),
            "concat_208D_r2": concat_results["results"][t].overall_r2,
            "concat_208D_rmse": concat_results["results"][t].overall_rmse,
            "unet_ms_192D_r2": unet_results["results"][t].overall_r2,
            "unet_ms_192D_rmse": unet_results["results"][t].overall_rmse,
            "delta_r2": (unet_results["results"][t].overall_r2 -
                         concat_results["results"][t].overall_r2),
        })
    df = pd.DataFrame(rows)
    path = output_dir / "regression_r2.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved R2 table: {path}")
    return path


# ---------------------------------------------------------------------------
# Spatial Maps
# ---------------------------------------------------------------------------


def _load_boundary():
    """Load Netherlands boundary in EPSG:28992."""
    from shapely import get_geometry, get_num_geometries
    paths = StudyAreaPaths(STUDY_AREA)
    boundary_path = paths.area_gdf_file()
    if not boundary_path.exists():
        return None
    boundary_gdf = gpd.read_file(boundary_path)
    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    boundary_gdf = boundary_gdf.to_crs(epsg=28992)
    geom = boundary_gdf.geometry.iloc[0]
    n_parts = get_num_geometries(geom)
    if n_parts > 1:
        euro_geom = max(
            (get_geometry(geom, i) for i in range(n_parts)),
            key=lambda g: g.area,
        )
        boundary_gdf = gpd.GeoDataFrame(geometry=[euro_geom], crs=boundary_gdf.crs)
    return boundary_gdf


def _rasterize_continuous(cx, cy, values, extent, width=2000, height=2400,
                          cmap_name="RdBu", vcenter=0.0, vmin=None, vmax=None):
    """Rasterize continuous values to RGBA image with diverging colormap."""
    minx, miny, maxx, maxy = extent
    mask = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    cx_m, cy_m, val_m = cx[mask], cy[mask], values[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    if vmin is None:
        vmin = float(np.nanquantile(val_m, 0.02))
    if vmax is None:
        vmax = float(np.nanquantile(val_m, 0.98))

    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    colormap = plt.get_cmap(cmap_name)
    rgb = colormap(norm(val_m))[:, :3].astype(np.float32)

    image = np.zeros((height, width, 4), dtype=np.float32)
    # stamp=2 for res9
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            sy = np.clip(py + dy, 0, height - 1)
            sx = np.clip(px + dx, 0, width - 1)
            image[sy, sx, :3] = rgb
            image[sy, sx, 3] = 1.0

    return image, norm, colormap


def plot_spatial_improvement(
    concat_results: Dict[str, Any],
    unet_results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Map where UNet multiscale beats concat (aggregated across targets)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute mean |residual| difference across all targets
    all_improvements = {}
    for t in TARGET_COLS:
        c_res = concat_results["results"][t]
        u_res = unet_results["results"][t]

        c_df = pd.DataFrame({
            "region_id": c_res.region_ids,
            "c_abs_resid": np.abs(c_res.actual_values - c_res.oof_predictions),
        }).set_index("region_id")

        u_df = pd.DataFrame({
            "region_id": u_res.region_ids,
            "u_abs_resid": np.abs(u_res.actual_values - u_res.oof_predictions),
        }).set_index("region_id")

        merged = c_df.join(u_df, how="inner")
        merged["improvement"] = merged["c_abs_resid"] - merged["u_abs_resid"]
        all_improvements[t] = merged["improvement"]

    # Average improvement across targets
    imp_df = pd.DataFrame(all_improvements)
    mean_imp = imp_df.mean(axis=1)

    # Get centroids
    db = SpatialDB.for_study_area(STUDY_AREA)
    hex_ids = mean_imp.index
    cx, cy = db.centroids(hex_ids, resolution=H3_RESOLUTION, crs=28992)

    boundary_gdf = _load_boundary()
    if boundary_gdf is not None:
        ext = boundary_gdf.total_bounds
    else:
        ext = db.extent(hex_ids, resolution=H3_RESOLUTION, crs=28992)
    minx, miny, maxx, maxy = ext
    pad = (maxx - minx) * 0.03
    extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    vals = mean_imp.values
    vmax = float(np.abs(vals).clip(0, np.quantile(np.abs(vals), 0.98)).max())
    if vmax == 0:
        vmax = 0.01

    image, norm, colormap = _rasterize_continuous(
        cx, cy, vals, extent, cmap_name="RdBu",
        vcenter=0.0, vmin=-vmax, vmax=vmax,
    )

    fig, ax = plt.subplots(figsize=(12, 14))
    fig.set_facecolor("white")

    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)

    ax.imshow(
        image,
        extent=[extent[0], extent[2], extent[1], extent[3]],
        origin="lower", aspect="equal", interpolation="nearest", zorder=2,
    )
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7,
                 label="|Concat resid| - |UNet resid| (mean over 6 targets)\n(+ = UNet better)")

    n_unet_better = int((vals > 0).sum())
    n_concat_better = int((vals < 0).sum())
    ax.set_title(
        f"Spatial Improvement: UNet Multiscale 192D vs Concat 208D\n"
        f"Blue = UNet better ({n_unet_better:,}) | Red = Concat better ({n_concat_better:,})\n"
        f"Mean across 6 targets | {len(vals):,} hexagons",
        fontsize=11, pad=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    path = output_dir / "spatial_improvement.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved spatial improvement map: {path}")
    return path


def plot_spatial_predictions_and_residuals(
    concat_results: Dict[str, Any],
    unet_results: Dict[str, Any],
    output_dir: Path,
    targets: List[str] = None,
):
    """Side-by-side spatial prediction and residual maps per target."""
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = targets or TARGET_COLS

    db = SpatialDB.for_study_area(STUDY_AREA)
    boundary_gdf = _load_boundary()

    for tc in targets:
        c_res = concat_results["results"][tc]
        u_res = unet_results["results"][tc]

        # Use concat's hex set for centroids (they overlap with UNet)
        c_df = pd.DataFrame({
            "region_id": c_res.region_ids,
            "c_pred": c_res.oof_predictions,
            "c_resid": c_res.actual_values - c_res.oof_predictions,
            "actual": c_res.actual_values,
        }).set_index("region_id")

        u_df = pd.DataFrame({
            "region_id": u_res.region_ids,
            "u_pred": u_res.oof_predictions,
            "u_resid": u_res.actual_values - u_res.oof_predictions,
        }).set_index("region_id")

        merged = c_df.join(u_df, how="inner").dropna()
        hex_ids = merged.index
        cx, cy = db.centroids(hex_ids, resolution=H3_RESOLUTION, crs=28992)

        if boundary_gdf is not None:
            ext = boundary_gdf.total_bounds
        else:
            ext = db.extent(hex_ids, resolution=H3_RESOLUTION, crs=28992)
        minx, miny, maxx, maxy = ext
        pad = (maxx - minx) * 0.03
        extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

        target_name = TARGET_NAMES.get(tc, tc)

        # --- Predictions side by side ---
        vmin_p = float(merged["actual"].quantile(0.02))
        vmax_p = float(merged["actual"].quantile(0.98))

        fig, axes = plt.subplots(1, 2, figsize=(22, 14))
        fig.set_facecolor("white")

        for ax_idx, (col, label) in enumerate([
            ("c_pred", "Concat 208D"),
            ("u_pred", "UNet MS 192D"),
        ]):
            ax = axes[ax_idx]
            vals = merged[col].values
            img, norm_p, cmap_p = _rasterize_continuous(
                cx, cy, vals, extent, cmap_name="viridis",
                vcenter=None, vmin=vmin_p, vmax=vmax_p,
            )
            if boundary_gdf is not None:
                boundary_gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)
            ax.imshow(img, extent=[extent[0], extent[2], extent[1], extent[3]],
                      origin="lower", aspect="equal", interpolation="nearest", zorder=2)
            ax.set_xlim(extent[0], extent[2])
            ax.set_ylim(extent[1], extent[3])
            ax.set_title(f"{label} predictions", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

        sm = cm.ScalarMappable(cmap=cmap_p, norm=norm_p)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, shrink=0.5, label=f"{target_name} predicted value")
        fig.suptitle(f"Predicted {target_name} ({tc})", fontsize=13, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = output_dir / f"regression_spatial_pred_{tc}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved prediction map: {path}")

        # --- Residuals side by side ---
        all_resid = np.concatenate([merged["c_resid"].values, merged["u_resid"].values])
        vmax_r = float(np.quantile(np.abs(all_resid), 0.98))

        fig, axes = plt.subplots(1, 2, figsize=(22, 14))
        fig.set_facecolor("white")

        for ax_idx, (col, label) in enumerate([
            ("c_resid", "Concat 208D"),
            ("u_resid", "UNet MS 192D"),
        ]):
            ax = axes[ax_idx]
            vals = merged[col].values
            img, norm_r, cmap_r = _rasterize_continuous(
                cx, cy, vals, extent, cmap_name="RdBu",
                vcenter=0.0, vmin=-vmax_r, vmax=vmax_r,
            )
            if boundary_gdf is not None:
                boundary_gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)
            ax.imshow(img, extent=[extent[0], extent[2], extent[1], extent[3]],
                      origin="lower", aspect="equal", interpolation="nearest", zorder=2)
            ax.set_xlim(extent[0], extent[2])
            ax.set_ylim(extent[1], extent[3])
            ax.set_title(f"{label} residuals", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

        sm = cm.ScalarMappable(cmap=cmap_r, norm=norm_r)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, shrink=0.5, label=f"Residual (actual - predicted)")
        fig.suptitle(f"Residuals: {target_name} ({tc})", fontsize=13, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = output_dir / f"regression_spatial_residual_{tc}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved residual map: {path}")


# ---------------------------------------------------------------------------
# Clustering Probe
# ---------------------------------------------------------------------------


def run_clustering_probe(
    concat_emb: pd.DataFrame,
    unet_ms_emb: pd.DataFrame,
    output_dir: Path,
    k: int = 10,
):
    """Train classifier to predict concat cluster labels from UNet MS embeddings.

    This tests whether UNet MS can recover the same spatial typology as concat.
    """
    import torch
    from stage3_analysis.dnn_probe import MLPProbeModel, _make_activation

    output_dir.mkdir(parents=True, exist_ok=True)

    # Align indices
    common = concat_emb.index.intersection(unet_ms_emb.index)
    logger.info(f"Clustering probe: {len(common):,} common hexagons")

    concat_aligned = concat_emb.loc[common]
    unet_aligned = unet_ms_emb.loc[common]

    # Cluster concat with KMeans k=10
    logger.info(f"Clustering concat 208D with KMeans k={k}...")
    scaler_c = StandardScaler()
    concat_scaled = scaler_c.fit_transform(concat_aligned.values)
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10_000, n_init=3)
    cluster_labels = kmeans.fit_predict(concat_scaled)

    # Train DNN classifier: UNet MS -> concat cluster labels
    # Simple 80/20 split (not spatial -- this is a cross-modal alignment test)
    n = len(common)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    X = unet_aligned.values.astype(np.float32)
    y = cluster_labels

    scaler_u = StandardScaler()
    X_train = scaler_u.fit_transform(X[train_idx])
    X_test = scaler_u.transform(X[test_idx])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPProbeModel(
        input_dim=X.shape[1],
        hidden_dim=128,
        num_layers=3,
        use_layer_norm=True,
        output_dim=k,
        activation=_make_activation("silu"),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    x_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.long, device=device)
    x_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    import copy
    best_state = None
    best_loss = float("inf")

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(x_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_out = model(x_test_t)
            test_loss = criterion(test_out, torch.tensor(y[test_idx], dtype=torch.long, device=device)).item()
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.cpu().eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test, dtype=torch.float32))
        test_pred = test_logits.argmax(dim=1).numpy()

    acc = accuracy_score(y[test_idx], test_pred)
    f1_macro = f1_score(y[test_idx], test_pred, average="macro")
    f1_weighted = f1_score(y[test_idx], test_pred, average="weighted")

    logger.info(f"Clustering probe: Acc={acc:.3f}, F1-macro={f1_macro:.3f}, F1-weighted={f1_weighted:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = {"Accuracy": acc, "F1 (macro)": f1_macro, "F1 (weighted)": f1_weighted}
    bars = ax.bar(metrics.keys(), metrics.values(), color=["steelblue", "coral", "seagreen"],
                  edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Clustering Probe: UNet MS 192D -> Concat k={k} Clusters\n"
        f"DNN classifier (80/20 split, {len(common):,} hexagons)",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    path = output_dir / "clustering_probe_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved clustering probe bar: {path}")

    # Save metrics
    with open(output_dir / "clustering_probe_metrics.json", "w") as f:
        json.dump({"k": k, "accuracy": acc, "f1_macro": f1_macro,
                    "f1_weighted": f1_weighted, "n_common": len(common)}, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Concat vs UNet MS probe comparison")
    parser.add_argument("--skip-spatial", action="store_true",
                        help="Skip spatial prediction/residual maps")
    parser.add_argument("--skip-clustering-probe", action="store_true",
                        help="Skip clustering probe")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Subset of targets for spatial maps (default: all 6)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()

    print("=" * 70)
    print("Concat 208D vs UNet Multiscale-Concat 192D -- DNN Probe Comparison")
    print("=" * 70)

    # 1. Load embeddings
    print("\n[1/6] Loading concat 208D...")
    concat_emb = pd.read_parquet(project_root / CONCAT_PATH)
    print(f"  Concat: {concat_emb.shape}")

    print("\n[2/6] Building UNet multiscale-concat 192D...")
    unet_ms_emb = build_unet_multiscale()
    print(f"  UNet MS: {unet_ms_emb.shape}")

    # 2. Run probes
    concat_out = OUTPUT_BASE / "concat_208D"
    unet_out = OUTPUT_BASE / "unet_ms_192D"

    print("\n[3/6] Running DNN probe on Concat 208D...")
    t1 = time.time()
    concat_results = run_dnn_probe(concat_emb, "concat_208D", concat_out)
    print(f"  Concat probe done in {_format_duration(time.time() - t1)}")

    print("\n[4/6] Running DNN probe on UNet Multiscale 192D...")
    t2 = time.time()
    unet_results = run_dnn_probe(unet_ms_emb, "unet_ms_192D", unet_out)
    print(f"  UNet MS probe done in {_format_duration(time.time() - t2)}")

    # 3. Comparison outputs
    print("\n[5/6] Generating comparison outputs...")
    save_r2_table(concat_results, unet_results, OUTPUT_BASE)
    plot_r2_comparison(concat_results, unet_results, OUTPUT_BASE)

    if not args.skip_spatial:
        spatial_targets = args.targets or TARGET_COLS
        plot_spatial_predictions_and_residuals(
            concat_results, unet_results, OUTPUT_BASE, targets=spatial_targets,
        )
        plot_spatial_improvement(concat_results, unet_results, OUTPUT_BASE)

    # 4. Clustering probe
    if not args.skip_clustering_probe:
        print("\n[6/6] Running clustering probe...")
        run_clustering_probe(concat_emb, unet_ms_emb, CLUSTER_OUTPUT, k=10)

    total = time.time() - t0
    print(f"\nAll done in {_format_duration(total)}")
    print(f"Outputs: {OUTPUT_BASE}")

    # Print R2 summary table
    print("\n" + "=" * 70)
    print("R-squared Summary:")
    print(f"{'Target':<20} {'Concat 208D':>12} {'UNet MS 192D':>14} {'Delta':>8}")
    print("-" * 55)
    for t in TARGET_COLS:
        c_r2 = concat_results["results"][t].overall_r2
        u_r2 = unet_results["results"][t].overall_r2
        d = u_r2 - c_r2
        print(f"{TARGET_NAMES.get(t, t):<20} {c_r2:>12.4f} {u_r2:>14.4f} {d:>+8.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
