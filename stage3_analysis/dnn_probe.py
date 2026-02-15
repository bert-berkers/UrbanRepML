#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe: MLP Probe for Liveability Prediction

Trains a multi-layer perceptron (MLP) probe to test whether non-linear
transformations of AlphaEarth embeddings improve liveability prediction
over the existing linear probe.

Key differences from the linear probe:
    - Uses a multi-layer MLP with residual connections, LayerNorm, and GELU
    - Early stopping with cosine annealing LR schedule
    - Each hexagon is an independent sample (no graph structure)

Produces TargetResult-compatible output so existing LinearProbeVisualizer works.

Usage:
    python -m stage3_analysis.dnn_probe --study-area netherlands
    python -m stage3_analysis.dnn_probe --study-area netherlands --hidden-dim 256 --num-layers 3
    python -m stage3_analysis.dnn_probe --study-area netherlands --compare data/study_areas/netherlands/analysis/linear_probe
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from spatialkfold.blocks import spatial_blocks

from stage3_analysis.linear_probe import (
    TARGET_COLS,
    TARGET_NAMES,
    FoldMetrics,
    TargetResult,
)
from utils import StudyAreaPaths
from utils.paths import write_run_info

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DNNProbeConfig:
    """Configuration for DNN MLP probe regression."""

    # Study area
    study_area: str = "netherlands"
    year: int = 2022
    h3_resolution: int = 10
    target_cols: List[str] = field(default_factory=lambda: list(TARGET_COLS))

    # Spatial block CV (same defaults as LinearProbeConfig)
    n_folds: int = 5
    block_width: int = 10_000   # meters (10km blocks for better fold mixing)
    block_height: int = 10_000  # meters
    random_state: int = 42

    # MLP architecture
    hidden_dim: int = 32            # First hidden dimension (halves per layer)
    num_layers: int = 3             # Number of narrowing layers
    dropout: float = 0.0            # Disabled -- replaced by progressive batch size
    use_layer_norm: bool = True     # LayerNorm after each layer

    # Training
    learning_rate: float = 5e-4
    initial_batch_size: int = 4096          # Starting mini-batch size
    batch_size_ramp_epochs: int = 10        # Double batch size every N epochs
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 30              # Early stopping patience
    min_delta: float = 1e-5         # Minimum improvement for early stopping

    # Data paths (relative to project root)
    embeddings_path: Optional[str] = None
    target_path: Optional[str] = None
    output_dir: Optional[str] = None

    # Run-level provenance: if non-empty, a dated run directory is created
    # under stage3_analysis/dnn_probe/{run_id}/ instead of writing to the
    # flat analysis directory.  When empty (default), behaviour is unchanged.
    run_descriptor: str = "default"

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    def __post_init__(self):
        paths = StudyAreaPaths(self.study_area)
        self.run_id: Optional[str] = None
        if self.embeddings_path is None:
            self.embeddings_path = str(
                paths.embedding_file("alphaearth", self.h3_resolution, self.year)
            )
        if self.target_path is None:
            self.target_path = str(
                paths.target_file("leefbaarometer", self.h3_resolution, self.year)
            )
        if self.output_dir is None:
            if self.run_descriptor:
                self.run_id = paths.create_run_id(self.run_descriptor)
                self.output_dir = str(paths.stage3_run("dnn_probe", self.run_id))
            else:
                self.output_dir = str(paths.stage3("dnn_probe"))
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------

class MLPProbeModel(nn.Module):
    """
    Multi-layer perceptron probe for per-hexagon regression.

    Architecture: progressively narrowing funnel from input_dim down to 1.
    Each layer halves the dimension: input_dim -> hidden_dim -> hidden_dim//2
    -> ... -> linear head -> 1.

    Each hexagon is treated as an independent sample -- no message passing
    or graph structure is used.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.num_layers = num_layers

        # Build progressively narrowing dimensions:
        # input_dim -> hidden_dim -> hidden_dim//2 -> ... -> head -> 1
        dims = [input_dim, hidden_dim]
        for _ in range(num_layers - 1):
            dims.append(max(dims[-1] // 2, 8))  # floor at 8

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.norms.append(
                nn.LayerNorm(dims[i + 1]) if use_layer_norm else nn.Identity()
            )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Regression head from final hidden dim
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Assumes input features are NaN-free. If NaN values are present
        (e.g. due to missing embeddings that slipped past upstream
        validation), they are replaced with 0.0 as a safety guard and
        a warning is logged.

        Args:
            x: Feature matrix [N, input_dim].

        Returns:
            Per-sample predictions [N, 1].
        """
        # Safety guard: replace NaN inputs to avoid silent propagation
        if torch.isnan(x).any():
            logger.warning("NaN detected in input features, replacing with 0.0")
            x = torch.nan_to_num(x, nan=0.0)

        # Progressive narrowing through layers
        h = x
        for i in range(len(self.layers)):
            h = self.layers[i](h)
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)

        # Regression head
        return self.head(h)


# ---------------------------------------------------------------------------
# Main regressor
# ---------------------------------------------------------------------------

class DNNProbeRegressor:
    """
    MLP probe with spatial block CV.

    Evaluates whether non-linear transformations of AlphaEarth embeddings
    improve liveability prediction over linear probes. Each hexagon is an
    independent sample -- no graph structure is used.

    Visualization compatibility:
        Produces TargetResult objects compatible with LinearProbeVisualizer,
        but because the MLP has no linear coefficients (``coefficients`` is
        a zero vector), only a subset of visualizer methods produce
        meaningful output.

        Works correctly:
            - plot_scatter_predicted_vs_actual (uses oof_predictions)
            - plot_spatial_residuals (uses oof_predictions + region_ids)
            - plot_fold_metrics (uses fold_metrics)
            - plot_metrics_comparison (uses overall_r2)

        Not meaningful (coefficients are zeros):
            - plot_coefficient_bars / plot_coefficient_bars_faceted
            - plot_coefficient_heatmap
            - plot_rgb_top3_map
            - plot_cross_target_correlation
    """

    def __init__(self, config: DNNProbeConfig, project_root: Optional[Path] = None):
        self.config = config
        self.project_root = project_root or Path(__file__).parent.parent
        self.results: Dict[str, TargetResult] = {}
        self.best_hparams: Dict[str, Dict[str, Any]] = {}
        self.training_curves: Dict[str, Dict[int, List[float]]] = {}
        self.data_gdf: Optional[gpd.GeoDataFrame] = None
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Data loading (mirrors LinearProbeRegressor)
    # ------------------------------------------------------------------

    def load_and_join_data(self) -> gpd.GeoDataFrame:
        """
        Load embeddings and target data, inner join on region_id / h3_index.

        Returns:
            GeoDataFrame with embeddings + target columns, indexed by region_id.
        """
        emb_path = self.project_root / self.config.embeddings_path
        target_path = self.project_root / self.config.target_path

        logger.info(f"Loading embeddings from {emb_path}")
        emb_df = pd.read_parquet(emb_path)

        # Normalize index: embeddings may use h3_index column
        if "h3_index" in emb_df.columns and emb_df.index.name != "region_id":
            emb_df = emb_df.set_index("h3_index")
            emb_df.index.name = "region_id"

        # Identify embedding feature columns (A00, A01, ..., A63)
        self.feature_names = [
            c for c in emb_df.columns
            if (c.startswith("A") and c[1:].isdigit()) or c.startswith("emb_")
        ]
        if not self.feature_names:
            # Fallback: numeric columns excluding metadata
            exclude = {"pixel_count", "tile_count", "geometry"}
            self.feature_names = [
                c for c in emb_df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(emb_df[c])
            ]

        logger.info(
            f"  Embedding features: {len(self.feature_names)} "
            f"({self.feature_names[0]}..{self.feature_names[-1]})"
        )
        logger.info(f"  Embedding rows: {len(emb_df):,}")

        emb_df = emb_df[self.feature_names]

        # Load target
        logger.info(f"Loading target from {target_path}")
        target_df = pd.read_parquet(target_path)
        if target_df.index.name != "region_id" and "region_id" in target_df.columns:
            target_df = target_df.set_index("region_id")
        logger.info(f"  Target rows: {len(target_df):,}")

        # Inner join
        joined = emb_df.join(target_df[self.config.target_cols], how="inner")
        logger.info(
            f"  Inner join: {len(joined):,} hexagons with both embeddings and targets"
        )

        # Drop rows with any NaN
        before = len(joined)
        joined = joined.dropna(subset=self.feature_names + self.config.target_cols)
        logger.info(f"  After dropna: {len(joined):,} (dropped {before - len(joined):,})")

        # Create geometry from H3 centroids for spatial blocking (SRAI-compliant)
        logger.info("  Creating geometry from H3 cell centroids for spatial blocking...")
        from srai.h3 import h3_to_geoseries
        hex_geom = h3_to_geoseries(joined.index)
        centroids = hex_geom.centroid
        centroids.index = joined.index
        joined = gpd.GeoDataFrame(joined, geometry=centroids, crs="EPSG:4326")

        self.data_gdf = joined
        return joined

    # ------------------------------------------------------------------
    # Spatial blocking (mirrors LinearProbeRegressor)
    # ------------------------------------------------------------------

    def create_spatial_blocks(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Create spatial block fold assignments using spatialkfold.

        Projects to EPSG:28992 (RD New) for metric block sizes.

        Returns:
            Array of fold assignments (1 to n_folds) per row, aligned to gdf index.
        """
        logger.info(
            f"Creating spatial blocks: {self.config.n_folds} folds, "
            f"{self.config.block_width}m x {self.config.block_height}m"
        )

        gdf_proj = gdf.to_crs(epsg=28992)

        blocks_gdf = spatial_blocks(
            gdf=gdf_proj,
            nfolds=self.config.n_folds,
            width=self.config.block_width,
            height=self.config.block_height,
            method="random",
            orientation="tb-lr",
            grid_type="rect",
            random_state=self.config.random_state,
        )

        logger.info(
            f"  Created {len(blocks_gdf)} spatial blocks "
            f"across {self.config.n_folds} folds"
        )

        points_with_folds = gpd.sjoin(
            gdf_proj[["geometry"]],
            blocks_gdf[["geometry", "folds"]],
            how="left",
            predicate="within",
        )

        # Handle points outside all blocks
        missing_mask = points_with_folds["folds"].isna()
        if missing_mask.any():
            n_missing = missing_mask.sum()
            logger.warning(
                f"  {n_missing} points outside all blocks, assigning to fold 1"
            )
            points_with_folds.loc[missing_mask, "folds"] = 1

        # Handle duplicates from sjoin (point on block boundary)
        points_with_folds = points_with_folds[
            ~points_with_folds.index.duplicated(keep="first")
        ]

        folds = points_with_folds.loc[gdf.index, "folds"].values.astype(int)

        unique_folds = np.unique(folds)
        fold_counts = {int(f): int((folds == f).sum()) for f in unique_folds}
        logger.info(f"  Fold sizes: {fold_counts}")

        return folds

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_one_fold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        hparams: Dict[str, Any],
    ) -> Tuple[MLPProbeModel, float, StandardScaler, StandardScaler, List[float]]:
        """
        Train MLP for one fold with early stopping.

        Per-fold feature AND target standardization: fit scalers on training
        samples, transform both train and validation independently. The model
        trains on standardized targets; predictions are inverse-transformed
        back to the original scale before being returned.

        Uses progressive mini-batch training: the batch size starts at
        ``initial_batch_size`` and doubles every ``batch_size_ramp_epochs``
        epochs, providing implicit regularization through gradient noise that
        decreases as training progresses.

        Args:
            X: Feature matrix [N, D] (numpy, raw -- standardized here).
            y: Target vector [N] (numpy).
            train_mask: Boolean array [N] for training samples.
            val_mask: Boolean array [N] for validation samples.
            hparams: Dict with keys: hidden_dim, num_layers, dropout,
                     use_layer_norm, learning_rate, weight_decay,
                     initial_batch_size, batch_size_ramp_epochs.

        Returns:
            (trained_model on CPU, best_val_loss, fitted_feature_scaler,
             fitted_target_scaler, val_loss_history)
        """
        device = torch.device(self.config.device)

        # Per-fold feature standardization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask]).astype(np.float32)
        X_val = scaler.transform(X[val_mask]).astype(np.float32)

        # Per-fold target standardization
        y_scaler = StandardScaler()
        y_train_raw = y[train_mask].reshape(-1, 1)
        y_val_raw = y[val_mask].reshape(-1, 1)
        y_train_std = y_scaler.fit_transform(y_train_raw).ravel().astype(np.float32)
        y_val_std = y_scaler.transform(y_val_raw).ravel().astype(np.float32)

        # Build tensors
        x_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train_std, dtype=torch.float32, device=device)
        x_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val_std, dtype=torch.float32, device=device)

        # Create model
        model = MLPProbeModel(
            input_dim=X.shape[1],
            hidden_dim=hparams["hidden_dim"],
            num_layers=hparams["num_layers"],
            dropout=hparams["dropout"],
            use_layer_norm=hparams.get("use_layer_norm", self.config.use_layer_norm),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_epochs
        )

        criterion = nn.MSELoss()

        # Progressive batch size parameters
        batch_size = hparams.get("initial_batch_size", 4096)
        batch_ramp = hparams.get("batch_size_ramp_epochs", 10)
        n_train = x_train_t.shape[0]

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[Dict] = None
        val_loss_history: List[float] = []

        for epoch in range(self.config.max_epochs):
            # Progressive batch size: double every batch_ramp epochs
            current_bs = min(batch_size * (2 ** (epoch // batch_ramp)), n_train)

            # -- Mini-batch training --
            model.train()
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, current_bs):
                idx = perm[start : start + current_bs]
                optimizer.zero_grad()
                out = model(x_train_t[idx])
                loss = criterion(out.squeeze(-1), y_train_t[idx])
                loss.backward()
                optimizer.step()

            scheduler.step()

            # -- Validate (full-batch inference) --
            model.eval()
            with torch.no_grad():
                val_out = model(x_val_t)
                val_loss = criterion(
                    val_out.squeeze(-1), y_val_t
                ).item()

            val_loss_history.append(val_loss)

            # -- Early stopping --
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model = model.cpu()
        return model, best_val_loss, scaler, y_scaler, val_loss_history

    # ------------------------------------------------------------------
    # Per-target pipeline
    # ------------------------------------------------------------------

    def run_for_target(
        self,
        target_col: str,
        X: np.ndarray,
        y_all: pd.DataFrame,
        folds: np.ndarray,
        region_ids: np.ndarray,
    ) -> TargetResult:
        """
        Run spatial CV pipeline for one target variable.

        Uses config defaults for all hyperparameters.

        Args:
            target_col: Target column name (e.g. 'lbm').
            X: Feature matrix [N, D].
            y_all: DataFrame with all target columns.
            folds: Spatial fold assignments.
            region_ids: Region IDs for each row.

        Returns:
            TargetResult with metrics and predictions.
        """
        y = y_all[target_col].values
        unique_folds = np.unique(folds)
        target_name = TARGET_NAMES.get(target_col, target_col)

        logger.info(f"\n--- Target: {target_col} ({target_name}) ---")
        logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

        # Use config defaults
        hparams = {
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "learning_rate": self.config.learning_rate,
            "dropout": self.config.dropout,
            "weight_decay": self.config.weight_decay,
            "use_layer_norm": self.config.use_layer_norm,
            "initial_batch_size": self.config.initial_batch_size,
            "batch_size_ramp_epochs": self.config.batch_size_ramp_epochs,
        }
        logger.info(
            f"  Using config defaults: "
            f"hidden={hparams['hidden_dim']}, "
            f"layers={hparams['num_layers']}, "
            f"lr={hparams['learning_rate']}"
        )

        self.best_hparams[target_col] = hparams

        # ----- Spatial CV evaluation -----
        logger.info("  Evaluating with spatial CV...")
        oof_predictions = np.full(len(y), np.nan)
        fold_metrics_list: List[FoldMetrics] = []
        training_curves: Dict[int, List[float]] = {}

        for fold_id in unique_folds:
            val_mask = folds == fold_id
            train_mask = ~val_mask

            model, best_val_loss, fold_scaler, fold_y_scaler, val_history = (
                self._train_one_fold(X, y, train_mask, val_mask, hparams)
            )
            training_curves[int(fold_id)] = val_history

            # Get predictions on validation set (in standardized space)
            device = torch.device(self.config.device)

            X_val_scaled = fold_scaler.transform(X[val_mask]).astype(np.float32)
            x_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)

            model = model.to(device)
            model.eval()
            with torch.no_grad():
                val_pred_std = model(x_val_t).squeeze(-1).cpu().numpy()

            # Inverse-transform predictions back to original target scale
            val_pred = fold_y_scaler.inverse_transform(
                val_pred_std.reshape(-1, 1)
            ).ravel()

            oof_predictions[val_mask] = val_pred

            y_val = y[val_mask]
            rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            mae = float(mean_absolute_error(y_val, val_pred))
            r2 = float(r2_score(y_val, val_pred))

            fold_metrics_list.append(
                FoldMetrics(
                    fold=int(fold_id),
                    rmse=rmse,
                    mae=mae,
                    r2=r2,
                    n_train=int(train_mask.sum()),
                    n_test=int(val_mask.sum()),
                )
            )

            logger.info(
                f"    Fold {fold_id}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f} "
                f"(train={train_mask.sum():,}, test={val_mask.sum():,}, "
                f"epochs={len(val_history)})"
            )

            # Free GPU memory
            model = model.cpu()
            del x_val_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Persist training curves for save_results()
        self.training_curves[target_col] = training_curves

        # Overall metrics from OOF predictions
        valid_mask = ~np.isnan(oof_predictions)
        overall_r2 = float(r2_score(y[valid_mask], oof_predictions[valid_mask]))
        overall_rmse = float(
            np.sqrt(mean_squared_error(y[valid_mask], oof_predictions[valid_mask]))
        )
        overall_mae = float(
            mean_absolute_error(y[valid_mask], oof_predictions[valid_mask])
        )

        logger.info(
            f"  Overall OOF: R2={overall_r2:.4f}, "
            f"RMSE={overall_rmse:.4f}, MAE={overall_mae:.4f}"
        )

        result = TargetResult(
            target=target_col,
            target_name=target_name,
            best_alpha=0.0,       # Not applicable for MLP
            best_l1_ratio=0.0,    # Not applicable for MLP
            fold_metrics=fold_metrics_list,
            overall_r2=overall_r2,
            overall_rmse=overall_rmse,
            overall_mae=overall_mae,
            coefficients=np.zeros(len(self.feature_names)),  # MLP has no linear coefficients
            intercept=0.0,
            feature_names=self.feature_names,
            oof_predictions=oof_predictions,
            actual_values=y,
            region_ids=region_ids,
        )

        self.results[target_col] = result
        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, TargetResult]:
        """
        Run the full DNN probe pipeline for all target variables.

        Returns:
            Dictionary mapping target column to TargetResult.
        """
        logger.info("=== DNN Probe Regression (MLP) ===")
        logger.info(f"Study area: {self.config.study_area}")
        logger.info(f"Year: {self.config.year}")
        logger.info(f"Device: {self.config.device}")
        logger.info(
            f"Architecture: MLP, "
            f"hidden={self.config.hidden_dim}, layers={self.config.num_layers}"
        )

        # Load and join data
        gdf = self.load_and_join_data()

        # Create spatial blocks
        region_ids = gdf.index.values
        folds = self.create_spatial_blocks(gdf)

        # Extract feature matrix
        X = gdf[self.feature_names].values
        y_all = gdf[self.config.target_cols]

        logger.info(f"\nFeature matrix: {X.shape}")
        logger.info(f"Targets: {self.config.target_cols}")

        # Run for each target
        for target_col in self.config.target_cols:
            self.run_for_target(
                target_col, X, y_all, folds, region_ids,
            )

        # Summary
        logger.info("\n=== Summary ===")
        for target_col, result in self.results.items():
            logger.info(
                f"  {target_col} ({result.target_name}): "
                f"R2={result.overall_r2:.4f}, RMSE={result.overall_rmse:.4f}"
            )

        return self.results

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save results in the same format as the linear probe.

        Produces:
            metrics_summary.csv, predictions_{target}.parquet, config.json,
            and optionally training_curves/{target}_fold{k}.json.
        """
        out_dir = output_dir or (self.project_root / self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ----- Metrics summary -----
        metrics_rows = []
        for target_col, result in self.results.items():
            row: Dict[str, Any] = {
                "target": target_col,
                "target_name": result.target_name,
                "best_alpha": result.best_alpha,
                "best_l1_ratio": result.best_l1_ratio,
                "overall_r2": result.overall_r2,
                "overall_rmse": result.overall_rmse,
                "overall_mae": result.overall_mae,
                "n_features": len(result.feature_names),
            }
            if result.fold_metrics:
                for fm in result.fold_metrics:
                    row[f"fold{fm.fold}_r2"] = fm.r2
                    row[f"fold{fm.fold}_rmse"] = fm.rmse
            metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
        logger.info(f"Saved metrics summary to {out_dir / 'metrics_summary.csv'}")

        # ----- Predictions per target -----
        for target_col, result in self.results.items():
            pred_df = pd.DataFrame(
                {
                    "region_id": result.region_ids,
                    "actual": result.actual_values,
                    "predicted": result.oof_predictions,
                    "residual": result.actual_values - result.oof_predictions,
                }
            ).set_index("region_id")
            pred_path = out_dir / f"predictions_{target_col}.parquet"
            pred_df.to_parquet(pred_path)

        logger.info(f"Saved predictions to {out_dir}")

        # ----- Config + best hyperparameters -----
        config_dict: Dict[str, Any] = {
            "study_area": self.config.study_area,
            "year": self.config.year,
            "h3_resolution": self.config.h3_resolution,
            "n_folds": self.config.n_folds,
            "block_width": self.config.block_width,
            "block_height": self.config.block_height,
            "random_state": self.config.random_state,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "device": self.config.device,
            "max_epochs": self.config.max_epochs,
            "patience": self.config.patience,
            "best_hyperparameters": {},
        }
        for target_col, hps in self.best_hparams.items():
            # Convert numpy types for JSON serialization
            config_dict["best_hyperparameters"][target_col] = {
                k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in hps.items()
            }

        with open(out_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved config to {out_dir / 'config.json'}")

        # Write run-level provenance when using a run directory
        if self.config.run_id is not None:
            # Auto-detect upstream run
            _paths = StudyAreaPaths(self.config.study_area)
            _latest = _paths.latest_run(_paths.stage1("alphaearth"))
            _upstream_label = _latest.name if _latest else "flat"
            write_run_info(
                out_dir,
                stage="stage3",
                study_area=self.config.study_area,
                config=config_dict,
                upstream_runs={"stage1/alphaearth": _upstream_label},
            )
            logger.info(f"Saved run_info.json to {out_dir / 'run_info.json'}")

        # ----- Training curves per target/fold -----
        if self.training_curves:
            curves_dir = out_dir / "training_curves"
            curves_dir.mkdir(parents=True, exist_ok=True)
            for target_col, fold_curves in self.training_curves.items():
                for fold_id, curve in fold_curves.items():
                    curve_path = curves_dir / f"{target_col}_fold{fold_id}.json"
                    with open(curve_path, "w") as f:
                        json.dump(
                            {"target": target_col, "fold": fold_id, "val_loss": curve},
                            f,
                        )
            logger.info(
                f"Saved training curves to {curves_dir} "
                f"({sum(len(fc) for fc in self.training_curves.values())} files)"
            )

        return out_dir

    # ------------------------------------------------------------------
    # Comparison with linear probe
    # ------------------------------------------------------------------

    def compare_with_linear(
        self,
        linear_results: Dict[str, TargetResult],
    ) -> pd.DataFrame:
        """
        Produce comparison table: linear vs DNN R2/RMSE/MAE per target.

        Args:
            linear_results: Dictionary of TargetResult from linear probe.

        Returns:
            DataFrame with columns: target, target_name,
            linear_r2, linear_rmse, linear_mae,
            dnn_r2, dnn_rmse, dnn_mae,
            r2_delta, rmse_delta.
        """
        rows = []
        for target_col in self.config.target_cols:
            lr = linear_results.get(target_col)
            dr = self.results.get(target_col)
            if lr is None or dr is None:
                continue

            rows.append(
                {
                    "target": target_col,
                    "target_name": TARGET_NAMES.get(target_col, target_col),
                    "linear_r2": lr.overall_r2,
                    "linear_rmse": lr.overall_rmse,
                    "linear_mae": lr.overall_mae,
                    "dnn_r2": dr.overall_r2,
                    "dnn_rmse": dr.overall_rmse,
                    "dnn_mae": dr.overall_mae,
                    "r2_delta": dr.overall_r2 - lr.overall_r2,
                    "rmse_delta": dr.overall_rmse - lr.overall_rmse,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def load_linear_results(linear_dir: Path) -> Dict[str, TargetResult]:
        """
        Load linear probe results from disk for comparison.

        Reads metrics_summary.csv and predictions_{target}.parquet files
        from the linear probe output directory and reconstructs a minimal
        Dict[str, TargetResult] sufficient for compare_with_linear().

        Args:
            linear_dir: Path to the linear probe output directory.

        Returns:
            Dictionary mapping target column to TargetResult (metrics only,
            coefficients/predictions may be empty).
        """
        metrics_path = linear_dir / "metrics_summary.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Linear probe metrics not found: {metrics_path}"
            )

        metrics_df = pd.read_csv(metrics_path)
        results: Dict[str, TargetResult] = {}

        for _, row in metrics_df.iterrows():
            target_col = row["target"]

            # Try to load predictions for full TargetResult
            pred_path = linear_dir / f"predictions_{target_col}.parquet"
            if pred_path.exists():
                pred_df = pd.read_parquet(pred_path)
                oof_preds = pred_df["predicted"].values
                actuals = pred_df["actual"].values
                region_ids = pred_df.index.values
            else:
                oof_preds = np.array([])
                actuals = np.array([])
                region_ids = np.array([])

            results[target_col] = TargetResult(
                target=target_col,
                target_name=row.get("target_name", target_col),
                best_alpha=row.get("best_alpha", 0.0),
                best_l1_ratio=row.get("best_l1_ratio", 0.0),
                fold_metrics=[],
                overall_r2=row["overall_r2"],
                overall_rmse=row["overall_rmse"],
                overall_mae=row["overall_mae"],
                coefficients=np.array([]),
                intercept=0.0,
                feature_names=[],
                oof_predictions=oof_preds,
                actual_values=actuals,
                region_ids=region_ids,
            )

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Run DNN probe regression with CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DNN Probe: MLP-based AlphaEarth -> Leefbaarometer"
    )
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Spatial CV folds (default: 5)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
        help="Max training epochs (default: 300)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension (default: 128)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of MLP layers (default: 2)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=10000,
        help="Spatial block size in meters (default: 10000)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device (default: auto)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Path to linear probe results dir for comparison table",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate DNN probe visualizations after saving results",
    )
    parser.add_argument(
        "--quick-viz",
        action="store_true",
        help="Generate only fast plots (skip spatial maps)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = DNNProbeConfig(
        study_area=args.study_area,
        n_folds=args.n_folds,
        max_epochs=args.max_epochs,
        block_width=args.block_size,
        block_height=args.block_size,
        device=args.device,
    )

    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        config.num_layers = args.num_layers

    regressor = DNNProbeRegressor(config)
    regressor.run()
    out_dir = regressor.save_results()

    # Comparison with linear probe
    linear_results = None
    if args.compare:
        linear_dir = Path(args.compare)
        logger.info(f"\nLoading linear probe results from {linear_dir}...")
        linear_results = DNNProbeRegressor.load_linear_results(linear_dir)
        comparison_df = regressor.compare_with_linear(linear_results)
        comparison_path = out_dir / "comparison_linear_vs_dnn.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nComparison saved to {comparison_path}")
        logger.info(f"\n{comparison_df.to_string(index=False)}")

    # Visualization
    if args.visualize:
        from stage3_analysis.dnn_probe_viz import DNNProbeVisualizer

        viz_dir = out_dir / "plots"
        logger.info(f"\nGenerating DNN probe visualizations to {viz_dir}...")
        viz = DNNProbeVisualizer(
            results=regressor.results,
            output_dir=viz_dir,
            training_curves=regressor.training_curves,
        )
        plot_paths = viz.plot_all(
            linear_results=linear_results,
            skip_spatial=args.quick_viz,
        )
        logger.info(f"Generated {len(plot_paths)} plots to {viz_dir}")


if __name__ == "__main__":
    main()
