#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Probe: Graph Neural Network Probe for Liveability Prediction

Trains a shallow GCN/GAT probe on the H3 adjacency graph to test whether
spatially-aware non-linear transformations of AlphaEarth embeddings improve
liveability prediction over the existing linear probe.

Key differences from the linear probe:
    - Operates on the H3 adjacency graph (message passing between hexagon neighbours)
    - Uses GCN or GAT convolutional layers (2-3 layers, local receptive field)
    - Transductive node masking for spatial block CV (full graph always intact)
    - Optuna HPO over GNN architecture and training hyperparameters

Produces TargetResult-compatible output so existing LinearProbeVisualizer works.

Usage:
    python -m stage3_analysis.dnn_probe --study-area netherlands
    python -m stage3_analysis.dnn_probe --study-area netherlands --skip-hpo
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
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from spatialkfold.blocks import spatial_blocks
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv

from stage3_analysis.linear_probe import (
    TARGET_COLS,
    TARGET_NAMES,
    FoldMetrics,
    TargetResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DNNProbeConfig:
    """Configuration for DNN graph probe regression."""

    # Study area
    study_area: str = "netherlands"
    year: int = 2022
    h3_resolution: int = 10
    target_cols: List[str] = field(default_factory=lambda: list(TARGET_COLS))

    # Spatial block CV (same defaults as LinearProbeConfig)
    n_folds: int = 5
    block_width: int = 25_000   # meters
    block_height: int = 25_000  # meters
    random_state: int = 42

    # GNN architecture
    conv_type: str = "gcn"          # "gcn" or "gat"
    hidden_dim: int = 128           # Hidden dimension
    num_layers: int = 2             # Number of GCN/GAT layers (2 or 3)
    num_heads: int = 4              # GAT attention heads (ignored for GCN)
    dropout: float = 0.1
    use_residual: bool = True       # Residual connections
    use_layer_norm: bool = True     # LayerNorm after each conv

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 30              # Early stopping patience
    min_delta: float = 1e-5         # Minimum improvement for early stopping

    # Optuna HPO
    n_trials: int = 30

    # Optuna search ranges
    hidden_dim_choices: List[int] = field(default_factory=lambda: [64, 128, 256])
    lr_low: float = 1e-4
    lr_high: float = 1e-2
    dropout_low: float = 0.0
    dropout_high: float = 0.3
    weight_decay_low: float = 1e-6
    weight_decay_high: float = 1e-3
    num_layers_choices: List[int] = field(default_factory=lambda: [2, 3])

    # Data paths (relative to project root)
    embeddings_path: Optional[str] = None
    target_path: Optional[str] = None
    output_dir: Optional[str] = None

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    def __post_init__(self):
        if self.embeddings_path is None:
            self.embeddings_path = (
                f"data/study_areas/{self.study_area}/embeddings/alphaearth/"
                f"{self.study_area}_res{self.h3_resolution}_{self.year}.parquet"
            )
        if self.target_path is None:
            self.target_path = (
                f"data/study_areas/{self.study_area}/target/leefbaarometer/"
                f"leefbaarometer_h3res{self.h3_resolution}_{self.year}.parquet"
            )
        if self.output_dir is None:
            self.output_dir = (
                f"data/study_areas/{self.study_area}/analysis/dnn_probe"
            )
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# GNN Model
# ---------------------------------------------------------------------------

class GNNProbeModel(nn.Module):
    """
    Shallow GCN/GAT probe for per-hexagon regression.

    Not a full U-Net -- just 2-3 message-passing layers with residual
    connections on a single-resolution H3 adjacency graph.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        conv_type: str,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # Convolutional layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gat":
                conv = GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                )
            elif conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type!r}. Use 'gcn' or 'gat'.")
            self.convs.append(conv)
            self.norms.append(
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            )

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Regression head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass.

        Assumes input features are NaN-free. If NaN values are present
        (e.g. due to missing embeddings that slipped past upstream
        validation), they are replaced with 0.0 as a safety guard and
        a warning is logged.

        Args:
            x: Node features [N, input_dim].
            edge_index: Edge indices [2, E].

        Returns:
            Per-node predictions [N, 1].
        """
        # Safety guard: replace NaN inputs to avoid silent propagation
        if torch.isnan(x).any():
            logger.warning("NaN detected in input features, replacing with 0.0")
            x = torch.nan_to_num(x, nan=0.0)

        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.activation(h)

        # Message-passing layers with residual connections
        for i in range(self.num_layers):
            h_in = h
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            if self.use_residual:
                h = h + h_in

        # Regression head
        return self.head(h)


# ---------------------------------------------------------------------------
# Main regressor
# ---------------------------------------------------------------------------

class DNNProbeRegressor:
    """
    GNN probe with spatial block CV and Optuna optimization.

    Evaluates whether spatially-aware (graph-based) transformations of
    AlphaEarth embeddings improve liveability prediction over linear probes.

    Visualization compatibility:
        Produces TargetResult objects compatible with LinearProbeVisualizer,
        but because the GNN has no linear coefficients (``coefficients`` is
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
            c for c in emb_df.columns if c.startswith("A") and c[1:].isdigit()
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
    # Graph construction
    # ------------------------------------------------------------------

    def build_h3_graph(self, region_ids: np.ndarray) -> Data:
        """
        Build H3 adjacency graph using SRAI H3Neighbourhood.

        Uses H3Neighbourhood.get_neighbours() to find the 1-ring (k=1)
        neighbours for each hexagon, then constructs a PyG Data object
        with bidirectional edge_index.

        Args:
            region_ids: Array of H3 hex ID strings.

        Returns:
            PyG Data with edge_index [2, E] and num_nodes.
        """
        import time

        from srai.neighbourhoods import H3Neighbourhood

        n = len(region_ids)
        logger.info(f"Building H3 adjacency graph for {n:,} hexagons...")

        neighbourhood = H3Neighbourhood()
        hex_to_idx: Dict[str, int] = {h: i for i, h in enumerate(region_ids)}

        src_list: List[int] = []
        dst_list: List[int] = []

        t0 = time.time()
        log_interval = max(1, n // 10)  # log progress ~10 times

        for i, hex_id in enumerate(region_ids):
            neighbours = neighbourhood.get_neighbours(hex_id)
            for nbr in neighbours:
                nbr_idx = hex_to_idx.get(nbr)
                if nbr_idx is not None:
                    src_list.append(i)
                    dst_list.append(nbr_idx)
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (n - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  Graph construction: {i + 1:,}/{n:,} hexagons "
                    f"({100 * (i + 1) / n:.0f}%) -- "
                    f"{rate:.0f} hex/s, ETA {eta:.0f}s"
                )

        elapsed = time.time() - t0
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        graph_data = Data(edge_index=edge_index, num_nodes=n)

        logger.info(
            f"  Graph built in {elapsed:.1f}s: "
            f"{n:,} nodes, {edge_index.shape[1]:,} edges, "
            f"avg degree {edge_index.shape[1] / n:.1f}"
        )

        return graph_data

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
        edge_index: torch.LongTensor,
        num_nodes: int,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        hparams: Dict[str, Any],
    ) -> Tuple[GNNProbeModel, float, StandardScaler, List[float]]:
        """
        Train GNN for one fold with early stopping.

        Per-fold feature standardization: fit scaler on train_mask nodes,
        transform all nodes. Train on full graph, compute loss only on
        train_mask. Early stopping based on val_mask loss.

        Args:
            X: Feature matrix [N, D] (numpy, raw -- standardized here).
            y: Target vector [N] (numpy).
            edge_index: Edge indices [2, E] (torch, on CPU).
            num_nodes: Number of nodes.
            train_mask: Boolean array [N] for training nodes.
            val_mask: Boolean array [N] for validation nodes.
            hparams: Dict with keys: hidden_dim, num_layers, conv_type,
                     num_heads, dropout, use_residual, use_layer_norm,
                     learning_rate, weight_decay.

        Returns:
            (trained_model on CPU, best_val_loss, fitted_scaler, val_loss_history)
        """
        device = torch.device(self.config.device)

        # Per-fold standardization
        scaler = StandardScaler()
        X_scaled = np.empty_like(X, dtype=np.float32)
        X_scaled[train_mask] = scaler.fit_transform(X[train_mask]).astype(np.float32)
        X_scaled[val_mask] = scaler.transform(X[val_mask]).astype(np.float32)
        # For nodes in neither mask (should not happen with binary folds, but be safe)
        other_mask = ~(train_mask | val_mask)
        if other_mask.any():
            X_scaled[other_mask] = scaler.transform(X[other_mask]).astype(np.float32)

        # Build PyG Data object for this fold
        data = Data(
            x=torch.tensor(X_scaled, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        data = data.to(device)

        train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
        val_mask_t = torch.tensor(val_mask, dtype=torch.bool, device=device)

        # Create model
        model = GNNProbeModel(
            input_dim=X.shape[1],
            hidden_dim=hparams["hidden_dim"],
            num_layers=hparams["num_layers"],
            conv_type=hparams["conv_type"],
            num_heads=hparams.get("num_heads", self.config.num_heads),
            dropout=hparams["dropout"],
            use_residual=hparams.get("use_residual", self.config.use_residual),
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

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: Optional[Dict] = None
        val_loss_history: List[float] = []

        for epoch in range(self.config.max_epochs):
            # -- Train --
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)  # [N, 1]
            loss = criterion(out[train_mask_t].squeeze(-1), data.y[train_mask_t])
            loss.backward()
            optimizer.step()
            scheduler.step()

            # -- Validate --
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                val_loss = criterion(
                    out[val_mask_t].squeeze(-1), data.y[val_mask_t]
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
        return model, best_val_loss, scaler, val_loss_history

    # ------------------------------------------------------------------
    # Optuna HPO
    # ------------------------------------------------------------------

    def _optuna_objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray,
        folds: np.ndarray,
        unique_folds: np.ndarray,
        edge_index: torch.LongTensor,
        num_nodes: int,
    ) -> float:
        """Optuna objective: minimize mean val RMSE across spatial folds."""
        hparams = {
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim", self.config.hidden_dim_choices
            ),
            "num_layers": trial.suggest_categorical(
                "num_layers", self.config.num_layers_choices
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.config.lr_low,
                self.config.lr_high,
                log=True,
            ),
            "dropout": trial.suggest_float(
                "dropout",
                self.config.dropout_low,
                self.config.dropout_high,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                self.config.weight_decay_low,
                self.config.weight_decay_high,
                log=True,
            ),
            "conv_type": trial.suggest_categorical("conv_type", ["gcn", "gat"]),
            "num_heads": self.config.num_heads,
            "use_residual": self.config.use_residual,
            "use_layer_norm": self.config.use_layer_norm,
        }

        fold_rmses: List[float] = []
        for fold_idx, fold_id in enumerate(unique_folds):
            val_mask = folds == fold_id
            train_mask = ~val_mask

            _, best_val_loss, _, _ = self._train_one_fold(
                X, y, edge_index, num_nodes, train_mask, val_mask, hparams
            )
            # val_loss is MSE, convert to RMSE
            fold_rmses.append(np.sqrt(best_val_loss))

            # Per-fold pruning: report running mean RMSE so Optuna can
            # prune unpromising trials before all folds complete.
            running_mean_rmse = float(np.mean(fold_rmses[: fold_idx + 1]))
            trial.report(running_mean_rmse, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rmses))

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
        graph_data: Data,
        skip_hpo: bool = False,
    ) -> TargetResult:
        """
        Run full HPO + CV pipeline for one target variable.

        Args:
            target_col: Target column name (e.g. 'lbm').
            X: Feature matrix [N, D].
            y_all: DataFrame with all target columns.
            folds: Spatial fold assignments.
            region_ids: Region IDs for each row.
            graph_data: PyG Data with edge_index.
            skip_hpo: If True, use default hyperparameters (no Optuna).

        Returns:
            TargetResult with metrics and predictions.
        """
        y = y_all[target_col].values
        unique_folds = np.unique(folds)
        target_name = TARGET_NAMES.get(target_col, target_col)

        logger.info(f"\n--- Target: {target_col} ({target_name}) ---")
        logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

        edge_index = graph_data.edge_index
        num_nodes = graph_data.num_nodes

        if skip_hpo:
            # Use default config hyperparameters
            best_hparams = {
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "learning_rate": self.config.learning_rate,
                "dropout": self.config.dropout,
                "weight_decay": self.config.weight_decay,
                "conv_type": self.config.conv_type,
                "num_heads": self.config.num_heads,
                "use_residual": self.config.use_residual,
                "use_layer_norm": self.config.use_layer_norm,
            }
            logger.info(
                f"  Skipping HPO, using defaults: "
                f"conv={best_hparams['conv_type']}, "
                f"hidden={best_hparams['hidden_dim']}, "
                f"layers={best_hparams['num_layers']}, "
                f"lr={best_hparams['learning_rate']}"
            )
        else:
            # Optuna hyperparameter optimization
            logger.info(
                f"  Running Optuna optimization ({self.config.n_trials} trials)..."
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
                pruner=optuna.pruners.MedianPruner(),
            )
            study.optimize(
                lambda trial: self._optuna_objective(
                    trial, X, y, folds, unique_folds, edge_index, num_nodes
                ),
                n_trials=self.config.n_trials,
                show_progress_bar=False,
            )

            best_hparams = {
                **study.best_params,
                "num_heads": self.config.num_heads,
                "use_residual": self.config.use_residual,
                "use_layer_norm": self.config.use_layer_norm,
            }
            logger.info(
                f"  Best HPO: conv={best_hparams['conv_type']}, "
                f"hidden={best_hparams['hidden_dim']}, "
                f"layers={best_hparams['num_layers']}, "
                f"lr={best_hparams['learning_rate']:.6f}, "
                f"dropout={best_hparams['dropout']:.3f}, "
                f"wd={best_hparams['weight_decay']:.6f}, "
                f"RMSE={study.best_value:.4f}"
            )

        self.best_hparams[target_col] = best_hparams

        # ----- Final CV evaluation with best hyperparameters -----
        logger.info("  Evaluating with spatial CV (best hyperparameters)...")
        oof_predictions = np.full(len(y), np.nan)
        fold_metrics_list: List[FoldMetrics] = []
        training_curves: Dict[int, List[float]] = {}

        for fold_id in unique_folds:
            val_mask = folds == fold_id
            train_mask = ~val_mask

            model, best_val_loss, fold_scaler, val_history = self._train_one_fold(
                X, y, edge_index, num_nodes, train_mask, val_mask, best_hparams
            )
            training_curves[int(fold_id)] = val_history

            # Get predictions on validation set using the scaler fitted during
            # training (reuse fold_scaler to avoid train/predict misalignment).
            device = torch.device(self.config.device)

            X_scaled = np.empty_like(X, dtype=np.float32)
            X_scaled[train_mask] = fold_scaler.transform(X[train_mask]).astype(
                np.float32
            )
            X_scaled[val_mask] = fold_scaler.transform(X[val_mask]).astype(np.float32)
            other_mask = ~(train_mask | val_mask)
            if other_mask.any():
                X_scaled[other_mask] = fold_scaler.transform(X[other_mask]).astype(
                    np.float32
                )

            data = Data(
                x=torch.tensor(X_scaled, dtype=torch.float32),
                y=torch.tensor(y, dtype=torch.float32),
                edge_index=edge_index,
                num_nodes=num_nodes,
            ).to(device)

            model = model.to(device)
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index).squeeze(-1).cpu().numpy()

            oof_predictions[val_mask] = out[val_mask]

            y_val = y[val_mask]
            y_pred_val = out[val_mask]
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            mae = float(mean_absolute_error(y_val, y_pred_val))
            r2 = float(r2_score(y_val, y_pred_val))

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
            del data
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
            best_alpha=0.0,       # Not applicable for GNN
            best_l1_ratio=0.0,    # Not applicable for GNN
            fold_metrics=fold_metrics_list,
            overall_r2=overall_r2,
            overall_rmse=overall_rmse,
            overall_mae=overall_mae,
            coefficients=np.zeros(len(self.feature_names)),  # GNN has no linear coefficients
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

    def run(self, skip_hpo: bool = False) -> Dict[str, TargetResult]:
        """
        Run the full DNN probe pipeline for all target variables.

        Args:
            skip_hpo: If True, use default hyperparameters (no Optuna).

        Returns:
            Dictionary mapping target column to TargetResult.
        """
        mode = "default HPs" if skip_hpo else f"Optuna ({self.config.n_trials} trials)"
        logger.info(f"=== DNN Probe Regression ({mode}) ===")
        logger.info(f"Study area: {self.config.study_area}")
        logger.info(f"Year: {self.config.year}")
        logger.info(f"Device: {self.config.device}")
        logger.info(
            f"Architecture: {self.config.conv_type.upper()}, "
            f"hidden={self.config.hidden_dim}, layers={self.config.num_layers}"
        )

        # Load and join data
        gdf = self.load_and_join_data()

        # Build H3 adjacency graph
        region_ids = gdf.index.values
        graph_data = self.build_h3_graph(region_ids)

        # Create spatial blocks
        folds = self.create_spatial_blocks(gdf)

        # Extract feature matrix
        X = gdf[self.feature_names].values
        y_all = gdf[self.config.target_cols]

        logger.info(f"\nFeature matrix: {X.shape}")
        logger.info(f"Targets: {self.config.target_cols}")

        # Run for each target
        for target_col in self.config.target_cols:
            self.run_for_target(
                target_col, X, y_all, folds, region_ids, graph_data,
                skip_hpo=skip_hpo,
            )

        # Summary
        logger.info("\n=== Summary ===")
        for target_col, result in self.results.items():
            logger.info(
                f"  {target_col} ({result.target_name}): "
                f"R2={result.overall_r2:.4f}, RMSE={result.overall_rmse:.4f}"
            )

        return self.results

    def run_simple(self) -> Dict[str, TargetResult]:
        """
        Run DNN probe with default hyperparameters, no Optuna.

        Convenience alias for ``run(skip_hpo=True)``.
        """
        return self.run(skip_hpo=True)

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
            "n_trials": self.config.n_trials,
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
        description="DNN Probe: GNN-based AlphaEarth -> Leefbaarometer"
    )
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument(
        "--conv-type",
        choices=["gcn", "gat"],
        default="gcn",
        help="GNN convolution type (default: gcn)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials (default: 30)",
    )
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
        help="Hidden dimension (skips HPO for this param if set)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of GCN/GAT layers (skips HPO for this param if set)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device (default: auto)",
    )
    parser.add_argument(
        "--skip-hpo",
        action="store_true",
        help="Use default hyperparameters, skip Optuna",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = DNNProbeConfig(
        study_area=args.study_area,
        conv_type=args.conv_type,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        max_epochs=args.max_epochs,
        device=args.device,
    )

    # If user fixes hidden_dim or num_layers via CLI, narrow the HPO search
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
        config.hidden_dim_choices = [args.hidden_dim]
    if args.num_layers is not None:
        config.num_layers = args.num_layers
        config.num_layers_choices = [args.num_layers]

    regressor = DNNProbeRegressor(config)
    regressor.run(skip_hpo=args.skip_hpo)
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
        )
        logger.info(f"Generated {len(plot_paths)} plots to {viz_dir}")


if __name__ == "__main__":
    main()
