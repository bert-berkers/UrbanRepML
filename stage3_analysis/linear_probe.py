#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear Probe Regression: AlphaEarth Embeddings -> Leefbaarometer

Fits ElasticNet regression with Optuna hyperparameter optimization and
spatial block cross-validation to evaluate whether AlphaEarth embeddings
encode Dutch liveability (leefbaarometer) signals.

Follows the approach from JohnKilbride/GEE_MediumBlog_Logic:
    - ElasticNet (L1+L2 regularization)
    - Optuna Bayesian hyperparameter optimization (alpha + l1_ratio)
    - Spatial block cross-validation via spatialkfold
    - Per-target training with out-of-fold predictions

Usage:
    python -m stage3_analysis.linear_probe --study-area netherlands
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from spatialkfold.blocks import spatial_blocks

logger = logging.getLogger(__name__)

# Leefbaarometer target columns
TARGET_COLS = ["lbm", "fys", "onv", "soc", "vrz", "won"]

# Target display names
TARGET_NAMES = {
    "lbm": "Overall Liveability",
    "fys": "Physical Environment",
    "onv": "Safety",
    "soc": "Social Cohesion",
    "vrz": "Amenities",
    "won": "Housing Quality",
}


@dataclass
class LinearProbeConfig:
    """Configuration for linear probe regression."""

    study_area: str = "netherlands"
    year: int = 2022
    h3_resolution: int = 10
    target_cols: List[str] = field(default_factory=lambda: list(TARGET_COLS))

    # Spatial block CV
    n_folds: int = 5
    block_width: int = 25_000   # meters
    block_height: int = 25_000  # meters
    random_state: int = 42

    # Optuna
    n_trials: int = 50
    alpha_low: float = 1e-5
    alpha_high: float = 10.0
    l1_ratio_low: float = 0.0
    l1_ratio_high: float = 1.0

    # Data paths (relative to project root)
    embeddings_path: Optional[str] = None
    pca_embeddings_path: Optional[str] = None
    target_path: Optional[str] = None
    output_dir: Optional[str] = None

    def __post_init__(self):
        if self.embeddings_path is None:
            self.embeddings_path = (
                f"data/study_areas/{self.study_area}/embeddings/alphaearth/"
                f"{self.study_area}_res{self.h3_resolution}_{self.year}.parquet"
            )
        if self.pca_embeddings_path is None:
            self.pca_embeddings_path = (
                f"data/study_areas/{self.study_area}/embeddings/alphaearth/"
                f"{self.study_area}_res{self.h3_resolution}_pca16_{self.year}.parquet"
            )
        if self.target_path is None:
            self.target_path = (
                f"data/study_areas/{self.study_area}/target/leefbaarometer/"
                f"leefbaarometer_h3res{self.h3_resolution}_{self.year}.parquet"
            )
        if self.output_dir is None:
            self.output_dir = (
                f"data/study_areas/{self.study_area}/analysis/linear_probe"
            )


@dataclass
class FoldMetrics:
    """Metrics for a single CV fold."""

    fold: int
    rmse: float
    mae: float
    r2: float
    n_train: int
    n_test: int


@dataclass
class TargetResult:
    """Results for a single target variable."""

    target: str
    target_name: str
    best_alpha: float
    best_l1_ratio: float
    fold_metrics: List[FoldMetrics]
    overall_r2: float
    overall_rmse: float
    overall_mae: float
    coefficients: np.ndarray
    intercept: float
    feature_names: List[str]
    oof_predictions: np.ndarray
    actual_values: np.ndarray
    region_ids: np.ndarray


class LinearProbeRegressor:
    """
    ElasticNet linear probe with spatial block CV and Optuna optimization.

    Evaluates whether AlphaEarth embeddings encode liveability signals by
    fitting regularized linear models to predict leefbaarometer scores.
    Spatial block cross-validation prevents spatial autocorrelation leakage.
    """

    def __init__(self, config: LinearProbeConfig, project_root: Optional[Path] = None):
        self.config = config
        self.project_root = project_root or Path(__file__).parent.parent
        self.results: Dict[str, TargetResult] = {}
        self.data_gdf: Optional[gpd.GeoDataFrame] = None
        self.feature_names: List[str] = []

    def load_and_join_data(self, use_pca: bool = False) -> gpd.GeoDataFrame:
        """
        Load embeddings and target data, inner join on region_id / h3_index.

        Args:
            use_pca: If True, use PCA-16 embeddings instead of full 64-dim.

        Returns:
            GeoDataFrame with embeddings + target columns, indexed by region_id.
        """
        emb_path_str = (self.config.pca_embeddings_path if use_pca
                        else self.config.embeddings_path)
        emb_path = self.project_root / emb_path_str
        target_path = self.project_root / self.config.target_path

        logger.info(f"Loading embeddings from {emb_path}")
        emb_df = pd.read_parquet(emb_path)

        # Normalize index: embeddings may use h3_index column
        if "h3_index" in emb_df.columns and emb_df.index.name != "region_id":
            emb_df = emb_df.set_index("h3_index")
            emb_df.index.name = "region_id"

        # Identify embedding feature columns
        if use_pca:
            self.feature_names = [c for c in emb_df.columns
                                  if c.startswith("PC") or c.startswith("pca_")]
            if not self.feature_names:
                # Fall back to numeric columns excluding metadata
                exclude = {"pixel_count", "tile_count", "geometry"}
                self.feature_names = [c for c in emb_df.columns
                                      if c not in exclude
                                      and pd.api.types.is_numeric_dtype(emb_df[c])]
        else:
            self.feature_names = [c for c in emb_df.columns
                                  if c.startswith("A") and c[1:].isdigit()]

        logger.info(f"  Embedding features: {len(self.feature_names)} "
                     f"({self.feature_names[0]}..{self.feature_names[-1]})")
        logger.info(f"  Embedding rows: {len(emb_df):,}")

        # Keep only embedding features (drop geometry, pixel_count, etc.)
        emb_df = emb_df[self.feature_names]

        # Load target
        logger.info(f"Loading target from {target_path}")
        target_df = pd.read_parquet(target_path)
        if target_df.index.name != "region_id" and "region_id" in target_df.columns:
            target_df = target_df.set_index("region_id")
        logger.info(f"  Target rows: {len(target_df):,}")

        # Inner join
        joined = emb_df.join(target_df[self.config.target_cols], how="inner")
        logger.info(f"  Inner join: {len(joined):,} hexagons with both embeddings and targets")

        # Drop rows with any NaN in targets or features
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

    def create_spatial_blocks(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Create spatial block fold assignments using spatialkfold.

        Projects to EPSG:28992 (RD New) for metric block sizes in the Netherlands.
        The spatial_blocks function returns block polygons with fold labels;
        we then sjoin data points to blocks to assign folds to each point.

        Returns:
            Array of fold assignments (1 to n_folds) per row, aligned to gdf index.
        """
        logger.info(f"Creating spatial blocks: {self.config.n_folds} folds, "
                     f"{self.config.block_width}m x {self.config.block_height}m")

        # spatialkfold needs projected CRS for metric block sizes
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

        logger.info(f"  Created {len(blocks_gdf)} spatial blocks across {self.config.n_folds} folds")

        # Assign folds to data points via spatial join
        # blocks_gdf has block polygons with 'folds' column
        # gdf_proj has data points -- sjoin to find which block each point falls in
        points_with_folds = gpd.sjoin(
            gdf_proj[["geometry"]],
            blocks_gdf[["geometry", "folds"]],
            how="left",
            predicate="within",
        )

        # Handle points that fall outside all blocks (edge cases)
        # Assign them to the nearest fold
        missing_mask = points_with_folds["folds"].isna()
        if missing_mask.any():
            n_missing = missing_mask.sum()
            logger.warning(f"  {n_missing} points outside all blocks, assigning to fold 1")
            points_with_folds.loc[missing_mask, "folds"] = 1

        # Handle duplicate entries from sjoin (point on block boundary)
        # Keep first match per original index
        points_with_folds = points_with_folds[~points_with_folds.index.duplicated(keep="first")]

        # Reindex to match original gdf order
        folds = points_with_folds.loc[gdf.index, "folds"].values.astype(int)

        unique_folds = np.unique(folds)
        fold_counts = {int(f): int((folds == f).sum()) for f in unique_folds}
        logger.info(f"  Fold sizes: {fold_counts}")

        return folds

    def _optuna_objective(
        self,
        trial: optuna.Trial,
        X: np.ndarray,
        y: np.ndarray,
        folds: np.ndarray,
        unique_folds: np.ndarray,
    ) -> float:
        """Optuna objective: minimize mean RMSE across spatial folds."""
        alpha = trial.suggest_float("alpha", self.config.alpha_low,
                                    self.config.alpha_high, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", self.config.l1_ratio_low,
                                       self.config.l1_ratio_high)

        rmse_scores = []
        for fold_id in unique_folds:
            test_mask = folds == fold_id
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # Standardize features per fold
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                               max_iter=1000, random_state=self.config.random_state)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    def _train_and_evaluate_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: np.ndarray,
        unique_folds: np.ndarray,
        alpha: float,
        l1_ratio: float,
    ) -> Tuple[np.ndarray, List[FoldMetrics]]:
        """
        Train and evaluate with spatial CV using optimal hyperparameters.

        Returns:
            Out-of-fold predictions and per-fold metrics.
        """
        oof_predictions = np.full(len(y), np.nan)
        fold_metrics_list = []

        for fold_id in unique_folds:
            test_mask = folds == fold_id
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # Standardize features per fold
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                               max_iter=1000, random_state=self.config.random_state)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            oof_predictions[test_mask] = y_pred

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            fold_metrics_list.append(FoldMetrics(
                fold=int(fold_id),
                rmse=rmse,
                mae=mae,
                r2=r2,
                n_train=int(train_mask.sum()),
                n_test=int(test_mask.sum()),
            ))

            logger.info(f"    Fold {fold_id}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f} "
                         f"(train={train_mask.sum():,}, test={test_mask.sum():,})")

        return oof_predictions, fold_metrics_list

    def _train_and_evaluate_cv_linear(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: np.ndarray,
        unique_folds: np.ndarray,
    ) -> Tuple[np.ndarray, List[FoldMetrics]]:
        """
        Train and evaluate plain LinearRegression with spatial CV (no regularization).

        Returns:
            Out-of-fold predictions and per-fold metrics.
        """
        oof_predictions = np.full(len(y), np.nan)
        fold_metrics_list = []

        for fold_id in unique_folds:
            test_mask = folds == fold_id
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            oof_predictions[test_mask] = y_pred

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            fold_metrics_list.append(FoldMetrics(
                fold=int(fold_id),
                rmse=rmse,
                mae=mae,
                r2=r2,
                n_train=int(train_mask.sum()),
                n_test=int(test_mask.sum()),
            ))

            logger.info(f"    Fold {fold_id}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f} "
                         f"(train={train_mask.sum():,}, test={test_mask.sum():,})")

        return oof_predictions, fold_metrics_list

    def _train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        l1_ratio: float,
    ) -> ElasticNet:
        """Train final model on all data with optimal hyperparameters."""
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                           max_iter=1000, random_state=self.config.random_state)
        model.fit(X_s, y)
        return model

    def run_for_target(
        self,
        target_col: str,
        X: np.ndarray,
        y_all: pd.DataFrame,
        folds: np.ndarray,
        region_ids: np.ndarray,
    ) -> TargetResult:
        """
        Run full optimization and evaluation pipeline for one target variable.

        Args:
            target_col: Target column name (e.g. 'lbm')
            X: Feature matrix (n_samples, n_features)
            y_all: DataFrame with all target columns
            folds: Spatial fold assignments
            region_ids: Region IDs for each row

        Returns:
            TargetResult with metrics, coefficients, and predictions.
        """
        y = y_all[target_col].values
        unique_folds = np.unique(folds)
        target_name = TARGET_NAMES.get(target_col, target_col)

        logger.info(f"\n--- Target: {target_col} ({target_name}) ---")
        logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

        # Optuna hyperparameter optimization
        logger.info(f"  Running Optuna optimization ({self.config.n_trials} trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(
                                        seed=self.config.random_state))
        study.optimize(
            lambda trial: self._optuna_objective(trial, X, y, folds, unique_folds),
            n_trials=self.config.n_trials,
            show_progress_bar=False,
        )

        best_alpha = study.best_params["alpha"]
        best_l1_ratio = study.best_params["l1_ratio"]
        logger.info(f"  Best: alpha={best_alpha:.6f}, l1_ratio={best_l1_ratio:.4f}, "
                     f"RMSE={study.best_value:.4f}")

        # Out-of-fold evaluation with best hyperparameters
        logger.info(f"  Evaluating with spatial CV...")
        oof_predictions, fold_metrics = self._train_and_evaluate_cv(
            X, y, folds, unique_folds, best_alpha, best_l1_ratio
        )

        # Overall metrics from OOF predictions
        valid_mask = ~np.isnan(oof_predictions)
        overall_r2 = r2_score(y[valid_mask], oof_predictions[valid_mask])
        overall_rmse = np.sqrt(mean_squared_error(y[valid_mask], oof_predictions[valid_mask]))
        overall_mae = mean_absolute_error(y[valid_mask], oof_predictions[valid_mask])

        logger.info(f"  Overall OOF: R2={overall_r2:.4f}, RMSE={overall_rmse:.4f}, "
                     f"MAE={overall_mae:.4f}")

        # Train final model on all data for coefficients
        final_model = self._train_final_model(X, y, best_alpha, best_l1_ratio)

        result = TargetResult(
            target=target_col,
            target_name=target_name,
            best_alpha=best_alpha,
            best_l1_ratio=best_l1_ratio,
            fold_metrics=fold_metrics,
            overall_r2=overall_r2,
            overall_rmse=overall_rmse,
            overall_mae=overall_mae,
            coefficients=final_model.coef_,
            intercept=final_model.intercept_,
            feature_names=self.feature_names,
            oof_predictions=oof_predictions,
            actual_values=y,
            region_ids=region_ids,
        )

        self.results[target_col] = result
        return result

    def run_simple(self, use_pca: bool = False) -> Dict[str, TargetResult]:
        """
        Run simple LinearRegression without regularization or CV.

        Fits plain LinearRegression on all data, computes in-sample metrics.
        No hyperparameter optimization, no spatial cross-validation.
        Useful as a baseline to compare against ElasticNet's regularization.

        Args:
            use_pca: If True, use PCA-16 embeddings instead of full 64-dim.

        Returns:
            Dictionary mapping target column to TargetResult.
        """
        emb_type = "PCA-16" if use_pca else f"full {len(self.feature_names) or '?'}-dim"
        logger.info(f"=== Linear Probe Regression (Simple Mode, {emb_type}) ===")
        logger.info(f"Study area: {self.config.study_area}")
        logger.info(f"Year: {self.config.year}")
        logger.info("Mode: Simple LinearRegression (no regularization, no CV)")

        # Load and join data
        gdf = self.load_and_join_data(use_pca=use_pca)

        # Extract feature matrix
        X = gdf[self.feature_names].values
        y_all = gdf[self.config.target_cols]
        region_ids = gdf.index.values

        logger.info(f"\nFeature matrix: {X.shape}")
        logger.info(f"Targets: {self.config.target_cols}")

        # Standardize features once for all targets
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit simple LinearRegression for each target
        for target_col in self.config.target_cols:
            y = y_all[target_col].values
            target_name = TARGET_NAMES.get(target_col, target_col)

            logger.info(f"\n--- Target: {target_col} ({target_name}) ---")
            logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

            # Fit LinearRegression (no regularization)
            model = LinearRegression()
            model.fit(X_scaled, y)

            # In-sample predictions
            y_pred = model.predict(X_scaled)

            # Compute metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            logger.info(f"  In-sample: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

            result = TargetResult(
                target=target_col,
                target_name=target_name,
                best_alpha=0.0,  # No regularization
                best_l1_ratio=0.0,
                fold_metrics=[],  # No CV folds
                overall_r2=r2,
                overall_rmse=rmse,
                overall_mae=mae,
                coefficients=model.coef_,
                intercept=model.intercept_,
                feature_names=self.feature_names,
                oof_predictions=y_pred,  # In-sample predictions
                actual_values=y,
                region_ids=region_ids,
            )

            self.results[target_col] = result

        # Summary
        logger.info("\n=== Summary ===")
        for target_col, result in self.results.items():
            logger.info(f"  {target_col} ({result.target_name}): "
                         f"R2={result.overall_r2:.4f}, RMSE={result.overall_rmse:.4f}")

        return self.results

    def run_linear_cv(self, use_pca: bool = False) -> Dict[str, TargetResult]:
        """
        Run plain LinearRegression with spatial block CV (no regularization, no Optuna).

        OLS regression with spatial cross-validation to get honest
        out-of-fold R² estimates.

        Args:
            use_pca: If True, use PCA-16 embeddings instead of full 64-dim.

        Returns:
            Dictionary mapping target column to TargetResult.
        """
        emb_type = "PCA-16" if use_pca else f"full {len(self.feature_names) or '?'}-dim"
        logger.info(f"=== Linear Probe Regression (Linear CV Mode, {emb_type}) ===")
        logger.info(f"Study area: {self.config.study_area}")
        logger.info(f"Year: {self.config.year}")
        logger.info("Mode: Plain LinearRegression with spatial block CV")

        gdf = self.load_and_join_data(use_pca=use_pca)
        folds = self.create_spatial_blocks(gdf)

        X = gdf[self.feature_names].values
        y_all = gdf[self.config.target_cols]
        region_ids = gdf.index.values

        logger.info(f"\nFeature matrix: {X.shape}")
        logger.info(f"Targets: {self.config.target_cols}")

        unique_folds = np.unique(folds)

        for target_col in self.config.target_cols:
            y = y_all[target_col].values
            target_name = TARGET_NAMES.get(target_col, target_col)

            logger.info(f"\n--- Target: {target_col} ({target_name}) ---")
            logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")

            oof_predictions, fold_metrics = self._train_and_evaluate_cv_linear(
                X, y, folds, unique_folds
            )

            valid_mask = ~np.isnan(oof_predictions)
            overall_r2 = r2_score(y[valid_mask], oof_predictions[valid_mask])
            overall_rmse = np.sqrt(mean_squared_error(y[valid_mask], oof_predictions[valid_mask]))
            overall_mae = mean_absolute_error(y[valid_mask], oof_predictions[valid_mask])

            logger.info(f"  Overall OOF: R2={overall_r2:.4f}, RMSE={overall_rmse:.4f}, "
                         f"MAE={overall_mae:.4f}")

            # Train final model on all data for coefficients
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            final_model = LinearRegression()
            final_model.fit(X_s, y)

            result = TargetResult(
                target=target_col,
                target_name=target_name,
                best_alpha=0.0,
                best_l1_ratio=0.0,
                fold_metrics=fold_metrics,
                overall_r2=overall_r2,
                overall_rmse=overall_rmse,
                overall_mae=overall_mae,
                coefficients=final_model.coef_,
                intercept=final_model.intercept_,
                feature_names=self.feature_names,
                oof_predictions=oof_predictions,
                actual_values=y,
                region_ids=region_ids,
            )

            self.results[target_col] = result

        # Summary
        logger.info("\n=== Summary ===")
        for target_col, result in self.results.items():
            logger.info(f"  {target_col} ({result.target_name}): "
                         f"R2={result.overall_r2:.4f}, RMSE={result.overall_rmse:.4f}")

        return self.results

    def run(self, use_pca: bool = False) -> Dict[str, TargetResult]:
        """
        Run the full linear probe pipeline for all target variables.

        Args:
            use_pca: If True, use PCA-16 embeddings instead of full 64-dim.

        Returns:
            Dictionary mapping target column to TargetResult.
        """
        emb_type = "PCA-16" if use_pca else f"full {len(self.feature_names) or '?'}-dim"
        logger.info(f"=== Linear Probe Regression ({emb_type}) ===")
        logger.info(f"Study area: {self.config.study_area}")
        logger.info(f"Year: {self.config.year}")

        # Load and join data
        gdf = self.load_and_join_data(use_pca=use_pca)

        # Create spatial blocks
        folds = self.create_spatial_blocks(gdf)

        # Extract feature matrix
        X = gdf[self.feature_names].values
        y_all = gdf[self.config.target_cols]
        region_ids = gdf.index.values

        logger.info(f"\nFeature matrix: {X.shape}")
        logger.info(f"Targets: {self.config.target_cols}")

        # Run for each target
        for target_col in self.config.target_cols:
            self.run_for_target(target_col, X, y_all, folds, region_ids)

        # Summary
        logger.info("\n=== Summary ===")
        for target_col, result in self.results.items():
            logger.info(f"  {target_col} ({result.target_name}): "
                         f"R2={result.overall_r2:.4f}, RMSE={result.overall_rmse:.4f}")

        return self.results

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """Save all results to disk."""
        out_dir = output_dir or (self.project_root / self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics summary
        metrics_rows = []
        for target_col, result in self.results.items():
            row = {
                "target": target_col,
                "target_name": result.target_name,
                "best_alpha": result.best_alpha,
                "best_l1_ratio": result.best_l1_ratio,
                "overall_r2": result.overall_r2,
                "overall_rmse": result.overall_rmse,
                "overall_mae": result.overall_mae,
                "n_features": len(result.feature_names),
            }
            # Only add fold metrics if they exist (ElasticNet mode has them, simple mode doesn't)
            if result.fold_metrics:
                for fm in result.fold_metrics:
                    row[f"fold{fm.fold}_r2"] = fm.r2
                    row[f"fold{fm.fold}_rmse"] = fm.rmse
            metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
        logger.info(f"Saved metrics summary to {out_dir / 'metrics_summary.csv'}")

        # Save coefficients per target
        coef_rows = []
        for target_col, result in self.results.items():
            for feat_name, coef_val in zip(result.feature_names, result.coefficients):
                coef_rows.append({
                    "target": target_col,
                    "feature": feat_name,
                    "coefficient": coef_val,
                })
        coef_df = pd.DataFrame(coef_rows)
        coef_df.to_csv(out_dir / "coefficients.csv", index=False)
        logger.info(f"Saved coefficients to {out_dir / 'coefficients.csv'}")

        # Save out-of-fold predictions per target
        for target_col, result in self.results.items():
            pred_df = pd.DataFrame({
                "region_id": result.region_ids,
                "actual": result.actual_values,
                "predicted": result.oof_predictions,
                "residual": result.actual_values - result.oof_predictions,
            }).set_index("region_id")
            pred_path = out_dir / f"predictions_{target_col}.parquet"
            pred_df.to_parquet(pred_path)

        logger.info(f"Saved predictions to {out_dir}")

        # Save config
        config_dict = {
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
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        return out_dir


def main():
    """Run linear probe regression with default config."""
    import argparse

    parser = argparse.ArgumentParser(description="Linear Probe: AlphaEarth -> Leefbaarometer")
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument("--pca", action="store_true", help="Use PCA-16 embeddings")
    parser.add_argument("--mode", choices=["elasticnet", "simple", "linear"], default="elasticnet",
                        help="Regression mode: elasticnet (regularized with CV), simple (no regularization, no CV), "
                             "or linear (plain OLS with spatial CV)")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials (ElasticNet mode only)")
    parser.add_argument("--n-folds", type=int, default=5, help="Spatial CV folds (ElasticNet mode only)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = LinearProbeConfig(
        study_area=args.study_area,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
    )

    regressor = LinearProbeRegressor(config)

    if args.mode == "simple":
        regressor.run_simple(use_pca=args.pca)
    elif args.mode == "linear":
        regressor.run_linear_cv(use_pca=args.pca)
    else:
        regressor.run(use_pca=args.pca)

    regressor.save_results()


if __name__ == "__main__":
    main()
