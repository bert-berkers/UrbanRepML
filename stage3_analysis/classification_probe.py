#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classification Probe: Embeddings -> Urban Taxonomy

Fits LogisticRegression with spatial block cross-validation to
evaluate whether embeddings encode urban morphological
type signals (urban taxonomy hierarchical classification).

    - LogisticRegression (multinomial, balanced class weights)
    - Spatial block cross-validation via spatialkfold
    - Per-target training with out-of-fold predictions
    - Hierarchical accuracy degradation across 7 taxonomy levels

Usage:
    python -m stage3_analysis.classification_probe --study-area netherlands
    python -m stage3_analysis.classification_probe --visualize
    python -m stage3_analysis.classification_probe --quick-viz
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from utils import StudyAreaPaths
from utils.paths import write_run_info

from .linear_probe import (
    TAXONOMY_TARGET_COLS,
    TAXONOMY_TARGET_NAMES,
    FoldMetrics,
    TargetResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationProbeConfig:
    """Configuration for classification probe (LogisticRegression)."""

    study_area: str = "netherlands"
    year: int = 2022
    h3_resolution: int = 10
    target_name: str = "urban_taxonomy"
    target_cols: List[str] = field(default_factory=lambda: list(TAXONOMY_TARGET_COLS))
    modality: str = "alphaearth"

    # Spatial block CV
    n_folds: int = 5
    block_width: int = 10_000   # meters (10km blocks)
    block_height: int = 10_000  # meters
    random_state: int = 42

    # Data paths (relative to project root)
    embeddings_path: Optional[str] = None
    target_path: Optional[str] = None
    output_dir: Optional[str] = None

    # Run-level provenance
    run_descriptor: str = "default"

    def __post_init__(self):
        paths = StudyAreaPaths(self.study_area)
        self.run_id: Optional[str] = None

        if self.embeddings_path is None:
            self.embeddings_path = str(
                paths.embedding_file(self.modality, self.h3_resolution, self.year)
            )
        if self.target_path is None:
            target_year = 2025 if self.target_name == "urban_taxonomy" else self.year
            self.target_path = str(
                paths.target_file(self.target_name, self.h3_resolution, target_year)
            )
        if self.output_dir is None:
            if self.run_descriptor:
                self.run_id = paths.create_run_id(self.run_descriptor)
                self.output_dir = str(
                    paths.stage3_run("classification_probe", self.run_id)
                )
            else:
                self.output_dir = str(paths.stage3("classification_probe"))


class ClassificationProber:
    """
    LogisticRegression classification probe with spatial block cross-validation.

    Evaluates whether embeddings encode urban morphological type
    signals by fitting LogisticRegression to predict urban taxonomy classes.
    Spatial block cross-validation prevents spatial autocorrelation leakage.
    """

    def __init__(
        self,
        config: ClassificationProbeConfig,
        project_root: Optional[Path] = None,
    ):
        self.config = config
        self.project_root = project_root or Path(__file__).parent.parent
        self.results: Dict[str, TargetResult] = {}
        self.data_gdf: Optional[gpd.GeoDataFrame] = None
        self.feature_names: List[str] = []

    def load_and_join_data(self) -> gpd.GeoDataFrame:
        """
        Load embeddings and target data, inner join on region_id.

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

        # Identify embedding feature columns (A00-A63 pattern + fallback)
        self.feature_names = [
            c for c in emb_df.columns
            if (c.startswith("A") and c[1:].isdigit()) or c.startswith("emb_")
        ]
        if not self.feature_names:
            # Numeric fallback for arbitrary embedding column names
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

        # Keep only embedding features
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

        # Drop rows with any NaN in targets or features
        before = len(joined)
        joined = joined.dropna(subset=self.feature_names + self.config.target_cols)
        logger.info(f"  After dropna: {len(joined):,} (dropped {before - len(joined):,})")

        # Load pre-computed regions for geometry centroids
        paths = StudyAreaPaths(self.config.study_area)
        region_path = self.project_root / paths.region_file(self.config.h3_resolution)
        logger.info(f"  Loading regions from {region_path}")
        regions_gdf = gpd.read_parquet(region_path)
        if regions_gdf.index.name != "region_id":
            regions_gdf.index.name = "region_id"
        joined = joined.join(regions_gdf[["geometry"]], how="inner")
        joined = gpd.GeoDataFrame(joined, crs="EPSG:4326")
        joined["geometry"] = joined.geometry.centroid
        logger.info(f"  After region join: {len(joined):,} hexagons")

        self.data_gdf = joined
        return joined

    def create_spatial_blocks(self, gdf: gpd.GeoDataFrame) -> np.ndarray:
        """
        Create spatial block fold assignments using spatialkfold.

        Projects to EPSG:28992 (RD New) for metric block sizes in the Netherlands.

        Returns:
            Array of fold assignments (1 to n_folds) per row, aligned to gdf index.
        """
        logger.info(
            f"Creating spatial blocks: {self.config.n_folds} folds, "
            f"{self.config.block_width}m x {self.config.block_height}m"
        )

        gdf_proj = gdf.to_crs(epsg=28992)

        from spatialkfold.blocks import spatial_blocks

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

        # Assign folds to data points via spatial join
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

        # Reindex to match original gdf order
        folds = points_with_folds.loc[gdf.index, "folds"].values.astype(int)

        unique_folds = np.unique(folds)
        fold_counts = {int(f): int((folds == f).sum()) for f in unique_folds}
        logger.info(f"  Fold sizes: {fold_counts}")

        return folds

    def _train_and_evaluate_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: np.ndarray,
        unique_folds: np.ndarray,
    ) -> Tuple[np.ndarray, List[FoldMetrics]]:
        """
        Train LogisticRegression and evaluate with spatial CV.

        Always classification: LogisticRegression (multinomial) with
        balanced class weights, accuracy/F1 metrics.

        Returns:
            Out-of-fold predictions and per-fold metrics.
        """
        oof_predictions = np.full(len(y), np.nan)
        fold_metrics_list = []

        from tqdm import tqdm

        for fold_id in tqdm(unique_folds, desc=f"  CV folds", leave=False):
            test_mask = folds == fold_id
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=self.config.random_state,
            )
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            oof_predictions[test_mask] = y_pred

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            fold_metrics_list.append(
                FoldMetrics(
                    fold=int(fold_id),
                    rmse=0.0,
                    mae=0.0,
                    r2=0.0,
                    n_train=int(train_mask.sum()),
                    n_test=int(test_mask.sum()),
                    accuracy=acc,
                    f1_macro=f1,
                )
            )

            logger.info(
                f"    Fold {fold_id}: Acc={acc:.4f}, F1={f1:.4f} "
                f"(train={train_mask.sum():,}, test={test_mask.sum():,})"
            )

        return oof_predictions, fold_metrics_list

    def _get_target_name(self, target_col: str) -> str:
        """Look up display name for a target column."""
        return TAXONOMY_TARGET_NAMES.get(target_col, target_col)

    def run(self) -> Dict[str, TargetResult]:
        """
        Run classification probe with spatial block CV.

        LogisticRegression with accuracy/F1 for each taxonomy level.

        Returns:
            Dictionary mapping target column to TargetResult.
        """
        logger.info("=== Classification Probe (Spatial CV) ===")
        logger.info(f"Study area: {self.config.study_area}")
        logger.info(f"Year: {self.config.year}")
        logger.info(f"Target: {self.config.target_name}")

        gdf = self.load_and_join_data()
        folds = self.create_spatial_blocks(gdf)

        X = gdf[self.feature_names].values
        y_all = gdf[self.config.target_cols]
        region_ids = gdf.index.values

        logger.info(f"\nFeature matrix: {X.shape}")
        logger.info(f"Targets: {self.config.target_cols}")

        unique_folds = np.unique(folds)

        from tqdm import tqdm

        for target_col in tqdm(self.config.target_cols, desc="Targets"):
            y = y_all[target_col].values
            target_name = self._get_target_name(target_col)

            n_classes = int(len(np.unique(y[~np.isnan(y)])))
            logger.info(f"\n--- Target: {target_col} ({target_name}) ---")
            logger.info(
                f"  n_classes={n_classes}, distribution: "
                f"{dict(zip(*np.unique(y.astype(int), return_counts=True)))}"
            )

            oof_predictions, fold_metrics = self._train_and_evaluate_cv(
                X, y, folds, unique_folds,
            )

            valid_mask = ~np.isnan(oof_predictions)

            overall_acc = accuracy_score(
                y[valid_mask], oof_predictions[valid_mask]
            )
            overall_f1 = f1_score(
                y[valid_mask],
                oof_predictions[valid_mask],
                average="macro",
                zero_division=0,
            )
            logger.info(
                f"  Overall OOF: Acc={overall_acc:.4f}, F1={overall_f1:.4f}"
            )

            result = TargetResult(
                target=target_col,
                target_name=target_name,
                best_alpha=0.0,
                best_l1_ratio=0.0,
                fold_metrics=fold_metrics,
                overall_r2=0.0,
                overall_rmse=0.0,
                overall_mae=0.0,
                coefficients=np.zeros(len(self.feature_names)),
                intercept=0.0,
                feature_names=self.feature_names,
                oof_predictions=oof_predictions,
                actual_values=y,
                region_ids=region_ids,
                overall_accuracy=overall_acc,
                overall_f1_macro=overall_f1,
                n_classes=n_classes,
                task_type="classification",
            )

            self.results[target_col] = result

        # Summary
        logger.info("\n=== Summary ===")
        for target_col, result in self.results.items():
            logger.info(
                f"  {target_col} ({result.target_name}): "
                f"Acc={result.overall_accuracy:.4f}, "
                f"F1={result.overall_f1_macro:.4f}, "
                f"n_classes={result.n_classes}"
            )

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
                "task_type": "classification",
                "overall_accuracy": result.overall_accuracy,
                "overall_f1_macro": result.overall_f1_macro,
                "n_classes": result.n_classes,
                "n_features": len(result.feature_names),
            }
            for fm in result.fold_metrics:
                row[f"fold{fm.fold}_accuracy"] = fm.accuracy
                row[f"fold{fm.fold}_f1_macro"] = fm.f1_macro
            metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
        logger.info(f"Saved metrics summary to {out_dir / 'metrics_summary.csv'}")

        # Save out-of-fold predictions per target
        for target_col, result in self.results.items():
            pred_dict = {
                "region_id": result.region_ids,
                "actual": result.actual_values,
                "predicted": result.oof_predictions,
            }
            pred_df = pd.DataFrame(pred_dict).set_index("region_id")
            pred_path = out_dir / f"predictions_{target_col}.parquet"
            pred_df.to_parquet(pred_path)

        logger.info(f"Saved predictions to {out_dir}")

        # Save config
        config_dict = {
            "study_area": self.config.study_area,
            "year": self.config.year,
            "h3_resolution": self.config.h3_resolution,
            "task_type": "classification",
            "target_name": self.config.target_name,
            "n_folds": self.config.n_folds,
            "block_width": self.config.block_width,
            "block_height": self.config.block_height,
            "random_state": self.config.random_state,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Write run-level provenance
        if self.config.run_id is not None:
            _paths = StudyAreaPaths(self.config.study_area)
            _latest = _paths.latest_run(_paths.stage1(self.config.modality))
            _upstream_label = _latest.name if _latest else "flat"
            write_run_info(
                out_dir,
                stage="stage3",
                study_area=self.config.study_area,
                config=config_dict,
                upstream_runs={f"stage1/{self.config.modality}": _upstream_label},
            )
            logger.info(f"Saved run_info.json to {out_dir / 'run_info.json'}")

        return out_dir


def main():
    """Run classification probe with default config."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Classification Probe: Embeddings -> Urban Taxonomy"
    )
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument("--modality", default="alphaearth",
                        help="Stage1 modality name (default: alphaearth)")
    parser.add_argument("--stage2-model", default=None,
                        help="Stage2 fusion model name (overrides --modality)")
    parser.add_argument("--embeddings-path", default=None,
                        help="Direct path to embeddings parquet (overrides all)")
    parser.add_argument(
        "--target-name",
        default="urban_taxonomy",
        help="Target dataset name (default: urban_taxonomy)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Spatial CV folds"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=10000,
        help="Spatial block size in meters (default: 10000)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations after probe",
    )
    parser.add_argument(
        "--quick-viz",
        action="store_true",
        help="Skip spatial maps (faster visualization)",
    )
    parser.add_argument(
        "--plot-only",
        type=str,
        default=None,
        help="Skip training, load results from this run dir and generate plots only",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # --plot-only mode: load saved results and generate visualizations
    if args.plot_only:
        from stage3_analysis.classification_viz import ClassificationVisualizer

        run_dir = Path(args.plot_only)
        logger.info(f"Plot-only mode: loading results from {run_dir}")

        results: Dict[str, TargetResult] = {}
        metrics_df = pd.read_csv(run_dir / "metrics_summary.csv")
        for _, row in metrics_df.iterrows():
            target_col = row["target"]
            pred_path = run_dir / f"predictions_{target_col}.parquet"
            pred_df = pd.read_parquet(pred_path)
            results[target_col] = TargetResult(
                target=target_col,
                target_name=row["target_name"],
                best_alpha=0.0,
                best_l1_ratio=0.0,
                fold_metrics=[],
                overall_r2=0.0,
                overall_rmse=0.0,
                overall_mae=0.0,
                coefficients=np.zeros(0),
                intercept=0.0,
                feature_names=[],
                oof_predictions=pred_df["predicted"].values,
                actual_values=pred_df["actual"].values,
                region_ids=pred_df.index.values,
                overall_accuracy=row.get("overall_accuracy"),
                overall_f1_macro=row.get("overall_f1_macro"),
                n_classes=int(row.get("n_classes", 0)),
                task_type="classification",
            )

        viz_dir = run_dir / "plots"
        logger.info(f"Generating plots to {viz_dir}...")
        viz = ClassificationVisualizer(
            results=results,
            output_dir=viz_dir,
            study_area=args.study_area,
        )
        plot_paths = viz.plot_all(skip_spatial=args.quick_viz)
        logger.info(f"Generated {len(plot_paths)} plots to {viz_dir}")
        return

    if args.embeddings_path:
        config = ClassificationProbeConfig(
            study_area=args.study_area,
            embeddings_path=args.embeddings_path,
            target_name=args.target_name,
            n_folds=args.n_folds,
            block_width=args.block_size,
            block_height=args.block_size,
        )
    elif args.stage2_model:
        paths = StudyAreaPaths(args.study_area)
        fused_path = paths.fused_embedding_file(args.stage2_model, 10)
        config = ClassificationProbeConfig(
            study_area=args.study_area,
            embeddings_path=str(fused_path),
            modality=args.stage2_model,
            target_name=args.target_name,
            n_folds=args.n_folds,
            block_width=args.block_size,
            block_height=args.block_size,
        )
    else:
        config = ClassificationProbeConfig(
            study_area=args.study_area,
            modality=args.modality,
            target_name=args.target_name,
            n_folds=args.n_folds,
            block_width=args.block_size,
            block_height=args.block_size,
        )

    prober = ClassificationProber(config)
    prober.run()
    out_dir = prober.save_results()

    if args.visualize or args.quick_viz:
        from stage3_analysis.classification_viz import ClassificationVisualizer

        viz = ClassificationVisualizer(
            results=prober.results,
            output_dir=out_dir / "plots",
            study_area=args.study_area,
        )
        viz.plot_all(skip_spatial=args.quick_viz)


if __name__ == "__main__":
    main()
