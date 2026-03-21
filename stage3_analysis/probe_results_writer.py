"""
Probe Results Writer
====================

Writes standardized probe results (predictions + metrics) to parquet files
for cross-approach comparison.

Output layout::

    data/study_areas/{area}/probe_results/{approach}/
    ├── predictions.parquet   # per-hex, per-target
    └── metrics.parquet       # per-target, per-metric

Lifetime: durable
Stage: 3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import pandas as pd

from utils.paths import StudyAreaPaths

if TYPE_CHECKING:
    from stage3_analysis.linear_probe import TargetResult

logger = logging.getLogger(__name__)


class ProbeResultsWriter:
    """Write probe results to a standardized parquet store."""

    def __init__(self, study_area: str, approach: str):
        self.paths = StudyAreaPaths(study_area)
        self.approach = approach

    def write(self, results: Dict[str, "TargetResult"]) -> Path:
        """Write predictions.parquet + metrics.parquet.

        Args:
            results: Mapping of target name to TargetResult dataclass.

        Returns:
            Path to the approach directory containing both parquets.
        """
        out_dir = self.paths.probe_results(self.approach)
        out_dir.mkdir(parents=True, exist_ok=True)

        predictions_df = self._build_predictions(results)
        metrics_df = self._build_metrics(results)

        pred_path = out_dir / "predictions.parquet"
        metrics_path = out_dir / "metrics.parquet"

        predictions_df.to_parquet(pred_path, index=False)
        metrics_df.to_parquet(metrics_path, index=False)

        logger.info(
            "Wrote probe results for approach '%s': "
            "%d prediction rows, %d metric rows -> %s",
            self.approach,
            len(predictions_df),
            len(metrics_df),
            out_dir,
        )

        return out_dir

    @classmethod
    def write_from_regressor(
        cls, regressor, approach: str, study_area: str
    ) -> Path:
        """Convenience: extract results from a regressor and write.

        Args:
            regressor: Any probe regressor with a ``.results`` dict attribute
                mapping target names to TargetResult instances.
            approach: Approach label (e.g. "ring_agg_k10").
            study_area: Study area name (e.g. "netherlands").

        Returns:
            Path to the approach directory containing both parquets.
        """
        writer = cls(study_area=study_area, approach=approach)
        return writer.write(regressor.results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_predictions(
        self, results: Dict[str, "TargetResult"]
    ) -> pd.DataFrame:
        """Build predictions DataFrame from all target results."""
        rows = []
        for target_result in results.values():
            n = len(target_result.region_ids)
            for i in range(n):
                rows.append(
                    {
                        "approach": self.approach,
                        "target_variable": target_result.target,
                        "region_id": str(target_result.region_ids[i]),
                        "y_true": float(target_result.actual_values[i]),
                        "y_pred": float(target_result.oof_predictions[i]),
                    }
                )

        return pd.DataFrame(
            rows,
            columns=[
                "approach",
                "target_variable",
                "region_id",
                "y_true",
                "y_pred",
            ],
        )

    def _build_metrics(
        self, results: Dict[str, "TargetResult"]
    ) -> pd.DataFrame:
        """Build metrics DataFrame from all target results."""
        rows = []
        for target_result in results.values():
            # Always emit regression metrics
            for metric_name, attr in [
                ("r2", "overall_r2"),
                ("rmse", "overall_rmse"),
                ("mae", "overall_mae"),
            ]:
                value = getattr(target_result, attr, None)
                if value is not None:
                    rows.append(
                        {
                            "approach": self.approach,
                            "target_variable": target_result.target,
                            "metric": metric_name,
                            "value": float(value),
                        }
                    )

            # Classification metrics when applicable
            if target_result.task_type == "classification":
                for metric_name, attr in [
                    ("accuracy", "overall_accuracy"),
                    ("f1_macro", "overall_f1_macro"),
                ]:
                    value = getattr(target_result, attr, None)
                    if value is not None:
                        rows.append(
                            {
                                "approach": self.approach,
                                "target_variable": target_result.target,
                                "metric": metric_name,
                                "value": float(value),
                            }
                        )

        return pd.DataFrame(
            rows,
            columns=["approach", "target_variable", "metric", "value"],
        )
