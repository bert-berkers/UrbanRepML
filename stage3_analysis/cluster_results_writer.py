"""
Write standardized cluster result parquets for cross-approach comparison.

Produces two files per approach:
- assignments.parquet: per-hex, per-k cluster labels (raw MiniBatchKMeans order)
- metrics.parquet: per-k silhouette and Calinski-Harabasz scores
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from utils.paths import StudyAreaPaths


class ClusterResultsWriter:
    """Write cluster assignments and quality metrics to standardized parquets.

    Schema
    ------
    assignments.parquet columns (no index):
        approach (str), k (int), region_id (str), cluster_label (int)

    metrics.parquet columns (no index):
        approach (str), k (int), metric (str), value (float64)
    """

    def __init__(self, study_area: str, approach: str):
        self.paths = StudyAreaPaths(study_area)
        self.approach = approach

    def write(
        self,
        cluster_assignments: Dict[int, np.ndarray],
        region_ids: pd.Index,
        embeddings: np.ndarray,
    ) -> Path:
        """Write assignments.parquet and metrics.parquet.

        Args:
            cluster_assignments: {k: labels_array} from MiniBatchKMeans.
                Each labels_array has shape (n_regions,) with values 0..k-1.
            region_ids: H3 hex IDs matching the label arrays.
            embeddings: Embedding matrix (n_regions, n_dims) for metric computation.

        Returns:
            The approach directory path containing both parquets.
        """
        out_dir = self.paths.cluster_results(self.approach)
        out_dir.mkdir(parents=True, exist_ok=True)

        assignment_rows = []
        metric_rows = []

        for k, labels in sorted(cluster_assignments.items()):
            # --- assignments ---
            for rid, label in zip(region_ids, labels):
                assignment_rows.append(
                    (self.approach, k, str(rid), int(label))
                )

            # --- metrics (need >= 2 distinct labels) ---
            n_unique = len(np.unique(labels))
            if n_unique < 2:
                continue

            # Calinski-Harabasz: fast, always compute
            ch = calinski_harabasz_score(embeddings, labels)
            metric_rows.append((self.approach, k, "calinski_harabasz", float(ch)))

            # Silhouette: subsample if large
            if len(embeddings) > 50_000:
                idx = np.random.RandomState(42).choice(
                    len(embeddings), 50_000, replace=False
                )
                sil = silhouette_score(embeddings[idx], labels[idx])
            else:
                sil = silhouette_score(embeddings, labels)
            metric_rows.append((self.approach, k, "silhouette", float(sil)))

        # Write assignments
        assignments_df = pd.DataFrame(
            assignment_rows,
            columns=["approach", "k", "region_id", "cluster_label"],
        )
        assignments_df.to_parquet(out_dir / "assignments.parquet", index=False)

        # Write metrics
        metrics_df = pd.DataFrame(
            metric_rows,
            columns=["approach", "k", "metric", "value"],
        )
        metrics_df.to_parquet(out_dir / "metrics.parquet", index=False)

        return out_dir

    @classmethod
    def write_from_clustering(
        cls,
        cluster_assignments: Dict[int, np.ndarray],
        region_ids: pd.Index,
        embeddings: np.ndarray,
        approach: str,
        study_area: str,
    ) -> Path:
        """Convenience classmethod to write in one call.

        Args:
            cluster_assignments: {k: labels_array} from MiniBatchKMeans.
            region_ids: H3 hex IDs matching the label arrays.
            embeddings: Embedding matrix for metric computation.
            approach: Name of the approach (e.g. "ring_agg_k10").
            study_area: Study area name (e.g. "netherlands").

        Returns:
            The approach directory path containing both parquets.
        """
        writer = cls(study_area=study_area, approach=approach)
        return writer.write(cluster_assignments, region_ids, embeddings)
