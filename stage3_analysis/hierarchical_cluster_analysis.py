#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hierarchical Cluster Analysis for Multi-Scale Spatial Embeddings

Performs beautiful cluster analysis across H3 resolutions 5-11, incorporating
topographical gradients, POI utilities, and holographic distance patterns
to reveal spatial patterns at multiple scales.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
# UMAP requires separate installation: pip install umap-learn
try:
    import umap
except ImportError:
    umap = None
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Low-level H3 hierarchy traversal (cell_to_parent) not wrapped by SRAI
import h3 as _h3
import warnings
warnings.filterwarnings('ignore')

from utils import StudyAreaPaths
from utils.paths import write_run_info

logger = logging.getLogger(__name__)


@dataclass
class ClusterMetrics:
    """Metrics for evaluating cluster quality"""
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    inertia: Optional[float]
    n_clusters: int
    n_samples: int
    cluster_sizes: List[int]


@dataclass
class HierarchicalClusterResult:
    """Results from hierarchical kmeans_clustering_1layer analysis"""
    resolution: int
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    metrics: ClusterMetrics
    feature_importance: Dict[str, float]
    spatial_coherence: float
    cross_resolution_consistency: Optional[float]


class HierarchicalClusterAnalyzer:
    """
    Analyzes spatial patterns through kmeans_clustering_1layer across multiple H3 resolutions.
    Reveals beautiful multi-scale spatial structures.
    """

    def __init__(
        self,
        resolutions: List[int] = [5, 6, 7, 8, 9, 10, 11],
        primary_resolution: int = 8,
        clustering_methods: List[str] = ['kmeans', 'gaussian_mixture', 'hierarchical'],
        paths: Optional[StudyAreaPaths] = None,
        run_descriptor: str = "default",
    ):
        self.resolutions = sorted(resolutions)
        self.primary_resolution = primary_resolution
        self.clustering_methods = clustering_methods
        self.paths = paths
        self.run_id: Optional[str] = None

        if paths is not None and run_descriptor:
            self.run_id = paths.create_run_id(run_descriptor)

        # Storage for results [old 2024]
        self.hierarchical_embeddings = {}
        self.cluster_results = {}
        self.cross_resolution_mappings = {}
        self.spatial_coherence_maps = {}

        logger.info(f"Initialized hierarchical cluster analyzer: resolutions {resolutions}")

    def load_hierarchical_embeddings(
        self,
        embeddings: Dict[int, pd.DataFrame]
    ) -> None:
        """
        Load hierarchical embeddings from the spatial embedding system.

        Args:
            embeddings: Dictionary mapping resolution to embedding DataFrames
        """
        self.hierarchical_embeddings = embeddings
        logger.info(f"Loaded embeddings for resolutions: {list(embeddings.keys())}")

        # Analyze embedding characteristics
        for res, df in embeddings.items():
            logger.info(f"  Resolution {res}: {len(df)} cells, {len(df.columns)} features")

    def prepare_clustering_features(
        self,
        resolution: int,
        feature_categories: Optional[List[str]] = None,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for kmeans_clustering_1layer analysis.

        Args:
            resolution: H3 resolution to analyze
            feature_categories: Categories of features to include
            scale_features: Whether to standardize features

        Returns:
            Feature matrix and feature names
        """
        if resolution not in self.hierarchical_embeddings:
            raise ValueError(f"No embeddings available for resolution {resolution}")

        df = self.hierarchical_embeddings[resolution]

        # Default feature categories
        if feature_categories is None:
            feature_categories = [
                'embedding',      # Embedding columns (e.g. A00-A63, emb_0-emb_N)
                'topographical',  # elevation, slope, aspect, curvature
                'poi',           # POI utility features
                'distance',      # Distance-based features
                'accessibility'  # Accessibility cost features
            ]

        # Select features based on categories
        selected_features = []
        feature_names = []

        for category in feature_categories:
            if category == 'embedding':
                # Try known prefix patterns, fall back to emb_ prefix
                emb_cols = [col for col in df.columns if col.startswith(('A', 'P', 'R', 'S', 'G')) and len(col) >= 2 and col[1:].isdigit()]
                if not emb_cols:
                    emb_cols = [col for col in df.columns if col.startswith('emb_')]
                selected_features.extend(emb_cols)
                feature_names.extend(emb_cols)

            elif category == 'topographical':
                topo_cols = [col for col in df.columns if any(
                    keyword in col.lower() for keyword in ['elevation', 'slope', 'aspect', 'curvature']
                )]
                selected_features.extend(topo_cols)
                feature_names.extend(topo_cols)

            elif category == 'poi':
                poi_cols = [col for col in df.columns if 'poi' in col.lower()]
                selected_features.extend(poi_cols)
                feature_names.extend(poi_cols)

            elif category == 'distance':
                dist_cols = [col for col in df.columns if 'distance' in col.lower()]
                selected_features.extend(dist_cols)
                feature_names.extend(dist_cols)

            elif category == 'accessibility':
                access_cols = [col for col in df.columns if 'accessibility' in col.lower()]
                selected_features.extend(access_cols)
                feature_names.extend(access_cols)

        # Remove duplicates while preserving order
        unique_features = []
        unique_names = []
        for feat, name in zip(selected_features, feature_names):
            if feat not in unique_features:
                unique_features.append(feat)
                unique_names.append(name)

        # Extract feature matrix
        feature_matrix = df[unique_features].values

        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)

        logger.info(f"Prepared {len(unique_features)} features for resolution {resolution} kmeans_clustering_1layer")

        return feature_matrix, unique_names

    def find_optimal_clusters(
        self,
        features: np.ndarray,
        method: str = 'kmeans',
        max_clusters: int = 20,
        min_clusters: int = 2
    ) -> Tuple[int, float]:
        """
        Find optimal number of clusters using multiple criteria.

        Args:
            features: Feature matrix
            method: Clustering method
            max_clusters: Maximum number of clusters to test
            min_clusters: Minimum number of clusters to test

        Returns:
            Optimal number of clusters and best score
        """
        n_samples = features.shape[0]
        max_clusters = min(max_clusters, n_samples // 2)  # Reasonable upper bound

        scores = []
        cluster_range = range(min_clusters, max_clusters + 1)

        for n_clusters in cluster_range:
            try:
                if method == 'kmeans':
                    clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = clustering.fit_predict(features)
                elif method == 'gaussian_mixture':
                    clustering = GaussianMixture(n_components=n_clusters, random_state=42)
                    labels = clustering.fit_predict(features)
                elif method == 'hierarchical':
                    clustering = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = clustering.fit_predict(features)
                else:
                    raise ValueError(f"Unknown kmeans_clustering_1layer method: {method}")

                # Calculate silhouette score
                if len(set(labels)) > 1:  # Need at least 2 clusters
                    score = silhouette_score(features, labels)
                    scores.append(score)
                else:
                    scores.append(-1.0)

            except Exception as e:
                logger.warning(f"Error kmeans_clustering_1layer with {n_clusters} clusters: {e}")
                scores.append(-1.0)

        # Find optimal number of clusters
        if scores:
            best_idx = np.argmax(scores)
            optimal_clusters = cluster_range[best_idx]
            best_score = scores[best_idx]
        else:
            optimal_clusters = min_clusters
            best_score = -1.0

        logger.info(f"Optimal clusters for {method}: {optimal_clusters} (score: {best_score:.3f})")

        return optimal_clusters, best_score

    def perform_clustering(
        self,
        resolution: int,
        method: str = 'kmeans',
        n_clusters: Optional[int] = None,
        feature_categories: Optional[List[str]] = None
    ) -> HierarchicalClusterResult:
        """
        Perform kmeans_clustering_1layer analysis for a specific resolution.

        Args:
            resolution: H3 resolution to analyze
            method: Clustering method
            n_clusters: Number of clusters (auto-detect if None)
            feature_categories: Feature categories to include

        Returns:
            Hierarchical cluster result
        """
        logger.info(f"Performing {method} kmeans_clustering_1layer for resolution {resolution}")

        # Prepare features
        features, feature_names = self.prepare_clustering_features(
            resolution, feature_categories
        )

        # Find optimal clusters if not specified
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters(features, method)

        # Perform kmeans_clustering_1layer
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clustering.fit_predict(features)
            cluster_centers = clustering.cluster_centers_
            inertia = clustering.inertia_

        elif method == 'gaussian_mixture':
            clustering = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = clustering.fit_predict(features)
            cluster_centers = clustering.means_
            inertia = None

        elif method == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(features)

            # Calculate cluster centers manually
            cluster_centers = []
            for cluster_id in range(n_clusters):
                cluster_mask = labels == cluster_id
                if np.any(cluster_mask):
                    center = features[cluster_mask].mean(axis=0)
                    cluster_centers.append(center)
            cluster_centers = np.array(cluster_centers)
            inertia = None

        elif method == 'dbscan':
            # DBSCAN doesn't require n_clusters
            clustering = DBSCAN(eps=0.5, min_samples=5)
            labels = clustering.fit_predict(features)

            # Calculate cluster centers for non-noise points
            unique_labels = set(labels)
            if -1 in unique_labels:  # Remove noise label
                unique_labels.remove(-1)

            cluster_centers = []
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                center = features[cluster_mask].mean(axis=0)
                cluster_centers.append(center)

            cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([[]])
            n_clusters = len(unique_labels)
            inertia = None

        else:
            raise ValueError(f"Unknown kmeans_clustering_1layer method: {method}")

        # Calculate kmeans_clustering_1layer metrics
        metrics = self._calculate_cluster_metrics(features, labels, inertia)

        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(
            features, labels, feature_names
        )

        # Calculate spatial coherence
        spatial_coherence = self._calculate_spatial_coherence(
            resolution, labels
        )

        # Calculate cross-resolution consistency if possible
        cross_resolution_consistency = self._calculate_cross_resolution_consistency(
            resolution, labels
        )

        result = HierarchicalClusterResult(
            resolution=resolution,
            cluster_labels=labels,
            cluster_centers=cluster_centers,
            metrics=metrics,
            feature_importance=feature_importance,
            spatial_coherence=spatial_coherence,
            cross_resolution_consistency=cross_resolution_consistency
        )

        # Store result
        if resolution not in self.cluster_results:
            self.cluster_results[resolution] = {}
        self.cluster_results[resolution][method] = result

        logger.info(f"Clustering complete: {n_clusters} clusters, silhouette score {metrics.silhouette_score:.3f}")

        return result

    def _calculate_cluster_metrics(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        inertia: Optional[float]
    ) -> ClusterMetrics:
        """Calculate comprehensive kmeans_clustering_1layer metrics."""

        n_clusters = len(set(labels))
        if -1 in labels:  # DBSCAN noise
            n_clusters -= 1

        n_samples = len(labels)
        cluster_sizes = [np.sum(labels == i) for i in set(labels) if i != -1]

        # Silhouette score
        if n_clusters > 1 and n_samples > n_clusters:
            silhouette = silhouette_score(features, labels)
        else:
            silhouette = -1.0

        # Calinski-Harabasz score
        if n_clusters > 1:
            calinski_harabasz = calinski_harabasz_score(features, labels)
        else:
            calinski_harabasz = 0.0

        # Davies-Bouldin score
        if n_clusters > 1:
            davies_bouldin = davies_bouldin_score(features, labels)
        else:
            davies_bouldin = 0.0

        return ClusterMetrics(
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            inertia=inertia,
            n_clusters=n_clusters,
            n_samples=n_samples,
            cluster_sizes=cluster_sizes
        )

    def _calculate_feature_importance(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance for kmeans_clustering_1layer."""

        # Use variance ratio as proxy for feature importance
        n_clusters = len(set(labels))
        if n_clusters <= 1:
            return {name: 0.0 for name in feature_names}

        importance_scores = {}

        for i, feature_name in enumerate(feature_names):
            feature_values = features[:, i]

            # Calculate between-cluster variance
            cluster_means = []
            for cluster_id in set(labels):
                if cluster_id != -1:  # Skip noise in DBSCAN
                    cluster_mask = labels == cluster_id
                    if np.any(cluster_mask):
                        cluster_mean = feature_values[cluster_mask].mean()
                        cluster_means.append(cluster_mean)

            if len(cluster_means) > 1:
                between_cluster_variance = np.var(cluster_means)
                within_cluster_variance = np.var(feature_values)

                if within_cluster_variance > 0:
                    importance = between_cluster_variance / within_cluster_variance
                else:
                    importance = 0.0
            else:
                importance = 0.0

            importance_scores[feature_name] = importance

        return importance_scores

    def _calculate_spatial_coherence(
        self,
        resolution: int,
        labels: np.ndarray
    ) -> float:
        """Calculate spatial coherence of clusters."""

        if resolution not in self.hierarchical_embeddings:
            return 0.0

        df = self.hierarchical_embeddings[resolution]
        h3_cells = df.index.tolist()

        coherence_scores = []

        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise
                continue

            cluster_mask = labels == cluster_id
            cluster_cells = [h3_cells[i] for i, mask in enumerate(cluster_mask) if mask]

            if len(cluster_cells) <= 1:
                continue

            # Calculate spatial compactness
            # Count how many cluster cells are neighbors of other cluster cells
            neighbor_connections = 0
            total_possible_connections = 0

            neighbourhood = H3Neighbourhood()
            for cell in cluster_cells:
                neighbors = neighbourhood.get_neighbours(cell)

                cell_neighbors_in_cluster = len(neighbors.intersection(set(cluster_cells)))
                neighbor_connections += cell_neighbors_in_cluster
                total_possible_connections += len(neighbors)

            if total_possible_connections > 0:
                cluster_coherence = neighbor_connections / total_possible_connections
                coherence_scores.append(cluster_coherence)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _calculate_cross_resolution_consistency(
        self,
        resolution: int,
        labels: np.ndarray
    ) -> Optional[float]:
        """Calculate consistency with kmeans_clustering_1layer at parent resolution."""

        parent_resolution = resolution - 1

        if (parent_resolution not in self.cluster_results or
            resolution not in self.hierarchical_embeddings or
            parent_resolution not in self.hierarchical_embeddings):
            return None

        # Get parent kmeans_clustering_1layer results [old 2024]
        parent_results = self.cluster_results[parent_resolution]
        if not parent_results:
            return None

        # Use first available kmeans_clustering_1layer method for parent
        parent_method = list(parent_results.keys())[0]
        parent_labels = parent_results[parent_method].cluster_labels

        # Map current cells to parent cells
        df = self.hierarchical_embeddings[resolution]
        parent_df = self.hierarchical_embeddings[parent_resolution]

        h3_cells = df.index.tolist()
        parent_h3_cells = parent_df.index.tolist()

        consistency_scores = []

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_mask = labels == cluster_id
            cluster_cells = [h3_cells[i] for i, mask in enumerate(cluster_mask) if mask]

            # Map to parent cells
            parent_cells = []
            for cell in cluster_cells:
                parent_cell = _h3.cell_to_parent(cell, parent_resolution)
                if parent_cell in parent_h3_cells:
                    parent_idx = parent_h3_cells.index(parent_cell)
                    parent_cluster = parent_labels[parent_idx]
                    parent_cells.append(parent_cluster)

            if parent_cells:
                # Calculate consistency as modal parent cluster proportion
                unique_parents, counts = np.unique(parent_cells, return_counts=True)
                max_consistency = np.max(counts) / len(parent_cells)
                consistency_scores.append(max_consistency)

        return np.mean(consistency_scores) if consistency_scores else 0.0

    def analyze_all_resolutions(
        self,
        methods: Optional[List[str]] = None,
        feature_categories: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, HierarchicalClusterResult]]:
        """
        Perform kmeans_clustering_1layer analysis across all resolutions.

        Args:
            methods: Clustering methods to use
            feature_categories: Feature categories to include

        Returns:
            Nested dictionary: resolution -> method -> results [old 2024]
        """
        if methods is None:
            methods = self.clustering_methods

        logger.info(f"Analyzing kmeans_clustering_1layer across {len(self.resolutions)} resolutions with methods: {methods}")

        results = {}

        for resolution in self.resolutions:
            if resolution not in self.hierarchical_embeddings:
                logger.warning(f"No embeddings for resolution {resolution}, skipping")
                continue

            logger.info(f"Processing resolution {resolution}")
            results[resolution] = {}

            for method in methods:
                try:
                    result = self.perform_clustering(
                        resolution, method, feature_categories=feature_categories
                    )
                    results[resolution][method] = result
                except Exception as e:
                    logger.error(f"Error kmeans_clustering_1layer resolution {resolution} with {method}: {e}")

        return results

    def create_cluster_summary(self) -> pd.DataFrame:
        """Create summary table of kmeans_clustering_1layer results [old 2024] across all resolutions."""

        summary_rows = []

        for resolution in self.resolutions:
            if resolution not in self.cluster_results:
                continue

            for method, result in self.cluster_results[resolution].items():
                row = {
                    'resolution': resolution,
                    'method': method,
                    'n_clusters': result.metrics.n_clusters,
                    'n_samples': result.metrics.n_samples,
                    'silhouette_score': result.metrics.silhouette_score,
                    'calinski_harabasz_score': result.metrics.calinski_harabasz_score,
                    'davies_bouldin_score': result.metrics.davies_bouldin_score,
                    'spatial_coherence': result.spatial_coherence,
                    'cross_resolution_consistency': result.cross_resolution_consistency or 0.0,
                    'mean_cluster_size': np.mean(result.metrics.cluster_sizes),
                    'cluster_size_std': np.std(result.metrics.cluster_sizes)
                }
                summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        if not summary_df.empty:
            # Sort by resolution and method
            summary_df = summary_df.sort_values(['resolution', 'method'])

        return summary_df

    def get_best_clustering_per_resolution(self) -> Dict[int, HierarchicalClusterResult]:
        """Get the best kmeans_clustering_1layer result for each resolution based on silhouette score."""

        best_results = {}

        for resolution in self.resolutions:
            if resolution not in self.cluster_results:
                continue

            best_score = -1.0
            best_result = None

            for method, result in self.cluster_results[resolution].items():
                if result.metrics.silhouette_score > best_score:
                    best_score = result.metrics.silhouette_score
                    best_result = result

            if best_result is not None:
                best_results[resolution] = best_result

        return best_results

    def save_cluster_assignments(self, output_dir: Optional[Path] = None) -> None:
        """Save cluster assignments for each resolution and method.

        Args:
            output_dir: Directory for output files. When None and ``paths``
                plus ``run_id`` are set, defaults to a dated run directory
                under ``stage3_analysis/hierarchical_clustering/{run_id}/``.
        """
        if output_dir is None and self.paths is not None and self.run_id is not None:
            output_dir = self.paths.stage3_run("hierarchical_clustering", self.run_id)
        elif output_dir is None:
            raise ValueError("output_dir is required when paths/run_id are not set")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for resolution in self.cluster_results:
            for method, result in self.cluster_results[resolution].items():
                # Get corresponding embedding dataframe
                if resolution in self.hierarchical_embeddings:
                    df = self.hierarchical_embeddings[resolution].copy()
                    df['cluster_id'] = result.cluster_labels

                    # Add cluster statistics
                    for cluster_id in set(result.cluster_labels):
                        if cluster_id != -1:
                            cluster_mask = result.cluster_labels == cluster_id
                            cluster_size = np.sum(cluster_mask)
                            df.loc[cluster_mask, 'cluster_size'] = cluster_size

                    # Save to file
                    filename = f"clusters_res{resolution}_{method}.parquet"
                    filepath = output_dir / filename
                    df.to_parquet(filepath)

                    logger.info(f"Saved cluster assignments: {filepath}")

        # Write run-level provenance when using a run directory
        if self.paths is not None and self.run_id is not None:
            write_run_info(
                output_dir,
                stage="stage3",
                study_area=self.paths.study_area,
                config={
                    "resolutions": self.resolutions,
                    "primary_resolution": self.primary_resolution,
                    "clustering_methods": self.clustering_methods,
                },
            )
            logger.info(f"Saved run_info.json to {output_dir / 'run_info.json'}")


def main():
    """Demonstration of hierarchical cluster analysis."""

    # This would typically be called after building hierarchical embeddings
    logger.info("Hierarchical Cluster Analysis System Ready!")
    logger.info("Use HierarchicalClusterAnalyzer.analyze_all_resolutions() to process embeddings")

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
