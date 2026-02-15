"""
Stage 3: Analysis Package
=========================

Analysis, kmeans_clustering_1layer, visualization, and validation utilities for urban embeddings.
"""

from .analytics import UrbanEmbeddingAnalyzer
from .hierarchical_cluster_analysis import HierarchicalClusterAnalyzer
from .hierarchical_visualization import HierarchicalLandscapeVisualizer
from .leefbaarometer_target import LeefbaarometerTargetBuilder, LeefbaarometerConfig
from .linear_probe import LinearProbeRegressor, LinearProbeConfig
from .linear_probe_viz import LinearProbeVisualizer
from .dnn_probe import DNNProbeRegressor, DNNProbeConfig
from .dnn_probe_viz import DNNProbeVisualizer

__all__ = [
    'UrbanEmbeddingAnalyzer',
    'HierarchicalClusterAnalyzer',
    'HierarchicalLandscapeVisualizer',
    'LeefbaarometerTargetBuilder',
    'LeefbaarometerConfig',
    'LinearProbeRegressor',
    'LinearProbeConfig',
    'LinearProbeVisualizer',
    'DNNProbeRegressor',
    'DNNProbeConfig',
    'DNNProbeVisualizer',
]
