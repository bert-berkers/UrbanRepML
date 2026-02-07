"""
Visualization subpackage for stage3_analysis.

Provides fast cluster visualization using dissolve + MiniBatchKMeans + datashader.
"""

from .cluster_viz import (
    load_and_prepare_embeddings,
    apply_pca_reduction,
    perform_minibatch_clustering,
    create_cluster_visualization,
    create_hierarchical_subplot,
    add_coordinate_grid_and_labels,
    STUDY_AREA_CONFIG,
)

__all__ = [
    'load_and_prepare_embeddings',
    'apply_pca_reduction',
    'perform_minibatch_clustering',
    'create_cluster_visualization',
    'create_hierarchical_subplot',
    'add_coordinate_grid_and_labels',
    'STUDY_AREA_CONFIG',
]
