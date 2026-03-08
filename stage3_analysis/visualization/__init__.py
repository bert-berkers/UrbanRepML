"""
Visualization subpackage for stage3_analysis.

Provides clustering utilities (PCA, MiniBatchKMeans) for embedding analysis.
Dissolve-based rendering has been archived to scripts/archive/visualization/cluster_viz_dissolve.py.
Current approach uses rasterized centroid rendering (see scripts/plot_embeddings.py).
"""

from .clustering_utils import (
    apply_pca_reduction,
    perform_minibatch_clustering,
)

__all__ = [
    'apply_pca_reduction',
    'perform_minibatch_clustering',
]
