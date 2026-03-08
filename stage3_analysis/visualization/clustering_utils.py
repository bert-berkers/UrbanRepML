"""
Reusable clustering utilities for stage3 analysis.

Extracted from cluster_viz.py when dissolve rendering was archived.
Contains PCA reduction, MiniBatchKMeans clustering, and study area config
used by both plot_embeddings.py and plot_cluster_maps.py.

Lifetime: durable
Stage: 3
"""

import os
import time
from typing import Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# CPU optimization
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())


def apply_pca_reduction(
    embeddings: np.ndarray, n_components: int = 16
) -> Tuple[np.ndarray, PCA]:
    """Apply PCA dimensionality reduction for computational efficiency."""
    print(f"Applying PCA: {embeddings.shape[1]}D -> {n_components}D...")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA variance retained: {variance_explained:.3f}")
    return embeddings_reduced, pca


def perform_minibatch_clustering(
    embeddings_reduced: np.ndarray,
    n_clusters_list: List[int],
    standardize: bool = False,
) -> Dict[int, np.ndarray]:
    """Apply MiniBatchKMeans clustering efficiently.

    Args:
        embeddings_reduced: Pre-processed embedding matrix.
        n_clusters_list: List of cluster counts to compute.
        standardize: Whether to standardize before clustering.

    Returns:
        Dict mapping n_clusters -> cluster label array.
    """
    print(f"MiniBatchKMeans clustering with {len(n_clusters_list)} configurations...")

    data = embeddings_reduced
    if standardize:
        data = StandardScaler().fit_transform(data)

    def cluster_single_k(k: int) -> Tuple[int, np.ndarray]:
        start_time = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=min(10000, len(data)),
            max_iter=100,
            n_init=3,
            init='k-means++',
            verbose=0,
        )
        clusters = kmeans.fit_predict(data)
        duration = time.time() - start_time
        print(f"  K={k} completed in {duration:.1f}s")
        return k, clusters

    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(cluster_single_k)(k) for k in n_clusters_list
    )

    cluster_results = {}
    for k, clusters in results:
        cluster_results[k] = clusters
        print(f"  Final K={k}: {len(set(clusters))} clusters")

    return cluster_results
