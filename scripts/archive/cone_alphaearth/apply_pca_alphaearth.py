#!/usr/bin/env python
"""
Apply PCA to AlphaEarth Embeddings
===================================

Reduces AlphaEarth embeddings from 64 dims to 16 dims via PCA for memory efficiency.

Usage:
    python scripts/netherlands/apply_pca_alphaearth.py --n-components 16
"""

import sys
from pathlib import Path
import argparse
import logging

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_pca_to_alphaearth(
    study_area: str = "netherlands",
    resolution: int = 10,
    n_components: int = 16,
    year: int = 2022
):
    """
    Apply PCA to AlphaEarth embeddings.

    Args:
        study_area: Name of study area
        resolution: H3 resolution
        n_components: Number of PCA components to keep
        year: Year of embeddings
    """
    logger.info("=" * 80)
    logger.info("PCA Preprocessing for AlphaEarth Embeddings")
    logger.info("=" * 80)
    logger.info(f"Study Area: {study_area}")
    logger.info(f"Resolution: {resolution}")
    logger.info(f"PCA Components: {n_components}")

    # Paths
    data_dir = Path(f"data/study_areas/{study_area}")
    embeddings_path = (
        data_dir / "embeddings" / "alphaearth" /
        f"{study_area}_res{resolution}_{year}.parquet"
    )

    output_path = (
        data_dir / "embeddings" / "alphaearth" /
        f"{study_area}_res{resolution}_pca{n_components}_{year}.parquet"
    )

    pca_model_path = (
        data_dir / "embeddings" / "alphaearth" /
        f"pca_model_{n_components}components.pkl"
    )

    # Load embeddings
    logger.info(f"\nLoading embeddings from: {embeddings_path}")
    embeddings_df = pd.read_parquet(embeddings_path)

    # Extract embedding columns (A00, A01, ..., A63)
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('A')]
    logger.info(f"Found {len(embedding_cols)} embedding dimensions")
    logger.info(f"Total hexagons: {len(embeddings_df):,}")

    # Get embeddings as numpy array
    embeddings_array = embeddings_df[embedding_cols].values
    logger.info(f"Embeddings shape: {embeddings_array.shape}")

    # Handle NaN values
    if np.isnan(embeddings_array).any():
        nan_count = np.isnan(embeddings_array).sum()
        logger.warning(f"Found {nan_count} NaN values, filling with 0")
        embeddings_array = np.nan_to_num(embeddings_array, nan=0.0)

    # Fit PCA
    logger.info(f"\nFitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_array)

    # Log variance explained
    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"Variance explained: {variance_explained:.2%}")
    logger.info(f"Per-component variance:")
    for i, var in enumerate(pca.explained_variance_ratio_[:10]):  # First 10
        logger.info(f"  PC{i:02d}: {var:.2%}")
    if n_components > 10:
        logger.info(f"  ... ({n_components - 10} more components)")

    # Create output DataFrame
    pca_cols = [f"P{i:02d}" for i in range(n_components)]
    pca_df = pd.DataFrame(
        embeddings_pca,
        index=embeddings_df.index,
        columns=pca_cols
    )

    # Copy geometry if present
    if 'geometry' in embeddings_df.columns:
        pca_df['geometry'] = embeddings_df['geometry']

    # Save PCA embeddings
    logger.info(f"\nSaving PCA embeddings to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pca_df.to_parquet(output_path)
    logger.info(f"Saved {len(pca_df):,} hexagons with {n_components} dimensions")

    # Save PCA model
    logger.info(f"Saving PCA model to: {pca_model_path}")
    joblib.dump(pca, pca_model_path)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PCA Preprocessing Complete")
    logger.info("=" * 80)
    logger.info(f"Input: {embeddings_array.shape}")
    logger.info(f"Output: {embeddings_pca.shape}")
    logger.info(f"Variance retained: {variance_explained:.2%}")
    logger.info(f"Compression ratio: {len(embedding_cols) / n_components:.1f}x")
    logger.info(f"Memory reduction: {(1 - n_components/len(embedding_cols)):.1%}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Apply PCA to AlphaEarth Embeddings')
    parser.add_argument('--study-area', type=str, default='netherlands',
                        help='Study area name (default: netherlands)')
    parser.add_argument('--resolution', type=int, default=10,
                        help='H3 resolution (default: 10)')
    parser.add_argument('--n-components', type=int, default=16,
                        help='Number of PCA components (default: 16)')
    parser.add_argument('--year', type=int, default=2022,
                        help='Year of embeddings (default: 2022)')

    args = parser.parse_args()

    apply_pca_to_alphaearth(
        study_area=args.study_area,
        resolution=args.resolution,
        n_components=args.n_components,
        year=args.year
    )


if __name__ == "__main__":
    main()
