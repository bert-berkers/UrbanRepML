#!/usr/bin/env python
"""
Aggregate AlphaEarth Embeddings from Res10 to Coarser Resolutions (Res5-9)
===========================================================================

Uses H3 parent-child hierarchy to aggregate fine-resolution embeddings
to coarser resolutions via mean pooling.

This enables multi-resolution hierarchical cone training by providing
embeddings at all resolutions from res5 (coarse) to res10 (fine).

Usage:
    python scripts/processing_modalities/alphaearth/aggregate_embeddings_hierarchical.py

Outputs:
    data/study_areas/netherlands/embeddings/alphaearth/netherlands_res{5-9}_2022.parquet
"""

import sys
from pathlib import Path
import logging
import time
from typing import Dict

import h3  # SRAI dependency for parent-child operations
import pandas as pd
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/aggregate_embeddings.log')
    ]
)
logger = logging.getLogger(__name__)


def aggregate_to_parent_resolution(
    embeddings_df: pd.DataFrame,
    source_resolution: int,
    target_resolution: int
) -> pd.DataFrame:
    """
    Aggregate embeddings from source resolution to coarser target resolution.

    Args:
        embeddings_df: DataFrame with h3_index and embedding columns
        source_resolution: Current H3 resolution (e.g., 10)
        target_resolution: Target H3 resolution (e.g., 9)

    Returns:
        DataFrame with aggregated embeddings at target resolution
    """
    logger.info(f"Aggregating res{source_resolution} to res{target_resolution}")

    # Get embedding columns (A00-A63)
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('A')]
    logger.info(f"  Found {len(embedding_cols)} embedding bands")

    # Create h3_index column if it's the index
    if 'h3_index' not in embeddings_df.columns:
        embeddings_df = embeddings_df.reset_index()

    # Get parent hexagons for each child
    logger.info("  Computing parent hexagons...")
    parents = []
    for h3_idx in tqdm(embeddings_df['h3_index'], desc=f"  Res{source_resolution} to {target_resolution}"):
        # Get parent at target resolution
        parent = h3_idx
        current_res = h3.get_resolution(h3_idx)

        # Traverse up the hierarchy to target resolution
        while current_res > target_resolution:
            parent = h3.cell_to_parent(parent, current_res - 1)
            current_res -= 1

        parents.append(parent)

    embeddings_df['parent_h3'] = parents

    # Aggregate embeddings by parent (mean pooling)
    logger.info("  Aggregating embeddings (mean pooling)...")

    # Group by parent and compute mean for embedding columns
    aggregated = embeddings_df.groupby('parent_h3')[embedding_cols].mean()

    # Rename index to h3_index
    aggregated.index.name = 'h3_index'
    aggregated = aggregated.reset_index()

    # Add metadata columns
    aggregated['pixel_count'] = embeddings_df.groupby('parent_h3')['pixel_count'].sum().values
    aggregated['tile_count'] = embeddings_df.groupby('parent_h3').size().values
    aggregated['h3_resolution'] = target_resolution

    logger.info(f"  Result: {len(aggregated):,} hexagons at res{target_resolution}")

    return aggregated


def generate_hierarchical_embeddings(
    study_area: str = "netherlands",
    base_resolution: int = 10,
    year: str = "2022"
):
    """
    Generate embeddings for all resolutions from base down to res5.

    Args:
        study_area: Name of study area
        base_resolution: Finest resolution with existing embeddings (10)
        year: Year of data (2022)
    """
    logger.info("="*70)
    logger.info("HIERARCHICAL EMBEDDING AGGREGATION")
    logger.info("="*70)
    logger.info(f"Study Area: {study_area}")
    logger.info(f"Base Resolution: {base_resolution}")
    logger.info(f"Target Resolutions: 5-{base_resolution-1}")
    logger.info(f"Year: {year}")

    start_time = time.time()

    # Setup paths
    embeddings_dir = Path(f"data/study_areas/{study_area}/embeddings/alphaearth")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load base resolution embeddings
    base_file = embeddings_dir / f"{study_area}_res{base_resolution}_{year}.parquet"

    if not base_file.exists():
        logger.error(f"Base embeddings not found: {base_file}")
        raise FileNotFoundError(f"Base embeddings not found: {base_file}")

    logger.info(f"\nLoading base embeddings: {base_file}")
    current_embeddings = pd.read_parquet(base_file)
    logger.info(f"  Loaded: {len(current_embeddings):,} hexagons")
    logger.info(f"  Columns: {current_embeddings.columns.tolist()[:10]}...")

    # Aggregate from base down to res5
    current_resolution = base_resolution

    for target_resolution in range(base_resolution - 1, 4, -1):  # 9, 8, 7, 6, 5
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Resolution {target_resolution}")
        logger.info(f"{'='*70}")

        # Check if target already exists
        target_file = embeddings_dir / f"{study_area}_res{target_resolution}_{year}.parquet"

        if target_file.exists():
            logger.info(f"✓ Already exists: {target_file}")
            logger.info(f"  Loading for next aggregation...")
            current_embeddings = pd.read_parquet(target_file)
            current_resolution = target_resolution
            continue

        # Aggregate from current to target
        aggregated = aggregate_to_parent_resolution(
            current_embeddings,
            current_resolution,
            target_resolution
        )

        # Save result
        logger.info(f"  Saving to: {target_file}")
        aggregated.to_parquet(target_file, index=False)

        # Log file size
        file_size = target_file.stat().st_size / (1024**2)  # MB
        logger.info(f"  Saved: {file_size:.1f} MB")

        # Update current for next iteration
        current_embeddings = aggregated
        current_resolution = target_resolution

    # Summary
    end_time = time.time()
    duration = end_time - start_time

    logger.info("\n" + "="*70)
    logger.info("AGGREGATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total time: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # List all generated files
    logger.info("\nGenerated embeddings:")
    for res in range(5, base_resolution + 1):
        res_file = embeddings_dir / f"{study_area}_res{res}_{year}.parquet"
        if res_file.exists():
            size = res_file.stat().st_size / (1024**2)
            df = pd.read_parquet(res_file)
            logger.info(f"  res{res}: {len(df):,} hexagons, {size:.1f} MB")


def verify_embeddings(study_area: str = "netherlands", year: str = "2022"):
    """Verify that all resolution embeddings exist and are valid."""
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION")
    logger.info("="*70)

    embeddings_dir = Path(f"data/study_areas/{study_area}/embeddings/alphaearth")

    all_valid = True
    for res in range(5, 11):
        res_file = embeddings_dir / f"{study_area}_res{res}_{year}.parquet"

        if not res_file.exists():
            logger.error(f"✗ Missing: res{res}")
            all_valid = False
            continue

        try:
            df = pd.read_parquet(res_file)
            embedding_cols = [col for col in df.columns if col.startswith('A')]

            logger.info(f"✓ res{res}: {len(df):,} hexagons, {len(embedding_cols)} bands")

            # Verify h3_index
            if 'h3_index' not in df.columns:
                logger.warning(f"  Warning: 'h3_index' column not found in res{res}")

            # Verify embedding dimensions
            if len(embedding_cols) != 64:
                logger.warning(f"  Warning: Expected 64 bands, found {len(embedding_cols)}")

        except Exception as e:
            logger.error(f"✗ Error reading res{res}: {e}")
            all_valid = False

    if all_valid:
        logger.info("\n✓ All embeddings verified successfully!")
    else:
        logger.error("\n✗ Some embeddings are missing or invalid")

    return all_valid


if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    try:
        # Generate hierarchical embeddings
        generate_hierarchical_embeddings(
            study_area="netherlands",
            base_resolution=10,
            year="2022"
        )

        # Verify all embeddings
        verify_embeddings(study_area="netherlands", year="2022")

        logger.info("\n" + "="*70)
        logger.info("SUCCESS: Hierarchical embeddings ready for training!")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"\nERROR: {e}", exc_info=True)
        sys.exit(1)