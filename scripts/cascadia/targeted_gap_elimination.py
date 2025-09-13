#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Targeted Gap Elimination for Cascadia Coastal Forests Processing
Focused approach to eliminate visible tile boundary gaps in SRAI visualizations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from scipy.spatial import cKDTree
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_actual_gaps(df: pd.DataFrame, gap_threshold_km: float = 3.0) -> List[Tuple[float, float]]:
    """
    Detect actual spatial gaps in hexagon coverage
    """
    logger.info(f"Detecting actual gaps with threshold {gap_threshold_km} km...")
    
    coords = df[['lng', 'lat']].values
    tree = cKDTree(coords)
    
    # Find areas with low spatial density (actual gaps)
    gap_threshold_deg = gap_threshold_km / 111.0
    gaps = []
    
    # Create a grid to sample potential gap locations
    min_lng, max_lng = df['lng'].min(), df['lng'].max()
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    
    # Sample grid points
    lng_samples = np.linspace(min_lng, max_lng, 50)
    lat_samples = np.linspace(min_lat, max_lat, 50)
    
    for lng in lng_samples:
        for lat in lat_samples:
            # Find nearest hexagon
            distance, _ = tree.query([lng, lat])
            
            # If distance is large, this might be a gap
            if distance > gap_threshold_deg:
                gaps.append((lng, lat))
    
    logger.info(f"Found {len(gaps)} potential gap locations")
    return gaps


def smooth_tile_boundaries(df: pd.DataFrame, 
                          smoothing_radius_km: float = 2.0,
                          strength: float = 0.2) -> pd.DataFrame:
    """
    Apply targeted smoothing to eliminate tile boundary artifacts
    """
    logger.info(f"Applying targeted boundary smoothing (radius: {smoothing_radius_km} km, strength: {strength})")
    
    # Focus only on hexagons that are likely on tile boundaries
    # These are hexagons with exactly 1 tile_count (no overlap = boundary)
    if 'tile_count' not in df.columns:
        logger.warning("No tile_count column found, skipping boundary smoothing")
        return df
    
    boundary_candidates = df[df['tile_count'] == 1].copy()
    logger.info(f"Identified {len(boundary_candidates)} boundary candidate hexagons")
    
    if len(boundary_candidates) == 0:
        return df
    
    # Create spatial index for all hexagons
    all_coords = df[['lng', 'lat']].values
    all_tree = cKDTree(all_coords)
    
    # Get embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('A')]
    smoothed_df = df.copy()
    
    search_radius_deg = smoothing_radius_km / 111.0
    smoothed_count = 0
    
    for idx, boundary_hex in boundary_candidates.iterrows():
        # Find all neighbors within smoothing radius
        neighbors = all_tree.query_ball_point([boundary_hex['lng'], boundary_hex['lat']], 
                                            search_radius_deg)
        
        # Filter out self and require at least 3 neighbors
        neighbors = [n for n in neighbors if all_coords[n][0] != boundary_hex['lng'] or 
                    all_coords[n][1] != boundary_hex['lat']]
        
        if len(neighbors) >= 3:
            # Calculate inverse distance weights
            neighbor_coords = all_coords[neighbors]
            boundary_coord = np.array([boundary_hex['lng'], boundary_hex['lat']])
            distances = np.linalg.norm(neighbor_coords - boundary_coord, axis=1)
            
            weights = 1.0 / (distances + 1e-8)
            weights = weights / np.sum(weights)
            
            # Get original embedding
            original_embedding = boundary_hex[embedding_cols].values
            
            # Calculate smoothed embedding from neighbors
            neighbor_embeddings = df.iloc[neighbors][embedding_cols].values
            smoothed_embedding = np.average(neighbor_embeddings, weights=weights, axis=0)
            
            # Apply gentle blending (preserve most of original)
            blended_embedding = ((1 - strength) * original_embedding + 
                               strength * smoothed_embedding)
            
            # Update the dataframe
            df_idx = df[df['h3_index'] == boundary_hex['h3_index']].index[0]
            for i, col in enumerate(embedding_cols):
                smoothed_df.at[df_idx, col] = blended_embedding[i]
                
            smoothed_count += 1
    
    logger.info(f"Applied boundary smoothing to {smoothed_count} hexagons")
    return smoothed_df


def fill_small_gaps(df: pd.DataFrame, max_gap_km: float = 1.5) -> pd.DataFrame:
    """
    Fill small gaps between hexagons using interpolation
    """
    logger.info(f"Filling small gaps up to {max_gap_km} km...")
    
    original_count = len(df)
    
    # Detect small gaps in the hexagon grid
    gaps = detect_actual_gaps(df, gap_threshold_km=max_gap_km)
    
    if not gaps:
        logger.info("No gaps detected to fill")
        return df
    
    # For each gap, try to interpolate from nearby hexagons
    coords = df[['lng', 'lat']].values
    tree = cKDTree(coords)
    embedding_cols = [col for col in df.columns if col.startswith('A')]
    
    filled_hexagons = []
    max_interpolation_distance_deg = max_gap_km / 111.0
    
    for gap_lng, gap_lat in gaps:
        # Find nearest hexagons for interpolation
        distances, indices = tree.query([gap_lng, gap_lat], k=6)
        
        # Only use hexagons within reasonable distance
        valid_mask = distances <= max_interpolation_distance_deg
        if np.sum(valid_mask) < 3:  # Need at least 3 neighbors
            continue
            
        valid_distances = distances[valid_mask]
        valid_indices = indices[valid_mask]
        
        # Inverse distance weighting
        weights = 1.0 / (valid_distances + 1e-8)
        weights = weights / np.sum(weights)
        
        # Interpolate embedding values
        neighbor_embeddings = df.iloc[valid_indices][embedding_cols].values
        interpolated_embedding = np.average(neighbor_embeddings, weights=weights, axis=0)
        
        # Create a pseudo H3 index for the gap (simplified approach)
        gap_h3_index = f"gap_{len(filled_hexagons):06d}"
        
        gap_record = {
            'h3_index': gap_h3_index,
            'lat': gap_lat,
            'lng': gap_lng,
            'tile_count': 0  # Mark as interpolated
        }
        
        # Add embedding values
        for i, col in enumerate(embedding_cols):
            gap_record[col] = interpolated_embedding[i]
            
        filled_hexagons.append(gap_record)
    
    if filled_hexagons:
        filled_df = pd.DataFrame(filled_hexagons)
        df = pd.concat([df, filled_df], ignore_index=True)
        logger.info(f"Filled {len(filled_hexagons)} gaps, total hexagons: {len(df)} (was {original_count})")
    
    return df


def apply_targeted_gap_elimination(df: pd.DataFrame, 
                                  config: dict = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Apply targeted gap elimination focused on visible tile boundary issues
    """
    if config is None:
        config = {
            'smoothing': {
                'smoothing_radius_km': 2.0,
                'smoothing_strength': 0.2,
                'max_gap_fill_km': 1.5,
                'enable_interpolation': True
            }
        }
    
    smoothing_config = config.get('smoothing', {})
    
    logger.info("="*60)
    logger.info("TARGETED GAP ELIMINATION")
    logger.info("="*60)
    
    original_count = len(df)
    logger.info(f"Input hexagons: {original_count:,}")
    
    # Step 1: Smooth tile boundaries (gentle)
    df_smoothed = smooth_tile_boundaries(
        df, 
        smoothing_radius_km=smoothing_config.get('smoothing_radius_km', 2.0),
        strength=smoothing_config.get('smoothing_strength', 0.2)
    )
    
    # Step 2: Fill small gaps if enabled
    if smoothing_config.get('enable_interpolation', True):
        df_filled = fill_small_gaps(
            df_smoothed,
            max_gap_km=smoothing_config.get('max_gap_fill_km', 1.5)
        )
    else:
        df_filled = df_smoothed
    
    # Compute quality metrics
    quality_metrics = {
        'original_count': original_count,
        'final_count': len(df_filled),
        'boundary_hexagons': len(df[df['tile_count'] == 1]) if 'tile_count' in df.columns else 0,
        'filled_gaps': len(df_filled) - original_count,
        'processing_success': True
    }
    
    logger.info("="*60)
    logger.info("TARGETED GAP ELIMINATION COMPLETE")
    logger.info(f"Final hexagon count: {len(df_filled):,}")
    logger.info(f"Gaps filled: {quality_metrics['filled_gaps']:,}")
    logger.info("="*60)
    
    return df_filled, quality_metrics


if __name__ == "__main__":
    # Example usage
    import yaml
    
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    print("Targeted gap elimination utilities loaded successfully")
    print("Use apply_targeted_gap_elimination(df, config) to process data")