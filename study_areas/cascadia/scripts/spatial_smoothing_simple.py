#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified Spatial Smoothing for Cascadia Coastal Forests Processing
Eliminates tile boundary gaps using geographic distance-based smoothing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class SimpleSpatialSmoother:
    """
    Simplified geographic continuity-aware spatial smoothing for H3 hexagon data
    Uses pure distance-based smoothing without H3 API dependencies
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Smoothing parameters from config
        smoothing_config = config.get('smoothing', {})
        self.boundary_buffer_km = smoothing_config.get('boundary_buffer_km', 2.0)
        self.smoothing_radius_km = smoothing_config.get('smoothing_radius_km', 1.5) 
        self.min_neighbors = smoothing_config.get('min_neighbors', 3)
        self.coverage_threshold = smoothing_config.get('coverage_threshold', 0.5)
        self.smoothing_strength = smoothing_config.get('smoothing_strength', 0.3)
        
        logger.info(f"Simple spatial smoother initialized:")
        logger.info(f"  Boundary buffer: {self.boundary_buffer_km} km")
        logger.info(f"  Smoothing radius: {self.smoothing_radius_km} km") 
        logger.info(f"  Min neighbors: {self.min_neighbors}")
        logger.info(f"  Coverage threshold: {self.coverage_threshold}")

    def detect_boundary_hexagons(self, df: pd.DataFrame) -> Set[str]:
        """
        Detect hexagons likely to be on tile boundaries based on spatial density
        """
        logger.info("Detecting boundary hexagons using spatial density...")
        
        boundary_hexagons = set()
        
        # Method 1: Low tile count (sparse coverage suggests boundary)
        if 'tile_count' in df.columns:
            low_coverage = df[df['tile_count'] == 1]['h3_index'].tolist()
            boundary_hexagons.update(low_coverage)
            logger.debug(f"Found {len(low_coverage)} single-tile hexagons")
        
        # Method 2: Geographic edge detection using spatial density
        coords = df[['lng', 'lat']].values
        tree = cKDTree(coords)
        
        # For each hexagon, count neighbors within smoothing radius
        neighbor_counts = []
        search_radius_deg = self.smoothing_radius_km / 111.0  # Approximate deg per km
        
        for i, (lng, lat) in enumerate(coords):
            neighbors = tree.query_ball_point([lng, lat], search_radius_deg)
            neighbor_counts.append(len(neighbors) - 1)  # Exclude self
            
        # Hexagons with few neighbors are likely on boundaries
        neighbor_counts = np.array(neighbor_counts)
        low_density_threshold = np.percentile(neighbor_counts, 30)  # Bottom 30%
        
        low_density_indices = np.where(neighbor_counts <= low_density_threshold)[0]
        low_density_hexagons = df.iloc[low_density_indices]['h3_index'].tolist()
        boundary_hexagons.update(low_density_hexagons)
        
        logger.info(f"Detected {len(boundary_hexagons)} boundary hexagons")
        return boundary_hexagons

    def smooth_boundary_hexagons(self, df: pd.DataFrame, 
                                boundary_hexagons: Set[str]) -> pd.DataFrame:
        """
        Apply spatial smoothing to boundary hexagons to eliminate discontinuities
        """
        logger.info(f"Applying spatial smoothing to {len(boundary_hexagons)} boundary hexagons...")
        
        if not boundary_hexagons:
            return df
            
        # Create spatial index
        coords = df[['lng', 'lat']].values
        tree = cKDTree(coords)
        hex_to_idx = {h3_idx: i for i, h3_idx in enumerate(df['h3_index'])}
        
        # Get embedding columns
        embedding_cols = [col for col in df.columns if col.startswith('A')]
        smoothed_df = df.copy()
        
        search_radius_deg = self.smoothing_radius_km / 111.0
        
        smoothed_count = 0
        for h3_idx in boundary_hexagons:
            if h3_idx not in hex_to_idx:
                continue
                
            hex_idx = hex_to_idx[h3_idx]
            hex_row = df.iloc[hex_idx]
            
            # Find neighbors within smoothing radius
            neighbors = tree.query_ball_point([hex_row['lng'], hex_row['lat']], 
                                            search_radius_deg)
            neighbors = [n for n in neighbors if n != hex_idx]  # Remove self
            
            if len(neighbors) >= self.min_neighbors:
                # Calculate distances for weighting
                neighbor_coords = coords[neighbors]
                hex_coord = np.array([hex_row['lng'], hex_row['lat']])
                distances = np.linalg.norm(neighbor_coords - hex_coord, axis=1)
                
                # Inverse distance weighting
                weights = 1.0 / (distances + 1e-10)
                weights = weights / np.sum(weights)
                
                # Get original embedding
                original_embedding = hex_row[embedding_cols].values
                
                # Calculate smoothed embedding
                neighbor_embeddings = df.iloc[neighbors][embedding_cols].values
                smoothed_embedding = np.average(neighbor_embeddings, weights=weights, axis=0)
                
                # Blend original and smoothed embeddings
                blended_embedding = ((1 - self.smoothing_strength) * original_embedding + 
                                   self.smoothing_strength * smoothed_embedding)
                
                # Update the smoothed dataframe
                for i, col in enumerate(embedding_cols):
                    smoothed_df.at[hex_idx, col] = blended_embedding[i]
                    
                smoothed_count += 1
        
        logger.info(f"Applied smoothing to {smoothed_count} hexagons")
        return smoothed_df

    def validate_spatial_continuity(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Validate spatial continuity and compute quality metrics
        """
        logger.info("Validating spatial continuity...")
        
        coords = df[['lng', 'lat']].values
        embedding_cols = [col for col in df.columns if col.startswith('A')]
        embeddings = df[embedding_cols].values
        
        # Build spatial tree
        tree = cKDTree(coords)
        
        # Calculate local spatial variance for continuity assessment
        search_radius_deg = self.smoothing_radius_km / 111.0
        local_variances = []
        
        for i, coord in enumerate(coords):
            neighbors = tree.query_ball_point(coord, search_radius_deg)
            if len(neighbors) > 1:
                neighbor_embeddings = embeddings[neighbors]
                local_variance = np.mean(np.var(neighbor_embeddings, axis=0))
                local_variances.append(local_variance)
        
        # Calculate continuity metrics
        metrics = {
            'mean_local_variance': np.mean(local_variances) if local_variances else 0.0,
            'std_local_variance': np.std(local_variances) if local_variances else 0.0,
            'continuity_score': 1.0 / (1.0 + np.mean(local_variances)) if local_variances else 1.0,
            'coverage_completeness': len(df) / max(self.estimate_expected_hexagons(df), len(df)),
            'boundary_smoothness': self.calculate_boundary_smoothness(df, tree)
        }
        
        logger.info("Spatial continuity validation complete:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        return metrics

    def estimate_expected_hexagons(self, df: pd.DataFrame) -> int:
        """
        Estimate expected number of hexagons for the study area
        """
        # Calculate bounding box area
        min_lng, max_lng = df['lng'].min(), df['lng'].max()
        min_lat, max_lat = df['lat'].min(), df['lat'].max()
        
        # Approximate area calculation (rough estimate)
        area_deg2 = (max_lng - min_lng) * (max_lat - min_lat)
        area_km2 = area_deg2 * 111.0 * 111.0 * np.cos(np.radians((min_lat + max_lat) / 2))
        
        # H3 resolution 8 hexagon area is approximately 0.737 km²
        h3_area_km2 = 0.737
        expected_hexagons = int(area_km2 / h3_area_km2)
        
        return expected_hexagons

    def calculate_boundary_smoothness(self, df: pd.DataFrame, tree: cKDTree) -> float:
        """
        Calculate boundary smoothness metric
        """
        embedding_cols = [col for col in df.columns if col.startswith('A')]
        embeddings = df[embedding_cols].values
        coords = df[['lng', 'lat']].values
        
        # Find boundary points (hexagons with fewer neighbors)
        search_radius_deg = self.smoothing_radius_km / 111.0
        boundary_discontinuities = []
        
        for i, coord in enumerate(coords):
            neighbors = tree.query_ball_point(coord, search_radius_deg)
            if len(neighbors) <= self.min_neighbors:  # Likely boundary point
                if len(neighbors) > 1:
                    # Calculate embedding variance with neighbors
                    neighbor_embeddings = embeddings[neighbors]
                    discontinuity = np.mean(np.var(neighbor_embeddings, axis=0))
                    boundary_discontinuities.append(discontinuity)
        
        if boundary_discontinuities:
            smoothness = 1.0 / (1.0 + np.mean(boundary_discontinuities))
        else:
            smoothness = 1.0
            
        return smoothness

    def process_spatial_smoothing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Main processing pipeline for spatial smoothing
        """
        logger.info("="*60)
        logger.info("SIMPLE SPATIAL SMOOTHING PIPELINE")
        logger.info("="*60)
        
        original_count = len(df)
        logger.info(f"Input hexagons: {original_count:,}")
        
        # Step 1: Detect boundary hexagons
        boundary_hexagons = self.detect_boundary_hexagons(df)
        
        # Step 2: Apply spatial smoothing to boundaries
        smoothed_df = self.smooth_boundary_hexagons(df, boundary_hexagons)
        
        # Step 3: Validate results
        quality_metrics = self.validate_spatial_continuity(smoothed_df)
        
        logger.info("="*60)
        logger.info("SIMPLE SPATIAL SMOOTHING COMPLETE")
        logger.info(f"Final hexagon count: {len(smoothed_df):,}")
        logger.info(f"Boundary hexagons processed: {len(boundary_hexagons):,}")
        logger.info(f"Continuity score: {quality_metrics['continuity_score']:.4f}")
        logger.info("="*60)
        
        return smoothed_df, quality_metrics


def apply_simple_spatial_smoothing(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Convenience function to apply simplified spatial smoothing to a DataFrame
    """
    smoother = SimpleSpatialSmoother(config)
    return smoother.process_spatial_smoothing(df)


if __name__ == "__main__":
    # Example usage
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with sample data
    print("Simple spatial smoothing utilities loaded successfully")
    print("Use apply_simple_spatial_smoothing(df, config) to process data")