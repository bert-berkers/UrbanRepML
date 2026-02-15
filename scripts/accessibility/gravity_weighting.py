"""
Gravity Model Weighting for Accessibility Graphs

Applies gravity model weighting using building density to create
meaningful edge weights for the accessibility graph.

Uses SRAI for all spatial operations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np
from srai.regionalizers import H3Regionalizer

logger = logging.getLogger(__name__)

# PLACEHOLDER - Implementation needed

def calculate_gravity_weights(
    travel_times: pd.DataFrame,
    building_density: pd.DataFrame,
    gravity_exponent: float = -2.0,
    use_srai_regions: bool = True
) -> pd.DataFrame:
    """
    Apply gravity model weighting to travel time edges.
    
    Args:
        travel_times: DataFrame with origin_h3, destination_h3, travel_time_minutes
        building_density: DataFrame with h3_id, building_count/density
        gravity_exponent: Exponent for distance decay (typically negative)
        use_srai_regions: Always True - use SRAI H3 regions
        
    Returns:
        DataFrame with origin_h3, destination_h3, gravity_weight
        
    Note:
        Gravity model: Weight = (Origin_density * Dest_density) / (Travel_time^abs(exponent))
    """
    if not use_srai_regions:
        raise ValueError("Must use SRAI regions - set use_srai_regions=True")
    
    logger.info("Calculating gravity model weights")
    
    # Merge with building densities
    travel_gravity = travel_times.copy()
    
    # Add origin densities
    travel_gravity = travel_gravity.merge(
        building_density.rename(columns={'h3_id': 'origin_h3', 'building_count': 'origin_density'}),
        on='origin_h3',
        how='left'
    )
    
    # Add destination densities  
    travel_gravity = travel_gravity.merge(
        building_density.rename(columns={'h3_id': 'destination_h3', 'building_count': 'dest_density'}),
        on='destination_h3',
        how='left'
    )
    
    # PLACEHOLDER: Implement actual gravity model calculation
    # This should:
    # 1. Handle missing building density data
    # 2. Apply gravity formula: (O_i * D_j) / (T_ij^α)
    # 3. Normalize weights appropriately
    # 4. Filter out zero/negative weights
    
    logger.warning("PLACEHOLDER: Gravity weighting calculation not implemented")
    
    # Calculate dummy gravity weights
    travel_gravity['gravity_weight'] = (
        travel_gravity.get('origin_density', 1.0) * 
        travel_gravity.get('dest_density', 1.0) / 
        np.power(travel_gravity['travel_time_minutes'], abs(gravity_exponent))
    )
    
    # Handle missing data
    travel_gravity['gravity_weight'] = travel_gravity['gravity_weight'].fillna(0.0)
    
    return travel_gravity[['origin_h3', 'destination_h3', 'gravity_weight']]


def load_building_density(
    study_area: str,
    h3_resolution: int = 9
) -> pd.DataFrame:
    """
    Load building density data for H3 regions.
    
    Args:
        study_area: Name of study area
        h3_resolution: H3 resolution level
        
    Returns:
        DataFrame with h3_id, building_count
    """
    
    # PLACEHOLDER: Load actual building data
    # This should load from processed building footprint data
    # aggregated to H3 regions using SRAI
    
    logger.warning("PLACEHOLDER: Building density loading not implemented")
    
    # Load H3 regions to generate dummy data
    regions_path = f"data/study_areas/{study_area}/regions_gdf/h3_res{h3_resolution}.parquet"
    
    if Path(regions_path).exists():
        regions_gdf = gpd.read_parquet(regions_path)
        
        # Generate dummy building densities
        dummy_densities = pd.DataFrame({
            'h3_id': regions_gdf['region_id'],
            'building_count': np.random.poisson(50, len(regions_gdf))  # Dummy data
        })
        
        return dummy_densities
    else:
        raise FileNotFoundError(f"H3 regions not found: {regions_path}")


def apply_gravity_weighting(
    study_area: str,
    h3_resolution: int = 9,
    gravity_exponent: float = -2.0
) -> pd.DataFrame:
    """
    Apply gravity weighting to accessibility graph for study area.
    
    Args:
        study_area: Name of study area
        h3_resolution: H3 resolution level
        gravity_exponent: Distance decay exponent
        
    Returns:
        Weighted accessibility graph
    """
    
    # Load travel times (from floodfill calculation)
    travel_times_path = f"data/study_areas/{study_area}/urban_embedding/graphs/travel_times_res{h3_resolution}.parquet"
    
    if not Path(travel_times_path).exists():
        raise FileNotFoundError(f"Travel times not found: {travel_times_path}")
    
    travel_times = pd.read_parquet(travel_times_path)
    
    # Load building densities
    building_density = load_building_density(study_area, h3_resolution)
    
    # Calculate gravity weights
    weighted_graph = calculate_gravity_weights(
        travel_times,
        building_density,
        gravity_exponent
    )
    
    return weighted_graph


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply gravity model weighting")
    parser.add_argument("--study-area", required=True, help="Study area name")
    parser.add_argument("--resolution", type=int, default=9, help="H3 resolution")
    parser.add_argument("--exponent", type=float, default=-2.0, help="Gravity exponent")
    parser.add_argument("--use-srai", action="store_true", default=True,
                       help="Use SRAI (always True)")
    
    args = parser.parse_args()
    
    if not args.use_srai:
        raise ValueError("Must use SRAI - set --use-srai flag")
    
    # Apply gravity weighting
    weighted_graph = apply_gravity_weighting(
        args.study_area,
        args.resolution,
        args.exponent
    )
    
    # Save results [old 2024]
    output_path = f"data/study_areas/{args.study_area}/urban_embedding/graphs/gravity_weighted_res{args.resolution}.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    weighted_graph.to_parquet(output_path)
    
    logger.info(f"Gravity-weighted graph saved to {output_path}")
    logger.info(f"Graph has {len(weighted_graph)} edges with mean weight {weighted_graph['gravity_weight'].mean():.4f}")