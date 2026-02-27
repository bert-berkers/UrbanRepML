"""
Floodfill Travel Time Calculation

Calculates travel times between H3 hexagons using floodfill algorithm
with local cutoff (few minutes) for accessibility graph construction.

Uses SRAI for all H3 operations and neighborhood analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood

logger = logging.getLogger(__name__)

# PLACEHOLDER - Implementation needed

def calculate_floodfill_travel_times(
    regions_gdf: gpd.GeoDataFrame,
    cutoff_minutes: int = 5,
    travel_mode: str = "walk",
    use_srai_neighbors: bool = True
) -> pd.DataFrame:
    """
    Calculate travel times between H3 regions using floodfill algorithm.
    
    Args:
        regions_gdf: H3 regions from SRAI H3Regionalizer
        cutoff_minutes: Maximum travel time to consider
        travel_mode: Mode of transport ('walk', 'bike', 'drive')
        use_srai_neighbors: Always True - use SRAI neighborhoods
        
    Returns:
        DataFrame with origin_region_id, destination_region_id, travel_time_minutes
        
    Note:
        This is a placeholder. Implementation should use:
        - SRAI H3Neighbourhood for spatial relationships
        - Local road network analysis
        - Dijkstra or similar for shortest paths
    """
    if not use_srai_neighbors:
        raise ValueError("Must use SRAI neighborhoods - set use_srai_neighbors=True")
    
    logger.info(f"Calculating floodfill travel times with {cutoff_minutes}min cutoff")
    
    # Get H3 neighborhoods using SRAI
    neighbourhood = H3Neighbourhood()
    neighbors = neighbourhood.get_neighbours(regions_gdf)
    
    # PLACEHOLDER: Implement actual travel time calculation
    # This should:
    # 1. Load local road network
    # 2. Calculate speeds by road type
    # 3. Run floodfill/Dijkstra from each origin
    # 4. Stop when cutoff_minutes reached
    # 5. Return travel time matrix
    
    logger.warning("PLACEHOLDER: Floodfill travel time calculation not implemented")
    
    # Return dummy data for now
    n_regions = len(regions_gdf)
    dummy_data = []
    for i in range(min(100, n_regions)):  # Limit for demo
        for j in range(min(10, n_regions)):  # Small neighborhood
            if i != j:
                dummy_data.append({
                    'origin_region_id': regions_gdf.index[i],
                    'destination_region_id': regions_gdf.index[j],
                    'travel_time_minutes': np.random.uniform(1, cutoff_minutes)
                })
    
    return pd.DataFrame(dummy_data)


def calculate_local_accessibility_matrix(
    study_area: str,
    h3_resolution: int = 9,
    cutoff_minutes: int = 5
) -> pd.DataFrame:
    """
    Calculate accessibility matrix for a study area.
    
    Args:
        study_area: Name of study area (e.g., 'netherlands')
        h3_resolution: H3 resolution level
        cutoff_minutes: Travel time cutoff
        
    Returns:
        Travel time matrix as DataFrame
    """
    
    # Load study area regions (created with SRAI)
    regions_path = f"data/study_areas/{study_area}/regions_gdf/h3_res{h3_resolution}.parquet"
    
    if not Path(regions_path).exists():
        raise FileNotFoundError(f"H3 regions not found: {regions_path}")
    
    regions_gdf = gpd.read_parquet(regions_path)
    
    # Calculate travel times
    travel_times = calculate_floodfill_travel_times(
        regions_gdf, 
        cutoff_minutes=cutoff_minutes
    )
    
    return travel_times


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate floodfill travel times")
    parser.add_argument("--study-area", required=True, help="Study area name")
    parser.add_argument("--resolution", type=int, default=9, help="H3 resolution")
    parser.add_argument("--cutoff", type=int, default=5, help="Travel time cutoff (minutes)")
    parser.add_argument("--use-srai", action="store_true", default=True, 
                       help="Use SRAI (always True)")
    
    args = parser.parse_args()
    
    if not args.use_srai:
        raise ValueError("Must use SRAI - set --use-srai flag")
    
    # Calculate travel times
    travel_times = calculate_local_accessibility_matrix(
        args.study_area,
        args.resolution, 
        args.cutoff
    )
    
    # Save results [old 2024]
    output_path = f"data/study_areas/{args.study_area}/urban_embedding/graphs/travel_times_res{args.resolution}.parquet"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    travel_times.to_parquet(output_path)
    
    logger.info(f"Travel times saved to {output_path}")