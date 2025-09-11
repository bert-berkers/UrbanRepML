"""
Percentile-Based Graph Pruning

Prunes accessibility graphs by keeping only the top percentile of edge strengths
per H3 resolution level. Creates sparse graphs for efficient GCN training.

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

# Default pruning thresholds per H3 resolution
DEFAULT_PRUNING_THRESHOLDS = {
    5: 0.99,   # Regional: very sparse (top 1%)
    8: 0.95,   # District: sparse (top 5%)
    9: 0.90,   # Neighborhood: moderate (top 10%)
    10: 0.85,  # Block: denser (top 15%)
    11: 0.80   # Detailed: densest (top 20%)
}


def prune_graph_by_percentile(
    weighted_graph: pd.DataFrame,
    h3_resolution: int,
    pruning_thresholds: Optional[Dict[int, float]] = None,
    use_srai_regions: bool = True
) -> pd.DataFrame:
    """
    Prune graph by keeping only top percentile of edge weights.
    
    Args:
        weighted_graph: DataFrame with origin_h3, destination_h3, gravity_weight
        h3_resolution: H3 resolution level
        pruning_thresholds: Dict mapping resolution to percentile threshold
        use_srai_regions: Always True - use SRAI H3 regions
        
    Returns:
        Pruned graph with only strongest edges
        
    Note:
        Higher threshold = sparser graph (keep fewer edges)
    """
    if not use_srai_regions:
        raise ValueError("Must use SRAI regions - set use_srai_regions=True")
    
    if pruning_thresholds is None:
        pruning_thresholds = DEFAULT_PRUNING_THRESHOLDS
    
    threshold = pruning_thresholds.get(h3_resolution, 0.90)
    
    logger.info(f"Pruning graph at resolution {h3_resolution} with threshold {threshold}")
    logger.info(f"Input graph has {len(weighted_graph)} edges")
    
    # Calculate percentile threshold
    weight_threshold = np.percentile(weighted_graph['gravity_weight'], threshold * 100)
    
    # Keep only edges above threshold
    pruned_graph = weighted_graph[
        weighted_graph['gravity_weight'] >= weight_threshold
    ].copy()
    
    # PLACEHOLDER: Implement additional pruning logic
    # This should:
    # 1. Ensure graph connectivity (no isolated nodes)
    # 2. Balance sparsity with connectivity
    # 3. Handle resolution-specific constraints
    # 4. Add edge type metadata if needed
    
    logger.warning("PLACEHOLDER: Advanced pruning logic not implemented")
    
    # Sort by weight (strongest first)
    pruned_graph = pruned_graph.sort_values('gravity_weight', ascending=False)
    
    logger.info(f"Pruned graph has {len(pruned_graph)} edges ({len(pruned_graph)/len(weighted_graph)*100:.1f}% kept)")
    logger.info(f"Weight range: {pruned_graph['gravity_weight'].min():.4f} - {pruned_graph['gravity_weight'].max():.4f}")
    
    return pruned_graph


def ensure_graph_connectivity(
    pruned_graph: pd.DataFrame,
    regions_gdf: gpd.GeoDataFrame,
    min_degree: int = 1
) -> pd.DataFrame:
    """
    Ensure all H3 regions have minimum connectivity.
    
    Args:
        pruned_graph: Pruned accessibility graph
        regions_gdf: H3 regions from SRAI
        min_degree: Minimum edges per node
        
    Returns:
        Graph with connectivity ensured
    """
    
    # PLACEHOLDER: Implement connectivity checking
    # This should:
    # 1. Find isolated nodes (degree < min_degree)
    # 2. Add edges to nearest neighbors using SRAI
    # 3. Ensure weakly connected components
    # 4. Preserve sparsity while ensuring connectivity
    
    logger.warning("PLACEHOLDER: Connectivity checking not implemented")
    
    return pruned_graph


def create_pruned_accessibility_graph(
    study_area: str,
    h3_resolution: int = 9,
    pruning_thresholds: Optional[Dict[int, float]] = None,
    ensure_connectivity: bool = True
) -> pd.DataFrame:
    """
    Create pruned accessibility graph for study area.
    
    Args:
        study_area: Name of study area
        h3_resolution: H3 resolution level
        pruning_thresholds: Custom pruning thresholds
        ensure_connectivity: Whether to ensure graph connectivity
        
    Returns:
        Pruned accessibility graph
    """
    
    # Load gravity-weighted graph
    weighted_graph_path = f"data/study_areas/{study_area}/urban_embedding/graphs/gravity_weighted_res{h3_resolution}.parquet"
    
    if not Path(weighted_graph_path).exists():
        raise FileNotFoundError(f"Weighted graph not found: {weighted_graph_path}")
    
    weighted_graph = pd.read_parquet(weighted_graph_path)
    
    # Prune by percentile
    pruned_graph = prune_graph_by_percentile(
        weighted_graph,
        h3_resolution,
        pruning_thresholds
    )
    
    # Ensure connectivity if requested
    if ensure_connectivity:
        regions_path = f"data/study_areas/{study_area}/regions_gdf/h3_res{h3_resolution}.parquet"
        
        if Path(regions_path).exists():
            regions_gdf = gpd.read_parquet(regions_path)
            pruned_graph = ensure_graph_connectivity(pruned_graph, regions_gdf)
        else:
            logger.warning(f"Could not load regions for connectivity check: {regions_path}")
    
    return pruned_graph


def create_multi_resolution_graphs(
    study_area: str,
    resolutions: List[int] = [5, 8, 9, 10],
    pruning_thresholds: Optional[Dict[int, float]] = None
) -> Dict[int, pd.DataFrame]:
    """
    Create pruned graphs for multiple H3 resolutions.
    
    Args:
        study_area: Name of study area
        resolutions: List of H3 resolutions to process
        pruning_thresholds: Custom pruning thresholds
        
    Returns:
        Dict mapping resolution to pruned graph
    """
    
    graphs = {}
    
    for resolution in resolutions:
        logger.info(f"Creating pruned graph for resolution {resolution}")
        
        try:
            pruned_graph = create_pruned_accessibility_graph(
                study_area,
                resolution,
                pruning_thresholds
            )
            graphs[resolution] = pruned_graph
            
            # Save individual graph
            output_path = f"data/study_areas/{study_area}/urban_embedding/graphs/pruned_res{resolution}.parquet"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            pruned_graph.to_parquet(output_path)
            
            logger.info(f"Saved pruned graph to {output_path}")
            
        except FileNotFoundError as e:
            logger.warning(f"Skipping resolution {resolution}: {e}")
            continue
    
    return graphs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prune accessibility graphs by percentile")
    parser.add_argument("--study-area", required=True, help="Study area name")
    parser.add_argument("--resolution", type=int, help="Single H3 resolution")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[5, 8, 9, 10],
                       help="Multiple H3 resolutions")
    parser.add_argument("--threshold", type=float, help="Custom pruning threshold")
    parser.add_argument("--use-srai", action="store_true", default=True,
                       help="Use SRAI (always True)")
    
    args = parser.parse_args()
    
    if not args.use_srai:
        raise ValueError("Must use SRAI - set --use-srai flag")
    
    # Set up custom thresholds if provided
    custom_thresholds = None
    if args.threshold:
        if args.resolution:
            custom_thresholds = {args.resolution: args.threshold}
        else:
            logger.warning("Custom threshold provided but no single resolution specified")
    
    # Process single resolution or multiple
    if args.resolution:
        pruned_graph = create_pruned_accessibility_graph(
            args.study_area,
            args.resolution,
            custom_thresholds
        )
        
        output_path = f"data/study_areas/{args.study_area}/urban_embedding/graphs/pruned_res{args.resolution}.parquet"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pruned_graph.to_parquet(output_path)
        
        logger.info(f"Pruned graph saved to {output_path}")
        
    else:
        graphs = create_multi_resolution_graphs(
            args.study_area,
            args.resolutions,
            custom_thresholds
        )
        
        logger.info(f"Created {len(graphs)} pruned graphs for resolutions: {list(graphs.keys())}")