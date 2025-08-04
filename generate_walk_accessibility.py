#!/usr/bin/env python3
"""
Generate missing walk accessibility graph for resolution 10
"""

import sys
import pickle
from pathlib import Path
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration for walk mode
WALK_CONFIG = {
    'speed': 1.4,  # m/s
    'max_travel_time': 300,  # 5 minutes
    'search_radius': 75,  # meters  
    'beta': 0.0020,  # decay parameter
    'percentile_threshold': 90  # Keep top 10% of edges
}

def load_walk_network(data_dir: Path) -> nx.Graph:
    """Load or download walk network."""
    print("[NETWORK] Loading walk network...")
    
    network_dir = data_dir / 'networks' / 'osm'
    walk_network_path = network_dir / 'south_holland_walk_network.pkl'
    
    if walk_network_path.exists():
        print(f"   [CACHE] Loading from {walk_network_path}")
        with open(walk_network_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("   [ERROR] Walk network not found!")
        raise FileNotFoundError(f"Walk network not found at {walk_network_path}")

def compute_walk_accessibility(regions_gdf: gpd.GeoDataFrame, network: nx.Graph) -> dict:
    """Compute walk accessibility graph for resolution 10."""
    print(f"[GRAPH] Computing walk accessibility for {len(regions_gdf)} regions...")
    
    # Get region centroids
    centroids = regions_gdf.geometry.centroid
    
    # Find nearest network nodes for each region
    region_nodes = {}
    print("   [MAPPING] Finding nearest network nodes...")
    
    for idx, centroid in tqdm(centroids.items(), desc="Mapping regions to network"):
        try:
            nearest_node = ox.distance.nearest_nodes(network, centroid.x, centroid.y)
            region_nodes[idx] = nearest_node
        except:
            continue
    
    print(f"   [INFO] Mapped {len(region_nodes)} regions to network nodes")
    
    # Compute accessibility matrix
    edges = []
    edge_weights = []
    
    print("   [COMPUTE] Computing pairwise accessibility...")
    
    region_list = list(region_nodes.keys())
    batch_size = 100
    
    for i in tqdm(range(0, len(region_list), batch_size), desc="Computing accessibility"):
        batch_regions = region_list[i:i+batch_size]
        
        for source_region in batch_regions:
            source_node = region_nodes[source_region]
            
            try:
                # Compute shortest path lengths from source
                lengths = nx.single_source_dijkstra_path_length(
                    network, source_node, 
                    cutoff=WALK_CONFIG['max_travel_time'], 
                    weight='travel_time'
                )
                
                # Find accessible regions
                for target_region in region_list:
                    if source_region == target_region:
                        continue
                        
                    target_node = region_nodes[target_region]
                    if target_node in lengths:
                        travel_time = lengths[target_node]
                        
                        # Compute accessibility with exponential decay
                        accessibility = np.exp(-WALK_CONFIG['beta'] * travel_time)
                        
                        if accessibility > 0.01:  # Threshold for meaningful accessibility
                            edges.append([source_region, target_region])
                            edge_weights.append(accessibility)
                            
            except:
                continue
    
    print(f"   [INFO] Generated {len(edges)} potential edges")
    
    # Filter to top percentile
    if len(edges) > 0:
        threshold_value = np.percentile(edge_weights, WALK_CONFIG['percentile_threshold'])
        
        filtered_edges = []
        filtered_weights = []
        
        for edge, weight in zip(edges, edge_weights):
            if weight >= threshold_value:
                filtered_edges.append(edge)
                filtered_weights.append(weight)
        
        print(f"   [FILTER] Kept {len(filtered_edges)} edges (top {100-WALK_CONFIG['percentile_threshold']}%)")
        
        # Create graph data structure
        edge_index = np.array(filtered_edges).T if filtered_edges else np.array([[], []])
        edge_attr = np.array(filtered_weights)
        
        return {
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'num_nodes': len(regions_gdf),
            'num_edges': len(filtered_edges)
        }
    else:
        print("   [WARN] No edges generated")
        return {
            'edge_index': np.array([[], []]),
            'edge_attr': np.array([]),
            'num_nodes': len(regions_gdf),
            'num_edges': 0
        }

def main():
    """Generate walk accessibility graph."""
    print("=== Generating Walk Accessibility Graph ===")
    
    # Paths
    project_root = Path("C:/Users/Bert Berkers/PycharmProjects/UrbanRepML")
    data_dir = project_root / "data"
    
    # Load resolution 10 regions
    print("[DATA] Loading resolution 10 regions...")
    regions_path = data_dir / "preprocessed" / "south_holland" / "regions_10_gdf.parquet"
    regions_gdf = gpd.read_parquet(regions_path)
    
    print(f"   [INFO] Loaded {len(regions_gdf)} regions")
    
    # Load walk network
    network = load_walk_network(data_dir)
    print(f"   [INFO] Network has {len(network.nodes)} nodes and {len(network.edges)} edges")
    
    # Compute accessibility
    graph_data = compute_walk_accessibility(regions_gdf, network)
    
    # Save graph
    output_path = data_dir / "networks" / "accessibility" / "south_holland_walk_res10.pkl"
    print(f"[SAVE] Saving to {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(graph_data, f)
    
    print(f"[SUCCESS] Walk accessibility graph generated!")
    print(f"   Nodes: {graph_data['num_nodes']}")
    print(f"   Edges: {graph_data['num_edges']}")
    if graph_data['num_edges'] > 0:
        print(f"   Avg accessibility: {graph_data['edge_attr'].mean():.6f}")
        print(f"   Max accessibility: {graph_data['edge_attr'].max():.6f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback  
        traceback.print_exc()
        sys.exit(1)