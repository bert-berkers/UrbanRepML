"""
Hierarchical accessibility graph computation for UrbanRepML.
Assumes all prerequisite data exists and just computes accessibility graphs.

Prerequisites:
- boundaries/area_study_gdf.parquet or {city}_area.parquet
- regions/regions_{res}_gdf.parquet or {city}_res{res}.parquet
- density/building_density_res{res}_preprocessed.parquet

This script ONLY computes accessibility graphs for the hierarchy.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import time
import json
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pickle
import h3
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import our graph constructors
from urban_embedding.hexagonal_graph_constructor import HexagonalLatticeConstructor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
MODES = {8: 'drive', 9: 'bike', 10: 'walk'}
SPEEDS = {'walk': 1.4, 'bike': 4.17, 'drive': 11.11}  # meters per second
BETA = {'walk': 0.002, 'bike': 0.0012, 'drive': 0.0008}  # impedance decay
CUTOFF_TIME = 300  # 5 minutes in seconds
PERCENTILE_THRESHOLD = 90  # Keep top 10% of edges
FSI_THRESHOLD = 0.1  # Minimum FSI to consider hex as active (much higher for faster computation)
BATCH_SIZE = 500  # Process hexagons in batches


class HierarchicalGraphBuilder:
    """Builds accessibility graphs with H3 hierarchical structure."""
    
    def __init__(self):
        self.hierarchical_mapping = {}
        self.active_hexagons = {8: set(), 9: set(), 10: set()}
        self.hex_to_column = {}  # Maps each hex to its res-8 parent
        
    def build_hierarchy(self, res8_gdf: gpd.GeoDataFrame) -> Dict:
        """Build hierarchical mapping from active res-8 hexagons."""
        
        print("[HIERARCHY] Building hierarchical hex columns...")
        
        # Filter res-8 hexagons by density threshold
        active_res8 = res8_gdf[res8_gdf['FSI_24'] > FSI_THRESHOLD]
        print(f"[INFO] Found {len(active_res8)} active res-8 hexagons (FSI > {FSI_THRESHOLD})")
        
        self.active_hexagons[8] = set(active_res8.index)
        
        # Build hierarchy for each active res-8 hex
        for res8_hex in tqdm(active_res8.index, desc="Building hierarchy"):
            # Get res-9 children
            children_9 = list(h3.cell_to_children(res8_hex, 9))
            self.active_hexagons[9].update(children_9)
            
            # Get res-10 grandchildren
            grandchildren_10 = []
            for child9 in children_9:
                gc = list(h3.cell_to_children(child9, 10))
                grandchildren_10.extend(gc)
                self.active_hexagons[10].update(gc)
                
                # Map grandchildren to res-8 column
                for gc_hex in gc:
                    self.hex_to_column[gc_hex] = res8_hex
                    
            # Map children to res-8 column
            for child9 in children_9:
                self.hex_to_column[child9] = res8_hex
                
            # Store hierarchy
            self.hierarchical_mapping[res8_hex] = {
                'children_9': children_9,
                'grandchildren_10': grandchildren_10,
                'fsi': float(active_res8.loc[res8_hex, 'FSI_24']),
                'volume': float(active_res8.loc[res8_hex, 'building_volume'])
            }
        
        # Self-map res-8 hexagons
        for res8_hex in self.active_hexagons[8]:
            self.hex_to_column[res8_hex] = res8_hex
            
        print(f"[OK] Created {len(self.hierarchical_mapping)} hex columns")
        print(f"     Res-8: {len(self.active_hexagons[8])} hexagons")
        print(f"     Res-9: {len(self.active_hexagons[9])} hexagons")
        print(f"     Res-10: {len(self.active_hexagons[10])} hexagons")
        
        return self.hierarchical_mapping
    
    def filter_regions_by_hierarchy(self, regions_gdf: gpd.GeoDataFrame, resolution: int) -> gpd.GeoDataFrame:
        """Filter regions to only include those in the hierarchy."""
        
        active_set = self.active_hexagons[resolution]
        filtered = regions_gdf[regions_gdf.index.isin(active_set)].copy()
        
        print(f"[FILTER] Res-{resolution}: {len(filtered)} / {len(regions_gdf)} hexagons kept")
        
        # Add column mapping
        filtered['column_id'] = filtered.index.map(self.hex_to_column)
        
        return filtered


def load_or_download_osm_network(mode: str, area_gdf: gpd.GeoDataFrame) -> nx.Graph:
    """Load cached OSM network or download if not exists."""
    
    # Create hash of area for cache key
    import hashlib
    area_str = str(area_gdf.geometry.values[0])
    area_hash = hashlib.md5(area_str.encode()).hexdigest()[:8]
    
    # Cache path based on area and mode
    network_path = Path(f"cache/networks/osm/{area_hash}_{mode}_network.pkl")
    network_path.parent.mkdir(parents=True, exist_ok=True)
    
    if network_path.exists():
        print(f"[CACHE] Loading cached {mode} network for area {area_hash}...")
        with open(network_path, 'rb') as f:
            network = pickle.load(f)
        print(f"[OK] Loaded network with {len(network.nodes)} nodes")
        return network
    
    print(f"[DOWNLOAD] Downloading {mode} network from OpenStreetMap...")
    
    # Buffer area for network download
    buffer_dist = {'walk': 200, 'bike': 300, 'drive': 500}[mode]
    area_rd = area_gdf.to_crs('EPSG:28992')
    area_buffered = area_rd.buffer(buffer_dist).unary_union
    area_wgs84 = gpd.GeoSeries([area_buffered], crs='EPSG:28992').to_crs('EPSG:4326')[0]
    
    # Download network
    network = ox.graph_from_polygon(
        area_wgs84,
        network_type='drive' if mode == 'drive' else 'all',
        simplify=True,
        truncate_by_edge=True
    )
    
    # Add travel times
    speed = SPEEDS[mode]
    for u, v, data in network.edges(data=True):
        data['time'] = data['length'] / speed
    
    # Ensure connectivity
    if mode == 'drive':
        largest_cc = max(nx.strongly_connected_components(network), key=len)
        network = network.subgraph(largest_cc).copy()
    else:
        network = network.to_undirected()
        largest_cc = max(nx.connected_components(network), key=len)
        network = network.subgraph(largest_cc).copy()
    
    # Cache network
    with open(network_path, 'wb') as f:
        pickle.dump(network, f)
    
    print(f"[OK] Downloaded and cached network with {len(network.nodes)} nodes")
    print(f"[CACHE] Saved to {network_path}")
    
    return network


def compute_accessibility_graph(
    regions_gdf: gpd.GeoDataFrame,
    network: nx.Graph,
    mode: str,
    resolution: int,
    hierarchy_builder: HierarchicalGraphBuilder
) -> Dict:
    """Compute accessibility graph with batch processing embeddings."""
    
    print(f"\n[GRAPH] Building accessibility graph for {mode} at resolution {resolution}")
    
    # Project to Dutch CRS
    regions_gdf = regions_gdf.to_crs('EPSG:28992')
    
    # Get hex IDs and create mapping
    hex_indices = regions_gdf.index.tolist()
    hex_id_to_idx = {hex_id: idx for idx, hex_id in enumerate(hex_indices)}
    
    # Calculate building volumes
    hex_areas = regions_gdf.geometry.area
    volumes = regions_gdf['FSI_24'] * hex_areas
    
    # Filter active hexagons
    active_hexes = [h for h in hex_indices if volumes.loc[h] > 0]
    print(f"[INFO] Processing {len(active_hexes)} active hexagons")
    
    # Get OSM nodes
    centroids = regions_gdf.geometry.centroid
    centroids_wgs = centroids.to_crs('EPSG:4326')
    osm_nodes = ox.nearest_nodes(
        network,
        centroids_wgs.x.values,
        centroids_wgs.y.values
    )
    
    # Process in batches
    edges = []
    beta = BETA[mode]
    
    # Create batches
    batches = [active_hexes[i:i+BATCH_SIZE] for i in range(0, len(active_hexes), BATCH_SIZE)]
    
    print(f"[BATCH] Processing {len(batches)} batches")
    
    # Process each batch
    for batch_idx, hex_batch in enumerate(tqdm(batches, desc=f"Computing {mode} accessibility")):
        
        for hex_id in hex_batch:
            source_idx = hex_id_to_idx[hex_id]
            source_node = osm_nodes[source_idx]
            
            if source_node is None:
                continue
                
            try:
                # Single-source Dijkstra with cutoff
                lengths = nx.single_source_dijkstra_path_length(
                    network,
                    source_node,
                    weight='time',
                    cutoff=CUTOFF_TIME
                )
            except nx.NetworkXNoPath:
                continue
            
            # Process reachable destinations
            for target_hex_id in hex_indices:
                if target_hex_id == hex_id:
                    continue
                    
                target_idx = hex_id_to_idx[target_hex_id]
                target_node = osm_nodes[target_idx]
                
                if target_node not in lengths:
                    continue
                    
                travel_time = lengths[target_node]
                target_volume = volumes.loc[target_hex_id]
                
                if target_volume <= 0:
                    continue
                    
                # Calculate accessibility
                impedance = np.exp(-beta * travel_time)
                accessibility = float(target_volume) * impedance
                
                if accessibility > 0:
                    edges.append((source_idx, target_idx, accessibility))
    
    print(f"[INFO] Generated {len(edges)} potential edges")
    
    # Apply percentile threshold and create bidirectional edges
    if edges:
        accessibilities = [e[2] for e in edges]
        threshold = np.percentile(accessibilities, PERCENTILE_THRESHOLD)
        
        # Filter and make bidirectional
        bidirectional_edges = []
        edge_set = set()
        
        for source, target, accessibility in edges:
            if accessibility >= threshold:
                # Add forward edge
                if (source, target) not in edge_set:
                    bidirectional_edges.append((source, target, accessibility))
                    edge_set.add((source, target))
                
                # Add reverse edge
                if (target, source) not in edge_set:
                    bidirectional_edges.append((target, source, accessibility))
                    edge_set.add((target, source))
        
        print(f"[INFO] Filtered to {len(bidirectional_edges)} edges (top {100-PERCENTILE_THRESHOLD}%)")
        
        # Add parent mapping for each hex
        parent_mapping = {}
        for hex_id in hex_indices:
            parent_mapping[hex_id] = hierarchy_builder.hex_to_column.get(hex_id, hex_id)
        
        # Create graph dictionary
        graph_data = {
            'resolution': resolution,
            'mode': mode,
            'num_nodes': len(regions_gdf),
            'edges': bidirectional_edges,
            'hex_indices': hex_indices,
            'parent_mapping': parent_mapping,
            'parameters': {
                'speed': SPEEDS[mode],
                'beta': beta,
                'cutoff_time': CUTOFF_TIME,
                'percentile_threshold': PERCENTILE_THRESHOLD,
                'fsi_threshold': FSI_THRESHOLD
            },
            'statistics': {
                'num_edges': len(bidirectional_edges),
                'avg_accessibility': np.mean([e[2] for e in bidirectional_edges]),
                'max_accessibility': max([e[2] for e in bidirectional_edges]),
                'min_accessibility': min([e[2] for e in bidirectional_edges]),
                'threshold_value': threshold,
                'num_columns': len(set(parent_mapping.values()))
            }
        }
        
        return graph_data
    else:
        raise ValueError(f"No edges created for {mode} at resolution {resolution}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate hierarchical accessibility graphs')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with filtered region/density data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for graphs (default: cache/graphs)')
    parser.add_argument('--city_name', type=str, default='south_holland',
                        help='City/region name for output files')
    parser.add_argument('--resolutions', type=str, default='8,9,10',
                        help='Comma-separated H3 resolutions')
    parser.add_argument('--fsi_threshold', type=float, default=0.1,
                        help='Minimum FSI for active hexagons')
    parser.add_argument('--cutoff_time', type=int, default=300,
                        help='Maximum travel time in seconds')
    parser.add_argument('--percentile_threshold', type=float, default=90,
                        help='Percentile threshold for edge filtering')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for graph computation')
    parser.add_argument('--graph_type', type=str, default='accessibility',
                        choices=['accessibility', 'hexagonal', 'hexagonal_lattice'],
                        help='Type of graph to construct')
    parser.add_argument('--neighbor_rings', type=int, default=1,
                        help='Number of neighbor rings for hexagonal graphs')
    parser.add_argument('--edge_weight', type=float, default=1.0,
                        help='Uniform edge weight for hexagonal graphs')
    return parser.parse_args()

def main():
    """Main workflow - just compute accessibility graphs."""
    
    args = parse_args()
    
    print("==== UrbanRepML Hierarchical Accessibility Graphs ====")
    print("=" * 55)
    
    start_time = time.time()
    
    # Parse resolutions
    resolutions = [int(r) for r in args.resolutions.split(',')]
    
    # Setup paths
    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("cache/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update global parameters
    global FSI_THRESHOLD, CUTOFF_TIME, PERCENTILE_THRESHOLD, BATCH_SIZE
    FSI_THRESHOLD = args.fsi_threshold
    CUTOFF_TIME = args.cutoff_time
    PERCENTILE_THRESHOLD = args.percentile_threshold
    BATCH_SIZE = args.batch_size
    
    print(f"[CONFIG] Data directory: {data_dir}")
    print(f"[CONFIG] Output directory: {output_dir}")
    print(f"[CONFIG] Graph type: {args.graph_type}")
    if args.graph_type in ['hexagonal', 'hexagonal_lattice']:
        print(f"[CONFIG] Neighbor rings: {args.neighbor_rings}")
        print(f"[CONFIG] Edge weight: {args.edge_weight}")
    else:
        print(f"[CONFIG] FSI threshold: {FSI_THRESHOLD}")
        print(f"[CONFIG] Cutoff time: {CUTOFF_TIME}s")
        print(f"[CONFIG] Percentile threshold: {PERCENTILE_THRESHOLD}%")
    
    # Load area boundary
    boundary_path = data_dir / "boundaries" / "area_study_gdf.parquet"
    if not boundary_path.exists():
        # Try alternative naming
        boundary_path = data_dir / "boundaries" / f"{args.city_name}_area.parquet"
    
    if not boundary_path.exists():
        raise FileNotFoundError(f"Area boundary not found at {boundary_path}")
    
    area_gdf = gpd.read_parquet(boundary_path)
    print(f"[LOAD] Loaded area boundary from {boundary_path}")
    
    # Initialize hierarchy builder
    hierarchy_builder = HierarchicalGraphBuilder()
    
    # Load base resolution data and build hierarchy
    base_res = min(resolutions)  # Use lowest resolution as base
    print(f"\n[STEP 1] Building hierarchy from resolution {base_res} data...")
    
    # Load regions with density data
    regions_path = data_dir / "regions" / f"regions_{base_res}_gdf.parquet"
    if not regions_path.exists():
        # Try alternative naming
        regions_path = data_dir / "regions" / f"{args.city_name}_res{base_res}.parquet"
    
    if not regions_path.exists():
        raise FileNotFoundError(f"Regions file not found: {regions_path}")
    
    res_base_gdf = gpd.read_parquet(regions_path)
    
    # Load density data
    density_path = data_dir / "density" / f"building_density_res{base_res}_preprocessed.parquet"
    if density_path.exists():
        density_df = pd.read_parquet(density_path)
        res_base_gdf = res_base_gdf.join(density_df[['FSI_24', 'building_volume']], how='left')
        res_base_gdf['FSI_24'] = res_base_gdf['FSI_24'].fillna(0)
        res_base_gdf['building_volume'] = res_base_gdf['building_volume'].fillna(0)
    else:
        print(f"[WARN] No density data found, using zero values")
        res_base_gdf['FSI_24'] = 0
        res_base_gdf['building_volume'] = 0
    
    hierarchy_builder.build_hierarchy(res_base_gdf)
    
    # Save hierarchical mapping
    mapping_path = output_dir / "hierarchical_mapping.pkl"
    
    with open(mapping_path, 'wb') as f:
        pickle.dump({
            'mapping': hierarchy_builder.hierarchical_mapping,
            'hex_to_column': hierarchy_builder.hex_to_column,
            'active_hexagons': hierarchy_builder.active_hexagons
        }, f)
    
    print(f"[SAVE] Hierarchical mapping saved to {mapping_path}")

    # Process each resolution
    print("\n[STEP 2] Computing accessibility graphs...")
    
    for resolution in resolutions:
        mode = MODES.get(resolution, 'walk')  # Default to walk if not in MODES
        print(f"\n--- Processing {mode} mode at resolution {resolution} ---")
        
        # Load regions data
        regions_path = data_dir / "regions" / f"regions_{resolution}_gdf.parquet"
        if not regions_path.exists():
            regions_path = data_dir / "regions" / f"{args.city_name}_res{resolution}.parquet"
        
        if not regions_path.exists():
            print(f"[SKIP] Regions file not found for resolution {resolution}")
            continue
            
        full_regions = gpd.read_parquet(regions_path)
        
        # Load density data
        density_path = data_dir / "density" / f"building_density_res{resolution}_preprocessed.parquet"
        if density_path.exists():
            density_df = pd.read_parquet(density_path)
            full_regions = full_regions.join(density_df[['FSI_24', 'building_volume']], how='left')
            full_regions['FSI_24'] = full_regions['FSI_24'].fillna(0)
            full_regions['building_volume'] = full_regions['building_volume'].fillna(0)
        else:
            full_regions['FSI_24'] = 0
            full_regions['building_volume'] = 0
        
        print(f"[LOAD] Loaded {len(full_regions)} regions")
        
        # Filter by hierarchy
        filtered_regions = hierarchy_builder.filter_regions_by_hierarchy(full_regions, resolution)
        
        # Compute graph based on type
        try:
            if args.graph_type in ['hexagonal', 'hexagonal_lattice']:
                # Use hexagonal lattice constructor
                print(f"[GRAPH] Building hexagonal lattice for {mode}...")
                hexagonal_constructor = HexagonalLatticeConstructor(
                    device='cpu',  # Use CPU for preprocessing auxiliary data
                    modes={resolution: mode},
                    neighbor_rings=args.neighbor_rings,
                    edge_weight=args.edge_weight
                )
                
                # Create mock data structures that the constructor expects
                hex_indices = {resolution: list(filtered_regions.index)}
                regions_by_res = {resolution: filtered_regions}
                
                # Construct hexagonal graphs
                edge_features = hexagonal_constructor.construct_graphs(
                    data_dir, args.city_name, hex_indices, regions_by_res
                )
                
                # Convert to compatible format for saving
                edge_data = edge_features[resolution]
                graph_data = {
                    'edge_index': edge_data.index.cpu().numpy(),
                    'edge_attr': edge_data.accessibility.cpu().numpy(),
                    'num_nodes': len(filtered_regions),
                    'graph_type': 'hexagonal_lattice',
                    'parameters': {
                        'neighbor_rings': args.neighbor_rings,
                        'edge_weight': args.edge_weight
                    },
                    'statistics': {
                        'num_edges': edge_data.index.shape[1],
                        'num_columns': len(filtered_regions),
                        'avg_accessibility': edge_data.accessibility.mean().item()
                    }
                }
            else:
                # Use accessibility-based graphs (original logic)
                network = load_or_download_osm_network(mode, area_gdf)
                graph_data = compute_accessibility_graph(
                    filtered_regions,
                    network,
                    mode,
                    resolution,
                    hierarchy_builder
                )
            
            # Save graph
            if args.graph_type in ['hexagonal', 'hexagonal_lattice']:
                graph_path = output_dir / f"{args.city_name}_{mode}_res{resolution}_hexagonal.pkl"
            else:
                graph_path = output_dir / f"{args.city_name}_{mode}_res{resolution}_hierarchical.pkl"
            with open(graph_path, 'wb') as f:
                pickle.dump(graph_data, f)
            
            print(f"[SAVE] Graph saved to {graph_path}")
            
            # Print statistics
            stats = graph_data['statistics']
            print(f"[STATS] {mode.upper()} graph:")
            print(f"   Nodes: {graph_data['num_nodes']}")
            print(f"   Edges: {stats['num_edges']}")
            print(f"   Columns: {stats['num_columns']}")
            print(f"   Avg degree: {stats['num_edges'] / graph_data['num_nodes']:.1f}")
            print(f"   Avg accessibility: {stats['avg_accessibility']:.1f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {mode}: {str(e)}")
            continue
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Accessibility graphs completed in {elapsed:.1f} seconds!")
    print(f"\n[INFO] Created {len(hierarchy_builder.hierarchical_mapping)} hex columns")
    print(f"\n[FILES] Graphs saved to: {output_dir}")
    print(f"   {args.city_name}_*_res*_hierarchical.pkl")
    print(f"   hierarchical_mapping.pkl")


if __name__ == "__main__":
    main()