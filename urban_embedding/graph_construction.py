import logging
from typing import Dict, List, Optional
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
from dataclasses import dataclass
import pickle
import json
from datetime import datetime
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class EdgeFeatures:
    """Container for edge features with type checking."""
    index: torch.Tensor
    accessibility: torch.Tensor

    def __post_init__(self):
        """Validate tensor types after initialization."""
        if self.index.dtype != torch.long:
            raise TypeError(f"Edge index must be torch.long, got {self.index.dtype}")
        if self.accessibility.dtype != torch.float32:
            self.accessibility = self.accessibility.float()

class CacheManager:
    """Manages caching for network graphs and accessibility graphs."""

    def __init__(self, cache_dir: Path, data_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        # Use data/networks for organized storage, cache/networks as fallback
        if data_dir:
            self.network_cache_dir = data_dir / 'networks' / 'osm'
            self.accessibility_cache_dir = data_dir / 'networks' / 'accessibility'
        else:
            self.network_cache_dir = cache_dir / 'networks' / 'osm'
            self.accessibility_cache_dir = cache_dir / 'networks' / 'accessibility'
        
        self.graph_cache_dir = cache_dir / 'graphs'

        # Create cache directories
        self.network_cache_dir.mkdir(parents=True, exist_ok=True)
        self.accessibility_cache_dir.mkdir(parents=True, exist_ok=True)
        self.graph_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_network_cache_path(self, city_name: str, mode: str) -> Path:
        """Get cache path for OSM network."""
        return self.network_cache_dir / f"{city_name}_{mode}_network.pkl"

    def get_graph_cache_path(self, city_name: str, graph_params: dict) -> Path:
        """Get cache path for accessibility graphs."""
        param_str = json.dumps(graph_params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return self.graph_cache_dir / f"{city_name}_graphs_{param_hash}"

    def load_network(self, city_name: str, mode: str) -> Optional[nx.Graph]:
        """Load cached OSM network if exists."""
        cache_path = self.get_network_cache_path(city_name, mode)
        if cache_path.exists():
            logger.info(f"Loading cached {mode} network from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_network(self, network: nx.Graph, city_name: str, mode: str):
        """Cache OSM network."""
        cache_path = self.get_network_cache_path(city_name, mode)
        logger.info(f"Caching {mode} network to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(network, f)

    def load_graphs(self, city_name: str, graph_params: dict) -> Optional[Dict[int, EdgeFeatures]]:
        """Load cached accessibility graphs if exist and params match."""
        graphs_dir = self.get_graph_cache_path(city_name, graph_params)
        if not graphs_dir.exists():
            return None

        try:
            # Load and validate metadata
            with open(graphs_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            # Validate parameters match
            if not self._validate_graph_params(metadata, graph_params):
                return None

            # Load graph data
            graphs = {}
            for res in metadata['resolutions']:
                graph_data = torch.load(
                    graphs_dir / f'graph_res{res}.pt',
                    map_location='cpu'
                )
                graphs[res] = EdgeFeatures(
                    index=graph_data['index'],
                    accessibility=graph_data['accessibility']
                )

            logger.info(f"Successfully loaded cached graphs from {graphs_dir}")
            return graphs

        except Exception as e:
            logger.warning(f"Failed to load cached graphs: {str(e)}")
            return None

    def save_graphs(
            self,
            graphs: Dict[int, EdgeFeatures],
            city_name: str,
            graph_params: dict
    ):
        """Cache accessibility graphs with metadata."""
        graphs_dir = self.get_graph_cache_path(city_name, graph_params)
        graphs_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            'city_name': city_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            **graph_params,
            'resolutions': list(graphs.keys())
        }

        with open(graphs_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save individual graphs
        for res, edge_features in graphs.items():
            graph_data = {
                'index': edge_features.index.cpu(),
                'accessibility': edge_features.accessibility.cpu(),
                'resolution': res
            }
            torch.save(
                graph_data,
                graphs_dir / f'graph_res{res}.pt'
            )

        logger.info(f"Cached graphs to {graphs_dir}")

    def load_accessibility_graph(self, city_name: str, resolution: int, mode: str) -> Optional[EdgeFeatures]:
        """Load accessibility graph from data/networks/accessibility structure."""
        graph_path = self.accessibility_cache_dir / f"{city_name}_{mode}_res{resolution}.pkl"
        
        if graph_path.exists():
            logger.info(f"Loading accessibility graph from {graph_path}")
            try:
                import pickle
                with open(graph_path, 'rb') as f:
                    graph_data = pickle.load(f)
                
                # Convert to EdgeFeatures format
                if hasattr(graph_data, 'edge_index') and hasattr(graph_data, 'edge_attr'):
                    return EdgeFeatures(
                        index=torch.tensor(graph_data.edge_index, dtype=torch.long),
                        accessibility=torch.tensor(graph_data.edge_attr, dtype=torch.float32)
                    )
                elif isinstance(graph_data, dict):
                    return EdgeFeatures(
                        index=torch.tensor(graph_data['edge_index'], dtype=torch.long),
                        accessibility=torch.tensor(graph_data['edge_attr'], dtype=torch.float32)
                    )
                else:
                    logger.warning(f"Unknown graph format in {graph_path}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error loading accessibility graph from {graph_path}: {str(e)}")
                return None
        
        return None

    def _validate_graph_params(self, metadata: dict, current_params: dict) -> bool:
        """Validate that cached graph parameters match current parameters."""
        for key in ['speeds', 'max_travel_time', 'search_radius', 'beta']:
            if metadata.get(key) != current_params.get(key):
                logger.info(f"Parameter mismatch for {key}")
                return False
        return True

class SpatialGraphConstructor:
    """Construct multi-scale spatial graphs with accessibility features."""

    def __init__(
            self,
            device: str = "cuda",
            modes: Dict[int, str] = None,
            cache_dir: Optional[Path] = None,
            data_dir: Optional[Path] = None,
            speeds: Dict[str, float] = None,
            max_travel_time: Dict[str, int] = None,
            search_radius: Dict[str, float] = None,
            beta: Dict[str, float] = None
    ):
        """Initialize graph constructor."""
        self.device = device
        self.modes = modes or {8: 'drive', 9: 'bike', 10: 'walk'}

        # Store graph parameters
        self.graph_params = {
            'speeds': speeds or {'walk': 1.4, 'bike': 4.17, 'drive': 11.11},
            'max_travel_time': max_travel_time or {'walk': 300, 'bike': 600, 'drive': 900},
            'search_radius': search_radius or {'walk': 200, 'bike': 300, 'drive': 500},
            'beta': beta or {'walk': 0.0020, 'bike': 0.0012, 'drive': 0.0008}
        }

        # Initialize cache manager if cache_dir provided
        self.cache_manager = CacheManager(cache_dir, data_dir) if cache_dir else None

        self._validate_parameters()

    def _validate_parameters(self):
        """Validate configuration parameters."""
        required_modes = set(mode for res, mode in self.modes.items())

        for param_name, param_dict in self.graph_params.items():
            missing_modes = required_modes - set(param_dict.keys())
            if missing_modes:
                raise ValueError(f"Missing {param_name} for modes: {missing_modes}")

    def _load_study_area(self, data_dir: Path, city_name: str) -> gpd.GeoDataFrame:
        """Load study area boundary."""
        study_area_path = data_dir / 'data' / 'preprocessed [TODO SORT & CLEAN UP]' / city_name / 'area_study_gdf.parquet'
        try:
            study_area = gpd.read_parquet(study_area_path)
            logger.info(f"Loaded study area boundary from {study_area_path}")
            return study_area
        except Exception as e:
            logger.error(f"Failed to load study area from {study_area_path}: {str(e)}")
            raise FileNotFoundError(f"Study area file not found at {study_area_path}")

    def _create_network(self, mode: str, study_area: gpd.GeoDataFrame) -> nx.Graph:
        """Create transport network for given mode."""
        logger.info(f"Creating {mode} network...")

        # Buffer the study area using mode-specific search radius
        search_radius = self.graph_params['search_radius'][mode]
        logger.info(f"Using search radius of {search_radius}m for {mode} mode")

        area_rd = study_area.to_crs('EPSG:28992')
        area_buffered = area_rd.buffer(search_radius).unary_union
        area_wgs84 = gpd.GeoSeries([area_buffered], crs='EPSG:28992').to_crs('EPSG:4326')[0]

        # Download network
        logger.info(f"Downloading {mode} network from OpenStreetMap...")
        G = ox.graph_from_polygon(
            area_wgs84,
            network_type='drive' if mode == 'drive' else 'all',
            simplify=True,
            truncate_by_edge=True
        )
        logger.info(f"Downloaded network with {len(G.nodes)} nodes and {len(G.edges)} edges")

        # Calculate travel times
        speed = self.graph_params['speeds'][mode]
        logger.info(f"Calculating travel times using speed: {speed}m/s")
        for u, v, data in G.edges(data=True):
            data['time'] = data['length'] / speed

        # Ensure network is connected
        if mode == 'drive':
            largest_cc = max(nx.strongly_connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        else:
            G = G.to_undirected()
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        logger.info(f"Final connected network: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G

    def _construct_accessibility_graph(
            self,
            G: nx.Graph,
            regions_gdf: gpd.GeoDataFrame,
            mode: str,
            max_time: int,
            beta: float
    ) -> EdgeFeatures:
        """Construct accessibility graph with proper weight handling."""
        logger.info(f"\nConstructing accessibility graph for {mode} mode:")
        logger.info(f"Max travel time: {max_time}s")
        logger.info(f"Beta (impedance decay): {beta}")
        logger.info(f"Number of regions: {len(regions_gdf)}")

        # Project to proper CRS
        regions_gdf = regions_gdf.to_crs('EPSG:28992')
        hex_indices = regions_gdf.index.tolist()
        hex_index_to_id = {hex_id: idx for idx, hex_id in enumerate(hex_indices)}

        # Calculate volumes
        hex_areas = regions_gdf.geometry.area
        volumes = regions_gdf['FSI_24'] * hex_areas

        # Store volumes as Series with proper index
        destination_mass = pd.Series(volumes, index=regions_gdf.index)
        logger.info(f"Volume stats - Min: {volumes.min():.2f}, Max: {volumes.max():.2f}, Mean: {volumes.mean():.2f}")

        # Pre-calculate buffers with proper indexing
        region_buffers = pd.Series(
            regions_gdf.geometry.buffer(self.graph_params['search_radius'][mode]),
            index=regions_gdf.index
        )

        # Get centroids with proper indexing
        centroids = regions_gdf.geometry.centroid
        centroids_wgs = centroids.to_crs('EPSG:4326')
        nodes, distances = ox.nearest_nodes(
            G,
            centroids_wgs.x.values,
            centroids_wgs.y.values,
            return_dist=True
        )

        # Process regions
        edges = []
        weights = []

        for hex_id in tqdm(hex_indices, desc=f"Building {mode} graph"):
            row_idx = hex_index_to_id[hex_id]
            source_node = nodes[row_idx]

            if source_node is None:
                continue

            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G, source_node, weight='time', cutoff=max_time
                )
            except nx.NetworkXNoPath:
                continue

            # Find neighbors using proper index-based access
            source_buffer = region_buffers.loc[hex_id]
            possible_neighbors = regions_gdf[regions_gdf.geometry.intersects(source_buffer)].index

            for neighbor_hex_id in possible_neighbors:
                if neighbor_hex_id == hex_id:
                    continue

                target_idx = hex_index_to_id[neighbor_hex_id]
                target_node = nodes[target_idx]

                if target_node not in lengths:
                    continue

                travel_time = lengths[target_node]
                if travel_time > max_time:
                    continue

                # Calculate accessibility using proper Series indexing
                impedance = np.exp(-beta * travel_time)
                accessibility = float(destination_mass.loc[neighbor_hex_id]) * impedance

                # Add significant edges
                if accessibility > 0.01:
                    source_id = hex_index_to_id[hex_id]
                    target_id = hex_index_to_id[neighbor_hex_id]
                    edges.extend([[source_id, target_id], [target_id, source_id]])
                    weights.extend([accessibility, accessibility])

        if not edges:
            raise ValueError(f"No edges were created for mode {mode}")

        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        edge_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Normalize weights
        edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)

        # Log statistics
        logger.info("\nFinal graph statistics:")
        logger.info(f"Nodes: {len(regions_gdf)}")
        logger.info(f"Edges: {len(edges)//2}")
        logger.info(f"Average edge weight: {edge_weights.mean().item():.3f}")
        logger.info(f"Max edge weight: {edge_weights.max().item():.3f}")
        logger.info(f"Min edge weight: {edge_weights.min().item():.3f}")
        logger.info(f"Average node degree: {len(edges)/len(regions_gdf):.1f}")

        return EdgeFeatures(index=edge_index, accessibility=edge_weights)

    def construct_graphs(
            self,
            data_dir: Path,
            city_name: str,
            hex_indices_by_res: Dict[int, List[str]],
            regions_gdf_by_res: Dict[int, gpd.GeoDataFrame]
    ) -> Dict[int, EdgeFeatures]:
        """Construct multi-scale urban graphs."""
        logger.info("Constructing multi-scale urban graphs...")

        # Try loading cached graphs first
        if self.cache_manager:
            cached_graphs = self.cache_manager.load_graphs(city_name, self.graph_params)
            if cached_graphs:
                # Move tensors to correct device
                for res in cached_graphs:
                    cached_graphs[res] = EdgeFeatures(
                        index=cached_graphs[res].index.to(self.device),
                        accessibility=cached_graphs[res].accessibility.to(self.device)
                    )
                return cached_graphs

        # Load study area for network construction
        study_area = self._load_study_area(data_dir, city_name)

        # Construct graphs for each resolution
        edge_features_dict = {}
        for res, mode in self.modes.items():
            logger.info(f"Processing resolution {res} ({mode})...")

            # Try loading cached network
            network = None
            if self.cache_manager:
                network = self.cache_manager.load_network(city_name, mode)

            # Create and cache network if not found
            if network is None:
                network = self._create_network(mode, study_area)
                if self.cache_manager:
                    self.cache_manager.save_network(network, city_name, mode)

            # Construct accessibility graph
            edge_features = self._construct_accessibility_graph(
                network,
                regions_gdf_by_res[res],
                mode,
                self.graph_params['max_travel_time'][mode],
                self.graph_params['beta'][mode]
            )
            edge_features_dict[res] = edge_features

        # Cache constructed graphs
        if self.cache_manager:
            self.cache_manager.save_graphs(edge_features_dict, city_name, self.graph_params)

        return edge_features_dict

    def visualize_graphs(self, edge_features_dict: Dict[int, EdgeFeatures]):
        """Visualize the constructed graphs and their properties."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from matplotlib.gridspec import GridSpec

            n_graphs = len(edge_features_dict)
            fig = plt.figure(figsize=(6*n_graphs, 10))
            gs = GridSpec(2, n_graphs, figure=fig)

            for idx, (res, edge_features) in enumerate(edge_features_dict.items()):
                mode = self.modes[res]

                # Convert to NetworkX graph for analysis
                G = nx.Graph()
                edges = edge_features.index.cpu().numpy().T
                weights = edge_features.accessibility.cpu().numpy()

                for (src, dst), w in zip(edges, weights):
                    G.add_edge(src, dst, weight=w)

                # Compute graph metrics
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                clustering_coeffs = nx.clustering(G)

                # Plot graph
                ax1 = fig.add_subplot(gs[0, idx])
                pos = nx.spring_layout(G)

                # Draw nodes with size proportional to degree centrality
                node_sizes = [v * 1000 for v in degree_centrality.values()]
                node_colors = list(betweenness_centrality.values())

                nx.draw_networkx_nodes(
                    G, pos,
                    node_size=node_sizes,
                    node_color=node_colors,
                    alpha=0.6,
                    cmap=plt.cm.viridis
                )

                # Draw edges with width proportional to accessibility
                edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                nx.draw_networkx_edges(
                    G, pos,
                    width=[w * 2 for w in edge_weights],
                    alpha=0.3
                )

                ax1.set_title(f'{mode.upper()} Network (h3.{res})')

                # Plot statistics
                ax2 = fig.add_subplot(gs[1, idx])
                stats_text = self._format_graph_stats(G, clustering_coeffs, betweenness_centrality, weights)
                ax2.text(
                    0.05, 0.95, stats_text,
                    transform=ax2.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
                ax2.axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error in graph visualization: {str(e)}")
            raise

    def _format_graph_stats(self, G, clustering_coeffs, betweenness_centrality, weights):
        """Format graph statistics for visualization."""
        return (
            f"Network Statistics:\n"
            f"Nodes: {G.number_of_nodes()}\n"
            f"Edges: {G.number_of_edges()}\n"
            f"Avg Degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}\n"
            f"Density: {nx.density(G):.3f}\n"
            f"Avg Clustering: {sum(clustering_coeffs.values())/len(clustering_coeffs):.3f}\n"
            f"Avg Path Length: {nx.average_shortest_path_length(G, weight='weight'):.2f}\n"
            f"Avg Betweenness: {np.mean(list(betweenness_centrality.values())):.3f}\n"
            f"Max Accessibility: {max(weights):.3f}\n"
            f"Min Accessibility: {min(weights):.3f}"
        )

    def graph_summary(self, edge_features_dict: Dict[int, EdgeFeatures]) -> pd.DataFrame:
        """Generate a summary DataFrame of graph properties."""
        summary_data = []

        for res, edge_features in edge_features_dict.items():
            mode = self.modes[res]

            # Convert to NetworkX
            G = nx.Graph()
            edges = edge_features.index.cpu().numpy().T
            weights = edge_features.accessibility.cpu().numpy()

            for (src, dst), w in zip(edges, weights):
                G.add_edge(src, dst, weight=w)

            # Compute metrics
            stats = {
                'resolution': res,
                'mode': mode,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'avg_degree': 2*G.number_of_edges()/G.number_of_nodes(),
                'density': nx.density(G),
                'avg_clustering': np.mean(list(nx.clustering(G).values())),
                'avg_path_length': nx.average_shortest_path_length(G, weight='weight'),
                'avg_betweenness': np.mean(list(nx.betweenness_centrality(G).values())),
                'max_accessibility': max(weights),
                'min_accessibility': min(weights)
            }

            summary_data.append(stats)

        return pd.DataFrame(summary_data)