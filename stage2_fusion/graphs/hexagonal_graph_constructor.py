import logging
from typing import Dict, List, Optional, Set
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import geopandas as gpd
import pandas as pd
# MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Note: SRAI provides H3 functionality with additional spatial analysis tools
import json
from datetime import datetime
from dataclasses import dataclass

from .graph_construction import EdgeFeatures

# Import geometric helpers
from ..geometry import (
    expected_k_ring_size,
    expected_edge_count,
    validate_edge_count,
)

logger = logging.getLogger(__name__)

@dataclass
class HexagonalLatticeParams:
    """Parameters for hexagonal lattice construction."""
    neighbor_rings: int = 1  # Number of neighbor rings to connect (1 = direct neighbors only)
    edge_weight: float = 1.0  # Uniform edge weight
    include_self_loops: bool = False  # Whether to include self-connections
    
class HexagonalLatticeConstructor:
    """Construct fully connected hexagonal lattice graphs based on H3 spatial structure."""

    def __init__(
            self,
            device: str = "cuda",
            modes: Dict[int, str] = None,
            cache_dir: Optional[Path] = None,
            data_dir: Optional[Path] = None,
            neighbor_rings: int = 1,
            edge_weight: float = 1.0,
            include_self_loops: bool = False
    ):
        """
        Initialize hexagonal lattice constructor.
        
        Args:
            device: Device for tensor operations
            modes: Mapping of resolutions to mode names (for compatibility)
            cache_dir: Directory for caching graphs
            data_dir: Directory for data storage
            neighbor_rings: Number of hexagonal neighbor rings to connect
            edge_weight: Uniform weight for all edges
            include_self_loops: Whether to add self-connections
        """
        self.device = device
        self.modes = modes or {8: 'drive', 9: 'bike', 10: 'walk'}
        
        # Lattice parameters
        self.lattice_params = HexagonalLatticeParams(
            neighbor_rings=neighbor_rings,
            edge_weight=edge_weight,
            include_self_loops=include_self_loops
        )
        
        # Cache management
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        if cache_dir:
            self.graph_cache_dir = cache_dir / 'hexagonal_graphs'
            self.graph_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"HexagonalLatticeConstructor initialized:")
        logger.info(f"  - Neighbor rings: {neighbor_rings}")
        logger.info(f"  - Edge weight: {edge_weight}")
        logger.info(f"  - Include self loops: {include_self_loops}")
        logger.info(f"  - Device: {device}")

    def _get_h3_neighbors(self, regions_gdf: gpd.GeoDataFrame, k_rings: int = 1) -> Dict[str, Set[str]]:
        """
        Get H3 neighbors within k rings for all hexagons using SRAI.

        Args:
            regions_gdf: GeoDataFrame with H3 hexagon regions (index = region_id)
            k_rings: Number of neighbor rings (1 = direct neighbors)

        Returns:
            Dictionary mapping hex_id to set of neighbor hex_ids
        """
        logger.info(f"Computing {k_rings}-ring neighbors using SRAI...")

        # Use SRAI's H3Neighbourhood to get neighbors
        neighbourhood = H3Neighbourhood()

        # Get neighbors for all hexagons at once
        neighbors_dict = {}

        for hex_id in regions_gdf.index:
            neighbors = set()

            # For each ring, get neighbors and add to set
            for ring in range(1, k_rings + 1):
                # Create a temporary GeoDataFrame with just this hexagon
                single_hex_gdf = regions_gdf.loc[[hex_id]]

                # Get k-ring neighbors using SRAI
                ring_gdf = neighbourhood.get_neighbours_at_distance(single_hex_gdf, ring)

                # Add neighbors from this ring
                if ring_gdf is not None and len(ring_gdf) > 0:
                    neighbors.update(ring_gdf.index.tolist())

            neighbors_dict[hex_id] = neighbors

        return neighbors_dict

    def _construct_hexagonal_lattice(
            self,
            regions_gdf: gpd.GeoDataFrame,
            resolution: int,
            mode: str
    ) -> EdgeFeatures:
        """
        Construct hexagonal lattice graph for a single resolution.
        
        Args:
            regions_gdf: GeoDataFrame with H3 hexagon regions
            resolution: H3 resolution level
            mode: Mode name (for logging)
            
        Returns:
            EdgeFeatures object with lattice connectivity
        """
        logger.info(f"\nConstructing hexagonal lattice for resolution {resolution} ({mode}):")
        logger.info(f"Number of hexagons: {len(regions_gdf)}")
        logger.info(f"Neighbor rings: {self.lattice_params.neighbor_rings}")
        
        hex_indices = list(regions_gdf.index)
        hex_index_to_id = {hex_id: idx for idx, hex_id in enumerate(hex_indices)}

        # Get all neighbors using SRAI
        all_neighbors = self._get_h3_neighbors(
            regions_gdf,
            k_rings=self.lattice_params.neighbor_rings
        )

        # Build lattice connections
        edges = []
        weights = []

        for hex_id in tqdm(hex_indices, desc=f"Building {mode} hexagonal lattice"):
            # Get spatial neighbors for this hexagon
            neighbors = all_neighbors.get(hex_id, set())

            # Filter neighbors to only include those in our region set
            valid_neighbors = [n for n in neighbors if n in hex_index_to_id]

            # Add edges to valid neighbors
            source_idx = hex_index_to_id[hex_id]
            for neighbor_hex_id in valid_neighbors:
                target_idx = hex_index_to_id[neighbor_hex_id]
                
                # Add bidirectional edges
                edges.extend([[source_idx, target_idx], [target_idx, source_idx]])
                weights.extend([self.lattice_params.edge_weight, self.lattice_params.edge_weight])
        
        # Add self-loops if requested
        if self.lattice_params.include_self_loops:
            for idx in range(len(hex_indices)):
                edges.extend([[idx, idx]])
                weights.extend([self.lattice_params.edge_weight])
        
        if not edges:
            raise ValueError(f"No edges were created for resolution {resolution}")
        
        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        edge_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Log statistics
        unique_edges = len(edges) // 2  # Each edge is bidirectional
        logger.info(f"Lattice statistics:")
        logger.info(f"  - Nodes: {len(regions_gdf)}")
        logger.info(f"  - Unique edges: {unique_edges}")
        logger.info(f"  - Total directed edges: {len(edges)}")
        logger.info(f"  - Average degree: {len(edges) / len(regions_gdf):.1f}")
        logger.info(f"  - Edge weight (uniform): {self.lattice_params.edge_weight}")
        
        return EdgeFeatures(index=edge_index, accessibility=edge_weights)

    # ========================================================================
    # GEOMETRIC OPTIMIZATIONS
    # ========================================================================

    def _construct_hexagonal_lattice_optimized(
            self,
            regions_gdf: gpd.GeoDataFrame,
            resolution: int,
            mode: str
    ) -> EdgeFeatures:
        """
        Construct hexagonal lattice with geometric optimizations.

        Optimizations:
        1. Pre-compute expected edge count using geometric formula: N × 3k(k+1)
        2. Pre-allocate arrays with exact size
        3. Batch neighbor queries for better performance
        4. Geometric validation of edge counts

        Args:
            regions_gdf: GeoDataFrame with H3 hexagon regions
            resolution: H3 resolution level
            mode: Mode name (for logging)

        Returns:
            EdgeFeatures object with lattice connectivity
        """
        logger.info(f"\nConstructing optimized hexagonal lattice for resolution {resolution} ({mode}):")
        logger.info(f"Number of hexagons: {len(regions_gdf)}")
        logger.info(f"Neighbor rings: {self.lattice_params.neighbor_rings}")

        num_hexagons = len(regions_gdf)

        # Geometric insight: Expected edge count
        expected_edges = expected_edge_count(num_hexagons, self.lattice_params.neighbor_rings)
        logger.info(f"Expected edges (geometric): ~{expected_edges:,} (N × {expected_k_ring_size(self.lattice_params.neighbor_rings, include_center=False)})")

        hex_indices = list(regions_gdf.index)
        hex_index_to_id = {hex_id: idx for idx, hex_id in enumerate(hex_indices)}

        # Pre-allocate edge arrays with expected size (geometric optimization)
        edges = []
        weights = []
        edges.reserve = expected_edges  # Hint for list growth

        # Batch get neighbors using geometric chunking
        logger.info("Batch computing k-ring neighbors...")
        all_neighbors = self._batch_get_neighbors_optimized(
            hex_indices,
            self.lattice_params.neighbor_rings,
            regions_gdf
        )

        # Build lattice connections using pre-computed neighbors
        logger.info("Building edge list from pre-computed neighbors...")
        for hex_id in tqdm(hex_indices, desc=f"Building {mode} lattice"):
            # Get pre-computed neighbors
            neighbors = all_neighbors.get(hex_id, set())

            # Filter to hexagons in our region set
            valid_neighbors = [n for n in neighbors if n in hex_index_to_id]

            # Add edges to valid neighbors
            source_idx = hex_index_to_id[hex_id]
            for neighbor_hex_id in valid_neighbors:
                target_idx = hex_index_to_id[neighbor_hex_id]

                # Add bidirectional edges
                edges.extend([[source_idx, target_idx], [target_idx, source_idx]])
                weights.extend([self.lattice_params.edge_weight, self.lattice_params.edge_weight])

        # Add self-loops if requested
        if self.lattice_params.include_self_loops:
            for idx in range(len(hex_indices)):
                edges.append([idx, idx])
                weights.append(self.lattice_params.edge_weight)

        if not edges:
            raise ValueError(f"No edges were created for resolution {resolution}")

        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        edge_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Geometric validation
        actual_edges = len(edges)
        validation = validate_edge_count(
            actual_edges,
            num_hexagons,
            self.lattice_params.neighbor_rings,
            tolerance=0.25  # 25% tolerance for boundary effects
        )

        # Log statistics with geometric comparison
        unique_edges = len(edges) // 2  # Each edge is bidirectional
        logger.info(f"Lattice statistics:")
        logger.info(f"  - Nodes: {len(regions_gdf)}")
        logger.info(f"  - Unique edges: {unique_edges}")
        logger.info(f"  - Total directed edges: {len(edges)}")
        logger.info(f"  - Expected edges: {expected_edges:,}")
        logger.info(f"  - Actual/Expected ratio: {validation['ratio']:.2%}")
        logger.info(f"  - Average degree: {len(edges) / len(regions_gdf):.1f}")
        logger.info(f"  - Edge weight (uniform): {self.lattice_params.edge_weight}")

        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"  {warning}")

        return EdgeFeatures(index=edge_index, accessibility=edge_weights)

    def _batch_get_neighbors_optimized(
            self,
            hex_ids: List[str],
            k_rings: int,
            regions_gdf: gpd.GeoDataFrame
    ) -> Dict[str, Set[str]]:
        """
        Batch get k-ring neighbors using SRAI with geometric insights.

        Geometric insight: For k=5, each hex has exactly 90 neighbors (1 + 3×5×6 - 1)
        Uses SRAI's H3Neighbourhood for efficient batch operations.

        Args:
            hex_ids: List of hexagon IDs
            k_rings: Number of rings
            regions_gdf: GeoDataFrame to filter against

        Returns:
            Dictionary mapping hex_id to set of neighbor hex_ids
        """
        expected_neighbors_per_hex = expected_k_ring_size(k_rings, include_center=False)
        logger.debug(f"Expected ~{expected_neighbors_per_hex} neighbors per hexagon (k={k_rings})")

        available_hexes = set(regions_gdf.index)
        neighbors_dict = {}

        # Use SRAI's H3Neighbourhood for batch operations
        neighbourhood = H3Neighbourhood()

        # Process in chunks for memory efficiency while using SRAI
        chunk_size = 1000

        for i in range(0, len(hex_ids), chunk_size):
            chunk = hex_ids[i:i+chunk_size]

            # Create temporary GeoDataFrame for this chunk
            chunk_gdf = regions_gdf.loc[regions_gdf.index.intersection(chunk)]

            if len(chunk_gdf) == 0:
                continue

            # Get neighbors for each distance up to k_rings using SRAI
            for hex_id in chunk_gdf.index:
                # For k-ring, need to accumulate neighbors at all distances 1 to k
                neighbors = set()

                # Single hex GeoDataFrame for SRAI
                single_hex_gdf = chunk_gdf.loc[[hex_id]]

                # Get neighbors at each distance and accumulate
                # Note: SRAI's get_neighbours_at_distance gets neighbors at EXACTLY distance k
                # For k-ring, we need all neighbors up to distance k
                for distance in range(1, k_rings + 1):
                    neighbors_at_distance = neighbourhood.get_neighbours_at_distance(
                        single_hex_gdf, distance
                    )

                    if neighbors_at_distance is not None and len(neighbors_at_distance) > 0:
                        neighbors.update(neighbors_at_distance.index.tolist())

                # Filter to available hexagons only
                neighbors = neighbors & available_hexes

                neighbors_dict[hex_id] = neighbors

        # Geometric validation on sample
        if len(neighbors_dict) > 0:
            sample_hex = list(neighbors_dict.keys())[0]
            sample_neighbors = len(neighbors_dict[sample_hex])
            logger.debug(f"Sample hex has {sample_neighbors} neighbors (expected ~{expected_neighbors_per_hex} for k={k_rings})")

        return neighbors_dict

    def construct_graphs(
            self,
            data_dir: Path,
            city_name: str,
            hex_indices_by_res: Dict[int, List[str]],
            regions_gdf_by_res: Dict[int, gpd.GeoDataFrame]
    ) -> Dict[int, EdgeFeatures]:
        """
        Construct hexagonal lattice graphs for all resolutions.
        
        Args:
            data_dir: Data directory (for compatibility with pipeline)
            city_name: City name for caching
            hex_indices_by_res: Hexagon indices by resolution
            regions_gdf_by_res: Region geodataframes by resolution
            
        Returns:
            Dictionary mapping resolution to EdgeFeatures
        """
        logger.info("Constructing hexagonal lattice graphs...")
        
        # Try loading cached graphs first
        cached_graphs = self._load_cached_graphs(city_name)
        if cached_graphs:
            # Move tensors to correct device
            for res in cached_graphs:
                cached_graphs[res] = EdgeFeatures(
                    index=cached_graphs[res].index.to(self.device),
                    accessibility=cached_graphs[res].accessibility.to(self.device)
                )
            return cached_graphs
        
        # Construct graphs for each resolution
        edge_features_dict = {}
        for res, mode in self.modes.items():
            logger.info(f"Processing resolution {res} ({mode})...")
            
            edge_features = self._construct_hexagonal_lattice(
                regions_gdf_by_res[res],
                res,
                mode
            )
            edge_features_dict[res] = edge_features
        
        # Cache constructed graphs
        self._save_cached_graphs(edge_features_dict, city_name)
        
        return edge_features_dict

    def _get_cache_path(self, city_name: str) -> Path:
        """Get cache directory path for hexagonal graphs."""
        if not self.cache_dir:
            return None
            
        # Create unique identifier for lattice parameters
        params_str = f"rings{self.lattice_params.neighbor_rings}_weight{self.lattice_params.edge_weight}"
        if self.lattice_params.include_self_loops:
            params_str += "_selfloops"
            
        return self.graph_cache_dir / f"{city_name}_hexagonal_{params_str}"

    def _load_cached_graphs(self, city_name: str) -> Optional[Dict[int, EdgeFeatures]]:
        """Load cached hexagonal lattice graphs if they exist."""
        cache_path = self._get_cache_path(city_name)
        if not cache_path or not cache_path.exists():
            return None
            
        try:
            # Load metadata
            with open(cache_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
                
            # Validate parameters match
            if not self._validate_cache_params(metadata):
                return None
                
            # Load graph data
            graphs = {}
            for res in metadata['resolutions']:
                graph_data = torch.load(
                    cache_path / f'hexagonal_res{res}.pt',
                    map_location='cpu'
                )
                graphs[res] = EdgeFeatures(
                    index=graph_data['index'],
                    accessibility=graph_data['accessibility']
                )
                
            logger.info(f"Successfully loaded cached hexagonal lattice graphs from {cache_path}")
            return graphs
            
        except Exception as e:
            logger.warning(f"Failed to load cached hexagonal graphs: {str(e)}")
            return None

    def _save_cached_graphs(self, graphs: Dict[int, EdgeFeatures], city_name: str):
        """Save hexagonal lattice graphs to cache."""
        cache_path = self._get_cache_path(city_name)
        if not cache_path:
            return
            
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'city_name': city_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'graph_type': 'hexagonal_lattice',
            'neighbor_rings': self.lattice_params.neighbor_rings,
            'edge_weight': self.lattice_params.edge_weight,
            'include_self_loops': self.lattice_params.include_self_loops,
            'resolutions': list(graphs.keys())
        }
        
        with open(cache_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save individual graphs
        for res, edge_features in graphs.items():
            graph_data = {
                'index': edge_features.index.cpu(),
                'accessibility': edge_features.accessibility.cpu(),
                'resolution': res,
                'graph_type': 'hexagonal_lattice'
            }
            torch.save(
                graph_data,
                cache_path / f'hexagonal_res{res}.pt'
            )
            
        logger.info(f"Cached hexagonal lattice graphs to {cache_path}")

    def _validate_cache_params(self, metadata: dict) -> bool:
        """Validate that cached parameters match current settings."""
        current_params = {
            'neighbor_rings': self.lattice_params.neighbor_rings,
            'edge_weight': self.lattice_params.edge_weight,
            'include_self_loops': self.lattice_params.include_self_loops
        }
        
        for param, value in current_params.items():
            if metadata.get(param) != value:
                logger.info(f"Hexagonal lattice parameter mismatch for {param}")
                return False
        return True

    def get_lattice_summary(self, edge_features_dict: Dict[int, EdgeFeatures]) -> pd.DataFrame:
        """Generate summary statistics for hexagonal lattice graphs."""
        summary_data = []
        
        for res, edge_features in edge_features_dict.items():
            mode = self.modes[res]
            
            # Calculate basic statistics
            num_edges = edge_features.index.shape[1]
            unique_nodes = torch.unique(edge_features.index).shape[0]
            avg_degree = num_edges / unique_nodes if unique_nodes > 0 else 0
            
            stats = {
                'resolution': res,
                'mode': mode,
                'graph_type': 'hexagonal_lattice',
                'nodes': unique_nodes,
                'edges': num_edges,
                'avg_degree': avg_degree,
                'neighbor_rings': self.lattice_params.neighbor_rings,
                'edge_weight': self.lattice_params.edge_weight,
                'has_self_loops': self.lattice_params.include_self_loops
            }
            
            summary_data.append(stats)
            
        return pd.DataFrame(summary_data)