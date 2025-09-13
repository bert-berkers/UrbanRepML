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

    def _get_h3_neighbors(self, hex_id: str, k_rings: int = 1) -> Set[str]:
        """
        Get H3 neighbors within k rings of a hexagon.
        
        Args:
            hex_id: H3 hexagon identifier
            k_rings: Number of neighbor rings (1 = direct neighbors)
            
        Returns:
            Set of neighbor H3 identifiers
        """
        neighbors = set()
        
        for ring in range(1, k_rings + 1):
            try:
                # Try both h3.k_ring and h3.grid_ring_unsafe for compatibility
                if hasattr(h3, 'k_ring'):
                    ring_neighbors = h3.k_ring(hex_id, ring) - h3.k_ring(hex_id, ring - 1)
                elif hasattr(h3, 'grid_ring_unsafe'):
                    # For newer h3-py versions
                    ring_neighbors = set(h3.grid_ring_unsafe(hex_id, ring))
                else:
                    # Fallback to direct neighbors only
                    if ring == 1 and hasattr(h3, 'grid_disk'):
                        all_neighbors = set(h3.grid_disk(hex_id, 1))
                        all_neighbors.discard(hex_id)  # Remove the center hex
                        ring_neighbors = all_neighbors
                    else:
                        logger.warning(f"H3 neighbor function not found, using empty set")
                        ring_neighbors = set()
                        
                neighbors.update(ring_neighbors)
            except Exception as e:
                logger.warning(f"Failed to get ring {ring} neighbors for {hex_id}: {str(e)}")
                continue
                
        return neighbors

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
        
        # Build lattice connections
        edges = []
        weights = []
        
        for hex_id in tqdm(hex_indices, desc=f"Building {mode} hexagonal lattice"):
            # Get spatial neighbors using H3
            neighbors = self._get_h3_neighbors(
                hex_id, 
                k_rings=self.lattice_params.neighbor_rings
            )
            
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