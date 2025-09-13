"""
Spatial batching system for memory-efficient training on large-scale datasets.
Implements overlapping spatial batches to enable training on datasets that don't fit in memory.
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
# MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Note: SRAI provides H3 functionality with additional spatial analysis tools
from pathlib import Path
from dataclasses import dataclass
import pickle
import json
from datetime import datetime

from .graph_construction import EdgeFeatures
from .hexagonal_graph_constructor import HexagonalLatticeConstructor

logger = logging.getLogger(__name__)


@dataclass
class SpatialBatch:
    """Container for a spatial batch of hexagons."""
    batch_id: str
    hex_ids: List[str]
    features: torch.Tensor
    edge_index: torch.Tensor
    edge_weights: torch.Tensor
    batch_boundary: Optional[object] = None  # Geometry boundary
    neighbors: Optional[List[str]] = None  # Neighboring batch IDs
    

@dataclass 
class BatchingConfig:
    """Configuration for spatial batching."""
    batch_size: int = 5000  # Target hexagons per batch
    overlap_ratio: float = 0.2  # Overlap between adjacent batches
    min_batch_size: int = 1000  # Minimum hexagons per batch
    max_batch_size: int = 10000  # Maximum hexagons per batch
    grouping_resolution: int = 6  # Resolution for spatial grouping
    boundary_buffer: float = 1000  # Buffer for batch boundaries (meters)
    

class SpatialBatchDataset(Dataset):
    """PyTorch Dataset for spatial batches."""
    
    def __init__(
        self,
        batches: List[SpatialBatch],
        include_batch_info: bool = True
    ):
        """
        Initialize spatial batch dataset.
        
        Args:
            batches: List of SpatialBatch objects
            include_batch_info: Whether to include batch metadata
        """
        self.batches = batches
        self.include_batch_info = include_batch_info
        
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        batch = self.batches[idx]
        
        result = {
            'features': batch.features,
            'edge_index': batch.edge_index,
            'edge_weights': batch.edge_weights,
            'hex_ids': batch.hex_ids
        }
        
        if self.include_batch_info:
            result.update({
                'batch_id': batch.batch_id,
                'neighbors': batch.neighbors or []
            })
            
        return result


class SpatialBatcher:
    """Create and manage spatial batches for large-scale processing embeddings."""
    
    def __init__(
        self,
        config: BatchingConfig = None,
        device: str = "cuda",
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize spatial batcher.
        
        Args:
            config: Batching configuration
            device: Device for tensor operations
            cache_dir: Directory for caching batches
        """
        self.config = config or BatchingConfig()
        self.device = device
        self.cache_dir = cache_dir
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"SpatialBatcher initialized:")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Overlap ratio: {self.config.overlap_ratio}")
        logger.info(f"  Grouping resolution: {self.config.grouping_resolution}")
    
    def create_spatial_batches(
        self,
        regions_gdf: gpd.GeoDataFrame,
        embeddings_df: pd.DataFrame
    ) -> List[SpatialBatch]:
        """
        Create spatial batches from regions and embeddings.
        
        Args:
            regions_gdf: GeoDataFrame with H3 hexagon regions
            embeddings_df: DataFrame with embeddings for hexagons
            
        Returns:
            List of SpatialBatch objects
        """
        logger.info("Creating spatial batches...")
        
        # Try loading cached batches
        if self.cache_dir:
            cached_batches = self._load_cached_batches()
            if cached_batches:
                return cached_batches
        
        # Align regions and embeddings
        common_hex_ids = list(
            set(regions_gdf.index).intersection(set(embeddings_df.index))
        )
        
        if not common_hex_ids:
            raise ValueError("No common hex_ids between regions and embeddings")
        
        logger.info(f"Processing {len(common_hex_ids)} hexagons")
        
        # Create spatial groups using parent hexagons
        spatial_groups = self._create_spatial_groups(common_hex_ids)
        
        # Convert spatial groups to batches
        batches = self._groups_to_batches(spatial_groups, regions_gdf, embeddings_df)
        
        # Add batch connectivity information
        batches = self._add_batch_connectivity(batches, regions_gdf)
        
        # Cache batches
        if self.cache_dir:
            self._cache_batches(batches)
        
        logger.info(f"Created {len(batches)} spatial batches")
        
        return batches
    
    def _create_spatial_groups(self, hex_ids: List[str]) -> Dict[str, List[str]]:
        """
        Group hexagons spatially using parent hexagons at lower resolution.
        
        Args:
            hex_ids: List of H3 hex_ids to group
            
        Returns:
            Dictionary mapping parent_id to list of child hex_ids
        """
        logger.info("Creating spatial groups...")
        
        parent_to_children = {}
        
        for hex_id in hex_ids:
            try:
                parent_id = h3.h3_to_parent(hex_id, self.config.grouping_resolution)
                if parent_id not in parent_to_children:
                    parent_to_children[parent_id] = []
                parent_to_children[parent_id].append(hex_id)
            except Exception as e:
                logger.warning(f"Error getting parent for {hex_id}: {str(e)}")
                continue
        
        logger.info(f"Created {len(parent_to_children)} spatial groups")
        
        return parent_to_children
    
    def _groups_to_batches(
        self,
        spatial_groups: Dict[str, List[str]],
        regions_gdf: gpd.GeoDataFrame,
        embeddings_df: pd.DataFrame
    ) -> List[SpatialBatch]:
        """
        Convert spatial groups to batches with size constraints.
        
        Args:
            spatial_groups: Dictionary of spatial groups
            regions_gdf: GeoDataFrame with regions
            embeddings_df: DataFrame with embeddings
            
        Returns:
            List of SpatialBatch objects
        """
        logger.info("Converting spatial groups to batches...")
        
        batches = []
        current_batch_hexes = []
        current_batch_parents = []
        batch_counter = 0
        
        for parent_id, child_hexes in spatial_groups.items():
            # Check if adding this group exceeds batch size
            potential_size = len(current_batch_hexes) + len(child_hexes)
            
            if (potential_size > self.config.batch_size and 
                len(current_batch_hexes) >= self.config.min_batch_size):
                
                # Create batch from current accumulation
                batch = self._create_batch_from_hexes(
                    current_batch_hexes,
                    regions_gdf,
                    embeddings_df,
                    batch_id=f"batch_{batch_counter:04d}",
                    parent_ids=current_batch_parents
                )
                batches.append(batch)
                batch_counter += 1
                
                # Start new batch
                current_batch_hexes = child_hexes.copy()
                current_batch_parents = [parent_id]
                
            else:
                # Add to current batch
                current_batch_hexes.extend(child_hexes)
                current_batch_parents.append(parent_id)
        
        # Create final batch if there are remaining hexes
        if len(current_batch_hexes) >= self.config.min_batch_size:
            batch = self._create_batch_from_hexes(
                current_batch_hexes,
                regions_gdf,
                embeddings_df,
                batch_id=f"batch_{batch_counter:04d}",
                parent_ids=current_batch_parents
            )
            batches.append(batch)
        elif current_batch_hexes and batches:
            # Add remaining hexes to last batch if it's small
            logger.info(f"Adding {len(current_batch_hexes)} remaining hexes to last batch")
            last_batch = batches[-1]
            extended_hexes = last_batch.hex_ids + current_batch_hexes
            
            # Recreate last batch with extended hexes
            batches[-1] = self._create_batch_from_hexes(
                extended_hexes,
                regions_gdf,
                embeddings_df,
                batch_id=last_batch.batch_id,
                parent_ids=current_batch_parents
            )
        
        return batches
    
    def _create_batch_from_hexes(
        self,
        hex_ids: List[str],
        regions_gdf: gpd.GeoDataFrame,
        embeddings_df: pd.DataFrame,
        batch_id: str,
        parent_ids: List[str] = None
    ) -> SpatialBatch:
        """
        Create a SpatialBatch from list of hex_ids.
        
        Args:
            hex_ids: List of hex_ids for this batch
            regions_gdf: GeoDataFrame with regions
            embeddings_df: DataFrame with embeddings
            batch_id: Unique identifier for batch
            parent_ids: List of parent hex_ids used for grouping
            
        Returns:
            SpatialBatch object
        """
        # Filter to available hexagons
        available_hexes = [
            h for h in hex_ids 
            if h in regions_gdf.index and h in embeddings_df.index
        ]
        
        if not available_hexes:
            raise ValueError(f"No valid hexagons for batch {batch_id}")
        
        # Get features
        features_df = embeddings_df.loc[available_hexes]
        features_tensor = torch.tensor(
            features_df.values,
            dtype=torch.float32,
            device=self.device
        )
        
        # Create hexagonal lattice graph for this batch
        batch_regions = regions_gdf.loc[available_hexes]
        
        # Use hexagonal lattice constructor
        lattice_constructor = HexagonalLatticeConstructor(
            device=self.device,
            neighbor_rings=1,  # Direct neighbors only
            edge_weight=1.0
        )
        
        # Create temporary mapping for lattice construction
        hex_indices_dict = {10: available_hexes}  # Use resolution 10 as default
        regions_dict = {10: batch_regions}
        
        # Construct lattice graph
        edge_features_dict = lattice_constructor.construct_graphs(
            data_dir=Path("temp"),  # Not used for caching in this context
            city_name=batch_id,
            hex_indices_by_res=hex_indices_dict,
            regions_gdf_by_res=regions_dict
        )
        
        edge_features = edge_features_dict[10]
        
        # Create batch boundary
        batch_boundary = batch_regions.unary_union
        
        return SpatialBatch(
            batch_id=batch_id,
            hex_ids=available_hexes,
            features=features_tensor,
            edge_index=edge_features.index,
            edge_weights=edge_features.accessibility,
            batch_boundary=batch_boundary
        )
    
    def _add_batch_connectivity(
        self,
        batches: List[SpatialBatch],
        regions_gdf: gpd.GeoDataFrame
    ) -> List[SpatialBatch]:
        """
        Add connectivity information between batches.
        
        Args:
            batches: List of SpatialBatch objects
            regions_gdf: GeoDataFrame with regions
            
        Returns:
            List of batches with neighbor information
        """
        logger.info("Adding batch connectivity information...")
        
        # Create spatial index for batch boundaries
        batch_boundaries = {}
        for batch in batches:
            if batch.batch_boundary is not None:
                batch_boundaries[batch.batch_id] = batch.batch_boundary
        
        # Find neighboring batches
        for i, batch in enumerate(batches):
            if batch.batch_boundary is None:
                continue
                
            neighbors = []
            batch_geom = batch.batch_boundary
            
            for other_batch in batches:
                if (other_batch.batch_id == batch.batch_id or 
                    other_batch.batch_boundary is None):
                    continue
                
                # Check if batches are adjacent (share boundary or are close)
                try:
                    distance = batch_geom.distance(other_batch.batch_boundary)
                    if distance < 0.01:  # Very close/touching (in degrees)
                        neighbors.append(other_batch.batch_id)
                except Exception as e:
                    logger.debug(f"Error calculating distance between batches: {str(e)}")
            
            # Update batch with neighbor information
            batches[i] = SpatialBatch(
                batch_id=batch.batch_id,
                hex_ids=batch.hex_ids,
                features=batch.features,
                edge_index=batch.edge_index,
                edge_weights=batch.edge_weights,
                batch_boundary=batch.batch_boundary,
                neighbors=neighbors
            )
        
        return batches
    
    def create_overlapping_batches(
        self,
        base_batches: List[SpatialBatch],
        regions_gdf: gpd.GeoDataFrame,
        embeddings_df: pd.DataFrame
    ) -> List[SpatialBatch]:
        """
        Create overlapping batches to handle boundary effects.
        
        Args:
            base_batches: Base non-overlapping batches
            regions_gdf: GeoDataFrame with regions
            embeddings_df: DataFrame with embeddings
            
        Returns:
            List of overlapping batches
        """
        logger.info("Creating overlapping batches...")
        
        overlapping_batches = []
        
        for batch in base_batches:
            if not batch.neighbors:
                # No neighbors, use original batch
                overlapping_batches.append(batch)
                continue
            
            # Calculate overlap size
            overlap_size = int(len(batch.hex_ids) * self.config.overlap_ratio)
            
            if overlap_size == 0:
                overlapping_batches.append(batch)
                continue
            
            # Find hexagons from neighboring batches to include
            neighbor_hexes = []
            for neighbor_id in batch.neighbors:
                neighbor_batch = next(
                    (b for b in base_batches if b.batch_id == neighbor_id),
                    None
                )
                if neighbor_batch:
                    # Add some hexagons from neighbor
                    n_to_add = min(overlap_size // len(batch.neighbors), 
                                 len(neighbor_batch.hex_ids) // 4)
                    neighbor_hexes.extend(neighbor_batch.hex_ids[:n_to_add])
            
            # Create extended batch
            extended_hex_ids = batch.hex_ids + neighbor_hexes
            
            try:
                overlapping_batch = self._create_batch_from_hexes(
                    extended_hex_ids,
                    regions_gdf,
                    embeddings_df,
                    batch_id=f"{batch.batch_id}_overlap"
                )
                overlapping_batches.append(overlapping_batch)
            except Exception as e:
                logger.warning(f"Error creating overlapping batch {batch.batch_id}: {str(e)}")
                overlapping_batches.append(batch)
        
        logger.info(f"Created {len(overlapping_batches)} overlapping batches")
        
        return overlapping_batches
    
    def _cache_batches(self, batches: List[SpatialBatch]):
        """Cache batches to disk."""
        if not self.cache_dir:
            return
            
        cache_path = self.cache_dir / "spatial_batches.pkl"
        
        # Prepare batches for caching (remove geometry which might not pickle well)
        cacheable_batches = []
        for batch in batches:
            cacheable_batch = SpatialBatch(
                batch_id=batch.batch_id,
                hex_ids=batch.hex_ids,
                features=batch.features.cpu(),  # Move to CPU for caching
                edge_index=batch.edge_index.cpu(),
                edge_weights=batch.edge_weights.cpu(),
                batch_boundary=None,  # Don't cache geometry
                neighbors=batch.neighbors
            )
            cacheable_batches.append(cacheable_batch)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cacheable_batches, f)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'num_batches': len(batches),
            'config': {
                'batch_size': self.config.batch_size,
                'overlap_ratio': self.config.overlap_ratio,
                'grouping_resolution': self.config.grouping_resolution
            },
            'batch_info': [
                {
                    'batch_id': b.batch_id,
                    'num_hexagons': len(b.hex_ids),
                    'feature_dims': b.features.shape[1] if b.features is not None else 0,
                    'num_edges': b.edge_index.shape[1] if b.edge_index is not None else 0
                }
                for b in batches
            ]
        }
        
        with open(self.cache_dir / "batch_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Cached {len(batches)} batches to {cache_path}")
    
    def _load_cached_batches(self) -> Optional[List[SpatialBatch]]:
        """Load cached batches from disk."""
        if not self.cache_dir:
            return None
            
        cache_path = self.cache_dir / "spatial_batches.pkl"
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_batches = pickle.load(f)
            
            # Move tensors back to device
            for batch in cached_batches:
                if batch.features is not None:
                    batch.features = batch.features.to(self.device)
                if batch.edge_index is not None:
                    batch.edge_index = batch.edge_index.to(self.device)
                if batch.edge_weights is not None:
                    batch.edge_weights = batch.edge_weights.to(self.device)
            
            logger.info(f"Loaded {len(cached_batches)} cached batches")
            return cached_batches
            
        except Exception as e:
            logger.warning(f"Error loading cached batches: {str(e)}")
            return None
    
    def create_dataloader(
        self,
        batches: List[SpatialBatch],
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create PyTorch DataLoader for batches.
        
        Args:
            batches: List of SpatialBatch objects
            batch_size: Number of spatial batches per training batch
            shuffle: Whether to shuffle batches
            num_workers: Number of worker processes
            
        Returns:
            DataLoader for spatial batches
        """
        dataset = SpatialBatchDataset(batches)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_spatial_batches
        )
    
    def _collate_spatial_batches(self, batch_list: List[Dict]) -> Dict:
        """
        Custom collate function for spatial batches.
        
        Args:
            batch_list: List of batch dictionaries
            
        Returns:
            Collated batch dictionary
        """
        if len(batch_list) == 1:
            # Single spatial batch - return as is
            return batch_list[0]
        
        # Multiple spatial batches - combine them
        all_features = []
        all_edge_indices = []
        all_edge_weights = []
        all_hex_ids = []
        batch_boundaries = []
        
        node_offset = 0
        
        for batch_dict in batch_list:
            features = batch_dict['features']
            edge_index = batch_dict['edge_index']
            edge_weights = batch_dict['edge_weights']
            hex_ids = batch_dict['hex_ids']
            
            all_features.append(features)
            all_hex_ids.extend(hex_ids)
            all_edge_weights.append(edge_weights)
            
            # Adjust edge indices for concatenation
            adjusted_edge_index = edge_index + node_offset
            all_edge_indices.append(adjusted_edge_index)
            
            # Track batch boundaries
            batch_boundaries.extend([len(batch_boundaries)] * len(hex_ids))
            
            node_offset += features.shape[0]
        
        return {
            'features': torch.cat(all_features, dim=0),
            'edge_index': torch.cat(all_edge_indices, dim=1),
            'edge_weights': torch.cat(all_edge_weights, dim=0),
            'hex_ids': all_hex_ids,
            'batch_boundaries': torch.tensor(batch_boundaries, dtype=torch.long)
        }
    
    def get_batch_statistics(self, batches: List[SpatialBatch]) -> pd.DataFrame:
        """
        Get statistics about the created batches.
        
        Args:
            batches: List of SpatialBatch objects
            
        Returns:
            DataFrame with batch statistics
        """
        stats = []
        
        for batch in batches:
            stat_dict = {
                'batch_id': batch.batch_id,
                'num_hexagons': len(batch.hex_ids),
                'feature_dims': batch.features.shape[1] if batch.features is not None else 0,
                'num_edges': batch.edge_index.shape[1] if batch.edge_index is not None else 0,
                'num_neighbors': len(batch.neighbors) if batch.neighbors else 0,
                'avg_degree': (batch.edge_index.shape[1] / len(batch.hex_ids)) if batch.edge_index is not None and len(batch.hex_ids) > 0 else 0
            }
            stats.append(stat_dict)
        
        stats_df = pd.DataFrame(stats)
        
        # Add summary statistics
        logger.info("Batch Statistics Summary:")
        logger.info(f"  Total batches: {len(batches)}")
        logger.info(f"  Avg hexagons per batch: {stats_df['num_hexagons'].mean():.1f}")
        logger.info(f"  Min/Max hexagons: {stats_df['num_hexagons'].min()}/{stats_df['num_hexagons'].max()}")
        logger.info(f"  Avg edges per batch: {stats_df['num_edges'].mean():.1f}")
        logger.info(f"  Avg degree per batch: {stats_df['avg_degree'].mean():.2f}")
        
        return stats_df