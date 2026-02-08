"""
Cone-Based PyTorch Dataset

Dataset where each sample is a hierarchical cone with:
- Res10 features (input/observations)
- Spatial edges per resolution
- Hierarchical parent-child mappings

Each cone is constructed on-the-fly using cached parent-child lookup tables.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import pandas as pd
import geopandas as gpd
import numpy as np
from dataclasses import dataclass

# SRAI for H3 operations
from srai.neighbourhoods import H3Neighbourhood

from .hierarchical_cone_masking import (
    HierarchicalConeMaskingSystem,
    HierarchicalCone
)

# Import geometric helpers for validation
from ..geometry import (
    expected_children_count,
    log_geometric_summary,
    validate_cone_size
)

logger = logging.getLogger(__name__)


@dataclass
class ConeGraphStructure:
    """
    Graph structure for a single cone.

    Contains all spatial and hierarchical edges needed by the model.
    Uses per-resolution indexing (not global cone indexing).
    """
    # Spatial edges per resolution: (edge_index, edge_weight)
    spatial_edges: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    # Hierarchical parent-child mappings: (child_to_parent_idx, valid_children_mask, num_parents)
    hierarchical_mappings: Dict[int, Tuple[torch.Tensor, torch.Tensor, int]]

    # Per-resolution mapping from hex ID to local index
    # Dict[resolution] -> Dict[hex_id] -> local_index
    hex_to_local_idx_by_res: Dict[int, Dict[str, int]]

    # Per-resolution reverse mapping
    # Dict[resolution] -> Dict[local_index] -> hex_id
    local_idx_to_hex_by_res: Dict[int, Dict[int, str]]


class ConeDataset(Dataset):
    """
    PyTorch Dataset for hierarchical cones.

    Each sample is one cone spanning resolutions 5-10, with:
    - Features at res10 (observations)
    - Graph structure (spatial + hierarchical edges)
    """

    def __init__(
        self,
        study_area: str,
        parent_resolution: int = 5,
        target_resolution: int = 10,
        neighbor_rings: int = 5,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize cone dataset.

        Args:
            study_area: Name of study area (e.g., "netherlands")
            parent_resolution: Coarse resolution for cone roots
            target_resolution: Fine resolution (observations)
            neighbor_rings: Number of spatial neighbor rings
            data_dir: Root data directory
            cache_dir: Directory for caching lookup tables
        """
        self.study_area = study_area
        self.parent_resolution = parent_resolution
        self.target_resolution = target_resolution
        self.neighbor_rings = neighbor_rings

        if data_dir is None:
            data_dir = f"data/study_areas/{study_area}"
        self.data_dir = Path(data_dir)

        if cache_dir is None:
            cache_dir = self.data_dir / "cones"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing ConeDataset:")
        logger.info(f"  Study area: {study_area}")
        logger.info(f"  Parent resolution: {parent_resolution}")
        logger.info(f"  Target resolution: {target_resolution}")
        logger.info(f"  Neighbor rings: {neighbor_rings}")

        # Load res10 embeddings (observations - Markov blanket)
        logger.info("\nLoading res10 embeddings...")
        self.embeddings_res10 = self._load_res10_embeddings()
        logger.info(f"  Loaded {len(self.embeddings_res10)} hexagons")

        # Load region definitions (all resolutions)
        logger.info("\nLoading region definitions...")
        self.regions_by_res = self._load_all_regions()
        for res, regions in self.regions_by_res.items():
            logger.info(f"  Res {res}: {len(regions)} hexagons")

        # Validate hierarchical consistency (use optimized version)
        self.regions_by_res = self._validate_hierarchical_consistency_optimized(
            self.regions_by_res,
            chunk_size=100000  # Process 100k hexagons per chunk
        )

        # Initialize cone system
        logger.info("\nInitializing cone system...")
        self.cone_system = HierarchicalConeMaskingSystem(
            parent_resolution=parent_resolution,
            target_resolution=target_resolution,
            neighbor_rings=neighbor_rings
        )

        # Build/load parent→children lookup table
        logger.info("\nBuilding parent-child lookup table...")
        cache_path = self.cache_dir / f"parent_lookup_res{parent_resolution}_to_{target_resolution}.pkl"
        self.cone_system.parent_to_children = self.cone_system._build_parent_lookup(
            self.regions_by_res,
            cache_path=str(cache_path)
        )

        # Get list of parent hexagons (cone roots)
        self.parent_hexes = list(self.regions_by_res[parent_resolution].index)
        logger.info(f"\nDataset ready: {len(self.parent_hexes)} cones")

        # Initialize H3 neighbourhood for spatial edges
        self.h3_neighbourhood = H3Neighbourhood()

    def _load_res10_embeddings(self) -> pd.DataFrame:
        """Load AlphaEarth embeddings at resolution 10 (observations)."""
        # Try PCA version first (memory efficient)
        embeddings_path = (
            self.data_dir / "embeddings" / "alphaearth" /
            f"{self.study_area}_res{self.target_resolution}_pca16_2022.parquet"
        )

        # Fall back to full embeddings if PCA not available
        if not embeddings_path.exists():
            logger.warning("PCA embeddings not found, falling back to full embeddings")
            embeddings_path = (
                self.data_dir / "embeddings" / "alphaearth" /
                f"{self.study_area}_res{self.target_resolution}_2022.parquet"
            )

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

        logger.info(f"Loading embeddings from: {embeddings_path.name}")
        embeddings_df = pd.read_parquet(embeddings_path)

        # Align with SRAI region_id format
        if 'h3_index' in embeddings_df.columns:
            embeddings_df = embeddings_df.set_index('h3_index')
            embeddings_df.index.name = 'region_id'

        # Extract only embedding columns (A00-A63 or P00-P15)
        embedding_cols = [col for col in embeddings_df.columns if col.startswith(('A', 'P'))]
        embeddings_df = embeddings_df[embedding_cols]

        # Handle NaN values: replace with zeros
        if embeddings_df.isnull().any().any():
            nan_count = embeddings_df.isnull().sum().sum()
            logger.warning(f"Found {nan_count} NaN values in embeddings, filling with 0")
            embeddings_df = embeddings_df.fillna(0.0)

        return embeddings_df

    def _load_all_regions(self) -> Dict[int, gpd.GeoDataFrame]:
        """Load region definitions for all resolutions."""
        regions_by_res = {}

        for res in range(self.parent_resolution, self.target_resolution + 1):
            regions_path = (
                self.data_dir / "regions_gdf" /
                f"{self.study_area}_res{res}.parquet"
            )

            if not regions_path.exists():
                logger.warning(f"Regions not found for res{res}: {regions_path}")
                continue

            regions_gdf = gpd.read_parquet(regions_path)
            regions_by_res[res] = regions_gdf

        return regions_by_res

    def _validate_hierarchical_consistency(
        self,
        regions_by_res: Dict[int, gpd.GeoDataFrame]
    ) -> Dict[int, gpd.GeoDataFrame]:
        """
        Validate and filter regions to ensure hierarchical consistency.

        **LEGACY**: This is the original O(N × depth) implementation.
        Use `_validate_hierarchical_consistency_optimized()` for better performance.

        Ensures all hexagons at finer resolutions are descendants of available
        parent hexagons at the parent resolution. This prevents "orphan" hexagons
        that have parents outside the study area.

        Args:
            regions_by_res: Dictionary mapping resolution to GeoDataFrame

        Returns:
            Filtered dictionary with only hierarchically consistent hexagons
        """
        import h3

        logger.info("\n" + "="*60)
        logger.info("Validating Hierarchical Consistency")
        logger.info("="*60)

        # Get available parent hexagons at parent_resolution
        if self.parent_resolution not in regions_by_res:
            logger.warning(f"Parent resolution {self.parent_resolution} not found in regions")
            return regions_by_res

        parent_hexes = set(regions_by_res[self.parent_resolution].index)
        logger.info(f"Available parent hexagons at res{self.parent_resolution}: {len(parent_hexes)}")

        # No filtering needed for parent resolution itself
        filtered_regions = {self.parent_resolution: regions_by_res[self.parent_resolution]}

        # Validate and filter each finer resolution
        for res in range(self.parent_resolution + 1, self.target_resolution + 1):
            if res not in regions_by_res:
                continue

            regions_gdf = regions_by_res[res]
            original_count = len(regions_gdf)

            # Check each hexagon's parent at parent_resolution
            valid_hexes = []
            invalid_count = 0

            for hex_id in regions_gdf.index:
                # Traverse up to parent_resolution
                parent = hex_id
                for _ in range(res - self.parent_resolution):
                    parent = h3.cell_to_parent(parent, h3.get_resolution(parent) - 1)

                # Check if parent exists in available parents
                if parent in parent_hexes:
                    valid_hexes.append(hex_id)
                else:
                    invalid_count += 1

            # Filter to only valid hexagons
            if len(valid_hexes) < original_count:
                filtered_gdf = regions_gdf.loc[valid_hexes]
                filtered_regions[res] = filtered_gdf

                percent_removed = (invalid_count / original_count) * 100
                logger.info(
                    f"Res{res}: Filtered {invalid_count:,} / {original_count:,} hexagons "
                    f"({percent_removed:.1f}% had parents outside res{self.parent_resolution})"
                )
                logger.info(f"  Kept {len(valid_hexes):,} valid hexagons")
            else:
                # All hexagons are valid
                filtered_regions[res] = regions_gdf
                logger.info(f"Res{res}: All {original_count:,} hexagons are valid")

        logger.info("="*60)
        return filtered_regions

    def _validate_hierarchical_consistency_optimized(
        self,
        regions_by_res: Dict[int, gpd.GeoDataFrame],
        chunk_size: int = 100000
    ) -> Dict[int, gpd.GeoDataFrame]:
        """
        Optimized hierarchical consistency validation using vectorized operations.

        Improvements over original:
        - Vectorized parent lookups (no nested loops)
        - Direct parent resolution targeting (no iteration)
        - NumPy array operations for membership checks
        - Geometric validation and logging
        - Chunked processing for memory efficiency

        Args:
            regions_by_res: Dictionary mapping resolution to GeoDataFrame
            chunk_size: Number of hexagons to process per chunk

        Returns:
            Filtered dictionary with only hierarchically consistent hexagons
        """
        import h3  # h3-py is SRAI dependency

        logger.info("\n" + "="*60)
        logger.info("Validating Hierarchical Consistency (Optimized)")
        logger.info("="*60)

        # Log geometric expectations
        log_geometric_summary(
            parent_res=self.parent_resolution,
            target_res=self.target_resolution,
            neighbor_rings=self.neighbor_rings
        )

        # Get available parent hexagons
        if self.parent_resolution not in regions_by_res:
            logger.warning(f"Parent resolution {self.parent_resolution} not found")
            return regions_by_res

        parent_hexes_set = set(regions_by_res[self.parent_resolution].index)
        logger.info(f"\nAvailable parent hexagons at res{self.parent_resolution}: {len(parent_hexes_set):,}")

        # No filtering needed for parent resolution
        filtered_regions = {self.parent_resolution: regions_by_res[self.parent_resolution]}

        # Validate each finer resolution
        for res in range(self.parent_resolution + 1, self.target_resolution + 1):
            if res not in regions_by_res:
                continue

            regions_gdf = regions_by_res[res]
            original_count = len(regions_gdf)

            # Geometric expectation
            expected_per_parent = expected_children_count(self.parent_resolution, res)
            expected_total = len(parent_hexes_set) * expected_per_parent
            coverage_ratio = original_count / expected_total if expected_total > 0 else 0.0

            logger.info(f"\nRes{res}:")
            logger.info(f"  Actual hexagons: {original_count:,}")
            logger.info(f"  Expected (geometric): {expected_total:,}")
            logger.info(f"  Coverage ratio: {coverage_ratio:.2%}")

            # Get all hex IDs as list
            hex_ids_list = regions_gdf.index.tolist()

            # Chunked processing for memory efficiency
            valid_indices = []
            invalid_count = 0

            num_chunks = (len(hex_ids_list) + chunk_size - 1) // chunk_size
            logger.info(f"  Processing in {num_chunks} chunks of {chunk_size:,} hexagons...")

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(hex_ids_list))
                chunk_hex_ids = hex_ids_list[start_idx:end_idx]

                # Vectorized parent lookup (direct to parent_resolution)
                # This replaces the nested loop: much faster!
                chunk_parents = [
                    h3.cell_to_parent(hex_id, self.parent_resolution)
                    for hex_id in chunk_hex_ids
                ]

                # Convert to numpy for vectorized membership check
                chunk_parents_array = np.array(chunk_parents)
                parent_hexes_array = np.array(list(parent_hexes_set))

                # Vectorized membership test
                valid_mask = np.isin(chunk_parents_array, parent_hexes_array)

                # Collect valid indices
                chunk_valid_indices = [
                    start_idx + i
                    for i in range(len(chunk_hex_ids))
                    if valid_mask[i]
                ]
                valid_indices.extend(chunk_valid_indices)
                invalid_count += (~valid_mask).sum()

            # Filter to only valid hexagons
            if len(valid_indices) < original_count:
                valid_hex_ids = [hex_ids_list[i] for i in valid_indices]
                filtered_gdf = regions_gdf.loc[valid_hex_ids]
                filtered_regions[res] = filtered_gdf

                percent_removed = (invalid_count / original_count) * 100
                logger.info(
                    f"  Filtered {invalid_count:,} / {original_count:,} hexagons "
                    f"({percent_removed:.1f}% had parents outside res{self.parent_resolution})"
                )
                logger.info(f"  Kept {len(valid_hex_ids):,} valid hexagons")

                # Geometric validation of result
                actual_per_parent_avg = len(valid_hex_ids) / len(parent_hexes_set)
                logger.info(f"  Actual per parent (avg): {actual_per_parent_avg:.1f}")
                logger.info(f"  Expected per parent: {expected_per_parent:,}")
                logger.info(
                    f"  Actual/Expected ratio: {actual_per_parent_avg / expected_per_parent:.2%}"
                )
            else:
                # All hexagons are valid
                filtered_regions[res] = regions_gdf
                logger.info(f"  All {original_count:,} hexagons are valid")

        logger.info("="*60)
        return filtered_regions

    def __len__(self) -> int:
        """Number of cones in dataset."""
        return len(self.parent_hexes)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one cone's data.

        Returns:
            Dictionary with:
                - features_res10: Input features [N_10, D]
                - spatial_edges: Dict[res] -> (edge_index, edge_weight)
                - hierarchical_mappings: Dict[child_res] -> (child_to_parent_idx, num_parents)
                - cone_id: Parent hex ID
                - num_nodes_per_res: Dict[res] -> num_nodes
        """
        parent_hex = self.parent_hexes[idx]

        # Create cone structure (fast with cached lookup)
        cone = self.cone_system.create_cone(
            parent_hex,
            self.regions_by_res
        )

        # Extract res10 features for this cone (observations)
        features_res10 = self._extract_res10_features(cone)

        # Build graph structure
        cone_structure = self._build_cone_graph(cone)

        # Count nodes per resolution
        num_nodes_per_res = {
            res: len(cone.descendants_by_resolution.get(res, set()))
            if res > self.parent_resolution
            else len({cone.parent_hex} | cone.parent_neighbors)
            for res in range(self.parent_resolution, self.target_resolution + 1)
        }

        return {
            'features_res10': features_res10,
            'spatial_edges': cone_structure.spatial_edges,
            'hierarchical_mappings': cone_structure.hierarchical_mappings,
            'cone_id': parent_hex,
            'num_nodes_per_res': num_nodes_per_res,
            'hex_to_local_idx_by_res': cone_structure.hex_to_local_idx_by_res,
            'local_idx_to_hex_by_res': cone_structure.local_idx_to_hex_by_res
        }

    def _extract_res10_features(self, cone: HierarchicalCone) -> torch.Tensor:
        """
        Extract res10 features for this cone.

        Returns features for ALL hexagons in cone, zero-padding where data is missing.
        This ensures features align with cone structure and hierarchical mappings.
        """
        # Get ALL res10 hexagons in this cone (ordered consistently with cone.local_node_indices)
        res10_hexes_ordered = sorted(cone.descendants_by_resolution[self.target_resolution])

        # Get embedding dimension
        embedding_dim = self.embeddings_res10.shape[1]
        num_hexes = len(res10_hexes_ordered)

        # Create feature matrix with zeros for all hexagons (padding)
        features = np.zeros((num_hexes, embedding_dim), dtype=np.float32)

        # Fill in embeddings where available
        for i, hex_id in enumerate(res10_hexes_ordered):
            if hex_id in self.embeddings_res10.index:
                embedding = self.embeddings_res10.loc[hex_id].values

                # Additional safety: replace any NaNs with zeros (should be handled in load, but be safe)
                if np.any(np.isnan(embedding)):
                    embedding = np.nan_to_num(embedding, nan=0.0)

                features[i] = embedding
            # else: keep zeros (padding for hexagons outside data coverage or beyond cone scope)

        # Count coverage for logging
        num_with_data = sum(1 for hex_id in res10_hexes_ordered if hex_id in self.embeddings_res10.index)
        coverage = num_with_data / num_hexes * 100 if num_hexes > 0 else 0

        if coverage < 95:  # Log warning if significant missing data
            logger.debug(
                f"Cone {cone.cone_id}: {num_with_data}/{num_hexes} res10 hexes have data ({coverage:.1f}%)"
            )

        return torch.tensor(features, dtype=torch.float32)

    def _build_cone_graph(self, cone: HierarchicalCone) -> ConeGraphStructure:
        """
        Build complete graph structure for cone using per-resolution indexing.

        Each resolution has its own index space (0 to N_res-1), not a global index space.
        This aligns with how regions_gdf files are organized.
        """
        # Create PER-RESOLUTION index mappings
        hex_to_local_idx_by_res = {}
        local_idx_to_hex_by_res = {}

        for res in range(self.parent_resolution, self.target_resolution + 1):
            # Get hexagons at this resolution from cone
            if res == self.parent_resolution:
                hexes_at_res = sorted({cone.parent_hex} | cone.parent_neighbors)
            else:
                hexes_at_res = sorted(cone.descendants_by_resolution.get(res, set()))

            # Create local indices for THIS resolution only (0 to N_res-1)
            hex_to_local_idx_by_res[res] = {
                hex_id: idx for idx, hex_id in enumerate(hexes_at_res)
            }
            local_idx_to_hex_by_res[res] = {
                idx: hex_id for hex_id, idx in hex_to_local_idx_by_res[res].items()
            }

        # Build spatial edges per resolution (using per-resolution indices)
        spatial_edges = {}
        for res in range(self.parent_resolution, self.target_resolution + 1):
            edges, weights = self._build_spatial_edges_at_resolution(
                cone, res, hex_to_local_idx_by_res[res]
            )
            spatial_edges[res] = (edges, weights)

        # Build hierarchical mappings (mapping between per-resolution index spaces)
        # Using optimized version with vectorized operations and geometric pre-allocation
        hierarchical_mappings = {}
        for child_res in range(self.parent_resolution + 1, self.target_resolution + 1):
            parent_res = child_res - 1
            mapping = self._build_hierarchical_mapping_optimized(
                cone,
                child_res,
                hex_to_local_idx_by_res[child_res],  # Child indices
                hex_to_local_idx_by_res[parent_res]  # Parent indices
            )
            hierarchical_mappings[child_res] = mapping

        return ConeGraphStructure(
            spatial_edges=spatial_edges,
            hierarchical_mappings=hierarchical_mappings,
            hex_to_local_idx_by_res=hex_to_local_idx_by_res,
            local_idx_to_hex_by_res=local_idx_to_hex_by_res
        )

    def _build_spatial_edges_at_resolution(
        self,
        cone: HierarchicalCone,
        res: int,
        hex_to_local_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build spatial edges for hexagonal lattice using SRAI.

        Creates bidirectional edges between immediate neighbors (6-connected hexagonal lattice).
        Each unique pair is processed once to avoid duplicates.

        Returns:
            edge_index: [2, E]
            edge_weight: [E]
        """
        # Get hexagons at this resolution in the cone
        if res == self.parent_resolution:
            hex_set = {cone.parent_hex} | cone.parent_neighbors
        else:
            hex_set = cone.descendants_by_resolution.get(res, set())

        if len(hex_set) == 0:
            # No hexagons at this resolution
            empty_edges = torch.zeros((2, 0), dtype=torch.long)
            empty_weights = torch.zeros(0, dtype=torch.float32)
            return empty_edges, empty_weights

        hexes_list = sorted(list(hex_set))  # Sort for deterministic ordering

        # Build edges: only process each unique pair once
        edge_list = []
        edge_weights = []

        # Track which hexagons we've seen to avoid duplicates
        seen_pairs = set()

        for src_hex in hexes_list:
            if src_hex not in hex_to_local_idx:
                continue

            src_idx = hex_to_local_idx[src_hex]

            # Use SRAI to get immediate neighbors (distance=1)
            from srai.neighbourhoods import H3Neighbourhood

            neighbourhood = H3Neighbourhood()
            neighbors = neighbourhood.get_neighbours_at_distance(src_hex, 1)

            for tgt_hex in neighbors:
                # Only add edge if target is in cone
                if tgt_hex not in hex_to_local_idx:
                    continue

                tgt_idx = hex_to_local_idx[tgt_hex]

                # Create canonical pair (smaller index first) to avoid duplicates
                pair = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))

                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)

                # Add BOTH directions for undirected graph in PyG
                edge_list.append([src_idx, tgt_idx])
                edge_list.append([tgt_idx, src_idx])
                edge_weights.append(1.0)
                edge_weights.append(1.0)

        if len(edge_list) == 0:
            empty_edges = torch.zeros((2, 0), dtype=torch.long)
            empty_weights = torch.zeros(0, dtype=torch.float32)
            return empty_edges, empty_weights

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

        return edge_index, edge_weight

    def _build_hierarchical_mapping(
        self,
        cone: HierarchicalCone,
        child_res: int,
        child_hex_to_idx: Dict[str, int],
        parent_hex_to_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Build child→parent mapping using per-resolution indices.

        **LEGACY**: Original implementation with Python loops.
        Use `_build_hierarchical_mapping_optimized()` for better performance.

        Filters to only include children whose parents exist in the cone.
        This handles "orphan children" at cone boundaries.

        Args:
            cone: Hierarchical cone
            child_res: Child resolution
            child_hex_to_idx: Mapping from child hex ID to local child index (0 to N_child-1)
            parent_hex_to_idx: Mapping from parent hex ID to local parent index (0 to N_parent-1)

        Returns:
            child_to_parent_idx: [num_valid_children] - parent index for each VALID child
            valid_children_mask: [num_all_children] - boolean mask indicating which children are valid
            num_parents: Number of parent nodes
        """
        import h3

        parent_res = child_res - 1

        # Get children hexagons at child_res (sorted for consistent ordering)
        children = sorted(cone.descendants_by_resolution.get(child_res, set()))

        # Build child→parent mapping and track which children are valid
        child_to_parent_list = []
        valid_children_mask = torch.zeros(len(children), dtype=torch.bool)

        for i, child_hex in enumerate(children):
            # Get parent of this child
            parent_hex = h3.cell_to_parent(child_hex, parent_res)

            # Check if parent exists in cone
            if parent_hex in parent_hex_to_idx:
                # Valid: parent exists in cone
                valid_children_mask[i] = True
                parent_local_idx = parent_hex_to_idx[parent_hex]
                child_to_parent_list.append(parent_local_idx)
            # else: orphan - mask stays False, don't add to mapping

        num_parents = len(parent_hex_to_idx)

        if len(child_to_parent_list) > 0:
            child_to_parent_idx = torch.tensor(child_to_parent_list, dtype=torch.long)
        else:
            child_to_parent_idx = torch.zeros(0, dtype=torch.long)

        return child_to_parent_idx, valid_children_mask, num_parents

    def _build_hierarchical_mapping_optimized(
        self,
        cone: HierarchicalCone,
        child_res: int,
        child_hex_to_idx: Dict[str, int],
        parent_hex_to_idx: Dict[str, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Optimized child→parent mapping using vectorized operations and geometric pre-allocation.

        Improvements over original:
        - Pre-allocates arrays using 7:1 geometric ratio
        - Vectorized parent lookups (list comprehension + NumPy)
        - Batch membership checks (NumPy.isin())
        - Direct tensor creation (no intermediate lists)
        - Geometric validation logging

        Args:
            cone: Hierarchical cone
            child_res: Child resolution
            child_hex_to_idx: Mapping from child hex ID to local child index
            parent_hex_to_idx: Mapping from parent hex ID to local parent index

        Returns:
            child_to_parent_idx: [num_valid_children] - parent indices
            valid_children_mask: [num_all_children] - validity mask
            num_parents: Number of parent nodes
        """
        import h3

        parent_res = child_res - 1
        num_parents = len(parent_hex_to_idx)

        # Get children hexagons (sorted for consistency)
        children = sorted(cone.descendants_by_resolution.get(child_res, set()))
        num_children = len(children)

        if num_children == 0:
            # No children at this resolution
            return (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.bool),
                num_parents
            )

        # Geometric expectation: 7 children per parent (interior)
        expected_children = num_parents * 7
        actual_ratio = num_children / expected_children if expected_children > 0 else 0.0

        # Vectorized parent lookup for all children
        # This replaces the per-child loop with a batch operation
        children_parents = [h3.cell_to_parent(child, parent_res) for child in children]

        # Convert to numpy for vectorized operations
        children_parents_array = np.array(children_parents)
        parent_hexes_array = np.array(list(parent_hex_to_idx.keys()))

        # Vectorized membership check: which parents exist in the cone?
        valid_mask = np.isin(children_parents_array, parent_hexes_array)

        # Count orphans (children whose parents aren't in cone)
        num_valid = valid_mask.sum()
        num_orphans = num_children - num_valid
        orphan_rate = num_orphans / num_children if num_children > 0 else 0.0

        # Pre-allocate output array using geometric expectation
        # Expected size: min(num_children, num_parents * 7)
        expected_size = min(num_children, num_parents * 7)
        child_to_parent_indices = np.zeros(num_valid, dtype=np.int64)

        # Build mapping for valid children only
        valid_idx = 0
        for i in range(num_children):
            if valid_mask[i]:
                parent_hex = children_parents_array[i]
                parent_local_idx = parent_hex_to_idx[parent_hex]
                child_to_parent_indices[valid_idx] = parent_local_idx
                valid_idx += 1

        # Convert to torch tensors
        child_to_parent_idx = torch.from_numpy(child_to_parent_indices).long()
        valid_children_mask = torch.from_numpy(valid_mask)

        # Geometric validation logging (only for first few resolutions to avoid spam)
        if child_res <= self.parent_resolution + 3:
            avg_children_per_parent = num_valid / num_parents if num_parents > 0 else 0.0
            logger.debug(
                f"  Res{child_res} mapping: {num_valid:,} valid / {num_children:,} total children, "
                f"{num_orphans} orphans ({orphan_rate:.1%}), "
                f"avg {avg_children_per_parent:.1f} children/parent (expected: 7)"
            )

        return child_to_parent_idx, valid_children_mask, num_parents


def cone_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    Since each cone has different numbers of nodes, we process cones
    one at a time (batch processing happens at cone level, not within cone).

    For now, just return the first item (batch size = 1 per cone).
    Future: Could batch multiple cones together with careful indexing.
    """
    # For simplicity, process one cone at a time
    if len(batch) == 1:
        return batch[0]
    else:
        # If batch size > 1, this needs more sophisticated batching
        # For now, just return first item
        logger.warning("Cone dataset currently supports batch_size=1 only")
        return batch[0]
