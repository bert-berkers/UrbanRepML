"""
Hierarchical Cone Masking for Multi-Resolution Graph Processing
================================================================

Creates computational "cones" where each cone contains:
- A parent hexagon at coarse resolution (e.g., res5)
- 5-ring neighborhood at that resolution
- All descendant hexagons through finer resolutions (res6->res7->...->res10)

Each cone is an independent computation that can be batched and processed in parallel.
This replaces the need to hold massive multi-resolution graphs entirely in memory.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from dataclasses import dataclass, field
from tqdm import tqdm
import pickle
from pathlib import Path

# Use SRAI for all H3 operations
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood

# Import geometric helpers
from ..geometry import (
    expected_cone_size,
    expected_k_ring_size,
    validate_cone_size,
    log_geometric_summary,
)

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalCone:
    """
    A hierarchical computation cone.

    Contains a parent hex, its neighbors, and all descendants across resolutions.
    """
    cone_id: str  # Parent hex ID at coarse resolution
    parent_resolution: int  # Starting resolution (e.g., 5)
    target_resolution: int  # Ending resolution (e.g., 10)

    # Hexagons by resolution
    parent_hex: str  # The root hexagon
    parent_neighbors: Set[str]  # 5-ring neighborhood at parent resolution
    descendants_by_resolution: Dict[int, Set[str]]  # res -> hexagons

    # For masking (computed in __post_init__)
    all_hexagons: Set[str] = field(init=False)  # All hexagons in cone (union across resolutions)
    hex_to_resolution: Dict[str, int] = field(init=False)  # Map each hex to its resolution

    # Indices for graph operations (computed in __post_init__)
    local_node_indices: Dict[str, int] = field(init=False)  # hex_id -> local index within cone

    def __post_init__(self):
        """Compute derived attributes."""
        # Combine all hexagons
        self.all_hexagons = {self.parent_hex} | self.parent_neighbors
        for hex_set in self.descendants_by_resolution.values():
            self.all_hexagons.update(hex_set)

        # Map hexagons to their resolution
        self.hex_to_resolution = {self.parent_hex: self.parent_resolution}
        for neighbor in self.parent_neighbors:
            self.hex_to_resolution[neighbor] = self.parent_resolution
        for res, hex_set in self.descendants_by_resolution.items():
            for hex_id in hex_set:
                self.hex_to_resolution[hex_id] = res

        # Create local indices
        self.local_node_indices = {
            hex_id: idx for idx, hex_id in enumerate(sorted(self.all_hexagons))
        }

    def get_mask(self, global_hex_list: List[str]) -> torch.Tensor:
        """
        Create a boolean mask for this cone's hexagons in the global graph.

        Args:
            global_hex_list: Ordered list of all hexagons in global graph

        Returns:
            Boolean tensor [num_global_nodes] where True = hexagon in this cone
        """
        mask = torch.zeros(len(global_hex_list), dtype=torch.bool)
        hex_to_global_idx = {h: i for i, h in enumerate(global_hex_list)}

        for hex_id in self.all_hexagons:
            if hex_id in hex_to_global_idx:
                mask[hex_to_global_idx[hex_id]] = True

        return mask

    def num_nodes(self) -> int:
        """Total number of nodes in this cone."""
        return len(self.all_hexagons)

    def nodes_per_resolution(self) -> Dict[int, int]:
        """Count nodes at each resolution."""
        counts = {self.parent_resolution: 1 + len(self.parent_neighbors)}
        for res, hex_set in self.descendants_by_resolution.items():
            counts[res] = len(hex_set)
        return counts


class HierarchicalConeMaskingSystem:
    """
    Creates and manages hierarchical computation cones for multi-resolution processing.
    """

    def __init__(
        self,
        parent_resolution: int = 5,
        target_resolution: int = 10,
        neighbor_rings: int = 5
    ):
        """
        Initialize cone masking system.

        Args:
            parent_resolution: Coarse resolution for cone roots (e.g., 5)
            target_resolution: Fine resolution for leaf nodes (e.g., 10)
            neighbor_rings: Number of neighbor rings at parent resolution (e.g., 5)
        """
        self.parent_resolution = parent_resolution
        self.target_resolution = target_resolution
        self.neighbor_rings = neighbor_rings

        # Resolution hierarchy
        self.resolutions = list(range(parent_resolution, target_resolution + 1))

        logger.info(f"Initializing HierarchicalConeMaskingSystem:")
        logger.info(f"  Parent resolution: {parent_resolution}")
        logger.info(f"  Target resolution: {target_resolution}")
        logger.info(f"  Neighbor rings: {neighbor_rings}")
        logger.info(f"  Resolution hierarchy: {self.resolutions}")

        # SRAI neighborhood
        self.neighbourhood = H3Neighbourhood()

        # Parent->children lookup table (will be built/loaded on demand)
        self.parent_to_children = None  # (parent_hex, child_res) -> set[child_hex]

    def _build_parent_lookup(
        self,
        regions_by_resolution: Dict[int, gpd.GeoDataFrame],
        cache_path: Optional[str] = None
    ) -> Dict[Tuple[str, int], Set[str]]:
        """
        Build parent->children lookup table for fast cone creation.

        This is a ONE-TIME computation that enables millisecond-speed cone creation.
        Maps: (parent_hex, child_resolution) -> set[child_hex]

        Args:
            regions_by_resolution: Dict mapping resolution -> GeoDataFrame
            cache_path: Optional path to cache the lookup table

        Returns:
            Dictionary mapping (parent_hex, child_res) to set of child hexagons
        """
        import h3
        import pickle
        from pathlib import Path
        from collections import defaultdict

        logger.info("\n" + "="*60)
        logger.info("Building Parent->Children Lookup Table")
        logger.info("="*60)

        # Try loading from cache first
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached lookup from {cache_path}")
            with open(cache_path, 'rb') as f:
                parent_to_children = pickle.load(f)
            logger.info(f"Loaded {len(parent_to_children):,} parent->children mappings")
            return parent_to_children

        logger.info(f"Building lookup for resolutions {self.resolutions}")
        logger.info("This is a one-time cost (~30 seconds)")

        parent_to_children = defaultdict(set)

        # For each resolution, map each hexagon to its parent at parent_resolution
        for child_res in tqdm(range(self.parent_resolution, self.target_resolution + 1), desc="Building lookup"):
            if child_res not in regions_by_resolution:
                continue

            hexagons = list(regions_by_resolution[child_res].index)
            logger.info(f"  Res{child_res}: Processing {len(hexagons):,} hexagons")

            for hex_id in hexagons:
                # Compute parent at parent_resolution
                parent = hex_id
                res = child_res
                while res > self.parent_resolution:
                    parent = h3.cell_to_parent(parent, res - 1)
                    res -= 1

                # Map parent->child at this resolution
                parent_to_children[(parent, child_res)].add(hex_id)

        logger.info(f"Built {len(parent_to_children):,} parent->children mappings")

        # Cache to disk if path provided
        if cache_path:
            logger.info(f"Caching lookup to {cache_path}")
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(dict(parent_to_children), f)
            logger.info(f"Cached lookup table")

        return dict(parent_to_children)

    def get_h3_children(self, parent_hex: str, child_resolution: int) -> Set[str]:
        """
        Get all children of a parent hexagon at specified resolution.
        Uses h3 directly for hierarchical parent-child mappings (per user guidance).

        Args:
            parent_hex: Parent hexagon ID
            child_resolution: Resolution of children

        Returns:
            Set of child hexagon IDs
        """
        import h3

        parent_res = h3.get_resolution(parent_hex)

        if child_resolution <= parent_res:
            return {parent_hex}

        # Use h3 directly for parent-to-child mappings (hierarchical, not spatial)
        children = set()
        if child_resolution == parent_res + 1:
            children = set(h3.cell_to_children(parent_hex, child_resolution))
        else:
            # Multi-level descent
            for intermediate_child in h3.cell_to_children(parent_hex, parent_res + 1):
                children.update(self.get_h3_children(intermediate_child, child_resolution))

        return children

    def get_neighbors_at_distance(
        self,
        hex_id: str,
        k_rings: int
    ) -> Set[str]:
        """
        Get k-ring neighbors using SRAI's H3Neighbourhood.

        Args:
            hex_id: Center hexagon
            k_rings: Number of rings

        Returns:
            Set of neighbor hexagon IDs (excluding center)
        """
        from srai.neighbourhoods import H3Neighbourhood

        neighbourhood = H3Neighbourhood()
        return neighbourhood.get_neighbours_up_to_distance(hex_id, k_rings)

    def create_cone(
        self,
        parent_hex: str,
        regions_by_resolution: Dict[int, gpd.GeoDataFrame]
    ) -> HierarchicalCone:
        """
        Create a hierarchical cone for a parent hexagon using fast lookup table.

        This is now a FAST operation using pre-computed parent-child mappings.
        No recursive h3 calls - just dict lookups!

        Args:
            parent_hex: Parent hexagon ID at parent_resolution
            regions_by_resolution: Dict mapping resolution -> GeoDataFrame of regions

        Returns:
            HierarchicalCone object
        """
        if self.parent_to_children is None:
            raise RuntimeError("Parent lookup table not initialized. Call create_all_cones() first.")

        # Get parent's neighbors via SRAI H3Neighbourhood
        parent_neighbors = self.get_neighbors_at_distance(
            parent_hex,
            self.neighbor_rings
        )

        # Filter to only neighbors that exist in our regions
        parent_regions = regions_by_resolution[self.parent_resolution]
        parent_neighbors = parent_neighbors & set(parent_regions.index)

        # Parent and neighbors define the "root" of this cone
        cone_roots = {parent_hex} | parent_neighbors

        # Get all descendants through each resolution using INVERTED LOOKUP TABLE
        descendants_by_resolution = {}

        for res in range(self.parent_resolution + 1, self.target_resolution + 1):
            if res not in regions_by_resolution:
                descendants_by_resolution[res] = set()
                continue

            # FAST: Direct lookup of children from parent
            # O(k) where k = number of cone roots (~10-20)
            # Instead of O(n) where n = all hexagons at this resolution (~5M)
            descendants = set()
            for root_hex in cone_roots:
                children = self.parent_to_children.get((root_hex, res), set())
                descendants.update(children)

            # Filter to only hexagons that exist in our regions
            available_hexagons = set(regions_by_resolution[res].index)
            descendants = descendants & available_hexagons

            descendants_by_resolution[res] = descendants

        # Create cone
        cone = HierarchicalCone(
            cone_id=parent_hex,
            parent_resolution=self.parent_resolution,
            target_resolution=self.target_resolution,
            parent_hex=parent_hex,
            parent_neighbors=parent_neighbors,
            descendants_by_resolution=descendants_by_resolution
        )

        return cone

    def cache_all_cones(
        self,
        regions_by_resolution: Dict[int, gpd.GeoDataFrame],
        cache_dir: str
    ) -> List[str]:
        """
        Build and cache all cones to disk (INDIVIDUAL FILES for true lazy loading).

        Creates cone for each parent hexagon and saves to separate pickle file.
        This enables loading only the cones needed (not all 60 GB at once!).

        Args:
            regions_by_resolution: Dict mapping resolution -> GeoDataFrame
            cache_dir: Directory to save individual cone files

        Returns:
            List of parent hexagon IDs (cone files saved as cone_{hex}.pkl)
        """
        cache_directory = Path(cache_dir)
        cache_directory.mkdir(parents=True, exist_ok=True)

        parent_hexes = sorted(regions_by_resolution[self.parent_resolution].index)

        # Check if cones already cached (check for first and last cone files)
        first_cone_file = cache_directory / f"cone_{parent_hexes[0]}.pkl"
        last_cone_file = cache_directory / f"cone_{parent_hexes[-1]}.pkl"

        if first_cone_file.exists() and last_cone_file.exists():
            # Count existing cone files
            existing_cones = list(cache_directory.glob("cone_*.pkl"))
            logger.info(f"Found {len(existing_cones)} cached cone files in: {cache_dir}")
            logger.info(f"  Skipping cone creation (already cached)")
            return parent_hexes

        # Build and save cones individually
        logger.info("Building all cones and saving individually...")
        logger.info(f"  This is a one-time operation (~10-15 minutes)")
        logger.info(f"  Creating {len(parent_hexes)} cone files in: {cache_dir}")

        total_size_mb = 0
        for parent_hex in tqdm(parent_hexes, desc="Building cones"):
            # Build cone
            cone = self.create_cone(parent_hex, regions_by_resolution)

            # Save to individual file
            cone_file = cache_directory / f"cone_{parent_hex}.pkl"
            with open(cone_file, 'wb') as f:
                pickle.dump(cone, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Track total size
            total_size_mb += cone_file.stat().st_size / 1e6

        logger.info(f"Cached {len(parent_hexes)} cones ({total_size_mb/1000:.1f} GB on disk)")
        logger.info(f"  Each cone is a separate file for true lazy loading!")
        logger.info(f"  Subsequent runs will load only needed cones (minimal memory)")

        return parent_hexes

    def create_all_cones(
        self,
        regions_by_resolution: Dict[int, gpd.GeoDataFrame],
        n_jobs: int = -1,
        cache_dir: Optional[str] = None
    ) -> List[HierarchicalCone]:
        """
        Create all cones for the study area with parallel processing.

        Uses fast parent-child lookup table for millisecond-speed cone creation.

        Args:
            regions_by_resolution: Dict mapping resolution -> GeoDataFrame
            n_jobs: Number of parallel jobs (-1 = use all CPU cores)
            cache_dir: Optional directory to cache parent lookup table

        Returns:
            List of HierarchicalCone objects
        """
        logger.info("\n" + "="*60)
        logger.info("Creating Hierarchical Computation Cones (Fast Lookup)")
        logger.info("="*60)

        # Build/load parent->children lookup table ONCE
        if cache_dir:
            cache_path = f"{cache_dir}/parent_to_children_res{self.parent_resolution}_to_{self.target_resolution}.pkl"
        else:
            cache_path = None

        self.parent_to_children = self._build_parent_lookup(
            regions_by_resolution,
            cache_path=cache_path
        )

        # Get all parent hexagons
        parent_regions = regions_by_resolution[self.parent_resolution]
        parent_hexagons = list(parent_regions.index)

        logger.info(f"\nParent hexagons at res{self.parent_resolution}: {len(parent_hexagons)}")

        # Create cones sequentially using fast lookup table
        # With dict lookups, this is fast enough (~1ms per cone = ~4 seconds for 408 cones)
        # No need for parallel processing (which adds pickle overhead)
        logger.info("Creating cones using fast dict lookups...")
        cones = []
        for parent_hex in tqdm(parent_hexagons, desc="Creating cones"):
            cone = self.create_cone(parent_hex, regions_by_resolution)
            cones.append(cone)

        # Statistics
        total_nodes = sum(cone.num_nodes() for cone in cones)
        avg_nodes = total_nodes / len(cones) if cones else 0

        logger.info(f"\nCone Statistics:")
        logger.info(f"  Total cones: {len(cones)}")
        logger.info(f"  Total nodes: {total_nodes}")
        logger.info(f"  Avg nodes per cone: {avg_nodes:.1f}")

        # Sample cone for inspection
        if cones:
            sample_cone = cones[0]
            logger.info(f"\nSample Cone ({sample_cone.cone_id}):")
            logger.info(f"  Parent neighbors: {len(sample_cone.parent_neighbors)}")
            for res, count in sample_cone.nodes_per_resolution().items():
                logger.info(f"  Res {res}: {count} hexagons")

        return cones

    # ========================================================================
    # GEOMETRIC OPTIMIZATIONS
    # ========================================================================

    def _spatial_sort_parent_hexagons(self, parent_hexagons: List[str]) -> List[str]:
        """
        Sort parent hexagons by H3 index for spatial locality.

        H3 indices follow a space-filling curve (hierarchical quadkey-like structure),
        so sorting by index groups spatially-adjacent hexagons together.

        Benefits:
        - Better cache locality when accessing descendants
        - Overlapping descendant sets processed consecutively
        - More predictable memory access patterns

        Args:
            parent_hexagons: List of parent hex IDs

        Returns:
            Spatially-sorted list of parent hex IDs
        """
        logger.info("Sorting parent hexagons by H3 space-filling curve order...")

        # H3 indices are strings, but they encode spatial position
        # Sorting lexicographically approximately follows spatial order
        sorted_hexes = sorted(parent_hexagons)

        logger.info(f"Sorted {len(sorted_hexes)} parent hexagons for spatial locality")

        return sorted_hexes

    def _batch_get_k_ring_neighbors(
        self,
        hex_ids: List[str],
        k_rings: int,
        regions_gdf: gpd.GeoDataFrame
    ) -> Dict[str, Set[str]]:
        """
        Get k-ring neighbors for multiple hexagons using SRAI batch operations.

        Geometric insight: For k=5, each hex has exactly 90 neighbors (1 + 3×5×6 - 1)
        Uses SRAI's H3Neighbourhood for consistent spatial operations.

        Args:
            hex_ids: List of hexagon IDs
            k_rings: Number of rings
            regions_gdf: GeoDataFrame to filter against

        Returns:
            Dictionary mapping hex_id to set of neighbor hex_ids
        """
        expected_neighbors = expected_k_ring_size(k_rings, include_center=False)
        logger.debug(f"Expected ~{expected_neighbors} neighbors per hexagon (k={k_rings})")

        available_hexes = set(regions_gdf.index)
        neighbors_dict = {}

        # Use SRAI's H3Neighbourhood for batch operations
        neighbourhood = H3Neighbourhood()

        # Batch process in chunks for better memory locality
        chunk_size = 1000
        for i in range(0, len(hex_ids), chunk_size):
            chunk = hex_ids[i:i+chunk_size]

            # Get chunk GeoDataFrame
            chunk_gdf = regions_gdf.loc[regions_gdf.index.intersection(chunk)]

            if len(chunk_gdf) == 0:
                continue

            # Use SRAI to get neighbors for each hexagon in chunk
            for hex_id in chunk_gdf.index:
                neighbors = set()

                # Single hex GeoDataFrame
                single_hex_gdf = chunk_gdf.loc[[hex_id]]

                # Accumulate neighbors at all distances 1 to k_rings
                for distance in range(1, k_rings + 1):
                    neighbors_at_distance = neighbourhood.get_neighbours_at_distance(
                        single_hex_gdf, distance
                    )

                    if neighbors_at_distance is not None and len(neighbors_at_distance) > 0:
                        neighbors.update(neighbors_at_distance.index.tolist())

                # Filter to available hexagons only
                neighbors = neighbors & available_hexes

                neighbors_dict[hex_id] = neighbors

        return neighbors_dict

    def _validate_cone_geometry(
        self,
        cone: HierarchicalCone,
        log_warnings: bool = True
    ) -> Dict[str, any]:
        """
        Validate cone against geometric expectations.

        Uses geometric formulas to check if cone size is reasonable,
        warning about significant deviations that might indicate errors.

        Args:
            cone: Hierarchical cone to validate
            log_warnings: Whether to log warnings

        Returns:
            Validation results [old 2024]
        """
        # Get actual counts
        actual_counts = cone.nodes_per_resolution()

        # Validate against geometric expectations
        result = validate_cone_size(
            actual_counts,
            self.parent_resolution,
            self.target_resolution,
            self.neighbor_rings,
            tolerance=0.30  # 30% tolerance for boundary effects
        )

        # Log warnings if requested
        if log_warnings and result['warnings']:
            logger.warning(f"Cone {cone.cone_id} geometry validation warnings:")
            for warning in result['warnings']:
                logger.warning(f"  {warning}")

        return result

    def create_all_cones_optimized(
        self,
        regions_by_resolution: Dict[int, gpd.GeoDataFrame],
        cache_dir: Optional[str] = None,
        spatial_sort: bool = True,
        validate_geometry: bool = True
    ) -> List[HierarchicalCone]:
        """
        Create all cones with geometric optimizations and validation.

        Optimizations:
        1. Spatial sorting for cache locality
        2. Geometric validation against expected sizes
        3. Enhanced logging with geometric insights
        4. Batch neighbor queries

        Args:
            regions_by_resolution: Dict mapping resolution -> GeoDataFrame
            cache_dir: Optional directory to cache parent lookup table
            spatial_sort: Sort cones by H3 space-filling curve
            validate_geometry: Validate cones against geometric expectations

        Returns:
            List of HierarchicalCone objects
        """
        logger.info("\n" + "="*60)
        logger.info("Creating Hierarchical Cones (Geometrically Optimized)")
        logger.info("="*60)

        # Log geometric expectations
        log_geometric_summary(
            self.parent_resolution,
            self.target_resolution,
            self.neighbor_rings,
            logger
        )

        # Build/load parent->children lookup table
        if cache_dir:
            cache_path = f"{cache_dir}/parent_to_children_res{self.parent_resolution}_to_{self.target_resolution}.pkl"
        else:
            cache_path = None

        self.parent_to_children = self._build_parent_lookup(
            regions_by_resolution,
            cache_path=cache_path
        )

        # Get parent hexagons
        parent_regions = regions_by_resolution[self.parent_resolution]
        parent_hexagons = list(parent_regions.index)

        logger.info(f"\nParent hexagons at res{self.parent_resolution}: {len(parent_hexagons)}")

        # Spatial sorting for cache locality
        if spatial_sort:
            parent_hexagons = self._spatial_sort_parent_hexagons(parent_hexagons)
            logger.info("Enabled spatial sorting for better cache locality")

        # Batch get k-ring neighbors for all parents (optimization)
        logger.info(f"Batch computing {self.neighbor_rings}-ring neighbors...")
        all_neighbors = self._batch_get_k_ring_neighbors(
            parent_hexagons,
            self.neighbor_rings,
            parent_regions
        )
        logger.info(f"Computed neighbors for {len(all_neighbors)} parents")

        # Create cones using pre-computed neighbors
        logger.info("Creating cones with geometric validation...")
        cones = []
        validation_results = []

        for parent_hex in tqdm(parent_hexagons, desc="Creating cones"):
            # Get pre-computed neighbors
            parent_neighbors = all_neighbors.get(parent_hex, set())

            # Build cone (similar to create_cone but with pre-computed neighbors)
            cone_roots = {parent_hex} | parent_neighbors

            descendants_by_resolution = {}
            for res in range(self.parent_resolution + 1, self.target_resolution + 1):
                if res not in regions_by_resolution:
                    descendants_by_resolution[res] = set()
                    continue

                # Lookup descendants from pre-computed table
                descendants = set()
                for root_hex in cone_roots:
                    children = self.parent_to_children.get((root_hex, res), set())
                    descendants.update(children)

                # Filter to available hexagons
                available_hexagons = set(regions_by_resolution[res].index)
                descendants = descendants & available_hexagons

                descendants_by_resolution[res] = descendants

            # Create cone
            cone = HierarchicalCone(
                cone_id=parent_hex,
                parent_resolution=self.parent_resolution,
                target_resolution=self.target_resolution,
                parent_hex=parent_hex,
                parent_neighbors=parent_neighbors,
                descendants_by_resolution=descendants_by_resolution
            )

            cones.append(cone)

            # Validate geometry if requested
            if validate_geometry and len(cones) % 100 == 0:
                # Validate sample cones
                result = self._validate_cone_geometry(cone, log_warnings=False)
                validation_results.append(result)

        # Aggregate validation results [old 2024]
        if validate_geometry and validation_results:
            num_valid = sum(1 for r in validation_results if r['valid'])
            logger.info(f"\nGeometric Validation: {num_valid}/{len(validation_results)} sample cones passed")

            if num_valid < len(validation_results):
                logger.warning(f"{len(validation_results) - num_valid} cones had geometric warnings (likely boundary effects)")

        # Statistics with geometric insights
        total_nodes = sum(cone.num_nodes() for cone in cones)
        avg_nodes = total_nodes / len(cones) if cones else 0

        # Expected vs actual comparison
        expected = expected_cone_size(
            self.parent_resolution,
            self.target_resolution,
            self.neighbor_rings
        )

        logger.info(f"\nCone Statistics:")
        logger.info(f"  Total cones: {len(cones)}")
        logger.info(f"  Total nodes (all cones, all resolutions): {total_nodes:,}")
        logger.info(f"  Avg nodes per cone: {avg_nodes:,.1f}")
        logger.info(f"  Expected avg (theoretical): {expected['total_descendants']:,}")
        logger.info(f"  Actual/Expected ratio: {avg_nodes / expected['total_descendants']:.2%}")

        # Sample cone for inspection with geometric comparison
        if cones:
            sample_cone = cones[0]
            logger.info(f"\nSample Cone ({sample_cone.cone_id}):")
            logger.info(f"  Parent neighbors: {len(sample_cone.parent_neighbors)}")

            expected_by_res = expected['descendants_per_resolution']
            for res, count in sample_cone.nodes_per_resolution().items():
                exp = expected_by_res.get(res, 0)
                ratio = count / exp if exp > 0 else 0
                logger.info(f"  Res {res}: {count:,} hexagons (expected: {exp:,}, ratio: {ratio:.2%})")

        return cones


class ConeBatcher:
    """
    LEGACY: Batches cones for parallel processing (ALL IN MEMORY).

    ⚠️ WARNING: Loads all cones into memory (~60 GB for Netherlands)

    USE INSTEAD: LazyConeBatcher (loads individual files on-demand, ~4.5 GB)

    Kept for backward compatibility with legacy inference scripts.
    New code should use LazyConeBatcher with cache_all_cones().
    """

    def __init__(self, cones: List[HierarchicalCone], batch_size: int = 32):
        """
        Initialize batcher.

        Args:
            cones: List of cones to batch (ALL LOADED IN MEMORY!)
            batch_size: Number of cones per batch
        """
        self.cones = cones
        self.batch_size = batch_size

        logger.warning("⚠️ Using LEGACY ConeBatcher - loads all cones in memory!")
        logger.warning("   Consider using LazyConeBatcher for 92% memory reduction")
        logger.info(f"Initialized ConeBatcher:")
        logger.info(f"  Total cones: {len(cones)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Num batches: {len(self)}")

    def __len__(self) -> int:
        """Number of batches."""
        return (len(self.cones) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches."""
        for i in range(0, len(self.cones), self.batch_size):
            yield self.cones[i:i + self.batch_size]

    def get_batch(self, batch_idx: int) -> List[HierarchicalCone]:
        """Get specific batch."""
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.cones))
        return self.cones[start:end]


class LazyConeBatcher:
    """
    TRUE lazy-loading cone batcher for memory-efficient training.

    Loads individual cone files from disk only when needed (batch by batch).
    NO massive pickle load - each cone is a separate file!

    MEMORY COMPARISON:
    - ConeBatcher (all in RAM): ~60 GB
    - LazyConeBatcher (individual files): ~4.5 GB per batch (32 cones only!)

    Reduction: 92% memory savings!
    """

    def __init__(
        self,
        parent_hexagons: List[str],
        cache_dir: str,
        batch_size: int = 32
    ):
        """
        Initialize lazy batcher.

        Args:
            parent_hexagons: Ordered list of parent hex IDs
            cache_dir: Directory containing individual cone files (cone_{hex}.pkl)
            batch_size: Number of cones per batch
        """
        self.parent_hexagons = parent_hexagons
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size

        # Verify cache directory exists
        if not self.cache_dir.exists():
            raise ValueError(f"Cache directory not found: {cache_dir}")

        # Count cached cone files
        cone_files = list(self.cache_dir.glob("cone_*.pkl"))
        logger.info(f"LazyConeBatcher initialized:")
        logger.info(f"  Cone cache dir: {cache_dir}")
        logger.info(f"  Cached cones: {len(cone_files)} files")
        logger.info(f"  Total cones: {len(parent_hexagons)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Num batches: {len(self)}")
        logger.info(f"  Memory per batch: ~{batch_size * 0.144:.1f} GB (vs {len(parent_hexagons) * 0.144:.1f} GB for all)")
        logger.info(f"  TRUE lazy loading - only loads 32 cones at a time!")

    def __len__(self) -> int:
        """Number of batches."""
        return (len(self.parent_hexagons) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Iterate over batches, loading individual cone files on-demand.

        Each iteration:
        1. Loads ONLY the cone files needed for this batch (32 × 144 MB = ~4.5 GB)
        2. Yields the batch
        3. Python GC automatically frees memory when batch goes out of scope
        4. Next batch loads fresh cone files

        NO massive 60 GB pickle load - true lazy loading!
        """
        for i in range(0, len(self.parent_hexagons), self.batch_size):
            batch_parent_hexes = self.parent_hexagons[i:i + self.batch_size]

            # Load ONLY the cone files for this batch
            batch_cones = []
            for parent_hex in batch_parent_hexes:
                cone_file = self.cache_dir / f"cone_{parent_hex}.pkl"
                with open(cone_file, 'rb') as f:
                    cone = pickle.load(f)
                batch_cones.append(cone)

            yield batch_cones

            # Batch goes out of scope after yield, Python GC frees memory

    def get_batch(self, batch_idx: int) -> List[HierarchicalCone]:
        """Get specific batch by index."""
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.parent_hexagons))
        batch_parent_hexes = self.parent_hexagons[start:end]

        # Load individual cone files
        batch_cones = []
        for parent_hex in batch_parent_hexes:
            cone_file = self.cache_dir / f"cone_{parent_hex}.pkl"
            with open(cone_file, 'rb') as f:
                cone = pickle.load(f)
            batch_cones.append(cone)

        return batch_cones


def example_usage():
    """Example of how to use the hierarchical cone masking system."""
    from utils.paths import StudyAreaPaths

    # Load regions at multiple resolutions
    study_area = "netherlands"
    paths = StudyAreaPaths(study_area)
    regions_by_resolution = {}

    for res in range(5, 11):
        regions_path = paths.region_file(res)
        regions_by_resolution[res] = gpd.read_parquet(regions_path)
        logger.info(f"Loaded res{res}: {len(regions_by_resolution[res])} hexagons")

    # Create cone system
    cone_system = HierarchicalConeMaskingSystem(
        parent_resolution=5,
        target_resolution=10,
        neighbor_rings=5
    )

    # Create all cones
    cones = cone_system.create_all_cones(regions_by_resolution)

    # Create batcher
    batcher = ConeBatcher(cones, batch_size=32)

    # Process batches
    for batch_idx, cone_batch in enumerate(batcher):
        logger.info(f"Processing batch {batch_idx+1}/{len(batcher)}")
        logger.info(f"  Cones in batch: {len(cone_batch)}")

        # Each cone in the batch can be processed independently
        for cone in cone_batch:
            # Get mask for global graph
            # mask = cone.get_mask(global_hex_list)

            # Apply mask to features/edges
            # cone_features = features[mask]
            # cone_edges = mask_edges(edges, mask)

            # Run forward pass on this cone
            # output = model(cone_features, cone_edges)

            pass

        # Only process first batch for example
        break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()