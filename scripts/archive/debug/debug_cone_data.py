#!/usr/bin/env python
"""
Debug Cone Data Structure

Inspect the cone data to understand dimension mismatches.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stage2_fusion.data.cone_dataset import ConeDataset

def debug_cone_data():
    """Debug cone data structure."""

    print("=" * 60)
    print("Debugging Cone Data Structure")
    print("=" * 60)

    # Create dataset
    print("\nCreating dataset...")
    dataset = ConeDataset(
        study_area="netherlands",
        parent_resolution=5,
        target_resolution=10,
        neighbor_rings=5
    )

    # Get first cone
    print("\nLoading first cone...")
    cone_data = dataset[0]

    print(f"\nCone ID: {cone_data['cone_id']}")
    print(f"\nFeatures shape: {cone_data['features_res10'].shape}")

    print(f"\nNodes per resolution:")
    total_nodes = 0
    for res in [5, 6, 7, 8, 9, 10]:
        count = cone_data['num_nodes_per_res'][res]
        print(f"  Res {res}: {count:,} nodes")
        total_nodes += count

    print(f"\nTotal nodes in cone: {total_nodes:,}")
    print(f"\nNumber of local indices per resolution:")
    for res in [5, 6, 7, 8, 9, 10]:
        if res in cone_data['hex_to_local_idx_by_res']:
            count = len(cone_data['hex_to_local_idx_by_res'][res])
            print(f"  Res {res}: {count:,} indices")

    print(f"\nHierarchical mappings:")
    for child_res in [6, 7, 8, 9, 10]:
        if child_res in cone_data['hierarchical_mappings']:
            child_to_parent, num_parents = cone_data['hierarchical_mappings'][child_res]
            print(f"  Res {child_res} -> {child_res-1}:")
            print(f"    child_to_parent length: {len(child_to_parent):,}")
            print(f"    num_parents: {num_parents:,}")
            print(f"    child_to_parent range: [{child_to_parent.min()}, {child_to_parent.max()}]")

    print(f"\nSpatial edges:")
    for res in [5, 6, 7, 8, 9, 10]:
        if res in cone_data['spatial_edges']:
            edge_index, edge_weight = cone_data['spatial_edges'][res]
            print(f"  Res {res}:")
            print(f"    edges: {edge_index.shape[1]:,}")
            if edge_index.shape[1] > 0:
                print(f"    node indices range: [{edge_index.min()}, {edge_index.max()}]")

    print("\n" + "=" * 60)
    print("Analysis:")
    print("=" * 60)
    print(f"Problem: Features are only for res10 ({cone_data['features_res10'].shape[0]:,} nodes)")
    print(f"But hierarchical mappings use global cone indices (0 to {total_nodes-1:,})")
    print(f"\nThe model architecture needs to:")
    print(f"1. Track which global indices belong to which resolution")
    print(f"2. Extract appropriate subsets of features per resolution")
    print(f"3. Or use per-resolution feature tensors instead of global indices")

if __name__ == "__main__":
    debug_cone_data()
