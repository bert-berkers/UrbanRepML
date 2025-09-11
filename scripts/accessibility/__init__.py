"""
Accessibility Graph Construction Scripts

Utilities for creating accessibility-based spatial graphs using SRAI.

The three-stage pipeline:
1. floodfill_travel_time.py - Calculate local travel times
2. gravity_weighting.py - Apply building density weighting
3. percentile_pruning.py - Create sparse graphs for GCN training

All scripts use SRAI for H3 operations and spatial analysis.
"""

from .floodfill_travel_time import calculate_floodfill_travel_times, calculate_local_accessibility_matrix
from .gravity_weighting import calculate_gravity_weights, apply_gravity_weighting
from .percentile_pruning import prune_graph_by_percentile, create_pruned_accessibility_graph

__all__ = [
    'calculate_floodfill_travel_times',
    'calculate_local_accessibility_matrix',
    'calculate_gravity_weights', 
    'apply_gravity_weighting',
    'prune_graph_by_percentile',
    'create_pruned_accessibility_graph'
]