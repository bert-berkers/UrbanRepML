"""
H3 Geometry Module
==================

Geometric properties and optimizations for H3 hierarchical hexagonal tessellation.

This module provides:
- Geometric formulas based on H3 properties (1+7k expansion, k-ring neighborhoods)
- Validation functions for cone sizes and connectivity
- Utilities for spatial ordering and optimization

Key Functions:
- expected_children_count: Calculate 7^k descendants
- expected_k_ring_size: Calculate 1 + 3k(k+1) neighbors
- expected_cone_size: Full cone geometry calculation
- validate_cone_size: Verify actual vs expected sizes
"""

from .h3_geometry import (
    # Hierarchical geometry
    expected_children_count,
    expected_total_descendants,
    descendants_by_resolution,

    # Spatial geometry
    expected_k_ring_size,
    expected_ring_size,

    # Cone geometry
    expected_cone_size,

    # Edge geometry
    expected_edge_count,
    expected_edge_count_bidirectional,

    # Validation
    validate_cone_size,
    validate_edge_count,

    # Utilities
    geometric_series_sum,
    log_geometric_summary,
)

__all__ = [
    'expected_children_count',
    'expected_total_descendants',
    'descendants_by_resolution',
    'expected_k_ring_size',
    'expected_ring_size',
    'expected_cone_size',
    'expected_edge_count',
    'expected_edge_count_bidirectional',
    'validate_cone_size',
    'validate_edge_count',
    'geometric_series_sum',
    'log_geometric_summary',
]
