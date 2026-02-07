"""
H3 Geometric Properties and Formulas
=====================================

This module provides geometric calculations based on H3's hierarchical hexagonal tessellation.
All formulas are derived from H3's fundamental geometric properties.

Key Geometric Constants:
- H3 hierarchy: Each parent → exactly 7 children (1+7k expansion)
- K-ring neighbors: 1 + 3k(k+1) hexagons in k-ring (including center)
- Hexagon symmetry: 6 immediate neighbors for interior hexagons

References:
- H3 Documentation: https://h3geo.org/docs/
- Hierarchical Expansion: 7-ary tree structure
- K-ring Formula: Sum of hexagonal rings = 1 + Σ(6i) for i=1 to k
"""

import logging
from typing import Dict, Tuple, Optional
import math

logger = logging.getLogger(__name__)


# ============================================================================
# HIERARCHICAL GEOMETRY (1+7k Expansion)
# ============================================================================

def expected_children_count(parent_res: int, child_res: int) -> int:
    """
    Calculate expected number of children from parent to child resolution.

    H3 Hierarchical Property: Each parent hexagon has exactly 7 children
    at the next finer resolution. This creates a 7-ary tree structure.

    Formula: 7^(child_res - parent_res)

    Examples:
        - Res5 → Res6: 7^1 = 7 children
        - Res5 → Res7: 7^2 = 49 grandchildren
        - Res5 → Res10: 7^5 = 16,807 descendants

    Args:
        parent_res: Parent resolution level
        child_res: Child resolution level (must be >= parent_res)

    Returns:
        Expected number of descendants

    Raises:
        ValueError: If child_res < parent_res
    """
    if child_res < parent_res:
        raise ValueError(f"Child resolution {child_res} must be >= parent {parent_res}")

    if child_res == parent_res:
        return 1

    depth = child_res - parent_res
    return 7 ** depth


def expected_total_descendants(parent_res: int, target_res: int) -> int:
    """
    Calculate total descendants across all resolutions from parent to target.

    This sums all descendants at each intermediate resolution level.

    Formula: Σ(7^i) for i=1 to (target_res - parent_res)
           = (7^(n+1) - 7) / 6  where n = target_res - parent_res

    Example (Res5 → Res10):
        Res6: 7
        Res7: 49
        Res8: 343
        Res9: 2,401
        Res10: 16,807
        Total: 19,607 descendants

    Args:
        parent_res: Starting resolution
        target_res: Ending resolution

    Returns:
        Total number of descendants across all levels
    """
    if target_res <= parent_res:
        return 0

    depth = target_res - parent_res

    # Geometric series sum: a(r^n - 1)/(r - 1) where a=7, r=7
    # = 7(7^n - 1)/6 = (7^(n+1) - 7)/6
    total = (7 ** (depth + 1) - 7) // 6

    return total


def descendants_by_resolution(parent_res: int, target_res: int) -> Dict[int, int]:
    """
    Get expected descendant counts at each resolution level.

    Returns a dictionary mapping each resolution to expected number of hexagons.

    Args:
        parent_res: Starting resolution
        target_res: Ending resolution

    Returns:
        Dictionary: {resolution: expected_count}
    """
    counts = {}

    for res in range(parent_res, target_res + 1):
        if res == parent_res:
            counts[res] = 1  # Just the parent itself
        else:
            counts[res] = expected_children_count(parent_res, res)

    return counts


# ============================================================================
# SPATIAL GEOMETRY (K-Ring Neighborhoods)
# ============================================================================

def expected_k_ring_size(k: int, include_center: bool = True) -> int:
    """
    Calculate expected size of k-ring neighborhood.

    K-Ring Formula: 1 + 3k(k+1) hexagons in k-ring (including center)

    Derivation:
        - Ring 0 (center): 1 hexagon
        - Ring 1: 6 hexagons (immediate neighbors)
        - Ring 2: 12 hexagons
        - Ring i: 6i hexagons
        - Total for k rings: 1 + 6(1) + 6(2) + ... + 6(k)
                            = 1 + 6·Σ(i) for i=1 to k
                            = 1 + 6·(k(k+1)/2)
                            = 1 + 3k(k+1)

    Examples:
        - k=1: 1 + 3(1)(2) = 7 hexagons (center + 6 neighbors)
        - k=2: 1 + 3(2)(3) = 19 hexagons
        - k=5: 1 + 3(5)(6) = 91 hexagons

    Args:
        k: Number of rings
        include_center: Whether to include center hexagon in count

    Returns:
        Expected number of hexagons in k-ring
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    total = 1 + 3 * k * (k + 1)

    if not include_center:
        total -= 1

    return total


def expected_ring_size(ring_distance: int) -> int:
    """
    Calculate expected size of a single ring at distance k.

    Formula: 6k hexagons at ring distance k (k > 0)
           : 1 hexagon at ring distance 0 (center)

    Args:
        ring_distance: Distance from center (0, 1, 2, ...)

    Returns:
        Expected number of hexagons in that specific ring
    """
    if ring_distance < 0:
        raise ValueError(f"Ring distance must be >= 0, got {ring_distance}")

    if ring_distance == 0:
        return 1

    return 6 * ring_distance


# ============================================================================
# CONE GEOMETRY (Hierarchical + Spatial Combined)
# ============================================================================

def expected_cone_size(
    parent_res: int,
    target_res: int,
    neighbor_rings: int,
    include_parent_in_descendants: bool = False
) -> Dict[str, int]:
    """
    Calculate expected size of a hierarchical cone.

    A cone consists of:
    1. Parent hexagon + k-ring neighbors at parent resolution
    2. All descendants of each through finer resolutions

    Formula:
        - Roots at parent_res: 1 + 3·k(k+1) = k-ring neighbors + parent
        - Each root expands: 7^(target_res - parent_res) descendants
        - Total at target_res: roots × 7^depth

    Example (Res5→Res10, k=5):
        - Roots at res5: 91 hexagons (parent + 5-ring neighbors)
        - Each root → 16,807 descendants at res10
        - Theoretical max: 91 × 16,807 = 1,529,437 hexagons
        - Actual: Less due to study area boundaries

    Args:
        parent_res: Coarse resolution (cone roots)
        target_res: Fine resolution (cone leaves)
        neighbor_rings: Number of neighbor rings around parent
        include_parent_in_descendants: Whether parent counts as descendant

    Returns:
        Dictionary with breakdown:
            - 'roots': Number of root hexagons at parent_res
            - 'total_descendants': Total across all resolutions
            - 'descendants_per_resolution': Dict[res, count]
            - 'theoretical_max': Maximum possible size
    """
    # Number of root hexagons at parent resolution
    roots = expected_k_ring_size(neighbor_rings, include_center=True)

    # Descendants at each resolution
    desc_by_res = {}
    total_descendants = 0

    for res in range(parent_res, target_res + 1):
        if res == parent_res:
            # Just the roots themselves
            count = roots
        else:
            # Each root expands to children at this resolution
            children_per_root = expected_children_count(parent_res, res)
            count = roots * children_per_root

        desc_by_res[res] = count

        if res > parent_res or include_parent_in_descendants:
            total_descendants += count

    return {
        'roots': roots,
        'total_descendants': total_descendants,
        'descendants_per_resolution': desc_by_res,
        'theoretical_max': desc_by_res[target_res]  # Max at finest resolution
    }


# ============================================================================
# EDGE GEOMETRY (Graph Connectivity)
# ============================================================================

def expected_edge_count(num_hexagons: int, neighbor_rings: int) -> int:
    """
    Calculate expected number of edges for k-ring connectivity.

    Formula (approximate, for interior hexagons):
        Each hexagon connects to all within k-ring: 3k(k+1) neighbors
        Total directed edges: N × 3k(k+1)
        Undirected edges: N × 3k(k+1) / 2

    Note: Actual count is lower due to:
        - Boundary hexagons (fewer neighbors)
        - Study area boundaries (missing hexagons)
        - Non-overlapping k-rings at edges

    This formula gives an upper bound for interior regions.

    Examples:
        - N=1000, k=1: ~1000 × 6 = 6,000 directed edges
        - N=1000, k=5: ~1000 × 90 = 90,000 directed edges

    Args:
        num_hexagons: Number of hexagons in region
        neighbor_rings: K-ring connectivity

    Returns:
        Expected number of directed edges (upper bound)
    """
    if num_hexagons <= 0:
        return 0

    neighbors_per_hex = expected_k_ring_size(neighbor_rings, include_center=False)

    # Directed edges (each hex → all neighbors)
    directed_edges = num_hexagons * neighbors_per_hex

    return directed_edges


def expected_edge_count_bidirectional(num_hexagons: int, neighbor_rings: int) -> int:
    """
    Calculate expected bidirectional edges (src→tgt and tgt→src).

    For graph neural networks with bidirectional message passing.

    Args:
        num_hexagons: Number of hexagons
        neighbor_rings: K-ring connectivity

    Returns:
        Expected bidirectional edge count (2× undirected)
    """
    # Each undirected edge becomes 2 directed edges
    undirected = expected_edge_count(num_hexagons, neighbor_rings)
    return undirected  # Already counting each edge once per direction


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_cone_size(
    actual_counts: Dict[int, int],
    parent_res: int,
    target_res: int,
    neighbor_rings: int,
    tolerance: float = 0.1
) -> Dict[str, any]:
    """
    Validate actual cone size against geometric expectations.

    Checks if actual counts are within expected range, accounting for
    boundary effects and study area limitations.

    Args:
        actual_counts: Actual hexagon counts by resolution
        parent_res: Parent resolution
        target_res: Target resolution
        neighbor_rings: Neighbor rings
        tolerance: Acceptable deviation ratio (default 10%)

    Returns:
        Validation results with warnings if outside expected range
    """
    expected = expected_cone_size(
        parent_res, target_res, neighbor_rings
    )['descendants_per_resolution']

    results = {
        'valid': True,
        'warnings': [],
        'comparisons': {}
    }

    for res, actual_count in actual_counts.items():
        if res not in expected:
            continue

        expected_count = expected[res]
        ratio = actual_count / expected_count if expected_count > 0 else 0

        results['comparisons'][res] = {
            'actual': actual_count,
            'expected': expected_count,
            'ratio': ratio,
            'deviation': abs(1 - ratio)
        }

        # Check if within tolerance (allowing for smaller due to boundaries)
        if ratio > (1 + tolerance):
            results['valid'] = False
            results['warnings'].append(
                f"Res{res}: Actual count {actual_count} exceeds expected {expected_count} "
                f"by {(ratio - 1) * 100:.1f}%"
            )
        elif ratio < (1 - tolerance) and res == parent_res:
            # Root count should match closely (not affected by descendants)
            results['warnings'].append(
                f"Res{res}: Actual count {actual_count} is {(1 - ratio) * 100:.1f}% "
                f"below expected {expected_count}"
            )

    return results


def validate_edge_count(
    actual_edges: int,
    num_hexagons: int,
    neighbor_rings: int,
    tolerance: float = 0.2
) -> Dict[str, any]:
    """
    Validate actual edge count against geometric expectations.

    Args:
        actual_edges: Actual number of edges
        num_hexagons: Number of hexagons
        neighbor_rings: K-ring connectivity
        tolerance: Acceptable deviation (default 20% due to boundaries)

    Returns:
        Validation results
    """
    expected_edges = expected_edge_count(num_hexagons, neighbor_rings)
    ratio = actual_edges / expected_edges if expected_edges > 0 else 0

    result = {
        'actual_edges': actual_edges,
        'expected_edges': expected_edges,
        'num_hexagons': num_hexagons,
        'neighbor_rings': neighbor_rings,
        'ratio': ratio,
        'valid': True,
        'warnings': []
    }

    if ratio > (1 + tolerance):
        result['valid'] = False
        result['warnings'].append(
            f"Edge count {actual_edges} exceeds expected {expected_edges} by "
            f"{(ratio - 1) * 100:.1f}%"
        )
    elif ratio < 0.5:  # Very low might indicate an error
        result['warnings'].append(
            f"Edge count {actual_edges} is only {ratio * 100:.1f}% of expected "
            f"{expected_edges} - check for connectivity issues"
        )

    return result


# ============================================================================
# UTILITIES
# ============================================================================

def geometric_series_sum(base: int, start_power: int, end_power: int) -> int:
    """
    Calculate sum of geometric series: Σ(base^i) for i=start to end.

    Formula: base^start × (base^(n+1) - 1) / (base - 1)
             where n = end - start

    Args:
        base: Base of the series (e.g., 7 for H3)
        start_power: Starting exponent
        end_power: Ending exponent (inclusive)

    Returns:
        Sum of geometric series
    """
    if end_power < start_power:
        return 0

    if base == 1:
        return end_power - start_power + 1

    n = end_power - start_power
    numerator = (base ** start_power) * (base ** (n + 1) - 1)
    denominator = base - 1

    return numerator // denominator


def log_geometric_summary(
    parent_res: int,
    target_res: int,
    neighbor_rings: int,
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Log a summary of geometric expectations for a cone configuration.

    Useful for debugging and understanding cone sizes before construction.

    Args:
        parent_res: Parent resolution
        target_res: Target resolution
        neighbor_rings: Neighbor rings
        logger_instance: Logger to use (default: module logger)
    """
    log = logger_instance or logger

    cone_size = expected_cone_size(parent_res, target_res, neighbor_rings)

    log.info("="*60)
    log.info("H3 Geometric Expectations")
    log.info("="*60)
    log.info(f"Configuration:")
    log.info(f"  Parent resolution: {parent_res}")
    log.info(f"  Target resolution: {target_res}")
    log.info(f"  Neighbor rings: {neighbor_rings}")
    log.info(f"")
    log.info(f"Cone Geometry:")
    log.info(f"  Root hexagons at res{parent_res}: {cone_size['roots']}")
    log.info(f"  Total descendants: {cone_size['total_descendants']:,}")
    log.info(f"  Theoretical max at res{target_res}: {cone_size['theoretical_max']:,}")
    log.info(f"")
    log.info(f"By Resolution:")
    for res, count in sorted(cone_size['descendants_per_resolution'].items()):
        log.info(f"  Res{res}: {count:,} hexagons")
    log.info("="*60)


if __name__ == "__main__":
    # Example usage and validation
    logging.basicConfig(level=logging.INFO)

    # Test geometric formulas
    print("\n=== H3 Geometric Formulas Demo ===\n")

    # Hierarchical expansion
    print("Hierarchical Expansion (1+7k):")
    for depth in range(1, 6):
        children = expected_children_count(5, 5 + depth)
        print(f"  Res5 -> Res{5+depth}: 7^{depth} = {children:,} descendants")

    # K-ring sizes
    print("\nK-Ring Sizes (1 + 3k(k+1)):")
    for k in range(1, 6):
        size = expected_k_ring_size(k)
        print(f"  {k}-ring: {size} hexagons")

    # Full cone geometry
    print("\nCone Geometry (Res5->Res10, k=5 neighbors):")
    log_geometric_summary(5, 10, 5)
