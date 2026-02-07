"""
Unit stage3_analysis for H3 geometric formulas.

Validates all geometric calculations against actual H3 library operations
to ensure formulas are correct.
"""

import pytest
import h3
from stage2_fusion.geometry import (
    expected_children_count,
    expected_total_descendants,
    descendants_by_resolution,
    expected_k_ring_size,
    expected_ring_size,
    expected_cone_size,
    expected_edge_count,
    validate_cone_size,
    validate_edge_count,
)


class TestHierarchicalGeometry:
    """Test hierarchical expansion formulas (1+7k)."""

    def test_single_level_expansion(self):
        """Test parent -> immediate children (7 children)."""
        # Each H3 parent has exactly 7 children
        assert expected_children_count(5, 6) == 7
        assert expected_children_count(8, 9) == 7
        assert expected_children_count(10, 11) == 7

    def test_multi_level_expansion(self):
        """Test multi-level descendant counts."""
        # 7^2 = 49 grandchildren
        assert expected_children_count(5, 7) == 49

        # 7^3 = 343
        assert expected_children_count(5, 8) == 343

        # 7^5 = 16,807 (res5 -> res10)
        assert expected_children_count(5, 10) == 16807

    def test_against_actual_h3(self):
        """Validate formula against actual H3 cell_to_children."""
        # Use a specific hex at res5
        test_hex = h3.latlng_to_cell(52.0, 5.0, 5)  # Netherlands

        # Test immediate children (res6)
        actual_children = list(h3.cell_to_children(test_hex, 6))
        expected = expected_children_count(5, 6)
        assert len(actual_children) == expected == 7

        # Test grandchildren (res7)
        actual_grandchildren = list(h3.cell_to_children(test_hex, 7))
        expected = expected_children_count(5, 7)
        assert len(actual_grandchildren) == expected == 49

    def test_total_descendants(self):
        """Test cumulative descendant count."""
        # Res5 -> Res6: 7
        # Res5 -> Res7: 7 + 49 = 56
        # Total should be (7^6 - 7) / 6 = (117,649 - 7) / 6 = 19,607
        total = expected_total_descendants(5, 10)

        # Verify by summing individual levels
        manual_sum = sum(
            expected_children_count(5, res)
            for res in range(6, 11)
        )
        assert total == manual_sum == 19607

    def test_descendants_by_resolution(self):
        """Test resolution breakdown."""
        desc = descendants_by_resolution(5, 10)

        assert desc[5] == 1  # Parent itself
        assert desc[6] == 7
        assert desc[7] == 49
        assert desc[8] == 343
        assert desc[9] == 2401
        assert desc[10] == 16807


class TestSpatialGeometry:
    """Test k-ring neighborhood formulas."""

    def test_k_ring_formula(self):
        """Test k-ring size formula: 1 + 3k(k+1)."""
        # k=0: just center
        assert expected_k_ring_size(0) == 1

        # k=1: center + 6 neighbors = 7
        assert expected_k_ring_size(1) == 7

        # k=2: 1 + 3(2)(3) = 1 + 18 = 19
        assert expected_k_ring_size(2) == 19

        # k=5: 1 + 3(5)(6) = 1 + 90 = 91
        assert expected_k_ring_size(5) == 91

    def test_k_ring_exclude_center(self):
        """Test k-ring without center hexagon."""
        # k=1 without center: 6 neighbors
        assert expected_k_ring_size(1, include_center=False) == 6

        # k=5 without center: 90 neighbors
        assert expected_k_ring_size(5, include_center=False) == 90

    def test_against_actual_h3_grid_disk(self):
        """Validate formula against actual H3 grid_disk."""
        test_hex = h3.latlng_to_cell(52.0, 5.0, 9)  # Netherlands

        # Test various k values
        for k in [1, 2, 3, 5]:
            actual_ring = list(h3.grid_disk(test_hex, k))
            expected = expected_k_ring_size(k)

            assert len(actual_ring) == expected, \
                f"k={k}: actual {len(actual_ring)} != expected {expected}"

    def test_ring_sizes(self):
        """Test individual ring sizes."""
        assert expected_ring_size(0) == 1  # Center
        assert expected_ring_size(1) == 6  # First ring
        assert expected_ring_size(2) == 12  # Second ring
        assert expected_ring_size(5) == 30  # Fifth ring

        # Verify sum equals k-ring formula
        for k in range(1, 6):
            ring_sum = sum(expected_ring_size(i) for i in range(k + 1))
            assert ring_sum == expected_k_ring_size(k)


class TestConeGeometry:
    """Test combined hierarchical + spatial cone geometry."""

    def test_cone_size_calculation(self):
        """Test full cone size calculation."""
        # Res5->Res10 with k=5 neighbors
        cone = expected_cone_size(5, 10, 5)

        # Root hexagons: 91 (1 parent + 90 neighbors)
        assert cone['roots'] == 91

        # Each root has 16,807 descendants at res10
        # Total: 91 × 16,807 = 1,529,437
        assert cone['theoretical_max'] == 91 * 16807 == 1529437

        # Verify breakdown by resolution
        assert cone['descendants_per_resolution'][5] == 91
        assert cone['descendants_per_resolution'][6] == 91 * 7
        assert cone['descendants_per_resolution'][10] == 91 * 16807

    def test_cone_scaling(self):
        """Test how cone size scales with parameters."""
        # Smaller k-ring -> smaller cone
        cone_k1 = expected_cone_size(5, 10, 1)
        cone_k5 = expected_cone_size(5, 10, 5)

        assert cone_k1['roots'] == 7  # 1 + 6 neighbors
        assert cone_k5['roots'] == 91  # 1 + 90 neighbors
        assert cone_k5['theoretical_max'] > cone_k1['theoretical_max']

        # Fewer resolution levels -> smaller cone
        cone_short = expected_cone_size(5, 7, 5)
        cone_long = expected_cone_size(5, 10, 5)

        assert cone_short['theoretical_max'] == 91 * 49  # Only to res7
        assert cone_long['theoretical_max'] == 91 * 16807  # To res10

    def test_realistic_cone_sizes(self):
        """Test cone sizes for actual use cases."""
        # Netherlands study area: res5->res10, k=5
        nl_cone = expected_cone_size(5, 10, 5)

        # Should have ~1.5M hexagons at finest resolution per cone
        assert nl_cone['theoretical_max'] > 1_000_000
        assert nl_cone['theoretical_max'] < 2_000_000

        # Total descendants across all levels
        assert nl_cone['total_descendants'] > 1_500_000


class TestEdgeGeometry:
    """Test graph edge count formulas."""

    def test_edge_count_formula(self):
        """Test edge count calculation."""
        # 1000 hexagons, k=1 connectivity
        # Each hex connects to ~6 neighbors
        # Total: 1000 × 6 = 6000 directed edges
        edges_k1 = expected_edge_count(1000, 1)
        assert edges_k1 == 1000 * 6 == 6000

        # k=5 connectivity
        # Each hex connects to 90 neighbors
        edges_k5 = expected_edge_count(1000, 5)
        assert edges_k5 == 1000 * 90 == 90000

    def test_edge_scaling(self):
        """Test how edges scale with k and N."""
        # More hexagons -> more edges (linear)
        edges_small = expected_edge_count(100, 5)
        edges_large = expected_edge_count(1000, 5)
        assert edges_large == 10 * edges_small

        # Larger k -> more edges per hex (quadratic in k)
        edges_k1 = expected_edge_count(1000, 1)
        edges_k2 = expected_edge_count(1000, 2)

        # k=2 should have more edges than k=1
        # k=2: 1000 × 18 = 18,000
        # k=1: 1000 × 6 = 6,000
        assert edges_k2 == 18000
        assert edges_k2 > edges_k1


class TestValidation:
    """Test validation functions."""

    def test_validate_cone_size_perfect_match(self):
        """Test validation with exact match."""
        actual = {
            5: 91,
            6: 637,
            7: 4459,
            8: 31213,
            9: 218491,
            10: 1529437
        }

        result = validate_cone_size(actual, 5, 10, 5, tolerance=0.01)

        assert result['valid'] == True
        assert len(result['warnings']) == 0

        # All ratios should be 1.0
        for res, comp in result['comparisons'].items():
            assert abs(comp['ratio'] - 1.0) < 0.001

    def test_validate_cone_size_with_boundaries(self):
        """Test validation with realistic boundary effects."""
        # Simulate ~20% reduction due to study area boundaries
        actual = {
            5: 91,  # Roots usually match exactly
            6: 500,  # ~80% of expected 637
            7: 3500,  # ~80% of expected 4459
            8: 25000,  # ~80% of expected 31213
            9: 175000,  # ~80% of expected 218491
            10: 1200000  # ~78% of expected 1529437
        }

        result = validate_cone_size(actual, 5, 10, 5, tolerance=0.25)

        # Should be valid with 25% tolerance
        # (might have warnings for being below expected, which is normal)
        for res, comp in result['comparisons'].items():
            # Actual should be less than expected (due to boundaries)
            assert comp['ratio'] <= 1.0

    def test_validate_edge_count(self):
        """Test edge count validation."""
        # Perfect interior region
        num_hexes = 1000
        k = 5
        actual_edges = 90000  # 1000 × 90

        result = validate_edge_count(actual_edges, num_hexes, k, tolerance=0.1)

        assert result['valid'] == True
        assert abs(result['ratio'] - 1.0) < 0.01

        # Boundary region (fewer edges)
        actual_edges_boundary = 70000  # ~80% of expected
        result_boundary = validate_edge_count(
            actual_edges_boundary, num_hexes, k, tolerance=0.25
        )

        # Should still be valid with larger tolerance
        assert result_boundary['ratio'] < 1.0


class TestGeometricSeriesUtilities:
    """Test utility functions."""

    def test_geometric_series(self):
        """Test geometric series summation."""
        from stage2_fusion.geometry.h3_geometry import geometric_series_sum

        # Sum of 7^1 + 7^2 + 7^3 = 7 + 49 + 343 = 399
        assert geometric_series_sum(7, 1, 3) == 399

        # Sum of 7^1 to 7^5 (used in total descendants)
        # = (7^6 - 7) / 6 = (117649 - 7) / 6 = 19607
        assert geometric_series_sum(7, 1, 5) == 19607

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Zero children
        with pytest.raises(ValueError):
            expected_children_count(10, 5)  # Child < parent

        # Negative k-ring
        with pytest.raises(ValueError):
            expected_k_ring_size(-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
