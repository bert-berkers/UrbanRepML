"""
Import smoke tests for all three stages of the UrbanRepML pipeline.

Verifies that every public module and class can be imported without error.
Each import is isolated in its own test function so failures are independent.

Also includes H3 compliance tests that verify no stage code uses banned
h3-py functions (tessellation/neighborhood) that should go through SRAI.
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Stage 1: Modality Encoders
# ---------------------------------------------------------------------------


class TestStage1Imports:
    """Import smoke tests for stage1_modalities package."""

    def test_import_base_module(self):
        from stage1_modalities.base import ModalityProcessor
        assert ModalityProcessor is not None

    def test_import_package_init(self):
        import stage1_modalities
        assert hasattr(stage1_modalities, "load_modality_processor")
        assert hasattr(stage1_modalities, "get_available_modalities")

    def test_import_load_modality_processor(self):
        from stage1_modalities import load_modality_processor
        assert callable(load_modality_processor)

    def test_import_alphaearth_processor(self):
        from stage1_modalities.alphaearth.processor import AlphaEarthProcessor
        assert AlphaEarthProcessor is not None

    def test_load_modality_processor_alphaearth(self):
        """Factory function returns an AlphaEarthProcessor instance."""
        from stage1_modalities import load_modality_processor
        from stage1_modalities.alphaearth.processor import AlphaEarthProcessor

        processor = load_modality_processor("alphaearth")
        assert isinstance(processor, AlphaEarthProcessor)

    def test_load_modality_processor_aerial_imagery_fails(self):
        """Known broken: AerialImageryProcessor class does not exist.

        The __init__.py imports AerialImageryProcessor from
        aerial_imagery/processor.py, but that file only contains imports
        and no class definition. This test documents the known failure.
        """
        from stage1_modalities import load_modality_processor

        with pytest.raises(ImportError):
            load_modality_processor("aerial_imagery")

    def test_load_modality_processor_unknown_raises(self):
        from stage1_modalities import load_modality_processor

        with pytest.raises(ValueError, match="Unknown modality"):
            load_modality_processor("nonexistent_modality")


# ---------------------------------------------------------------------------
# Stage 2: Fusion Models
# ---------------------------------------------------------------------------


class TestStage2ModelImports:
    """Import smoke tests for stage2_fusion model classes."""

    def test_import_full_area_unet(self):
        from stage2_fusion.models.full_area_unet import FullAreaUNet
        assert FullAreaUNet is not None

    def test_import_cone_batching_unet(self):
        from stage2_fusion.models.cone_batching_unet import ConeBatchingUNet
        assert ConeBatchingUNet is not None

    def test_import_cone_batching_unet_config(self):
        from stage2_fusion.models.cone_batching_unet import ConeBatchingUNetConfig
        assert ConeBatchingUNetConfig is not None


class TestStage2DataImports:
    """Import smoke tests for stage2_fusion data loading classes."""

    def test_import_cone_dataset(self):
        from stage2_fusion.data.cone_dataset import ConeDataset
        assert ConeDataset is not None

    def test_import_cone_graph_structure(self):
        from stage2_fusion.data.cone_dataset import ConeGraphStructure
        assert ConeGraphStructure is not None

    def test_import_hierarchical_cone_masking_system(self):
        from stage2_fusion.data.hierarchical_cone_masking import (
            HierarchicalConeMaskingSystem,
        )
        assert HierarchicalConeMaskingSystem is not None

    def test_import_lazy_cone_batcher(self):
        from stage2_fusion.data.hierarchical_cone_masking import LazyConeBatcher
        assert LazyConeBatcher is not None

    def test_import_hierarchical_cone(self):
        from stage2_fusion.data.hierarchical_cone_masking import HierarchicalCone
        assert HierarchicalCone is not None

    def test_import_multimodal_loader(self):
        from stage2_fusion.data.multimodal_loader import MultiModalLoader
        assert MultiModalLoader is not None

    def test_import_study_area_loader(self):
        from stage2_fusion.data.study_area_loader import StudyAreaLoader
        assert StudyAreaLoader is not None

    def test_import_feature_processing(self):
        from stage2_fusion.data.feature_processing import UrbanFeatureProcessor
        assert UrbanFeatureProcessor is not None


class TestStage2GeometryImports:
    """Import smoke tests for stage2_fusion geometry helpers."""

    def test_import_h3_geometry_helpers(self):
        from stage2_fusion.geometry.h3_geometry import (
            expected_children_count,
            validate_cone_size,
            log_geometric_summary,
        )
        assert callable(expected_children_count)
        assert callable(validate_cone_size)
        assert callable(log_geometric_summary)


# ---------------------------------------------------------------------------
# Stage 3: Analysis & Visualization
# ---------------------------------------------------------------------------


class TestStage3Imports:
    """Import smoke tests for stage3_analysis package."""

    def test_import_package_init(self):
        import stage3_analysis
        assert hasattr(stage3_analysis, "UrbanEmbeddingAnalyzer")
        assert hasattr(stage3_analysis, "HierarchicalClusterAnalyzer")
        assert hasattr(stage3_analysis, "HierarchicalLandscapeVisualizer")

    def test_import_urban_embedding_analyzer(self):
        from stage3_analysis.analytics import UrbanEmbeddingAnalyzer
        assert UrbanEmbeddingAnalyzer is not None

    def test_import_hierarchical_cluster_analyzer(self):
        from stage3_analysis.hierarchical_cluster_analysis import (
            HierarchicalClusterAnalyzer,
        )
        assert HierarchicalClusterAnalyzer is not None

    def test_import_hierarchical_landscape_visualizer(self):
        from stage3_analysis.hierarchical_visualization import (
            HierarchicalLandscapeVisualizer,
        )
        assert HierarchicalLandscapeVisualizer is not None


# ---------------------------------------------------------------------------
# H3 Compliance Tests
# ---------------------------------------------------------------------------

# Banned h3-py functions that MUST go through SRAI instead.
# See CLAUDE.md: "h3 is acceptable for parent-child hierarchy operations
# that SRAI does not wrap" -- everything else is banned.
BANNED_H3_PATTERNS = [
    r"h3\.grid_disk",
    r"h3\.grid_ring",
    r"h3\.cell_to_boundary",
    r"h3\.latlng_to_cell",
]

# Aliases used in some modules (import h3 as _h3)
BANNED_H3_ALIAS_PATTERNS = [
    r"_h3\.grid_disk",
    r"_h3\.grid_ring",
    r"_h3\.cell_to_boundary",
    r"_h3\.latlng_to_cell",
]

# Directories to scan (stage code only, not scripts/archive or tests)
STAGE_DIRS = [
    "stage1_modalities",
    "stage2_fusion",
    "stage3_analysis",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestH3Compliance:
    """Verify that stage code does not use banned h3-py functions.

    Per CLAUDE.md, h3-py is only allowed for parent-child hierarchy
    operations (cell_to_parent, cell_to_children, get_resolution, etc.).
    Tessellation and neighborhood queries must go through SRAI.
    """

    @pytest.mark.parametrize("pattern", BANNED_H3_PATTERNS + BANNED_H3_ALIAS_PATTERNS)
    def test_no_banned_h3_usage(self, pattern):
        """Grep stage directories for banned h3 function calls."""
        violations = []

        for stage_dir in STAGE_DIRS:
            search_path = PROJECT_ROOT / stage_dir
            if not search_path.exists():
                continue

            for py_file in search_path.rglob("*.py"):
                rel_path = py_file.relative_to(PROJECT_ROOT)
                with open(py_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        # Skip comments
                        stripped = line.lstrip()
                        if stripped.startswith("#"):
                            continue
                        # Check for the pattern using simple string matching
                        # (the pattern without the regex backslash)
                        search_str = pattern.replace(r"\.", ".")
                        if search_str in line:
                            violations.append(
                                f"{rel_path}:{line_num}: {line.rstrip()}"
                            )

        if violations:
            violation_report = "\n".join(violations)
            pytest.fail(
                f"Found banned h3 usage matching '{pattern}' in stage code "
                f"(should use SRAI instead):\n{violation_report}"
            )

    def test_no_h3_tessellation_import(self):
        """Ensure no stage code imports h3 tessellation functions directly."""
        banned_imports = [
            "from h3 import grid_disk",
            "from h3 import grid_ring",
            "from h3 import cell_to_boundary",
            "from h3 import latlng_to_cell",
        ]
        violations = []

        for stage_dir in STAGE_DIRS:
            search_path = PROJECT_ROOT / stage_dir
            if not search_path.exists():
                continue

            for py_file in search_path.rglob("*.py"):
                rel_path = py_file.relative_to(PROJECT_ROOT)
                with open(py_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        stripped = line.lstrip()
                        if stripped.startswith("#"):
                            continue
                        for banned in banned_imports:
                            if banned in line:
                                violations.append(
                                    f"{rel_path}:{line_num}: {line.rstrip()}"
                                )

        if violations:
            violation_report = "\n".join(violations)
            pytest.fail(
                f"Found direct imports of banned h3 functions:\n{violation_report}"
            )
