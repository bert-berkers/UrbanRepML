"""
Urban Embedding Package
=======================

Multi-modal urban representation learning with clean modular architecture.

NOTE: Most imports temporarily disabled to avoid cascading import errors.
Import directly from submodules as needed.
"""

# Temporarily disabled to avoid cascading import errors
# Users should import directly from submodules, e.g.:
#   from stage2_fusion.models.cone_batching_unet import ConeBatchingUNet
#   from stage2_fusion.data.study_area_loader import StudyAreaLoader

__all__ = [
    # Pipeline
    'UrbanEmbeddingPipeline',

    # Models
    'ConeBatchingUNet', 'ConeBatchingUNetConfig',
    'FullAreaUNet',
    'AccessibilityUNet',

    # Data
    'StudyAreaLoader',
    'MultiModalLoader',
    'UrbanFeatureProcessor',
    'SpatialBatch',
    'StudyAreaFilter',

    # Training
    'UnifiedTrainer',
    'EmbeddingExtractor',
    'RenormalizingTrainer',
    'RenormalizingPipeline',

    # Graphs
    'SpatialGraphConstructor',
    'HexagonalLatticeConstructor',

    # Analysis — moved to stage3_analysis package
]