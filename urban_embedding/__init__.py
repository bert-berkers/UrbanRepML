"""
Urban Embedding Package
=======================

Multi-modal urban representation learning with clean modular architecture.

NOTE: Most imports temporarily disabled to avoid cascading import errors.
Import directly from submodules as needed.
"""

# Temporarily disabled to avoid cascading import errors
# Users should import directly from submodules, e.g.:
#   from urban_embedding.models.lattice_unet import LatticeUNet
#   from urban_embedding.data.study_area_loader import StudyAreaLoader

__all__ = [
    # Pipeline
    'UrbanEmbeddingPipeline',

    # Models
    'BaseUNet', 'BaseUNetConfig',
    'LatticeUNet', 'LatticeUNetConfig',
    'UrbanUNet',
    'HierarchicalSpatialUNet',
    'RenormalizingUNet',

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

    # Analysis
    'UrbanEmbeddingAnalyzer'
]