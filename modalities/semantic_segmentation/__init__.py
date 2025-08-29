"""
Semantic Segmentation Modality - AlphaEarth + DINOv3 Fusion

This modality combines:
1. AlphaEarth embeddings (from Google Earth Engine) as conditioning
2. DINOv3 features (from PDOK aerial imagery) for segmentation
3. Attentional U-Net for hierarchical feature fusion
4. Categorical outputs for semantic land use/land cover classes

The key innovation is using AlphaEarth embeddings to condition the 
segmentation network, improving DINOv3's performance through global context.
"""

from .processor import SemanticSegmentationProcessor
from .fusion_network import AlphaEarthConditionedUNet
from .segmentation_classes import SegmentationClasses

__all__ = [
    'SemanticSegmentationProcessor',
    'AlphaEarthConditionedUNet',
    'SegmentationClasses'
]