"""
Urban Embedding Models
======================

UNet architectures for urban spatial representation learning.

Models:
- FullAreaUNet: Full study area U-Net with lateral accessibility graph (OG that worked)
- ConeBatchingUNet: Cone-based hierarchical U-Net (most promising future direction)
- AccessibilityUNet: Accessibility-weighted U-Net (planned)
"""

from .full_area_unet import FullAreaUNet
from .cone_batching_unet import ConeBatchingUNet, ConeBatchingUNetConfig
from .accessibility_unet import AccessibilityUNet

__all__ = [
    'FullAreaUNet',
    'ConeBatchingUNet',
    'ConeBatchingUNetConfig',
    'AccessibilityUNet',
]
