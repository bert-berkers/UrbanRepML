"""
Urban Embedding Models
======================

UNet architectures for urban spatial representation learning.

Models:
- BaseUNet: Abstract base class for all U-Net variants
- AccessibilityUNet: Accessibility-weighted U-Net (planned)
- UrbanUNet: Full study area U-Net with lateral accessibility graph (OG that worked)
- ConeLatticeUNet: Cone-based hierarchical U-Net (most promising future direction)
"""

from .base import BaseUNet, BaseUNetConfig

__all__ = [
    'BaseUNet',
    'BaseUNetConfig',
]
