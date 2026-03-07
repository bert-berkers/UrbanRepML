"""
Urban Embedding Models
======================

Models for urban spatial representation learning.

Models:
- FullAreaUNet: Full study area U-Net with lateral accessibility graph (OG that worked)
- ConeBatchingUNet: Cone-based hierarchical U-Net (most promising future direction)
- SimpleRingAggregator: K-ring spatial averaging (non-parametric baseline)
- LatticeGCN: GCN encoder-decoder on H3 hexagonal lattice (self-supervised)
"""

from .full_area_unet import FullAreaUNet
from .cone_batching_unet import ConeBatchingUNet, ConeBatchingUNetConfig
from .ring_aggregation import SimpleRingAggregator
from .lattice_gcn import LatticeGCN

__all__ = [
    'FullAreaUNet',
    'ConeBatchingUNet',
    'ConeBatchingUNetConfig',
    'SimpleRingAggregator',
    'LatticeGCN',
]
