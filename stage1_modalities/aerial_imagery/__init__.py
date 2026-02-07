"""
Aerial Imagery Modality - PDOK RGB Images with DINOv3 Encoding

This modality fetches high-resolution aerial RGB images from PDOK (Netherlands)
and encodes them using DINOv3 (especially the remote sensing variant).
The embeddings are hierarchically aggregated to H3 hexagons.
"""

from .processor import AerialImageryProcessor
from .pdok_client import PDOKClient
from .dinov3_encoder import DINOv3Encoder

__all__ = [
    'AerialImageryProcessor',
    'PDOKClient', 
    'DINOv3Encoder'
]