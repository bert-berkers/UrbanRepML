"""
Aerial Imagery Modality - PDOK RGB Images with DINOv3 Encoding

Fetches per-hexagon aerial RGB images from PDOK (Netherlands) and encodes
them with DINOv3 ViT-L/16 (satellite-pretrained, 1024D embeddings).
"""

from .processor import AerialImageryProcessor
from .pdok_client import PDOKClient
from .dinov3_encoder import DINOv3Encoder

__all__ = [
    'AerialImageryProcessor',
    'PDOKClient', 
    'DINOv3Encoder'
]