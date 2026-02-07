"""
Roads modality for UrbanRepML.

Processes OpenStreetMap road network data into H3 hexagon embeddings using SRAI.
"""

from .processor import RoadsProcessor

__all__ = ['RoadsProcessor']