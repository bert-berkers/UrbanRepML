"""
Points of Interest (POI) modality for UrbanRepML.

Processes OpenStreetMap POI data into H3 hexagon embeddings using SRAI.
"""

from .processor import POIProcessor

__all__ = ['POIProcessor']