"""
GTFS (General Transit Feed Specification) modality for UrbanRepML.

Processes public transit data into H3 hexagon embeddings using SRAI's GTFS2VecEmbedder.
"""

from .processor import GTFSProcessor

__all__ = ['GTFSProcessor']
