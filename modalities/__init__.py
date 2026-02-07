"""
UrbanRepML Modalities Package

Processors for different urban data modalities:
- AlphaEarth: Satellite imagery embeddings (primary)
- POI: Points of interest embeddings
- Roads: Road network embeddings
- GTFS: Public transit embeddings (planned)
- Aerial Imagery: High-res RGB images with DINOv3 encoding
"""

from modalities.base import ModalityProcessor


def get_available_modalities():
    """Get list of available modality processors."""
    return {
        'alphaearth': 'Satellite imagery embeddings from AlphaEarth',
        'aerial_imagery': 'High-resolution RGB images with DINOv3 encoding',
        'poi': 'Points of interest embeddings using Hex2Vec',
        'gtfs': 'Public transit accessibility embeddings',
        'roads': 'Road network topology embeddings',
    }


def load_modality_processor(modality, config=None):
    """Load a specific modality processor."""
    if config is None:
        config = {}

    if modality == 'alphaearth':
        from .alphaearth.processor import AlphaEarthProcessor
        return AlphaEarthProcessor(config)
    elif modality == 'aerial_imagery':
        from .aerial_imagery.processor import AerialImageryProcessor
        return AerialImageryProcessor(config)
    elif modality == 'poi':
        from .poi.processor import POIProcessor
        return POIProcessor(config)
    elif modality == 'roads':
        from .roads.processor import RoadsProcessor
        return RoadsProcessor(config)
    else:
        available = list(get_available_modalities().keys())
        raise ValueError(f"Unknown modality '{modality}'. Available: {available}")
