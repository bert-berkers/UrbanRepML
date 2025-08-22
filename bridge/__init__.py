"""
Bridge module for integrating UrbanRepML with GEO-INFER framework.

This module provides seamless data exchange and method integration between:
- UrbanRepML: Urban embedding and multi-resolution analysis
- GEO-INFER: Active Inference geospatial framework
"""

from .data_exchange import (
    urbanreml_to_geoinfer,
    geoinfer_to_urbanreml,
    h3_data_bridge
)

from .model_integration import (
    combine_embeddings,
    active_inference_wrapper,
    multi_resolution_adapter
)

__all__ = [
    'urbanreml_to_geoinfer',
    'geoinfer_to_urbanreml', 
    'h3_data_bridge',
    'combine_embeddings',
    'active_inference_wrapper',
    'multi_resolution_adapter'
]