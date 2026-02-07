#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaEarth Satellite Imagery Modality

Processes AlphaEarth satellite embeddings (64-dimensional) from GeoTIFF files
into H3 hexagon-based representations for urban analysis.

This modality provides deep learned representations of satellite imagery
covering various spectral and spatial features for urban understanding.
"""

from .processor import AlphaEarthProcessor

__version__ = "1.0.0"
__author__ = "UrbanRepML Team"

__all__ = ["AlphaEarthProcessor"]