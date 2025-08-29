#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UrbanRepML Modalities Package

This package contains processors for different urban data modalities:
- AlphaEarth: Satellite imagery embeddings
- Aerial Imagery: High-res RGB images with DINOv3 encoding
- POI: Points of interest embeddings  
- GTFS: Public transit embeddings
- Roads: Road network embeddings
- Buildings: Building footprint embeddings
- StreetView: Street-level imagery embeddings

Each modality follows a standardized interface for processing raw data
into H3 hexagon-based embeddings.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import geopandas as gpd
import pandas as pd
import numpy as np


class ModalityProcessor(ABC):
    """Abstract base class for all modality processors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize processor with configuration."""
        self.config = config
        self.name = self.__class__.__name__.replace('Processor', '').lower()
    
    @abstractmethod
    def download(self, study_area: str, **kwargs) -> Path:
        """Download raw data for the given study area."""
        pass
    
    @abstractmethod
    def process(self, raw_data_path: Path, **kwargs) -> gpd.GeoDataFrame:
        """Process raw data into structured format."""
        pass
    
    @abstractmethod
    def to_h3(self, gdf: gpd.GeoDataFrame, resolution: int, **kwargs) -> pd.DataFrame:
        """Convert processed data to H3 hexagon format."""
        pass
    
    @abstractmethod
    def create_embeddings(self, h3_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Create embeddings from H3 data."""
        pass
    
    def run_pipeline(self, study_area: str, h3_resolution: int, 
                    output_dir: Path, force_download: bool = False) -> Path:
        """Run complete processing pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already processed
        output_file = output_dir / f"{self.name}_{study_area}_res{h3_resolution}.parquet"
        if output_file.exists() and not force_download:
            print(f"Using existing embeddings: {output_file}")
            return output_file
        
        print(f"Processing {self.name} modality for {study_area}")
        
        # Download raw data
        raw_data_path = self.download(study_area)
        
        # Process to structured format
        gdf = self.process(raw_data_path)
        
        # Convert to H3
        h3_df = self.to_h3(gdf, h3_resolution)
        
        # Create embeddings
        embeddings = self.create_embeddings(h3_df)
        
        # Add embeddings to dataframe
        embedding_cols = [f"emb_{i:03d}" for i in range(embeddings.shape[1])]
        for i, col in enumerate(embedding_cols):
            h3_df[col] = embeddings[:, i]
        
        # Save result
        h3_df.to_parquet(output_file, index=False)
        print(f"Saved {self.name} embeddings to: {output_file}")
        
        return output_file


def get_available_modalities() -> Dict[str, str]:
    """Get list of available modality processors."""
    return {
        'alphaearth': 'Satellite imagery embeddings from AlphaEarth',
        'aerial_imagery': 'High-resolution RGB images with DINOv3 encoding',
        'poi': 'Points of interest embeddings using Hex2Vec',
        'gtfs': 'Public transit accessibility embeddings',
        'roads': 'Road network topology embeddings', 
        'buildings': 'Building footprint density embeddings',
        'streetview': 'Street-level imagery embeddings'
    }


def load_modality_processor(modality: str, config: Optional[Dict] = None):
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
    elif modality == 'gtfs':
        from .gtfs.processor import GTFSProcessor
        return GTFSProcessor(config)
    elif modality == 'roads':
        from .roads.processor import RoadProcessor
        return RoadProcessor(config)
    elif modality == 'buildings':
        from .buildings.processor import BuildingProcessor
        return BuildingProcessor(config)
    elif modality == 'streetview':
        from .streetview.processor import StreetViewProcessor
        return StreetViewProcessor(config)
    else:
        available = list(get_available_modalities().keys())
        raise ValueError(f"Unknown modality '{modality}'. Available: {available}")