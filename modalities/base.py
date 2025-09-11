"""
Base classes for modality processors in UrbanRepML.

All modality processors should inherit from ModalityProcessor and implement
the required abstract methods.
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
    def validate_config(self):
        """Validate configuration parameters."""
        pass
    
    @abstractmethod
    def load_data(self, study_area: str) -> gpd.GeoDataFrame:
        """Load raw data for the given study area."""
        pass
    
    @abstractmethod
    def process_to_h3(self, data: gpd.GeoDataFrame, h3_resolution: int) -> pd.DataFrame:
        """Process data into H3 hexagon format with embeddings."""
        pass
    
    @abstractmethod
    def run_pipeline(self, study_area: str, h3_resolution: int, 
                    output_dir: str) -> str:
        """Run complete processing embeddings pipeline."""
        pass
    
    def save_embeddings(self, embeddings_df: pd.DataFrame, output_dir: str, 
                       filename: str = None) -> str:
        """Save embeddings DataFrame to parquet file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"{self.name}_embeddings.parquet"
        
        full_path = output_path / filename
        embeddings_df.to_parquet(full_path, index=False)
        
        return str(full_path)