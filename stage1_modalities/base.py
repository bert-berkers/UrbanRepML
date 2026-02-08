"""
Base classes for modality processors in UrbanRepML.

All modality processors inherit from ModalityProcessor, which provides shared
infrastructure (config, name, save_embeddings). No abstract methods are enforced
because the three active processor types have fundamentally different interfaces:

- AlphaEarthProcessor (raster): process(raw_data_path, regions_gdf) -> GeoDataFrame
- POIProcessor (vector/SRAI): load_data(area_gdf) + process_to_h3(...) + run_pipeline(...)
- RoadsProcessor (vector/SRAI): load_data(area_gdf) + highway2vec(...) + run_pipeline(...)

The common contract is: each processor takes a config dict at init and ultimately
produces a DataFrame/GeoDataFrame indexed by region_id (or h3_index at the
stage1-stage2 boundary) with embedding columns.
"""

from abc import ABC
from pathlib import Path
from typing import Dict, Any
import pandas as pd


class ModalityProcessor(ABC):
    """Base class for all modality processors.

    Provides shared infrastructure: config storage, modality name derivation,
    and parquet output. Subclasses implement their own processing interface
    appropriate to their data type (raster, vector, etc.).

    All processors share this lifecycle:
        1. __init__(config) -- store configuration
        2. (processor-specific data loading and processing methods)
        3. Output: DataFrame/GeoDataFrame with region_id index + embedding columns
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize processor with configuration.

        Args:
            config: Dictionary of processor-specific settings.
        """
        self.config = config
        self.name = self.__class__.__name__.replace('Processor', '').lower()

    def save_embeddings(self, embeddings_df: pd.DataFrame, output_dir: str,
                       filename: str = None) -> str:
        """Save embeddings DataFrame to parquet file.

        Args:
            embeddings_df: Embeddings to save.
            output_dir: Directory to write the parquet file.
            filename: Optional filename override; defaults to {name}_embeddings.parquet.

        Returns:
            Absolute path to the written parquet file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{self.name}_embeddings.parquet"

        full_path = output_path / filename
        embeddings_df.to_parquet(full_path, index=False)

        return str(full_path)