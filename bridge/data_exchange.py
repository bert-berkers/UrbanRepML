"""
Data exchange utilities for UrbanRepML <-> GEO-INFER integration.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Union
import h3


def urbanreml_to_geoinfer(
    data_path: Union[str, Path],
    data_type: str = "h3_embeddings",
    resolution: int = 8
) -> Dict:
    """
    Convert UrbanRepML data format to GEO-INFER compatible format.
    
    Args:
        data_path: Path to UrbanRepML data (parquet file)
        data_type: Type of data ('h3_embeddings', 'alphaearth', 'density')
        resolution: H3 resolution level
    
    Returns:
        Dictionary with GEO-INFER formatted data
    """
    df = pd.read_parquet(data_path)
    
    # Handle H3 index extraction
    h3_index = []
    if df.index.name and 'h3' in str(df.index.name):
        h3_index = df.index.tolist()
    elif 'h3' in df.columns:
        h3_index = df['h3'].tolist()
    elif 'hex_id' in df.columns:
        h3_index = df['hex_id'].tolist()
    else:
        # Try to use index if it looks like H3
        h3_index = df.index.tolist()
    
    geoinfer_data = {
        "metadata": {
            "source": "UrbanRepML",
            "type": data_type,
            "h3_resolution": resolution,
            "crs": "EPSG:4326"
        },
        "data": df.to_dict('records'),
        "h3_index": h3_index
    }
    
    return geoinfer_data


def geoinfer_to_urbanreml(
    geoinfer_data: Dict,
    target_format: str = "parquet"
) -> pd.DataFrame:
    """
    Convert GEO-INFER data to UrbanRepML compatible format.
    
    Args:
        geoinfer_data: GEO-INFER formatted data dictionary
        target_format: Output format ('parquet', 'geodataframe')
    
    Returns:
        DataFrame or GeoDataFrame in UrbanRepML format
    """
    df = pd.DataFrame(geoinfer_data['data'])
    
    if 'h3_index' in geoinfer_data:
        df['h3'] = geoinfer_data['h3_index']
        df.set_index('h3', inplace=True)
    
    if target_format == "geodataframe":
        # Convert H3 to geometry
        df['geometry'] = df.index.map(lambda x: h3.h3_to_geo_boundary(x, geo_json=True))
        return gpd.GeoDataFrame(df, crs="EPSG:4326")
    
    return df


def h3_data_bridge(
    urbanreml_regions: pd.DataFrame,
    geoinfer_modules: List[str],
    resolution: int = 8
) -> Dict:
    """
    Bridge H3 regional data between UrbanRepML and GEO-INFER modules.
    
    Args:
        urbanreml_regions: DataFrame with H3 regions from UrbanRepML
        geoinfer_modules: List of GEO-INFER modules to integrate
        resolution: H3 resolution level
    
    Returns:
        Dictionary mapping modules to processed data
    """
    bridged_data = {}
    
    for module in geoinfer_modules:
        if module == "agricultural_analysis":
            # Prepare data for GEO-INFER agricultural module
            bridged_data[module] = {
                "h3_cells": urbanreml_regions.index.tolist(),
                "resolution": resolution,
                "features": urbanreml_regions.select_dtypes(include='number').to_dict('list')
            }
        elif module == "climate_impact":
            # Prepare for climate analysis
            bridged_data[module] = {
                "spatial_index": urbanreml_regions.index,
                "temporal_range": None,  # To be filled from metadata
                "variables": list(urbanreml_regions.columns)
            }
    
    return bridged_data