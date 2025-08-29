#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AlphaEarth Satellite Imagery Modality Processor

Processes AlphaEarth GeoTIFF files (64-band embeddings) into H3 hexagon representations.
Implements the standardized ModalityProcessor interface.
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from rasterio.windows import Window
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import os
import concurrent.futures
from tqdm.auto import tqdm

# H3 and Projection imports
import h3
from pyproj import Transformer
from shapely.geometry import Polygon

# Import base class
from ..  import ModalityProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def process_subtile(data: np.ndarray, transform, h3_resolution: int, min_pixels_per_hex: int,
                    transformer: Transformer) -> Dict:
    """Process a subtile: Coordinates -> CRS Transform -> H3 Indexing -> Aggregation."""
    n_bands, height, width = data.shape

    if height == 0 or width == 0 or np.isnan(data).all():
        return {}

    # Create coordinate grids for all pixels (using pixel centers: +0.5)
    rows, cols = np.meshgrid(
        np.arange(height) + 0.5,
        np.arange(width) + 0.5,
        indexing='ij'
    )

    # Transform to geographic coordinates
    xs, ys = transform * (cols.flatten(), rows.flatten())

    # Transform coordinates to WGS84 (EPSG:4326)
    lons, lats = transformer.transform(xs, ys)

    # Get H3 index for each pixel
    h3_indices = [
        h3.latlng_to_cell(lat, lon, h3_resolution)
        for lat, lon in zip(lats, lons)
    ]

    # Flatten data array: (bands, pixels) -> (pixels, bands)
    data_flat = data.reshape(n_bands, -1).T.astype(np.float32)

    # Create DataFrame
    df = pd.DataFrame(data_flat)
    df['h3_index'] = h3_indices

    # Filter out nodata (NaNs) and invalid H3 indices
    df = df.dropna()
    df = df[df['h3_index'].notna()]

    if df.empty:
        return {}

    # Aggregate by H3 index (Mean and Count)
    grouped = df.groupby('h3_index')
    embedding_cols = [col for col in df.columns if col != 'h3_index']

    # Calculate mean and count simultaneously
    aggregations = grouped[embedding_cols].agg(['mean', 'count'])

    # Format results
    result = {}
    for h3_index, row in aggregations.iterrows():
        # Check minimum pixel count
        pixel_count = row[(0, 'count')]

        if pixel_count >= min_pixels_per_hex:
            # Extract mean values for all bands
            mean_values = row.xs('mean', level=1).values

            result[h3_index] = {
                'embedding': mean_values.tolist(),
                'pixel_count': int(pixel_count)
            }

    return result


def process_tile_worker(tiff_path: Path, h3_resolution: int, subtile_size: int, min_pixels_per_hex: int,
                        intermediate_dir: Path):
    """Worker function to process a single TIFF tile."""
    tile_name = tiff_path.name
    tile_stem = tiff_path.stem

    # Checkpoint: Check if intermediate result exists
    intermediate_file = intermediate_dir / f"{tile_stem}.json"
    if intermediate_file.exists():
        return True

    tile_results = {}

    try:
        # Open raster with masked=True to handle nodata as NaN
        with rioxarray.open_rasterio(tiff_path, masked=True) as ds:
            crs = ds.rio.crs
            n_bands, height, width = ds.shape

            if crs is None:
                logger.error(f"Tile {tile_name} has no CRS defined. Cannot process.")
                return False

            # Initialize transformer
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

            # Process in subtiles
            for row in range(0, height, subtile_size):
                for col in range(0, width, subtile_size):
                    # Define the window for reading
                    y_slice = slice(row, min(row + subtile_size, height))
                    x_slice = slice(col, min(col + subtile_size, width))
                    subtile_ds = ds.isel(y=y_slice, x=x_slice)
                    subtile_data = subtile_ds.values

                    # Get the affine transform for the subtile
                    subtile_transform = subtile_ds.rio.transform()

                    # Process subtile
                    subtile_results = process_subtile(
                        subtile_data, subtile_transform, h3_resolution, min_pixels_per_hex, transformer
                    )

                    # Merge results (Handle overlaps between subtiles)
                    for h3_index, values in subtile_results.items():
                        if h3_index not in tile_results:
                            tile_results[h3_index] = values
                        else:
                            # Weighted average
                            existing = tile_results[h3_index]
                            total_count = existing['pixel_count'] + values['pixel_count']

                            # Calculate weighted average
                            existing_array = np.array(existing['embedding'])
                            new_array = np.array(values['embedding'])

                            weighted_avg = (
                                existing_array * existing['pixel_count'] +
                                new_array * values['pixel_count']
                            ) / total_count

                            tile_results[h3_index] = {
                                'embedding': weighted_avg.tolist(),
                                'pixel_count': total_count
                            }

            # Save intermediate result
            with open(intermediate_file, 'w') as f:
                json.dump(tile_results, f)

            return True

    except Exception as e:
        logger.error(f"Error processing tile {tile_name}: {e}", exc_info=True)
        if intermediate_file.exists():
            intermediate_file.unlink()
        return False


class AlphaEarthProcessor(ModalityProcessor):
    """AlphaEarth satellite imagery processor following modality interface."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.subtile_size = config.get('subtile_size', 512)
        self.min_pixels_per_hex = config.get('min_pixels_per_hex', 1)
        self.max_workers = config.get('max_workers', max(1, os.cpu_count() - 2))
        
    def download(self, study_area: str, **kwargs) -> Path:
        """Download or locate AlphaEarth TIFF files for study area."""
        # For AlphaEarth, we assume files are already downloaded
        source_dir = self.config.get('source_dir')
        if not source_dir:
            raise ValueError("AlphaEarth source_dir must be specified in config")
            
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"AlphaEarth source directory not found: {source_path}")
            
        return source_path
    
    def process(self, raw_data_path: Path, **kwargs) -> gpd.GeoDataFrame:
        """Process AlphaEarth TIFFs to H3 hexagons."""
        year_filter = kwargs.get('year_filter', '2021')
        h3_resolution = kwargs.get('h3_resolution', 8)
        
        # Get TIFF files
        tiff_files = self.get_tiff_files(raw_data_path, year_filter)
        
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {raw_data_path} with year filter {year_filter}")
        
        # Create intermediate directory
        intermediate_dir = raw_data_path / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing {len(tiff_files)} TIFF files with {self.max_workers} workers")
        
        # Process tiles in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tile = {
                executor.submit(
                    process_tile_worker,
                    tiff_path,
                    h3_resolution,
                    self.subtile_size,
                    self.min_pixels_per_hex,
                    intermediate_dir
                ): tiff_path for tiff_path in tiff_files
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_tile), 
                             total=len(tiff_files), desc="Processing Tiles"):
                try:
                    success = future.result()
                    if not success:
                        tiff_path = future_to_tile[future]
                        logger.warning(f"Processing failed for: {tiff_path.name}")
                except Exception as exc:
                    tiff_path = future_to_tile[future]
                    logger.error(f'{tiff_path.name} generated an exception: {exc}')

        # Merge all intermediate results
        return self.merge_intermediate_results(intermediate_dir)
    
    def get_tiff_files(self, source_dir: Path, year_filter: Optional[str] = None) -> List[Path]:
        """Get list of TIFF files, optionally filtered by year."""
        if year_filter:
            patterns = [f'*{year_filter}*.tif', f'*{year_filter}*.tiff']
        else:
            patterns = ['*.tif', '*.tiff']
            
        all_tiff_files = []
        for pattern in patterns:
            all_tiff_files.extend(source_dir.glob(pattern))
        
        return sorted(all_tiff_files)
    
    def merge_intermediate_results(self, intermediate_dir: Path) -> gpd.GeoDataFrame:
        """Merge intermediate JSON results into a GeoDataFrame."""
        logger.info("Merging intermediate results...")
        
        intermediate_files = list(intermediate_dir.glob("*.json"))
        if not intermediate_files:
            raise ValueError("No intermediate files found for merging")
        
        logger.info(f"Loading and combining {len(intermediate_files)} intermediate files...")
        
        # Combine all results with overlap handling
        merged = {}
        
        for intermediate_file in tqdm(intermediate_files, desc="Loading Intermediate Files"):
            try:
                with open(intermediate_file, 'r') as f:
                    tile_results = json.load(f)
            except Exception as e:
                logger.error(f"Error loading {intermediate_file}: {e}. Skipping.")
                continue
            
            for h3_index_str, values in tile_results.items():
                if h3_index_str not in merged:
                    merged[h3_index_str] = {
                        'embeddings': [],
                        'pixel_counts': []
                    }
                
                merged[h3_index_str]['embeddings'].append(np.array(values['embedding'], dtype=np.float32))
                merged[h3_index_str]['pixel_counts'].append(values['pixel_count'])
        
        logger.info(f"Calculating final weighted averages for {len(merged)} unique hexagons...")
        
        # Average overlapping hexagons
        final_data = []
        
        for h3_index_str, data in tqdm(merged.items(), desc="Calculating Weighted Averages"):
            embeddings = data['embeddings']
            pixel_counts = np.array(data['pixel_counts'])
            
            if len(embeddings) > 1:
                # Multiple tiles contributed: calculate weighted average
                weights = pixel_counts / pixel_counts.sum()
                final_embedding = np.average(embeddings, weights=weights, axis=0)
            else:
                final_embedding = embeddings[0]
            
            # Create record
            record = {'h3_index': h3_index_str}
            
            # Add embedding dimensions
            for i, value in enumerate(final_embedding):
                record[f'A{i:02d}'] = value
            
            record['pixel_count'] = pixel_counts.sum()
            record['tile_count'] = len(embeddings)
            
            # Add H3 geometry
            boundary = h3.cell_to_boundary(h3_index_str)
            boundary_lonlat = [(lng, lat) for lat, lng in boundary]
            record['geometry'] = Polygon(boundary_lonlat)
            
            final_data.append(record)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(final_data, geometry='geometry', crs="EPSG:4326")
        logger.info(f"Final dataset generated: {len(gdf)} hexagons")
        
        return gdf
    
    def to_h3(self, gdf: gpd.GeoDataFrame, resolution: int, **kwargs) -> pd.DataFrame:
        """Convert to H3 format - already in H3 format from processing."""
        # AlphaEarth processor already outputs in H3 format
        return gdf.drop('geometry', axis=1)
    
    def create_embeddings(self, h3_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Extract embeddings from AlphaEarth data."""
        # Extract embedding columns (A00-A63)
        embedding_cols = [col for col in h3_data.columns if col.startswith('A') and col[1:].isdigit()]
        return h3_data[embedding_cols].values