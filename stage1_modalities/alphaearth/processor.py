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
# Using SRAI as primary interface (per CLAUDE.md)
# Note: h3 is a dependency of SRAI and needed for geometry conversions
import h3  # SRAI dependency - used for hex geometry
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
from pyproj import Transformer
from shapely.geometry import Polygon, Point, box
import rasterio

# Import base class
from ..  import ModalityProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def process_subtile(data: np.ndarray, transform, tile_hexagons: gpd.GeoDataFrame, min_pixels_per_hex: int,
                    transformer: Transformer) -> Dict:
    """Process a subtile: Coordinates -> CRS Transform -> Spatial Join with pre-defined hexagons -> Aggregation."""
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

    # Create point geometries for pixels
    from shapely.geometry import Point
    pixel_points = [Point(lon, lat) for lon, lat in zip(lons, lats)]

    # Flatten data array: (bands, pixels) -> (pixels, bands)
    data_flat = data.reshape(n_bands, -1).T.astype(np.float32)

    # Create GeoDataFrame for pixels
    pixels_gdf = gpd.GeoDataFrame(
        data_flat,
        geometry=pixel_points,
        crs='EPSG:4326'
    )

    # Filter out nodata (NaNs)
    pixels_gdf = pixels_gdf.dropna()

    if pixels_gdf.empty:
        return {}

    # Spatial join pixels with hexagons
    joined = gpd.sjoin(
        pixels_gdf,
        tile_hexagons[['geometry']],
        how='inner',
        predicate='within'
    )

    if joined.empty:
        return {}

    # Get h3_index from the join
    joined['h3_index'] = joined.index_right

    # Aggregate by H3 index (Mean and Count)
    embedding_cols = [col for col in joined.columns if isinstance(col, int)]
    grouped = joined.groupby('h3_index')[embedding_cols]

    # Calculate mean and count simultaneously
    aggregations = grouped.agg(['mean', 'count'])

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


def process_tile_worker(tiff_path: Path, tile_hexagons: gpd.GeoDataFrame, subtile_size: int, min_pixels_per_hex: int,
                        intermediate_dir: Path):
    """Worker function to process a single TIFF tile with pre-defined hexagons."""
    tile_name = tiff_path.name
    tile_stem = tiff_path.stem

    # Checkpoint: Check if intermediate embeddings stage1_modalities result exists
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

                    # Process subtile with pre-defined hexagons
                    subtile_results = process_subtile(
                        subtile_data, subtile_transform, tile_hexagons, min_pixels_per_hex, transformer
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

            # Save intermediate embeddings stage1_modalities result
            with open(intermediate_file, 'w') as f:
                json.dump(tile_results, f)

            return True

    except Exception as e:
        logger.error(f"Error processing_modalities tile {tile_name}: {e}", exc_info=True)
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
    
    def process(self, raw_data_path: Path, regions_gdf: gpd.GeoDataFrame = None, **kwargs) -> gpd.GeoDataFrame:
        """Process AlphaEarth TIFFs to H3 hexagons using pre-defined study area regions."""
        year_filter = kwargs.get('year_filter', '2021')
        h3_resolution = kwargs.get('h3_resolution', 8)

        # Get TIFF files
        tiff_files = self.get_tiff_files(raw_data_path, year_filter)

        if not tiff_files:
            raise ValueError(f"No TIFF files found in {raw_data_path} with year filter {year_filter}")

        # Check if intermediate_dir was passed as a parameter
        if 'intermediate_dir' in self.config and self.config['intermediate_dir']:
            # Use the provided intermediate directory
            intermediate_dir = Path(self.config['intermediate_dir'])
            intermediate_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Create intermediate embeddings stage1_modalities directory in new structure
            intermediate_base = Path('data/study_areas/default/embeddings/intermediate/alphaearth')
            intermediate_base.mkdir(parents=True, exist_ok=True)

            # Create study-specific intermediate embeddings stage1_modalities directory
            study_name = raw_data_path.parent.name  # e.g., 'cascadia_2021'
            intermediate_dir = intermediate_base / study_name
            intermediate_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing {len(tiff_files)} TIFF files with {self.max_workers} workers")

        # If regions_gdf provided, use it for all tiles
        if regions_gdf is not None:
            # Ensure h3_index column exists
            if 'h3_index' not in regions_gdf.columns:
                regions_gdf = regions_gdf.reset_index().rename(columns={'index': 'h3_index'})
            regions_gdf = regions_gdf.set_index('h3_index')

            # Process each tile with its intersecting hexagons
            from shapely.geometry import box
            import rasterio

            results = []
            for tiff_path in tiff_files:
                # Get tile bounds
                with rasterio.open(tiff_path) as src:
                    bounds = src.bounds
                    crs = src.crs

                # Transform bounds to WGS84 if needed
                if crs.to_epsg() != 4326:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                    lons, lats = transformer.transform(
                        [bounds.left, bounds.right, bounds.left, bounds.right],
                        [bounds.bottom, bounds.bottom, bounds.top, bounds.top]
                    )
                    wgs84_bounds = (min(lons), min(lats), max(lons), max(lats))
                else:
                    wgs84_bounds = (bounds.left, bounds.bottom, bounds.right, bounds.top)

                # Get hexagons that intersect with tile
                bbox = box(*wgs84_bounds)
                tile_hexagons = regions_gdf[regions_gdf.intersects(bbox)].copy()

                if len(tile_hexagons) > 0:
                    # Process this tile with its hexagons
                    success = process_tile_worker(
                        tiff_path,
                        tile_hexagons,
                        self.subtile_size,
                        self.min_pixels_per_hex,
                        intermediate_dir
                    )
                    if success:
                        results.append(tiff_path)
                else:
                    logger.warning(f"No hexagons found for tile {tiff_path.name}")
        else:
            # Fall back to original processing without pre-defined regions
            logger.warning("No regions_gdf provided, cannot process without study area regions")
            raise ValueError("regions_gdf is required for processing AlphaEarth data")

        # Merge all intermediate embeddings stage1_modalities results
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
        """Merge intermediate embeddings stage1_modalities JSON results into a GeoDataFrame."""
        logger.info("Merging intermediate embeddings stage1_modalities results...")
        
        intermediate_files = list(intermediate_dir.glob("*.json"))
        if not intermediate_files:
            raise ValueError("No intermediate embeddings stage1_modalities files found for merging")
        
        logger.info(f"Loading and combining {len(intermediate_files)} intermediate embeddings stage1_modalities files...")
        
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
        """Convert to H3 format - already in H3 format from processing_modalities."""
        # AlphaEarth processor already outputs in H3 format
        return gdf.drop('geometry', axis=1)
    
    def create_embeddings(self, h3_data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Extract embeddings from AlphaEarth data."""
        # Extract embedding columns (A00-A63)
        embedding_cols = [col for col in h3_data.columns if col.startswith('A') and col[1:].isdigit()]
        return h3_data[embedding_cols].values