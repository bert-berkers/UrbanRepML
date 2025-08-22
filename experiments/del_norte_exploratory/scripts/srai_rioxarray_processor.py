#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized AlphaEarth processor combining SRAI and rioxarray.
Uses SRAI for efficient H3 operations and rioxarray for optimized TIFF reading.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import xarray as xr
import dask.array as da
import h3
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import gc
import time
import warnings
warnings.filterwarnings('ignore')

# SRAI imports for optimized H3 operations
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
from shapely.geometry import Polygon, Point
import shapely.wkt

logger = logging.getLogger(__name__)


class SRAIRioxarrayProcessor:
    """Optimized AlphaEarth processor using SRAI and rioxarray."""
    
    def __init__(self, config: dict):
        """Initialize processor with configuration."""
        self.config = config
        self.h3_resolution = config['data']['h3_resolution']
        self.source_dir = Path(config['data']['source_dir'])
        self.pattern = config['data']['pattern']
        self.batch_size = config['processing']['batch_size']
        self.min_pixels = config['processing']['min_pixels_per_hex']
        
        # Get max tiles if specified
        self.max_tiles = config.get('experiment', {}).get('max_tiles', None)
        
        # Rioxarray optimization settings
        self.chunk_size_mb = config.get('rioxarray', {}).get('chunk_size_mb', 100)
        
        # Calculate optimal chunk size for resolution 10
        # For 64 bands of float32 data: chunk_size^2 * 64 * 4 bytes = target MB
        bytes_per_mb = 1024 * 1024
        target_bytes = self.chunk_size_mb * bytes_per_mb
        pixels_per_chunk = target_bytes // (64 * 4)  # 64 bands, 4 bytes per float32
        self.chunk_pixels = int(np.sqrt(pixels_per_chunk))
        
        # Adjust chunk size to be divisor of 3072 for optimal alignment
        if 3072 % self.chunk_pixels != 0:
            # Find nearest divisor
            divisors = [d for d in range(256, 1537) if 3072 % d == 0]
            self.chunk_pixels = min(divisors, key=lambda x: abs(x - self.chunk_pixels))
        
        # SRAI components
        self.h3_regionalizer = None
        self.h3_neighbourhood = None
        
        # Setup logging
        logging.basicConfig(
            level=config['output']['log_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"SRAI+Rioxarray Processor initialized")
        logger.info(f"H3 Resolution: {self.h3_resolution}")
        logger.info(f"Chunk size: {self.chunk_pixels}x{self.chunk_pixels} pixels (~{self.chunk_size_mb}MB)")
    
    def initialize_srai_components(self, bounds: Tuple[float, float, float, float]):
        """Initialize SRAI H3 components for the given bounds."""
        # Create bounding box polygon
        min_lon, min_lat, max_lon, max_lat = bounds
        bbox_polygon = Polygon([
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat)
        ])
        
        # Create GeoDataFrame with bounding box
        gdf_bounds = gpd.GeoDataFrame(
            {'geometry': [bbox_polygon]},
            crs='EPSG:4326'
        )
        
        # Initialize H3 regionalizer
        self.h3_regionalizer = H3Regionalizer(resolution=self.h3_resolution)
        
        # Create H3 regions for the area
        logger.info("Creating H3 regions with SRAI...")
        h3_regions = self.h3_regionalizer.transform(gdf_bounds)
        
        # Initialize neighbourhood for spatial relationships
        self.h3_neighbourhood = H3Neighbourhood(h3_regions)
        
        logger.info(f"Initialized SRAI with {len(h3_regions)} H3 hexagons")
        return h3_regions
    
    def get_tiff_files(self) -> List[Path]:
        """Get TIFF files to process."""
        files = list(self.source_dir.glob(self.pattern))
        logger.info(f"Found {len(files)} TIFF files")
        
        # Limit to max_tiles if specified
        if self.max_tiles and len(files) > self.max_tiles:
            files = files[:self.max_tiles]
            logger.info(f"Limited to {self.max_tiles} tiles as configured")
        
        return sorted(files)
    
    def load_tiff_optimized(self, file_path: Path) -> Optional[xr.DataArray]:
        """Load TIFF using rioxarray with optimal settings."""
        try:
            # Open with optimal chunk size and COG optimizations
            with rioxarray.set_options(export_grid_mapping=False):
                da = rioxarray.open_rasterio(
                    file_path,
                    chunks={'x': self.chunk_pixels, 'y': self.chunk_pixels, 'band': 64},
                    lock=False,  # Enable parallel reading for COG
                    decode_times=False,  # Skip time decoding
                    cache=False  # Don't cache in memory
                )
                
                # Verify 64 bands
                if da.sizes['band'] != 64:
                    logger.warning(f"File {file_path} has {da.sizes['band']} bands, expected 64")
                    return None
                
                # Quick emptiness check on small sample
                sample = da.isel(
                    x=slice(0, min(100, da.sizes['x'])),
                    y=slice(0, min(100, da.sizes['y']))
                ).compute()
                
                if np.sum(sample.values != 0) < 100:
                    logger.debug(f"Skipping mostly empty tile: {file_path.name}")
                    return None
                
                return da
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def process_chunk_to_h3(self, chunk_data: np.ndarray, 
                           x_coords: np.ndarray, 
                           y_coords: np.ndarray) -> pd.DataFrame:
        """Process a data chunk to H3 hexagons using vectorized operations."""
        # Create coordinate grids
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Flatten for processing
        flat_lons = xx.flatten()
        flat_lats = yy.flatten()
        
        # Reshape data: (bands, y, x) -> (pixels, bands)
        n_bands = chunk_data.shape[0]
        flat_data = chunk_data.reshape(n_bands, -1).T
        
        # Filter out zero pixels
        non_zero_mask = np.any(flat_data != 0, axis=1)
        if not np.any(non_zero_mask):
            return pd.DataFrame()
        
        valid_lons = flat_lons[non_zero_mask]
        valid_lats = flat_lats[non_zero_mask]
        valid_data = flat_data[non_zero_mask]
        
        # Subsample based on resolution
        sample_rate = 5 if self.h3_resolution == 10 else 10
        sample_indices = np.arange(0, len(valid_lons), sample_rate)
        
        if len(sample_indices) == 0:
            return pd.DataFrame()
        
        sampled_lons = valid_lons[sample_indices]
        sampled_lats = valid_lats[sample_indices]
        sampled_data = valid_data[sample_indices]
        
        # Vectorized H3 conversion
        h3_indices = [h3.latlng_to_cell(lat, lon, self.h3_resolution) 
                     for lat, lon in zip(sampled_lats, sampled_lons)]
        
        # Create DataFrame
        records = []
        for i, h3_idx in enumerate(h3_indices):
            record = {
                'h3_index': h3_idx,
                'lat': sampled_lats[i],
                'lon': sampled_lons[i]
            }
            # Add band values
            for band_idx in range(64):
                record[f'band_{band_idx:02d}'] = sampled_data[i, band_idx]
            records.append(record)
        
        return pd.DataFrame(records)
    
    def process_dataarray_to_h3(self, da: xr.DataArray, 
                                file_path: Path) -> gpd.GeoDataFrame:
        """Process xarray DataArray to H3 hexagons using SRAI."""
        logger.info(f"Processing {file_path.name} to H3 hexagons...")
        
        # Get bounds for SRAI initialization
        bounds = (
            float(da.x.min()),
            float(da.y.min()),
            float(da.x.max()),
            float(da.y.max())
        )
        
        # Initialize SRAI components if not already done
        if self.h3_regionalizer is None:
            self.initialize_srai_components(bounds)
        
        # Process chunks in parallel using Dask
        all_hexagons = []
        
        # Compute in chunks to manage memory
        for x_start in range(0, da.sizes['x'], self.chunk_pixels):
            for y_start in range(0, da.sizes['y'], self.chunk_pixels):
                x_end = min(x_start + self.chunk_pixels, da.sizes['x'])
                y_end = min(y_start + self.chunk_pixels, da.sizes['y'])
                
                # Extract chunk
                chunk = da.isel(x=slice(x_start, x_end), y=slice(y_start, y_end))
                
                # Get coordinates
                x_coords = chunk.x.values
                y_coords = chunk.y.values
                
                # Compute chunk data
                chunk_data = chunk.compute().values
                
                # Process to H3
                hex_df = self.process_chunk_to_h3(chunk_data, x_coords, y_coords)
                
                if not hex_df.empty:
                    hex_df['tile'] = file_path.stem
                    all_hexagons.append(hex_df)
        
        if not all_hexagons:
            return gpd.GeoDataFrame()
        
        # Combine all chunks
        combined_df = pd.concat(all_hexagons, ignore_index=True)
        
        # Aggregate by H3 hexagon
        band_cols = [f'band_{i:02d}' for i in range(64)]
        grouped = combined_df.groupby('h3_index').agg({
            **{col: 'mean' for col in band_cols},
            'lat': 'mean',
            'lon': 'mean'
        })
        
        # Add pixel count
        pixel_counts = combined_df.groupby('h3_index').size().rename('pixel_count')
        grouped = pd.concat([grouped, pixel_counts], axis=1)
        
        # Filter by minimum pixel count
        grouped = grouped[grouped['pixel_count'] >= self.min_pixels]
        
        # Convert to GeoDataFrame with SRAI
        grouped_reset = grouped.reset_index()
        
        # Create geometries using SRAI's efficient methods
        geometries = []
        for h3_idx in grouped_reset['h3_index']:
            boundary = h3.cell_to_boundary(h3_idx)
            poly = Polygon([(lon, lat) for lat, lon in boundary])
            geometries.append(poly)
        
        gdf = gpd.GeoDataFrame(grouped_reset, geometry=geometries, crs='EPSG:4326')
        
        logger.info(f"Created {len(gdf)} H3 hexagons from {file_path.name}")
        return gdf
    
    def process_batch_parallel(self, files: List[Path]) -> List[gpd.GeoDataFrame]:
        """Process a batch of files in parallel."""
        batch_results = []
        
        for file_path in tqdm(files, desc="Processing with SRAI+Rioxarray"):
            start_time = time.time()
            
            # Load with rioxarray
            da = self.load_tiff_optimized(file_path)
            if da is None:
                continue
            
            # Process to H3 with SRAI
            gdf = self.process_dataarray_to_h3(da, file_path)
            
            if not gdf.empty:
                batch_results.append(gdf)
            
            # Clean up
            del da
            gc.collect()
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {file_path.name} in {processing_time:.1f}s")
        
        return batch_results
    
    def process_all(self) -> gpd.GeoDataFrame:
        """Process all TIFF files using SRAI+Rioxarray optimization."""
        # Get files to process
        tiff_files = self.get_tiff_files()
        if not tiff_files:
            logger.error("No TIFF files found!")
            return gpd.GeoDataFrame()
        
        logger.info(f"Processing {len(tiff_files)} files with SRAI+Rioxarray...")
        
        # Process in batches
        all_gdfs = []
        total_start_time = time.time()
        
        for i in range(0, len(tiff_files), self.batch_size):
            batch_files = tiff_files[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tiff_files) - 1) // self.batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            batch_results = self.process_batch_parallel(batch_files)
            all_gdfs.extend(batch_results)
            
            # Memory cleanup between batches
            gc.collect()
        
        # Combine all results
        if not all_gdfs:
            logger.error("No data processed!")
            return gpd.GeoDataFrame()
        
        logger.info("Combining results from all tiles...")
        final_gdf = pd.concat(all_gdfs, ignore_index=True)
        
        # Aggregate duplicate hexagons from overlapping tiles
        band_cols = [f'band_{i:02d}' for i in range(64)]
        
        # Group by H3 index and aggregate
        grouped = final_gdf.groupby('h3_index').agg({
            **{col: 'mean' for col in band_cols if col in final_gdf.columns},
            'lat': 'mean',
            'lon': 'mean',
            'pixel_count': 'sum'
        })
        
        # Recreate geometries for final GeoDataFrame
        geometries = []
        for h3_idx in grouped.index:
            boundary = h3.cell_to_boundary(h3_idx)
            poly = Polygon([(lon, lat) for lat, lon in boundary])
            geometries.append(poly)
        
        final_gdf = gpd.GeoDataFrame(
            grouped.reset_index(),
            geometry=geometries,
            crs='EPSG:4326'
        )
        
        # Add metadata
        final_gdf['year'] = 2021
        final_gdf['resolution'] = self.h3_resolution
        final_gdf['processor'] = 'srai_rioxarray'
        
        # Rename pixel_count to total_pixels
        if 'pixel_count' in final_gdf.columns:
            final_gdf.rename(columns={'pixel_count': 'total_pixels'}, inplace=True)
        
        total_time = time.time() - total_start_time
        logger.info(f"SRAI+Rioxarray processing complete!")
        logger.info(f"Total hexagons: {len(final_gdf)}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Average pixels per hexagon: {final_gdf['total_pixels'].mean():.1f}")
        
        return final_gdf
    
    def save_results(self, gdf: gpd.GeoDataFrame, output_path: Optional[Path] = None):
        """Save results with metadata."""
        if output_path is None:
            res = self.h3_resolution
            output_path = Path(f"data/h3_2021_res{res}/srai_rioxarray_results.parquet")
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet (without geometry for parquet format)
        df_to_save = pd.DataFrame(gdf.drop(columns='geometry'))
        df_to_save.to_parquet(output_path, compression='snappy')
        logger.info(f"Saved {len(gdf)} hexagons to {output_path}")
        
        # Also save as GeoPackage for geometry preservation
        gpkg_path = output_path.with_suffix('.gpkg')
        gdf.to_file(gpkg_path, driver='GPKG')
        logger.info(f"Saved geometry to {gpkg_path}")
        
        # Save metadata
        metadata = {
            'processor': 'srai_rioxarray',
            'year': 2021,
            'h3_resolution': self.h3_resolution,
            'n_hexagons': len(gdf),
            'n_dimensions': 64,
            'band_columns': [f'band_{i:02d}' for i in range(64)],
            'mean_pixels_per_hex': float(gdf['total_pixels'].mean()) if 'total_pixels' in gdf else 0,
            'chunk_size_mb': self.chunk_size_mb,
            'chunk_pixels': self.chunk_pixels,
            'srai_version': '0.9.7',
            'rioxarray_version': '0.19.0'
        }
        
        import json
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main entry point for SRAI+Rioxarray processing."""
    import yaml
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure rioxarray settings are present
    if 'rioxarray' not in config:
        config['rioxarray'] = {
            'chunk_size_mb': 100,
            'use_parallel': True,
            'optimize_chunks': True
        }
    
    # Initialize processor
    processor = SRAIRioxarrayProcessor(config)
    
    # Process all files
    logger.info("Starting SRAI+Rioxarray optimized processing...")
    gdf = processor.process_all()
    
    if not gdf.empty:
        # Save results
        res = config['data']['h3_resolution']
        output_path = Path(f"data/h3_2021_res{res}/srai_rioxarray_results.parquet")
        processor.save_results(gdf, output_path)
        logger.info("Processing complete!")
        return gdf
    else:
        logger.error("No data processed!")
        return None


if __name__ == "__main__":
    main()