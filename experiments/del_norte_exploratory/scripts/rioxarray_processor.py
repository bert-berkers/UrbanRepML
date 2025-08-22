#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rioxarray-based AlphaEarth TIFF processor with dask optimization.
Improved performance for large-scale multi-band satellite data processing.
"""

import numpy as np
import pandas as pd
import rasterio

# Try to import rioxarray components
try:
    import rioxarray
    import xarray as xr
    RIOXARRAY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: rioxarray not available ({e}), falling back to rasterio")
    RIOXARRAY_AVAILABLE = False
    rioxarray = None
    xr = None

import h3
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import gc
import warnings
import time
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RioxarrayAlphaEarthProcessor:
    """Process AlphaEarth TIFF files using rioxarray with dask optimization."""
    
    def __init__(self, config: dict):
        """Initialize processor with configuration."""
        self.config = config
        self.h3_resolution = config['data']['h3_resolution']
        self.source_dir = Path(config['data']['source_dir'])
        self.pattern = config['data']['pattern']
        self.batch_size = config['processing']['batch_size']
        self.min_pixels = config['processing']['min_pixels_per_hex']
        
        # Rioxarray-specific settings
        self.chunk_size_mb = config.get('rioxarray', {}).get('chunk_size_mb', 100)
        self.use_parallel = config.get('rioxarray', {}).get('use_parallel', True)
        self.optimize_chunks = config.get('rioxarray', {}).get('optimize_chunks', True)
        
        # Calculate optimal chunk size for TIFF files
        # 3072x3072 pixels * 64 bands * 4 bytes (float32) = ~2.4GB per file
        # Target chunks of ~100MB each
        pixels_per_chunk = (self.chunk_size_mb * 1024 * 1024) // (64 * 4)  # 64 bands, 4 bytes per pixel
        self.chunk_pixels = int(np.sqrt(pixels_per_chunk))  # Square chunks
        
        # Setup logging
        logging.basicConfig(
            level=config['output']['log_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"RioxarrayProcessor initialized with {self.chunk_pixels}x{self.chunk_pixels} chunks")
        
    def get_tiff_files(self) -> List[Path]:
        """Get all 2021 TIFF files from source directory."""
        files = list(self.source_dir.glob(self.pattern))
        logger.info(f"Found {len(files)} TIFF files for processing")
        
        # For benchmark, limit to smaller subset initially
        if len(files) > 10:
            files = files[:10]
            logger.info(f"Limited to first {len(files)} files for rioxarray benchmark")
        
        return sorted(files)
    
    def load_tiff_with_rioxarray(self, file_path: Path) -> Optional[xr.DataArray]:
        """Load TIFF using rioxarray with optimized chunking."""
        if not RIOXARRAY_AVAILABLE:
            logger.error("Rioxarray not available, cannot process with this method")
            return None
            
        try:
            # Set rioxarray options for performance
            with rioxarray.set_options(export_grid_mapping=False):
                # Load with dask chunks for memory efficiency
                da = rioxarray.open_rasterio(
                    file_path,
                    chunks={'x': self.chunk_pixels, 'y': self.chunk_pixels, 'band': 64},
                    lock=False,  # Allow parallel reading for COG files
                    decode_times=False  # Skip time decoding for performance
                )
                
                # Ensure we have 64 bands
                if da.sizes['band'] != 64:
                    logger.warning(f"File {file_path} has {da.sizes['band']} bands, expected 64")
                    return None
                
                # Skip if mostly empty (check a sample)
                sample = da.isel(x=slice(0, 100), y=slice(0, 100)).compute()
                if np.sum(sample != 0) < 100:
                    logger.debug(f"Skipping mostly empty tile: {file_path.name}")
                    return None
                
                return da
                
        except Exception as e:
            logger.error(f"Error loading {file_path} with rioxarray: {e}")
            return None
    
    def da_to_h3_vectorized(self, da: xr.DataArray) -> pd.DataFrame:
        """Convert xarray DataArray to H3 hexagons using vectorized operations."""
        logger.info("Converting to H3 hexagons using vectorized rioxarray approach...")
        
        # Get coordinate grids
        lons, lats = np.meshgrid(da.x.values, da.y.values)
        
        # Flatten coordinates for vectorized H3 conversion
        flat_lons = lons.flatten()
        flat_lats = lats.flatten()
        
        # Subsample for memory efficiency (every 10th pixel like original)
        subsample_idx = np.arange(0, len(flat_lats), 10)
        sample_lons = flat_lons[subsample_idx]
        sample_lats = flat_lats[subsample_idx]
        
        # Vectorized H3 conversion
        logger.info(f"Converting {len(sample_lats)} pixels to H3 indices...")
        h3_indices = []
        for lat, lon in zip(sample_lats, sample_lons):
            try:
                h3_idx = h3.latlng_to_cell(lat, lon, self.h3_resolution)
                h3_indices.append(h3_idx)
            except:
                h3_indices.append(None)
        
        # Get corresponding pixel values
        # Reshape to match original sampling
        y_indices = subsample_idx // da.sizes['x']
        x_indices = subsample_idx % da.sizes['x']
        
        # Extract band values for sampled pixels
        pixel_data = []
        for i, (y_idx, x_idx) in enumerate(zip(y_indices, x_indices)):
            if h3_indices[i] is not None:
                # Get all 64 band values for this pixel
                pixel_values = da.isel(y=y_idx, x=x_idx).compute().values
                
                # Skip zero pixels
                if np.all(pixel_values == 0):
                    continue
                
                pixel_dict = {
                    'h3_index': h3_indices[i],
                    'lat': sample_lats[i],
                    'lon': sample_lons[i],
                    'tile': file_path.stem if 'file_path' in locals() else 'unknown'
                }
                
                # Add all 64 bands
                for band_idx in range(64):
                    pixel_dict[f'band_{band_idx:02d}'] = pixel_values[band_idx]
                
                pixel_data.append(pixel_dict)
        
        if not pixel_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(pixel_data)
        
        # Aggregate by H3 hexagon (mean of all pixels in same hexagon)
        if not df.empty:
            band_cols = [f'band_{i:02d}' for i in range(64)]
            grouped = df.groupby('h3_index')[band_cols].mean()
            
            # Add location info (use mean lat/lon)
            locations = df.groupby('h3_index')[['lat', 'lon']].mean()
            grouped = pd.concat([grouped, locations], axis=1)
            
            # Add pixel count for quality assessment
            pixel_counts = df.groupby('h3_index').size().rename('pixel_count')
            grouped = pd.concat([grouped, pixel_counts], axis=1)
            
            # Filter by minimum pixel count
            grouped = grouped[grouped['pixel_count'] >= self.min_pixels]
            
            logger.info(f"Created {len(grouped)} H3 hexagons at resolution {self.h3_resolution}")
            return grouped.reset_index()
        
        return pd.DataFrame()
    
    def process_file_batch_rioxarray(self, files: List[Path]) -> List[pd.DataFrame]:
        """Process a batch of files using rioxarray with parallel processing."""
        batch_results = []
        
        for file_path in tqdm(files, desc="Processing with rioxarray"):
            start_time = time.time()
            
            # Load with rioxarray
            da = self.load_tiff_with_rioxarray(file_path)
            if da is None:
                continue
            
            # Convert to H3
            hex_df = self.da_to_h3_vectorized(da)
            if not hex_df.empty:
                batch_results.append(hex_df)
            
            # Clean up
            del da
            gc.collect()
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {file_path.name} in {processing_time:.1f}s with rioxarray")
        
        return batch_results
    
    def process_all_rioxarray(self) -> pd.DataFrame:
        """Process all files using rioxarray approach."""
        if not RIOXARRAY_AVAILABLE:
            logger.error("Rioxarray not available. Install with: pip install rioxarray")
            return pd.DataFrame()
        
        # Get files
        tiff_files = self.get_tiff_files()
        if not tiff_files:
            logger.error("No TIFF files found!")
            return pd.DataFrame()
        
        # Process in batches
        all_hexagons = []
        total_start_time = time.time()
        
        for i in range(0, len(tiff_files), self.batch_size):
            batch_files = tiff_files[i:i+self.batch_size]
            logger.info(f"Processing rioxarray batch {i//self.batch_size + 1}/{(len(tiff_files)-1)//self.batch_size + 1}")
            
            batch_results = self.process_file_batch_rioxarray(batch_files)
            all_hexagons.extend(batch_results)
            
            # Memory cleanup
            gc.collect()
        
        # Combine all results
        if all_hexagons:
            final_df = pd.concat(all_hexagons, ignore_index=True)
            
            # Aggregate duplicates (hexagons appearing in multiple tiles)
            band_cols = [f'band_{i:02d}' for i in range(64)]
            final_grouped = final_df.groupby('h3_index')[band_cols + ['lat', 'lon']].mean()
            
            # Add total pixel count
            pixel_counts = final_df.groupby('h3_index')['pixel_count'].sum()
            final_grouped['total_pixels'] = pixel_counts
            
            # Add metadata
            final_grouped['year'] = 2021
            final_grouped['resolution'] = self.h3_resolution
            final_grouped['processor'] = 'rioxarray'
            
            total_time = time.time() - total_start_time
            logger.info(f"Rioxarray processing complete: {len(final_grouped)} hexagons in {total_time:.1f}s")
            logger.info(f"Average: {final_grouped['total_pixels'].mean():.1f} pixels per hexagon")
            
            return final_grouped.reset_index()
        
        return pd.DataFrame()
    
    def save_results(self, df: pd.DataFrame, output_path: Optional[Path] = None):
        """Save results with rioxarray processor metadata."""
        if output_path is None:
            res = self.h3_resolution
            output_path = Path(f"data/h3_2021_res{res}/rioxarray_results.parquet")
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        df.to_parquet(output_path, compression='snappy')
        logger.info(f"Saved {len(df)} hexagons to {output_path}")
        
        # Save metadata
        metadata = {
            'processor': 'rioxarray',
            'year': 2021,
            'h3_resolution': self.h3_resolution,
            'n_hexagons': len(df),
            'n_dimensions': 64,
            'band_columns': [f'band_{i:02d}' for i in range(64)],
            'mean_pixels_per_hex': float(df['total_pixels'].mean()) if 'total_pixels' in df else 0,
            'chunk_size_mb': self.chunk_size_mb,
            'chunk_pixels': self.chunk_pixels,
            'rioxarray_available': RIOXARRAY_AVAILABLE
        }
        
        import json
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main entry point for rioxarray processing test."""
    import yaml
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add rioxarray-specific settings if not present
    if 'rioxarray' not in config:
        config['rioxarray'] = {
            'chunk_size_mb': 100,
            'use_parallel': True,
            'optimize_chunks': True
        }
    
    # Initialize processor
    processor = RioxarrayAlphaEarthProcessor(config)
    
    # Process files
    logger.info("Starting rioxarray-based AlphaEarth processing...")
    hex_df = processor.process_all_rioxarray()
    
    if not hex_df.empty:
        # Save results
        res = config['data']['h3_resolution']
        output_path = Path(f"data/h3_2021_res{res}/rioxarray_benchmark.parquet")
        processor.save_results(hex_df, output_path)
        logger.info("Rioxarray processing complete!")
        return hex_df
    else:
        logger.error("No data processed!")
        return None


if __name__ == "__main__":
    main()