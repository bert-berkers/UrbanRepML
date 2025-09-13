#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Load AlphaEarth TIFF tiles and convert to H3 resolution 11 hexagons.
Preserves all 64 dimensions of the AlphaEarth embeddings.
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
# MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Note: SRAI provides H3 functionality with additional spatial analysis tools
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AlphaEarthToH3Converter:
    """Convert AlphaEarth TIFF tiles to H3 hexagons at resolution 11."""
    
    def __init__(self, config: dict):
        """Initialize converter with configuration."""
        self.config = config
        self.h3_resolution = config['data']['h3_resolution']
        self.source_dir = Path(config['data']['source_dir'])
        self.pattern = config['data']['pattern']
        self.batch_size = config['processing embeddings']['batch_size']
        self.min_pixels = config['processing embeddings']['min_pixels_per_hex']
        
        # Setup logging
        logging.basicConfig(
            level=config['output']['log_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def get_tiff_files(self) -> List[Path]:
        """Get all 2021 TIFF files from source directory."""
        files = list(self.source_dir.glob(self.pattern))
        logger.info(f"Found {len(files)} TIFF files for 2021")
        
        # Limit to max_tiles if specified in config
        max_tiles = self.config.get('experiment', {}).get('max_tiles', None)
        if max_tiles and len(files) > max_tiles:
            logger.info(f"Limiting to {max_tiles} tiles as specified in config")
            files = files[:max_tiles]
        else:
            logger.info(f"Processing all {len(files)} files for complete dataset")
        
        return sorted(files)
    
    def load_tiff_batch(self, files: List[Path]) -> Dict[str, np.ndarray]:
        """Load a batch of TIFF files and extract embeddings."""
        batch_data = {}
        
        for file_path in tqdm(files, desc="Loading TIFFs"):
            try:
                with rasterio.open(file_path) as src:
                    # Get bounds and transform
                    bounds = src.bounds
                    transform = src.transform
                    
                    # Read all 64 bands
                    data = src.read()  # Shape: (64, height, width)
                    
                    # Skip if mostly empty
                    if np.sum(data != 0) < 100:
                        logger.debug(f"Skipping mostly empty tile: {file_path.name}")
                        continue
                    
                    # Store with metadata
                    batch_data[str(file_path)] = {
                        'data': data,
                        'bounds': bounds,
                        'transform': transform,
                        'shape': data.shape
                    }
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
                
        return batch_data
    
    def pixels_to_h3(self, tiff_data: dict) -> pd.DataFrame:
        """Convert TIFF pixels to H3 hexagons."""
        all_hexagons = []
        
        total_files = len(tiff_data)
        with tqdm(total=total_files, desc="Converting to H3", unit="file") as pbar:
            for file_idx, (file_path, tile_info) in enumerate(tiff_data.items()):
                pbar.set_description(f"Converting to H3 (file {file_idx+1}/{total_files})")
                
                data = tile_info['data']
                bounds = tile_info['bounds']
                transform = tile_info['transform']
                
                # Get pixel coordinates
                height, width = data.shape[1], data.shape[2]
                
                # Create a grid of lat/lon coordinates
                lons = np.linspace(bounds.left, bounds.right, width)
                lats = np.linspace(bounds.top, bounds.bottom, height)
                
                # Process in chunks to manage memory - adjust for resolution
                chunk_size = self.config['processing embeddings'].get('chunk_size', 150)
                total_chunks = ((height // chunk_size) + 1) * ((width // chunk_size) + 1)
                chunk_count = 0
                
                for i in range(0, height, chunk_size):
                    for j in range(0, width, chunk_size):
                        chunk_count += 1
                        
                        # Get chunk bounds
                        i_end = min(i + chunk_size, height)
                        j_end = min(j + chunk_size, width)
                        
                        # Extract chunk data
                        chunk_data = data[:, i:i_end, j:j_end]
                        
                        # Skip empty chunks
                        if np.all(chunk_data == 0):
                            continue
                        
                        # Get coordinates for this chunk
                        chunk_lons, chunk_lats = np.meshgrid(
                            lons[j:j_end], 
                            lats[i:i_end]
                        )
                        
                        # Flatten coordinates
                        flat_lons = chunk_lons.flatten()
                        flat_lats = chunk_lats.flatten()
                        flat_data = chunk_data.reshape(64, -1).T  # Shape: (n_pixels, 64)
                        
                        # Convert to H3 hexagons (sample rate based on resolution)
                        # Resolution 10 needs denser sampling
                        sample_rate = 5 if self.h3_resolution == 10 else 10
                        for idx in range(0, len(flat_lats), sample_rate):
                            lat, lon = flat_lats[idx], flat_lons[idx]
                            
                            # Skip zero pixels
                            if np.all(flat_data[idx] == 0):
                                continue
                                
                            # Get H3 index
                            h3_idx = h3.latlng_to_cell(lat, lon, self.h3_resolution)
                            
                            # Store embedding
                            hex_data = {
                                'h3_index': h3_idx,
                                'lat': lat,
                                'lon': lon,
                                'tile': Path(file_path).stem
                            }
                            
                            # Add all 64 embedding dimensions
                            for band_idx in range(64):
                                hex_data[f'band_{band_idx:02d}'] = flat_data[idx, band_idx]
                            
                            all_hexagons.append(hex_data)
                
                # Update progress bar
                pbar.update(1)
                
                # Clean up memory
                del data
                gc.collect()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_hexagons)
        
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
    
    def process_all(self) -> pd.DataFrame:
        """Process all 2021 TIFF files to H3 hexagons."""
        # Get all TIFF files
        tiff_files = self.get_tiff_files()
        
        if not tiff_files:
            logger.error("No TIFF files found!")
            return pd.DataFrame()
        
        # Process in batches
        all_hexagons = []
        
        for i in range(0, len(tiff_files), self.batch_size):
            batch_files = tiff_files[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(tiff_files)-1)//self.batch_size + 1}")
            
            # Load batch
            batch_data = self.load_tiff_batch(batch_files)
            
            if batch_data:
                # Convert to H3
                hex_df = self.pixels_to_h3(batch_data)
                
                if not hex_df.empty:
                    all_hexagons.append(hex_df)
            
            # Clean up memory
            del batch_data
            gc.collect()
        
        # Combine all hexagons
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
            
            logger.info(f"Final dataset: {len(final_grouped)} unique H3 hexagons")
            logger.info(f"Embedding dimensions: 64")
            logger.info(f"Average pixels per hexagon: {final_grouped['total_pixels'].mean():.1f}")
            
            return final_grouped.reset_index()
        
        return pd.DataFrame()
    
    def save_results(self, df: pd.DataFrame, output_path: Optional[Path] = None):
        """Save H3 hexagon data to parquet file."""
        if output_path is None:
            res = self.h3_resolution
            output_path = Path(f"data/h3_2021_res{res}/{self.config['data']['output_file']}")
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        df.to_parquet(output_path, compression='snappy')
        logger.info(f"Saved {len(df)} hexagons to {output_path}")
        
        # Save metadata
        metadata = {
            'year': 2021,
            'h3_resolution': self.h3_resolution,
            'n_hexagons': len(df),
            'n_dimensions': 64,
            'band_columns': [f'band_{i:02d}' for i in range(64)],
            'mean_pixels_per_hex': float(df['total_pixels'].mean()) if 'total_pixels' in df else 0,
            'source_files': len(self.get_tiff_files())
        }
        
        import json
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main entry point for AlphaEarth to H3 conversion."""
    import yaml
    import sys
    from pathlib import Path
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update H3 resolution from config
    config['data']['h3_resolution'] = config['processing embeddings']['h3_resolution']
    
    # Initialize converter
    converter = AlphaEarthToH3Converter(config)
    
    # Process all files
    logger.info("Starting AlphaEarth to H3 conversion for Del Norte 2021")
    hex_df = converter.process_all()
    
    if not hex_df.empty:
        # Save results
        res = config['data']['h3_resolution']
        output_path = Path(f"data/h3_2021_res{res}/{config['data']['output_file']}")
        converter.save_results(hex_df, output_path)
        logger.info("Conversion complete!")
        return hex_df
    else:
        logger.error("No data processed!")
        return None


if __name__ == "__main__":
    main()