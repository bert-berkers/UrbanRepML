"""
Process Cascadia AlphaEarth tiles to H3 hexagons at multiple resolutions (5-11).

This script processes AlphaEarth GeoTIFF tiles and converts them to H3 hexagonal
representations at resolutions 5 through 11, maintaining hierarchical relationships
and optimizing for memory efficiency.

Usage:
    python process_cascadia_multires.py --year 2023 --resolution 8
    python process_cascadia_multires.py --year 2023 --all_resolutions
    python process_cascadia_multires.py --all_years --all_resolutions
"""

import os
import gc
import glob
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import h3
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console only for now
    ]
)
logger = logging.getLogger(__name__)


class CascadiaMultiResProcessor:
    """Process Cascadia AlphaEarth data to multiple H3 resolutions."""
    
    # H3 resolution specifications
    H3_SPECS = {
        5: {'edge_km': 9.2, 'area_km2': 252.9, 'batch_size': 100000},
        6: {'edge_km': 3.2, 'area_km2': 31.0, 'batch_size': 50000},
        7: {'edge_km': 1.2, 'area_km2': 3.65, 'batch_size': 20000},
        8: {'edge_km': 0.46, 'area_km2': 0.46, 'batch_size': 10000},
        9: {'edge_km': 0.17, 'area_km2': 0.054, 'batch_size': 5000},
        10: {'edge_km': 0.066, 'area_km2': 0.0063, 'batch_size': 2000},
        11: {'edge_km': 0.025, 'area_km2': 0.00074, 'batch_size': 1000}
    }
    
    def __init__(self, 
                 input_dir: str = "../../data/alphaearth_raw",
                 output_dir: str = "../../data/h3_processed",
                 cache_dir: str = "../../data/cache"):
        """
        Initialize processor.
        
        Args:
            input_dir: Directory containing AlphaEarth tiles
            output_dir: Directory for H3 processed outputs
            cache_dir: Directory for intermediate caching
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'tiles_processed': 0,
            'hexagons_created': {},
            'processing_time': {},
            'memory_peaks': {}
        }
        
        logger.info("Initialized Cascadia Multi-Resolution Processor")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Cache: {cache_dir}")
    
    def get_tile_files(self, year: int) -> List[str]:
        """
        Get list of tile files for a specific year.
        
        Args:
            year: Year to process
            
        Returns:
            List of tile file paths
        """
        pattern = os.path.join(self.input_dir, str(year), f"Cascadia_AlphaEarth_{year}_*.tif")
        files = glob.glob(pattern)
        
        logger.info(f"Found {len(files)} tiles for year {year}")
        return sorted(files)
    
    def process_tile_to_h3(self, 
                           tile_path: str, 
                           resolution: int,
                           verbose: bool = False) -> Optional[gpd.GeoDataFrame]:
        """
        Process a single tile to H3 hexagons at specified resolution.
        
        Args:
            tile_path: Path to GeoTIFF tile
            resolution: H3 resolution (5-11)
            verbose: Enable verbose logging
            
        Returns:
            GeoDataFrame with H3 hexagons and embeddings
        """
        tile_name = os.path.basename(tile_path).replace('.tif', '')
        
        if verbose:
            logger.info(f"Processing tile {tile_name} to resolution {resolution}")
        
        try:
            with rasterio.open(tile_path) as src:
                # Get metadata
                transform = src.transform
                crs = src.crs
                bounds = src.bounds
                n_bands = src.count
                
                if verbose:
                    logger.info(f"  Tile shape: {src.height}x{src.width}, Bands: {n_bands}")
                
                # Determine sampling strategy based on resolution
                sample_rate = self.get_sampling_rate(resolution, src.height * src.width)
                
                # Read data with potential downsampling
                if sample_rate < 1.0:
                    # Downsample for lower resolutions
                    step = int(1 / sample_rate)
                    rows = np.arange(0, src.height, step)
                    cols = np.arange(0, src.width, step)
                    
                    # Read sampled data
                    data = np.zeros((n_bands, len(rows), len(cols)))
                    for i, row in enumerate(rows):
                        for j, col in enumerate(cols):
                            window = ((row, row+1), (col, col+1))
                            data[:, i, j] = src.read(window=window).squeeze()
                    
                    # Update coordinates
                    row_coords, col_coords = np.meshgrid(rows, cols, indexing='ij')
                else:
                    # Read full data for high resolutions
                    data = src.read()
                    row_coords, col_coords = np.meshgrid(
                        range(src.height), range(src.width), indexing='ij'
                    )
                
                # Get geographic coordinates
                from rasterio.transform import xy
                xs, ys = xy(transform, row_coords.flatten(), col_coords.flatten())
                
                # Convert to lat/lon if needed
                if crs and crs.to_epsg() != 4326:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                    lons, lats = transformer.transform(xs, ys)
                else:
                    lons, lats = xs, ys
                
                # Reshape data
                if sample_rate < 1.0:
                    data_reshaped = data.reshape(n_bands, -1).T
                else:
                    data_reshaped = data.reshape(n_bands, -1).T
                
                # Clear original data from memory
                del data
                gc.collect()
                
                # Process to H3 in batches
                batch_size = self.H3_SPECS[resolution]['batch_size']
                h3_accumulator = {}
                
                for batch_start in range(0, len(lons), batch_size):
                    batch_end = min(batch_start + batch_size, len(lons))
                    
                    for i in range(batch_start, batch_end):
                        # Skip nodata pixels
                        pixel_embeddings = data_reshaped[i]
                        if np.all(pixel_embeddings == 0) or np.any(np.isnan(pixel_embeddings)):
                            continue
                        
                        # Get H3 index
                        h3_index = h3.latlng_to_cell(lats[i], lons[i], resolution)
                        
                        # Accumulate embeddings
                        if h3_index not in h3_accumulator:
                            h3_accumulator[h3_index] = {
                                'embeddings': [],
                                'pixel_count': 0
                            }
                        
                        h3_accumulator[h3_index]['embeddings'].append(pixel_embeddings)
                        h3_accumulator[h3_index]['pixel_count'] += 1
                    
                    # Periodic garbage collection
                    if batch_end % (batch_size * 10) == 0:
                        gc.collect()
                
                # Clear arrays
                del data_reshaped
                gc.collect()
                
                # Aggregate H3 cells
                if not h3_accumulator:
                    logger.warning(f"  No valid data in tile {tile_name}")
                    return None
                
                aggregated = []
                for h3_index, data in h3_accumulator.items():
                    # Compute mean embeddings
                    embeddings_stack = np.vstack(data['embeddings'])
                    mean_embeddings = np.mean(embeddings_stack, axis=0)
                    
                    # Get H3 geometry
                    boundary = h3.cell_to_boundary(h3_index)
                    polygon = Polygon([(lon, lat) for lat, lon in boundary])
                    
                    # Store result
                    result = {
                        'h3_index': h3_index,
                        'resolution': resolution,
                        'geometry': polygon,
                        'pixel_count': data['pixel_count'],
                        'tile_id': tile_name
                    }
                    
                    # Add embedding dimensions
                    for j, val in enumerate(mean_embeddings):
                        result[f'embed_{j}'] = val
                    
                    aggregated.append(result)
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(aggregated, crs='EPSG:4326')
                
                if verbose:
                    logger.info(f"  Created {len(gdf)} H3 cells at resolution {resolution}")
                
                return gdf
                
        except Exception as e:
            logger.error(f"Error processing tile {tile_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_sampling_rate(self, resolution: int, total_pixels: int) -> float:
        """
        Determine sampling rate based on H3 resolution and data size.
        
        Args:
            resolution: H3 resolution
            total_pixels: Total number of pixels in tile
            
        Returns:
            Sampling rate (0-1)
        """
        # For lower resolutions, we can sample more aggressively
        if resolution <= 6:
            return 0.1  # Sample 10% of pixels
        elif resolution == 7:
            return 0.25  # Sample 25%
        elif resolution == 8:
            return 0.5  # Sample 50%
        else:
            return 1.0  # Use all pixels for high resolutions
    
    def process_year_resolution(self, year: int, resolution: int) -> gpd.GeoDataFrame:
        """
        Process all tiles for a year at a specific resolution.
        
        Args:
            year: Year to process
            resolution: H3 resolution
            
        Returns:
            Combined GeoDataFrame for all tiles
        """
        logger.info(f"\nProcessing year {year} at resolution {resolution}")
        start_time = datetime.now()
        
        # Check if already processed
        output_file = os.path.join(
            self.output_dir, 
            f"resolution_{resolution}",
            f"cascadia_{year}_h3_res{resolution}.parquet"
        )
        
        if os.path.exists(output_file):
            logger.info(f"Already processed: {output_file}")
            return gpd.read_parquet(output_file)
        
        # Get tile files
        tile_files = self.get_tile_files(year)
        
        if not tile_files:
            logger.error(f"No tiles found for year {year}")
            return None
        
        # Process each tile
        all_gdfs = []
        for i, tile_path in enumerate(tqdm(tile_files, desc=f"Res {resolution}")):
            logger.info(f"Processing tile {i+1}/{len(tile_files)}")
            
            gdf = self.process_tile_to_h3(tile_path, resolution)
            if gdf is not None:
                all_gdfs.append(gdf)
            
            # Periodic memory cleanup
            if (i + 1) % 5 == 0:
                gc.collect()
        
        if not all_gdfs:
            logger.error(f"No valid data processed for year {year}")
            return None
        
        # Combine all tiles
        logger.info("Combining tiles...")
        combined_gdf = pd.concat(all_gdfs, ignore_index=True)
        
        # Aggregate overlapping hexagons
        logger.info("Aggregating overlapping hexagons...")
        embed_cols = [col for col in combined_gdf.columns if col.startswith('embed_')]
        
        aggregated = combined_gdf.groupby('h3_index').agg({
            'resolution': 'first',
            'geometry': 'first',
            'pixel_count': 'sum',
            **{col: 'mean' for col in embed_cols}
        }).reset_index()
        
        # Convert back to GeoDataFrame
        final_gdf = gpd.GeoDataFrame(aggregated, crs='EPSG:4326', geometry='geometry')
        
        # Save to parquet
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_gdf.to_parquet(output_file)
        
        # Update statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        self.stats['hexagons_created'][f"{year}_res{resolution}"] = len(final_gdf)
        self.stats['processing_time'][f"{year}_res{resolution}"] = elapsed
        
        logger.info(f"Completed year {year} resolution {resolution}")
        logger.info(f"  Hexagons: {len(final_gdf)}")
        logger.info(f"  Time: {elapsed:.1f} seconds")
        logger.info(f"  Saved to: {output_file}")
        
        return final_gdf
    
    def create_hierarchical_mapping(self, year: int, resolutions: List[int] = None):
        """
        Create parent-child mappings between H3 resolutions.
        
        Args:
            year: Year to process
            resolutions: List of resolutions (default: 5-11)
        """
        if resolutions is None:
            resolutions = list(range(5, 12))
        
        logger.info(f"\nCreating hierarchical mappings for year {year}")
        
        mappings = {}
        
        for i in range(len(resolutions) - 1):
            parent_res = resolutions[i]
            child_res = resolutions[i + 1]
            
            logger.info(f"Mapping resolution {parent_res} -> {child_res}")
            
            # Load data
            parent_file = os.path.join(
                self.output_dir,
                f"resolution_{parent_res}",
                f"cascadia_{year}_h3_res{parent_res}.parquet"
            )
            child_file = os.path.join(
                self.output_dir,
                f"resolution_{child_res}",
                f"cascadia_{year}_h3_res{child_res}.parquet"
            )
            
            if not os.path.exists(parent_file) or not os.path.exists(child_file):
                logger.warning(f"Missing data files for mapping")
                continue
            
            parent_gdf = gpd.read_parquet(parent_file)
            child_gdf = gpd.read_parquet(child_file)
            
            # Create mapping
            mapping = {}
            for parent_h3 in parent_gdf['h3_index']:
                # Get children
                children = h3.cell_to_children(parent_h3, child_res)
                # Filter to only children that exist in our data
                existing_children = [c for c in children if c in child_gdf['h3_index'].values]
                if existing_children:
                    mapping[parent_h3] = existing_children
            
            mappings[f"res{parent_res}_to_res{child_res}"] = mapping
            
            logger.info(f"  Mapped {len(mapping)} parent cells to children")
        
        # Save mappings
        output_file = os.path.join(
            self.output_dir,
            f"hierarchical_mappings_{year}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        
        logger.info(f"Saved hierarchical mappings to {output_file}")
    
    def process_all(self, 
                   years: List[int] = None,
                   resolutions: List[int] = None):
        """
        Process all years and resolutions.
        
        Args:
            years: List of years (default: 2017-2024)
            resolutions: List of H3 resolutions (default: 5-11)
        """
        if years is None:
            years = [2023]  # Default to 2023 for testing
        if resolutions is None:
            resolutions = list(range(5, 12))
        
        logger.info("="*60)
        logger.info("Starting Cascadia Multi-Resolution Processing")
        logger.info(f"Years: {years}")
        logger.info(f"Resolutions: {resolutions}")
        logger.info("="*60)
        
        for year in years:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing Year {year}")
            logger.info(f"{'='*40}")
            
            for resolution in resolutions:
                self.process_year_resolution(year, resolution)
                
                # Memory cleanup between resolutions
                gc.collect()
            
            # Create hierarchical mappings for the year
            self.create_hierarchical_mapping(year, resolutions)
        
        # Save processing statistics
        self.save_statistics()
        
        logger.info("\n" + "="*60)
        logger.info("Processing Complete!")
        logger.info("="*60)
        self.print_statistics()
    
    def save_statistics(self):
        """Save processing statistics to JSON."""
        stats_file = os.path.join(self.output_dir, "processing_statistics.json")
        
        self.stats['tiles_processed'] = sum(
            1 for k in self.stats['hexagons_created'] if self.stats['hexagons_created'][k] > 0
        )
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")
    
    def print_statistics(self):
        """Print processing statistics."""
        logger.info("\nProcessing Statistics:")
        logger.info(f"  Tiles processed: {self.stats['tiles_processed']}")
        
        if self.stats['hexagons_created']:
            logger.info("\n  Hexagons created:")
            for key, count in self.stats['hexagons_created'].items():
                logger.info(f"    {key}: {count:,}")
        
        if self.stats['processing_time']:
            total_time = sum(self.stats['processing_time'].values())
            logger.info(f"\n  Total processing time: {total_time:.1f} seconds")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Process Cascadia AlphaEarth to H3")
    parser.add_argument('--year', type=int, help='Single year to process')
    parser.add_argument('--years', nargs='+', type=int, help='Multiple years')
    parser.add_argument('--all_years', action='store_true', 
                       help='Process all years (2017-2024)')
    parser.add_argument('--resolution', type=int, help='Single H3 resolution (5-11)')
    parser.add_argument('--resolutions', nargs='+', type=int, 
                       help='Multiple resolutions')
    parser.add_argument('--all_resolutions', action='store_true',
                       help='Process all resolutions (5-11)')
    parser.add_argument('--input_dir', type=str, 
                       default='../../data/alphaearth_raw',
                       help='Input directory with AlphaEarth tiles')
    parser.add_argument('--output_dir', type=str,
                       default='../../data/h3_processed',
                       help='Output directory for H3 data')
    
    args = parser.parse_args()
    
    # Determine years
    if args.all_years:
        years = list(range(2017, 2025))
    elif args.years:
        years = args.years
    elif args.year:
        years = [args.year]
    else:
        years = [2023]  # Default
    
    # Determine resolutions
    if args.all_resolutions:
        resolutions = list(range(5, 12))
    elif args.resolutions:
        resolutions = args.resolutions
    elif args.resolution:
        resolutions = [args.resolution]
    else:
        resolutions = [8]  # Default to GEO-INFER standard
    
    # Create processor
    processor = CascadiaMultiResProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Process
    processor.process_all(years, resolutions)


if __name__ == "__main__":
    main()