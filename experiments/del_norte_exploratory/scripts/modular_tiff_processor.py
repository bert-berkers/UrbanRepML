#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modular TIFF to H3 Processor - Simple, grounded approach with checkpointing
Uses SRAI for H3 operations with pre-regionalization for efficiency
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# SRAI imports for H3 operations
from srai.regionalizers import H3Regionalizer
from shapely.geometry import Polygon, Point
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class ModularTiffProcessor:
    """Process AlphaEarth TIFFs to H3 hexagons with modular, resumable approach"""
    
    def __init__(self, config: dict):
        self.config = config
        self.h3_resolution = config['data']['h3_resolution']
        self.source_dir = Path(config['data']['source_dir'])
        self.output_dir = Path(f"data/h3_2021_res{self.h3_resolution}_modular")
        self.checkpoint_dir = Path("data/checkpoints")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.subtile_size = config['processing'].get('subtile_size', 256)
        self.batch_size = config['processing'].get('subtiles_per_batch', 10)
        self.min_pixels = config['processing']['min_pixels_per_hex']
        
        # Checkpoint tracking
        self.checkpoint_file = self.checkpoint_dir / "modular_progress.json"
        self.completed_tiles = set()
        self.completed_subtiles = {}
        
        self.load_checkpoint()
        
        # Pre-regionalize study area with SRAI
        self.h3_gdf = None
        self.hex_lookup = {}
        self.hex_tree = None
        self.preregionalize_study_area()
        
        logger.info(f"Modular processor initialized")
        logger.info(f"  H3 Resolution: {self.h3_resolution}")
        logger.info(f"  Subtile size: {self.subtile_size}x{self.subtile_size}")
        logger.info(f"  Pre-computed hexagons: {len(self.h3_gdf) if self.h3_gdf is not None else 0}")
        logger.info(f"  Previously completed: {len(self.completed_tiles)} tiles")
    
    def load_checkpoint(self):
        """Load processing checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            self.completed_tiles = set(checkpoint.get('completed_tiles', []))
            self.completed_subtiles = checkpoint.get('completed_subtiles', {})
            logger.info(f"Loaded checkpoint with {len(self.completed_tiles)} completed tiles")
    
    def save_checkpoint(self):
        """Save processing checkpoint"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'completed_tiles': list(self.completed_tiles),
            'completed_subtiles': self.completed_subtiles,
            'h3_resolution': self.h3_resolution
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def preregionalize_study_area(self):
        """Pre-compute all H3 hexagons for the study area using SRAI"""
        logger.info("Pre-regionalizing study area with SRAI...")
        
        # Get study area bounds from config
        study_area = self.config.get('study_area', {})
        bounds = study_area.get('bounds', {})
        
        # Use Cascadia bounds if not specified
        west = bounds.get('west', -124.703589)
        east = bounds.get('east', -117.352765)
        south = bounds.get('south', 38.667442)
        north = bounds.get('north', 43.372548)
        
        # Create bounding box polygon
        bbox_polygon = Polygon([
            (west, south),
            (east, south),
            (east, north),
            (west, north),
            (west, south)
        ])
        
        # Create GeoDataFrame with bounding box
        gdf_bounds = gpd.GeoDataFrame(
            {'geometry': [bbox_polygon]},
            crs='EPSG:4326'
        )
        
        # Initialize H3 regionalizer
        regionalizer = H3Regionalizer(resolution=self.h3_resolution)
        
        # Generate H3 hexagons for the area
        logger.info(f"Generating H3 hexagons at resolution {self.h3_resolution}...")
        self.h3_gdf = regionalizer.transform(gdf_bounds)
        
        # Create lookup structures for fast pixel-to-hexagon mapping
        logger.info("Building spatial index for hexagons...")
        
        # Store hexagon centroids for KDTree
        centroids = []
        for idx, row in self.h3_gdf.iterrows():
            # Get hexagon centroid
            centroid = row.geometry.centroid
            centroids.append([centroid.x, centroid.y])
            self.hex_lookup[idx] = {
                'geometry': row.geometry,
                'centroid': (centroid.x, centroid.y),
                'values': []  # Will store pixel values for averaging
            }
        
        # Build KDTree for fast nearest-neighbor lookup
        self.hex_tree = cKDTree(np.array(centroids))
        
        logger.info(f"Pre-regionalization complete: {len(self.h3_gdf)} hexagons created")
    
    def filter_coastal_tiles(self, tiles: List[Path]) -> List[Path]:
        """Filter tiles to only include those in the coastal study area (west of -121°)"""
        logger.info("Filtering tiles for coastal study area...")
        
        coastal_tiles = []
        skipped_count = 0
        
        for tile in tiles:
            try:
                with rioxarray.open_rasterio(tile) as da:
                    bounds = da.rio.bounds()
                    # Check if tile has any coverage west of -121°
                    if bounds[2] < -121.0:  # Eastern edge is west of -121°
                        coastal_tiles.append(tile)
                    else:
                        skipped_count += 1
            except Exception as e:
                logger.warning(f"Could not read bounds for {tile.name}: {e}")
                # Include tile if we can't read bounds (conservative approach)
                coastal_tiles.append(tile)
        
        logger.info(f"Spatial filtering complete:")
        logger.info(f"  Coastal tiles: {len(coastal_tiles)}")
        logger.info(f"  Skipped tiles: {skipped_count}")
        logger.info(f"  Efficiency gain: {skipped_count/(len(tiles))*100:.1f}% reduction")
        
        return coastal_tiles
    
    def get_subtile_bounds(self, tile_bounds: Tuple[float, float, float, float], 
                          subtile_idx: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Calculate geographic bounds for a subtile"""
        min_lon, min_lat, max_lon, max_lat = tile_bounds
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        
        # Calculate how many subtiles fit in this tile (3072 / 256 = 12)
        n_subtiles = 3072 // self.subtile_size
        
        row, col = subtile_idx
        subtile_min_lon = min_lon + (col * lon_range / n_subtiles)
        subtile_max_lon = min_lon + ((col + 1) * lon_range / n_subtiles)
        subtile_min_lat = min_lat + (row * lat_range / n_subtiles)
        subtile_max_lat = min_lat + ((row + 1) * lat_range / n_subtiles)
        
        return (subtile_min_lon, subtile_min_lat, subtile_max_lon, subtile_max_lat)
    
    def process_subtile(self, data_chunk: np.ndarray, 
                       bounds: Tuple[float, float, float, float]) -> Dict[str, np.ndarray]:
        """Process a single subtile to H3 hexagons using pre-computed hexagons"""
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Skip if chunk is mostly empty
        if np.sum(data_chunk != 0) < 100:
            return {}
        
        # Create coordinate grids for this subtile
        lons = np.linspace(min_lon, max_lon, self.subtile_size)
        lats = np.linspace(max_lat, min_lat, self.subtile_size)  # Note: reversed for raster
        
        # Process each pixel and assign to nearest hexagon
        hex_values = {}
        
        # Sample pixels (every 5th pixel for efficiency)
        sample_stride = 5
        for i in range(0, min(self.subtile_size, data_chunk.shape[1]), sample_stride):
            for j in range(0, min(self.subtile_size, data_chunk.shape[2]), sample_stride):
                # Skip if pixel is empty
                if data_chunk[0, i, j] == 0:
                    continue
                
                # Get pixel coordinates
                pixel_lon = lons[j]
                pixel_lat = lats[i]
                
                # Find nearest hexagon using KDTree
                distance, hex_idx = self.hex_tree.query([pixel_lon, pixel_lat])
                
                # Only include if pixel is reasonably close to a hexagon
                # (within ~1km at equator, adjusting for latitude)
                max_distance = 0.01  # ~1km in degrees
                if distance > max_distance:
                    continue
                
                # Get hexagon ID
                hex_id = self.h3_gdf.index[hex_idx]
                
                # Store pixel values for this hexagon
                if hex_id not in hex_values:
                    hex_values[hex_id] = []
                
                # Add all band values for this pixel
                hex_values[hex_id].append(data_chunk[:, i, j])
        
        # Average values for each hexagon
        averaged_hex_values = {}
        for hex_id, pixel_list in hex_values.items():
            if len(pixel_list) >= self.min_pixels:
                # Average across all pixels in this hexagon
                averaged_hex_values[hex_id] = np.mean(pixel_list, axis=0)
        
        return averaged_hex_values
    
    def process_single_tile_worker(self, tile_path: Path) -> Dict[str, np.ndarray]:
        """Process a single tile - worker function for parallel processing"""
        tile_name = tile_path.stem
        
        # Check if already completed
        if tile_name in self.completed_tiles:
            logger.debug(f"Skipping completed tile: {tile_name}")
            return {}
        
        logger.debug(f"Processing tile: {tile_name}")
        
        try:
            # Open the TIFF file
            with rioxarray.open_rasterio(tile_path, chunks={'band': 64}) as da:
                # Get tile bounds
                bounds = da.rio.bounds()
                
                # Process in subtiles
                n_subtiles = 3072 // self.subtile_size  # 12x12 = 144 subtiles
                all_hex_values = {}
                
                for row in range(n_subtiles):
                    for col in range(n_subtiles):
                        # Extract subtile data
                        row_start = row * self.subtile_size
                        row_end = (row + 1) * self.subtile_size
                        col_start = col * self.subtile_size
                        col_end = (col + 1) * self.subtile_size
                        
                        subtile_data = da.isel(
                            x=slice(col_start, col_end),
                            y=slice(row_start, row_end)
                        ).compute()
                        
                        # Get subtile bounds
                        subtile_bounds = self.get_subtile_bounds(bounds, (row, col))
                        
                        # Process subtile
                        hex_values = self.process_subtile(subtile_data.values, subtile_bounds)
                        
                        # Merge results
                        for hex_id, values in hex_values.items():
                            if hex_id not in all_hex_values:
                                all_hex_values[hex_id] = []
                            all_hex_values[hex_id].append(values)
                
                # Average values for hexagons that appear in multiple subtiles
                final_hex_values = {}
                for hex_id, values_list in all_hex_values.items():
                    final_hex_values[hex_id] = np.mean(values_list, axis=0).tolist()
                
                logger.info(f"  Completed tile {tile_name} with {len(final_hex_values)} hexagons")
                return {tile_name: final_hex_values}
                
        except Exception as e:
            logger.error(f"Error processing tile {tile_name}: {e}")
            return {}
    
    def process_tiles_parallel(self, tiles: List[Path], n_workers: int = 4) -> Dict[str, Dict]:
        """Process multiple tiles in parallel"""
        logger.info(f"Starting parallel processing with {n_workers} workers...")
        
        # Filter out already completed tiles
        remaining_tiles = [t for t in tiles if t.stem not in self.completed_tiles]
        logger.info(f"Processing {len(remaining_tiles)} remaining tiles (skipping {len(tiles)-len(remaining_tiles)} completed)")
        
        all_results = {}
        completed_count = len(self.completed_tiles)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_tile = {
                executor.submit(process_tile_worker, tile_path, self.config, 
                               self.h3_gdf, self.hex_lookup, self.hex_tree): tile_path 
                for tile_path in remaining_tiles
            }
            
            # Process completed tasks
            for future in as_completed(future_to_tile):
                tile_path = future_to_tile[future]
                try:
                    result = future.result()
                    if result:
                        all_results.update(result)
                        completed_count += 1
                        
                        # Update checkpoint every 10 tiles
                        if completed_count % 10 == 0:
                            self.completed_tiles.update(result.keys())
                            self.save_checkpoint()
                            logger.info(f"Progress: {completed_count}/{len(tiles)} tiles completed ({completed_count/len(tiles)*100:.1f}%)")
                            
                except Exception as e:
                    logger.error(f"Error in parallel processing for {tile_path}: {e}")
        
        return all_results
    
    def merge_parallel_results(self, tile_results: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Merge results from parallel processing, averaging overlapping hexagons"""
        logger.info("Merging parallel processing results...")
        
        all_hexagons = {}
        
        # Collect all hexagon values
        for tile_name, hex_data in tile_results.items():
            for hex_id, values in hex_data.items():
                if hex_id not in all_hexagons:
                    all_hexagons[hex_id] = []
                all_hexagons[hex_id].append(values)
        
        # Average overlapping hexagons
        final_hexagons = {}
        for hex_id, values_list in all_hexagons.items():
            if len(values_list) > 1:
                logger.debug(f"Averaging {len(values_list)} values for hexagon {hex_id}")
            final_hexagons[hex_id] = np.mean(values_list, axis=0)
        
        logger.info(f"Merged results: {len(final_hexagons)} unique hexagons")
        return final_hexagons
    
    def run(self, n_workers: int = 4):
        """Run the complete processing pipeline with spatial filtering and parallel processing"""
        start_time = time.time()
        
        # Get all TIFF files
        pattern = self.config['data']['pattern']
        all_tiff_files = sorted(self.source_dir.glob(pattern))
        logger.info(f"Found {len(all_tiff_files)} TIFF files total")
        
        # Filter to coastal tiles only
        coastal_tiles = self.filter_coastal_tiles(all_tiff_files)
        logger.info(f"Filtered to {len(coastal_tiles)} coastal tiles")
        
        # Process tiles in parallel
        n_workers = min(n_workers, multiprocessing.cpu_count(), len(coastal_tiles))
        logger.info(f"Using {n_workers} parallel workers")
        
        tile_results = self.process_tiles_parallel(coastal_tiles, n_workers)
        
        # Merge results from all tiles
        final_hex_values = self.merge_parallel_results(tile_results)
        
        # Convert to DataFrame
        final_data = []
        for hex_id, values in final_hex_values.items():
            # Get hex centroid from pre-computed GeoDataFrame
            if hex_id in self.hex_lookup:
                lon, lat = self.hex_lookup[hex_id]['centroid']
            else:
                # Fallback: get from H3 GeoDataFrame
                try:
                    hex_geom = self.h3_gdf.loc[hex_id].geometry
                    centroid = hex_geom.centroid
                    lon, lat = centroid.x, centroid.y
                except:
                    logger.warning(f"Could not find geometry for hex {hex_id}")
                    continue
            
            row = {
                'hex_id': hex_id,
                'latitude': lat,
                'longitude': lon,
                'h3_resolution': self.h3_resolution
            }
            
            # Add band values
            for i, val in enumerate(values):
                row[f'A{i:02d}'] = val
            
            final_data.append(row)
        
        final_df = pd.DataFrame(final_data)
        logger.info(f"Final dataset: {len(final_df)} hexagons with 64 bands")
        
        # Save final results
        output_parquet = self.output_dir / f"cascadia_coastal_forests_2021_res{self.h3_resolution}.parquet"
        final_df.to_parquet(output_parquet, compression='snappy')
        logger.info(f"Saved final results to {output_parquet}")
        
        elapsed = time.time() - start_time
        logger.info(f"Processing complete in {elapsed/3600:.2f} hours")
        logger.info(f"Processed {len(coastal_tiles)} tiles using {n_workers} workers")
        logger.info(f"Average processing rate: {len(coastal_tiles)/(elapsed/3600):.1f} tiles/hour")
        
        return final_df


# Worker function for parallel processing (needs to be at module level)
def process_tile_worker(tile_path: Path, config: dict, h3_gdf, hex_lookup: dict, hex_tree) -> Dict[str, Dict]:
    """Worker function for processing a single tile (module level for pickling)"""
    
    # Recreate processor instance for this worker
    processor = ModularTiffProcessor.__new__(ModularTiffProcessor)  # Create without __init__
    processor.config = config
    processor.h3_gdf = h3_gdf
    processor.hex_lookup = hex_lookup
    processor.hex_tree = hex_tree
    processor.subtile_size = config['processing'].get('subtile_size', 256)
    processor.min_pixels = config['processing']['min_pixels_per_hex']
    processor.completed_tiles = set()  # Worker doesn't need checkpoint state
    
    return processor.process_single_tile_worker(tile_path)


def main():
    """Main entry point"""
    import yaml
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run processor
    processor = ModularTiffProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()


