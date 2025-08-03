"""
Process AlphaEarth 2023 embedding tiles and convert to H3 resolution 10.

This script:
1. Loads AlphaEarth .tif tiles one by one (memory efficient)
2. Converts 10m x 10m pixel embeddings to H3 resolution 10 hexagons
3. Saves both raw tiles and H3 aggregated data as parquet files
"""

import os
import gc
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from rasterio.warp import calculate_default_transform, reproject, Resampling
import h3
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class AlphaEarthProcessor:
    """Process AlphaEarth embedding tiles to H3 hexagons."""
    
    def __init__(self, input_dir, output_dir, year=2023):
        """
        Initialize processor.
        
        Args:
            input_dir: Directory containing .tif files
            output_dir: Directory for output parquet files
            year: Year to process
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.year = year
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tiles'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'h3_aggregated'), exist_ok=True)
        
        # H3 resolution 10 corresponds to ~15m edge length
        self.h3_resolution = 10
        
    def get_tile_files(self):
        """Get list of tile files for the specified year."""
        pattern = os.path.join(self.input_dir, f"Netherlands_Embedding_{self.year}_Mosaic-*.tif")
        files = glob.glob(pattern)
        print(f"Found {len(files)} tiles for year {self.year}")
        return sorted(files)
    
    def process_single_tile(self, tile_path):
        """
        Process a single tile file.
        
        Args:
            tile_path: Path to .tif file
            
        Returns:
            GeoDataFrame with H3 aggregated embeddings
        """
        tile_name = os.path.basename(tile_path).replace('.tif', '')
        print(f"\nProcessing tile: {tile_name}")
        
        # Check if already processed
        h3_output_path = os.path.join(self.output_dir, 'h3_aggregated', f'{tile_name}_h3.parquet')
        if os.path.exists(h3_output_path):
            print(f"  Tile already processed, loading from cache")
            return gpd.read_parquet(h3_output_path)
        
        try:
            with rasterio.open(tile_path) as src:
                # Get metadata
                transform = src.transform
                crs = src.crs
                bounds = src.bounds
                
                # Read data - AlphaEarth has 64 bands (embedding dimensions)
                n_bands = src.count
                print(f"  Bands: {n_bands}, Shape: {src.height}x{src.width}")
                print(f"  CRS: {crs}")
                print(f"  Bounds: {bounds}")
                
                # Read all bands at once
                print("  Reading tile data...")
                data = src.read()  # Shape: (n_bands, height, width)
                
                # Get nodata value
                nodata = src.nodata
                
                # Create mesh of pixel coordinates
                rows, cols = np.meshgrid(range(src.height), range(src.width), indexing='ij')
                
                # Get geographic coordinates for all pixels
                xs, ys = xy(transform, rows.flatten(), cols.flatten())
                
                # Convert to lat/lon if needed
                if crs and crs.to_epsg() != 4326:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                    lons, lats = transformer.transform(xs, ys)
                else:
                    lons, lats = xs, ys
                
                # Reshape data for easier processing
                # From (bands, height, width) to (height*width, bands)
                data_reshaped = data.reshape(n_bands, -1).T
                
                # Create dictionary to accumulate embeddings by H3 cell
                h3_accumulator = {}
                
                print("  Converting to H3 cells...")
                for i in tqdm(range(len(lons)), desc="  Processing pixels", position=0, leave=True, ascii=True, ncols=80):
                    # Get pixel embeddings
                    pixel_embeddings = data_reshaped[i]
                    
                    # Skip nodata pixels
                    if nodata is not None and np.any(pixel_embeddings == nodata):
                        continue
                    
                    # Skip if all zeros (sometimes used as nodata)
                    if np.all(pixel_embeddings == 0):
                        continue
                    
                    # Get H3 index for this pixel
                    h3_index = h3.latlng_to_cell(lats[i], lons[i], self.h3_resolution)
                    
                    # Accumulate embeddings for this H3 cell
                    if h3_index not in h3_accumulator:
                        h3_accumulator[h3_index] = {
                            'embeddings': [],
                            'pixel_count': 0
                        }
                    
                    h3_accumulator[h3_index]['embeddings'].append(pixel_embeddings)
                    h3_accumulator[h3_index]['pixel_count'] += 1
                
                # Clear large arrays from memory
                del data
                del data_reshaped
                gc.collect()
                
            # Aggregate by H3 cell
            print(f"  Aggregating {len(h3_accumulator)} H3 cells...")
            if not h3_accumulator:
                print("  No valid data in tile")
                return None
            
            # Create list of aggregated results
            aggregated = []
            for h3_index, data in tqdm(h3_accumulator.items(), desc="  Creating H3 cells", position=0, leave=True, ascii=True, ncols=80):
                # Compute mean embeddings
                embeddings_stack = np.vstack(data['embeddings'])
                mean_embeddings = np.mean(embeddings_stack, axis=0)
                
                # Get H3 geometry
                boundary = h3.cell_to_boundary(h3_index)
                polygon = Polygon([(lon, lat) for lat, lon in boundary])
                
                # Store result
                result = {
                    'h3_index': h3_index,
                    'geometry': polygon,
                    'pixel_count': data['pixel_count']
                }
                
                # Add embedding dimensions
                for i, val in enumerate(mean_embeddings):
                    result[f'embed_{i}'] = val
                
                aggregated.append(result)
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(aggregated, crs='EPSG:4326')
            
            # Save to parquet
            print(f"  Saving {len(gdf)} H3 cells to parquet...")
            gdf.to_parquet(h3_output_path)
            
            # Also save tile metadata
            metadata = {
                'tile_name': tile_name,
                'n_h3_cells': len(gdf),
                'bounds': bounds,
                'original_shape': (src.height, src.width),
                'n_bands': n_bands
            }
            metadata_path = h3_output_path.replace('.parquet', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Completed: {len(gdf)} H3 cells")
            return gdf
            
        except Exception as e:
            print(f"  Error processing tile: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def stitch_all_tiles(self):
        """Stitch all processed tiles into a single GeoDataFrame."""
        print("\nStitching all tiles...")
        
        # Check if final file exists
        final_path = os.path.join(self.output_dir, f'netherlands_{self.year}_h3_res{self.h3_resolution}.parquet')
        if os.path.exists(final_path):
            print(f"Final stitched file already exists: {final_path}")
            return gpd.read_parquet(final_path)
        
        # Load all H3 aggregated tiles
        h3_files = glob.glob(os.path.join(self.output_dir, 'h3_aggregated', '*_h3.parquet'))
        print(f"Found {len(h3_files)} processed tiles to stitch")
        
        if not h3_files:
            print("No processed tiles found!")
            return None
        
        # Load and concatenate in batches
        batch_size = 10
        all_gdfs = []
        
        for i in range(0, len(h3_files), batch_size):
            batch_files = h3_files[i:i+batch_size]
            batch_gdfs = []
            
            for file in batch_files:
                try:
                    gdf = gpd.read_parquet(file)
                    batch_gdfs.append(gdf)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            
            if batch_gdfs:
                # Concatenate batch
                batch_combined = pd.concat(batch_gdfs, ignore_index=True)
                
                # Group by H3 index in case of overlaps
                if 'h3_index' in batch_combined.columns:
                    # Aggregate overlapping cells
                    embed_cols = [col for col in batch_combined.columns if col.startswith('embed_')]
                    
                    grouped = batch_combined.groupby('h3_index').agg({
                        'geometry': 'first',
                        'pixel_count': 'sum',
                        **{col: 'mean' for col in embed_cols}
                    }).reset_index()
                    
                    all_gdfs.append(grouped)
                else:
                    all_gdfs.append(batch_combined)
                
                # Clear memory
                del batch_gdfs
                del batch_combined
                gc.collect()
        
        # Final concatenation
        print("Performing final concatenation...")
        final_gdf = pd.concat(all_gdfs, ignore_index=True)
        
        # Final deduplication if needed
        if 'h3_index' in final_gdf.columns:
            embed_cols = [col for col in final_gdf.columns if col.startswith('embed_')]
            final_gdf = final_gdf.groupby('h3_index').agg({
                'geometry': 'first',
                'pixel_count': 'sum',
                **{col: 'mean' for col in embed_cols}
            }).reset_index()
        
        # Convert back to GeoDataFrame
        final_gdf = gpd.GeoDataFrame(final_gdf, crs='EPSG:4326', geometry='geometry')
        
        # Save final result
        print(f"Saving final stitched file with {len(final_gdf)} H3 cells...")
        final_gdf.to_parquet(final_path)
        
        # Save summary statistics
        stats = {
            'year': self.year,
            'h3_resolution': self.h3_resolution,
            'total_h3_cells': len(final_gdf),
            'total_pixels_processed': int(final_gdf['pixel_count'].sum()),
            'embedding_dimensions': len([col for col in final_gdf.columns if col.startswith('embed_')])
        }
        stats_path = final_path.replace('.parquet', '_stats.json')
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Complete! Saved to {final_path}")
        print(f"Statistics: {stats}")
        
        return final_gdf
    
    def run(self):
        """Run the full processing pipeline."""
        print(f"Starting AlphaEarth processing for year {self.year}")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Get tile files
        tile_files = self.get_tile_files()
        
        if not tile_files:
            print("No tiles found!")
            return None
        
        # Process each tile
        for tile_path in tqdm(tile_files, desc="Processing tiles", position=0, leave=True, ascii=True, ncols=80):
            self.process_single_tile(tile_path)
            
            # Force garbage collection after each tile
            gc.collect()
        
        # Stitch all tiles together
        final_gdf = self.stitch_all_tiles()
        
        return final_gdf


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = r"G:\My Drive\EarthEngine_Exports_Corrected"
    OUTPUT_DIR = r"C:\Users\Bert Berkers\PycharmProjects\UrbanRepML\data\embeddings_AlphaEarth\processed"
    YEAR = 2023
    
    # Create processor and run
    processor = AlphaEarthProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        year=YEAR
    )
    
    # Run processing
    final_result = processor.run()
    
    if final_result is not None:
        print(f"\nProcessing complete!")
        print(f"Final dataset shape: {final_result.shape}")
        print(f"Columns: {list(final_result.columns[:10])}...")