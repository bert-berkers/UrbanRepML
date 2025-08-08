"""
Export AlphaEarth embeddings for Cascadia region from Google Earth Engine.

This script exports AlphaEarth satellite embeddings for the Cascadia bioregion
(Northern California and Oregon) for years 2017-2024. The exports are tiled
and sent to Google Drive for local synchronization.

Usage:
    python export_cascadia_alphaearth.py --year 2023 --dry_run
    python export_cascadia_alphaearth.py --all_years

Note: This script requires Google Earth Engine Python API authentication.
"""

import ee
import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import logging
import os
import sys

# Add parent directory to path for config access
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Initialize Earth Engine
try:
    ee.Initialize(project='boreal-union-296021')
    print("Google Earth Engine initialized successfully with project boreal-union-296021")
except Exception as e:
    print(f"Failed to initialize Earth Engine: {e}")
    print("Please authenticate using: earthengine authenticate --project=boreal-union-296021")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Console only for now
    ]
)
logger = logging.getLogger(__name__)


class CascadiaAlphaEarthExporter:
    """Export AlphaEarth data for Cascadia region."""
    
    # California counties in Cascadia (Northern CA)
    CALIFORNIA_COUNTIES = [
        "Butte", "Colusa", "Del Norte", "Glenn", "Humboldt", "Lake",
        "Lassen", "Mendocino", "Modoc", "Nevada", "Plumas", "Shasta",
        "Sierra", "Siskiyou", "Tehama", "Trinity"
    ]
    
    # Oregon - all counties (we'll get the full state boundary)
    OREGON_STATE = "Oregon"
    
    # AlphaEarth collection name pattern
    ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    
    def __init__(self, export_folder: str = "AlphaEarth_Cascadia", 
                 tile_size: int = 3072,
                 scale: int = 10):
        """
        Initialize exporter.
        
        Args:
            export_folder: Google Drive folder for exports
            tile_size: Size of export tiles in pixels
            scale: Export scale in meters (10m for AlphaEarth)
        """
        self.export_folder = export_folder
        self.tile_size = tile_size
        self.scale = scale
        self.tasks = []
        
        logger.info(f"Initialized Cascadia AlphaEarth Exporter")
        logger.info(f"Export folder: {export_folder}")
        logger.info(f"Tile size: {tile_size}x{tile_size} pixels")
        logger.info(f"Scale: {scale}m")
        
    def get_cascadia_boundary(self) -> ee.Geometry:
        """
        Get Cascadia region boundary from county definitions.
        
        Returns:
            ee.Geometry: Cascadia region boundary
        """
        logger.info("Creating Cascadia boundary...")
        
        # Get US counties dataset
        counties = ee.FeatureCollection("TIGER/2018/Counties")
        
        # Filter California counties
        ca_counties = counties.filter(
            ee.Filter.And(
                ee.Filter.eq('STATEFP', '06'),  # California FIPS
                ee.Filter.inList('NAME', self.CALIFORNIA_COUNTIES)
            )
        )
        
        # Get all Oregon counties
        or_counties = counties.filter(ee.Filter.eq('STATEFP', '41'))  # Oregon FIPS
        
        # Combine both regions
        cascadia = ca_counties.merge(or_counties)
        
        # Get dissolved boundary
        boundary = cascadia.geometry().dissolve(1000)  # 1km tolerance
        
        # Get bounds for logging
        bounds = boundary.bounds().getInfo()['coordinates'][0]
        min_lon = min(coord[0] for coord in bounds)
        max_lon = max(coord[0] for coord in bounds)
        min_lat = min(coord[1] for coord in bounds)
        max_lat = max(coord[1] for coord in bounds)
        
        logger.info(f"Cascadia boundary created:")
        logger.info(f"  Bounds: [{min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f}]")
        logger.info(f"  CA Counties: {len(self.CALIFORNIA_COUNTIES)}")
        logger.info(f"  OR Counties: All (36)")
        
        return boundary
    
    def check_alphaearth_availability(self, year: int) -> bool:
        """
        Check if AlphaEarth data is available for a given year.
        
        Args:
            year: Year to check
            
        Returns:
            bool: True if data is available
        """
        collection_id = self.ALPHAEARTH_COLLECTION
        
        try:
            # Try to load the collection
            collection = ee.ImageCollection(collection_id)
            
            # Check if collection has any images
            count = collection.size()
            
            if count.getInfo() > 0:
                # Get first image to check bands
                first = ee.Image(collection.first())
                bands = first.bandNames().getInfo()
                
                logger.info(f"Year {year}: Available")
                logger.info(f"  Collection: {collection_id}")
                logger.info(f"  Images: {count.getInfo()}")
                logger.info(f"  Bands: {len(bands)} dimensions")
                
                return True
            else:
                logger.warning(f"Year {year}: Collection exists but is empty")
                return False
                
        except Exception as e:
            logger.warning(f"Year {year}: Not available - {e}")
            return False
    
    def create_export_tiles(self, boundary: ee.Geometry, year: int) -> List[Dict]:
        """
        Create tile definitions for export.
        
        Args:
            boundary: Region boundary
            year: Year to export
            
        Returns:
            List of tile definitions
        """
        logger.info(f"Creating export tiles for year {year}...")
        
        # Get boundary extent
        bounds = boundary.bounds()
        coords = bounds.coordinates().get(0).getInfo()
        
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)
        
        # Calculate tile size in degrees (approximate)
        # At ~42°N, 1 degree longitude ≈ 82.5 km
        # 10m resolution, 3072 pixels = 30.72 km
        tile_size_deg_lon = (self.tile_size * self.scale / 1000) / 82.5
        tile_size_deg_lat = (self.tile_size * self.scale / 1000) / 111.0
        
        tiles = []
        tile_id = 0
        
        lat = min_lat
        while lat < max_lat:
            lon = min_lon
            while lon < max_lon:
                # Create tile boundary
                tile_bounds = ee.Geometry.Rectangle([
                    lon, 
                    lat,
                    min(lon + tile_size_deg_lon, max_lon),
                    min(lat + tile_size_deg_lat, max_lat)
                ])
                
                # Check if tile intersects with Cascadia boundary
                intersection = tile_bounds.intersection(boundary, 100)
                
                # Only include tiles that actually overlap with region
                tiles.append({
                    'id': f"{year}_{tile_id:04d}",
                    'year': year,
                    'geometry': tile_bounds,
                    'intersection': intersection,
                    'bounds': [lon, lat, 
                              min(lon + tile_size_deg_lon, max_lon),
                              min(lat + tile_size_deg_lat, max_lat)]
                })
                
                tile_id += 1
                lon += tile_size_deg_lon
            lat += tile_size_deg_lat
        
        logger.info(f"Created {len(tiles)} tiles for year {year}")
        logger.info(f"  Tile size: ~{tile_size_deg_lon:.3f}° x {tile_size_deg_lat:.3f}°")
        
        return tiles
    
    def export_tile(self, tile: Dict, dry_run: bool = False) -> str:
        """
        Export a single tile to Google Drive.
        
        Args:
            tile: Tile definition
            dry_run: If True, don't actually start export
            
        Returns:
            Task ID
        """
        year = tile['year']
        tile_id = tile['id']
        
        # Load AlphaEarth image for the year
        collection_id = self.ALPHAEARTH_COLLECTION
        
        try:
            # Get the mosaic for the year
            image = ee.ImageCollection(collection_id).mosaic()
            
            # Clip to tile intersection with Cascadia
            clipped = image.clip(tile['intersection'])
            
            # Export parameters
            export_params = {
                'image': clipped,
                'description': f'Cascadia_AlphaEarth_{tile_id}',
                'folder': self.export_folder,
                'fileNamePrefix': f'Cascadia_AlphaEarth_{tile_id}',
                'scale': self.scale,
                'region': tile['geometry'],
                'fileFormat': 'GeoTIFF',
                'maxPixels': 1e9,
                'formatOptions': {
                    'cloudOptimized': True
                }
            }
            
            if not dry_run:
                # Start the export task
                task = ee.batch.Export.image.toDrive(**export_params)
                task.start()
                
                logger.info(f"Started export for tile {tile_id}")
                logger.info(f"  Bounds: {tile['bounds']}")
                
                return task.id
            else:
                logger.info(f"[DRY RUN] Would export tile {tile_id}")
                logger.info(f"  Bounds: {tile['bounds']}")
                return f"dry_run_{tile_id}"
                
        except Exception as e:
            logger.error(f"Failed to export tile {tile_id}: {e}")
            return None
    
    def export_year(self, year: int, dry_run: bool = False) -> List[str]:
        """
        Export all tiles for a given year.
        
        Args:
            year: Year to export
            dry_run: If True, don't actually start exports
            
        Returns:
            List of task IDs
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Exporting AlphaEarth data for year {year}")
        logger.info(f"{'='*60}")
        
        # Check availability
        if not self.check_alphaearth_availability(year):
            logger.error(f"AlphaEarth data not available for year {year}")
            return []
        
        # Get Cascadia boundary
        boundary = self.get_cascadia_boundary()
        
        # Create tiles
        tiles = self.create_export_tiles(boundary, year)
        
        # Export each tile
        task_ids = []
        for i, tile in enumerate(tiles):
            logger.info(f"\nProcessing tile {i+1}/{len(tiles)}")
            
            task_id = self.export_tile(tile, dry_run)
            if task_id:
                task_ids.append(task_id)
                self.tasks.append({
                    'year': year,
                    'tile_id': tile['id'],
                    'task_id': task_id,
                    'status': 'STARTED' if not dry_run else 'DRY_RUN'
                })
            
            # Small delay to avoid overwhelming EE
            if not dry_run:
                time.sleep(0.5)
        
        logger.info(f"\nYear {year} export summary:")
        logger.info(f"  Total tiles: {len(tiles)}")
        logger.info(f"  Tasks started: {len(task_ids)}")
        
        return task_ids
    
    def export_all_years(self, years: List[int] = None, dry_run: bool = False):
        """
        Export all available years.
        
        Args:
            years: List of years to export (default: 2017-2024)
            dry_run: If True, don't actually start exports
        """
        if years is None:
            years = list(range(2017, 2025))
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"Starting Cascadia AlphaEarth export for years: {years}")
        logger.info(f"{'#'*60}")
        
        all_task_ids = []
        
        for year in years:
            task_ids = self.export_year(year, dry_run)
            all_task_ids.extend(task_ids)
            
            # Longer delay between years
            if not dry_run and year != years[-1]:
                logger.info(f"Waiting 5 seconds before next year...")
                time.sleep(5)
        
        # Save task list
        self.save_task_list()
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"Export complete!")
        logger.info(f"  Years processed: {len(years)}")
        logger.info(f"  Total tasks: {len(all_task_ids)}")
        logger.info(f"  Task list saved to: export_tasks.json")
        logger.info(f"{'#'*60}")
        
        if not dry_run:
            logger.info("\nMonitor progress at:")
            logger.info("https://code.earthengine.google.com/tasks")
    
    def save_task_list(self):
        """Save task list to JSON file."""
        output_file = "../../logs/export_tasks.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'export_time': datetime.now().isoformat(),
                'export_folder': self.export_folder,
                'tile_size': self.tile_size,
                'scale': self.scale,
                'total_tasks': len(self.tasks),
                'tasks': self.tasks
            }, f, indent=2)
        
        logger.info(f"Task list saved to {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Export Cascadia AlphaEarth data from GEE")
    parser.add_argument('--year', type=int, help='Single year to export')
    parser.add_argument('--years', nargs='+', type=int, help='Multiple years to export')
    parser.add_argument('--all_years', action='store_true', help='Export all years (2017-2024)')
    parser.add_argument('--dry_run', action='store_true', help='Test run without starting exports')
    parser.add_argument('--export_folder', default='AlphaEarth_Cascadia', 
                       help='Google Drive folder name')
    parser.add_argument('--tile_size', type=int, default=3072, 
                       help='Tile size in pixels')
    parser.add_argument('--scale', type=int, default=10, 
                       help='Export scale in meters')
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = CascadiaAlphaEarthExporter(
        export_folder=args.export_folder,
        tile_size=args.tile_size,
        scale=args.scale
    )
    
    # Determine which years to export
    if args.all_years:
        years = list(range(2017, 2025))
    elif args.years:
        years = args.years
    elif args.year:
        years = [args.year]
    else:
        # Default: check availability for all years
        logger.info("Checking AlphaEarth availability for years 2017-2024...")
        for year in range(2017, 2025):
            exporter.check_alphaearth_availability(year)
        return
    
    # Export selected years
    exporter.export_all_years(years, dry_run=args.dry_run)


if __name__ == "__main__":
    main()