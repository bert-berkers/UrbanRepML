"""
Enhanced AlphaEarth fetcher with tiled export capability.

This script addresses the integration gap between Earth Engine exports and the existing
AlphaEarth processing pipeline by providing tiled exports that match the expected
naming conventions and coordinate systems.

Key improvements:
- Automatic tiling for large study areas
- Naming conventions that match existing processors
- Tile boundary metadata for seamless stitching
- Overlap handling for processing continuity
- Integration with existing H3 processing pipeline

Prerequisites:
1. Google Cloud project with Earth Engine API enabled
2. Service account with appropriate permissions  
3. .env file in keys/ directory with credentials
4. Study area boundary files in data/boundaries/

Example usage:
    python scripts/alphaearth_earthengine_retrieval/fetch_alphaearth_embeddings_tiled.py \\
        --study-area cascadia_oldremove --year 2021 --tile-size-km 50 --overlap-km 5
"""

import argparse
import logging
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import ee
import geopandas as gpd
from dotenv import load_dotenv
from shapely.geometry import box, Polygon
from shapely.ops import transform
import pyproj
from functools import partial

# Configuration
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

ALPHAEARTH_COLLECTION_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

# Study area naming conventions (matches existing processors)
NAMING_CONVENTIONS = {
    'cascadia_oldremove': 'Cascadia_AlphaEarth_{year}_{tile_id}.tif',
    'netherlands': 'Netherlands_Embedding_{year}_{tile_id}.tif',
    'pearl_river_delta': 'PRD_AlphaEarth_{year}_{tile_id}.tif',
    'default': '{study_area}_AlphaEarth_{year}_{tile_id}.tif'
}


class TiledAlphaEarthExporter:
    """Handles tiled export of AlphaEarth data with proper integration."""
    
    def __init__(self, study_area: str, year: int, tile_size_km: float, 
                 overlap_km: float, scale: int, output_folder: str):
        self.study_area = study_area
        self.year = year
        self.tile_size_km = tile_size_km
        self.overlap_km = overlap_km
        self.scale = scale
        self.output_folder = output_folder
        
        # Initialize Earth Engine
        self.initialize_ee()
        
        # Load study area geometry
        self.geometry = self.load_study_area_geometry()
        
        # Create tile grid
        self.tiles = self.create_tile_grid()
        
        logger.info(f"Initialized exporter for {study_area}: {len(self.tiles)} tiles")
    
    def initialize_ee(self):
        """Initialize Google Earth Engine with service account."""
        logger.info("Initializing Earth Engine...")
        load_dotenv('keys/.env')
        try:
            import os
            project_id = os.getenv('GEE_PROJECT_ID')
            if not project_id:
                raise ValueError("GEE_PROJECT_ID not found in .env file")
            
            ee.Initialize(project=project_id)
            logger.info(f"Earth Engine initialized successfully with project: {project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            logger.error("Please ensure your service account credentials are properly configured")
            raise
    
    def load_study_area_geometry(self) -> ee.Geometry:
        """Load study area boundary and convert to Earth Engine geometry."""
        boundary_path = Path(f"data/boundaries/{self.study_area}/{self.study_area}_states.geojson")
        
        if not boundary_path.exists():
            # Try alternative paths
            alt_paths = [
                Path(f"study_areas/{self.study_area}/area_gdf/{self.study_area}_boundary.geojson"),
                Path(f"data/study_areas/{self.study_area}/area_gdf/boundary.geojson")
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    boundary_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Boundary file not found for {self.study_area}")
        
        logger.info(f"Loading boundary from: {boundary_path}")
        gdf = gpd.read_file(boundary_path)
        gdf_dissolved = gdf.dissolve()
        geom = gdf_dissolved.geometry.iloc[0]
        
        # Convert to Earth Engine geometry
        ee_geom = ee.Geometry(geom.__geo_interface__)
        logger.info("Study area geometry loaded and converted.")
        return ee_geom
    
    def create_tile_grid(self) -> List[Dict]:
        """Create a grid of tiles covering the study area with overlap."""
        logger.info(f"Creating tile grid: {self.tile_size_km}km tiles with {self.overlap_km}km overlap")
        
        # Get bounds of study area in WGS84
        gdf = gpd.read_file(f"data/boundaries/{self.study_area}/{self.study_area}_states.geojson")
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        
        # Convert km to degrees (approximate at study area center)
        center_lat = (bounds[1] + bounds[3]) / 2
        km_to_deg_lat = 1.0 / 111.0  # 1 degree ≈ 111 km
        km_to_deg_lon = 1.0 / (111.0 * math.cos(math.radians(center_lat)))
        
        tile_size_deg = self.tile_size_km * km_to_deg_lat
        overlap_deg = self.overlap_km * km_to_deg_lat
        step_size_deg = tile_size_deg - overlap_deg
        
        tiles = []
        tile_id = 0
        
        # Create grid from southwest to northeast
        y = bounds[1] - overlap_deg  # Start south of bounds
        while y < bounds[3] + overlap_deg:  # Until north of bounds
            x = bounds[0] - overlap_deg  # Start west of bounds
            while x < bounds[2] + overlap_deg:  # Until east of bounds
                
                # Create tile bounds
                tile_bounds = [
                    x, y, 
                    x + tile_size_deg, 
                    y + tile_size_deg
                ]
                
                # Create tile geometry
                tile_geom = box(*tile_bounds)
                
                # Check if tile intersects study area
                study_area_shapely = gdf.unary_union
                if tile_geom.intersects(study_area_shapely):
                    
                    # Generate tile ID (4-digit zero-padded)
                    tile_id_str = f"{tile_id:04d}"
                    
                    # Create tile info
                    tile_info = {
                        'id': tile_id_str,
                        'bounds': tile_bounds,
                        'geometry': tile_geom,
                        'ee_geometry': ee.Geometry.Rectangle(tile_bounds),
                        'center_lat': y + tile_size_deg/2,
                        'center_lon': x + tile_size_deg/2
                    }
                    
                    tiles.append(tile_info)
                    tile_id += 1
                
                x += step_size_deg
            y += step_size_deg
        
        logger.info(f"Created {len(tiles)} tiles covering study area")
        return tiles
    
    def get_tile_filename(self, tile_info: Dict) -> str:
        """Generate filename following existing naming conventions."""
        naming_pattern = NAMING_CONVENTIONS.get(
            self.study_area.lower(), 
            NAMING_CONVENTIONS['default']
        )
        
        return naming_pattern.format(
            study_area=self.study_area.title(),
            year=self.year,
            tile_id=tile_info['id']
        )
    
    def export_tile(self, tile_info: Dict) -> ee.batch.Task:
        """Export a single tile with proper naming and metadata."""
        filename = self.get_tile_filename(tile_info)
        
        logger.info(f"Exporting tile {tile_info['id']}: {filename}")
        
        # Filter and prepare image for this tile
        start_date = f"{self.year}-01-01"
        end_date = f"{self.year}-12-31"
        
        # Intersect tile with study area for precise clipping
        tile_study_intersection = tile_info['ee_geometry'].intersection(self.geometry)
        
        image = (
            ee.ImageCollection(ALPHAEARTH_COLLECTION_ID)
            .filterDate(start_date, end_date)
            .filterBounds(tile_study_intersection)
            .mosaic()
            .clip(tile_study_intersection)
        )
        
        # Configure export task
        task_config = {
            "image": image,
            "description": f"{self.study_area}_{self.year}_tile_{tile_info['id']}",
            "folder": self.output_folder,
            "fileNamePrefix": filename.replace('.tif', ''),  # EE adds .tif automatically
            "scale": self.scale,
            "region": tile_study_intersection,
            "fileFormat": "GeoTIFF",
            "maxPixels": 1e13,
            "crs": "EPSG:4326",  # Ensure consistent projection
        }
        
        task = ee.batch.Export.image.toDrive(**task_config)
        task.start()
        
        logger.info(f"Started export task: {task.id}")
        return task
    
    def export_all_tiles(self) -> List[ee.batch.Task]:
        """Export all tiles and return list of tasks."""
        logger.info(f"Starting export of {len(self.tiles)} tiles...")
        
        tasks = []
        for i, tile_info in enumerate(self.tiles):
            logger.info(f"Exporting tile {i+1}/{len(self.tiles)}")
            
            try:
                task = self.export_tile(tile_info)
                tasks.append(task)
                
                # Add small delay to avoid overwhelming Earth Engine
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to export tile {tile_info['id']}: {e}")
                continue
        
        logger.info(f"Successfully started {len(tasks)} export tasks")
        return tasks
    
    def save_tile_metadata(self, output_path: Optional[Path] = None) -> Path:
        """Save tile metadata for processing pipeline integration."""
        if output_path is None:
            output_path = Path(f"data/study_areas/{self.study_area}/tiles_metadata_{self.year}.json")
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata = {
            'study_area': self.study_area,
            'year': self.year,
            'tile_size_km': self.tile_size_km,
            'overlap_km': self.overlap_km,
            'scale_meters': self.scale,
            'total_tiles': len(self.tiles),
            'naming_convention': NAMING_CONVENTIONS.get(
                self.study_area.lower(), 
                NAMING_CONVENTIONS['default']
            ),
            'tiles': []
        }
        
        # Add tile information
        for tile_info in self.tiles:
            tile_metadata = {
                'id': tile_info['id'],
                'filename': self.get_tile_filename(tile_info),
                'bounds_wgs84': tile_info['bounds'],
                'center_lat': tile_info['center_lat'],
                'center_lon': tile_info['center_lon'],
                'overlap_km': self.overlap_km
            }
            metadata['tiles'].append(tile_metadata)
        
        # Save metadata
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Tile metadata saved to: {output_path}")
        return output_path
    
    def monitor_all_tasks(self, tasks: List[ee.batch.Task]):
        """Monitor all export tasks until completion."""
        logger.info(f"Monitoring {len(tasks)} export tasks...")
        
        active_tasks = tasks.copy()
        completed = 0
        failed = 0
        
        while active_tasks:
            time.sleep(60)  # Check every minute
            
            for task in active_tasks.copy():
                status = task.status()
                
                if not task.active():
                    active_tasks.remove(task)
                    
                    if status["state"] == "COMPLETED":
                        completed += 1
                        logger.info(f"✓ Task completed: {status['description']} ({completed}/{len(tasks)})")
                    else:
                        failed += 1
                        logger.error(f"✗ Task failed: {status['description']} - {status.get('error_message', 'Unknown error')}")
            
            if active_tasks:
                logger.info(f"Still monitoring {len(active_tasks)} tasks... (Completed: {completed}, Failed: {failed})")
        
        logger.info(f"All tasks finished. Completed: {completed}, Failed: {failed}")
        
        if failed > 0:
            logger.warning(f"{failed} tasks failed. Check Earth Engine console for details.")


def get_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch AlphaEarth embeddings with tiled export for processing integration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--study-area", type=str, required=True,
                       help="Name of study area (e.g., 'netherlands', 'cascadia_oldremove')")
    parser.add_argument("--year", type=int, default=2022,
                       help="Year to filter AlphaEarth data")
    parser.add_argument("--tile-size-km", type=float, default=50,
                       help="Size of each tile in kilometers")
    parser.add_argument("--overlap-km", type=float, default=5,
                       help="Overlap between tiles in kilometers")
    parser.add_argument("--scale", type=int, default=10,
                       help="Resolution for Earth Engine export in meters")
    parser.add_argument("--output-folder", type=str, default="UrbanRepML_Tiled_Exports",
                       help="Google Drive folder for exports")
    parser.add_argument("--export-only", action="store_true",
                       help="Only start exports, don't monitor completion")
    parser.add_argument("--save-metadata", action="store_true", default=True,
                       help="Save tile metadata for processing integration")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = get_arguments()
    
    try:
        # Initialize tiled exporter
        exporter = TiledAlphaEarthExporter(
            study_area=args.study_area,
            year=args.year,
            tile_size_km=args.tile_size_km,
            overlap_km=args.overlap_km,
            scale=args.scale,
            output_folder=args.output_folder
        )
        
        # Save tile metadata
        if args.save_metadata:
            exporter.save_tile_metadata()
        
        # Export all tiles
        tasks = exporter.export_all_tiles()
        
        if not args.export_only and tasks:
            # Monitor task completion
            exporter.monitor_all_tasks(tasks)
        else:
            logger.info("Export tasks started. Check Google Drive and Earth Engine console for progress.")
        
        logger.info("Tiled AlphaEarth export complete!")
        logger.info(f"Files will be named following pattern: {exporter.get_tile_filename({'id': 'XXXX'})}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())