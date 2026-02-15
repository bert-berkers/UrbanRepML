#!/usr/bin/env python3
"""
Setup Pearl River Delta study area with proper SRAI-based H3 regionalization.

This script follows the CLAUDE.md principles:
- Uses SRAI for ALL H3 operations (never h3 directly)
- Creates proper study area structure per documentation
- Generates regions_gdf required for AlphaEarth processing

The Pearl River Delta (PRD) is one of the world's largest urban agglomerations,
including Guangzhou, Shenzhen, Hong Kong, Macau, and surrounding cities.
"""

import geopandas as gpd
from shapely.geometry import box, Polygon
from srai.regionalizers import H3Regionalizer  # ALWAYS use SRAI, never h3 directly!
from srai.neighbourhoods import H3Neighbourhood
from pathlib import Path
import json
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_prd_boundary() -> gpd.GeoDataFrame:
    """
    Create Pearl River Delta boundary polygon.
    
    The PRD metropolitan area includes:
    - Core cities: Guangzhou, Shenzhen, Hong Kong, Macau
    - Major cities: Dongguan, Foshan, Zhongshan, Zhuhai, Jiangmen
    - Parts of: Huizhou, Zhaoqing
    
    Returns:
        GeoDataFrame with PRD boundary
    """
    logger.info("Creating Pearl River Delta boundary...")
    
    # Define PRD boundary with more detail (simplified polygon covering the main urban area)
    # These coordinates roughly encompass the Greater Bay Area
    prd_coords = [
        (112.5, 21.5),   # Southwest corner (west of Macau)
        (112.5, 23.5),   # Northwest corner (north of Guangzhou)
        (114.5, 23.5),   # Northeast corner (north of Shenzhen)
        (114.5, 22.1),   # Southeast corner (east of Hong Kong)
        (114.3, 21.5),   # South (covering southern islands)
        (112.5, 21.5)    # Close polygon
    ]
    
    prd_polygon = Polygon(prd_coords)
    
    # Create GeoDataFrame
    area_gdf = gpd.GeoDataFrame(
        {
            'name': ['Pearl River Delta'],
            'name_zh': ['珠江三角洲'],
            'region': ['Greater Bay Area'],
            'country': ['China'],
            'area_km2': [prd_polygon.area * 111 * 111],  # Approximate conversion
            'population_millions': [70]  # Approximate 2023 population
        },
        geometry=[prd_polygon],
        crs='EPSG:4326'
    )
    
    logger.info(f"Created PRD boundary: {prd_polygon.bounds}")
    logger.info(f"Approximate area: {area_gdf['area_km2'].iloc[0]:.0f} km²")
    
    return area_gdf

def setup_study_area_structure(study_area_name: str = 'pearl_river_delta') -> Path:
    """
    Create the standard study area directory structure per CLAUDE.md.
    
    Args:
        study_area_name: Name of the study area
        
    Returns:
        Path to study area root directory
    """
    logger.info(f"Setting up study area structure for {study_area_name}...")
    
    # Base path
    study_area_path = Path(f'study_areas/{study_area_name}')
    
    # Create all required subdirectories per documentation
    directories = [
        'area_gdf',           # Study area boundary
        'regions_gdf',        # H3 tessellation (via SRAI!)
        'embeddings/alphaearth',  # AlphaEarth embeddings
        'embeddings/poi',     # POI embeddings (future)
        'embeddings/roads',   # Roads embeddings (future)
        'embeddings/gtfs',    # Transit embeddings (future)
        'stage2_fusion',    # Fused results [old 2024]
        'plots',              # Visualizations
        'cache'               # Temporary files
    ]
    
    for dir_name in directories:
        dir_path = study_area_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Created: {dir_path}")
    
    # Also create data/boundaries structure for Earth Engine
    boundaries_path = Path(f'data/boundaries/{study_area_name}')
    boundaries_path.mkdir(parents=True, exist_ok=True)
    
    return study_area_path

def create_h3_regions_with_srai(area_gdf: gpd.GeoDataFrame, resolution: int = 10) -> gpd.GeoDataFrame:
    """
    Create H3 hexagon tessellation using SRAI (NOT h3 directly!).
    
    This is CRITICAL for the processing pipeline - all AlphaEarth processing
    requires pre-computed regions_gdf to map pixels to hexagons.
    
    Args:
        area_gdf: Study area boundary GeoDataFrame
        resolution: H3 resolution (10 as specified for high detail)
        
    Returns:
        GeoDataFrame with H3 hexagons
    """
    logger.info(f"Creating H3 regions at resolution {resolution} using SRAI...")
    logger.info("NOTE: Using SRAI's H3Regionalizer per CLAUDE.md - never h3 directly!")
    
    # Initialize SRAI H3Regionalizer
    regionalizer = H3Regionalizer(resolution=resolution)
    
    # Transform study area to H3 hexagons
    regions_gdf = regionalizer.transform(area_gdf)
    
    # Add metadata
    regions_gdf['h3_resolution'] = resolution
    regions_gdf['study_area'] = 'pearl_river_delta'
    
    logger.info(f"✓ Created {len(regions_gdf)} H3 hexagons at resolution {resolution}")
    
    # Calculate statistics
    hex_area_km2 = regions_gdf.geometry.iloc[0].area * 111 * 111  # Approximate
    logger.info(f"  Average hexagon area: ~{hex_area_km2:.3f} km²")
    logger.info(f"  Total hexagons: {len(regions_gdf):,}")
    
    return regions_gdf

def save_study_area_files(
    area_gdf: gpd.GeoDataFrame,
    regions_gdf: gpd.GeoDataFrame,
    study_area_path: Path
) -> None:
    """
    Save all study area files in the correct formats and locations.
    
    Args:
        area_gdf: Study area boundary
        regions_gdf: H3 hexagon tessellation
        study_area_path: Root path for study area
    """
    logger.info("Saving study area files...")
    
    # 1. Save area boundary
    area_file = study_area_path / 'area_gdf' / 'pearl_river_delta_boundary.geojson'
    area_gdf.to_file(area_file, driver='GeoJSON')
    logger.info(f"  ✓ Saved area boundary: {area_file}")
    
    # 2. Save H3 regions (critical for processing!)
    regions_file = study_area_path / 'regions_gdf' / 'h3_res8.parquet'
    regions_gdf.to_parquet(regions_file)
    logger.info(f"  ✓ Saved H3 regions: {regions_file}")
    
    # Also save as GeoJSON for visualization
    regions_geojson = study_area_path / 'regions_gdf' / 'h3_res8_sample.geojson'
    # Save only first 1000 hexagons for visualization (full set is too large for GeoJSON)
    regions_gdf.head(1000).to_file(regions_geojson, driver='GeoJSON')
    logger.info(f"  ✓ Saved sample regions for viz: {regions_geojson}")
    
    # 3. Save boundary for Earth Engine
    ee_boundary_path = Path('data/boundaries/pearl_river_delta')
    ee_boundary_path.mkdir(parents=True, exist_ok=True)
    
    # Save as states file (for compatibility with Earth Engine script)
    ee_states_file = ee_boundary_path / 'pearl_river_delta_states.geojson'
    area_gdf.to_file(ee_states_file, driver='GeoJSON')
    logger.info(f"  ✓ Saved Earth Engine boundary: {ee_states_file}")
    
    # 4. Create metadata file
    metadata = {
        'study_area': 'pearl_river_delta',
        'h3_resolution': 8,
        'total_hexagons': len(regions_gdf),
        'boundary_bounds': list(area_gdf.total_bounds),
        'area_km2': float(area_gdf['area_km2'].iloc[0]),
        'crs': 'EPSG:4326',
        'created_with': 'SRAI H3Regionalizer',
        'note': 'All H3 operations use SRAI per CLAUDE.md - never h3 directly'
    }
    
    metadata_file = study_area_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Saved metadata: {metadata_file}")

def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("PEARL RIVER DELTA STUDY AREA SETUP")
    logger.info("="*60)
    
    try:
        # 1. Create study area structure
        study_area_path = setup_study_area_structure('pearl_river_delta')
        
        # 2. Create PRD boundary
        area_gdf = create_prd_boundary()
        
        # 3. Create H3 regions using SRAI (CRITICAL!)
        regions_gdf = create_h3_regions_with_srai(area_gdf, resolution=8)
        
        # 4. Save all files
        save_study_area_files(area_gdf, regions_gdf, study_area_path)
        
        # 5. Print summary
        logger.info("="*60)
        logger.info("✅ SETUP COMPLETE!")
        logger.info("="*60)
        logger.info(f"Study area: Pearl River Delta")
        logger.info(f"H3 resolution: 8")
        logger.info(f"Total hexagons: {len(regions_gdf):,}")
        logger.info(f"Area: ~{area_gdf['area_km2'].iloc[0]:.0f} km²")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Ensure Earth Engine credentials are set in keys/.env")
        logger.info("2. Run: python scripts/alphaearth_earthengine_retrieval/fetch_alphaearth_embeddings_tiled.py \\")
        logger.info("        --study-area pearl_river_delta --year 2023")
        logger.info("3. Download tiles from Google Drive")
        logger.info("4. Process with AlphaEarth processor using the regions_gdf")
        logger.info("")
        logger.info("Files created:")
        logger.info(f"  - {study_area_path}/area_gdf/pearl_river_delta_boundary.geojson")
        logger.info(f"  - {study_area_path}/regions_gdf/h3_res8.parquet")
        logger.info(f"  - data/boundaries/pearl_river_delta/pearl_river_delta_states.geojson")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())