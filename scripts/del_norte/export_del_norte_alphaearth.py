#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Google Earth Engine Export Script for Del Norte County AlphaEarth Embeddings

This script exports AlphaEarth satellite embeddings for Del Norte County, California.
Del Norte is the northwesternmost county in California, part of the Cascadia bioregion.

Area: ~3,180 km² (much smaller than full Cascadia's 421,000 km²)
Expected data size: ~3GB per year at 10m resolution with 64 bands

Workflow:
1. Run this script to start export tasks in Google Earth Engine
2. Monitor tasks at: https://code.earthengine.google.com/tasks
3. Once complete, download from Google Drive
4. Process with H3 aggregation scripts

Based on successful Netherlands export pipeline.
"""

import ee
import time
from datetime import datetime
import argparse
import sys

# Configuration
GCLOUD_PROJECT_ID = 'boreal-union-296021'
GDRIVE_FOLDER = 'AlphaEarth_DelNorte'
ALPHAEARTH_COLLECTION = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'

# Del Norte County details
DEL_NORTE_FIPS = '06015'  # California (06) + Del Norte (015)
DEL_NORTE_NAME = 'Del Norte'


def initialize_earth_engine():
    """Initialize Google Earth Engine with project."""
    try:
        ee.Initialize(project=GCLOUD_PROJECT_ID)
        print(f"[OK] Google Earth Engine initialized with project: {GCLOUD_PROJECT_ID}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize Earth Engine: {e}")
        print("\nTo authenticate, run:")
        print(f"  earthengine authenticate --project={GCLOUD_PROJECT_ID}")
        return False


def get_del_norte_boundary():
    """
    Get Del Norte County boundary from US Census TIGER dataset.
    
    Returns:
        ee.Geometry: Del Norte County boundary
    """
    print("\nLoading Del Norte County boundary...")
    
    # Use TIGER/2018/Counties dataset (consistent with GEO-INFER)
    counties = ee.FeatureCollection("TIGER/2018/Counties")
    
    # Filter for Del Norte County, California
    del_norte = counties.filter(
        ee.Filter.And(
            ee.Filter.eq('STATEFP', '06'),  # California
            ee.Filter.eq('NAME', 'Del Norte')
        )
    )
    
    # Get the geometry
    boundary = del_norte.geometry()
    
    # Get bounds for display
    bounds_info = boundary.bounds().getInfo()['coordinates'][0]
    min_lon = min(coord[0] for coord in bounds_info)
    max_lon = max(coord[0] for coord in bounds_info)
    min_lat = min(coord[1] for coord in bounds_info)
    max_lat = max(coord[1] for coord in bounds_info)
    
    print(f"[OK] Del Norte County boundary loaded")
    print(f"  Bounds: [{min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}]")
    print(f"  Approximate area: 3,180 km^2")
    
    return boundary


def check_alphaearth_availability(year):
    """
    Check if AlphaEarth data is available for a given year.
    
    Args:
        year: Year to check (int)
        
    Returns:
        bool: True if data is available
    """
    try:
        collection = ee.ImageCollection(ALPHAEARTH_COLLECTION)
        
        # Filter for the year
        yearly_collection = collection.filter(
            ee.Filter.date(
                ee.Date.fromYMD(year, 1, 1),
                ee.Date.fromYMD(year, 12, 31)
            )
        )
        
        # Check if any images exist
        count = yearly_collection.size().getInfo()
        
        if count > 0:
            # Get band information from first image
            first_image = ee.Image(yearly_collection.first())
            bands = first_image.bandNames().getInfo()
            
            print(f"  Year {year}: [OK] Available ({count} images, {len(bands)} bands)")
            return True
        else:
            print(f"  Year {year}: [X] No data available")
            return False
            
    except Exception as e:
        print(f"  Year {year}: [X] Error checking availability: {e}")
        return False


def export_year(year, boundary, dry_run=False):
    """
    Export AlphaEarth data for Del Norte County for a specific year.
    
    Args:
        year: Year to export (int)
        boundary: County boundary (ee.Geometry)
        dry_run: If True, only show what would be exported
        
    Returns:
        dict: Export task information
    """
    print(f"\n{'='*60}")
    print(f"Processing Year: {year}")
    print(f"{'='*60}")
    
    # Load and filter the collection
    collection = ee.ImageCollection(ALPHAEARTH_COLLECTION)
    
    # Filter by date and region
    filtered = collection.filter(
        ee.Filter.date(
            ee.Date.fromYMD(year, 1, 1),
            ee.Date.fromYMD(year, 12, 31)
        )
    ).filterBounds(boundary)
    
    # Check collection size
    collection_size = filtered.size().getInfo()
    print(f"Found {collection_size} image tiles for year {year}")
    
    if collection_size == 0:
        print(f"No images found for year {year}. Skipping.")
        return None
    
    # Create mosaic and clip to county boundary
    print("Creating mosaic from tiles...")
    mosaic = filtered.mosaic()
    final_image = mosaic.clip(boundary)
    
    # Get image properties
    band_names = final_image.bandNames().getInfo()
    print(f"Image has {len(band_names)} bands")
    print(f"First 5 bands: {band_names[:5]}")
    
    # Define export parameters
    export_name = f'DelNorte_AlphaEarth_{year}'
    export_params = {
        'image': final_image,
        'description': export_name,
        'folder': GDRIVE_FOLDER,
        'fileNamePrefix': export_name,
        'scale': 10,  # 10 meters (native AlphaEarth resolution)
        'region': boundary,
        'fileFormat': 'GeoTIFF',
        'maxPixels': 1e10,  # Increased for safety
        'formatOptions': {
            'cloudOptimized': True  # Create Cloud-Optimized GeoTIFF
        }
    }
    
    # Calculate approximate file size
    # Del Norte: ~3,180 km^2 = 3.18e9 m^2
    # At 10m resolution: 3.18e9 / 100 = 3.18e7 pixels
    # With 64 bands * 4 bytes/band: 3.18e7 * 64 * 4 = 8.14 GB (uncompressed)
    # With compression: ~2-3 GB expected
    pixels_estimate = 3.18e7
    size_estimate_gb = (pixels_estimate * 64 * 4) / 1e9
    print(f"Estimated size: ~{size_estimate_gb:.1f} GB (uncompressed)")
    print(f"Expected compressed size: ~{size_estimate_gb/3:.1f} GB")
    
    if dry_run:
        print("\n[DRY RUN] Would export with parameters:")
        print(f"  Name: {export_name}")
        print(f"  Folder: {GDRIVE_FOLDER}")
        print(f"  Scale: 10m")
        print(f"  Format: Cloud-Optimized GeoTIFF")
        return export_params
    
    # Start the export task
    print("\nStarting export task...")
    task = ee.batch.Export.image.toDrive(**export_params)
    task.start()
    
    print(f"[OK] Export task started: {export_name}")
    print(f"  Monitor at: https://code.earthengine.google.com/tasks")
    print(f"  Task ID: {task.id}")
    
    return {
        'year': year,
        'task_id': task.id,
        'name': export_name,
        'status': 'STARTED',
        'size_estimate_gb': size_estimate_gb/3
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Export AlphaEarth embeddings for Del Norte County'
    )
    parser.add_argument(
        '--years', 
        nargs='+', 
        type=int,
        default=[2020, 2021, 2022, 2023],
        help='Years to export (default: 2020-2023)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be exported without starting tasks'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check data availability, do not export'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Del Norte County AlphaEarth Export Script")
    print("="*60)
    print(f"Target years: {args.years}")
    print(f"Google Drive folder: {GDRIVE_FOLDER}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'EXPORT'}")
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        return 1
    
    # Get Del Norte County boundary
    boundary = get_del_norte_boundary()
    
    # Check data availability
    print("\nChecking AlphaEarth data availability...")
    available_years = []
    for year in args.years:
        if check_alphaearth_availability(year):
            available_years.append(year)
    
    if not available_years:
        print("\n[X] No data available for any requested years")
        return 1
    
    print(f"\n[OK] Data available for years: {available_years}")
    
    if args.check_only:
        print("\nCheck-only mode. Exiting without export.")
        return 0
    
    # Export each year
    tasks = []
    total_size = 0
    
    for year in available_years:
        task_info = export_year(year, boundary, dry_run=args.dry_run)
        if task_info:
            tasks.append(task_info)
            total_size += task_info.get('size_estimate_gb', 0)
    
    # Summary
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"County: Del Norte, California")
    print(f"Years processed: {len(tasks)}")
    print(f"Total estimated size: ~{total_size:.1f} GB")
    
    if not args.dry_run and tasks:
        print(f"\n[OK] {len(tasks)} export tasks started")
        print("\nNext steps:")
        print("1. Monitor progress at: https://code.earthengine.google.com/tasks")
        print(f"2. Files will appear in Google Drive folder: {GDRIVE_FOLDER}")
        print("3. Download completed files to local machine")
        print("4. Process with H3 aggregation script")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())