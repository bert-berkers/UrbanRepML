#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Google Earth Engine Export-to-Drive Script for AlphaEarth Data

This script EXPORTS data from Google Earth Engine to Google Drive.
It does NOT download data to your local machine!

The workflow is:
1. Run this script to start export tasks in Google Earth Engine
2. Monitor tasks in the Earth Engine Code Editor Tasks tab
3. Once complete, sync the Google Drive folder to your local machine
4. Run process_alphaearth_2023.py on the synced files
"""

import ee
import time

# --- Configuration ---
# 1. Your Google Cloud Project ID
gcloud_project_id = 'boreal-union-296021'

# 2. Define the years you want to export
years_to_export = [2020, 2021, 2022, 2023]

# 3. Define your Google Drive folder for the exports
gdrive_folder = 'EarthEngine_Exports_Corrected'

print("=== CORRECTED ALPHAEARTH EXPORT SCRIPT ===")
print("Using proper mosaicking approach recommended by Gemini")

# --- Authenticate and Initialize GEE ---
try:
    ee.Initialize(project=gcloud_project_id)
except Exception as e:
    print("Authentication required. Please follow the prompts.")
    ee.Authenticate()
    ee.Initialize(project=gcloud_project_id)

print("Google Earth Engine Initialized Successfully.")

# --- Define Area of Interest (Netherlands) ---
print("Loading Netherlands boundary...")
countries = ee.FeatureCollection('FAO/GAUL/2015/level0')
netherlands_aoi = countries.filter(ee.Filter.eq('ADM0_NAME', 'Netherlands')).geometry()

print("Netherlands boundary loaded successfully.")

# --- Load the Satellite Embedding Collection ---
print("Loading AlphaEarth embedding collection...")
embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

# --- Process Each Year ---
for year in years_to_export:
    print(f"\n{'='*60}")
    print(f"PROCESSING YEAR: {year}")
    print(f"{'='*60}")
    
    try:
        # 1. Filter the Collection
        print(f"Step 1: Filtering collection for year {year}...")
        filtered_embeddings = embeddings.filter(
            ee.Filter.date(
                ee.Date.fromYMD(year, 1, 1), 
                ee.Date.fromYMD(year, 12, 31)
            )
        ).filter(ee.Filter.bounds(netherlands_aoi))
        
        # Check if we have any images for this year
        collection_size = filtered_embeddings.size().getInfo()
        print(f"Found {collection_size} image tiles for year {year}")
        
        if collection_size == 0:
            print(f"No images found for year {year}. Skipping.")
            continue
        
        # 2. Create Mosaic
        print(f"Step 2: Creating mosaic from {collection_size} tiles...")
        yearly_mosaic = filtered_embeddings.mosaic()
        
        # 3. Clip to Area of Interest
        print(f"Step 3: Clipping mosaic to Netherlands boundary...")
        final_image = yearly_mosaic.clip(netherlands_aoi)
        
        # 4. Get image info for validation
        print(f"Step 4: Validating final image...")
        try:
            # Get some basic info about the final image
            band_names = final_image.bandNames().getInfo()
            print(f"Number of bands: {len(band_names)}")
            print(f"First few band names: {band_names[:5]}")
            
            # Get image bounds
            bounds = final_image.geometry().bounds().getInfo()
            print(f"Image bounds: {bounds}")
            
        except Exception as e:
            print(f"Warning: Could not validate image properties: {e}")
        
        # 5. Export the Final Image
        print(f"Step 5: Setting up export task...")
        task_description = f'Netherlands_Embedding_{year}_Mosaic'
        
        export_task = ee.batch.Export.image.toDrive(
            image=final_image,
            description=task_description,
            folder=gdrive_folder,
            fileNamePrefix=task_description,
            scale=10,  # Native 10-meter resolution
            region=netherlands_aoi,
            maxPixels=1e11,  # Large number to handle big areas
            crs='EPSG:28992',  # Amersfoort / RD New for Netherlands
            fileFormat='GeoTIFF'
        )
        
        # Start the export task
        export_task.start()
        print(f"✓ Export task started for {year}")
        print(f"  Task description: {task_description}")
        print(f"  Output folder: {gdrive_folder}")
        print(f"  Resolution: 10 meters")
        print(f"  CRS: EPSG:28992")
        
    except Exception as e:
        print(f"✗ Error processing year {year}: {e}")
        continue

# --- Monitor Export Tasks ---
print(f"\n{'='*60}")
print("MONITORING EXPORT TASKS")
print(f"{'='*60}")
print("Export tasks have been started.")
print("You can monitor progress in the Google Earth Engine Code Editor 'Tasks' tab.")
print("Or wait here for completion monitoring...")

# Optional: Monitor task completion
monitor_tasks = input("\nMonitor task completion? (y/n): ").lower().startswith('y')

if monitor_tasks:
    print("\nMonitoring export tasks (Ctrl+C to stop)...")
    try:
        while True:
            # Get tasks that are still running
            all_tasks = ee.batch.Task.list()
            running_tasks = [
                t for t in all_tasks 
                if t.state in ['READY', 'RUNNING'] and 
                'Netherlands_Embedding_' in t.config.get('description', '')
            ]
            
            if not running_tasks:
                print("\n✓ All AlphaEarth export tasks are complete!")
                
                # Show completed tasks
                completed_tasks = [
                    t for t in all_tasks 
                    if 'Netherlands_Embedding_' in t.config.get('description', '') and 
                    t.state in ['COMPLETED', 'FAILED']
                ]
                
                print(f"\nTask Summary:")
                for task in completed_tasks[-len(years_to_export):]:  # Show recent tasks
                    status_icon = "✓" if task.state == "COMPLETED" else "✗"
                    print(f"  {status_icon} {task.config.get('description')} - {task.state}")
                
                break
            
            print(f"\nTasks still in progress ({time.ctime()}):")
            for task in running_tasks:
                print(f"  ⏳ {task.config.get('description')} - {task.state}")
            
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Tasks will continue running in Google Earth Engine.")

print(f"\n{'='*60}")
print("EXPORT SCRIPT COMPLETE")
print(f"{'='*60}")
print("Key improvements in this corrected approach:")
print("✓ Proper filtering by date AND location")
print("✓ Mosaicking to create seamless single images")
print("✓ Clipping to exact area of interest")
print("✓ Better error handling and validation")
print("✓ One clean file per year instead of many tiles")
print(f"\nCheck your Google Drive folder '{gdrive_folder}' for the exported files.")