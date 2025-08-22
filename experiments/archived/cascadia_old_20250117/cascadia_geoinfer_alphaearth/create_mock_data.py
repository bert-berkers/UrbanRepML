#!/usr/bin/env python3
"""
Create mock AlphaEarth data for testing Cascadia experiment pipeline.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon
from datetime import datetime
import json

print("="*60)
print("CREATING MOCK ALPHAEARTH DATA FOR CASCADIA EXPERIMENT")
print("="*60)
print(f"Timestamp: {datetime.now().isoformat()}")

# Create data directories
data_dirs = [
    'data/h3_processed/resolution_8',
    'data/h3_processed/resolution_9', 
    'data/h3_processed/resolution_10'
]

for dir_path in data_dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# Mock Cascadia region bounds (Northern California + Oregon)
cascadia_bounds = {
    'north': 46.3,
    'south': 39.0, 
    'west': -124.6,
    'east': -116.5
}

def create_mock_h3_data(resolution, year=2023, n_samples=1000):
    """Create mock H3 data for a given resolution."""
    print(f"\nCreating mock data for resolution {resolution}, year {year}")
    
    # Generate random points within Cascadia bounds
    np.random.seed(42 + resolution + year)  # Reproducible
    
    lats = np.random.uniform(cascadia_bounds['south'], cascadia_bounds['north'], n_samples)
    lons = np.random.uniform(cascadia_bounds['west'], cascadia_bounds['east'], n_samples)
    
    # Convert to H3 indices
    h3_indices = []
    geometries = []
    
    for lat, lon in zip(lats, lons):
        h3_index = h3.latlng_to_cell(lat, lon, resolution)
        h3_indices.append(h3_index)
        
        # Get H3 cell geometry
        boundary = h3.cell_to_boundary(h3_index)
        polygon = Polygon([(lon, lat) for lat, lon in boundary])
        geometries.append(polygon)
    
    # Remove duplicates (multiple points in same H3 cell)
    unique_data = {}
    for i, h3_index in enumerate(h3_indices):
        if h3_index not in unique_data:
            unique_data[h3_index] = geometries[i]
    
    # Create embedding data (64 dimensions like AlphaEarth)
    n_unique = len(unique_data)
    embeddings = np.random.randn(n_unique, 64) * 0.5  # Realistic embedding scale
    
    # Create GeoDataFrame
    rows = []
    for i, (h3_index, geometry) in enumerate(unique_data.items()):
        row = {
            'h3_index': h3_index,
            'resolution': resolution,
            'geometry': geometry,
            'pixel_count': np.random.randint(5, 50),  # Mock pixel count
            'tile_id': f'mock_tile_{i // 100}'  # Mock tile assignment
        }
        
        # Add embedding dimensions
        for j in range(64):
            row[f'embed_{j}'] = embeddings[i, j]
        
        rows.append(row)
    
    gdf = gpd.GeoDataFrame(rows, crs='EPSG:4326')
    
    print(f"  Created {len(gdf)} unique H3 cells")
    print(f"  Embedding dimensions: 64")
    print(f"  Sample H3 indices: {list(gdf['h3_index'].head(3))}")
    
    return gdf

# Create mock data for multiple resolutions and years
resolutions = [8, 9, 10]
years = [2023, 2024]

# Different sample sizes for different resolutions
sample_sizes = {8: 500, 9: 1500, 10: 3000}

for year in years:
    print(f"\n{'='*40}")
    print(f"CREATING DATA FOR YEAR {year}")
    print(f"{'='*40}")
    
    for resolution in resolutions:
        # Create mock data
        gdf = create_mock_h3_data(
            resolution, 
            year, 
            sample_sizes[resolution]
        )
        
        # Save to parquet
        output_file = f'data/h3_processed/resolution_{resolution}/cascadia_{year}_h3_res{resolution}.parquet'
        gdf.to_parquet(output_file)
        
        print(f"  Saved: {output_file}")
        
        # Create metadata
        metadata = {
            'year': year,
            'resolution': resolution,
            'n_h3_cells': len(gdf),
            'embedding_dimensions': 64,
            'created_timestamp': datetime.now().isoformat(),
            'data_type': 'mock_alphaearth',
            'region': 'cascadia',
            'bounds': cascadia_bounds
        }
        
        metadata_file = output_file.replace('.parquet', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

print("\n" + "="*60)
print("MOCK DATA CREATION SUMMARY")
print("="*60)

# Summary statistics
total_files = 0
total_cells = 0

for year in years:
    print(f"\nYear {year}:")
    for resolution in resolutions:
        file_path = f'data/h3_processed/resolution_{resolution}/cascadia_{year}_h3_res{resolution}.parquet'
        if os.path.exists(file_path):
            gdf = gpd.read_parquet(file_path)
            print(f"  Resolution {resolution}: {len(gdf):,} H3 cells")
            total_files += 1
            total_cells += len(gdf)

print(f"\nTotal files created: {total_files}")
print(f"Total H3 cells: {total_cells:,}")
print(f"Mock data ready for pipeline testing!")
print("="*60)