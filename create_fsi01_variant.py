#!/usr/bin/env python3
"""
Create South Holland FSI 0.1 variant by filtering regions with FSI >= 0.1
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

def create_fsi01_variant():
    """Create FSI 0.1 filtered variant of South Holland data."""
    
    print("=== Creating South Holland FSI 0.1 Variant ===")
    
    # Paths
    project_root = Path("C:/Users/Bert Berkers/PycharmProjects/UrbanRepML")
    skeleton_dir = project_root / "data" / "skeleton"
    output_dir = project_root / "data" / "preprocessed" / "south_holland_fsi01"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy area study (unchanged)
    print("[1/3] Copying study area...")
    area_path = skeleton_dir / "boundaries" / "south_holland_area.parquet"
    area_gdf = gpd.read_parquet(area_path)
    area_gdf.to_parquet(output_dir / "area_study_gdf.parquet")
    print(f"   [OK] Saved area_study_gdf.parquet")
    
    # 2. Process each resolution
    for resolution in [8, 9, 10]:
        print(f"\n[2/3] Processing resolution {resolution}...")
        
        # Load region data
        regions_path = skeleton_dir / "regions" / f"south_holland_res{resolution}.parquet"
        regions_gdf = gpd.read_parquet(regions_path)
        print(f"   [INFO] Loaded {len(regions_gdf)} regions")
        
        # Load density data
        density_path = skeleton_dir / "density" / f"south_holland_res{resolution}_density.parquet"
        if density_path.exists():
            density_df = pd.read_parquet(density_path)
            print(f"   [INFO] Loaded density data: {len(density_df)} records")
            
            # Join density with regions
            regions_with_density = regions_gdf.join(density_df[['FSI_24']], how='left')
            regions_with_density['FSI_24'] = regions_with_density['FSI_24'].fillna(0.0)
            
            # Filter by FSI >= 0.1
            fsi_filtered = regions_with_density[regions_with_density['FSI_24'] >= 0.1].copy()
            fsi_filtered['in_study_area'] = True
            
            print(f"   [FILTER] FSI >= 0.1 filter: {len(regions_gdf)} -> {len(fsi_filtered)} regions")
            print(f"   [STATS] FSI range: [{fsi_filtered['FSI_24'].min():.3f}, {fsi_filtered['FSI_24'].max():.3f}]")
            print(f"   [STATS] Mean FSI: {fsi_filtered['FSI_24'].mean():.3f}")
            
        else:
            print(f"   [WARN] No density data found, keeping all regions")
            fsi_filtered = regions_gdf.copy()
            fsi_filtered['FSI_24'] = 0.0
            fsi_filtered['in_study_area'] = True
        
        # Save filtered regions
        regions_output = output_dir / f"regions_{resolution}_gdf.parquet"
        fsi_filtered.to_parquet(regions_output)
        print(f"   [OK] Saved {regions_output.name}")
        
        # Create processed density file
        density_output = output_dir / f"building_density_res{resolution}_preprocessed.parquet"
        if len(fsi_filtered) > 0:
            density_processed = pd.DataFrame({
                'FSI_24': fsi_filtered['FSI_24'],
                'in_study_area': fsi_filtered['in_study_area']
            }, index=fsi_filtered.index)
        else:
            density_processed = pd.DataFrame({
                'FSI_24': pd.Series(dtype=float),
                'in_study_area': pd.Series(dtype=bool)
            })
            
        density_processed.to_parquet(density_output)
        print(f"   [OK] Saved {density_output.name}")
    
    print(f"\n[3/3] FSI 0.1 variant created successfully!")
    print(f"[OUTPUT] Directory: {output_dir}")
    print(f"[SUMMARY] Total regions by resolution:")
    
    # Summary statistics
    for resolution in [8, 9, 10]:
        regions_file = output_dir / f"regions_{resolution}_gdf.parquet"
        if regions_file.exists():
            regions_count = len(gpd.read_parquet(regions_file))
            print(f"   Resolution {resolution}: {regions_count} regions")

if __name__ == "__main__":
    try:
        create_fsi01_variant()
        print("\n[SUCCESS] Complete!")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)