#!/usr/bin/env python3
"""
Create base south_holland preprocessed data (without FSI filtering)
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np

def create_base_south_holland():
    """Create base South Holland data without FSI filtering."""
    
    print("=== Creating Base South Holland Data ===")
    
    # Paths
    project_root = Path("C:/Users/Bert Berkers/PycharmProjects/UrbanRepML")
    preprocessed_dir = project_root / "data" / "preprocessed"
    output_dir = preprocessed_dir / "south_holland"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each resolution
    for resolution in [8, 9, 10]:
        print(f"\n[INFO] Processing resolution {resolution}...")
        
        # Load region data (already copied)
        regions_path = output_dir / f"regions_{resolution}_gdf.parquet"
        regions_gdf = gpd.read_parquet(regions_path)
        print(f"   [INFO] Loaded {len(regions_gdf)} regions")
        
        # Load density data
        density_path = preprocessed_dir / "density" / f"south_holland_res{resolution}_density.parquet"
        if density_path.exists():
            density_df = pd.read_parquet(density_path)
            print(f"   [INFO] Loaded density data: {len(density_df)} records")
            
            # Join density with regions (no filtering)
            regions_with_density = regions_gdf.join(density_df[['FSI_24']], how='left')
            regions_with_density['FSI_24'] = regions_with_density['FSI_24'].fillna(0.0)
            regions_with_density['in_study_area'] = True  # All regions are in study area for base
            
            print(f"   [STATS] All regions kept: {len(regions_with_density)}")
            print(f"   [STATS] FSI range: [{regions_with_density['FSI_24'].min():.3f}, {regions_with_density['FSI_24'].max():.3f}]")
            print(f"   [STATS] Mean FSI: {regions_with_density['FSI_24'].mean():.3f}")
            
        else:
            print(f"   [WARN] No density data found")
            regions_with_density = regions_gdf.copy()
            regions_with_density['FSI_24'] = 0.0
            regions_with_density['in_study_area'] = True
        
        # Save updated regions
        regions_with_density.to_parquet(regions_path)
        print(f"   [OK] Updated {regions_path.name}")
        
        # Create processed density file
        density_output = output_dir / f"building_density_res{resolution}_preprocessed.parquet"
        density_processed = pd.DataFrame({
            'FSI_24': regions_with_density['FSI_24'],
            'in_study_area': regions_with_density['in_study_area']
        }, index=regions_with_density.index)
            
        density_processed.to_parquet(density_output)
        print(f"   [OK] Created {density_output.name}")
    
    print(f"\n[SUCCESS] Base South Holland data created!")
    print(f"[OUTPUT] Directory: {output_dir}")

if __name__ == "__main__":
    try:
        create_base_south_holland()
        print("\n[SUCCESS] Complete!")
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)