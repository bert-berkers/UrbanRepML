"""
Density setup script for UrbanRepML preprocessing.
Calculates building density for H3 regions using real building data.

This script:
1. Loads H3 regions from data/skeleton/regions/
2. Calculates building density using PV28_00_basis_bouwblok_shp.shp
3. Saves density data to data/skeleton/density/
4. Updates combined dataset in data/skeleton/total/
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_regions() -> gpd.GeoDataFrame:
    """Load H3 regions created by setup_regions.py."""
    regions_path = Path("data/skeleton/regions/south_holland_res10.parquet")
    
    if not regions_path.exists():
        raise FileNotFoundError(
            f"Regions file not found: {regions_path}\n"
            "Please run setup_regions.py first!"
        )
    
    print("📦 Loading H3 regions...")
    regions_gdf = gpd.read_parquet(regions_path)
    print(f"✅ Loaded {len(regions_gdf)} H3 regions")
    return regions_gdf

def calculate_building_density(regions_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Calculate building density using PV28_00_basis_bouwblok_shp data."""
    
    building_path = Path("data/skeleton/density/PV28_00_basis_bouwblok_shp.shp")
    
    if not building_path.exists():
        raise FileNotFoundError(
            f"Building data not found: {building_path}\n"
            "Please place the PV28_00_basis_bouwblok_shp shapefile there!"
        )
    
    print("🏢 Loading building data...")
    building_gdf = gpd.read_file(building_path)
    print(f"   Loaded {len(building_gdf)} building blocks")
    
    # Project to Dutch CRS for accurate area calculations
    print("🔄 Reprojecting to Dutch CRS (EPSG:28992)...")
    building_gdf = building_gdf.to_crs(epsg=28992)
    regions_gdf = regions_gdf.to_crs(epsg=28992)
    
    print("🔍 Performing spatial intersection...")
    # Spatial intersection between buildings and H3 regions
    overlay = gpd.overlay(building_gdf, regions_gdf.reset_index(), how='intersection')
    
    # Initialize results DataFrame
    results_df = pd.DataFrame(
        index=regions_gdf.index,
        columns=['FSI_24', 'building_volume', 'total_area', 'in_study_area'],
        data={
            'FSI_24': 0.0, 
            'building_volume': 0.0,
            'total_area': 0.0,
            'in_study_area': False
        }
    )
    
    if not overlay.empty:
        print("📊 Calculating building density metrics...")
        
        # Calculate intersection area and building volume
        overlay['intersection_area'] = overlay.geometry.area
        overlay['building_volume'] = overlay['FSI_24'] * overlay['intersection_area']
        
        # Aggregate by H3 region
        aggregated = overlay.groupby('region_id').agg({
            'building_volume': 'sum',
            'intersection_area': 'sum'
        }).reset_index()
        
        # Calculate weighted average FSI (building density)
        aggregated['FSI_24'] = (
            aggregated['building_volume'] / aggregated['intersection_area']
        ).fillna(0)
        
        # Update results with calculated values
        valid_indices = aggregated['region_id'].intersection(results_df.index)
        results_df.loc[valid_indices, 'FSI_24'] = aggregated.set_index('region_id').loc[valid_indices, 'FSI_24']
        results_df.loc[valid_indices, 'building_volume'] = aggregated.set_index('region_id').loc[valid_indices, 'building_volume']  
        results_df.loc[valid_indices, 'total_area'] = aggregated.set_index('region_id').loc[valid_indices, 'intersection_area']
        results_df.loc[valid_indices, 'in_study_area'] = True
        
        print(f"✅ Calculated density for {len(valid_indices)} regions with buildings")
    else:
        print("⚠️ No spatial intersections found!")
    
    return results_df

def save_density_data(density_df: pd.DataFrame):
    """Save density data to density folder."""
    
    density_path = Path("data/skeleton/density/south_holland_res10_density.parquet")
    density_df.to_parquet(density_path)
    print(f"💾 Density data saved: {density_path}")

def update_total_dataset(regions_gdf: gpd.GeoDataFrame, density_df: pd.DataFrame):
    """Update or create combined dataset in total folder."""
    
    total_path = Path("data/skeleton/total/south_holland_res10_complete.parquet")
    
    # Start with regions as base
    combined_gdf = regions_gdf.copy()
    
    # Add density columns
    combined_gdf['FSI_24'] = density_df['FSI_24']
    combined_gdf['building_volume'] = density_df['building_volume'] 
    combined_gdf['total_area'] = density_df['total_area']
    combined_gdf['in_study_area'] = density_df['in_study_area']
    
    # If total file exists, merge with existing data
    if total_path.exists():
        print("🔄 Updating existing total dataset...")
        existing_gdf = gpd.read_parquet(total_path)
        
        # Update columns that we're adding
        for col in ['FSI_24', 'building_volume', 'total_area', 'in_study_area']:
            existing_gdf[col] = combined_gdf[col]
        
        combined_gdf = existing_gdf
    else:
        print("🆕 Creating new total dataset...")
    
    # Save combined dataset  
    combined_gdf.to_parquet(total_path)
    print(f"💾 Total dataset saved: {total_path}")
    
    # Print summary statistics
    buildings_count = combined_gdf[combined_gdf['in_study_area']].shape[0]
    avg_density = combined_gdf[combined_gdf['FSI_24'] > 0]['FSI_24'].mean()
    
    print(f"\n📊 Density Summary:")
    print(f"   Regions with buildings: {buildings_count:,}")
    print(f"   Average FSI density: {avg_density:.3f}")

def main():
    """Main density setup workflow."""
    
    print("🏢 UrbanRepML Density Setup")
    print("=" * 40)
    
    # Load H3 regions
    regions_gdf = load_regions()
    
    # Calculate building density
    density_df = calculate_building_density(regions_gdf)
    
    # Save density data
    save_density_data(density_df)
    
    # Update total combined dataset
    update_total_dataset(regions_gdf, density_df)
    
    print("\n🎉 Density setup completed!")
    print("\n📁 Files created:")
    print("   data/skeleton/density/south_holland_res10_density.parquet")
    print("   data/skeleton/total/south_holland_res10_complete.parquet")
    print("\n🔧 Next steps:")
    print("   Add more features (transport, POI, etc.) or run experiments!")

if __name__ == "__main__":
    main()