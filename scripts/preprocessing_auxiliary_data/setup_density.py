"""
Density setup script for UrbanRepML preprocessing_auxiliary_data.
Calculates building density for H3 regions using real building data.

This script:
1. Loads H3 regions from specified input directory
2. Calculates building density using PV28__00_Basis_Bouwblok.shp
3. Saves density data to specified output directory
4. Updates combined dataset in total folder
"""

import sys
import warnings
from pathlib import Path
from typing import Optional, Dict
import argparse

import pandas as pd
import geopandas as gpd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_regions(input_dir: Path, city_name: str, resolutions: list) -> Dict[int, gpd.GeoDataFrame]:
    """Load H3 regions created by setup_regions.py for all resolutions."""
    regions_dict = {}
    
    for resolution in resolutions:
        regions_path = input_dir / "regions" / f"{city_name}_res{resolution}.parquet"
        
        if not regions_path.exists():
            raise FileNotFoundError(
                f"Regions file not found: {regions_path}\n"
                "Please run setup_regions.py first!"
            )
        
        print(f"[LOAD] Loading H3 regions for resolution {resolution}...")
        regions_gdf = gpd.read_parquet(regions_path)
        print(f"[OK] Loaded {len(regions_gdf)} H3 regions at resolution {resolution}")
        regions_dict[resolution] = regions_gdf
    
    return regions_dict

def calculate_building_density(regions_gdf: gpd.GeoDataFrame, building_data_path: Path) -> pd.DataFrame:
    """Calculate building density using PV28__00_Basis_Bouwblok data."""
    
    if not building_data_path.exists():
        raise FileNotFoundError(
            f"Building data not found: {building_data_path}\n"
            "Please ensure the building shapefile is available!"
        )
    
    print("[LOAD] Loading building data...")
    building_gdf = gpd.read_file(building_data_path)
    print(f"   Loaded {len(building_gdf)} building blocks")
    
    # Project to Dutch CRS for accurate area calculations
    print("[PROJ] Reprojecting to Dutch CRS (EPSG:28992)...")
    building_gdf = building_gdf.to_crs(epsg=28992)
    regions_gdf = regions_gdf.to_crs(epsg=28992)
    
    print("[CALC] Performing spatial intersection...")
    # Spatial intersection between buildings and H3 regions  
    # If region_id is already a column, drop it before reset_index
    regions_for_overlay = regions_gdf.copy()
    if 'region_id' in regions_for_overlay.columns:
        regions_for_overlay = regions_for_overlay.drop(columns=['region_id'])
    overlay = gpd.overlay(building_gdf, regions_for_overlay.reset_index(), how='intersection')
    
    # Initialize results [old 2024] DataFrame
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
        print("[CALC] Calculating building density metrics...")
        
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
        
        # Update results [old 2024] with calculated values
        # Use pandas Index intersection
        valid_indices = pd.Index(aggregated['region_id']).intersection(results_df.index)
        aggregated_indexed = aggregated.set_index('region_id')
        results_df.loc[valid_indices, 'FSI_24'] = aggregated_indexed.loc[valid_indices, 'FSI_24']
        results_df.loc[valid_indices, 'building_volume'] = aggregated_indexed.loc[valid_indices, 'building_volume']  
        results_df.loc[valid_indices, 'total_area'] = aggregated_indexed.loc[valid_indices, 'intersection_area']
        results_df.loc[valid_indices, 'in_study_area'] = True
        
        print(f"[OK] Calculated density for {len(valid_indices)} regions with buildings")
    else:
        print("[WARN] No spatial intersections found!")
    
    return results_df

def save_density_data(density_dict: Dict[int, pd.DataFrame], city_name: str, output_dir: Path):
    """Save density data to density folder for all resolutions."""
    
    for resolution, density_df in density_dict.items():
        density_path = output_dir / "density" / f"{city_name}_res{resolution}_density.parquet"
        density_df.to_parquet(density_path)
        print(f"[SAVE] Density data saved for resolution {resolution}: {density_path}")

def update_total_dataset(regions_dict: Dict[int, gpd.GeoDataFrame], 
                        density_dict: Dict[int, pd.DataFrame], 
                        city_name: str, output_dir: Path):
    """Update or create combined dataset in total folder for all resolutions."""
    
    for resolution in regions_dict.keys():
        regions_gdf = regions_dict[resolution]
        density_df = density_dict[resolution]
        total_path = output_dir / "total" / f"{city_name}_res{resolution}_complete.parquet"
    
        # Start with regions as base
        combined_gdf = regions_gdf.copy()
        
        # Add density columns
        combined_gdf['FSI_24'] = density_df['FSI_24']
        combined_gdf['building_volume'] = density_df['building_volume'] 
        combined_gdf['total_area'] = density_df['total_area']
        combined_gdf['in_study_area'] = density_df['in_study_area']
        
        # If total file exists, merge with existing data
        if total_path.exists():
            print(f"[UPDATE] Updating existing total dataset for resolution {resolution}...")
            existing_gdf = gpd.read_parquet(total_path)
            
            # Update columns that we're adding
            for col in ['FSI_24', 'building_volume', 'total_area', 'in_study_area']:
                existing_gdf[col] = combined_gdf[col]
            
            combined_gdf = existing_gdf
        else:
            print(f"[NEW] Creating new total dataset for resolution {resolution}...")
        
        # Save combined dataset  
        combined_gdf.to_parquet(total_path)
        print(f"[SAVE] Total dataset saved for resolution {resolution}: {total_path}")
        
        # Print summary statistics
        buildings_count = combined_gdf[combined_gdf['in_study_area']].shape[0]
        avg_density = combined_gdf[combined_gdf['FSI_24'] > 0]['FSI_24'].mean()
        
        print(f"\n[STATS] Density Summary for resolution {resolution}:")
        print(f"   Regions with buildings: {buildings_count:,}")
        print(f"   Average FSI density: {avg_density:.3f}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate building density for H3 regions')
    parser.add_argument('--city_name', type=str, default='south_holland',
                        help='Name of the city/region to process')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory with region data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for density data')
    parser.add_argument('--building_data', type=str, 
                        default='data/preprocessed [TODO SORT & CLEAN UP]/density/PV28__00_Basis_Bouwblok.shp',
                        help='Path to building density shapefile')
    parser.add_argument('--resolutions', type=str, default='8,9,10',
                        help='Comma-separated H3 resolutions to process')
    return parser.parse_args()

def main():
    """Main density setup workflow."""
    
    args = parse_args()
    
    print("==== UrbanRepML Density Setup ====")
    print("=" * 40)
    
    # Parse resolutions
    resolutions = [int(r) for r in args.resolutions.split(',')]
    
    # Setup directories
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path(f"data/preprocessed [TODO SORT & CLEAN UP]/{args.city_name}_base")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir  # Same as input by default
    
    building_data_path = Path(args.building_data)
    
    # Load H3 regions for all resolutions
    regions_dict = load_regions(input_dir, args.city_name, resolutions)
    
    # Calculate building density for each resolution
    density_dict = {}
    for resolution, regions_gdf in regions_dict.items():
        print(f"\n[PROCESS] Calculating density for resolution {resolution}...")
        density_df = calculate_building_density(regions_gdf, building_data_path)
        density_dict[resolution] = density_df
    
    # Save density data
    save_density_data(density_dict, args.city_name, output_dir)
    
    # Update total combined dataset
    update_total_dataset(regions_dict, density_dict, args.city_name, output_dir)
    
    print("\n[SUCCESS] Density setup completed!")
    print(f"\n[INFO] Files created in {output_dir}:")
    for resolution in resolutions:
        print(f"   density/{args.city_name}_res{resolution}_density.parquet")
        print(f"   total/{args.city_name}_res{resolution}_complete.parquet")
    print("\n[NEXT] Next steps:")
    print(f"   Run setup_fsi_filter.py to filter by FSI threshold")

if __name__ == "__main__":
    main()