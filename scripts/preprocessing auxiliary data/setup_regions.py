"""
Region setup script for UrbanRepML preprocessing auxiliary data.
Sets up H3 regionalization and building density data using SRAI and RUDIFUN.

This is the centralized script for creating empty H3 hexagon GeoDataFrames
that are used throughout the UrbanRepML project. It should be the only place
where we define base regional structures.
"""

import sys
import warnings
from pathlib import Path
from typing import Optional, Dict
import argparse

import pandas as pd
import geopandas as gpd
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_project_paths(output_dir: Path):
    """Ensure all necessary directories exist."""
    paths = [
        output_dir / "boundaries",
        output_dir / "regions",
        output_dir / "density", 
        output_dir / "total",
        output_dir.parent.parent / "cache"
    ]
    
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        
    print(f"[OK] Project directories created under {output_dir}")

def get_study_area(city_name: str) -> gpd.GeoDataFrame:
    """Get study area boundary using SRAI."""
    # Map city names to geocoding queries
    city_queries = {
        "south_holland": "South Holland, Netherlands",
        "amsterdam": "Amsterdam, Netherlands",
        "rotterdam": "Rotterdam, Netherlands",
        "utrecht": "Utrecht, Netherlands",
        "the_hague": "The Hague, Netherlands"
    }
    
    query = city_queries.get(city_name.lower(), f"{city_name}, Netherlands")
    print(f"[INFO] Fetching {city_name} boundary using SRAI...")
    area_gdf = geocode_to_region_gdf(query)
    print(f"[OK] Found {city_name} boundary ({len(area_gdf)} regions)")
    return area_gdf

def create_h3_regions(area_gdf: gpd.GeoDataFrame, resolution: int = 10) -> gpd.GeoDataFrame:
    """Create H3 regions using SRAI."""
    print(f"[INFO] Creating H3 regionalization at resolution {resolution}...")
    
    regionalizer = H3Regionalizer(resolution=resolution)
    regions_gdf = regionalizer.transform(area_gdf)
    
    # Add region_id column if not present (SRAI uses index as region_id)
    if 'region_id' not in regions_gdf.columns:
        regions_gdf['region_id'] = regions_gdf.index
        
    print(f"[OK] Created {len(regions_gdf)} H3 regions at resolution {resolution}")
    return regions_gdf


def save_region_data(area_gdf: gpd.GeoDataFrame, regions_dict: Dict[int, gpd.GeoDataFrame], 
                     city_name: str, output_dir: Path):
    """Save basic region data to organized structure."""
    
    print("[SAVE] Saving region data...")
    
    # Save study area boundary
    area_path = output_dir / "boundaries" / f"{city_name}_area.parquet"
    area_gdf.to_parquet(area_path)
    print(f"   Study area: {area_path}")
    
    # Save H3 regions for each resolution
    for resolution, regions_gdf in regions_dict.items():
        regions_path = output_dir / "regions" / f"{city_name}_res{resolution}.parquet"
        regions_gdf.to_parquet(regions_path)
        print(f"   H3 regions (res {resolution}): {regions_path}")
        
        # Initialize total dataset with base region data
        total_path = output_dir / "total" / f"{city_name}_res{resolution}_complete.parquet"
        regions_gdf.to_parquet(total_path)
        print(f"   Total dataset initialized (res {resolution}): {total_path}")
    
    print("[OK] Basic region data saved")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup H3 regions for urban analysis')
    parser.add_argument('--city_name', type=str, default='south_holland',
                        help='Name of the city/region to process')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for processed data')
    parser.add_argument('--resolutions', type=str, default='8,9,10',
                        help='Comma-separated H3 resolutions to generate (e.g., "8,9,10")')
    return parser.parse_args()

def main():
    """Main region setup workflow - creates basic H3 hexagon regions."""
    
    args = parse_args()
    
    print("==== UrbanRepML Region Setup ====")
    print("=" * 40)
    
    # Parse resolutions
    resolutions = [int(r) for r in args.resolutions.split(',')]
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/preprocessed/{args.city_name}_base")
    
    # Setup directories
    setup_project_paths(output_dir)
    
    # Get city boundary
    area_gdf = get_study_area(args.city_name)
    
    # Create H3 regionalization for each resolution
    regions_dict = {}
    for resolution in resolutions:
        regions_gdf = create_h3_regions(area_gdf, resolution=resolution)
        regions_dict[resolution] = regions_gdf
    
    # Save basic region data
    save_region_data(area_gdf, regions_dict, args.city_name, output_dir)
    
    print("\n[SUCCESS] Region setup completed!")
    print(f"   Study area: {args.city_name}")
    print(f"   H3 resolutions: {resolutions}")
    print(f"   Total regions:")
    for res, gdf in regions_dict.items():
        print(f"      Resolution {res}: {len(gdf)} regions")
    print(f"\n[INFO] Data saved to {output_dir}")
    print("\n[NEXT] Next steps:")
    print(f"   Run setup_density.py --city_name {args.city_name} --input_dir {output_dir}")

if __name__ == "__main__":
    main()