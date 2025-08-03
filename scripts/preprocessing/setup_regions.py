"""
Region setup script for UrbanRepML preprocessing.
Sets up H3 regionalization and building density data using SRAI and RUDIFUN.

This is the centralized script for creating empty H3 hexagon GeoDataFrames
that are used throughout the UrbanRepML project. It should be the only place
where we define base regional structures.
"""

import sys
import warnings
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import geopandas as gpd
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_project_paths():
    """Ensure all necessary directories exist."""
    paths = [
        "data/skeleton/boundaries",
        "data/skeleton/regions",
        "data/skeleton/density", 
        "data/skeleton/total",
        "cache"
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        
    print("✅ Project directories created")

def get_study_area() -> gpd.GeoDataFrame:
    """Get study area boundary using SRAI."""
    print("📍 Fetching South Holland boundary using SRAI...")
    area_gdf = geocode_to_region_gdf("South Holland, Netherlands")
    print(f"✅ Found South Holland boundary ({len(area_gdf)} regions)")
    return area_gdf

def create_h3_regions(area_gdf: gpd.GeoDataFrame, resolution: int = 10) -> gpd.GeoDataFrame:
    """Create H3 regions using SRAI."""
    print(f"🔷 Creating H3 regionalization at resolution {resolution}...")
    
    regionalizer = H3Regionalizer(resolution=resolution)
    regions_gdf = regionalizer.transform(area_gdf)
    
    # Add region_id column if not present (SRAI uses index as region_id)
    if 'region_id' not in regions_gdf.columns:
        regions_gdf['region_id'] = regions_gdf.index
        
    print(f"✅ Created {len(regions_gdf)} H3 regions at resolution {resolution}")
    return regions_gdf


def save_region_data(area_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame):
    """Save basic region data to organized structure."""
    
    print("💾 Saving region data...")
    
    # Save study area boundary
    area_path = Path("data/skeleton/boundaries/south_holland_area.parquet")
    area_gdf.to_parquet(area_path)
    print(f"   Study area: {area_path}")
    
    # Save H3 base regions
    regions_path = Path("data/skeleton/regions/south_holland_res10.parquet") 
    regions_gdf.to_parquet(regions_path)
    print(f"   H3 regions: {regions_path}")
    
    # Initialize total dataset with base region data
    total_path = Path("data/skeleton/total/south_holland_res10_complete.parquet")
    regions_gdf.to_parquet(total_path)
    print(f"   Total dataset initialized: {total_path}")
    
    print("✅ Basic region data saved")

def main():
    """Main region setup workflow - creates basic H3 hexagon regions."""
    
    print("🏗️ UrbanRepML Region Setup")
    print("=" * 40)
    
    # Setup directories
    setup_project_paths()
    
    # Get South Holland boundary (this is area_gdf)
    area_gdf = get_study_area()
    
    # Create H3 regionalization (this creates regions_gdf)
    regions_gdf = create_h3_regions(area_gdf, resolution=10)
    
    # Save basic region data
    save_region_data(area_gdf, regions_gdf)
    
    print("\n🎉 Region setup completed!")
    print(f"   Study area: South Holland")
    print(f"   H3 resolution: 10")
    print(f"   Total regions: {len(regions_gdf)}")
    print("\n📁 Data saved to data/skeleton/")
    print("\n🔧 Next steps:")
    print("   Run setup_density.py to add building density data")

if __name__ == "__main__":
    main()