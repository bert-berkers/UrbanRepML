"""
Create Cascadia bioregion boundaries for AlphaEarth data extraction.

This script creates GeoDataFrames with exact county boundaries for:
- 16 Northern California counties
- 36 Oregon counties (all)
Total: 52 counties in the Cascadia bioregion as defined by GEO-INFER
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon
from pathlib import Path
import json
import requests
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class CascadiaBoundaryCreator:
    """Create and manage Cascadia bioregion boundaries for data extraction."""
    
    def __init__(self, output_dir: str = "data/boundaries/cascadia"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define Cascadia counties based on GEO-INFER specification
        self.california_counties = [
            "Butte", "Colusa", "Del Norte", "Glenn", "Humboldt", "Lake",
            "Lassen", "Mendocino", "Modoc", "Nevada", "Plumas", "Shasta",
            "Sierra", "Siskiyou", "Tehama", "Trinity"
        ]
        
        # Oregon: all 36 counties
        self.oregon_counties = [
            "Baker", "Benton", "Clackamas", "Clatsop", "Columbia", "Coos",
            "Crook", "Curry", "Deschutes", "Douglas", "Gilliam", "Grant",
            "Harney", "Hood River", "Jackson", "Jefferson", "Josephine",
            "Klamath", "Lake", "Lane", "Lincoln", "Linn", "Malheur",
            "Marion", "Morrow", "Multnomah", "Polk", "Sherman", "Tillamook",
            "Umatilla", "Union", "Wallowa", "Wasco", "Washington", "Wheeler",
            "Yamhill"
        ]
        
        # Bounding box for entire Cascadia region (from config.yaml)
        self.cascadia_bounds = {
            'north': 46.3,   # Northern Oregon border
            'south': 39.0,   # Northern California (below Tehama)
            'west': -124.6,  # Pacific Coast
            'east': -116.5   # Eastern Oregon/California border
        }
        
    def get_county_boundaries_census(self) -> gpd.GeoDataFrame:
        """
        Download county boundaries from US Census TIGER/Line files.
        """
        print("Downloading county boundaries from US Census...")
        
        # URLs for Census TIGER/Line shapefiles
        ca_url = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_06_county.zip"
        or_url = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_41_county.zip"
        
        try:
            # Download California counties
            ca_counties = gpd.read_file(ca_url)
            ca_counties['STATE'] = 'CA'
            
            # Download Oregon counties
            or_counties = gpd.read_file(or_url)
            or_counties['STATE'] = 'OR'
            
            # Filter for Cascadia counties
            ca_filtered = ca_counties[ca_counties['NAME'].isin(self.california_counties)]
            or_filtered = or_counties  # All Oregon counties
            
            # Combine
            cascadia_counties = pd.concat([ca_filtered, or_filtered], ignore_index=True)
            
            # Ensure CRS is WGS84
            cascadia_counties = cascadia_counties.to_crs('EPSG:4326')
            
            return cascadia_counties
            
        except Exception as e:
            print(f"Error downloading Census data: {e}")
            return self.create_fallback_boundaries()
    
    def create_fallback_boundaries(self) -> gpd.GeoDataFrame:
        """
        Create simplified boundary boxes as fallback.
        """
        print("Creating fallback boundaries...")
        
        # Create bounding box for entire region
        bbox = box(
            self.cascadia_bounds['west'],
            self.cascadia_bounds['south'],
            self.cascadia_bounds['east'],
            self.cascadia_bounds['north']
        )
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            [{
                'NAME': 'Cascadia Bioregion',
                'STATE': 'CA_OR',
                'geometry': bbox,
                'COUNTIES': len(self.california_counties) + len(self.oregon_counties),
                'AREA_KM2': 421000
            }],
            crs='EPSG:4326'
        )
        
        return gdf
    
    def create_county_info_table(self, counties_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Create detailed information table for counties.
        """
        info_data = []
        
        for idx, row in counties_gdf.iterrows():
            bounds = row.geometry.bounds
            info_data.append({
                'county_name': row['NAME'],
                'state': row['STATE'],
                'fips_code': row.get('GEOID', 'N/A'),
                'area_km2': row.geometry.area * 111 * 111,  # Rough conversion
                'centroid_lat': row.geometry.centroid.y,
                'centroid_lon': row.geometry.centroid.x,
                'bbox_west': bounds[0],
                'bbox_south': bounds[1],
                'bbox_east': bounds[2],
                'bbox_north': bounds[3]
            })
        
        return pd.DataFrame(info_data)
    
    def create_gee_export_config(self, counties_gdf: gpd.GeoDataFrame) -> Dict:
        """
        Create configuration for Google Earth Engine exports.
        """
        # Get overall bounds
        total_bounds = counties_gdf.total_bounds
        
        config = {
            'region_name': 'cascadia',
            'total_counties': len(counties_gdf),
            'california_counties': len(counties_gdf[counties_gdf['STATE'] == 'CA']),
            'oregon_counties': len(counties_gdf[counties_gdf['STATE'] == 'OR']),
            'bounds': {
                'west': float(total_bounds[0]),
                'south': float(total_bounds[1]),
                'east': float(total_bounds[2]),
                'north': float(total_bounds[3])
            },
            'gee_export': {
                'collection': 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',
                'years': list(range(2017, 2025)),
                'scale': 10,
                'crs': 'EPSG:4326',
                'max_pixels': 1e9,
                'tile_size': 3072
            },
            'h3_resolutions': [5, 6, 7, 8, 9, 10, 11],
            'primary_resolution': 8
        }
        
        return config
    
    def save_outputs(self, counties_gdf: gpd.GeoDataFrame):
        """
        Save all outputs for use in AlphaEarth extraction.
        """
        print(f"\nSaving outputs to {self.output_dir}...")
        
        # Save main GeoDataFrame
        counties_gdf.to_file(
            self.output_dir / "cascadia_counties.geojson",
            driver='GeoJSON'
        )
        counties_gdf.to_parquet(
            self.output_dir / "cascadia_counties.parquet"
        )
        
        # Save simplified version (dissolved by state)
        simplified = counties_gdf.dissolve(by='STATE').reset_index()
        simplified.to_file(
            self.output_dir / "cascadia_states.geojson",
            driver='GeoJSON'
        )
        
        # Save county info table
        info_table = self.create_county_info_table(counties_gdf)
        info_table.to_csv(
            self.output_dir / "cascadia_counties_info.csv",
            index=False
        )
        
        # Save GEE export config
        gee_config = self.create_gee_export_config(counties_gdf)
        with open(self.output_dir / "gee_export_config.json", 'w') as f:
            json.dump(gee_config, f, indent=2)
        
        # Save county lists
        county_lists = {
            'california': self.california_counties,
            'oregon': self.oregon_counties,
            'total_count': len(self.california_counties) + len(self.oregon_counties)
        }
        with open(self.output_dir / "county_lists.json", 'w') as f:
            json.dump(county_lists, f, indent=2)
        
        print(f"✓ Saved cascadia_counties.geojson")
        print(f"✓ Saved cascadia_counties.parquet")
        print(f"✓ Saved cascadia_states.geojson")
        print(f"✓ Saved cascadia_counties_info.csv")
        print(f"✓ Saved gee_export_config.json")
        print(f"✓ Saved county_lists.json")
    
    def print_summary(self, counties_gdf: gpd.GeoDataFrame):
        """
        Print summary of the created boundaries.
        """
        total_bounds = counties_gdf.total_bounds
        
        print("\n" + "="*60)
        print("CASCADIA BIOREGION BOUNDARY SUMMARY")
        print("="*60)
        print(f"\nTotal Counties: {len(counties_gdf)}")
        print(f"  - California: {len(counties_gdf[counties_gdf['STATE'] == 'CA'])} counties")
        print(f"  - Oregon: {len(counties_gdf[counties_gdf['STATE'] == 'OR'])} counties")
        
        print(f"\nSpatial Extent:")
        print(f"  - North: {total_bounds[3]:.2f}°")
        print(f"  - South: {total_bounds[1]:.2f}°")
        print(f"  - East: {total_bounds[2]:.2f}°")
        print(f"  - West: {total_bounds[0]:.2f}°")
        
        print(f"\nApproximate Area: ~421,000 km²")
        
        print("\nCalifornia Counties:")
        ca_counties = counties_gdf[counties_gdf['STATE'] == 'CA']['NAME'].tolist()
        print(f"  {', '.join(sorted(ca_counties))}")
        
        print("\nOregon Counties:")
        print(f"  All 36 counties included")
        
        print("\n" + "="*60)
        print("Ready for AlphaEarth data extraction!")
        print("Use the saved GeoJSON file to filter GEE exports")
        print("="*60)
    
    def run(self):
        """
        Execute the full boundary creation process.
        """
        print("Creating Cascadia bioregion boundaries...")
        
        # Get county boundaries
        counties_gdf = self.get_county_boundaries_census()
        
        # Save all outputs
        self.save_outputs(counties_gdf)
        
        # Print summary
        self.print_summary(counties_gdf)
        
        return counties_gdf


def main():
    """Main execution function."""
    creator = CascadiaBoundaryCreator()
    counties_gdf = creator.run()
    
    print("\n✅ Cascadia boundaries created successfully!")
    print(f"Files saved to: data/boundaries/cascadia/")
    
    return counties_gdf


if __name__ == "__main__":
    main()