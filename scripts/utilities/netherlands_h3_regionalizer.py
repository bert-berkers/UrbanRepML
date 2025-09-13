"""
Create Netherlands H3 hexagons at multiple resolutions using SRAI
and verify AlphaEarth data coverage
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import h3
from shapely.geometry import Polygon, box
from srai.regionalizers import H3Regionalizer
import warnings
import rasterio
from rasterio.warp import transform_bounds
from pyproj import CRS
warnings.filterwarnings('ignore')


def get_netherlands_boundary():
    """Get Netherlands country boundary from Natural Earth or OSM"""
    try:
        # Try to load from Natural Earth
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        netherlands = world[world['name'] == 'Netherlands'].copy()
        
        if len(netherlands) == 0:
            # Fallback to bounding box if not found
            # Netherlands approximate bounds in WGS84
            bounds = box(3.36, 50.75, 7.23, 53.55)
            netherlands = gpd.GeoDataFrame(
                {'name': ['Netherlands'], 'geometry': [bounds]},
                crs='EPSG:4326'
            )
        
        return netherlands
    except Exception as e:
        print(f"Warning: Could not load Natural Earth data: {e}")
        # Use Netherlands bounding box
        bounds = box(3.36, 50.75, 7.23, 53.55)
        netherlands = gpd.GeoDataFrame(
            {'name': ['Netherlands'], 'geometry': [bounds]},
            crs='EPSG:4326'
        )
        return netherlands


def verify_alphaearth_coverage():
    """Check the spatial extent of AlphaEarth TIFFs"""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv('keys/.env')
    
    # Get path from environment or use default
    tiff_path = os.getenv('ALPHAEARTH_NETHERLANDS_PATH', 'G:/My Drive/AlphaEarth_Netherlands/')
    tiff_dir = Path(tiff_path)
    
    if not tiff_dir.exists():
        print("Warning: AlphaEarth directory not found")
        return None
    
    # Sample TIFFs to get overall bounds
    tiff_files = list(tiff_dir.glob('Netherlands_Embedding_2023_*.tif'))
    
    if not tiff_files:
        print("Warning: No AlphaEarth TIFFs found")
        return None
    
    print(f"Found {len(tiff_files)} AlphaEarth TIFFs for 2023")
    
    # Get bounds from all TIFFs
    all_bounds = []
    for i, tiff_path in enumerate(tiff_files[:10]):  # Sample first 10
        with rasterio.open(tiff_path) as src:
            # Transform from Dutch RD (EPSG:28992) to WGS84
            bounds_wgs84 = transform_bounds(
                src.crs, 
                CRS.from_epsg(4326),
                *src.bounds
            )
            all_bounds.append(bounds_wgs84)
    
    # Calculate overall extent
    min_lon = min(b[0] for b in all_bounds)
    max_lon = max(b[2] for b in all_bounds)
    min_lat = min(b[1] for b in all_bounds)
    max_lat = max(b[3] for b in all_bounds)
    
    print(f"\nAlphaEarth data extent (WGS84):")
    print(f"  Longitude: {min_lon:.3f}° to {max_lon:.3f}°")
    print(f"  Latitude: {min_lat:.3f}° to {max_lat:.3f}°")
    
    # Create coverage polygon
    coverage_polygon = box(min_lon, min_lat, max_lon, max_lat)
    coverage_gdf = gpd.GeoDataFrame(
        {'source': ['AlphaEarth'], 'geometry': [coverage_polygon]},
        crs='EPSG:4326'
    )
    
    return coverage_gdf


def create_h3_hexagons(boundary_gdf, resolutions=[10, 9, 8, 7, 6, 5]):
    """Create H3 hexagons at multiple resolutions using SRAI"""
    
    hexagon_gdfs = {}
    
    for resolution in resolutions:
        print(f"\nGenerating H3 hexagons at resolution {resolution}...")
        
        # Use SRAI H3Regionalizer
        regionalizer = H3Regionalizer(resolution=resolution)
        
        # Transform boundary to H3 regions
        h3_gdf = regionalizer.transform(boundary_gdf)
        
        # Add resolution info
        h3_gdf['h3_resolution'] = resolution
        
        # Calculate hexagon area
        if len(h3_gdf) > 0:
            sample_hex = h3_gdf.index[0]
            hex_area = h3.cell_area(sample_hex, unit='km^2')
            h3_gdf['area_km2'] = hex_area
        
        hexagon_gdfs[resolution] = h3_gdf
        
        print(f"  Created {len(h3_gdf):,} hexagons")
        print(f"  Approximate area per hexagon: {hex_area:.3f} km²")
        
        # Calculate total coverage
        total_area = len(h3_gdf) * hex_area
        print(f"  Total coverage: {total_area:,.0f} km²")
    
    return hexagon_gdfs


def save_data(boundary_gdf, hexagon_gdfs, coverage_gdf=None):
    """Save boundary and hexagon data to processed folder"""
    
    # Create output directory
    output_dir = Path('data/processed/h3_regions/netherlands')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Netherlands boundary
    boundary_path = output_dir / 'netherlands_boundary.geojson'
    boundary_gdf.to_file(boundary_path, driver='GeoJSON')
    print(f"\nSaved boundary to: {boundary_path}")
    
    # Save AlphaEarth coverage if available
    if coverage_gdf is not None:
        coverage_path = output_dir / 'alphaearth_coverage.geojson'
        coverage_gdf.to_file(coverage_path, driver='GeoJSON')
        print(f"Saved AlphaEarth coverage to: {coverage_path}")
    
    # Save hexagons for each resolution
    for resolution, h3_gdf in hexagon_gdfs.items():
        # Save as GeoParquet for efficiency
        hex_path = output_dir / f'netherlands_h3_res{resolution}.parquet'
        h3_gdf.to_parquet(hex_path)
        print(f"Saved {len(h3_gdf):,} hexagons at resolution {resolution} to: {hex_path}")
        
        # Also save a sample as GeoJSON for visualization (first 1000 hexagons)
        if len(h3_gdf) <= 10000:  # Only save as GeoJSON if manageable size
            sample_gdf = h3_gdf.head(1000)
            sample_path = output_dir / f'netherlands_h3_res{resolution}_sample.geojson'
            sample_gdf.to_file(sample_path, driver='GeoJSON')
    
    # Create summary statistics
    summary = []
    for resolution, h3_gdf in hexagon_gdfs.items():
        summary.append({
            'resolution': resolution,
            'hexagon_count': len(h3_gdf),
            'area_per_hex_km2': h3_gdf['area_km2'].iloc[0] if len(h3_gdf) > 0 else 0,
            'total_area_km2': len(h3_gdf) * (h3_gdf['area_km2'].iloc[0] if len(h3_gdf) > 0 else 0)
        })
    
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / 'h3_summary_stats.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary statistics to: {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


def main():
    """Main execution"""
    print("=" * 60)
    print("Netherlands H3 Hexagon Generation with SRAI")
    print("=" * 60)
    
    # Step 1: Get Netherlands boundary
    print("\n1. Loading Netherlands boundary...")
    netherlands_boundary = get_netherlands_boundary()
    print(f"   Boundary CRS: {netherlands_boundary.crs}")
    print(f"   Boundary area: {netherlands_boundary.geometry.area[0]:.2f} deg²")
    
    # Step 2: Verify AlphaEarth coverage
    print("\n2. Verifying AlphaEarth data coverage...")
    coverage_gdf = verify_alphaearth_coverage()
    
    if coverage_gdf is not None:
        # Check overlap
        overlap = gpd.overlay(netherlands_boundary, coverage_gdf, how='intersection')
        if len(overlap) > 0:
            overlap_pct = (overlap.geometry.area.sum() / netherlands_boundary.geometry.area.sum()) * 100
            print(f"   AlphaEarth covers ~{overlap_pct:.1f}% of Netherlands boundary")
    
    # Step 3: Create H3 hexagons
    print("\n3. Creating H3 hexagons at multiple resolutions...")
    hexagon_gdfs = create_h3_hexagons(
        netherlands_boundary,
        resolutions=[10, 9, 8, 7, 6, 5]
    )
    
    # Step 4: Save all data
    print("\n4. Saving data to processed folder...")
    save_data(netherlands_boundary, hexagon_gdfs, coverage_gdf)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()