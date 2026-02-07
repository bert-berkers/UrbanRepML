#!/usr/bin/env python3
"""
Test script to verify Pearl River Delta Resolution 10 setup before full processing.
"""

import sys
from pathlib import Path
import geopandas as gpd

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def test_setup():
    """Test the setup for Pearl River Delta Resolution 10 processing."""

    print("="*60)
    print("TESTING PEARL RIVER DELTA RESOLUTION 10 SETUP")
    print("="*60)

    # 1. Check H3 regions file
    print("\n1. Checking H3 regions file...")
    project_root = Path(__file__).parent.parent.parent.parent
    regions_file = project_root / 'data/study_areas/pearl_river_delta/regions_gdf/h3_res10.parquet'

    if regions_file.exists():
        regions_gdf = gpd.read_parquet(regions_file)
        print(f"   [OK] H3 regions file exists")
        print(f"   - Total hexagons: {len(regions_gdf):,}")
        print(f"   - File size: {regions_file.stat().st_size / (1024**2):.1f} MB")

        if len(regions_gdf) < 2000000:
            print(f"   [WARNING] Only {len(regions_gdf):,} hexagons - expected ~3M")
        else:
            print(f"   [OK] Hexagon count looks good for resolution 10")
    else:
        print(f"   [ERROR] H3 regions file NOT FOUND: {regions_file}")
        return False

    # 2. Check source TIFFs
    print("\n2. Checking AlphaEarth source TIFFs...")
    source_path = Path('G:/My Drive/AlphaEarth_PRD/')

    if source_path.exists():
        tiff_files = list(source_path.glob('*2023*.tif'))
        print(f"   [OK] Source directory exists")
        print(f"   - TIFF files found: {len(tiff_files)}")

        if len(tiff_files) == 0:
            print(f"   [ERROR] No 2023 TIFF files found")
            return False

        # Sample first few files
        for f in tiff_files[:3]:
            print(f"   - {f.name} ({f.stat().st_size / (1024**2):.1f} MB)")
        if len(tiff_files) > 3:
            print(f"   ... and {len(tiff_files) - 3} more files")
    else:
        print(f"   [ERROR] Source directory NOT FOUND: {source_path}")
        print(f"   Please update the path in the processing script")
        return False

    # 3. Check intermediate directory structure
    print("\n3. Checking intermediate directory structure...")
    intermediate_base = project_root / 'data/study_areas/netherlands/embeddings/intermediate/alphaearth'

    if intermediate_base.exists():
        print(f"   [OK] Intermediate base directory exists")
    else:
        print(f"   [INFO] Intermediate directory will be created during processing")

    # 4. Check existing outputs
    print("\n4. Checking existing outputs...")
    output_dir = project_root / 'data/study_areas/pearl_river_delta/embeddings/alphaearth'

    if output_dir.exists():
        existing_files = list(output_dir.glob('*.parquet'))
        print(f"   Output directory exists with {len(existing_files)} parquet files")

        # Check if we have previous res 10 attempts
        res10_files = [f for f in existing_files if 'res10' in f.name]
        if res10_files:
            print(f"   Found {len(res10_files)} previous res10 attempts:")
            for f in res10_files:
                print(f"   - {f.name} ({f.stat().st_size / (1024**2):.1f} MB)")

                # Load and check hexagon count
                try:
                    test_gdf = gpd.read_parquet(f)
                    print(f"     Hexagons: {len(test_gdf):,}")
                    if len(test_gdf) < 100000:
                        print(f"     [WARNING] Low hexagon count - likely failed processing")
                except:
                    pass
    else:
        print(f"   Output directory will be created during processing")

    # 5. Test configuration
    print("\n5. Testing configuration...")
    print("   FIXED parameters for Resolution 10:")
    print("   - min_pixels_per_hex: 1 (critical!)")
    print("   - subtile_size: 128")
    print("   - h3_resolution: 10")
    print("   - max_workers: 8")

    # Summary
    print("\n" + "="*60)
    print("SETUP TEST SUMMARY")
    print("="*60)

    all_good = regions_file.exists() and source_path.exists() and len(tiff_files) > 0

    if all_good:
        print("[OK] All checks passed! Ready to run fixed processing.")
        print("\nRun the fixed processing script with:")
        print("python scripts/processing_modalities/alphaearth/process_alphaearth_pearl_river_delta_res10_2023_fixed.py")
    else:
        print("[ERROR] Some issues found. Please fix before running processing.")

    return all_good


if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)