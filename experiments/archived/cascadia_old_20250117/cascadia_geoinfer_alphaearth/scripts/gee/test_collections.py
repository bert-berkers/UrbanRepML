"""
Test script to check what satellite embedding collections are available.
"""

import ee

# Initialize Earth Engine
try:
    ee.Initialize(project='boreal-union-296021')
    print("Google Earth Engine initialized successfully")
except Exception as e:
    print(f"Failed to initialize Earth Engine: {e}")
    exit(1)

# Test different collection IDs that might work
test_collections = [
    "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
    "GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL",
    "GOOGLE/Research/open-buildings/v1/polygons",  # Test a known public collection
    "COPERNICUS/S2_SR_HARMONIZED",  # Test Sentinel-2 (known to work)
    "LANDSAT/LC08/C02/T1_L2"  # Test Landsat 8 (known to work)
]

print("\nTesting various collections:")
print("=" * 60)

for collection_id in test_collections:
    print(f"\nTesting: {collection_id}")
    try:
        collection = ee.ImageCollection(collection_id)
        size = collection.size().getInfo()
        
        if size > 0:
            print(f"  SUCCESS: {size} images")
            
            # Get first image info
            first_image = ee.Image(collection.first())
            bands = first_image.bandNames().getInfo()
            
            print(f"    Bands ({len(bands)}): {bands[:5] if len(bands) > 5 else bands}")
            
            # Check date range
            dates = collection.aggregate_array('system:time_start').getInfo()
            if dates:
                from datetime import datetime
                min_date = datetime.fromtimestamp(min(dates)/1000).strftime('%Y-%m-%d')
                max_date = datetime.fromtimestamp(max(dates)/1000).strftime('%Y-%m-%d')
                print(f"    Date range: {min_date} to {max_date}")
        else:
            print(f"  WARNING: Collection exists but is empty")
            
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            print(f"  ERROR: Collection not found")
        else:
            print(f"  ERROR: {error_msg}")

print("\n" + "=" * 60)
print("Collection test complete")