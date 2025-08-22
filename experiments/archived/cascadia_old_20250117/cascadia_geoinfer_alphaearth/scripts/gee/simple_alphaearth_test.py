"""
Simple test for AlphaEarth collection access.
"""

import ee

# Initialize Earth Engine
try:
    ee.Initialize(project='boreal-union-296021')
    print("Google Earth Engine initialized successfully")
except Exception as e:
    print(f"Failed to initialize Earth Engine: {e}")
    exit(1)

# Test the specific AlphaEarth collection from the documentation
collection_id = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

print(f"\nTesting AlphaEarth collection: {collection_id}")
print("=" * 60)

try:
    # Try to access the collection
    collection = ee.ImageCollection(collection_id)
    size = collection.size().getInfo()
    
    print(f"Collection size: {size} images")
    
    if size > 0:
        # Get first image
        first_image = ee.Image(collection.first())
        
        # Get bands
        bands = first_image.bandNames().getInfo()
        print(f"Number of bands: {len(bands)}")
        print(f"Band names (first 10): {bands[:10]}")
        
        # Get projection info
        projection = first_image.projection().getInfo()
        print(f"Resolution: {projection.get('nominalScale', 'Unknown')}m")
        print(f"CRS: {projection.get('crs', 'Unknown')}")
        
        # Check temporal coverage for Cascadia region
        cascadia_bounds = ee.Geometry.Rectangle([-124.6, 39.0, -116.5, 46.3])
        
        # Filter by region and get date range
        regional_collection = collection.filterBounds(cascadia_bounds)
        regional_size = regional_collection.size().getInfo()
        
        print(f"\nCascadia region coverage: {regional_size} images")
        
        if regional_size > 0:
            # Get date range for Cascadia
            dates = regional_collection.aggregate_array('system:time_start').getInfo()
            if dates:
                from datetime import datetime
                min_date = datetime.fromtimestamp(min(dates)/1000).strftime('%Y-%m-%d')
                max_date = datetime.fromtimestamp(max(dates)/1000).strftime('%Y-%m-%d')
                print(f"Date range: {min_date} to {max_date}")
                
                # Check specific years
                available_years = []
                for year in range(2017, 2025):
                    start_date = f"{year}-01-01"
                    end_date = f"{year+1}-01-01"
                    
                    yearly = regional_collection.filterDate(start_date, end_date)
                    yearly_size = yearly.size().getInfo()
                    
                    if yearly_size > 0:
                        available_years.append(year)
                        print(f"  {year}: {yearly_size} images")
                
                print(f"\nAvailable years: {available_years}")
        else:
            print("No images found for Cascadia region")
    
    else:
        print("Collection is empty")
        
except Exception as e:
    error_msg = str(e)
    if "Collection asset" in error_msg and "does not exist" in error_msg:
        print("ERROR: Collection does not exist or access denied")
        print("This could mean:")
        print("  1. The collection requires special permissions")
        print("  2. The collection ID has changed")
        print("  3. The collection is not available in your region/account")
    else:
        print(f"ERROR: {error_msg}")

print("\nTest complete")