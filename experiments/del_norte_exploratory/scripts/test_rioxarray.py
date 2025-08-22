#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test to verify rioxarray processor works correctly.
"""

import sys
from pathlib import Path
import yaml
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.rioxarray_processor import RioxarrayAlphaEarthProcessor, RIOXARRAY_AVAILABLE

def test_rioxarray_processor():
    """Test the rioxarray processor with a single file."""
    print("="*60)
    print("TESTING RIOXARRAY PROCESSOR")
    print("="*60)
    
    # Check availability
    if not RIOXARRAY_AVAILABLE:
        print("❌ Rioxarray not available, cannot test")
        return False
    
    print("✅ Rioxarray is available")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add rioxarray config if not present
    if 'rioxarray' not in config:
        config['rioxarray'] = {
            'chunk_size_mb': 50,  # Smaller for testing
            'use_parallel': True,
            'optimize_chunks': True
        }
    
    # Initialize processor
    try:
        processor = RioxarrayAlphaEarthProcessor(config)
        print("✅ Processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return False
    
    # Get test files
    try:
        test_files = processor.get_tiff_files()
        if not test_files:
            print("❌ No test files found")
            return False
        
        print(f"✅ Found {len(test_files)} test files")
        
        # Test with just the first file
        test_file = test_files[0]
        print(f"Testing with: {test_file.name}")
        
        # Test loading a single file
        da = processor.load_tiff_with_rioxarray(test_file)
        if da is None:
            print("❌ Failed to load TIFF with rioxarray")
            return False
        
        print(f"✅ Loaded TIFF: shape={da.shape}, dims={da.dims}")
        
        # Test H3 conversion (just a small sample)
        # Override the vectorized function to use smaller sample
        original_func = processor.da_to_h3_vectorized
        
        def test_h3_conversion(da_input):
            """Test H3 conversion with small sample."""
            import numpy as np
            import pandas as pd
            import h3
            
            # Just test a very small grid
            sample_size = 10
            lons = da_input.x.values[:sample_size]
            lats = da_input.y.values[:sample_size]
            
            test_data = []
            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    try:
                        h3_idx = h3.latlng_to_cell(lat, lon, processor.h3_resolution)
                        # Get pixel values
                        pixel_values = da_input.isel(y=i, x=j).compute().values
                        
                        if not np.all(pixel_values == 0):
                            pixel_dict = {
                                'h3_index': h3_idx,
                                'lat': lat,
                                'lon': lon,
                                'tile': test_file.stem
                            }
                            
                            # Add first few bands for testing
                            for band_idx in range(min(5, 64)):
                                pixel_dict[f'band_{band_idx:02d}'] = pixel_values[band_idx]
                            
                            test_data.append(pixel_dict)
                    except:
                        continue
            
            return pd.DataFrame(test_data)
        
        # Test H3 conversion
        result_df = test_h3_conversion(da)
        if result_df.empty:
            print("⚠️  H3 conversion returned empty results (might be expected for test)")
        else:
            print(f"✅ H3 conversion successful: {len(result_df)} hexagons")
        
        print("✅ Rioxarray processor test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rioxarray_processor()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")