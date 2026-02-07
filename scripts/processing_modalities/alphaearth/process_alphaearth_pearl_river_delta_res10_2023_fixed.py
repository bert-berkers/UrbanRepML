#!/usr/bin/env python3
"""
FIXED Process AlphaEarth 2023 TIFFs for Pearl River Delta to H3 resolution 10.

Critical fixes applied:
1. min_pixels_per_hex = 1 (not 3) - appropriate for small res 10 hexagons
2. subtile_size = 128 (not 256) - better memory management
3. Proper intermediate directory structure
4. Comprehensive validation

This script follows CLAUDE.md principles:
- Uses SRAI for ALL H3 operations (never h3 directly)
- Study-area based processing
- Maps AlphaEarth embeddings to pre-computed H3 regions
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from datetime import datetime
import time
import json
import warnings
import shutil
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv('keys/.env')

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import the AlphaEarthProcessor
from stage1_modalities.alphaearth.processor import AlphaEarthProcessor

# Set up comprehensive logging
def setup_logging():
    """Setup detailed logging for Pearl River Delta resolution 10 FIXED processing."""

    # Create logs directory at project root
    project_root = Path(__file__).parent.parent.parent.parent
    log_dir = project_root / 'logs' / 'alphaearth'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pearl_river_delta_res10_FIXED_{timestamp}.log'

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    return logger


def log_system_info(logger):
    """Log system information for debugging."""

    try:
        import psutil
        logger.info("SYSTEM INFORMATION")
        logger.info(f"  CPU cores: {os.cpu_count()}")
        logger.info(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        logger.info(f"  Total memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        logger.info("SYSTEM INFORMATION")
        logger.info(f"  CPU cores: {os.cpu_count()}")
        logger.info("  Memory info: psutil not available")

    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  Working directory: {Path.cwd()}")


def load_h3_regions(study_area: str, resolution: int, logger):
    """Load pre-computed H3 regions from SRAI processing.

    CRITICAL: We always use pre-computed regions from SRAI, never generate on the fly!
    """
    # Navigate to project root (4 levels up from scripts/processing_modalities/alphaearth/)
    project_root = Path(__file__).parent.parent.parent.parent
    regions_file = project_root / f'data/study_areas/{study_area}/regions_gdf/h3_res{resolution}.parquet'

    if not regions_file.exists():
        raise FileNotFoundError(
            f"H3 regions file not found: {regions_file}\n"
            f"Run setup_pearl_river_delta.py first to create SRAI regions!"
        )

    logger.info(f"Loading SRAI H3 regions from: {regions_file}")
    regions_gdf = gpd.read_parquet(regions_file)
    logger.info(f"  Loaded {len(regions_gdf):,} H3 hexagons at resolution {resolution}")

    # Validate resolution
    if len(regions_gdf) < 2000000:  # Resolution 10 should have ~3M hexagons
        logger.warning(f"  WARNING: Only {len(regions_gdf):,} regions loaded - expected ~3M for resolution 10")

    # Log bounds
    bounds = regions_gdf.total_bounds
    logger.info(f"  Geographic bounds:")
    logger.info(f"    West: {bounds[0]:.3f}°, East: {bounds[2]:.3f}°")
    logger.info(f"    South: {bounds[1]:.3f}°, North: {bounds[3]:.3f}°")

    return regions_gdf


def clean_intermediate_directory(intermediate_dir: Path, logger):
    """Clean intermediate directory to ensure fresh processing."""
    if intermediate_dir.exists():
        logger.info(f"Cleaning existing intermediate directory: {intermediate_dir}")
        shutil.rmtree(intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created clean intermediate directory: {intermediate_dir}")


def process_alphaearth_prd_fixed(logger):
    """FIXED: Process AlphaEarth 2023 TIFFs for Pearl River Delta to H3 resolution 10."""

    logger.info("="*60)
    logger.info("STARTING FIXED PEARL RIVER DELTA ALPHAEARTH PROCESSING (RES 10)")
    logger.info("="*60)
    logger.info("CRITICAL FIXES APPLIED:")
    logger.info("  1. min_pixels_per_hex = 1 (was 3 - too high for res 10!)")
    logger.info("  2. subtile_size = 128 (was 256 - better memory management)")
    logger.info("  3. Clean intermediate directory structure")
    logger.info("  4. Proper parameter passing to processor")
    logger.info("="*60)

    start_time = time.time()

    # FIXED Configuration for Pearl River Delta at resolution 10
    config = {
        'source_dir': 'G:/My Drive/AlphaEarth_PRD/',  # Direct path to PRD data
        'subtile_size': 128,  # FIXED: Smaller tiles for resolution 10 (was 256)
        'min_pixels_per_hex': 1,  # FIXED: Critical change from 3 to 1!
        'max_workers': 8,  # Slightly reduced for stability
    }

    # Study area parameters
    study_area = 'pearl_river_delta'
    h3_resolution = 10
    year_filter = '2023'

    logger.info("FIXED PROCESSING CONFIGURATION")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  h3_resolution: {h3_resolution}")
    logger.info(f"  year_filter: {year_filter}")
    logger.info(f"  study_area: {study_area}")

    # Verify source directory
    source_path = Path(config['source_dir'])
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_path}")
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    logger.info(f"Source directory verified: {source_path}")

    # Count 2023 TIFF files
    tiff_pattern = f"*{year_filter}*.tif"
    tiff_files = list(source_path.glob(tiff_pattern))
    logger.info(f"Found {len(tiff_files)} TIFF files for {year_filter}")

    if not tiff_files:
        logger.error(f"No {year_filter} TIFF files found")
        raise ValueError(f"No {year_filter} TIFF files found")

    # Log file statistics
    total_size = sum(f.stat().st_size for f in tiff_files)
    avg_size = total_size / len(tiff_files) if tiff_files else 0
    logger.info(f"Total input data size: {total_size / (1024**3):.1f} GB")
    logger.info(f"Average file size: {avg_size / (1024**2):.1f} MB")

    # Load pre-computed H3 regions (CRITICAL: using SRAI regions!)
    regions_gdf = load_h3_regions(study_area, h3_resolution, logger)

    # Clean and prepare intermediate directory using new structure
    project_root = Path(__file__).parent.parent.parent.parent
    intermediate_base = project_root / 'data' / 'processed' / 'intermediate embeddings stage1_modalities' / 'alphaearth'
    intermediate_base.mkdir(parents=True, exist_ok=True)

    # Create study-area and resolution specific subdirectory
    intermediate_dir = intermediate_base / study_area / f'res{h3_resolution}_{year_filter}_FIXED'
    clean_intermediate_directory(intermediate_dir, logger)

    # Initialize processor with fixed configuration
    logger.info("Initializing AlphaEarthProcessor with FIXED parameters...")
    processor = AlphaEarthProcessor(config)

    # Process to resolution 10
    logger.info("STARTING FIXED TILE PROCESSING")
    logger.info(f"  Study area: Pearl River Delta")
    logger.info(f"  Target resolution: {h3_resolution}")
    logger.info(f"  Expected hexagon area: ~0.015 km²")  # Res 10 hexagon area
    logger.info(f"  Expected pixels per hexagon: ~150")  # At 10m AlphaEarth resolution
    logger.info(f"  Processing {len(tiff_files)} tiles...")
    logger.info(f"  Expected output: ~2.7M hexagons (49x more than res 8)")

    processing_start = time.time()

    # Process with explicit parameters and pre-computed regions
    try:
        # Override the processor's intermediate directory handling
        # Save original config
        original_intermediate = config.get('intermediate_dir', None)

        # Set our fixed intermediate directory
        config['intermediate_dir'] = str(intermediate_dir)

        gdf = processor.process(
            raw_data_path=source_path,
            year_filter=year_filter,
            h3_resolution=h3_resolution,
            regions_gdf=regions_gdf  # Pass pre-computed SRAI regions
        )

        # Restore original config if needed
        if original_intermediate:
            config['intermediate_dir'] = original_intermediate
        else:
            config.pop('intermediate_dir', None)

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise

    processing_end = time.time()
    processing_duration = processing_end - processing_start

    logger.info("PROCESSING COMPLETED")
    logger.info(f"  Duration: {processing_duration:.1f} seconds ({processing_duration/60:.1f} minutes)")
    logger.info(f"  Hexagons generated: {len(gdf):,}")

    # CRITICAL VALIDATION: Check if we got the expected number of hexagons
    expected_min = 2000000  # At least 2M hexagons for resolution 10
    expected_max = 3500000  # At most 3.5M hexagons

    if len(gdf) < expected_min:
        logger.error(f"⚠️ WARNING: Only {len(gdf):,} hexagons generated!")
        logger.error(f"  Expected: {expected_min:,} - {expected_max:,} hexagons")
        logger.error("  This indicates the resolution 10 processing may still have issues!")
    else:
        logger.info(f"✅ SUCCESS: Generated {len(gdf):,} hexagons - within expected range!")

    # Compare with resolution 8 baseline
    res8_baseline = 55331  # Known resolution 8 count
    ratio = len(gdf) / res8_baseline
    logger.info(f"  Ratio vs resolution 8: {ratio:.1f}x (expected: ~49x)")

    if ratio < 30:
        logger.warning(f"⚠️ Low ratio ({ratio:.1f}x) - processing may need review")

    logger.info(f"  Processing rate: {len(gdf) / processing_duration:.0f} hexagons/second")

    # Log embedding statistics
    embedding_cols = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
    logger.info(f"  Embedding dimensions: {len(embedding_cols)}")

    if embedding_cols:
        embedding_data = gdf[embedding_cols]
        logger.info(f"  Embedding value range: [{embedding_data.min().min():.3f}, {embedding_data.max().max():.3f}]")
        logger.info(f"  Embedding mean: {embedding_data.mean().mean():.3f}")
        logger.info(f"  Embedding std: {embedding_data.std().mean():.3f}")

    # Log pixel count statistics
    if 'pixel_count' in gdf.columns:
        logger.info(f"  Pixel count range: [{gdf['pixel_count'].min()}, {gdf['pixel_count'].max()}]")
        mean_pixels = gdf['pixel_count'].mean()
        logger.info(f"  Average pixels per hexagon: {mean_pixels:.1f}")

        if mean_pixels > 1000:
            logger.warning(f"⚠️ High average pixel count ({mean_pixels:.1f}) - unexpected for res 10")

    return gdf, processing_duration


def save_processed_data(gdf, processing_duration, logger):
    """Save processed data following study area structure."""

    logger.info("SAVING PROCESSED DATA")

    # Create output directory in data/study_areas structure
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / 'data/study_areas/pearl_river_delta/embeddings/alphaearth'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main embeddings file with FIXED suffix
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'prd_res10_2023_FIXED_{timestamp}.parquet'
    logger.info(f"  Saving to: {output_file}")

    save_start = time.time()
    gdf.to_parquet(output_file)
    save_duration = time.time() - save_start

    file_size = output_file.stat().st_size
    logger.info(f"  File saved in {save_duration:.1f} seconds")
    logger.info(f"  File size: {file_size / (1024**2):.1f} MB")

    # Create comprehensive metadata file
    metadata = {
        'processing_date': datetime.now().isoformat(),
        'processing_duration_seconds': processing_duration,
        'study_area': 'pearl_river_delta',
        'hexagon_count': len(gdf),
        'h3_resolution': 10,
        'year': 2023,
        'region': 'Pearl River Delta, China',
        'coverage_area_km2': len(gdf) * 0.015,  # Accurate area for res 10
        'embedding_dimensions': len([col for col in gdf.columns if col.startswith('A')]),
        'file_size_mb': file_size / (1024**2),
        'source_tiles': 114,  # From our count
        'coordinate_system': 'EPSG:4326',
        'processor_version': 'AlphaEarthProcessor v1.0 FIXED',
        'fixes_applied': [
            'min_pixels_per_hex set to 1 (was 3)',
            'subtile_size set to 128 (was 256)',
            'Clean intermediate directory',
            'Proper parameter passing'
        ],
        'configuration': {
            'subtile_size': 128,
            'min_pixels_per_hex': 1,  # CRITICAL FIX
            'max_workers': 8,
            'source_dir': 'G:/My Drive/AlphaEarth_PRD/'
        },
        'validation': {
            'expected_hexagons': '2M-3.5M',
            'actual_hexagons': len(gdf),
            'ratio_vs_res8': len(gdf) / 55331,
            'validation_passed': len(gdf) >= 2000000
        },
        'notes': [
            'FIXED version with correct resolution 10 parameters',
            'Processed using SRAI H3Regionalizer per CLAUDE.md',
            'Pre-computed H3 regions from setup_pearl_river_delta.py',
            'Never used h3 directly, only SRAI',
            'Critical fix: min_pixels_per_hex reduced from 3 to 1'
        ]
    }

    # Add embedding statistics
    embedding_cols = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
    if embedding_cols:
        embedding_data = gdf[embedding_cols]
        metadata['embedding_statistics'] = {
            'min_value': float(embedding_data.min().min()),
            'max_value': float(embedding_data.max().max()),
            'mean_value': float(embedding_data.mean().mean()),
            'std_value': float(embedding_data.std().mean())
        }

    # Add pixel count statistics
    if 'pixel_count' in gdf.columns:
        metadata['pixel_statistics'] = {
            'min_pixels': int(gdf['pixel_count'].min()),
            'max_pixels': int(gdf['pixel_count'].max()),
            'mean_pixels': float(gdf['pixel_count'].mean()),
            'total_pixels': int(gdf['pixel_count'].sum())
        }

    # Save metadata
    metadata_file = output_dir / f'prd_res10_2023_FIXED_{timestamp}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata saved to: {metadata_file}")

    return output_file, metadata_file


def validate_output(gdf, logger):
    """Validate the processed data quality with FIXED expectations."""

    logger.info("VALIDATING OUTPUT DATA (FIXED CRITERIA)")

    # Check for required columns
    required_cols = ['geometry'] if 'geometry' in gdf.columns else []
    h3_col = 'h3_index' if 'h3_index' in gdf.columns else gdf.index.name

    logger.info(f"  Total hexagons: {len(gdf):,}")
    logger.info(f"  H3 index column: {h3_col}")

    # CRITICAL: Validate hexagon count
    res8_baseline = 55331
    expected_ratio_min = 40  # At least 40x more than res 8
    expected_ratio_max = 60  # At most 60x more than res 8
    actual_ratio = len(gdf) / res8_baseline

    logger.info(f"  Resolution 8 baseline: {res8_baseline:,} hexagons")
    logger.info(f"  Resolution 10 actual: {len(gdf):,} hexagons")
    logger.info(f"  Ratio: {actual_ratio:.1f}x (expected: {expected_ratio_min}x-{expected_ratio_max}x)")

    hexagon_count_valid = expected_ratio_min <= actual_ratio <= expected_ratio_max
    if not hexagon_count_valid:
        logger.error(f"  ❌ HEXAGON COUNT VALIDATION FAILED!")
        logger.error(f"     Expected: {res8_baseline * expected_ratio_min:,} - {res8_baseline * expected_ratio_max:,}")
        logger.error(f"     Actual: {len(gdf):,}")
    else:
        logger.info(f"  ✅ Hexagon count validation PASSED")

    # Check embedding completeness
    embedding_cols = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
    if embedding_cols:
        nan_count = gdf[embedding_cols].isna().sum().sum()
        total_values = len(gdf) * len(embedding_cols)
        completeness = (total_values - nan_count) / total_values if total_values > 0 else 0
        logger.info(f"  Embedding completeness: {completeness:.1%} ({total_values - nan_count}/{total_values} values)")
        completeness_valid = completeness > 0.95
    else:
        completeness_valid = False

    # Check geometry validity if present
    if 'geometry' in gdf.columns:
        valid_geom = gdf.geometry.is_valid.sum()
        geom_validity = valid_geom / len(gdf)
        logger.info(f"  Geometry validity: {geom_validity:.1%} ({valid_geom}/{len(gdf)} geometries)")

        # Check coordinate range (should be within Pearl River Delta bounds)
        bounds = gdf.total_bounds
        logger.info(f"  Geographic bounds:")
        logger.info(f"    West: {bounds[0]:.3f}°, East: {bounds[2]:.3f}°")
        logger.info(f"    South: {bounds[1]:.3f}°, North: {bounds[3]:.3f}°")

        # Validate bounds are within expected PRD range
        expected_bounds = {
            'west': 112.5, 'east': 114.5,
            'south': 21.5, 'north': 23.5
        }
        bounds_valid = (
            bounds[0] >= expected_bounds['west'] - 0.5 and
            bounds[2] <= expected_bounds['east'] + 0.5 and
            bounds[1] >= expected_bounds['south'] - 0.5 and
            bounds[3] <= expected_bounds['north'] + 0.5
        )
        logger.info(f"  Bounds validation: {'PASSED' if bounds_valid else 'WARNING - outside expected PRD range'}")
    else:
        geom_validity = 1.0
        bounds_valid = True

    # Check pixel statistics
    if 'pixel_count' in gdf.columns:
        mean_pixels = gdf['pixel_count'].mean()
        logger.info(f"  Mean pixels per hexagon: {mean_pixels:.1f}")

        # For resolution 10, expect ~150 pixels per hexagon
        pixel_count_valid = 50 <= mean_pixels <= 500
        if not pixel_count_valid:
            logger.warning(f"  ⚠️ Unusual pixel count: {mean_pixels:.1f} (expected: 50-500)")
    else:
        pixel_count_valid = True

    # Overall validation
    overall_valid = (
        hexagon_count_valid and
        completeness_valid and
        (geom_validity > 0.95) and
        bounds_valid and
        pixel_count_valid
    )

    if overall_valid:
        logger.info("="*60)
        logger.info("  🎉 VALIDATION PASSED: Resolution 10 processing SUCCESSFUL!")
        logger.info("="*60)
    else:
        logger.warning("="*60)
        logger.warning("  ⚠️ VALIDATION WARNING: Some issues detected")
        logger.warning("  Please review the logs above")
        logger.warning("="*60)

    return overall_valid


def main():
    """Main execution pipeline."""

    try:
        # Setup logging
        logger = setup_logging()

        # Log session start
        logger.info("FIXED PEARL RIVER DELTA ALPHAEARTH PROCESSING SESSION")
        logger.info(f"Session start: {datetime.now()}")
        logger.info("Following CLAUDE.md principles: SRAI for all H3 operations")
        log_system_info(logger)

        # Process AlphaEarth data with FIXED parameters
        gdf, processing_duration = process_alphaearth_prd_fixed(logger)

        # Validate output with FIXED criteria
        validation_passed = validate_output(gdf, logger)

        # Save data with metadata
        output_file, metadata_file = save_processed_data(gdf, processing_duration, logger)

        # Final summary
        total_duration = time.time() - processing_duration
        logger.info("="*60)
        logger.info("FIXED PROCESSING SESSION COMPLETE")
        logger.info("="*60)
        logger.info(f"Study area: Pearl River Delta")
        logger.info(f"Resolution: 10 (FIXED)")
        logger.info(f"Total session duration: {total_duration:.1f} seconds")
        logger.info(f"Hexagons processed: {len(gdf):,}")
        logger.info(f"Ratio vs res 8: {len(gdf)/55331:.1f}x")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Metadata file: {metadata_file}")
        logger.info(f"Data validation: {'✅ PASSED' if validation_passed else '⚠️ WARNING'}")

        if len(gdf) >= 2000000:
            logger.info("")
            logger.info("🎉 SUCCESS: Resolution 10 processing completed with expected high resolution!")
            logger.info("   The Pearl River Delta now has proper ~2.7M hexagon coverage")
        else:
            logger.info("")
            logger.info("⚠️ WARNING: Resolution 10 processing completed but with fewer hexagons than expected")
            logger.info("   Please review the logs for potential issues")

        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review the processed embeddings and metadata")
        logger.info("2. Visualize sample hexagons to verify dense coverage")
        logger.info("3. Compare with resolution 8 to confirm improved detail")
        logger.info("4. Process other stage1_modalities at resolution 10 if needed")

        return 0

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Processing failed: {e}")
            logger.error("Full traceback:", exc_info=True)
        else:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())