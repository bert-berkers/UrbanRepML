#!/usr/bin/env python3
"""
Process AlphaEarth 2023 TIFFs for Pearl River Delta to H3 resolution 8.

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
    """Setup detailed logging for Pearl River Delta processing."""

    # Create logs directory at project root
    project_root = Path(__file__).parent.parent.parent.parent
    log_dir = project_root / 'logs' / 'alphaearth'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pearl_river_delta_res8_2023_{timestamp}.log'

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
    logger.info(f"  Loaded {len(regions_gdf)} H3 hexagons at resolution {resolution}")

    # Log bounds
    bounds = regions_gdf.total_bounds
    logger.info(f"  Geographic bounds:")
    logger.info(f"    West: {bounds[0]:.3f}°, East: {bounds[2]:.3f}°")
    logger.info(f"    South: {bounds[1]:.3f}°, North: {bounds[3]:.3f}°")

    return regions_gdf


def process_alphaearth_prd(logger):
    """Process AlphaEarth 2023 TIFFs for Pearl River Delta to H3 resolution 8."""

    logger.info("="*60)
    logger.info("STARTING PEARL RIVER DELTA ALPHAEARTH PROCESSING")
    logger.info("="*60)

    start_time = time.time()

    # Configuration for Pearl River Delta at resolution 8
    config = {
        'source_dir': 'G:/My Drive/AlphaEarth_PRD/',  # Direct path to PRD data
        'subtile_size': 512,  # Larger tiles for resolution 8 (vs 256 for res 10)
        'min_pixels_per_hex': 5,  # Standard threshold for res 8
        'max_workers': 10,  # Parallel processing
        'intermediate_dir': 'intermediate_prd_res8'  # Separate intermediate embeddings stage1_modalities directory
    }

    logger.info("PROCESSING CONFIGURATION")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Verify source directory
    source_path = Path(config['source_dir'])
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_path}")
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    logger.info(f"Source directory verified: {source_path}")

    # Count 2023 TIFF files
    tiff_pattern = "*2023*.tif"
    tiff_files = list(source_path.glob(tiff_pattern))
    logger.info(f"Found {len(tiff_files)} TIFF files for 2023")

    if not tiff_files:
        logger.error("No 2023 TIFF files found")
        raise ValueError("No 2023 TIFF files found")

    # Log file statistics
    total_size = sum(f.stat().st_size for f in tiff_files)
    avg_size = total_size / len(tiff_files) if tiff_files else 0
    logger.info(f"Total input data size: {total_size / (1024**3):.1f} GB")
    logger.info(f"Average file size: {avg_size / (1024**2):.1f} MB")

    # Load pre-computed H3 regions (CRITICAL: using SRAI regions!)
    h3_resolution = 8
    regions_gdf = load_h3_regions('pearl_river_delta', h3_resolution, logger)

    # Initialize processor
    logger.info("Initializing AlphaEarthProcessor...")
    processor = AlphaEarthProcessor(config)

    # Process to resolution 8
    logger.info("STARTING TILE PROCESSING")
    logger.info(f"  Study area: Pearl River Delta")
    logger.info(f"  Target resolution: {h3_resolution}")
    logger.info(f"  Expected hexagon area: ~0.6 km²")  # Res 8 hexagon area
    logger.info(f"  Processing {len(tiff_files)} tiles...")

    processing_start = time.time()

    # Process with explicit study area and pre-computed regions
    gdf = processor.process(
        raw_data_path=source_path,
        year_filter='2023',
        h3_resolution=h3_resolution,
        intermediate_dir=config['intermediate_dir'],
        regions_gdf=regions_gdf  # Pass pre-computed SRAI regions
    )

    processing_end = time.time()
    processing_duration = processing_end - processing_start

    logger.info("PROCESSING COMPLETED")
    logger.info(f"  Duration: {processing_duration:.1f} seconds ({processing_duration/60:.1f} minutes)")
    logger.info(f"  Hexagons generated: {len(gdf):,}")
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
        logger.info(f"  Average pixels per hexagon: {gdf['pixel_count'].mean():.1f}")

    return gdf, processing_duration


def save_processed_data(gdf, processing_duration, logger):
    """Save processed data following study area structure."""

    logger.info("SAVING PROCESSED DATA")

    # Create output directory in data/study_areas structure
    # Navigate to project root (4 levels up from scripts/processing_modalities/alphaearth/)
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / 'data/study_areas/pearl_river_delta/embeddings/alphaearth'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main embeddings file
    output_file = output_dir / 'prd_res8_2023.parquet'
    logger.info(f"  Saving to: {output_file}")

    save_start = time.time()
    gdf.to_parquet(output_file)
    save_duration = time.time() - save_start

    file_size = output_file.stat().st_size
    logger.info(f"  File saved in {save_duration:.1f} seconds")
    logger.info(f"  File size: {file_size / (1024**2):.1f} MB")

    # Create metadata file
    metadata = {
        'processing_date': datetime.now().isoformat(),
        'processing_duration_seconds': processing_duration,
        'study_area': 'pearl_river_delta',
        'hexagon_count': len(gdf),
        'h3_resolution': 8,
        'year': 2023,
        'region': 'Pearl River Delta, China',
        'coverage_area_km2': len(gdf) * 0.6,  # Approximate area for res 8
        'embedding_dimensions': len([col for col in gdf.columns if col.startswith('A')]),
        'file_size_mb': file_size / (1024**2),
        'source_tiles': 114,  # From our count
        'coordinate_system': 'EPSG:4326',
        'processor_version': 'AlphaEarthProcessor v1.0',
        'configuration': {
            'subtile_size': 512,
            'min_pixels_per_hex': 5,
            'max_workers': 10,
            'source_dir': 'G:/My Drive/AlphaEarth_PRD/'
        },
        'notes': [
            'Processed using SRAI H3Regionalizer per CLAUDE.md',
            'Pre-computed H3 regions from setup_pearl_river_delta.py',
            'Never used h3 directly, only SRAI'
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
    metadata_file = output_dir / 'prd_res8_2023_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata saved to: {metadata_file}")

    return output_file, metadata_file


def validate_output(gdf, logger):
    """Validate the processed data quality."""

    logger.info("VALIDATING OUTPUT DATA")

    # Check for required columns
    required_cols = ['geometry'] if 'geometry' in gdf.columns else []
    h3_col = 'h3_index' if 'h3_index' in gdf.columns else gdf.index.name

    logger.info(f"  Total hexagons: {len(gdf):,}")
    logger.info(f"  H3 index column: {h3_col}")

    # Check embedding completeness
    embedding_cols = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
    if embedding_cols:
        nan_count = gdf[embedding_cols].isna().sum().sum()
        total_values = len(gdf) * len(embedding_cols)
        completeness = (total_values - nan_count) / total_values if total_values > 0 else 0
        logger.info(f"  Embedding completeness: {completeness:.1%} ({total_values - nan_count}/{total_values} values)")

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

    # Overall validation
    overall_valid = (
        len(gdf) > 0 and
        (not embedding_cols or completeness > 0.95) and
        ('geometry' not in gdf.columns or geom_validity > 0.95)
    )

    if overall_valid:
        logger.info("  VALIDATION PASSED: Data quality meets standards")
    else:
        logger.warning("  VALIDATION WARNING: Some quality issues detected")

    return overall_valid


def main():
    """Main execution pipeline."""

    try:
        # Setup logging
        logger = setup_logging()

        # Log session start
        logger.info("PEARL RIVER DELTA ALPHAEARTH PROCESSING SESSION")
        logger.info(f"Session start: {datetime.now()}")
        logger.info("Following CLAUDE.md principles: SRAI for all H3 operations")
        log_system_info(logger)

        # Process AlphaEarth data
        gdf, processing_duration = process_alphaearth_prd(logger)

        # Validate output
        validation_passed = validate_output(gdf, logger)

        # Save data with metadata
        output_file, metadata_file = save_processed_data(gdf, processing_duration, logger)

        # Final summary
        total_duration = time.time() - processing_duration
        logger.info("="*60)
        logger.info("PROCESSING SESSION COMPLETE")
        logger.info("="*60)
        logger.info(f"Study area: Pearl River Delta")
        logger.info(f"Total session duration: {total_duration:.1f} seconds")
        logger.info(f"Hexagons processed: {len(gdf):,}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Metadata file: {metadata_file}")
        logger.info(f"Data validation: {'PASSED' if validation_passed else 'WARNING'}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review the processed embeddings")
        logger.info("2. Visualize sample hexagons to verify coverage")
        logger.info("3. Process other stage1_modalities (POI, roads, etc.) if needed")
        logger.info("4. Run fusion pipeline when all stage1_modalities are ready")

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