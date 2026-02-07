"""
Process raw AlphaEarth 2022 TIFFs for Netherlands to H3 resolution 10
with comprehensive logging and efficient storage
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from stage1_modalities.alphaearth.processor import AlphaEarthProcessor

# Set up comprehensive logging
def setup_logging():
    """Setup detailed logging for resolution 10 processing_modalities."""
    
    # Create logs directory
    log_dir = Path('logs/alphaearth')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'netherlands_res10_2022_{timestamp}.log'
    
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
    
    import psutil
    
    logger.info("SYSTEM INFORMATION")
    logger.info(f"  CPU cores: {os.cpu_count()}")
    logger.info(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    logger.info(f"  Total memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  Working directory: {Path.cwd()}")


def process_alphaearth_res10_2022(logger):
    """Process AlphaEarth 2022 TIFFs to H3 resolution 10."""
    
    logger.info("="*60)
    logger.info("STARTING ALPHAEARTH RESOLUTION 10 PROCESSING")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Configuration for resolution 10
    config = {
        'source_dir': os.getenv('ALPHAEARTH_NETHERLANDS_PATH', 'G:/My Drive/AlphaEarth_Netherlands/'),  # Use env var
        'subtile_size': 256,  # Smaller for finer resolution
        'min_pixels_per_hex': 3,  # Lower threshold for res 10
        'max_workers': 10,
        'intermediate_dir': 'intermediate_res10'  # Separate directory for res 10
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
    
    # Count 2022 TIFF files
    tiff_pattern = "*2022*.tif"
    tiff_files = list(source_path.glob(tiff_pattern))
    logger.info(f"Found {len(tiff_files)} TIFF files for 2022")
    
    if not tiff_files:
        logger.error("No 2022 TIFF files found")
        raise ValueError("No 2022 TIFF files found")
    
    # Log file size statistics
    total_size = sum(f.stat().st_size for f in tiff_files)
    avg_size = total_size / len(tiff_files)
    logger.info(f"Total input data size: {total_size / (1024**3):.1f} GB")
    logger.info(f"Average file size: {avg_size / (1024**2):.1f} MB")
    
    # Initialize processor
    logger.info("Initializing AlphaEarthProcessor...")
    processor = AlphaEarthProcessor(config)
    
    # Process to resolution 10
    logger.info("STARTING TILE PROCESSING")
    logger.info(f"  Target resolution: 10")
    logger.info(f"  Expected hexagon area: ~0.013 km²")
    logger.info(f"  Processing {len(tiff_files)} tiles...")
    
    processing_start = time.time()
    
    gdf = processor.process(
        raw_data_path=source_path,
        year_filter='2022',
        h3_resolution=10,
        intermediate_dir='intermediate_res10'
    )
    
    processing_end = time.time()
    processing_duration = processing_end - processing_start
    
    logger.info("PROCESSING COMPLETED")
    logger.info(f"  Duration: {processing_duration:.1f} seconds ({processing_duration/60:.1f} minutes)")
    logger.info(f"  Hexagons generated: {len(gdf):,}")
    logger.info(f"  Processing rate: {len(gdf) / processing_duration:.0f} hexagons/second")
    logger.info(f"  Processing rate: {len(gdf) / (processing_duration/60):.0f} hexagons/minute")
    
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
    """Save processed data with metadata."""
    
    logger.info("SAVING PROCESSED DATA")
    
    # Create output directory
    output_dir = Path('data/study_areas/netherlands/embeddings/alphaearth')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main embeddings file
    output_file = output_dir / 'netherlands_res10_2022.parquet'
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
        'hexagon_count': len(gdf),
        'h3_resolution': 10,
        'year': 2022,
        'region': 'netherlands',
        'coverage_area_km2': len(gdf) * 0.013,  # Approximate area
        'embedding_dimensions': len([col for col in gdf.columns if col.startswith('A')]),
        'file_size_mb': file_size / (1024**2),
        'source_tiles': 99,
        'coordinate_system': 'EPSG:4326',
        'processor_version': 'AlphaEarthProcessor v1.0',
        'configuration': {
            'subtile_size': 256,
            'min_pixels_per_hex': 3,
            'max_workers': 10
        }
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
    metadata_file = output_dir / 'netherlands_res10_2022_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata saved to: {metadata_file}")
    
    return output_file, metadata_file


def validate_output(gdf, logger):
    """Validate the processed data quality."""
    
    logger.info("VALIDATING OUTPUT DATA")
    
    # Check H3 index validity
    # MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)
    from srai.regionalizers import H3Regionalizer
    from srai.neighbourhoods import H3Neighbourhood
    # Note: SRAI provides H3 functionality with additional spatial analysis tools
    h3_indices = gdf['h3_index'] if 'h3_index' in gdf.columns else gdf.index

    if hasattr(h3_indices, 'tolist'):
        indices_to_check = h3_indices.tolist()[:1000]  # Sample for speed
    else:
        indices_to_check = list(h3_indices)[:1000]

    valid_count = sum(1 for idx in indices_to_check if h3.is_valid_cell(str(idx)))
    validity_rate = valid_count / len(indices_to_check)
    logger.info(f"  H3 index validity: {validity_rate:.1%} ({valid_count}/{len(indices_to_check)} sampled)")

    # Check embedding completeness
    embedding_cols = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
    if embedding_cols:
        nan_count = gdf[embedding_cols].isna().sum().sum()
        total_values = len(gdf) * len(embedding_cols)
        completeness = (total_values - nan_count) / total_values
        logger.info(f"  Embedding completeness: {completeness:.1%} ({total_values - nan_count}/{total_values} values)")

    # Check geometry validity
    if 'geometry' in gdf.columns:
        valid_geom = gdf.geometry.is_valid.sum()
        geom_validity = valid_geom / len(gdf)
        logger.info(f"  Geometry validity: {geom_validity:.1%} ({valid_geom}/{len(gdf)} geometries)")

    # Check coordinate range
    if 'geometry' in gdf.columns:
        bounds = gdf.total_bounds
        logger.info(f"  Geographic bounds:")
        logger.info(f"    West: {bounds[0]:.3f}°")
        logger.info(f"    East: {bounds[2]:.3f}°")
        logger.info(f"    South: {bounds[1]:.3f}°")
        logger.info(f"    North: {bounds[3]:.3f}°")

    # Overall validation
    overall_valid = (validity_rate > 0.95 and
                    (not embedding_cols or completeness > 0.95) and
                    ('geometry' not in gdf.columns or geom_validity > 0.95))

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
        logger.info("NETHERLANDS ALPHAEARTH RESOLUTION 10 PROCESSING SESSION")
        logger.info(f"Session start: {datetime.now()}")
        log_system_info(logger)
        
        # Process AlphaEarth data
        gdf, processing_duration = process_alphaearth_res10_2022(logger)
        
        # Validate output
        validation_passed = validate_output(gdf, logger)
        
        # Save data with metadata
        output_file, metadata_file = save_processed_data(gdf, processing_duration, logger)
        
        # Final summary
        total_duration = time.time() - processing_duration
        logger.info("="*60)
        logger.info("PROCESSING SESSION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total session duration: {total_duration:.1f} seconds")
        logger.info(f"Hexagons processed: {len(gdf):,}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Metadata file: {metadata_file}")
        logger.info(f"Data validation: {'PASSED' if validation_passed else 'WARNING'}")
        
        # Update processing_modalities log
        update_processing_log(len(gdf), processing_duration, validation_passed, logger)
        
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


def update_processing_log(hexagon_count, duration, validation_passed, logger):
    """Update the processing_modalities log with this session's results."""
    
    try:
        # Read current processing_modalities log
        log_file = Path('docs/PROCESSING_LOG.md')
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
        else:
            content = "# UrbanRepML Processing Log\n\n"
        
        # Create new entry
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        new_entry = f"""
### Operation: AlphaEarth Processing (Resolution 10)
**Date**: {timestamp}
**Script**: `scripts/process_alphaearth_netherlands_res10_2022.py`
**Duration**: {duration:.1f} seconds ({duration/60:.1f} minutes)

**Configuration**:
```python
{{
    'source_dir': '[From ALPHAEARTH_NETHERLANDS_PATH env var]',
    'subtile_size': 256,
    'min_pixels_per_hex': 3,
    'max_workers': 10,
    'year_filter': '2022',
    'h3_resolution': 10
}}
```

**Results**:
- **Input tiles**: 99 GeoTIFF files (2022)
- **Output hexagons**: {hexagon_count:,}
- **Processing rate**: {hexagon_count / duration:.0f} hexagons/second
- **Processing rate**: {hexagon_count / (duration/60):.0f} hexagons/minute
- **Validation**: {'PASSED' if validation_passed else 'FAILED'}
- **Output file**: `data/study_areas/netherlands/embeddings/alphaearth/netherlands_res10_2022.parquet`

"""
        
        # Find insertion point (after the existing Netherlands section)
        if "## 2025-08-30: Netherlands AlphaEarth Processing" in content:
            # Insert after the last operation in today's section
            insert_point = content.find("### Quality Validation Results")
            if insert_point > 0:
                content = content[:insert_point] + new_entry + "\n" + content[insert_point:]
        else:
            # Append to end
            content += new_entry
        
        # Write back
        with open(log_file, 'w') as f:
            f.write(content)
        
        logger.info(f"Updated processing_modalities log: {log_file}")
        
    except Exception as e:
        logger.warning(f"Failed to update processing_modalities log: {e}")


if __name__ == "__main__":
    exit(main())