"""
Generate Landuse-based Hex2Vec and GeoVex embeddings for Netherlands at resolution 10.
TEST VERSION: Processes in smaller chunks or with limited landuse categories.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our landuse processor
from modalities.landuse.processor import LanduseProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/landuse_netherlands_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def load_netherlands_boundary(test_mode=True):
    """Load Netherlands boundary for processing embeddings."""
    logger.info("Loading Netherlands boundary...")
    
    boundary_path = Path("data/processed/h3_regions/netherlands/netherlands_boundary.geojson")
    if not boundary_path.exists():
        # Try alternative path
        boundary_path = Path("study_areas/netherlands/cache/netherlands_boundary.parquet")
        if not boundary_path.exists():
            raise FileNotFoundError(f"Netherlands boundary not found. Tried:\n{boundary_path}")
    
    # Load boundary
    if boundary_path.suffix == '.parquet':
        boundary_gdf = gpd.read_parquet(boundary_path)
    else:
        boundary_gdf = gpd.read_file(boundary_path)
    
    if test_mode:
        # For testing, use a smaller area (Amsterdam region)
        logger.info("TEST MODE: Using Amsterdam region only")
        # Amsterdam approximate bounds
        test_bounds = box(4.7, 52.3, 5.1, 52.45)  # Smaller box around Amsterdam
        test_gdf = gpd.GeoDataFrame([1], geometry=[test_bounds], crs='EPSG:4326')
        boundary_gdf = test_gdf
        logger.info(f"Test area bounds: {test_bounds.bounds}")
    
    logger.info(f"Loaded boundary with {len(boundary_gdf)} features")
    logger.info(f"Boundary CRS: {boundary_gdf.crs}")
    logger.info(f"Boundary bounds: {boundary_gdf.total_bounds}")
    
    return boundary_gdf


def verify_outputs(output_paths):
    """Verify that output files have the correct structure."""
    logger.info("\n" + "="*60)
    logger.info("Verifying output files...")
    logger.info("="*60)
    
    for embedding_type, path in output_paths.items():
        if Path(path).exists():
            df = pd.read_parquet(path)
            logger.info(f"\n{embedding_type.upper()}:")
            logger.info(f"  File: {Path(path).name}")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Has h3_index: {'h3_index' in df.columns}")
            logger.info(f"  Index name: {df.index.name}")
            logger.info(f"  First 5 columns: {df.columns[:5].tolist()}")
            
            # Check h3_index format
            if 'h3_index' in df.columns:
                sample_h3 = df['h3_index'].iloc[0] if len(df) > 0 else None
                logger.info(f"  Sample h3_index: {sample_h3}")
                
                # Verify it's a valid H3 index
                if sample_h3:
# MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Note: SRAI provides H3 functionality with additional spatial analysis tools
                    try:
                        resolution = h3.h3_get_resolution(sample_h3)
                        logger.info(f"  H3 resolution: {resolution}")
                    except:
                        logger.warning(f"  Could not validate H3 index: {sample_h3}")
        else:
            logger.warning(f"{embedding_type}: File not found at {path}")
    
    logger.info("\n" + "="*60)


def main():
    """Main execution pipeline."""
    logger.info("="*60)
    logger.info("Starting Netherlands landuse embeddings generation (TEST MODE)")
    logger.info("="*60)
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Step 1: Load boundary (TEST MODE - Amsterdam only)
        logger.info("\nStep 1: Loading study area boundary")
        boundary_gdf = load_netherlands_boundary(test_mode=True)
        
        # Step 2: Configure processor with limited categories for faster processing embeddings
        logger.info("\nStep 2: Configuring landuse processor (limited categories for test)")
        landuse_config = {
            'data_source': 'osm_online',
            'tags': {
                # Focus on main landuse categories only for test
                'landuse': ['residential', 'commercial', 'industrial', 'retail', 
                           'farmland', 'forest', 'meadow', 'orchard',
                           'recreation_ground', 'cemetery', 'construction'],
                'natural': ['water', 'wetland', 'wood'],
                'water': ['canal', 'river', 'lake']  # Key for Netherlands
            },
            'compute_diversity_metrics': True,
            'use_hex2vec': True,
            'use_geovex': True,
            'hex2vec_epochs': 10,  # Reduced for test
            'geovex_epochs': 8,    # Reduced for test
            'batch_size': 256,     # Smaller batch for test
            'save_intermediate': True,
            'intermediate_dir': 'data/intermediate/landuse',
            'output_dir': 'data/processed/embeddings/landuse'
        }
        
        # Initialize processor
        processor = LanduseProcessor(landuse_config)
        
        # Step 3: Run pipeline
        logger.info("\nStep 3: Running landuse processing embeddings pipeline (TEST MODE)")
        logger.info("Processing Amsterdam region as test...")
        
        output_paths = processor.run_pipeline(
            study_area=boundary_gdf,
            h3_resolution=10,  # Match AlphaEarth resolution
            output_dir=landuse_config['output_dir'],
            study_area_name='netherlands_test'  # Add _test suffix
        )
        
        # Step 4: Verify outputs
        logger.info("\nStep 4: Verifying outputs")
        verify_outputs(output_paths)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST LANDUSE EMBEDDINGS GENERATION COMPLETED!")
        logger.info("="*60)
        logger.info("\nOutput files saved to:")
        for embedding_type, path in output_paths.items():
            logger.info(f"  {embedding_type}: {path}")
        
        logger.info("\nIntermediate files saved to: data/intermediate/landuse/")
        
        logger.info("\n" + "="*60)
        logger.info("TEST SUCCESSFUL!")
        logger.info("To process full Netherlands, run the full script with test_mode=False")
        logger.info("Or process in regional chunks to avoid timeout")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error during landuse embeddings generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()