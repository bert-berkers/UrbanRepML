"""
Generate Landuse-based Hex2Vec and GeoVex embeddings for Netherlands at resolution 10.
Saves separate files for counts, diversity metrics, hex2vec, and geovex embeddings.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import geopandas as gpd

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
        logging.FileHandler(f'logs/landuse_netherlands_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def load_netherlands_boundary():
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
                    import h3
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
    logger.info("Starting Netherlands landuse embeddings generation")
    logger.info("="*60)
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # Step 1: Load boundary
        logger.info("\nStep 1: Loading study area boundary")
        boundary_gdf = load_netherlands_boundary()
        
        # Step 2: Configure processor
        logger.info("\nStep 2: Configuring landuse processor")
        landuse_config = {
            'data_source': 'osm_online',
            'tags': {
                'landuse': True,  # Get ALL landuse values
                'natural': ['water', 'wetland', 'beach', 'sand', 'wood', 'grassland', 'heath', 'scrub'],
                'water': True,  # All water features
                'wetland': True  # Netherlands has many wetlands
            },
            'compute_diversity_metrics': True,
            'use_hex2vec': True,
            'use_geovex': True,
            'hex2vec_epochs': 25,  # Good for landuse patterns
            'geovex_epochs': 20,   # Capture spatial transitions
            'batch_size': 512,     # GPU optimized
            'save_intermediate': True,
            'intermediate_dir': 'data/intermediate/landuse',
            'output_dir': 'data/processed/embeddings/landuse'
        }
        
        # Initialize processor
        processor = LanduseProcessor(landuse_config)
        
        # Step 3: Run pipeline
        logger.info("\nStep 3: Running landuse processing embeddings pipeline")
        logger.info("This will:")
        logger.info("  1. Download landuse polygons from OSM")
        logger.info("  2. Process to H3 resolution 10")
        logger.info("  3. Calculate area coverage per hexagon")
        logger.info("  4. Generate diversity metrics")
        logger.info("  5. Train Hex2Vec embeddings (32D)")
        logger.info("  6. Train GeoVex embeddings (32D)")
        logger.info("  7. Save 4 separate output files")
        
        output_paths = processor.run_pipeline(
            study_area=boundary_gdf,
            h3_resolution=10,  # Match AlphaEarth resolution
            output_dir=landuse_config['output_dir'],
            study_area_name='netherlands'
        )
        
        # Step 4: Verify outputs
        logger.info("\nStep 4: Verifying outputs")
        verify_outputs(output_paths)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("LANDUSE EMBEDDINGS GENERATION COMPLETED!")
        logger.info("="*60)
        logger.info("\nOutput files saved to:")
        for embedding_type, path in output_paths.items():
            logger.info(f"  {embedding_type}: {path}")
        
        logger.info("\nIntermediate files saved to: data/intermediate/landuse/")
        
        # Check compatibility with AlphaEarth
        logger.info("\nChecking compatibility with AlphaEarth data...")
        alphaearth_path = Path("data/processed/embeddings/alphaearth/netherlands_res10_2022.parquet")
        if alphaearth_path.exists():
            alpha_df = pd.read_parquet(alphaearth_path)
            
            if 'counts' in output_paths and Path(output_paths['counts']).exists():
                landuse_df = pd.read_parquet(output_paths['counts'])
                
                alpha_h3_set = set(alpha_df['h3_index'])
                landuse_h3_set = set(landuse_df['h3_index'])
                
                overlap = len(alpha_h3_set.intersection(landuse_h3_set))
                logger.info(f"  H3 overlap with AlphaEarth: {overlap:,} cells")
                logger.info(f"  Overlap percentage: {overlap/len(alpha_h3_set)*100:.1f}%")
        else:
            logger.info("  AlphaEarth data not found for comparison")
        
        logger.info("\nProcessing complete! All embeddings ready for multi-modal fusion.")
        
    except Exception as e:
        logger.error(f"Error during landuse embeddings generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()