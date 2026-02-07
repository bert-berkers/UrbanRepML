"""
Generate Roads embeddings for Netherlands at resolution 10.
This script specifically generates the missing roads embeddings.
"""

import logging
from pathlib import Path
import geopandas as gpd
from datetime import datetime

# Import roads processor
from stage1_modalities.roads.processor import RoadsProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Generate roads embeddings for Netherlands."""
    logger.info("Starting Netherlands roads embedding generation...")
    
    # Load Netherlands boundary
    boundary_path = Path("data/study_areas/netherlands/area_gdf/netherlands_boundary.geojson")
    if not boundary_path.exists():
        raise FileNotFoundError(f"Netherlands boundary not found at {boundary_path}")
    
    boundary_gdf = gpd.read_file(boundary_path)
    logger.info(f"Loaded boundary with {len(boundary_gdf)} features")
    logger.info(f"Boundary bounds: {boundary_gdf.total_bounds}")
    
    # Configure Roads processor
    roads_config = {
        'data_source': 'osm_online',
        'output_dir': 'data/study_areas/netherlands/embeddings/roads',
        'embedding_size': 30,  # 30D Highway2Vec embeddings
        'hidden_size': 64,
        'highway2vec_epochs': 25,  # Reduced for faster training
        # Save intermediate embeddings stage1_modalities data
        'save_intermediate': True,
        'intermediate_dir': 'data/study_areas/netherlands/embeddings/intermediate/roads'
    }
    
    logger.info("Configuration:")
    logger.info(f"  - Embedding size: {roads_config['embedding_size']}D")
    logger.info(f"  - Hidden size: {roads_config['hidden_size']}")
    logger.info(f"  - Training epochs: {roads_config['highway2vec_epochs']}")
    logger.info(f"  - Save intermediate embeddings modalities: {roads_config['save_intermediate']}")
    
    # Initialize processor
    processor = RoadsProcessor(roads_config)
    
    # Generate embeddings
    start_time = datetime.now()
    logger.info(f"Starting processing at {start_time}")
    
    try:
        output_path = processor.run_pipeline(
            study_area=boundary_gdf,
            h3_resolution=10,  # Match AlphaEarth and POI resolution
            output_dir=roads_config['output_dir'],
            study_area_name='netherlands'
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Roads embeddings completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info(f"Processing time: {duration}")
        
        # Verify the output
        if Path(output_path).exists():
            import pandas as pd
            df = pd.read_parquet(output_path)
            logger.info(f"Generated embeddings shape: {df.shape}")
            logger.info(f"H3 cells: {df['h3_index'].nunique()}")
            logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error generating roads embeddings: {e}")
        raise


if __name__ == "__main__":
    main()