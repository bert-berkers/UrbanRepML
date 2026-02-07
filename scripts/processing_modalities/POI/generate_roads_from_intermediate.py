"""
Generate Roads embeddings from existing intermediate embeddings stage1_modalities data.
This script uses the already processed intermediate embeddings stage1_modalities roads data to generate embeddings.
Optimized for RTX 3090 24GB VRAM.
"""

import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd
import yaml
from datetime import datetime

# Import roads processor
from stage1_modalities.roads.processor import RoadsProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Generate roads embeddings from intermediate embeddings stage1_modalities data."""
    logger.info("Starting roads embedding generation from intermediate embeddings stage1_modalities data...")
    
    # Check if intermediate embeddings stage1_modalities data exists
    intermediate_base = Path("data/study_areas/netherlands/embeddings/intermediate/roads")
    features_path = intermediate_base / "features_gdf" / "netherlands_res10_features.parquet"
    regions_path = intermediate_base / "regions_gdf" / "netherlands_res10_regions.parquet"
    joint_path = intermediate_base / "joint_gdf" / "netherlands_res10_joint.parquet"
    
    if not all(p.exists() for p in [features_path, regions_path, joint_path]):
        raise FileNotFoundError("Intermediate data not found. Please run the full pipeline first.")
    
    logger.info("Loading intermediate embeddings stage1_modalities data...")
    logger.info(f"Features: {features_path}")
    logger.info(f"Regions: {regions_path}")
    logger.info(f"Joint: {joint_path}")
    
    # Load intermediate embeddings stage1_modalities data
    roads_gdf = gpd.read_parquet(features_path)
    regions_gdf = gpd.read_parquet(regions_path)
    # Load joint_gdf as regular pandas DataFrame since it doesn't need geometry
    joint_gdf = pd.read_parquet(joint_path)
    
    logger.info(f"Loaded intermediate embeddings stage1_modalities data:")
    logger.info(f"  Roads: {roads_gdf.shape}")
    logger.info(f"  Regions: {regions_gdf.shape}")
    logger.info(f"  Joints: {joint_gdf.shape}")
    
    # Load optimized configuration from YAML
    logger.info("Loading Highway2Vec configuration optimized for RTX 3090...")
    config_path = Path("stage1_modalities/roads/config.yaml")
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Build configuration using highway2vec section for RTX 3090 optimization
    highway2vec_config = yaml_config.get('highway2vec', {})
    roads_config = {
        'data_source': 'osm_online',
        'output_dir': 'data/study_areas/netherlands/embeddings/roads',
        'highway2vec': highway2vec_config,  # Use the full highway2vec section
        'save_intermediate': False  # Don't save again
    }
    
    logger.info("RTX 3090 optimized configuration:")
    logger.info(f"  - Embedding size: {highway2vec_config.get('embedding_size', 'default')}D")
    logger.info(f"  - Hidden size: {highway2vec_config.get('hidden_size', 'default')}")  
    logger.info(f"  - Training epochs: {highway2vec_config.get('epochs', 'default')}")
    logger.info(f"  - Batch size: {highway2vec_config.get('batch_size', 'default')} (optimized for 24GB VRAM)")
    
    # Initialize processor
    processor = RoadsProcessor(roads_config)
    
    # Call highway2vec training method directly with intermediate embeddings stage1_modalities data
    start_time = datetime.now()
    logger.info(f"Starting Highway2Vec training at {start_time}")
    
    try:
        # Call the private method directly since we have all the data
        embeddings_df = processor._train_highway2vec_with_data(roads_gdf, regions_gdf, joint_gdf, 10, "netherlands")
        
        # Save embeddings manually since we're calling highway2vec directly
        output_dir = Path(roads_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = "roads_embeddings_res10.parquet"
        output_path = processor.save_embeddings(embeddings_df, str(output_dir), output_filename)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"Roads embeddings completed successfully!")
        logger.info(f"Output: {output_path}")
        logger.info(f"Processing time: {duration}")
        
        # Verify the output
        if Path(output_path).exists():
            df = pd.read_parquet(output_path)
            logger.info(f"Generated embeddings shape: {df.shape}")
            logger.info(f"H3 cells: {df['h3_index'].nunique()}")
            logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
    except Exception as e:
        logger.error(f"Error generating roads embeddings: {e}")
        raise


if __name__ == "__main__":
    main()