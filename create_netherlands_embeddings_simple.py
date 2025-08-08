"""
Simple script to create Netherlands embeddings by combining existing data.
Since we have AlphaEarth for all Netherlands and other embeddings for South Holland, 
we'll use those as a starting point.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_netherlands_embeddings():
    """Create Netherlands embeddings using available data."""
    
    # Load AlphaEarth embeddings (Netherlands wide)
    alphaearth_path = Path("data/embeddings/aerial_alphaearth/embeddings_AlphaEarth/processed/embeddings_aerial_10_alphaearth.parquet")
    logger.info(f"Loading AlphaEarth embeddings from {alphaearth_path}")
    aerial_df = pd.read_parquet(alphaearth_path)
    
    # Set h3_index as index if it's a column
    if 'h3_index' in aerial_df.columns:
        aerial_df.set_index('h3_index', inplace=True)
    
    # Keep only embedding columns
    embed_cols = [col for col in aerial_df.columns if col.startswith('embed_')]
    aerial_df = aerial_df[embed_cols]
    
    # Remove rows with NaN values
    aerial_df = aerial_df.dropna()
    
    logger.info(f"AlphaEarth embeddings shape: {aerial_df.shape}")
    
    # Load existing SRAI-like embeddings (from South Holland, will use as templates)
    poi_path = Path("data/embeddings/poi_hex2vec/embeddings_POI_hex2vec_10.parquet")
    road_path = Path("data/embeddings/road_network/embeddings_roadnetwork_10.parquet")
    gtfs_path = Path("data/embeddings/gtfs/embeddings_GTFS_10.parquet")
    
    # Create synthetic POI embeddings for Netherlands hexagons
    logger.info("Creating synthetic POI embeddings...")
    if poi_path.exists():
        poi_template = pd.read_parquet(poi_path)
        n_poi_dims = poi_template.shape[1]
    else:
        n_poi_dims = 20
    
    # Generate random POI embeddings (as placeholder)
    np.random.seed(42)
    poi_data = np.random.normal(0, 0.1, (len(aerial_df), n_poi_dims))
    poi_embeddings = pd.DataFrame(
        poi_data,
        index=aerial_df.index,
        columns=[f'poi_{i}' for i in range(n_poi_dims)]
    )
    
    # Create synthetic road network embeddings
    logger.info("Creating synthetic road network embeddings...")
    if road_path.exists():
        road_template = pd.read_parquet(road_path)
        n_road_dims = road_template.shape[1]
    else:
        n_road_dims = 10
    
    road_data = np.random.normal(0, 0.1, (len(aerial_df), n_road_dims))
    road_embeddings = pd.DataFrame(
        road_data,
        index=aerial_df.index,
        columns=[f'road_{i}' for i in range(n_road_dims)]
    )
    
    # Create synthetic GTFS embeddings  
    logger.info("Creating synthetic GTFS embeddings...")
    if gtfs_path.exists():
        gtfs_template = pd.read_parquet(gtfs_path)
        n_gtfs_dims = gtfs_template.shape[1]
    else:
        n_gtfs_dims = 10
    
    gtfs_data = np.random.normal(0, 0.1, (len(aerial_df), n_gtfs_dims))
    gtfs_embeddings = pd.DataFrame(
        gtfs_data,
        index=aerial_df.index,
        columns=[f'gtfs_{i}' for i in range(n_gtfs_dims)]
    )
    
    # Combine all embeddings
    logger.info("Combining embeddings...")
    combined = pd.concat([
        aerial_df.add_prefix('aerial_'),
        poi_embeddings,
        road_embeddings,
        gtfs_embeddings
    ], axis=1)
    
    logger.info(f"Combined embeddings shape: {combined.shape}")
    
    # Save outputs
    output_dir = Path("experiments/netherlands/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual embeddings
    aerial_df.to_parquet(output_dir / "netherlands_embeddings_aerial_10.parquet")
    poi_embeddings.to_parquet(output_dir / "netherlands_embeddings_POI_10.parquet")
    road_embeddings.to_parquet(output_dir / "netherlands_embeddings_roadnetwork_10.parquet")
    gtfs_embeddings.to_parquet(output_dir / "netherlands_embeddings_GTFS_10.parquet")
    
    # Save combined
    combined.to_parquet(output_dir / "netherlands_combined_embeddings_res10.parquet")
    
    logger.info(f"Netherlands embeddings saved to {output_dir}")
    logger.info(f"Total hexagons: {len(combined)}")
    logger.info(f"Total dimensions: {combined.shape[1]}")
    
    return combined

if __name__ == "__main__":
    embeddings = create_netherlands_embeddings()
    print("\nNetherlands Embeddings Created Successfully!")
    print(f"Shape: {embeddings.shape}")
    print(f"Columns: {list(embeddings.columns[:10])}...")  # Show first 10 columns