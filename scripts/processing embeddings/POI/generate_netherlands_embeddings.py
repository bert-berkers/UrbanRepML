"""
Generate POI and Roads embeddings for Netherlands at resolution 10
Compatible with AlphaEarth Netherlands data for multi-modal fusion.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
from datetime import datetime

# Import our processors
from modalities.poi.processor import POIProcessor  
from modalities.roads.processor import RoadsProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_netherlands_boundary():
    """Load Netherlands boundary and check compatibility with AlphaEarth data."""
    logger.info("Loading Netherlands boundary...")
    
    boundary_path = Path("data/processed/h3_regions/netherlands/netherlands_boundary.geojson")
    if not boundary_path.exists():
        raise FileNotFoundError(f"Netherlands boundary not found at {boundary_path}")
    
    # Load boundary
    boundary_gdf = gpd.read_file(boundary_path)
    logger.info(f"Loaded boundary with {len(boundary_gdf)} features")
    logger.info(f"Boundary CRS: {boundary_gdf.crs}")
    logger.info(f"Boundary bounds: {boundary_gdf.total_bounds}")
    
    return boundary_gdf

def check_alphaearth_compatibility():
    """Check existing AlphaEarth data for compatibility."""
    logger.info("Checking AlphaEarth data compatibility...")
    
    alphaearth_path = Path("data/processed/embeddings/alphaearth/netherlands_res10_2022.parquet")
    if not alphaearth_path.exists():
        logger.warning(f"AlphaEarth data not found at {alphaearth_path}")
        return None
    
    # Load sample to check structure
    alphaearth_df = pd.read_parquet(alphaearth_path)
    logger.info(f"AlphaEarth data shape: {alphaearth_df.shape}")
    logger.info(f"AlphaEarth columns: {alphaearth_df.columns.tolist()[:10]}...")
    logger.info(f"AlphaEarth H3 cells: {alphaearth_df['h3_index'].nunique()}")
    
    # Check H3 resolution
# MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Note: SRAI provides H3 functionality with additional spatial analysis tools
    sample_indices = alphaearth_df['h3_index'].head(3).tolist()
    resolutions = [h3.get_resolution(idx) for idx in sample_indices]
    logger.info(f"AlphaEarth H3 resolutions: {resolutions}")
    
    return alphaearth_df

def generate_poi_embeddings(boundary_gdf):
    """Generate POI embeddings focused on GeoVex and Hex2Vec for ML."""
    logger.info("Generating POI embeddings with GeoVex and Hex2Vec...")
    
    # Configure POI processor for GPU-optimized processing embeddings
    poi_config = {
        'data_source': 'osm_online',
        'output_dir': 'data/processed/embeddings/poi',
        # No poi_categories specified = get all POI types automatically
        'compute_diversity_metrics': True,  # For human inspection
        'use_hex2vec': True,   # PRIMARY: Skip-gram embeddings for ML
        'use_geovex': True,    # PRIMARY: Hexagonal convolutional embeddings for ML
        # GPU optimization settings
        'hex2vec_epochs': 10,  # Reduced from default
        'geovex_epochs': 8,    # Reduced for faster training
        'batch_size': 256,     # Larger batch size for better GPU utilization
        # Save intermediate data for debugging
        'save_intermediate': True,
        'intermediate_dir': 'data/processed/intermediate/poi'
    }
    
    # Initialize processor
    processor = POIProcessor(poi_config)
    
    # Generate embeddings
    output_path = processor.run_pipeline(
        study_area=boundary_gdf,
        h3_resolution=10,  # Match AlphaEarth resolution
        output_dir=poi_config['output_dir'],
        study_area_name='netherlands'
    )
    
    logger.info(f"POI embeddings completed: {output_path}")
    return output_path

def generate_roads_embeddings(boundary_gdf):
    """Generate Roads embeddings with Highway2Vec for ML."""
    logger.info("Generating Roads embeddings with Highway2Vec...")
    
    # Configure Roads processor for GPU-optimized training
    roads_config = {
        'data_source': 'osm_online',
        'output_dir': 'data/processed/embeddings/roads',
        'embedding_size': 30,  # PRIMARY: 30D learned features for ML
        'hidden_size': 64,
        'highway2vec_epochs': 25,  # Reduced for faster GPU training
        # Save intermediate data for debugging
        'save_intermediate': True,
        'intermediate_dir': 'data/processed/intermediate/roads'
    }
    
    # Initialize processor
    processor = RoadsProcessor(roads_config)
    
    # Generate embeddings
    output_path = processor.run_pipeline(
        study_area=boundary_gdf,
        h3_resolution=10,  # Match AlphaEarth resolution
        output_dir=roads_config['output_dir'],
        study_area_name='netherlands'
    )
    
    logger.info(f"Roads embeddings completed: {output_path}")
    return output_path

def verify_output_compatibility(poi_path, roads_path, alphaearth_df):
    """Verify that outputs are compatible with AlphaEarth data."""
    logger.info("Verifying output compatibility with AlphaEarth data...")
    
    # Load generated embeddings
    if poi_path:
        poi_df = pd.read_parquet(poi_path)
        logger.info(f"POI embeddings shape: {poi_df.shape}")
        logger.info(f"POI columns: {poi_df.columns.tolist()[:10]}...")
        
        # Check for ML-focused columns
        geovex_cols = [c for c in poi_df.columns if c.startswith('geovex_')]
        hex2vec_cols = [c for c in poi_df.columns if c.startswith('hex2vec_')]
        logger.info(f"GeoVex dimensions: {len(geovex_cols)}")
        logger.info(f"Hex2Vec dimensions: {len(hex2vec_cols)}")
    
    if roads_path:
        roads_df = pd.read_parquet(roads_path)
        logger.info(f"Roads embeddings shape: {roads_df.shape}")
        logger.info(f"Roads embedding dimensions: {roads_df.shape[1] - 2}")  # Minus h3_index and resolution
    
    # Check H3 compatibility if AlphaEarth data available
    if alphaearth_df is not None and poi_path:
        poi_h3_set = set(poi_df['h3_index'])
        alpha_h3_set = set(alphaearth_df['h3_index'])
        
        overlap = len(poi_h3_set.intersection(alpha_h3_set))
        poi_only = len(poi_h3_set - alpha_h3_set)
        alpha_only = len(alpha_h3_set - poi_h3_set)
        
        logger.info(f"H3 overlap: {overlap} cells")
        logger.info(f"POI-only cells: {poi_only}")
        logger.info(f"AlphaEarth-only cells: {alpha_only}")
        logger.info(f"Overlap percentage: {overlap/len(alpha_h3_set)*100:.1f}%")

def main():
    """Main execution pipeline."""
    logger.info("Starting Netherlands embeddings generation...")
    
    try:
        # Step 1: Load boundary and check compatibility
        boundary_gdf = load_netherlands_boundary()
        alphaearth_df = check_alphaearth_compatibility()
        
        # Step 2: Generate POI embeddings (focus on GeoVex + Hex2Vec)
        poi_path = generate_poi_embeddings(boundary_gdf)
        
        # Step 3: Generate Roads embeddings (Highway2Vec)
        roads_path = generate_roads_embeddings(boundary_gdf)
        
        # Step 4: Verify compatibility
        verify_output_compatibility(poi_path, roads_path, alphaearth_df)
        
        logger.info("Netherlands embeddings generation completed successfully!")
        logger.info(f"POI embeddings: {poi_path}")
        logger.info(f"Roads embeddings: {roads_path}")
        
    except Exception as e:
        logger.error(f"Error during embeddings generation: {e}")
        raise

if __name__ == "__main__":
    main()