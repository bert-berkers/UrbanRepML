"""
Train Hex2Vec embeddings from landuse data.
Self-contained script that handles intermediate data generation if needed.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import geopandas as gpd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# SRAI imports
from srai.embedders import Hex2VecEmbedder
from srai.neighbourhoods import H3Neighbourhood

# Import our landuse processor for fallback data generation
from modalities.landuse.processor import LanduseProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_intermediate_data(study_area_name: str = "netherlands", h3_resolution: int = 10):
    """Check if intermediate data exists."""
    intermediate_dir = Path("data/intermediate/landuse")
    base_name = f"{study_area_name}_res{h3_resolution}"
    
    features_path = intermediate_dir / f"{base_name}_features.parquet"
    regions_path = intermediate_dir / f"{base_name}_regions.parquet"
    joint_path = intermediate_dir / f"{base_name}_joint.parquet"
    
    all_exist = all([features_path.exists(), regions_path.exists(), joint_path.exists()])
    
    if all_exist:
        logger.info("✅ Intermediate data found")
        return True, (features_path, regions_path, joint_path)
    else:
        logger.info("❌ Intermediate data missing - will generate")
        return False, None


def generate_intermediate_data(study_area_name: str = "netherlands", h3_resolution: int = 10):
    """Generate intermediate data using existing AlphaEarth H3 regions."""
    logger.info("Generating intermediate data using existing AlphaEarth H3 regions...")
    
    # Load existing AlphaEarth H3 regions (no boundary needed!)
    if study_area_name == "netherlands":
        alpha_path = Path("data/processed/embeddings/alphaearth/netherlands_res10_2022.parquet")
        if not alpha_path.exists():
            raise FileNotFoundError(f"AlphaEarth data not found at: {alpha_path}")
        
        logger.info("Loading existing AlphaEarth H3 regions...")
        alpha_df = pd.read_parquet(alpha_path)
        h3_indices = alpha_df['h3_index'].tolist()
        logger.info(f"Found {len(h3_indices):,} H3 hexagons from AlphaEarth data")
        
        # Create regions_gdf from H3 indices
        import h3
        from shapely.geometry import Polygon
        
        logger.info("Creating regions GeoDataFrame from H3 indices...")
        geometries = []
        for h3_idx in h3_indices:
            # Get H3 hexagon boundary coordinates
            boundary = h3.cell_to_boundary(h3_idx)
            # Convert to Shapely polygon (boundary returns lat,lng tuples)
            coords = [(lng, lat) for lat, lng in boundary]
            geom = Polygon(coords)
            geometries.append(geom)
        
        regions_gdf = gpd.GeoDataFrame(
            {'region_id': h3_indices}, 
            geometry=geometries, 
            crs='EPSG:4326'
        )
        regions_gdf = regions_gdf.set_index('region_id')
        logger.info(f"Created regions_gdf with {len(regions_gdf):,} hexagons")
        
        # Get union boundary for OSM download
        logger.info("Computing union boundary for OSM download...")
        boundary_gdf = gpd.GeoDataFrame([1], geometry=[regions_gdf.unary_union], crs='EPSG:4326')
        
    elif study_area_name == "netherlands_test":
        # Amsterdam test area (fallback)
        from shapely.geometry import box
        test_bounds = box(4.7, 52.3, 5.1, 52.45)
        boundary_gdf = gpd.GeoDataFrame([1], geometry=[test_bounds], crs='EPSG:4326')
        logger.info("Using Amsterdam test area")
        
        # For test, create regions normally
        from srai.regionalizers import H3Regionalizer
        regionalizer = H3Regionalizer(h3_resolution)
        regions_gdf = regionalizer.transform(boundary_gdf)
        
    else:
        raise ValueError(f"Boundary loading not implemented for: {study_area_name}")
    
    # Configure processor for OSM download only
    config = {
        'data_source': 'osm_online',
        'tags': {
            'landuse': ['residential', 'commercial', 'industrial', 'retail', 
                       'farmland', 'forest', 'meadow', 'orchard',
                       'recreation_ground', 'cemetery', 'construction'],
            'natural': ['water', 'wetland', 'wood'],
            'water': ['canal', 'river', 'lake']
        },
        'save_intermediate': True,
        'intermediate_dir': 'data/intermediate/landuse'
    }
    
    # Initialize processor
    processor = LanduseProcessor(config)
    
    # Download OSM data for the boundary
    logger.info("Downloading OSM landuse data...")
    features_gdf = processor.load_data(boundary_gdf)
    
    # Create spatial joins between features and our existing regions
    logger.info("Creating spatial joins...")
    from srai.joiners import IntersectionJoiner
    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(regions_gdf, features_gdf)
    
    # Save intermediate data
    processor._save_intermediate_data(
        features_gdf, regions_gdf, joint_gdf, 
        study_area_name, h3_resolution
    )
    
    logger.info("✅ Intermediate data generated successfully using existing H3 regions")
    return features_gdf, regions_gdf, joint_gdf


def load_intermediate_data(study_area_name: str = "netherlands", h3_resolution: int = 10):
    """Load intermediate data from disk."""
    intermediate_dir = Path("data/intermediate/landuse")
    base_name = f"{study_area_name}_res{h3_resolution}"
    
    features_path = intermediate_dir / f"{base_name}_features.parquet"
    regions_path = intermediate_dir / f"{base_name}_regions.parquet"
    joint_path = intermediate_dir / f"{base_name}_joint.parquet"
    
    logger.info(f"Loading intermediate data...")
    
    # Load with fallback for geometry issues
    try:
        features_gdf = gpd.read_parquet(features_path)
    except:
        features_df = pd.read_parquet(features_path)
        features_gdf = gpd.GeoDataFrame(features_df)
    
    try:
        regions_gdf = gpd.read_parquet(regions_path)
    except:
        regions_df = pd.read_parquet(regions_path)
        regions_gdf = gpd.GeoDataFrame(regions_df)
    
    try:
        joint_gdf = gpd.read_parquet(joint_path)
    except:
        joint_df = pd.read_parquet(joint_path)
        joint_gdf = gpd.GeoDataFrame(joint_df)
    
    logger.info(f"✅ Loaded: {len(features_gdf)} features, {len(regions_gdf)} regions, {len(joint_gdf)} joints")
    return features_gdf, regions_gdf, joint_gdf


def get_landuse_features(features_gdf):
    """Extract landuse feature categories."""
    features = []
    
    # Get landuse categories
    if 'landuse' in features_gdf.columns:
        landuse_vals = features_gdf['landuse'].dropna().unique()
        features.extend([f'landuse_{val}' for val in landuse_vals])
    
    # Get natural categories  
    if 'natural' in features_gdf.columns:
        natural_vals = features_gdf['natural'].dropna().unique()
        features.extend([f'natural_{val}' for val in natural_vals])
    
    # Get water categories
    if 'water' in features_gdf.columns:
        water_vals = features_gdf['water'].dropna().unique()
        features.extend([f'water_{val}' for val in water_vals])
    
    logger.info(f"Found {len(features)} landuse feature categories")
    return sorted(features)


def train_hex2vec_embeddings(features_gdf, regions_gdf, joint_gdf, expected_features, 
                            epochs=10, batch_size=2048):
    """Train Hex2Vec embeddings using SRAI fit_transform."""
    logger.info("="*50)
    logger.info("TRAINING HEX2VEC EMBEDDINGS")
    logger.info("="*50)
    
    # Build neighborhood
    logger.info("Building H3 neighborhood graph...")
    neighbourhood = H3Neighbourhood(regions_gdf)
    logger.info(f"Neighborhood ready with {len(regions_gdf)} regions")
    
    # Initialize Hex2VecEmbedder
    logger.info(f"Initializing Hex2Vec with {len(expected_features)} features, 32D embeddings")
    hex2vec = Hex2VecEmbedder(
        encoder_sizes=[64, 32],  # 2-layer encoder ending in 32D
        expected_output_features=expected_features
    )
    
    # Use fit_transform to train and get embeddings in one call
    logger.info(f"Training Hex2Vec model ({epochs} epochs, batch_size={batch_size})...")
    trainer_kwargs = {
        'accelerator': 'auto',
        'devices': 1,
        'max_epochs': epochs,
        'enable_progress_bar': True,
        'logger': False
    }
    
    # This does both training AND returns embeddings
    embeddings = hex2vec.fit_transform(
        regions_gdf=regions_gdf,
        features_gdf=features_gdf,
        joint_gdf=joint_gdf,
        neighbourhood=neighbourhood,
        batch_size=batch_size,
        trainer_kwargs=trainer_kwargs
    )
    
    logger.info(f"✅ Hex2Vec training complete! Shape: {embeddings.shape}")
    
    return embeddings, hex2vec


def save_hex2vec_outputs(embeddings, hex2vec_model, regions_gdf, study_area_name, h3_resolution):
    """Save hex2vec embeddings and trained model."""
    logger.info("="*50)
    logger.info("SAVING HEX2VEC OUTPUTS")
    logger.info("="*50)
    
    # Prepare embeddings DataFrame
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.columns = [f'hex2vec_{i}' for i in range(embeddings_df.shape[1])]
    
    # Add h3_index (get from regions_gdf index)
    if hasattr(regions_gdf.index, 'name') and regions_gdf.index.name == 'region_id':
        embeddings_df['h3_index'] = regions_gdf.index.astype(str)
    else:
        embeddings_df['h3_index'] = regions_gdf.index.astype(str)
    
    # Reorder columns (h3_index first)
    cols = ['h3_index'] + [col for col in embeddings_df.columns if col != 'h3_index']
    embeddings_df = embeddings_df[cols]
    
    # Save embeddings
    output_dir = Path("data/processed/embeddings/landuse")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_path = output_dir / f'landuse_hex2vec_{study_area_name}_res{h3_resolution}.parquet'
    embeddings_df.to_parquet(embeddings_path, index=False)
    logger.info(f"✅ Saved embeddings: {embeddings_path}")
    
    # Save trained model
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"hex2vec_landuse_{study_area_name}.pkl"
    hex2vec_model.save(model_path)
    logger.info(f"✅ Saved trained model: {model_path}")
    
    return embeddings_path, model_path


def verify_outputs(embeddings_path):
    """Verify output file structure."""
    logger.info("="*50)
    logger.info("VERIFYING OUTPUTS")
    logger.info("="*50)
    
    # Load and check embeddings
    df = pd.read_parquet(embeddings_path)
    logger.info(f"Embeddings shape: {df.shape}")
    logger.info(f"Has h3_index column: {'h3_index' in df.columns}")
    logger.info(f"First 5 columns: {df.columns[:5].tolist()}")
    
    # Verify H3 index
    if 'h3_index' in df.columns and len(df) > 0:
        import h3
        sample_h3 = df['h3_index'].iloc[0]
        try:
            resolution = h3.get_resolution(sample_h3)
            logger.info(f"H3 resolution verified: {resolution}")
        except:
            logger.warning(f"Could not validate H3 index: {sample_h3}")
    
    logger.info("✅ Output verification complete")


def main():
    """Main execution."""
    logger.info("="*60)
    logger.info("HEX2VEC LANDUSE EMBEDDINGS TRAINING")
    logger.info("="*60)
    
    study_area_name = "netherlands"
    h3_resolution = 10
    
    try:
        # Step 1: Check for intermediate data
        has_data, data_paths = check_intermediate_data(study_area_name, h3_resolution)
        
        if not has_data:
            # Generate intermediate data if missing
            features_gdf, regions_gdf, joint_gdf = generate_intermediate_data(
                study_area_name, h3_resolution
            )
        else:
            # Load existing intermediate data
            features_gdf, regions_gdf, joint_gdf = load_intermediate_data(
                study_area_name, h3_resolution
            )
        
        # Step 2: Get feature categories
        expected_features = get_landuse_features(features_gdf)
        
        # Step 3: Train Hex2Vec embeddings
        embeddings, hex2vec_model = train_hex2vec_embeddings(
            features_gdf, regions_gdf, joint_gdf, expected_features,
            epochs=10, batch_size=2048
        )
        
        # Step 4: Save outputs
        embeddings_path, model_path = save_hex2vec_outputs(
            embeddings, hex2vec_model, regions_gdf, study_area_name, h3_resolution
        )
        
        # Step 5: Verify outputs
        verify_outputs(embeddings_path)
        
        logger.info("="*60)
        logger.info("🎉 HEX2VEC TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"📁 Embeddings: {embeddings_path}")
        logger.info(f"🤖 Model: {model_path}")
        logger.info("Ready for multi-modal fusion!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()