"""
Roads Network Modality Processor

Processes OpenStreetMap road network data into H3 hexagon embeddings using SRAI's Highway2VecEmbedder.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union
import warnings

import pandas as pd
import geopandas as gpd

# SRAI imports
from srai.loaders import OSMPbfLoader, OSMOnlineLoader
from srai.regionalizers import H3Regionalizer
from srai.joiners import IntersectionJoiner

# Import Highway2VecEmbedder
try:
    from srai.embedders import Highway2VecEmbedder
    HIGHWAY2VEC_AVAILABLE = True
except ImportError:
    HIGHWAY2VEC_AVAILABLE = False
    logging.warning("Highway2VecEmbedder not available. Install with: pip install srai[torch]")

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


from stage1_modalities.base import ModalityProcessor
from utils import StudyAreaPaths


# Default road types to consider
DEFAULT_ROAD_TYPES = [
    'motorway', 'motorway_link', 'trunk', 'trunk_link', 
    'primary', 'primary_link', 'secondary', 'secondary_link',
    'tertiary', 'tertiary_link', 'unclassified', 'residential',
    'living_street', 'service', 'road'
]


class RoadsProcessor(ModalityProcessor):
    """Process road network data into H3 hexagon embeddings using Highway2Vec."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize roads processor with configuration."""
        super().__init__(config)
        
        if not HIGHWAY2VEC_AVAILABLE:
            raise RuntimeError("RoadsProcessor requires Highway2VecEmbedder. Install with: pip install srai[torch]")
        
        self.validate_config()
        
        # Data configuration
        self.data_source = config.get('data_source', 'osm_online')
        self.pbf_path = config.get('pbf_path', None)
        self.road_types = config.get('road_types', DEFAULT_ROAD_TYPES)
        
        # Highway2Vec parameters from config section
        highway2vec_config = config.get('highway2vec', {})
        self.embedding_size = highway2vec_config.get('embedding_size', config.get('embedding_size', 30))
        self.hidden_size = highway2vec_config.get('hidden_size', config.get('hidden_size', 64))
        self.highway2vec_epochs = highway2vec_config.get('epochs', config.get('highway2vec_epochs', 25))
        self.highway2vec_batch_size = highway2vec_config.get('batch_size', config.get('highway2vec_batch_size', 128))
        
        # Intermediate data saving
        self.save_intermediate = config.get('save_intermediate', False)
        _roads_paths = StudyAreaPaths(config.get('study_area', 'default'))
        self.intermediate_dir = Path(config.get('intermediate_dir', str(_roads_paths.intermediate("roads"))))
        
        logger.info(f"Initialized RoadsProcessor with Highway2Vec (embedding_size={self.embedding_size})")
        logger.info(f"Highway2Vec epochs: {self.highway2vec_epochs}, batch_size: {self.highway2vec_batch_size}")
        if self.save_intermediate:
            logger.info(f"Intermediate data will be saved to: {self.intermediate_dir}")

    def validate_config(self):
        """Validate configuration parameters."""
        if self.config.get('data_source') == 'pbf' and not self.config.get('pbf_path'):
            raise ValueError("PBF data source requires 'pbf_path' in config")

    def load_data(self, area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Load road network data for the study area."""
        logger.info(f"Loading road network data using {self.data_source}")
        
        # Ensure WGS84
        if area_gdf.crs != 'EPSG:4326':
            area_gdf = area_gdf.to_crs('EPSG:4326')
        
        # Define tags for OSM loading
        tags = {'highway': self.road_types}
        
        # Load roads
        if self.data_source == 'pbf':
            loader = OSMPbfLoader()
            roads_gdf = loader.load(self.pbf_path, tags=tags, area=area_gdf)
        else:
            loader = OSMOnlineLoader()
            roads_gdf = loader.load(area_gdf, tags=tags)
        
        # Filter to line geometries only
        if not roads_gdf.empty:
            initial_count = len(roads_gdf)
            roads_gdf = roads_gdf[roads_gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
            if len(roads_gdf) < initial_count:
                logger.info(f"Filtered {initial_count - len(roads_gdf)} non-linear geometries")
        
        logger.info(f"Loaded {len(roads_gdf)} road segments")
        return roads_gdf

    def highway2vec(self, roads_gdf: gpd.GeoDataFrame, area_gdf_or_regions: gpd.GeoDataFrame, 
                   h3_resolution: int, study_area_name: str = "unnamed") -> pd.DataFrame:
        """Train Highway2Vec model and generate road network embeddings using GPU."""
        logger.info(f"Starting Highway2Vec training for H3 resolution {h3_resolution}")
        logger.info(f"Road segments: {len(roads_gdf):,}")
        
        # Check if we're passed regions directly or need to create them
        if 'geometry' in area_gdf_or_regions.columns and len(area_gdf_or_regions) > 1000:
            # This looks like pre-computed regions
            logger.info("Using pre-computed H3 regions...")
            regions_gdf = area_gdf_or_regions
            logger.info(f"Using {len(regions_gdf):,} pre-computed H3 regions")
            
            # For intermediate embeddings stage1_modalities data, we need to create a joint_gdf
            # This is a simplified version - in real use, joint_gdf should be loaded too
            logger.info("Creating spatial join for Highway2Vec (this may take time)...")
            from srai.joiners import IntersectionJoiner
            joiner = IntersectionJoiner()
            joint_gdf = joiner.transform(regions=regions_gdf, features=roads_gdf)
            logger.info(f"Created {len(joint_gdf):,} road-region pairs")
            
        else:
            # Standard workflow - create regions from boundary
            area_gdf = area_gdf_or_regions
            
            # 1. H3 Regionalization
            logger.info("Step 1: Creating H3 hexagonal regions...")
            regionalizer = H3Regionalizer(resolution=h3_resolution)
            regions_gdf = regionalizer.transform(area_gdf)
            logger.info(f"Created {len(regions_gdf):,} H3 regions at resolution {h3_resolution}")
            
            # 2. Spatial Joining
            logger.info("Step 2: Spatially joining roads to H3 regions...")
            joiner = IntersectionJoiner()
            joint_gdf = joiner.transform(regions=regions_gdf, features=roads_gdf)
            road_region_matches = len(joint_gdf)
            logger.info(f"Joined {road_region_matches:,} road-region pairs")
            
            # Save intermediate embeddings stage1_modalities data if requested
            if self.save_intermediate:
                self._save_intermediate_data(roads_gdf, regions_gdf, joint_gdf, h3_resolution, study_area_name)
        
        # 3. Prepare features for Highway2Vec
        logger.info("Step 3: Preparing features for Highway2Vec training...")
        
        # Highway2Vec works with the spatial relationships, not individual road features
        # We need to create a feature matrix from the roads data
        features_for_training = self._prepare_highway2vec_features(roads_gdf, regions_gdf, joint_gdf)
        logger.info(f"Prepared features shape: {features_for_training.shape}")
        
        # 4. Highway2Vec Training
        logger.info("Step 4: Training Highway2Vec autoencoder...")
        logger.info(f"Model architecture: {self.hidden_size} -> {self.embedding_size} dimensions")
        logger.info("Initializing Highway2Vec (using RTX 3090 if available)...")
        
        # Initialize Highway2Vec embedder
        embedder = Highway2VecEmbedder(
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size
        )
        
        # Train and generate embeddings with GPU acceleration
        try:
            logger.info("Training autoencoder on road network patterns...")
            embeddings_gdf = embedder.fit_transform(
                regions_gdf=regions_gdf,
                features_gdf=features_for_training,
                joint_gdf=joint_gdf,
                trainer_kwargs={
                    'accelerator': 'auto',  # Use GPU if available
                    'devices': 1,
                    'max_epochs': self.highway2vec_epochs,  # GPU-optimized epochs
                    'enable_progress_bar': True
                },
                dataloader_kwargs={
                    'batch_size': self.highway2vec_batch_size
                }
            )
            
            logger.info("Highway2Vec training completed successfully!")
            logger.info(f"Generated embeddings for {len(embeddings_gdf):,} hexagons")
            
            # Convert to DataFrame format
            embeddings_df = pd.DataFrame(embeddings_gdf.drop(columns='geometry', errors='ignore'))
            
            # Format output for consistency
            if embeddings_df.index.name == 'region_id':
                embeddings_df.index.name = 'h3_index'
            elif embeddings_df.index.name is None:
                embeddings_df.index.name = 'h3_index'
                
            embeddings_df['h3_resolution'] = h3_resolution
            embeddings_df.index = embeddings_df.index.astype(str)
            
            # Log embedding statistics
            embedding_cols = [col for col in embeddings_df.columns 
                            if col not in ['h3_resolution', 'h3_index']]
            if embedding_cols:
                sample_values = embeddings_df[embedding_cols].iloc[0].values
                logger.info(f"Embedding dimensions: {len(embedding_cols)}")
                logger.info(f"Sample embedding values: {sample_values[:5]}...") 
            
            return embeddings_df.reset_index()
            
        except Exception as e:
            logger.error(f"Highway2Vec training failed: {e}")
            if "CUDA" in str(e) or "GPU" in str(e) or "torch" in str(e).lower():
                logger.info("GPU training failed. Suggestions:")
                logger.info("  - Check CUDA installation: nvidia-smi")
                logger.info("  - Check PyTorch GPU: python -c 'import torch; print(torch.cuda.is_available())'")
                logger.info("  - Fallback to CPU: set CUDA_VISIBLE_DEVICES=''")
            raise
    
    def _prepare_highway2vec_features(self, roads_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame, 
                                     joint_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Prepare numerical features for Highway2Vec training."""
        logger.info("Preparing numerical features from roads data...")
        
        # Extract highway types and create numerical encodings
        highway_types = roads_gdf['highway'].value_counts()
        logger.info(f"Found {len(highway_types)} highway types: {list(highway_types.head().index)}")
        
        # Create a copy of roads_gdf for feature engineering
        features_gdf = roads_gdf.copy()
        
        # 1. One-hot encode highway types (limited to most common to avoid too many features)
        top_highway_types = highway_types.head(10).index  # Top 10 most common types
        for highway_type in top_highway_types:
            features_gdf[f'highway_{highway_type}'] = (features_gdf['highway'] == highway_type).astype(int)
        
        # 2. Calculate geometric features
        if features_gdf.crs != 'EPSG:4326':
            features_gdf = features_gdf.to_crs('EPSG:4326')
            
        # Road length (in degrees, will be normalized anyway)
        features_gdf['road_length'] = features_gdf.geometry.length
        
        # Road complexity (number of coordinates)
        features_gdf['road_complexity'] = features_gdf.geometry.apply(
            lambda geom: len(geom.coords) if hasattr(geom, 'coords') else 1
        )
        
        # 3. Keep only numerical columns for Highway2Vec
        numerical_cols = [col for col in features_gdf.columns 
                         if col.startswith('highway_') or col in ['road_length', 'road_complexity']]
        
        features_numerical = features_gdf[numerical_cols + ['geometry']].copy()
        
        # 4. Fill any remaining NaN values
        for col in numerical_cols:
            features_numerical[col] = features_numerical[col].fillna(0).astype(float)
        
        logger.info(f"Created {len(numerical_cols)} numerical features: {numerical_cols[:5]}...")
        logger.info(f"Feature matrix shape: {features_numerical.shape}")
        
        return features_numerical
    
    def _train_highway2vec_with_data(self, roads_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame, 
                                   joint_gdf: gpd.GeoDataFrame, h3_resolution: int, study_area_name: str) -> pd.DataFrame:
        """Train Highway2Vec with pre-loaded intermediate embeddings stage1_modalities data."""
        logger.info(f"Training Highway2Vec with pre-loaded data for H3 resolution {h3_resolution}")
        logger.info(f"Roads: {len(roads_gdf):,}, Regions: {len(regions_gdf):,}, Joints: {len(joint_gdf):,}")
        
        # Prepare features for Highway2Vec
        logger.info("Preparing features for Highway2Vec training...")
        features_for_training = self._prepare_highway2vec_features(roads_gdf, regions_gdf, joint_gdf)
        logger.info(f"Prepared features shape: {features_for_training.shape}")
        
        # Highway2Vec Training
        logger.info("Training Highway2Vec autoencoder...")
        logger.info(f"Model architecture: {self.hidden_size} -> {self.embedding_size} dimensions")
        logger.info("Initializing Highway2Vec (using RTX 3090 if available)...")
        
        # Get batch size from config (default to 128 if not specified)
        batch_size = getattr(self, 'highway2vec_batch_size', 128)
        logger.info(f"Using batch size: {batch_size} for Highway2Vec training")
        
        # Initialize Highway2Vec embedder
        from srai.embedders import Highway2VecEmbedder
        embedder = Highway2VecEmbedder(
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size
        )
        
        # Train and generate embeddings with GPU acceleration
        try:
            logger.info("Training autoencoder on road network patterns...")
            
            embeddings_gdf = embedder.fit_transform(
                regions_gdf=regions_gdf,
                features_gdf=features_for_training,
                joint_gdf=joint_gdf,
                trainer_kwargs={
                    'accelerator': 'auto',  # Use GPU if available
                    'devices': 1,
                    'max_epochs': self.highway2vec_epochs,  # GPU-optimized epochs
                    'enable_progress_bar': True
                },
                dataloader_kwargs={
                    'batch_size': self.highway2vec_batch_size
                }
            )
            
            logger.info("Highway2Vec training completed successfully!")
            logger.info(f"Generated embeddings for {len(embeddings_gdf):,} hexagons")
            
            # Convert to DataFrame format
            embeddings_df = pd.DataFrame(embeddings_gdf.drop(columns='geometry', errors='ignore'))
            
            # Format output for consistency
            if embeddings_df.index.name == 'region_id':
                embeddings_df.index.name = 'h3_index'
            elif embeddings_df.index.name is None:
                embeddings_df.index.name = 'h3_index'
                
            embeddings_df['h3_resolution'] = h3_resolution
            embeddings_df.index = embeddings_df.index.astype(str)
            
            # Log embedding statistics
            embedding_cols = [col for col in embeddings_df.columns 
                            if col not in ['h3_resolution', 'h3_index']]
            if embedding_cols:
                sample_values = embeddings_df[embedding_cols].iloc[0].values
                logger.info(f"Embedding dimensions: {len(embedding_cols)}")
                logger.info(f"Sample embedding values: {sample_values[:5]}...") 
            
            return embeddings_df.reset_index()
            
        except Exception as e:
            logger.error(f"Highway2Vec training failed: {e}")
            if "CUDA" in str(e) or "GPU" in str(e) or "torch" in str(e).lower():
                logger.info("GPU training failed. Suggestions:")
                logger.info("  - Check CUDA installation: nvidia-smi")
                logger.info("  - Check PyTorch GPU: python -c 'import torch; print(torch.cuda.is_available())'")
                logger.info("  - Fallback to CPU: set CUDA_VISIBLE_DEVICES=''")
            raise
    
    def _save_intermediate_data(self, features_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame, 
                               joint_gdf: gpd.GeoDataFrame, h3_resolution: int, study_area_name: str):
        """Save intermediate embeddings stage1_modalities SRAI data for debugging and analysis."""
        logger.info("Saving intermediate embeddings stage1_modalities data...")
        
        # Create directories
        features_dir = self.intermediate_dir / 'features_gdf'
        regions_dir = self.intermediate_dir / 'regions_gdf'
        joint_dir = self.intermediate_dir / 'joint_gdf'
        
        for dir_path in [features_dir, regions_dir, joint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames with study area and resolution
        base_name = f"{study_area_name}_res{h3_resolution}"
        
        # Save features (roads)
        features_path = features_dir / f"{base_name}_features.parquet"
        features_gdf.to_parquet(features_path)
        logger.info(f"Saved features_gdf to {features_path}")
        
        # Save regions (H3 hexagons)
        regions_path = regions_dir / f"{base_name}_regions.parquet"
        regions_gdf.to_parquet(regions_path)
        logger.info(f"Saved regions_gdf to {regions_path}")
        
        # Save joint (spatial join results [old 2024])
        joint_path = joint_dir / f"{base_name}_joint.parquet"
        joint_gdf.to_parquet(joint_path)
        logger.info(f"Saved joint_gdf to {joint_path}")
        
        logger.info(f"Intermediate data saved for {study_area_name} at resolution {h3_resolution}")

    def run_pipeline(self, study_area: Union[str, gpd.GeoDataFrame],
                    h3_resolution: int,
                    output_dir: str = None,
                    study_area_name: str = None) -> str:
        """Execute complete road processing_modalities pipeline."""
        logger.info(f"Starting roads pipeline for resolution {h3_resolution}")
        
        # Use configured output dir if not specified
        if output_dir is None:
            _roads_paths = StudyAreaPaths(self.config.get('study_area', 'default'))
            output_dir = self.config.get('output_dir', str(_roads_paths.stage1("roads")))
        
        # Load study area
        if isinstance(study_area, str):
            area_gdf = gpd.read_file(study_area)
            if study_area_name is None:
                study_area_name = Path(study_area).stem
        else:
            area_gdf = study_area
            if study_area_name is None:
                study_area_name = "unnamed"
        
        # Load road data
        roads_gdf = self.load_data(area_gdf)
        if roads_gdf.empty:
            logger.warning("No road data found for study area")
            return None
        
        # Train Highway2Vec and generate embeddings
        embeddings_df = self.highway2vec(roads_gdf, area_gdf, h3_resolution, study_area_name)
        
        # Save embeddings
        output_filename = f"roads_embeddings_res{h3_resolution}.parquet"
        output_path = self.save_embeddings(embeddings_df, output_dir, output_filename)
        
        logger.info(f"Roads embeddings saved to {output_path}")
        logger.info(f"Completed! Processed {len(embeddings_df):,} hexagons with {self.embedding_size}D Highway2Vec embeddings")
        
        return output_path