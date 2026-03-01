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

        # Study area path helper (needed early for PBF auto-resolve)
        self._study_area_name = config.get('study_area', 'default')
        self._paths = StudyAreaPaths(self._study_area_name)

        self.validate_config()

        # Data configuration
        self.data_source = config.get('data_source', 'osm_online')
        if config.get('pbf_path'):
            self.pbf_path = Path(config['pbf_path'])
        elif self.data_source == 'pbf':
            # Auto-resolve from study area osm/ directory
            osm_date = config.get('osm_date', 'latest')
            self.pbf_path = self._paths.osm_snapshot_pbf(osm_date)
            logger.info(f"Auto-resolved PBF path: {self.pbf_path}")
        else:
            self.pbf_path = None
        self.road_types = config.get('road_types', DEFAULT_ROAD_TYPES)

        # Highway2Vec parameters from config section
        highway2vec_config = config.get('highway2vec', {})
        self.embedding_size = highway2vec_config.get('embedding_size', config.get('embedding_size', 30))
        self.hidden_size = highway2vec_config.get('hidden_size', config.get('hidden_size', 64))
        self.highway2vec_epochs = highway2vec_config.get('epochs', config.get('highway2vec_epochs', 25))
        self.highway2vec_batch_size = highway2vec_config.get('batch_size', config.get('highway2vec_batch_size', 128))

        # Intermediate data saving
        self.save_intermediate = config.get('save_intermediate', False)
        self.intermediate_dir = Path(config.get('intermediate_dir', str(self._paths.intermediate("roads"))))
        
        logger.info(f"Initialized RoadsProcessor with Highway2Vec (embedding_size={self.embedding_size})")
        logger.info(f"Highway2Vec epochs: {self.highway2vec_epochs}, batch_size: {self.highway2vec_batch_size}")
        if self.save_intermediate:
            logger.info(f"Intermediate data will be saved to: {self.intermediate_dir}")

    def validate_config(self):
        """Validate configuration parameters.

        When ``data_source`` is ``'pbf'`` and no explicit ``pbf_path`` is
        provided, the processor auto-resolves the PBF path from the study
        area's ``osm/`` directory via :pymethod:`StudyAreaPaths.osm_snapshot_pbf`.
        """
        # No longer raise when pbf_path is missing -- auto-resolve in __init__
        pass

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
            loader = OSMPbfLoader(pbf_file=self.pbf_path)
            roads_gdf = loader.load(area_gdf, tags=tags)
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
        if not HIGHWAY2VEC_AVAILABLE:
            raise RuntimeError(
                "RoadsProcessor.highway2vec() requires Highway2VecEmbedder. "
                "Install with: pip install srai[torch]"
            )

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

            # Rename embedding columns to R-prefixed convention (R00, R01, ..., Rxx)
            from stage1_modalities import MODALITY_PREFIXES
            prefix = MODALITY_PREFIXES["roads"]  # "R"
            rename_map = {}
            for col in embeddings_df.columns:
                if isinstance(col, int) or (isinstance(col, str) and col.isdigit()):
                    idx = int(col)
                    rename_map[col] = f"{prefix}{idx:02d}"
            if rename_map:
                embeddings_df = embeddings_df.rename(columns=rename_map)
                logger.info(f"Renamed {len(rename_map)} columns with '{prefix}' prefix")

            # Keep only R-prefixed embedding columns (no metadata columns)
            embedding_cols = [col for col in embeddings_df.columns
                            if col.startswith(prefix) and col[len(prefix):].isdigit()]
            embeddings_df = embeddings_df[embedding_cols]

            # Ensure index is string-typed region_id
            embeddings_df.index = embeddings_df.index.astype(str)
            if embeddings_df.index.name != 'region_id':
                embeddings_df.index.name = 'region_id'

            # Log embedding statistics
            if embedding_cols:
                sample_values = embeddings_df[embedding_cols].iloc[0].values
                logger.info(f"Embedding dimensions: {len(embedding_cols)}")
                logger.info(f"Sample embedding values: {sample_values[:5]}...")

            return embeddings_df

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
        """Prepare numerical features for Highway2Vec training.

        Highway2VecEmbedder.fit() strips geometry, then trains an autoencoder
        on features_gdf.values. The index must be a named index that matches
        joint_gdf.index.names[1] (the feature-level of the MultiIndex).
        """
        logger.info("Preparing numerical features from roads data...")

        # Highway type distribution
        highway_col = 'highway' if 'highway' in roads_gdf.columns else None
        if highway_col:
            highway_types = roads_gdf[highway_col].value_counts()
            logger.info(f"Found {len(highway_types)} highway types: {list(highway_types.head().index)}")
        else:
            highway_types = pd.Series(dtype=int)
            logger.warning("No 'highway' column found in roads data")

        # Create a copy preserving the original index (from OSM loader)
        features_gdf = roads_gdf.copy()

        # 1. One-hot encode highway types (limited to most common)
        if len(highway_types) > 0:
            top_highway_types = highway_types.head(10).index
            for highway_type in top_highway_types:
                features_gdf[f'highway_{highway_type}'] = (features_gdf[highway_col] == highway_type).astype(int)

        # 2. Calculate geometric features
        if features_gdf.crs and features_gdf.crs != 'EPSG:4326':
            features_gdf = features_gdf.to_crs('EPSG:4326')

        # Road length (in degrees, will be normalized by autoencoder anyway)
        features_gdf['road_length'] = features_gdf.geometry.length

        # Road complexity (number of coordinates)
        features_gdf['road_complexity'] = features_gdf.geometry.apply(
            lambda geom: len(geom.coords) if hasattr(geom, 'coords') else 1
        )

        # 3. Keep only numerical columns + geometry for Highway2Vec
        numerical_cols = [col for col in features_gdf.columns
                         if col.startswith('highway_') or col in ['road_length', 'road_complexity']]

        features_numerical = features_gdf[numerical_cols + ['geometry']].copy()

        # 4. Fill NaN values
        for col in numerical_cols:
            features_numerical[col] = features_numerical[col].fillna(0).astype(float)

        # 5. Ensure index name matches joint_gdf second level (SRAI _validate_indexes requirement)
        if joint_gdf is not None and isinstance(joint_gdf.index, pd.MultiIndex):
            expected_name = joint_gdf.index.names[1]
            if features_numerical.index.name != expected_name:
                logger.info(f"Aligning features index name: {features_numerical.index.name!r} -> {expected_name!r}")
                features_numerical.index.name = expected_name

        logger.info(f"Created {len(numerical_cols)} numerical features: {numerical_cols[:5]}...")
        logger.info(f"Feature matrix shape: {features_numerical.shape}")

        return features_numerical
    
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
        
        # Save joint (spatial join results)
        joint_path = joint_dir / f"{base_name}_joint.parquet"
        joint_gdf.to_parquet(joint_path)
        logger.info(f"Saved joint_gdf to {joint_path}")
        
        logger.info(f"Intermediate data saved for {study_area_name} at resolution {h3_resolution}")

    def run_pipeline(self, study_area: Union[str, gpd.GeoDataFrame],
                    h3_resolution: int,
                    output_dir: str = None,
                    study_area_name: str = None,
                    year: int = 2022) -> str:
        """Execute complete road processing_modalities pipeline."""
        logger.info(f"Starting roads pipeline for resolution {h3_resolution}")

        study_area_key = self.config.get('study_area', 'default')
        paths = StudyAreaPaths(study_area_key)

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

        # Save embeddings using canonical StudyAreaPaths filename
        # Convention: region_id as index, only embedding columns, saved with index=True
        if output_dir is not None:
            # Caller override: save to custom directory with canonical filename
            out_path = Path(output_dir) / paths.embedding_file("roads", h3_resolution, year).name
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            embeddings_df.to_parquet(out_path)
            output_path = str(out_path)
        else:
            # Standard path: use StudyAreaPaths.embedding_file() directly
            out_path = paths.embedding_file("roads", h3_resolution, year)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            embeddings_df.to_parquet(out_path)
            output_path = str(out_path)

        logger.info(f"Roads embeddings saved to {output_path}")
        logger.info(f"Completed! Processed {len(embeddings_df):,} hexagons with {self.embedding_size}D Highway2Vec embeddings")

        return output_path