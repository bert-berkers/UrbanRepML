"""
Points of Interest (POI) Modality Processor

Processes OpenStreetMap POI data into H3 hexagon embeddings using SRAI.
Generates count-based, diversity, and optionally Hex2Vec and GeoVex embeddings.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np

# SRAI imports
from srai.loaders import OSMPbfLoader, OSMOnlineLoader
from srai.regionalizers import H3Regionalizer
from srai.embedders import CountEmbedder
from srai.joiners import IntersectionJoiner
from srai.neighbourhoods import H3Neighbourhood

# Import optional embedders
try:
    from srai.embedders import Hex2VecEmbedder
    HEX2VEC_AVAILABLE = True
except ImportError:
    HEX2VEC_AVAILABLE = False
    logging.warning("Hex2VecEmbedder not available. Install with: pip install srai[torch]")

try:
    from srai.embedders import GeoVexEmbedder
    GEOVEX_AVAILABLE = True
except ImportError:
    GEOVEX_AVAILABLE = False
    logging.warning("GeoVexEmbedder not available. Install with: pip install srai[torch]")

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


from stage1_modalities.base import ModalityProcessor


class POIProcessor(ModalityProcessor):
    """Process POI data into H3 hexagon embeddings using SRAI."""

    # Default POI categories to load (memory-efficient subset)
    DEFAULT_POI_CATEGORIES = {
        'amenity': ['restaurant', 'cafe', 'school', 'hospital', 'pharmacy', 'bank', 'fuel', 'parking'],
        'shop': ['supermarket', 'convenience', 'bakery', 'clothes', 'hairdresser', 'car_repair'],
        'leisure': ['park', 'playground', 'sports_centre', 'swimming_pool', 'fitness_centre'],
        'tourism': ['hotel', 'attraction', 'museum', 'viewpoint'],
        'office': ['government', 'company', 'lawyer', 'estate_agent'],
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize POI processor with configuration."""
        super().__init__(config)
        self.validate_config()

        # Data configuration
        self.data_source = config.get('data_source', 'osm_online')
        self.pbf_path = config.get('pbf_path', None)
        self.poi_categories = config.get('poi_categories', self.DEFAULT_POI_CATEGORIES)

        # Feature configuration
        self.compute_diversity_metrics = config.get('compute_diversity_metrics', True)
        self.use_hex2vec = config.get('use_hex2vec', False) and HEX2VEC_AVAILABLE
        self.use_geovex = config.get('use_geovex', False) and GEOVEX_AVAILABLE
        
        # GPU optimization parameters
        self.hex2vec_epochs = config.get('hex2vec_epochs', 10)
        self.geovex_epochs = config.get('geovex_epochs', 8)  
        self.batch_size = config.get('batch_size', 256)
        
        # Intermediate data saving
        self.save_intermediate = config.get('save_intermediate', False)
        self.intermediate_dir = Path(config.get('intermediate_dir', 'data/study_areas/default/embeddings/intermediate/poi'))

        logger.info(f"Initialized POIProcessor. Hex2Vec: {self.use_hex2vec}, GeoVex: {self.use_geovex}")
        logger.info(f"GPU settings - Hex2Vec epochs: {self.hex2vec_epochs}, GeoVex epochs: {self.geovex_epochs}, Batch size: {self.batch_size}")
        if self.save_intermediate:
            logger.info(f"Intermediate data will be saved to: {self.intermediate_dir}")

    def validate_config(self):
        """Validate configuration parameters."""
        if self.config.get('data_source') == 'pbf' and not self.config.get('pbf_path'):
            raise ValueError("PBF data source requires 'pbf_path' in config")

    def load_data(self, area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Load POI data for the study area."""
        logger.info(f"Loading POI data using {self.data_source}")
        
        if not self.poi_categories:
            logger.info("Loading ALL POI categories from OSM")
        else:
            logger.info(f"Loading {len(self.poi_categories)} POI categories: {list(self.poi_categories.keys())}")

        # Ensure WGS84
        if area_gdf.crs != 'EPSG:4326':
            logger.info("Converting study area to WGS84...")
            area_gdf = area_gdf.to_crs('EPSG:4326')

        # Load POIs
        logger.info("Starting OSM data download...")
        if self.data_source == 'pbf':
            logger.info(f"Loading from PBF file: {self.pbf_path}")
            loader = OSMPbfLoader()
            pois_gdf = loader.load(self.pbf_path, tags=self.poi_categories, area=area_gdf)
        else:
            logger.info("Downloading from OSM Overpass API (with caching)...")
            loader = OSMOnlineLoader()
            pois_gdf = loader.load(area_gdf, tags=self.poi_categories)

        logger.info(f"Downloaded {len(pois_gdf)} POIs from OSM")
        
        if len(pois_gdf) > 0:
            unique_tags = pois_gdf.columns.drop(['geometry'], errors='ignore').tolist()
            logger.info(f"POI feature columns: {unique_tags[:10]}{'...' if len(unique_tags) > 10 else ''}")
        
        return pois_gdf

    def process_to_h3(self, pois_gdf: gpd.GeoDataFrame, area_gdf: gpd.GeoDataFrame, 
                     h3_resolution: int, study_area_name: str = "unnamed") -> pd.DataFrame:
        """Process POI data into H3 hexagon embeddings."""
        logger.info(f"Processing POIs to H3 resolution {h3_resolution}")

        # 1. Regionalization
        logger.info("Step 1: Creating H3 hexagonal regions...")
        regionalizer = H3Regionalizer(resolution=h3_resolution)
        regions_gdf = regionalizer.transform(area_gdf)
        logger.info(f"Created {len(regions_gdf):,} H3 regions at resolution {h3_resolution}")

        # 2. Join features to regions
        logger.info("Step 2: Spatially joining POIs to H3 regions...")
        joiner = IntersectionJoiner()
        joint_gdf = joiner.transform(regions=regions_gdf, features=pois_gdf)
        poi_region_matches = len(joint_gdf)
        logger.info(f"Joined {poi_region_matches:,} POI-region pairs")
        
        # Save intermediate embeddings stage1_modalities data if requested
        if self.save_intermediate:
            self._save_intermediate_data(pois_gdf, regions_gdf, joint_gdf, h3_resolution, study_area_name)

        # 3. Count embeddings
        logger.info("Step 3: Computing count-based embeddings...")
        count_embedder = CountEmbedder()
        embeddings_gdf = count_embedder.transform(
            regions_gdf=regions_gdf,
            features_gdf=pois_gdf,
            joint_gdf=joint_gdf
        )
        logger.info(f"Count embeddings complete: {embeddings_gdf.shape[1]-1} feature columns")

        # Convert to DataFrame
        embeddings_df = pd.DataFrame(embeddings_gdf.drop(columns='geometry', errors='ignore'))

        # Add total count
        total_counts = joint_gdf.groupby(joint_gdf.index).size()
        embeddings_df['total_poi_count'] = total_counts
        embeddings_df['total_poi_count'] = embeddings_df['total_poi_count'].fillna(0).astype(int)

        # 4. Diversity metrics
        if self.compute_diversity_metrics:
            logger.info("Calculating diversity metrics")
            diversity_df = self._calculate_diversity_metrics(embeddings_df)
            embeddings_df = embeddings_df.merge(diversity_df, left_index=True, right_index=True, how='left')

        # 5. Hex2Vec embeddings (if enabled)
        if self.use_hex2vec:
            logger.info("Starting Hex2Vec embeddings generation...")
            logger.info("Computing H3 neighborhood graph...")
            try:
                neighbourhood = H3Neighbourhood(regions_gdf)
                logger.info(f"Neighborhood graph ready with {len(regions_gdf)} regions")
                
                # Hex2VecEmbedder uses features to create skip-gram embeddings
                logger.info("Initializing Hex2Vec model (32D skip-gram)...")
                hex2vec = Hex2VecEmbedder(
                    encoder_sizes=[32],  # Single hidden layer with 32 units
                    expected_output_features=list(self.poi_categories.keys())
                )
                
                logger.info(f"Training Hex2Vec model (GPU, {self.hex2vec_epochs} epochs)...")
                hex2vec_embeddings = hex2vec.fit_transform(
                    regions_gdf=regions_gdf,
                    features_gdf=pois_gdf,
                    joint_gdf=joint_gdf,
                    neighbourhood=neighbourhood,
                    trainer_kwargs={
                        'accelerator': 'auto', 
                        'devices': 1,
                        'max_epochs': self.hex2vec_epochs,
                        'enable_progress_bar': True
                    }
                )
                
                # Add hex2vec columns
                hex2vec_df = pd.DataFrame(hex2vec_embeddings)
                hex2vec_df.columns = [f'hex2vec_{i}' for i in range(hex2vec_df.shape[1])]
                embeddings_df = embeddings_df.merge(hex2vec_df, left_index=True, right_index=True, how='left')
                logger.info(f"Hex2Vec complete! Added {hex2vec_df.shape[1]} dimensions")
            except Exception as e:
                logger.error(f"Hex2Vec embedding failed: {e}")

        # 6. GeoVex embeddings (if enabled)
        if self.use_geovex:
            logger.info("Starting GeoVex hexagonal convolutional embeddings...")
            logger.info(f"Initializing GeoVex model (32D, {self.geovex_epochs} epochs, batch_size={self.batch_size})...")
            try:
                # GeoVexEmbedder uses hexagonal convolutional approach with Poisson distribution
                geovex = GeoVexEmbedder(
                    target_features=list(self.poi_categories.keys()),
                    embedding_size=32,
                    neighbourhood_radius=3,  # Smaller radius for faster training
                    convolutional_layers=2,
                    batch_size=self.batch_size  # GPU-optimized batch size
                )
                
                logger.info("Training GeoVex hexagonal convolutional model (GPU-optimized)...")
                geovex_embeddings = geovex.fit_transform(
                    regions_gdf=regions_gdf,
                    features_gdf=pois_gdf,
                    joint_gdf=joint_gdf,
                    trainer_kwargs={
                        'accelerator': 'auto', 
                        'devices': 1, 
                        'max_epochs': self.geovex_epochs,
                        'batch_size': self.batch_size,
                        'enable_progress_bar': True
                    }
                )
                
                # Add geovex columns
                geovex_df = pd.DataFrame(geovex_embeddings)
                geovex_df.columns = [f'geovex_{i}' for i in range(geovex_df.shape[1])]
                embeddings_df = embeddings_df.merge(geovex_df, left_index=True, right_index=True, how='left')
                logger.info(f"GeoVex complete! Added {geovex_df.shape[1]} dimensions")
            except Exception as e:
                logger.error(f"GeoVex embedding failed: {e}")
                if "CUDA" in str(e) or "GPU" in str(e):
                    logger.info("GPU error detected. You may need to check CUDA/PyTorch setup.")

        # Format output
        embeddings_df['h3_resolution'] = h3_resolution
        if embeddings_df.index.name == 'region_id':
            embeddings_df.index.name = 'h3_index'
        embeddings_df.index = embeddings_df.index.astype(str)

        return embeddings_df.reset_index()

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
        
        # Save features (POIs)
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

    def _calculate_diversity_metrics(self, embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate POI diversity metrics."""
        # Get category columns
        category_cols = [col for col in embeddings_df.columns 
                        if col in self.poi_categories.keys()]
        
        if not category_cols:
            return pd.DataFrame(index=embeddings_df.index)

        counts = embeddings_df[category_cols].fillna(0)
        total = counts.sum(axis=1)
        
        # Avoid division by zero
        proportions = counts.divide(total.replace(0, 1), axis=0)

        # Shannon entropy
        shannon_entropy = -1 * (proportions * np.log(proportions.replace(0, np.nan))).sum(axis=1, skipna=True)
        
        # Simpson diversity
        simpson_diversity = 1 - (proportions**2).sum(axis=1)
        
        # Richness
        richness = (counts > 0).sum(axis=1)
        
        # Evenness
        max_entropy = np.log(richness.replace(0, 1))
        evenness = shannon_entropy / max_entropy.replace(0, 1)

        return pd.DataFrame({
            'poi_shannon_entropy': shannon_entropy.fillna(0),
            'poi_simpson_diversity': simpson_diversity.fillna(0),
            'poi_richness': richness,
            'poi_evenness': evenness.fillna(0)
        }, index=embeddings_df.index)

    def run_pipeline(self, study_area: Union[str, gpd.GeoDataFrame],
                    h3_resolution: int,
                    output_dir: str = None,
                    study_area_name: str = None) -> str:
        """Execute complete POI processing_modalities pipeline."""
        logger.info(f"Starting POI pipeline for resolution {h3_resolution}")

        # Use configured output dir if not specified
        if output_dir is None:
            output_dir = self.config.get('output_dir', 'data/study_areas/default/embeddings/poi')

        # Load study area
        if isinstance(study_area, str):
            area_gdf = gpd.read_file(study_area)
            if study_area_name is None:
                study_area_name = Path(study_area).stem
        else:
            area_gdf = study_area
            if study_area_name is None:
                study_area_name = "unnamed"

        # Load POI data
        pois_gdf = self.load_data(area_gdf)
        if pois_gdf.empty:
            logger.warning("No POI data found for study area")
            return None

        # Process to H3
        embeddings_df = self.process_to_h3(pois_gdf, area_gdf, h3_resolution, study_area_name)

        # Save embeddings
        output_filename = f"poi_embeddings_res{h3_resolution}.parquet"
        output_path = self.save_embeddings(embeddings_df, output_dir, output_filename)

        logger.info(f"POI embeddings saved to {output_path}")
        logger.info(f"Processed {len(embeddings_df)} hexagons with {embeddings_df.shape[1]} features")

        return output_path