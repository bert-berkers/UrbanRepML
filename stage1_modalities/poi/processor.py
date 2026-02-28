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
from utils import StudyAreaPaths


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

        # Study area and year for path construction
        self.study_area_name = config.get('study_area', 'default')
        self.year = config.get('year', 2022)

        # Data configuration
        self.data_source = config.get('data_source', 'osm_online')
        self.pbf_path = Path(config['pbf_path']) if config.get('pbf_path') else None
        self.poi_categories = config.get('poi_categories', self.DEFAULT_POI_CATEGORIES)

        # Feature configuration
        self.compute_diversity_metrics = config.get('compute_diversity_metrics', True)
        self.use_hex2vec = config.get('use_hex2vec', False) and HEX2VEC_AVAILABLE
        self.use_geovex = config.get('use_geovex', False) and GEOVEX_AVAILABLE

        # GPU optimization parameters
        self.hex2vec_epochs = config.get('hex2vec_epochs', 10)
        self.hex2vec_encoder_sizes = config.get('hex2vec_encoder_sizes', [150, 75, 50])
        self.geovex_epochs = config.get('geovex_epochs', 8)
        self.geovex_embedding_size = config.get('geovex_embedding_size', 32)
        self.geovex_neighbourhood_radius = config.get('geovex_neighbourhood_radius', 4)
        self.geovex_convolutional_layers = config.get('geovex_convolutional_layers', 2)
        self.batch_size = config.get('batch_size', 4096)

        # Intermediate data saving
        self.save_intermediate = config.get('save_intermediate', False)
        self._paths = StudyAreaPaths(self.study_area_name)
        self.intermediate_dir = Path(config.get('intermediate_dir', str(self._paths.intermediate("poi"))))

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
            loader = OSMPbfLoader(pbf_file=self.pbf_path)
            pois_gdf = loader.load(area_gdf, tags=self.poi_categories)
        else:
            logger.info("Downloading from OSM Overpass API (with caching)...")
            loader = OSMOnlineLoader()
            pois_gdf = loader.load(area_gdf, tags=self.poi_categories)

        logger.info(f"Downloaded {len(pois_gdf)} POIs from OSM")
        
        if len(pois_gdf) > 0:
            unique_tags = pois_gdf.columns.drop(['geometry'], errors='ignore').tolist()
            logger.info(f"POI feature columns: {unique_tags[:10]}{'...' if len(unique_tags) > 10 else ''}")
        
        return pois_gdf

    def load_intermediates(
        self, h3_resolution: int, study_area_name: Optional[str] = None
    ) -> tuple:
        """Load pre-saved intermediate data (regions, features, joint).

        Loads the three parquet files that ``_save_intermediate_data`` writes
        during a full ``process_to_h3`` run.

        Args:
            h3_resolution: H3 resolution that was used during processing.
            study_area_name: Study area name embedded in filenames.
                Defaults to ``self.study_area_name``.

        Returns:
            Tuple of ``(regions_gdf, features_gdf, joint_gdf)``.

        Raises:
            FileNotFoundError: If any of the three parquet files is missing.
        """
        if study_area_name is None:
            study_area_name = self.study_area_name

        base_name = f"{study_area_name}_res{h3_resolution}"

        regions_path = self.intermediate_dir / "regions_gdf" / f"{base_name}_regions.parquet"
        features_path = self.intermediate_dir / "features_gdf" / f"{base_name}_features.parquet"
        joint_path = self.intermediate_dir / "joint_gdf" / f"{base_name}_joint.parquet"

        for path, label in [
            (regions_path, "regions_gdf"),
            (features_path, "features_gdf"),
            (joint_path, "joint_gdf"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Intermediate {label} not found at {path}. "
                    f"Run the full pipeline with save_intermediate=True first."
                )

        logger.info(f"Loading intermediates for {study_area_name} res{h3_resolution}")

        # Load parquets as plain DataFrames first (they may lack geo metadata)
        regions_df = pd.read_parquet(regions_path)
        features_df = pd.read_parquet(features_path)
        joint_df = pd.read_parquet(joint_path)

        # Convert to GeoDataFrames if geometry column exists
        # Geometry may be stored as WKB binary, so deserialize it
        def _to_geodataframe(df, name="data"):
            if 'geometry' not in df.columns:
                return df
            from shapely import wkb
            try:
                # Try to deserialize WKB geometry
                df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x)
                return gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            except Exception as e:
                logger.warning(f"Failed to parse geometry for {name}: {e}. Returning as DataFrame.")
                return df

        regions_gdf = _to_geodataframe(regions_df, "regions")
        features_gdf = _to_geodataframe(features_df, "features")
        joint_gdf = _to_geodataframe(joint_df, "joint")

        logger.info(
            f"Loaded {len(regions_gdf):,} regions, "
            f"{len(features_gdf):,} features, "
            f"{len(joint_gdf):,} joint pairs"
        )
        return regions_gdf, features_gdf, joint_gdf

    def run_count_embeddings(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """Compute count-based POI embeddings with optional diversity metrics.

        Runs SRAI's ``CountEmbedder``, adds a total POI count column, and
        optionally appends Shannon entropy / Simpson diversity / richness /
        evenness columns (controlled by ``self.compute_diversity_metrics``).

        Args:
            regions_gdf: H3 hexagonal regions.
            features_gdf: POI features GeoDataFrame.
            joint_gdf: Spatial join of regions and features.

        Returns:
            DataFrame with original column names (e.g. ``amenity_restaurant``,
            ``poi_shannon_entropy``), indexed by ``region_id``.
        """
        logger.info("Computing count-based embeddings...")
        count_embedder = CountEmbedder()
        count_df = count_embedder.transform(
            regions_gdf=regions_gdf,
            features_gdf=features_gdf,
            joint_gdf=joint_gdf,
        )
        logger.info(f"Count embeddings complete: {count_df.shape[1]} feature columns")

        # CountEmbedder.transform returns pd.DataFrame (no geometry column)
        embeddings_df = pd.DataFrame(count_df)

        # Add total POI count per region (joint_gdf has MultiIndex: region_id, feature_id)
        total_counts = joint_gdf.groupby(level=0).size()
        embeddings_df["total_poi_count"] = total_counts
        embeddings_df["total_poi_count"] = embeddings_df["total_poi_count"].fillna(0).astype(int)

        # Diversity metrics
        if self.compute_diversity_metrics:
            logger.info("Calculating diversity metrics")
            diversity_df = self._calculate_diversity_metrics(embeddings_df)
            embeddings_df = embeddings_df.merge(
                diversity_df, left_index=True, right_index=True, how="left"
            )

        # Ensure region_id index
        embeddings_df.index = embeddings_df.index.astype(str)
        if embeddings_df.index.name != "region_id":
            embeddings_df.index.name = "region_id"

        return embeddings_df

    def run_hex2vec(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """Compute Hex2Vec skip-gram embeddings.

        Builds an H3 neighbourhood graph and trains a Hex2Vec encoder on the
        POI feature matrix.

        Args:
            regions_gdf: H3 hexagonal regions.
            features_gdf: POI features GeoDataFrame.
            joint_gdf: Spatial join of regions and features.

        Returns:
            DataFrame with columns ``hex2vec_0 .. hex2vec_N``, indexed by
            ``region_id``.

        Raises:
            ImportError: If ``srai[torch]`` is not installed.
            RuntimeError: If the Hex2Vec training fails.
        """
        if not HEX2VEC_AVAILABLE:
            raise ImportError(
                "Hex2VecEmbedder not available. Install with: pip install srai[torch]"
            )

        logger.info("Starting Hex2Vec embeddings generation...")
        logger.info("Computing H3 neighborhood graph...")
        neighbourhood = H3Neighbourhood(regions_gdf)
        logger.info(f"Neighborhood graph ready with {len(regions_gdf)} regions")

        logger.info(f"Initializing Hex2Vec model (encoder_sizes={self.hex2vec_encoder_sizes})...")
        hex2vec = Hex2VecEmbedder(
            encoder_sizes=self.hex2vec_encoder_sizes,
            expected_output_features=self.poi_categories,
        )

        logger.info(f"Training Hex2Vec model (GPU, {self.hex2vec_epochs} epochs, batch_size={self.batch_size})...")
        hex2vec_embeddings = hex2vec.fit_transform(
            regions_gdf=regions_gdf,
            features_gdf=features_gdf,
            joint_gdf=joint_gdf,
            neighbourhood=neighbourhood,
            batch_size=self.batch_size,
            trainer_kwargs={
                "accelerator": "auto",
                "devices": 1,
                "max_epochs": self.hex2vec_epochs,
                "enable_progress_bar": True,
            },
        )

        hex2vec_df = pd.DataFrame(hex2vec_embeddings)
        hex2vec_df.columns = [f"hex2vec_{i}" for i in range(hex2vec_df.shape[1])]
        logger.info(f"Hex2Vec complete! Added {hex2vec_df.shape[1]} dimensions")

        # Ensure region_id index
        hex2vec_df.index = hex2vec_df.index.astype(str)
        if hex2vec_df.index.name != "region_id":
            hex2vec_df.index.name = "region_id"

        return hex2vec_df

    def run_geovex(
        self,
        regions_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        joint_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """Compute GeoVex hexagonal convolutional embeddings.

        Trains a GeoVex variational autoencoder on the POI feature matrix
        using a hexagonal convolutional neighbourhood.

        Args:
            regions_gdf: H3 hexagonal regions.
            features_gdf: POI features GeoDataFrame.
            joint_gdf: Spatial join of regions and features.

        Returns:
            DataFrame with columns ``geovex_0 .. geovex_N``, indexed by
            ``region_id``.

        Raises:
            ImportError: If ``srai[torch]`` is not installed.
            RuntimeError: If the GeoVex training fails.
        """
        if not GEOVEX_AVAILABLE:
            raise ImportError(
                "GeoVexEmbedder not available. Install with: pip install srai[torch]"
            )

        logger.info("Starting GeoVex hexagonal convolutional embeddings...")
        logger.info(
            f"Initializing GeoVex model ({self.geovex_embedding_size}D, "
            f"{self.geovex_epochs} epochs, batch_size={self.batch_size})..."
        )
        neighbourhood = H3Neighbourhood(regions_gdf)

        geovex = GeoVexEmbedder(
            target_features=self.poi_categories,
            embedding_size=self.geovex_embedding_size,
            neighbourhood_radius=self.geovex_neighbourhood_radius,
            convolutional_layers=self.geovex_convolutional_layers,
            batch_size=self.batch_size,
        )

        logger.info("Training GeoVex hexagonal convolutional model (GPU-optimized)...")
        geovex_embeddings = geovex.fit_transform(
            regions_gdf=regions_gdf,
            features_gdf=features_gdf,
            joint_gdf=joint_gdf,
            neighbourhood=neighbourhood,
            trainer_kwargs={
                "accelerator": "auto",
                "devices": 1,
                "max_epochs": self.geovex_epochs,
                "enable_progress_bar": True,
            },
        )

        geovex_df = pd.DataFrame(geovex_embeddings)
        geovex_df.columns = [f"geovex_{i}" for i in range(geovex_df.shape[1])]
        logger.info(f"GeoVex complete! Added {geovex_df.shape[1]} dimensions")

        # Ensure region_id index
        geovex_df.index = geovex_df.index.astype(str)
        if geovex_df.index.name != "region_id":
            geovex_df.index.name = "region_id"

        return geovex_df

    def process_to_h3(self, pois_gdf: gpd.GeoDataFrame, area_gdf: gpd.GeoDataFrame,
                     h3_resolution: int, study_area_name: str = "unnamed") -> pd.DataFrame:
        """Process POI data into H3 hexagon embeddings.

        Orchestrates regionalization, spatial joining, and embedding generation.
        Delegates to ``run_count_embeddings``, ``run_hex2vec``, and
        ``run_geovex`` for the actual computation.
        """
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

        # Save intermediate data if requested
        if self.save_intermediate:
            self._save_intermediate_data(pois_gdf, regions_gdf, joint_gdf, h3_resolution, study_area_name)

        # 3. Count embeddings (includes diversity metrics)
        embeddings_df = self.run_count_embeddings(regions_gdf, pois_gdf, joint_gdf)

        # 4. Hex2Vec embeddings (if enabled)
        if self.use_hex2vec:
            try:
                hex2vec_df = self.run_hex2vec(regions_gdf, pois_gdf, joint_gdf)
                embeddings_df = embeddings_df.merge(
                    hex2vec_df, left_index=True, right_index=True, how="left"
                )
            except Exception as e:
                logger.error(f"Hex2Vec embedding failed: {e}")

        # 5. GeoVex embeddings (if enabled)
        if self.use_geovex:
            try:
                geovex_df = self.run_geovex(regions_gdf, pois_gdf, joint_gdf)
                embeddings_df = embeddings_df.merge(
                    geovex_df, left_index=True, right_index=True, how="left"
                )
            except Exception as e:
                logger.error(f"GeoVex embedding failed: {e}")
                if "CUDA" in str(e) or "GPU" in str(e):
                    logger.info("GPU error detected. You may need to check CUDA/PyTorch setup.")

        # Drop metadata columns -- only embedding features should remain
        metadata_cols = ['total_poi_count', 'h3_resolution']
        embeddings_df = embeddings_df.drop(
            columns=[c for c in metadata_cols if c in embeddings_df.columns]
        )

        # Rename all columns to P00, P01, ... Pxx (canonical prefix convention)
        n_cols = len(embeddings_df.columns)
        width = max(2, len(str(n_cols - 1)))  # At least 2 digits
        new_col_names = {
            old: f"P{i:0{width}d}"
            for i, old in enumerate(embeddings_df.columns)
        }
        embeddings_df = embeddings_df.rename(columns=new_col_names)

        # Ensure region_id is the canonical index name
        embeddings_df.index = embeddings_df.index.astype(str)
        if embeddings_df.index.name != 'region_id':
            embeddings_df.index.name = 'region_id'

        return embeddings_df

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
        """Calculate POI diversity metrics.

        CountEmbedder produces columns like ``amenity_restaurant``,
        ``shop_bakery``, etc.  We aggregate by category prefix (the keys
        of ``self.poi_categories``) to obtain per-category counts, then
        compute Shannon entropy, Simpson diversity, richness, and evenness
        across the category-level totals.
        """
        # Aggregate sub-type columns into per-category totals.
        # E.g. amenity_restaurant + amenity_cafe + ... -> amenity_total
        cat_totals = {}
        for cat_key in self.poi_categories.keys():
            prefix = cat_key + '_'
            matching_cols = [c for c in embeddings_df.columns if c.startswith(prefix)]
            if matching_cols:
                cat_totals[cat_key] = embeddings_df[matching_cols].fillna(0).sum(axis=1)

        if not cat_totals:
            return pd.DataFrame(index=embeddings_df.index)

        counts = pd.DataFrame(cat_totals, index=embeddings_df.index).fillna(0)
        total = counts.sum(axis=1)

        # Avoid division by zero
        proportions = counts.divide(total.replace(0, 1), axis=0)

        # Shannon entropy
        shannon_entropy = -1 * (proportions * np.log(proportions.replace(0, np.nan))).sum(axis=1, skipna=True)

        # Simpson diversity
        simpson_diversity = 1 - (proportions**2).sum(axis=1)

        # Richness (number of categories present)
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
                    study_area_name: str = None) -> str:
        """Execute complete POI processing pipeline.

        Args:
            study_area: Path to a boundary file or a GeoDataFrame.
            h3_resolution: H3 resolution level (e.g. 9, 10).
            study_area_name: Optional override for study area name.

        Returns:
            Absolute path to the saved parquet file.
        """
        logger.info(f"Starting POI pipeline for resolution {h3_resolution}")

        # Load study area
        if isinstance(study_area, str):
            area_gdf = gpd.read_file(study_area)
            if study_area_name is None:
                study_area_name = Path(study_area).stem
        else:
            area_gdf = study_area
            if study_area_name is None:
                study_area_name = self.study_area_name

        # Load POI data
        pois_gdf = self.load_data(area_gdf)
        if pois_gdf.empty:
            logger.warning("No POI data found for study area")
            return None

        # Process to H3 -- returns DataFrame indexed by region_id with P-prefixed columns
        embeddings_df = self.process_to_h3(pois_gdf, area_gdf, h3_resolution, study_area_name)

        # Save using StudyAreaPaths convention:
        #   {study_area}_res{resolution}_{year}.parquet
        output_path = self._paths.embedding_file("poi", h3_resolution, self.year)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_df.to_parquet(output_path)

        logger.info(f"POI embeddings saved to {output_path}")
        logger.info(f"Processed {len(embeddings_df)} hexagons with {embeddings_df.shape[1]} features")

        return str(output_path)