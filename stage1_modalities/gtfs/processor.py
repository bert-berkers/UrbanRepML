"""
GTFS (General Transit Feed Specification) Modality Processor

Processes public transit data into H3 hexagon embeddings using SRAI's GTFS2VecEmbedder.
Downloads GTFS feeds (default: Dutch OVapi), loads stop/trip data via GTFSLoader,
and learns transit pattern embeddings via an autoencoder.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings

import pandas as pd
import geopandas as gpd

# SRAI imports
from srai.regionalizers import H3Regionalizer
from srai.joiners import IntersectionJoiner

# Import GTFSLoader (requires gtfs_kit)
try:
    from srai.loaders import GTFSLoader
    GTFS_LOADER_AVAILABLE = True
except (ImportError, Exception):
    GTFS_LOADER_AVAILABLE = False
    logging.warning("GTFSLoader not available. Install with: pip install gtfs_kit")

# Import GTFS2VecEmbedder (requires torch)
try:
    from srai.embedders import GTFS2VecEmbedder
    GTFS2VEC_AVAILABLE = True
except ImportError:
    GTFS2VEC_AVAILABLE = False
    logging.warning("GTFS2VecEmbedder not available. Install with: pip install srai[torch]")

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


from stage1_modalities.base import ModalityProcessor
from utils import StudyAreaPaths


# Default Dutch GTFS feed URL
DEFAULT_GTFS_URL = "https://gtfs.ovapi.nl/nl/gtfs-nl.zip"


class GTFSProcessor(ModalityProcessor):
    """Process GTFS transit data into H3 hexagon embeddings using GTFS2Vec."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize GTFS processor with configuration.

        Args:
            config: Dictionary containing processor settings. Keys:
                - data_source: 'download' or 'local'
                - gtfs_url: URL to download GTFS feed (for 'download' source)
                - gtfs_path: Path to local GTFS .zip (for 'local' source)
                - study_area: Study area name for path resolution
                - skip_validation: Skip GTFS feed validation (default False)
                - fail_on_validation_errors: Fail on validation errors (default False)
                - gtfs2vec: Dict with embedding_size, hidden_size, skip_autoencoder,
                  epochs, batch_size
                - save_intermediate: Save intermediate SRAI data (default False)
        """
        super().__init__(config)
        self.validate_config()

        # Data configuration
        self.data_source = config.get('data_source', 'download')
        self.gtfs_url = config.get('gtfs_url', DEFAULT_GTFS_URL)
        self.gtfs_path = config.get('gtfs_path', None)

        # Feed validation settings
        self.skip_validation = config.get('skip_validation', False)
        self.fail_on_validation_errors = config.get('fail_on_validation_errors', False)

        # GTFS2Vec parameters from config section
        gtfs2vec_config = config.get('gtfs2vec', {})
        self.embedding_size = gtfs2vec_config.get('embedding_size', config.get('embedding_size', 64))
        self.hidden_size = gtfs2vec_config.get('hidden_size', config.get('hidden_size', 48))
        self.skip_autoencoder = gtfs2vec_config.get('skip_autoencoder', config.get('skip_autoencoder', False))
        self.gtfs2vec_epochs = gtfs2vec_config.get('epochs', config.get('gtfs2vec_epochs', 10))
        self.gtfs2vec_batch_size = gtfs2vec_config.get('batch_size', config.get('gtfs2vec_batch_size', 256))

        # Intermediate data saving
        self.save_intermediate = config.get('save_intermediate', False)
        _gtfs_paths = StudyAreaPaths(config.get('study_area', 'default'))
        self.intermediate_dir = Path(config.get('intermediate_dir', str(_gtfs_paths.intermediate("gtfs"))))

        logger.info(
            f"Initialized GTFSProcessor with GTFS2Vec "
            f"(embedding_size={self.embedding_size}, hidden_size={self.hidden_size})"
        )
        logger.info(f"GTFS2Vec epochs: {self.gtfs2vec_epochs}, batch_size: {self.gtfs2vec_batch_size}")
        if self.save_intermediate:
            logger.info(f"Intermediate data will be saved to: {self.intermediate_dir}")

    def validate_config(self):
        """Validate configuration parameters."""
        if self.config.get('data_source') == 'local' and not self.config.get('gtfs_path'):
            raise ValueError("Local data source requires 'gtfs_path' in config")

    def download_feed(self, url: Optional[str] = None, path: Optional[Path] = None) -> Path:
        """Download a GTFS feed from a URL.

        Args:
            url: URL to download from. Defaults to self.gtfs_url.
            path: Local path to save the file. Defaults to intermediate_dir/gtfs-nl.zip.

        Returns:
            Path to the downloaded .zip file.
        """
        import requests

        if url is None:
            url = self.gtfs_url
        if path is None:
            path = self.intermediate_dir / "gtfs-nl.zip"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Skip download if file already exists and is non-empty
        if path.exists() and path.stat().st_size > 0:
            logger.info(f"GTFS feed already exists at {path} ({path.stat().st_size / 1e6:.1f} MB), skipping download")
            return path

        logger.info(f"Downloading GTFS feed from {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (10 * 1024 * 1024) < 8192:
                    progress = downloaded / total_size * 100
                    logger.info(f"Download progress: {progress:.0f}% ({downloaded / 1e6:.1f} MB)")

        logger.info(f"Downloaded GTFS feed to {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return path

    def load_data(self, gtfs_path: Path) -> gpd.GeoDataFrame:
        """Load GTFS feed data using SRAI's GTFSLoader.

        GTFSLoader reads a GTFS .zip file and returns a GeoDataFrame with:
        - Stop locations as Point geometries
        - Trip count columns (trips_at_0 through trips_at_23) for hourly aggregation
        - Direction columns for each hour

        Args:
            gtfs_path: Path to the GTFS .zip file.

        Returns:
            GeoDataFrame with stop features indexed by feature_id.
        """
        if not GTFS_LOADER_AVAILABLE:
            raise RuntimeError(
                "GTFSLoader requires gtfs_kit package. "
                "Install with: pip install gtfs_kit"
            )

        logger.info(f"Loading GTFS feed from {gtfs_path}")
        loader = GTFSLoader()
        features_gdf = loader.load(
            gtfs_file=Path(gtfs_path),
            fail_on_validation_errors=self.fail_on_validation_errors,
            skip_validation=self.skip_validation,
        )

        logger.info(f"Loaded {len(features_gdf)} transit stops from GTFS feed")
        if len(features_gdf) > 0:
            trip_cols = [c for c in features_gdf.columns if c.startswith('trips_at_')]
            direction_cols = [c for c in features_gdf.columns if c.startswith('directions_at_')]
            logger.info(f"Trip columns: {len(trip_cols)}, Direction columns: {len(direction_cols)}")
            logger.info(f"Feature columns: {list(features_gdf.columns[:10])}{'...' if len(features_gdf.columns) > 10 else ''}")

        return features_gdf

    def process_to_h3(self, features_gdf: gpd.GeoDataFrame, area_gdf: gpd.GeoDataFrame,
                      h3_resolution: int, study_area_name: str = "unnamed") -> pd.DataFrame:
        """Process GTFS features into H3 hexagon embeddings using GTFS2Vec.

        Pipeline:
        1. Create H3 tessellation from study area boundary
        2. Spatial join stops to hexagons
        3. Train GTFS2Vec autoencoder on transit patterns
        4. Output region-indexed embeddings

        Args:
            features_gdf: GeoDataFrame from load_data() with stop features.
            area_gdf: Study area boundary GeoDataFrame.
            h3_resolution: H3 resolution for tessellation.
            study_area_name: Name for intermediate file naming.

        Returns:
            DataFrame with region_id index, gtfs2vec embedding columns, and h3_resolution.
        """
        if not GTFS2VEC_AVAILABLE:
            raise RuntimeError(
                "GTFS2VecEmbedder requires PyTorch. "
                "Install with: pip install srai[torch]"
            )

        logger.info(f"Processing GTFS stops to H3 resolution {h3_resolution}")

        # 1. Regionalization
        logger.info("Step 1: Creating H3 hexagonal regions...")
        regionalizer = H3Regionalizer(resolution=h3_resolution)
        regions_gdf = regionalizer.transform(area_gdf)
        logger.info(f"Created {len(regions_gdf):,} H3 regions at resolution {h3_resolution}")

        # 2. Spatial join stops to regions
        logger.info("Step 2: Spatially joining transit stops to H3 regions...")
        joiner = IntersectionJoiner()
        joint_gdf = joiner.transform(regions=regions_gdf, features=features_gdf)
        stop_region_matches = len(joint_gdf)
        logger.info(f"Joined {stop_region_matches:,} stop-region pairs")

        if stop_region_matches == 0:
            logger.warning("No transit stops fall within study area hexagons. Returning empty embeddings.")
            empty_df = pd.DataFrame(index=pd.Index([], name='region_id'))
            empty_df['h3_resolution'] = h3_resolution
            return empty_df.reset_index()

        # Save intermediate data if requested
        if self.save_intermediate:
            self._save_intermediate_data(features_gdf, regions_gdf, joint_gdf, h3_resolution, study_area_name)

        # 3. GTFS2Vec embedding
        logger.info("Step 3: Training GTFS2Vec autoencoder on transit patterns...")
        logger.info(
            f"Model architecture: input -> {self.hidden_size} (hidden) -> "
            f"{self.embedding_size} (embedding)"
        )

        embedder = GTFS2VecEmbedder(
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            skip_autoencoder=self.skip_autoencoder,
        )

        try:
            logger.info(f"Training GTFS2Vec ({self.gtfs2vec_epochs} epochs)...")
            embeddings_df = embedder.fit_transform(
                regions_gdf=regions_gdf,
                features_gdf=features_gdf,
                joint_gdf=joint_gdf,
                trainer_kwargs={
                    'accelerator': 'auto',
                    'devices': 1,
                    'max_epochs': self.gtfs2vec_epochs,
                    'enable_progress_bar': True,
                },
            )

            logger.info(f"GTFS2Vec training completed. Generated embeddings for {len(embeddings_df):,} regions")

            # Rename columns to gtfs2vec_N for clarity
            embedding_cols = [c for c in embeddings_df.columns if c != 'geometry']
            col_rename = {old: f'gtfs2vec_{i}' for i, old in enumerate(embedding_cols)}
            embeddings_df = embeddings_df.rename(columns=col_rename)

            logger.info(f"Embedding dimensions: {len(embedding_cols)}")

        except Exception as e:
            logger.error(f"GTFS2Vec training failed: {e}")
            if "CUDA" in str(e) or "GPU" in str(e) or "torch" in str(e).lower():
                logger.info("GPU training failed. Suggestions:")
                logger.info("  - Check CUDA installation: nvidia-smi")
                logger.info("  - Check PyTorch GPU: python -c 'import torch; print(torch.cuda.is_available())'")
                logger.info("  - Fallback to CPU: set CUDA_VISIBLE_DEVICES=''")
            raise

        # Format output: ensure region_id is the canonical index name
        embeddings_df['h3_resolution'] = h3_resolution
        embeddings_df.index = embeddings_df.index.astype(str)
        if embeddings_df.index.name != 'region_id':
            embeddings_df.index.name = 'region_id'

        return embeddings_df.reset_index()

    def _save_intermediate_data(self, features_gdf: gpd.GeoDataFrame, regions_gdf: gpd.GeoDataFrame,
                                joint_gdf: gpd.GeoDataFrame, h3_resolution: int, study_area_name: str):
        """Save intermediate SRAI data for debugging and analysis."""
        logger.info("Saving intermediate data...")

        features_dir = self.intermediate_dir / 'features_gdf'
        regions_dir = self.intermediate_dir / 'regions_gdf'
        joint_dir = self.intermediate_dir / 'joint_gdf'

        for dir_path in [features_dir, regions_dir, joint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        base_name = f"{study_area_name}_res{h3_resolution}"

        features_path = features_dir / f"{base_name}_features.parquet"
        features_gdf.to_parquet(features_path)
        logger.info(f"Saved features_gdf to {features_path}")

        regions_path = regions_dir / f"{base_name}_regions.parquet"
        regions_gdf.to_parquet(regions_path)
        logger.info(f"Saved regions_gdf to {regions_path}")

        joint_path = joint_dir / f"{base_name}_joint.parquet"
        joint_gdf.to_parquet(joint_path)
        logger.info(f"Saved joint_gdf to {joint_path}")

        logger.info(f"Intermediate data saved for {study_area_name} at resolution {h3_resolution}")

    def run_pipeline(self, study_area: Union[str, gpd.GeoDataFrame],
                     h3_resolution: int,
                     output_dir: str = None,
                     study_area_name: str = None) -> str:
        """Execute complete GTFS processing pipeline.

        Orchestrates: download (if needed) -> load -> process_to_h3 -> save.

        Args:
            study_area: Path to study area boundary file, or GeoDataFrame.
            h3_resolution: H3 resolution for tessellation.
            output_dir: Directory to save embeddings. Defaults to StudyAreaPaths.
            study_area_name: Name for file naming. Derived from path if not provided.

        Returns:
            Path to the saved embeddings parquet file, or None if no data found.
        """
        logger.info(f"Starting GTFS pipeline for resolution {h3_resolution}")

        # Use configured output dir if not specified
        if output_dir is None:
            _gtfs_paths = StudyAreaPaths(self.config.get('study_area', 'default'))
            output_dir = self.config.get('output_dir', str(_gtfs_paths.stage1("gtfs")))

        # Load study area boundary
        if isinstance(study_area, str):
            area_gdf = gpd.read_file(study_area)
            if study_area_name is None:
                study_area_name = Path(study_area).stem
        else:
            area_gdf = study_area
            if study_area_name is None:
                study_area_name = "unnamed"

        # Ensure WGS84
        if area_gdf.crs != 'EPSG:4326':
            logger.info("Converting study area to WGS84...")
            area_gdf = area_gdf.to_crs('EPSG:4326')

        # Obtain GTFS feed path
        if self.data_source == 'download':
            gtfs_path = self.download_feed()
        elif self.data_source == 'local':
            gtfs_path = Path(self.gtfs_path)
            if not gtfs_path.exists():
                raise FileNotFoundError(f"GTFS file not found: {gtfs_path}")
        else:
            raise ValueError(f"Unknown data_source: {self.data_source}. Use 'download' or 'local'.")

        # Load GTFS data
        features_gdf = self.load_data(gtfs_path)
        if features_gdf.empty:
            logger.warning("No transit data found in GTFS feed")
            return None

        # Process to H3 embeddings
        embeddings_df = self.process_to_h3(features_gdf, area_gdf, h3_resolution, study_area_name)

        # Save embeddings
        output_filename = f"gtfs_embeddings_res{h3_resolution}.parquet"
        output_path = self.save_embeddings(embeddings_df, output_dir, output_filename)

        logger.info(f"GTFS embeddings saved to {output_path}")
        logger.info(
            f"Completed! Processed {len(embeddings_df):,} hexagons with "
            f"{self.embedding_size}D GTFS2Vec embeddings"
        )

        return output_path
