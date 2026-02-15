"""
Aerial Imagery Processor for UrbanRepML Stage 1.

Fetches per-hexagon aerial images from PDOK (224x224) and encodes them
with DINOv3 ViT-L/16 (satellite-pretrained, 1024D embeddings).

Two-phase pipeline:
  Phase 1 - FETCH: Parallel download 224x224 images from PDOK WMS
  Phase 2 - ENCODE: GPU batch encoding with DINOv3

Output contract (matches other modalities):
  - region_id: string column (after reset_index)
  - dinov3_0 .. dinov3_1023: float embedding columns
  - h3_resolution: int column
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import logging

from stage1_modalities.base import ModalityProcessor
from .pdok_client import PDOKClient
from .dinov3_encoder import DINOv3Encoder

logger = logging.getLogger(__name__)


class AerialImageryProcessor(ModalityProcessor):
    """Process aerial imagery into H3-indexed DINOv3 embeddings.

    Lifecycle:
        1. __init__(config) -- set up PDOKClient + DINOv3Encoder
        2. process(regions_gdf, h3_resolution) -- fetch + encode
        3. run_pipeline(study_area, h3_resolution) -- full end-to-end

    Config keys:
        pdok_year: str       -- PDOK imagery year (default 'current')
        image_size: int      -- Tile size in pixels (default 224)
        max_workers: int     -- Parallel fetch threads (default 64)
        batch_size: int      -- GPU encoding batch size (default 32)
        device: str          -- 'cuda' or 'cpu' (default auto-detect)
        study_area: str      -- Study area name (default 'default')
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # PDOK client for image fetching
        self.pdok_client = PDOKClient(
            year=config.get('pdok_year', 'current'),
            image_size=config.get('image_size', 224),
            max_retries=config.get('max_retries', 3),
            max_workers=config.get('max_workers', 64),
        )

        # DINOv3 encoder (lazy-loaded on first use to avoid GPU allocation
        # when only fetching images)
        self._encoder: Optional[DINOv3Encoder] = None
        self._encoder_config = {
            'device': config.get('device', None),
            'batch_size': config.get('batch_size', 32),
        }

        # Paths via StudyAreaPaths
        from utils import StudyAreaPaths
        study_area = config.get('study_area', 'default')
        self._paths = StudyAreaPaths(study_area)
        self.tile_dir = self._paths.intermediate('aerial_imagery') / 'tiles'

    @property
    def encoder(self) -> DINOv3Encoder:
        """Lazy-load DINOv3 encoder on first access."""
        if self._encoder is None:
            import torch
            device = self._encoder_config['device']
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._encoder = DINOv3Encoder(
                device=device,
                batch_size=self._encoder_config['batch_size'],
            )
        return self._encoder

    def process(
        self,
        regions_gdf: gpd.GeoDataFrame,
        h3_resolution: int = 10,
    ) -> pd.DataFrame:
        """Process aerial imagery for given regions.

        Phase 1 - FETCH: Parallel download 224x224 images from PDOK.
        Phase 2 - ENCODE: GPU batch encoding with DINOv3.

        Args:
            regions_gdf: GeoDataFrame indexed by region_id with geometry
                (from SRAI H3Regionalizer, CRS EPSG:4326).
            h3_resolution: H3 resolution of the regions (for metadata).

        Returns:
            DataFrame with columns:
                region_id (str), dinov3_0..dinov3_1023 (float),
                h3_resolution (int).
        """
        hex_ids = regions_gdf.index.tolist()
        logger.info(
            f"Processing {len(hex_ids)} hexagons at res {h3_resolution}"
        )

        # Phase 1: Fetch images
        logger.info("Phase 1: Fetching images from PDOK...")
        self.tile_dir.mkdir(parents=True, exist_ok=True)
        image_paths = self.pdok_client.fetch_images_parallel(
            regions_gdf,
            cache_dir=self.tile_dir,
        )
        logger.info(
            f"Phase 1 complete: {len(image_paths)}/{len(hex_ids)} "
            f"images fetched"
        )

        if not image_paths:
            logger.warning("No images fetched, returning empty DataFrame")
            return self._empty_output(h3_resolution)

        # Phase 2: Encode images
        logger.info("Phase 2: Encoding images with DINOv3...")
        valid_hexes = list(image_paths.keys())
        valid_paths = [image_paths[h] for h in valid_hexes]
        embeddings = self.encoder.encode_images(valid_paths)
        logger.info(
            f"Phase 2 complete: {embeddings.shape[0]} images encoded "
            f"to {embeddings.shape[1]}D embeddings"
        )

        # Build output DataFrame
        cols = [f'dinov3_{i}' for i in range(embeddings.shape[1])]
        df = pd.DataFrame(embeddings, index=valid_hexes, columns=cols)
        df.index.name = 'region_id'
        df.index = df.index.astype(str)
        df['h3_resolution'] = h3_resolution

        return df.reset_index()

    def run_pipeline(
        self,
        study_area: str,
        h3_resolution: int = 10,
        output_dir: Optional[str] = None,
    ) -> str:
        """Full pipeline: load regions -> fetch -> encode -> save.

        Args:
            study_area: Name of the study area.
            h3_resolution: Target H3 resolution.
            output_dir: Override output directory. Defaults to
                StudyAreaPaths.stage1('aerial_imagery').

        Returns:
            Path to saved parquet file.
        """
        from utils import StudyAreaPaths

        paths = StudyAreaPaths(study_area)

        # Load pre-computed regions
        regions_path = paths.region_file(h3_resolution)
        if not regions_path.exists():
            raise FileNotFoundError(
                f"Regions file not found: {regions_path}. "
                f"Run H3Regionalizer first."
            )
        regions_gdf = gpd.read_parquet(regions_path)
        logger.info(
            f"Loaded {len(regions_gdf)} regions from {regions_path}"
        )

        # Update tile directory for this study area
        self.tile_dir = paths.intermediate('aerial_imagery') / 'tiles'

        # Process
        embeddings_df = self.process(regions_gdf, h3_resolution)

        # Save
        if output_dir is None:
            output_dir = str(paths.stage1('aerial_imagery'))

        output_path = self.save_embeddings(embeddings_df, output_dir)
        logger.info(f"Saved embeddings to {output_path}")

        return output_path

    def _empty_output(self, h3_resolution: int) -> pd.DataFrame:
        """Return an empty DataFrame matching the output contract."""
        cols = ['region_id'] + [f'dinov3_{i}' for i in range(1024)]
        cols.append('h3_resolution')
        return pd.DataFrame(columns=cols)
