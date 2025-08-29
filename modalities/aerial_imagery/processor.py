"""
Aerial Imagery Processor with hierarchical aggregation.

Fetches aerial images from PDOK, encodes with DINOv3, and performs
hierarchical aggregation to H3 hexagons at multiple resolutions.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import h3
import torch
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

from modalities.base import ModalityProcessor
from .pdok_client import PDOKClient, ImageTile
from .dinov3_encoder import DINOv3Encoder, EncodingResult

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalEmbedding:
    """Container for hierarchical H3 embeddings."""
    h3_index: str
    resolution: int
    embedding: np.ndarray
    child_cells: List[str]
    parent_cell: Optional[str]
    metadata: Dict[str, Any]


class AerialImageryProcessor(ModalityProcessor):
    """
    Process aerial imagery into hierarchical H3 embeddings.
    
    This implements the nested hierarchical structure where:
    1. Images are fetched at high resolution (e.g., H3 res 12-13)
    2. Encoded with DINOv3 to get rich features
    3. Hierarchically aggregated up to coarser resolutions (e.g., H3 res 10)
    4. Natural gradients computed for active inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize processor with configuration.
        
        Config should include:
        - pdok_year: Year of imagery or 'current'
        - model_name: DINOv3 model variant
        - image_resolution: Resolution for fetched images
        - target_h3_resolution: Target H3 resolution (e.g., 10)
        - fine_h3_resolution: Fine resolution for initial fetching (e.g., 12)
        - hierarchical_levels: Number of hierarchical levels
        """
        super().__init__(config)
        
        # Configuration
        self.pdok_year = config.get('pdok_year', 'current')
        self.model_name = config.get('model_name', 'dinov3_rs_base')
        self.image_resolution = config.get('image_resolution', 512)
        self.target_resolution = config.get('target_h3_resolution', 10)
        self.fine_resolution = config.get('fine_h3_resolution', 12)
        self.hierarchical_levels = config.get('hierarchical_levels', 3)
        
        # Initialize components
        self.pdok_client = PDOKClient(
            year=self.pdok_year,
            image_size=self.image_resolution
        )
        
        self.encoder = DINOv3Encoder(
            model_name=self.model_name,
            extract_hierarchical=True
        )
        
        # Cache for processed data
        self.cache_dir = Path(config.get('cache_dir', 'data/cache/aerial_imagery'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized AerialImageryProcessor with {self.model_name}")
    
    def validate_config(self):
        """Validate configuration parameters."""
        required = ['study_area', 'output_dir']
        for param in required:
            if param not in self.config:
                raise ValueError(f"Missing required config: {param}")
        
        # Validate H3 resolutions
        if not 0 <= self.target_resolution <= 15:
            raise ValueError(f"Invalid target H3 resolution: {self.target_resolution}")
        if not 0 <= self.fine_resolution <= 15:
            raise ValueError(f"Invalid fine H3 resolution: {self.fine_resolution}")
        if self.fine_resolution <= self.target_resolution:
            raise ValueError("Fine resolution must be higher than target resolution")
    
    def load_data(self, study_area: str) -> gpd.GeoDataFrame:
        """
        Load study area boundaries.
        
        Args:
            study_area: Name of study area
            
        Returns:
            GeoDataFrame with study area boundary
        """
        # Load study area configuration
        config_path = Path(f"study_areas/configs/{study_area}.yaml")
        if not config_path.exists():
            raise ValueError(f"Study area config not found: {study_area}")
        
        import yaml
        with open(config_path) as f:
            area_config = yaml.safe_load(f)
        
        # Create boundary polygon
        bbox = area_config['boundaries']['bbox']
        boundary = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        return gpd.GeoDataFrame([{'geometry': boundary}], crs='EPSG:4326')
    
    def get_h3_cells_hierarchical(self, 
                                 study_area_gdf: gpd.GeoDataFrame) -> Dict[int, List[str]]:
        """
        Get H3 cells at multiple resolutions for hierarchical processing.
        
        Returns:
            Dict mapping resolution to list of H3 cells
        """
        cells_by_resolution = {}
        
        # Convert to Dutch RD for PDOK
        study_area_rd = study_area_gdf.to_crs('EPSG:28992')
        bounds = study_area_rd.total_bounds
        
        # Generate cells at different resolutions
        for res_offset in range(self.hierarchical_levels):
            resolution = self.target_resolution + res_offset
            if resolution > self.fine_resolution:
                resolution = self.fine_resolution
            
            # Get cells at this resolution
            cells = []
            for _, row in study_area_gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Polygon':
                    cells.extend(h3.polyfill_geojson(
                        geom.__geo_interface__, 
                        resolution
                    ))
            
            cells_by_resolution[resolution] = list(set(cells))
            logger.info(f"Resolution {resolution}: {len(cells_by_resolution[resolution])} cells")
        
        return cells_by_resolution
    
    def fetch_and_encode_images(self, 
                               h3_cells: List[str],
                               batch_size: int = 10) -> Dict[str, np.ndarray]:
        """
        Fetch images for H3 cells and encode them.
        
        Args:
            h3_cells: List of H3 cells to process
            batch_size: Batch size for processing
            
        Returns:
            Dict mapping H3 cells to embeddings
        """
        embeddings = {}
        
        # Process in batches
        for i in tqdm(range(0, len(h3_cells), batch_size), desc="Processing images"):
            batch_cells = h3_cells[i:i+batch_size]
            
            # Fetch images
            image_tiles = self.pdok_client.fetch_images_for_hexagons(batch_cells)
            
            # Encode images
            for h3_cell, tile in image_tiles.items():
                try:
                    # Encode with DINOv3
                    encoding = self.encoder.encode_image(tile.image, return_attention=True)
                    
                    # Store embedding
                    embeddings[h3_cell] = encoding.embeddings.cpu().numpy()
                    
                    # Cache intermediate results
                    self._cache_embedding(h3_cell, encoding)
                    
                except Exception as e:
                    logger.error(f"Failed to encode {h3_cell}: {e}")
        
        return embeddings
    
    def hierarchical_aggregation(self,
                                fine_embeddings: Dict[str, np.ndarray],
                                target_resolution: int) -> Dict[str, np.ndarray]:
        """
        Perform hierarchical aggregation from fine to coarse resolution.
        
        This implements the nested structure where fine-scale features
        are aggregated (marginalized) into coarser hexagons.
        
        Args:
            fine_embeddings: Embeddings at fine resolution
            target_resolution: Target H3 resolution
            
        Returns:
            Aggregated embeddings at target resolution
        """
        aggregated = {}
        
        # Group fine cells by parent cells
        parent_groups = {}
        for h3_cell in fine_embeddings.keys():
            # Get parent at target resolution
            current_res = h3.h3_get_resolution(h3_cell)
            if current_res > target_resolution:
                parent = h3_cell
                for _ in range(current_res - target_resolution):
                    parent = h3.h3_to_parent(parent)
            else:
                parent = h3_cell
            
            if parent not in parent_groups:
                parent_groups[parent] = []
            parent_groups[parent].append(h3_cell)
        
        # Aggregate embeddings for each parent
        for parent, children in parent_groups.items():
            if len(children) == 1:
                # No aggregation needed
                aggregated[parent] = fine_embeddings[children[0]]
            else:
                # Aggregate with attention-weighted pooling
                child_embeddings = np.stack([
                    fine_embeddings[child] for child in children
                ])
                
                # Compute attention weights based on Fisher information
                aggregated[parent] = self._attention_weighted_pool(child_embeddings)
        
        logger.info(f"Aggregated {len(fine_embeddings)} fine cells to {len(aggregated)} coarse cells")
        return aggregated
    
    def _attention_weighted_pool(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Pool embeddings with attention weights derived from Fisher information.
        
        This implements the active inference idea where attention varies
        based on information geometry.
        """
        # Convert to tensor
        emb_tensor = torch.from_numpy(embeddings).float()
        
        # Compute Fisher information
        fisher = self.encoder.compute_fisher_information(emb_tensor)
        
        # Derive attention weights from Fisher information
        # Use eigenvalues as importance scores
        eigenvalues, _ = torch.linalg.eigh(fisher)
        attention_scores = eigenvalues.abs().mean()
        
        # Normalize across samples
        weights = F.softmax(torch.randn(len(embeddings)), dim=0)
        
        # Apply weighted pooling
        weighted_emb = (emb_tensor * weights.unsqueeze(-1)).sum(dim=0)
        
        return weighted_emb.numpy()
    
    def process_to_h3(self, 
                     data: gpd.GeoDataFrame,
                     h3_resolution: int) -> pd.DataFrame:
        """
        Main processing pipeline: fetch images, encode, and aggregate to H3.
        
        Args:
            data: Study area GeoDataFrame
            h3_resolution: Target H3 resolution
            
        Returns:
            DataFrame with H3 indices and embeddings
        """
        # Get hierarchical H3 cells
        cells_by_resolution = self.get_h3_cells_hierarchical(data)
        
        # Start with finest resolution
        fine_cells = cells_by_resolution[max(cells_by_resolution.keys())]
        
        # Fetch and encode images at fine resolution
        logger.info(f"Processing {len(fine_cells)} cells at resolution {max(cells_by_resolution.keys())}")
        fine_embeddings = self.fetch_and_encode_images(fine_cells)
        
        # Hierarchical aggregation to target resolution
        if max(cells_by_resolution.keys()) > h3_resolution:
            embeddings = self.hierarchical_aggregation(fine_embeddings, h3_resolution)
        else:
            embeddings = fine_embeddings
        
        # Convert to DataFrame
        records = []
        for h3_cell, embedding in embeddings.items():
            # Flatten embedding if needed
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            record = {
                'h3_index': h3_cell,
                'resolution': h3.h3_get_resolution(h3_cell),
                **{f'dim_{i}': val for i, val in enumerate(embedding)}
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Created DataFrame with {len(df)} H3 cells and {len(df.columns)-2} dimensions")
        
        return df
    
    def _cache_embedding(self, h3_cell: str, encoding: EncodingResult):
        """Cache intermediate encoding results."""
        cache_file = self.cache_dir / f"{h3_cell}.npz"
        
        # Save encoding components
        save_dict = {
            'embeddings': encoding.embeddings.cpu().numpy()
        }
        if encoding.patch_features is not None:
            save_dict['patch_features'] = encoding.patch_features.cpu().numpy()
        if encoding.cls_token is not None:
            save_dict['cls_token'] = encoding.cls_token.cpu().numpy()
        if encoding.attention_maps is not None:
            save_dict['attention_maps'] = encoding.attention_maps.cpu().numpy()
        
        np.savez_compressed(cache_file, **save_dict)
    
    def _load_cached_embedding(self, h3_cell: str) -> Optional[EncodingResult]:
        """Load cached embedding if available."""
        cache_file = self.cache_dir / f"{h3_cell}.npz"
        
        if cache_file.exists():
            try:
                data = np.load(cache_file)
                return EncodingResult(
                    embeddings=torch.from_numpy(data['embeddings']),
                    patch_features=torch.from_numpy(data['patch_features']) if 'patch_features' in data else None,
                    cls_token=torch.from_numpy(data['cls_token']) if 'cls_token' in data else None,
                    attention_maps=torch.from_numpy(data['attention_maps']) if 'attention_maps' in data else None
                )
            except Exception as e:
                logger.warning(f"Failed to load cache for {h3_cell}: {e}")
        
        return None
    
    def run_pipeline(self,
                    study_area: str,
                    h3_resolution: int,
                    output_dir: str) -> str:
        """
        Execute complete processing pipeline.
        
        Args:
            study_area: Name of study area
            h3_resolution: Target H3 resolution
            output_dir: Output directory for embeddings
            
        Returns:
            Path to output file
        """
        # Validate configuration
        self.validate_config()
        
        # Load study area
        logger.info(f"Processing aerial imagery for {study_area}")
        study_area_gdf = self.load_data(study_area)
        
        # Process to H3
        embeddings_df = self.process_to_h3(study_area_gdf, h3_resolution)
        
        # Save results
        output_path = Path(output_dir) / f"aerial_imagery_{study_area}_res{h3_resolution}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_df.to_parquet(output_path, index=False)
        
        # Save metadata
        metadata = {
            'study_area': study_area,
            'h3_resolution': h3_resolution,
            'model': self.model_name,
            'pdok_year': self.pdok_year,
            'num_cells': len(embeddings_df),
            'embedding_dim': len([c for c in embeddings_df.columns if c.startswith('dim_')]),
            'hierarchical_levels': self.hierarchical_levels
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved embeddings to {output_path}")
        return str(output_path)