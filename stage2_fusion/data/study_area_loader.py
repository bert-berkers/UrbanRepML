"""
Unified Study Area Data Loader
==============================

Single source of truth for loading data from study areas.
Handles regions, embeddings, and model artifacts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np
import torch
import json
from datetime import datetime

from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)


class StudyAreaLoader:
    """
    Unified loader for study area data.

    Provides consistent interface for loading regions, embeddings,
    and saving processed results [old 2024].
    """

    def __init__(self, study_area: str, base_path: str = "data/study_areas",
                 study_area_paths: Optional[StudyAreaPaths] = None):
        """
        Initialize loader for specific study area.

        Args:
            study_area: Name of study area (e.g., 'netherlands')
            base_path: Base path to study areas directory (ignored if study_area_paths provided)
            study_area_paths: Optional pre-constructed StudyAreaPaths instance
        """
        self.study_area = study_area

        # Use provided StudyAreaPaths or construct one
        if study_area_paths is not None:
            self._study_area_paths = study_area_paths
        else:
            self._study_area_paths = StudyAreaPaths(study_area)

        self.base_path = self._study_area_paths.root

        if not self.base_path.exists():
            raise ValueError(f"Study area path does not exist: {self.base_path}")

        # Define standard paths (delegating to StudyAreaPaths where possible)
        self.paths = {
            'boundaries': self._study_area_paths.boundaries(),
            'regions': self._study_area_paths.regions(),
            'models': self.base_path / 'models',
            'metadata': self.base_path / 'metadata'
        }

        # Available resolutions (will be detected)
        self.available_resolutions = self._detect_available_resolutions()

        logger.info(f"Initialized StudyAreaLoader for {study_area}")
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Available resolutions: {self.available_resolutions}")

    def _detect_available_resolutions(self) -> List[int]:
        """Detect available H3 resolutions."""
        resolutions = []

        if self.paths['regions'].exists():
            for file in self.paths['regions'].glob("*_res*.parquet"):
                # Extract resolution from filename
                try:
                    res = int(file.stem.split('_res')[-1])
                    resolutions.append(res)
                except:
                    continue

        return sorted(resolutions)

    def load_boundary(self) -> gpd.GeoDataFrame:
        """
        Load study area boundary.

        Returns:
            GeoDataFrame with study area boundary
        """
        boundary_file = self.paths['boundaries'] / f"{self.study_area}_boundary.geojson"

        if not boundary_file.exists():
            raise FileNotFoundError(f"Boundary file not found: {boundary_file}")

        return gpd.read_file(boundary_file)

    def load_regions(self, resolution: int,
                    with_geometry: bool = True) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Load H3 regions for specific resolution.

        Args:
            resolution: H3 resolution (5-10)
            with_geometry: Whether to include geometry column

        Returns:
            GeoDataFrame or DataFrame with H3 regions
        """
        region_file = self.paths['regions'] / f"{self.study_area}_res{resolution}.parquet"

        if not region_file.exists():
            raise FileNotFoundError(f"Region file not found for resolution {resolution}: {region_file}")

        if with_geometry:
            gdf = gpd.read_parquet(region_file)
            logger.info(f"Loaded {len(gdf):,} regions for resolution {resolution}")
            return gdf
        else:
            # Load without geometry for memory efficiency
            df = pd.read_parquet(region_file, columns=lambda x: x != 'geometry')
            logger.info(f"Loaded {len(df):,} regions for resolution {resolution} (no geometry)")
            return df

    def load_embeddings(self, modality: str, resolution: Optional[int] = None,
                       year: Optional[int] = None) -> pd.DataFrame:
        """
        Load embeddings for specific modality.

        Args:
            modality: Embedding type (alphaearth, poi, roads, gtfs, lattice_unet)
            resolution: H3 resolution (optional, will try to detect)
            year: Year for temporal data (optional)

        Returns:
            DataFrame with embeddings
        """
        embeddings_dir = self._study_area_paths.stage1(modality)

        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

        # Build filename pattern
        pattern = f"{self.study_area}"
        if resolution is not None:
            pattern += f"_res{resolution}"
        if year is not None:
            pattern += f"_{year}"

        # Find matching files
        files = list(embeddings_dir.glob(f"{pattern}*.parquet"))

        if not files:
            raise FileNotFoundError(f"No embeddings found for {modality} with pattern {pattern}")

        if len(files) > 1:
            logger.warning(f"Multiple files found, using first: {files[0]}")

        # Load embeddings
        df = pd.read_parquet(files[0])
        logger.info(f"Loaded {modality} embeddings: {len(df):,} rows, "
                   f"{len([c for c in df.columns if c.startswith('A') or c.startswith('emb')]):,} dimensions")

        return df

    def save_embeddings(self, embeddings: Union[pd.DataFrame, torch.Tensor, np.ndarray],
                       model_name: str, resolution: int,
                       metadata: Optional[Dict] = None):
        """
        Save processed embeddings.

        Args:
            embeddings: Embeddings to save (DataFrame, Tensor, or numpy array)
            model_name: Name of model that generated embeddings
            resolution: H3 resolution
            metadata: Optional metadata to save alongside
        """
        # Create output directory
        output_dir = self._study_area_paths.model_embeddings(model_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        if isinstance(embeddings, np.ndarray):
            # Create DataFrame with embedding columns
            n_dims = embeddings.shape[1] if len(embeddings.shape) > 1 else 1
            columns = [f'emb_{i}' for i in range(n_dims)]
            embeddings = pd.DataFrame(embeddings, columns=columns)

        # Save embeddings
        output_file = output_dir / f"res{resolution}_embeddings.parquet"
        embeddings.to_parquet(output_file)
        logger.info(f"Saved {len(embeddings):,} embeddings to {output_file}")

        # Save metadata if provided
        if metadata:
            metadata_file = output_dir / f"res{resolution}_metadata.json"
            metadata['timestamp'] = datetime.now().isoformat()
            metadata['shape'] = list(embeddings.shape)
            metadata['resolution'] = resolution

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_file}")

    def save_model(self, model: torch.nn.Module, model_name: str,
                  config: Optional[Dict] = None, metrics: Optional[Dict] = None):
        """
        Save trained model.

        Args:
            model: PyTorch model to save
            model_name: Name for saved model
            config: Model configuration
            metrics: Training metrics
        """
        model_dir = self.paths['models']
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = model_dir / f"{model_name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        logger.info(f"Saved model to {model_path}")

    def load_model(self, model_name: str) -> Dict:
        """
        Load saved model.

        Args:
            model_name: Name of model to load

        Returns:
            Dictionary with model state and metadata
        """
        model_path = self.paths['models'] / f"{model_name}.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        return checkpoint

    def get_multi_resolution_data(self, modality: str,
                                  resolutions: Optional[List[int]] = None) -> Dict[int, pd.DataFrame]:
        """
        Load data for multiple resolutions.

        Args:
            modality: Data modality
            resolutions: List of resolutions (None for all available)

        Returns:
            Dictionary mapping resolution to DataFrame
        """
        if resolutions is None:
            resolutions = self.available_resolutions

        data = {}
        for res in resolutions:
            try:
                if modality == 'regions':
                    data[res] = self.load_regions(res)
                else:
                    data[res] = self.load_embeddings(modality, resolution=res)
            except FileNotFoundError:
                logger.warning(f"Data not found for {modality} at resolution {res}")
                continue

        return data

    def create_hierarchical_mapping(self, parent_res: int, child_res: int) -> pd.DataFrame:
        """
        Create parent-child H3 mapping between resolutions.

        Args:
            parent_res: Parent resolution
            child_res: Child resolution (must be parent_res + 1)

        Returns:
            DataFrame with parent-child mappings
        """
        if child_res != parent_res + 1:
            raise ValueError("Child resolution must be exactly parent_res + 1")

        # Load child regions
        child_regions = self.load_regions(child_res, with_geometry=False)

        # Import h3 for parent calculation
        import h3

        # Create mapping
        mapping_data = []
        for child_hex in child_regions.index:
            try:
                parent_hex = h3.cell_to_parent(child_hex, parent_res)
                mapping_data.append({
                    'parent_hex': parent_hex,
                    'child_hex': child_hex,
                    'parent_res': parent_res,
                    'child_res': child_res
                })
            except:
                continue

        mapping_df = pd.DataFrame(mapping_data)
        logger.info(f"Created mapping: {len(mapping_df):,} parent-child relationships")

        return mapping_df

    def get_data_summary(self) -> Dict:
        """
        Get summary of available data.

        Returns:
            Dictionary with data availability summary
        """
        summary = {
            'study_area': self.study_area,
            'base_path': str(self.base_path),
            'available_resolutions': self.available_resolutions,
            'stage1_modalities': {}
        }

        # Check each modality
        for modality in ['alphaearth', 'poi', 'roads', 'gtfs', 'lattice_unet']:
            modality_dir = self._study_area_paths.stage1(modality)
            if modality_dir.exists():
                files = list(modality_dir.glob("*.parquet"))
                summary['stage1_modalities'][modality] = {
                    'exists': True,
                    'num_files': len(files),
                    'files': [f.name for f in files]
                }
            else:
                summary['stage1_modalities'][modality] = {'exists': False}

        # Check models
        if self.paths['models'].exists():
            models = list(self.paths['models'].glob("*.pt"))
            summary['models'] = [m.name for m in models]

        return summary

    def validate_data_consistency(self) -> bool:
        """
        Validate data consistency across resolutions.

        Returns:
            True if data is consistent
        """
        issues = []

        # Check if boundary exists
        if not (self.paths['boundaries'] / f"{self.study_area}_boundary.geojson").exists():
            issues.append("Missing boundary file")

        # Check region files
        for res in range(5, 11):
            region_file = self.paths['regions'] / f"{self.study_area}_res{res}.parquet"
            if not region_file.exists():
                issues.append(f"Missing region file for resolution {res}")

        if issues:
            logger.warning(f"Data consistency issues: {issues}")
            return False

        logger.info("Data consistency check passed")
        return True