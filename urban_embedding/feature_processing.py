# feature_processing.py

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Optional, List, Tuple, Union, Any
import torch
import torch.nn.functional as F
import h3
from dataclasses import dataclass

import logging
import sys
# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('urban_embedding.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # This will use system's default encoding
    ]
)

# Add this line to get the logger for this module
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingStats:
    """Statistics for preprocessing auxiliary data validation"""
    shape: Tuple[int, int]
    original_range: Tuple[float, float]
    normalized_range: Tuple[float, float]
    l2_norm_mean: float
    zero_fraction: float
    explained_variance: float
    n_components: int

class UrbanFeatureProcessor:
    def __init__(
            self,
            variance_threshold: float = 0.95,
            min_components: Optional[Dict[str, int]] = None,
            max_components: Optional[int] = None,
            device: str = "cuda",
            cache_dir: Optional[Path] = None,
            preprocessed_dir: Optional[Path] = None,
            eps: float = 1e-8
    ):
        self.variance_threshold = self._validate_threshold(variance_threshold)
        self.min_components = min_components if min_components else {}
        self.max_components = max_components
        self.device = device
        self.cache_dir = cache_dir
        self.preprocessed_dir = preprocessed_dir
        self.eps = eps
        self.pca_models = {}
        self.preprocessing_stats = {}

    def _validate_threshold(self, threshold: float) -> float:
        if not (0 < threshold <= 1):
            raise ValueError("variance_threshold must be between 0 and 1")
        return threshold

    def _compute_stats(
            self,
            features: np.ndarray,
            normalized_features: np.ndarray,
            pca: PCA,
            n_components: int
    ) -> PreprocessingStats:
        """Compute comprehensive preprocessing auxiliary data statistics"""
        return PreprocessingStats(
            shape=features.shape,
            original_range=(float(np.min(features)), float(np.max(features))),
            normalized_range=(float(np.min(normalized_features)), float(np.max(normalized_features))),
            l2_norm_mean=float(np.mean(np.linalg.norm(normalized_features, axis=1))),
            zero_fraction=float(np.sum(features == 0) / features.size),
            explained_variance=float(np.sum(pca.explained_variance_ratio_[:n_components])),
            n_components=n_components
        )

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """L2 normalize features"""
        features_tensor = torch.from_numpy(features)
        normalized = F.normalize(features_tensor, p=2, dim=1, eps=self.eps)
        return normalized.numpy()

    def _determine_components(
            self,
            explained_variance_ratio: np.ndarray,
            modality: str,
            n_features: int
    ) -> int:
        """Determine number of components using variance threshold"""
        cumsum = np.cumsum(explained_variance_ratio)
        variance_components = np.searchsorted(cumsum, self.variance_threshold) + 1
        min_required = self.min_components.get(modality, 1)

        n_components = max(
            min_required,
            min(
                variance_components,
                n_features,
                self.max_components if self.max_components else n_features
            )
        )

        logger.info(f"Component selection for {modality}:")
        logger.info(f"- Variance-based: {variance_components}")
        logger.info(f"- Minimum required: {min_required}")
        logger.info(f"- Maximum allowed: {self.max_components or n_features}")
        logger.info(f"- Final selection: {n_components}")

        return n_components

    def fit_transform(
            self,
            features_dict: Dict[str, np.ndarray],
            city_name: str
    ) -> Dict[str, np.ndarray]:
        """Process features with L2 normalization and PCA"""
        transformed = {}

        for modality, features in features_dict.items():
            logger.info(f"\nProcessing {modality} features")

            # Input validation
            if not isinstance(features, np.ndarray):
                raise TypeError(f"Features for {modality} must be numpy array")
            if features.ndim != 2:
                raise ValueError(f"Features for {modality} must be 2-dimensional")

            # Handle missing values
            features = np.nan_to_num(features, nan=0.0)

            # L2 normalize
            features_normalized = self._normalize_features(features)

            # Fit initial PCA to determine components
            temp_pca = PCA()
            temp_pca.fit(features_normalized)

            # Determine components
            n_components = self._determine_components(
                temp_pca.explained_variance_ratio_,
                modality,
                features.shape[1]
            )

            # Final PCA
            pca = PCA(n_components=n_components)
            transformed_features = pca.fit_transform(features_normalized)

            # Normalize PCA output
            transformed_features = self._normalize_features(transformed_features)

            # Store results
            self.pca_models[modality] = pca
            transformed[modality] = transformed_features

            # Compute and store statistics
            stats = self._compute_stats(
                features,
                transformed_features,
                pca,
                n_components
            )
            self.preprocessing_stats[modality] = stats

            # Log results
            logger.info(f"Processing results for {modality}:")
            logger.info(f"Dimensions: {stats.shape[1]} -> {stats.n_components}")
            logger.info(f"Explained variance: {stats.explained_variance:.3%}")
            logger.info(f"L2 norm mean: {stats.l2_norm_mean:.3f}")
            logger.info(f"Output range: [{stats.normalized_range[0]:.3f}, {stats.normalized_range[1]:.3f}]")

        return transformed

    # Cross-scale mapping methods remain unchanged
    def _create_fine_to_coarse_mapping(
            self,
            fine_res: int,
            coarse_res: int,
            fine_hexes: List[str],
            city_name: str
    ) -> torch.sparse.Tensor:
        """Create mapping from fine to coarse resolution ensuring index correspondence."""
        logger.info(f"Creating new mapping {fine_res}->{coarse_res}")

        if not self.preprocessed_dir:
            raise ValueError("preprocessed_dir must be set to create mappings")

        # Load coarse regions to ensure correct ordering
        coarse_regions = pd.read_parquet(
            self.preprocessed_dir / city_name / f'regions_{coarse_res}_gdf.parquet'
        )
        coarse_hexes = list(coarse_regions.index)

        # Create mapping using indices from our ordered lists
        fine_to_idx = {h: i for i, h in enumerate(fine_hexes)}
        coarse_to_idx = {h: i for i, h in enumerate(coarse_hexes)}

        indices = []
        values = []

        for fine_hex in fine_hexes:
            coarse_hex = h3.cell_to_parent(fine_hex, coarse_res)
            if coarse_hex in coarse_to_idx:
                fine_idx = fine_to_idx[fine_hex]
                coarse_idx = coarse_to_idx[coarse_hex]
                indices.append([fine_idx, coarse_idx])
                values.append(1.0)

        if not indices:
            raise ValueError(f"No valid mappings found between resolutions {fine_res} and {coarse_res}")

        indices = torch.tensor(indices, device=self.device, dtype=torch.long).t()
        values = torch.tensor(values, device=self.device, dtype=torch.float32)
        mapping_size = (len(fine_hexes), len(coarse_hexes))

        # Create sparse tensor
        mapping = torch.sparse_coo_tensor(
            indices,
            values,
            size=mapping_size,
            device=self.device
        ).coalesce()

        # Normalize rows
        row_sums = torch.sparse.sum(mapping, dim=1).to_dense()
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        normalized_values = mapping.values() / row_sums[mapping.indices()[0]]

        mapping = torch.sparse_coo_tensor(
            mapping.indices(),
            normalized_values,
            size=mapping_size,
            device=self.device
        ).coalesce()

        logger.info(f"Created mapping with shape {mapping_size}")
        return mapping

    def load_cross_scale_mappings(
            self,
            city_name: str,
            resolutions: List[int]
    ) -> Dict[Tuple[int, int], torch.sparse.Tensor]:
        """Load or create fine->coarse mappings."""
        logger.info("Loading cross-scale mappings...")
        mappings = {}

        if not self.preprocessed_dir:
            raise ValueError("preprocessed_dir must be set to load mappings")

        city_dir = self.preprocessed_dir / city_name
        resolutions = sorted(resolutions, reverse=True)

        for i in range(len(resolutions) - 1):
            res_fine = resolutions[i]
            res_coarse = resolutions[i + 1]

            mapping_file = city_dir / f'mapping_{res_fine}_to_{res_coarse}.pt'

            try:
                if mapping_file.exists():
                    logger.info(f"Loading existing mapping {res_fine}->{res_coarse}")
                    mapping = torch.load(mapping_file, map_location=self.device)
                else:
                    logger.info(f"Creating new mapping {res_fine}->{res_coarse}")
                    regions_file = city_dir / f'regions_{res_fine}_gdf.parquet'
                    if not regions_file.exists():
                        raise FileNotFoundError(f"Regions file not found: {regions_file}")

                    fine_regions = pd.read_parquet(regions_file)
                    mapping = self._create_fine_to_coarse_mapping(
                        res_fine,
                        res_coarse,
                        list(fine_regions.index),
                        city_name
                    )
                    torch.save(mapping, mapping_file)
                    logger.info(f"Saved mapping to {mapping_file}")

                mappings[(res_fine, res_coarse)] = mapping
                logger.info(f"Created mapping tensor of size {mapping.size()}")

            except Exception as e:
                logger.error(f"Error processing mapping {res_fine}->{res_coarse}: {str(e)}")
                raise

        return mappings

    @property
    def feature_dims(self) -> Dict[str, int]:
        """Get number of components per modality"""
        return {name: pca.n_components_ for name, pca in self.pca_models.items()}