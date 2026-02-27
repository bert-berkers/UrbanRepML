"""
Multi-modal data loader for urban embeddings.
Handles loading and alignment of different modality embeddings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)


class MultiModalLoader:
    """Load and align multi-modal urban embeddings."""
    
    def __init__(self, config_path: str = None, config: Dict = None):
        """Initialize with configuration file or dictionary."""
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or {}
            
        self.modalities = {}
        self.aligned_data = None
        self.region_id = None
        
    def load_modality(self, name: str, config: Dict) -> pd.DataFrame:
        """Load a single modality's embeddings."""
        if not config.get('enabled', True):
            logger.info(f"Skipping disabled modality: {name}")
            return None
            
        source_path = Path(config['source'])
        if not source_path.exists():
            logger.warning(f"Source file not found for {name}: {source_path}")
            return None
            
        logger.info(f"Loading {name} from {source_path}")
        df = pd.read_parquet(source_path)
        
        # Ensure region_id column exists
        if 'region_id' not in df.columns:
            if df.index.name == 'region_id':
                df = df.reset_index()
            else:
                raise ValueError(f"No region_id found in {name} data")

        # Set region_id as index for alignment
        df = df.set_index('region_id')
        
        # Remove metadata columns if present
        metadata_cols = ['h3_resolution', 'geometry']
        df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
        
        logger.info(f"Loaded {name}: {len(df)} hexagons, {df.shape[1]} features")
        
        # Apply preprocessing_auxiliary_data if specified
        if 'preprocessing_auxiliary_data' in config:
            df = self._preprocess_modality(df, config['preprocessing_auxiliary_data'], name)
            
        return df
    
    def _preprocess_modality(self, df: pd.DataFrame, preprocess_config: Dict, 
                            name: str) -> pd.DataFrame:
        """Apply preprocessing_auxiliary_data to a modality."""
        # Handle missing values
        if df.isnull().any().any():
            logger.warning(f"{name} has missing values, filling with 0")
            df = df.fillna(0)
        
        # Normalization
        if preprocess_config.get('normalize', False):
            method = preprocess_config.get('normalization_method', 'standard')
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            logger.info(f"Applying {method} normalization to {name}")
            df_normalized = pd.DataFrame(
                scaler.fit_transform(df),
                index=df.index,
                columns=df.columns
            )
            df = df_normalized
        
        # PCA reduction if specified
        if 'pca_components' in preprocess_config:
            n_components = preprocess_config['pca_components']
            if n_components < df.shape[1]:
                from sklearn.decomposition import PCA
                
                logger.info(f"Applying PCA to {name}: {df.shape[1]} -> {n_components} dimensions")
                pca = PCA(n_components=n_components)
                df_pca = pd.DataFrame(
                    pca.fit_transform(df),
                    index=df.index,
                    columns=[f'{name}_pca_{i}' for i in range(n_components)]
                )
                
                logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
                df = df_pca
        
        return df
    
    def load_all_modalities(self) -> Dict[str, pd.DataFrame]:
        """Load all enabled stage1_modalities from configuration."""
        logger.info("Loading all stage1_modalities...")
        
        modality_configs = self.config.get('stage1_modalities', {})
        
        for name, mod_config in modality_configs.items():
            df = self.load_modality(name, mod_config)
            if df is not None:
                self.modalities[name] = df
                
        logger.info(f"Loaded {len(self.modalities)} stage1_modalities: {list(self.modalities.keys())}")
        return self.modalities
    
    def align_modalities(self, method: str = 'intersection') -> pd.DataFrame:
        """Align stage1_modalities to common H3 cells."""
        if not self.modalities:
            raise ValueError("No stage1_modalities loaded")
            
        logger.info(f"Aligning {len(self.modalities)} stage1_modalities using {method} method")
        
        # Get all region_id sets
        region_id_sets = {name: set(df.index) for name, df in self.modalities.items()}

        if method == 'intersection':
            # Only keep region_ids present in all stage1_modalities
            common_region_ids = set.intersection(*region_id_sets.values())
            logger.info(f"Common H3 cells (intersection): {len(common_region_ids)}")

        elif method == 'union':
            # Keep all region_ids, fill missing with zeros
            common_region_ids = set.union(*region_id_sets.values())
            logger.info(f"Total H3 cells (union): {len(common_region_ids)}")
            
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        # Report coverage
        for name, region_id_set in region_id_sets.items():
            coverage = len(region_id_set.intersection(common_region_ids)) / len(common_region_ids) * 100
            logger.info(f"  {name} coverage: {coverage:.1f}%")

        # Store common region_id index
        self.region_id = sorted(list(common_region_ids))

        # Align all stage1_modalities to common index
        aligned = {}
        for name, df in self.modalities.items():
            if method == 'intersection':
                aligned[name] = df.loc[self.region_id]
            else:  # union
                aligned[name] = df.reindex(self.region_id, fill_value=0)
                
        self.aligned_modalities = aligned
        return aligned
    
    def fuse_modalities(self, method: str = 'concatenate', 
                       weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Fuse aligned stage1_modalities into single feature matrix."""
        if not hasattr(self, 'aligned_modalities'):
            self.align_modalities()
            
        fusion_method = self.config.get('feature_processing', {}).get('fusion', {}).get('method', method)
        logger.info(f"Fusing stage1_modalities using {fusion_method} method")
        
        if fusion_method == 'concatenate':
            # Simple concatenation of features
            dfs_to_concat = []
            
            for name, df in self.aligned_modalities.items():
                # Add modality prefix to column names
                df_prefixed = df.add_prefix(f'{name}_')
                dfs_to_concat.append(df_prefixed)
                
            self.aligned_data = pd.concat(dfs_to_concat, axis=1)
            logger.info(f"Fused features shape: {self.aligned_data.shape}")
            
        elif fusion_method == 'weighted_average':
            # Weighted average (requires same dimensionality)
            if weights is None:
                weights = {name: self.config['stage1_modalities'][name].get('weight', 1.0)
                          for name in self.aligned_modalities}
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            logger.info(f"Using weights: {weights}")
            
            # This requires same feature dimensions - would need additional logic
            raise NotImplementedError("Weighted average fusion requires same dimensionality")
            
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
            
        return self.aligned_data
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for analysis."""
        if self.aligned_data is None:
            raise ValueError("No aligned data available")
            
        feature_groups = {}
        
        for modality in self.modalities.keys():
            modality_features = [col for col in self.aligned_data.columns 
                               if col.startswith(f'{modality}_')]
            feature_groups[modality] = modality_features
            
        return feature_groups
    
    def save_aligned_data(self, output_path: str):
        """Save aligned multi-modal data."""
        if self.aligned_data is None:
            raise ValueError("No aligned data to save")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add region_id back as column
        output_df = self.aligned_data.reset_index()
        output_df.to_parquet(output_path)
        
        logger.info(f"Saved aligned data to {output_path}")
        logger.info(f"Shape: {output_df.shape}")
        logger.info(f"Memory usage: {output_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Save metadata
        metadata = {
            'stage1_modalities': list(self.modalities.keys()),
            'n_hexagons': len(output_df),
            'n_features': output_df.shape[1] - 1,  # Minus region_id
            'feature_groups': self.get_feature_groups(),
            'alignment_method': getattr(self, 'alignment_method', 'intersection')
        }
        
        import json
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved metadata to {metadata_path}")
        
        return output_path


def load_multimodal_embeddings(config_path: str,
                              alignment: str = 'intersection',
                              save_output: bool = True,
                              output_dir: str = None,
                              study_area: str = 'netherlands') -> pd.DataFrame:
    """Convenience function to load and align multi-modal embeddings."""
    if output_dir is None:
        output_dir = str(StudyAreaPaths(study_area).stage2("multimodal"))

    loader = MultiModalLoader(config_path=config_path)
    
    # Load all stage1_modalities
    loader.load_all_modalities()
    
    # Align to common H3 cells
    loader.align_modalities(method=alignment)
    
    # Fuse stage1_modalities
    fused_data = loader.fuse_modalities()
    
    # Save if requested
    if save_output:
        output_path = Path(output_dir) / 'netherlands_multimodal_res10.parquet'
        loader.save_aligned_data(output_path)
        
    return fused_data


if __name__ == "__main__":
    # Test loading
    config_path = "configs/netherlands_pipeline.yaml"
    
    if Path(config_path).exists():
        logger.info("Testing multi-modal loader...")
        data = load_multimodal_embeddings(
            config_path,
            alignment='intersection',
            save_output=True
        )
        
        print(f"\nLoaded multi-modal data:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {list(data.columns[:10])}...")
        print(f"Memory: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")