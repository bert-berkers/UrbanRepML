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
        self.h3_index = None
        
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
        
        # Ensure h3_index column exists
        if 'h3_index' not in df.columns:
            if df.index.name == 'h3_index':
                df = df.reset_index()
            else:
                raise ValueError(f"No h3_index found in {name} data")
        
        # Set h3_index as index for alignment
        df = df.set_index('h3_index')
        
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
        
        # Get all H3 indices
        h3_sets = {name: set(df.index) for name, df in self.modalities.items()}
        
        if method == 'intersection':
            # Only keep H3 cells present in all stage1_modalities
            common_h3 = set.intersection(*h3_sets.values())
            logger.info(f"Common H3 cells (intersection): {len(common_h3)}")
            
        elif method == 'union':
            # Keep all H3 cells, fill missing with zeros
            common_h3 = set.union(*h3_sets.values())
            logger.info(f"Total H3 cells (union): {len(common_h3)}")
            
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        # Report coverage
        for name, h3_set in h3_sets.items():
            coverage = len(h3_set.intersection(common_h3)) / len(common_h3) * 100
            logger.info(f"  {name} coverage: {coverage:.1f}%")
        
        # Store common H3 index
        self.h3_index = sorted(list(common_h3))
        
        # Align all stage1_modalities to common index
        aligned = {}
        for name, df in self.modalities.items():
            if method == 'intersection':
                aligned[name] = df.loc[self.h3_index]
            else:  # union
                aligned[name] = df.reindex(self.h3_index, fill_value=0)
                
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
        
        # Add h3_index back as column
        output_df = self.aligned_data.reset_index()
        output_df.to_parquet(output_path)
        
        logger.info(f"Saved aligned data to {output_path}")
        logger.info(f"Shape: {output_df.shape}")
        logger.info(f"Memory usage: {output_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Save metadata
        metadata = {
            'stage1_modalities': list(self.modalities.keys()),
            'n_hexagons': len(output_df),
            'n_features': output_df.shape[1] - 1,  # Minus h3_index
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
                              output_dir: str = 'data/study_areas/default/stage2_fusion/multimodal') -> pd.DataFrame:
    """Convenience function to load and align multi-modal embeddings."""
    
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