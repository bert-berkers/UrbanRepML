#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Active Inference Processor for Del Norte County AlphaEarth Data

Processes AlphaEarth satellite embeddings using active inference principles,
calculating free energy, information gain, and spatial representations.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import h3
from pathlib import Path
import yaml
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import rasterio
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from urban_embedding.active_inference import (
    ActiveInferenceModule,
    SpatialMarkovBlanket,
    HierarchicalActiveInference,
    FreeEnergyComponents
)
from urban_embedding.information_gain import (
    SpatialInformationCalculator,
    SpatialInformationGain,
    InformationBottleneck,
    InformationMetrics
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DelNorteActiveInferenceProcessor:
    """
    Main processor for Del Norte County active inference analysis.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize processor with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        if config_path is None:
            config_path = "experiments/del_norte_active_inference/config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        
        # Setup directories
        self.experiment_dir = Path("experiments/del_norte_active_inference")
        self.data_dir = self.experiment_dir / "data"
        self.results_dir = self.experiment_dir / "results"
        self.logs_dir = self.experiment_dir / "logs"
        
        for dir_path in [self.data_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Track processing state
        self.processing_stats = {
            'hexagons_processed': 0,
            'total_free_energy': 0.0,
            'total_information_gain': 0.0,
            'gaps_detected': 0,
            'synthetic_generated': 0
        }
    
    def _init_components(self):
        """Initialize active inference and information theory components."""
        
        # Active inference module
        ai_config = self.config['active_inference']
        self.active_inference = ActiveInferenceModule(
            input_dim=self.config['alphaearth']['bands'],
            hidden_dims=self.config['representation_learning']['embedding']['hidden_dims'],
            latent_dim=self.config['representation_learning']['embedding']['latent_dim'],
            precision_init=ai_config['free_energy']['precision_init'],
            complexity_weight=ai_config['free_energy']['complexity_weight'],
            accuracy_weight=ai_config['free_energy']['accuracy_weight']
        )
        
        # Hierarchical active inference
        self.hierarchical_ai = HierarchicalActiveInference(
            resolutions=self.config['h3_processing']['resolutions'],
            primary_resolution=ai_config['hierarchy']['primary_level'],
            coupling_strength=ai_config['hierarchy']['parent_child_influence']
        )
        
        # Spatial Markov blankets
        self.markov_blanket = SpatialMarkovBlanket(
            h3_resolution=ai_config['hierarchy']['primary_level'],
            spatial_radius=ai_config['markov_blankets']['spatial_radius'],
            include_diagonal=ai_config['markov_blankets']['include_diagonal']
        )
        
        # Information calculators
        info_config = self.config['information_metrics']
        self.info_calculator = SpatialInformationCalculator(
            method=info_config['entropy']['method'],
            n_bins=info_config['entropy']['n_bins'],
            normalize=info_config['entropy']['normalize']
        )
        
        self.spatial_info = SpatialInformationGain(
            h3_resolution=ai_config['hierarchy']['primary_level'],
            temporal_window=ai_config['markov_blankets']['temporal_window']
        )
        
        # Information bottleneck
        self.info_bottleneck = InformationBottleneck(
            compression_dim=self.config['representation_learning']['embedding']['latent_dim'],
            beta=0.01  # From config if available
        )
        
        logger.info("Initialized all active inference components")
    
    def load_alphaearth_data(
        self,
        file_path: Optional[Path] = None,
        year: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load AlphaEarth data from file or create mock data for testing.
        
        Args:
            file_path: Path to AlphaEarth GeoTIFF
            year: Year of data (for mock data)
            
        Returns:
            Data array and metadata dictionary
        """
        if file_path and file_path.exists():
            logger.info(f"Loading AlphaEarth data from {file_path}")
            
            with rasterio.open(file_path) as src:
                data = src.read()  # Shape: (bands, height, width)
                metadata = {
                    'crs': src.crs.to_string(),
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'shape': data.shape
                }
                return data, metadata
        else:
            # Create mock data for testing
            logger.warning("Creating mock AlphaEarth data for testing")
            
            # Simulate 64-band satellite embeddings
            n_bands = self.config['alphaearth']['bands']
            height, width = 100, 100  # Small test size
            
            # Create spatially correlated mock data
            data = np.zeros((n_bands, height, width))
            for b in range(n_bands):
                # Generate smooth spatial patterns
                freq = np.random.uniform(0.05, 0.2)
                phase = np.random.uniform(0, 2*np.pi)
                
                x = np.linspace(0, 10, width)
                y = np.linspace(0, 10, height)
                X, Y = np.meshgrid(x, y)
                
                pattern = np.sin(freq * X + phase) * np.cos(freq * Y)
                noise = np.random.normal(0, 0.1, (height, width))
                data[b] = pattern + noise
            
            # Mock metadata
            metadata = {
                'crs': 'EPSG:4326',
                'bounds': self.config['region']['bounds'],
                'shape': data.shape,
                'mock': True
            }
            
            return data, metadata
    
    def process_to_h3(
        self,
        data: np.ndarray,
        metadata: Dict,
        resolution: int = 8
    ) -> pd.DataFrame:
        """
        Process AlphaEarth data to H3 hexagons with information metrics.
        
        Args:
            data: AlphaEarth data array (bands, height, width)
            metadata: Data metadata
            resolution: H3 resolution
            
        Returns:
            DataFrame with H3 hexagons and embeddings
        """
        logger.info(f"Processing to H3 resolution {resolution}")
        
        bands, height, width = data.shape
        bounds = metadata['bounds']
        
        # Create H3 hexagons for the region
        h3_indices = set()
        
        # Generate H3 indices covering the bounds
        lat_step = (bounds['north'] - bounds['south']) / height
        lon_step = (bounds['east'] - bounds['west']) / width
        
        for i in range(0, height, 10):  # Sample for efficiency
            for j in range(0, width, 10):
                lat = bounds['south'] + i * lat_step
                lon = bounds['west'] + j * lon_step
                h3_idx = h3.latlng_to_cell(lat, lon, resolution)
                h3_indices.add(h3_idx)
        
        logger.info(f"Generated {len(h3_indices)} H3 hexagons")
        
        # Process each hexagon
        rows = []
        for h3_idx in h3_indices:
            # Get hexagon center
            lat, lon = h3.cell_to_latlng(h3_idx)
            
            # Map to pixel coordinates (simplified)
            i = int((lat - bounds['south']) / lat_step)
            j = int((lon - bounds['west']) / lon_step)
            
            # Ensure within bounds
            i = max(0, min(height - 1, i))
            j = max(0, min(width - 1, j))
            
            # Extract embedding for this location
            embedding = data[:, i, j]
            
            # Calculate information metrics
            entropy = self.info_calculator.calculate_entropy(embedding.reshape(1, -1))
            
            row = {
                'h3': h3_idx,
                'lat': lat,
                'lon': lon,
                'entropy': entropy
            }
            
            # Add embedding dimensions
            for b in range(bands):
                row[f'A{b:02d}'] = embedding[b]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.set_index('h3', inplace=True)
        
        # Add geometry for visualization
        df['geometry'] = df.index.map(
            lambda x: Polygon(h3.cell_to_boundary(x))
        )
        
        self.processing_stats['hexagons_processed'] = len(df)
        logger.info(f"Processed {len(df)} hexagons with embeddings")
        
        return df
    
    def calculate_free_energy(
        self,
        embeddings: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate free energy for each hexagon.
        
        Args:
            embeddings: DataFrame with H3 embeddings
            
        Returns:
            DataFrame with free energy components added
        """
        logger.info("Calculating free energy for all hexagons")
        
        # Extract embedding columns
        embed_cols = [col for col in embeddings.columns if col.startswith('A')]
        X = torch.tensor(embeddings[embed_cols].values, dtype=torch.float32)
        
        # Process through active inference module
        X = X.to(self.active_inference.device)
        
        with torch.no_grad():
            x_recon, mu, logvar, free_energy = self.active_inference(X)
        
        # Add free energy components to dataframe
        embeddings['free_energy_total'] = free_energy.total
        embeddings['free_energy_complexity'] = free_energy.complexity
        embeddings['free_energy_accuracy'] = free_energy.accuracy
        embeddings['posterior_entropy'] = free_energy.entropy
        embeddings['expected_free_energy'] = free_energy.expected
        
        # Calculate statistics
        self.processing_stats['total_free_energy'] = free_energy.total
        
        logger.info(f"Free energy calculated: {free_energy.total:.4f}")
        
        return embeddings
    
    def calculate_information_metrics(
        self,
        embeddings: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate comprehensive information metrics.
        
        Args:
            embeddings: DataFrame with H3 embeddings
            
        Returns:
            DataFrame with information metrics added
        """
        logger.info("Calculating information theory metrics")
        
        # Convert to dictionary format for spatial calculations
        embed_cols = [col for col in embeddings.columns if col.startswith('A')]
        embeddings_dict = {
            idx: row[embed_cols].values
            for idx, row in embeddings.iterrows()
        }
        
        # Calculate hexagon information
        hex_info = self.spatial_info.calculate_hexagon_information(embeddings_dict)
        embeddings['information_content'] = embeddings.index.map(hex_info)
        
        # Calculate spatial mutual information
        mi_values = {}
        for h3_idx in embeddings.index:
            mi = self.spatial_info.calculate_spatial_mutual_information(
                embeddings_dict, h3_idx, neighbor_ring=1
            )
            mi_values[h3_idx] = mi
        
        embeddings['spatial_mutual_info'] = embeddings.index.map(mi_values)
        
        # Calculate Markov blanket conditional independence
        independence = {}
        for h3_idx in embeddings.index:
            indep = self.markov_blanket.calculate_conditional_independence(
                embeddings_dict, h3_idx
            )
            independence[h3_idx] = indep
        
        embeddings['conditional_independence'] = embeddings.index.map(independence)
        
        # Information gain (using entropy as baseline)
        baseline_entropy = embeddings['entropy'].mean()
        embeddings['information_gain'] = embeddings['entropy'] - baseline_entropy
        
        self.processing_stats['total_information_gain'] = embeddings['information_gain'].sum()
        
        logger.info(f"Information metrics calculated for {len(embeddings)} hexagons")
        
        return embeddings
    
    def detect_gaps(
        self,
        embeddings: pd.DataFrame,
        coverage_threshold: float = 0.7
    ) -> List[str]:
        """
        Detect spatial gaps using expected free energy.
        
        Args:
            embeddings: DataFrame with processed embeddings
            coverage_threshold: Minimum coverage threshold
            
        Returns:
            List of H3 indices identified as gaps
        """
        logger.info("Detecting spatial gaps using expected free energy")
        
        # Create coverage map
        all_possible = set()
        for h3_idx in embeddings.index:
            # Add neighbors to check coverage
            neighbors = set(h3.grid_disk(h3_idx, 2))
            all_possible.update(neighbors)
        
        covered = set(embeddings.index)
        gaps = all_possible - covered
        
        # Filter gaps by expected information gain
        coverage_map = {h3_idx: h3_idx in covered for h3_idx in all_possible}
        
        embeddings_dict = {
            idx: embeddings.loc[idx, [col for col in embeddings.columns if col.startswith('A')]].values
            for idx in embeddings.index
        }
        
        high_info_gaps = self.spatial_info.identify_high_information_gaps(
            embeddings_dict,
            coverage_map,
            threshold_percentile=90
        )
        
        self.processing_stats['gaps_detected'] = len(high_info_gaps)
        logger.info(f"Detected {len(high_info_gaps)} high-information gaps")
        
        return high_info_gaps
    
    def generate_synthetic(
        self,
        embeddings: pd.DataFrame,
        gap_indices: List[str]
    ) -> pd.DataFrame:
        """
        Generate synthetic data for gaps using active inference.
        
        Args:
            embeddings: Existing embeddings
            gap_indices: H3 indices of gaps
            
        Returns:
            DataFrame with synthetic embeddings
        """
        logger.info(f"Generating synthetic data for {len(gap_indices)} gaps")
        
        if not gap_indices:
            return pd.DataFrame()
        
        synthetic_rows = []
        embed_cols = [col for col in embeddings.columns if col.startswith('A')]
        
        for gap_idx in gap_indices[:100]:  # Limit for testing
            # Get neighbors
            neighbors = set(h3.grid_disk(gap_idx, 1))
            neighbors.discard(gap_idx)
            
            # Get neighbor embeddings
            neighbor_embeddings = []
            for neighbor in neighbors:
                if neighbor in embeddings.index:
                    neighbor_embeddings.append(
                        embeddings.loc[neighbor, embed_cols].values
                    )
            
            if neighbor_embeddings:
                # Use active inference to generate synthetic embedding
                neighbor_array = np.array(neighbor_embeddings, dtype=np.float32)
                neighbor_tensor = torch.tensor(
                    neighbor_array,
                    dtype=torch.float32
                ).to(self.active_inference.device)
                
                with torch.no_grad():
                    # Encode neighbors
                    mu, logvar = self.active_inference.encode(neighbor_tensor)
                    
                    # Average latent representation
                    mu_avg = mu.mean(dim=0, keepdim=True)
                    logvar_avg = logvar.mean(dim=0, keepdim=True)
                    
                    # Sample and decode
                    z = self.active_inference.reparameterize(mu_avg, logvar_avg)
                    synthetic = self.active_inference.decode(z)
                    
                    synthetic_np = synthetic.cpu().numpy().flatten()
                
                # Create row
                lat, lon = h3.cell_to_latlng(gap_idx)
                row = {
                    'h3': gap_idx,
                    'lat': lat,
                    'lon': lon,
                    'synthetic': True
                }
                
                for i, val in enumerate(synthetic_np[:len(embed_cols)]):
                    row[embed_cols[i]] = val
                
                synthetic_rows.append(row)
        
        synthetic_df = pd.DataFrame(synthetic_rows)
        if not synthetic_df.empty:
            synthetic_df.set_index('h3', inplace=True)
            
            # Add geometry
            synthetic_df['geometry'] = synthetic_df.index.map(
                lambda x: Polygon(h3.cell_to_boundary(x))
            )
        
        self.processing_stats['synthetic_generated'] = len(synthetic_df)
        logger.info(f"Generated {len(synthetic_df)} synthetic embeddings")
        
        return synthetic_df
    
    def save_results(
        self,
        embeddings: pd.DataFrame,
        synthetic: Optional[pd.DataFrame] = None,
        year: Optional[int] = None
    ):
        """
        Save processed results.
        
        Args:
            embeddings: Processed embeddings
            synthetic: Synthetic embeddings (optional)
            year: Year of data
        """
        year_str = f"_{year}" if year else ""
        
        # Save main embeddings (drop geometry for parquet compatibility)
        embeddings_save = embeddings.copy()
        if 'geometry' in embeddings_save.columns:
            embeddings_save = embeddings_save.drop('geometry', axis=1)
        
        output_path = self.results_dir / f"del_norte_embeddings{year_str}.parquet"
        embeddings_save.to_parquet(output_path)
        logger.info(f"Saved embeddings to {output_path}")
        
        # Save synthetic if available
        if synthetic is not None and not synthetic.empty:
            synthetic_save = synthetic.copy()
            if 'geometry' in synthetic_save.columns:
                synthetic_save = synthetic_save.drop('geometry', axis=1)
                
            synthetic_path = self.results_dir / f"del_norte_synthetic{year_str}.parquet"
            synthetic_save.to_parquet(synthetic_path)
            logger.info(f"Saved synthetic data to {synthetic_path}")
        
        # Save processing statistics
        stats_path = self.results_dir / f"processing_stats{year_str}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.processing_stats, f, indent=2)
        
        # Save information metrics summary
        metrics_summary = {
            'mean_entropy': float(embeddings['entropy'].mean()),
            'mean_free_energy': float(embeddings.get('free_energy_total', 0).mean()),
            'mean_information_gain': float(embeddings.get('information_gain', 0).mean()),
            'mean_spatial_mi': float(embeddings.get('spatial_mutual_info', 0).mean()),
            'processing_stats': self.processing_stats
        }
        
        metrics_path = self.results_dir / f"information_metrics{year_str}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        logger.info("Results saved successfully")
    
    def run(
        self,
        data_path: Optional[Path] = None,
        year: Optional[int] = 2023
    ):
        """
        Run complete active inference processing pipeline.
        
        Args:
            data_path: Path to AlphaEarth data (optional, will use mock if None)
            year: Year of data
        """
        logger.info("="*60)
        logger.info("Del Norte Active Inference Processing Pipeline")
        logger.info("="*60)
        
        # Load data
        data, metadata = self.load_alphaearth_data(data_path, year)
        
        # Process to H3
        embeddings = self.process_to_h3(
            data, 
            metadata,
            resolution=self.config['active_inference']['hierarchy']['primary_level']
        )
        
        # Calculate free energy
        embeddings = self.calculate_free_energy(embeddings)
        
        # Calculate information metrics
        embeddings = self.calculate_information_metrics(embeddings)
        
        # Detect gaps
        gaps = self.detect_gaps(embeddings)
        
        # Generate synthetic data for gaps
        synthetic = self.generate_synthetic(embeddings, gaps)
        
        # Save results
        self.save_results(embeddings, synthetic, year)
        
        logger.info("="*60)
        logger.info("Processing Complete!")
        logger.info(f"Hexagons processed: {self.processing_stats['hexagons_processed']}")
        logger.info(f"Total free energy: {self.processing_stats['total_free_energy']:.4f}")
        logger.info(f"Total information gain: {self.processing_stats['total_information_gain']:.4f}")
        logger.info(f"Gaps detected: {self.processing_stats['gaps_detected']}")
        logger.info(f"Synthetic generated: {self.processing_stats['synthetic_generated']}")
        logger.info("="*60)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process Del Norte AlphaEarth data with active inference'
    )
    parser.add_argument(
        '--config',
        default='experiments/del_norte_active_inference/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data',
        type=Path,
        help='Path to AlphaEarth GeoTIFF (optional, will use mock if not provided)'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2023,
        help='Year of data'
    )
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = DelNorteActiveInferenceProcessor(args.config)
    processor.run(args.data, args.year)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())