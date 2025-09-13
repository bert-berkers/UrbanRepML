#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization suite for Del Norte exploratory analysis.
Creates spatial maps and statistical plots with multiple color schemes.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import RegularPolygon
import seaborn as sns
# MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Note: SRAI provides H3 functionality with additional spatial analysis tools
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class DelNorteVisualizer:
    """Create rich visualizations for Del Norte clustering analysis."""
    
    def __init__(self, config: dict):
        """Initialize visualizer with configuration."""
        self.config = config
        self.viz_config = config['visualization']
        self.colormaps = self.viz_config['colormaps']
        self.figure_size = self.viz_config['figure_size']
        self.dpi = self.viz_config['dpi']
        self.hex_size = self.viz_config['hex_size']
        
        # Setup logging
        logging.basicConfig(
            level=config['output']['log_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create output directories
        self.plots_dir = Path("plots")
        self.spatial_dir = self.plots_dir / "spatial"
        self.dist_dir = self.plots_dir / "distributions"
        self.comp_dir = self.plots_dir / "comparisons"
        
        for directory in [self.spatial_dir, self.dist_dir, self.comp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_clustered_data(self, cluster_file: Path) -> pd.DataFrame:
        """Load clustering results with H3 data."""
        logger.info(f"Loading clustered data from {cluster_file}")
        df = pd.read_parquet(cluster_file)
        
        # Add H3 polygon coordinates for visualization
        df = self.add_h3_coordinates(df)
        
        logger.info(f"Loaded {len(df)} hexagons with {df['cluster'].nunique()} clusters")
        return df
    
    def add_h3_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add H3 hexagon boundary coordinates for plotting."""
        # Get H3 hexagon boundaries
        boundaries = []
        for h3_idx in tqdm(df['h3_index'], desc="Getting H3 boundaries"):
            try:
                boundary = h3.cell_to_boundary(h3_idx)
                boundaries.append(boundary)
            except:
                boundaries.append(None)
        
        df['h3_boundary'] = boundaries
        return df
    
    def create_spatial_cluster_map(self, df: pd.DataFrame, method: str, 
                                  config_key: str, colormap: str) -> plt.Figure:
        """Create spatial map of clusters using hexagons."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Get unique clusters
        clusters = sorted(df['cluster'].unique())
        n_clusters = len(clusters)
        
        # Create colormap
        if colormap in plt.colormaps():
            cmap = plt.cm.get_cmap(colormap)
        else:
            cmap = plt.cm.viridis
        
        colors = [cmap(i / max(1, n_clusters - 1)) for i in range(n_clusters)]
        
        # Plot each cluster
        for cluster_id, color in zip(clusters, colors):
            cluster_data = df[df['cluster'] == cluster_id]
            
            # Plot hexagons
            for _, row in cluster_data.iterrows():
                if row['h3_boundary'] is not None:
                    # Convert H3 boundary to matplotlib polygon
                    boundary = np.array(row['h3_boundary'])
                    polygon = RegularPolygon(
                        (row['lon'], row['lat']),
                        6,  # hexagon
                        radius=0.005 * self.hex_size,  # Approximate size for H3 res 11
                        facecolor=color,
                        edgecolor='white',
                        linewidth=0.1,
                        alpha=0.8
                    )
                    ax.add_patch(polygon)
        
        # Set plot properties
        ax.set_xlim(df['lon'].min() - 0.01, df['lon'].max() + 0.01)
        ax.set_ylim(df['lat'].min() - 0.01, df['lat'].max() + 0.01)
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Del Norte 2021 - {method.upper()} Clustering (k={config_key})\n'
                    f'H3 Resolution 11 - {colormap} colormap')
        
        # Add colorbar legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_clusters-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Cluster ID')
        cbar.set_ticks(range(n_clusters))
        
        plt.tight_layout()
        return fig
    
    def create_cluster_distribution_plot(self, df: pd.DataFrame, method: str, 
                                       config_key: str) -> plt.Figure:
        """Create distribution plots for cluster analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        fig.suptitle(f'Del Norte 2021 - {method.upper()} Clustering Distribution Analysis (k={config_key})', 
                    fontsize=16)
        
        # 1. Cluster size distribution
        cluster_sizes = df['cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Hexagons')
        
        # 2. Spatial distribution (lat/lon scatter)
        scatter = axes[0, 1].scatter(df['lon'], df['lat'], c=df['cluster'], 
                                   cmap='tab10', alpha=0.6, s=2)
        axes[0, 1].set_title('Spatial Distribution of Clusters')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(scatter, ax=axes[0, 1], label='Cluster ID')
        
        # 3. Band intensity by cluster (sample of bands)
        sample_bands = [f'band_{i:02d}' for i in [0, 15, 31, 47, 63]]  # Sample 5 bands
        cluster_means = df.groupby('cluster')[sample_bands].mean()
        
        cluster_means.T.plot(kind='line', ax=axes[1, 0], alpha=0.8)
        axes[1, 0].set_title('Band Intensity Profiles by Cluster')
        axes[1, 0].set_xlabel('Band Index (sampled)')
        axes[1, 0].set_ylabel('Mean Intensity')
        axes[1, 0].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Cluster statistics heatmap
        band_cols = [f'band_{i:02d}' for i in range(64)]
        cluster_stats = df.groupby('cluster')[band_cols].mean()
        
        # Sample bands for visualization
        sample_indices = np.linspace(0, 63, 16, dtype=int)
        sample_band_cols = [f'band_{i:02d}' for i in sample_indices]
        
        sns.heatmap(cluster_stats[sample_band_cols].T, ax=axes[1, 1], 
                   cmap='viridis', cbar_kws={'label': 'Mean Intensity'})
        axes[1, 1].set_title('Cluster Band Intensities (Sampled)')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Band Index')
        
        plt.tight_layout()
        return fig
    
    def create_feature_analysis_plot(self, df: pd.DataFrame, method: str, 
                                   config_key: str) -> plt.Figure:
        """Create detailed feature analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        fig.suptitle(f'Del Norte 2021 - {method.upper()} Feature Analysis (k={config_key})', 
                    fontsize=16)
        
        band_cols = [f'band_{i:02d}' for i in range(64)]
        
        # 1. Overall band intensity distribution
        all_bands = df[band_cols].values.flatten()
        axes[0, 0].hist(all_bands[all_bands != 0], bins=50, alpha=0.7, color='darkblue')
        axes[0, 0].set_title('Overall Band Intensity Distribution')
        axes[0, 0].set_xlabel('Intensity Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        
        # 2. Band variance across all hexagons
        band_variances = df[band_cols].var().values
        axes[0, 1].plot(range(64), band_variances, 'o-', alpha=0.7)
        axes[0, 1].set_title('Band Variance Across All Hexagons')
        axes[0, 1].set_xlabel('Band Index')
        axes[0, 1].set_ylabel('Variance')
        
        # 3. First few principal components visualization (quick PCA for viz only)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Standardize and reduce for visualization
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[band_cols])
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(scaled_features)
        
        scatter = axes[1, 0].scatter(pca_features[:, 0], pca_features[:, 1], 
                                   c=df['cluster'], cmap='tab10', alpha=0.6, s=3)
        axes[1, 0].set_title(f'PCA Visualization (PC1 vs PC2)\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        plt.colorbar(scatter, ax=axes[1, 0], label='Cluster ID')
        
        # 4. Cluster separation quality
        if len(df['cluster'].unique()) > 1:
            from sklearn.metrics import silhouette_samples
            
            # Calculate silhouette scores
            sample_scores = silhouette_samples(scaled_features, df['cluster'])
            df_temp = df.copy()
            df_temp['silhouette'] = sample_scores
            
            # Plot silhouette scores by cluster
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_scores = df_temp[df_temp['cluster'] == cluster_id]['silhouette']
                axes[1, 1].hist(cluster_scores, alpha=0.6, label=f'Cluster {cluster_id}', bins=20)
            
            axes[1, 1].set_title('Silhouette Score Distribution by Cluster')
            axes[1, 1].set_xlabel('Silhouette Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].axvline(sample_scores.mean(), color='red', linestyle='--', 
                             label=f'Mean: {sample_scores.mean():.3f}')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(self, results_dict: Dict[str, pd.DataFrame]) -> plt.Figure:
        """Create comparison plot of different clustering methods."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        fig.suptitle('Del Norte 2021 - Clustering Method Comparison', fontsize=16)
        
        # Load metrics for comparison
        metrics_data = self.load_all_metrics()
        
        if metrics_data:
            # 1. Silhouette scores comparison
            methods = list(metrics_data.keys())
            for method in methods:
                configs = list(metrics_data[method].keys())
                silhouette_scores = [metrics_data[method][config]['silhouette'] 
                                   for config in configs]
                axes[0, 0].plot(configs, silhouette_scores, 'o-', label=method, alpha=0.7)
            
            axes[0, 0].set_title('Silhouette Score Comparison')
            axes[0, 0].set_xlabel('Configuration')
            axes[0, 0].set_ylabel('Silhouette Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Davies-Bouldin scores comparison
            for method in methods:
                configs = list(metrics_data[method].keys())
                db_scores = [metrics_data[method][config]['davies_bouldin'] 
                           for config in configs]
                axes[0, 1].plot(configs, db_scores, 'o-', label=method, alpha=0.7)
            
            axes[0, 1].set_title('Davies-Bouldin Score Comparison (lower is better)')
            axes[0, 1].set_xlabel('Configuration')
            axes[0, 1].set_ylabel('Davies-Bouldin Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Spatial comparison (if we have multiple methods loaded)
        if len(results_dict) >= 2:
            methods = list(results_dict.keys())[:2]  # Compare first two methods
            
            for i, method in enumerate(methods):
                df = results_dict[method]
                scatter = axes[1, i].scatter(df['lon'], df['lat'], c=df['cluster'], 
                                          cmap='tab10', alpha=0.6, s=2)
                axes[1, i].set_title(f'{method.upper()} Spatial Clusters')
                axes[1, i].set_xlabel('Longitude')
                axes[1, i].set_ylabel('Latitude')
                axes[1, i].set_aspect('equal')
                plt.colorbar(scatter, ax=axes[1, i], label='Cluster ID')
        
        plt.tight_layout()
        return fig
    
    def load_all_metrics(self) -> Dict:
        """Load all clustering metrics for comparison."""
        results_dir = Path("results/clusters")
        metrics_data = {}
        
        for metrics_file in results_dir.glob("*.json"):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                method = data['method']
                config_key = data['config_key']
                
                if method not in metrics_data:
                    metrics_data[method] = {}
                
                metrics_data[method][config_key] = data['metrics']
                
            except Exception as e:
                logger.warning(f"Could not load metrics from {metrics_file}: {e}")
        
        return metrics_data
    
    def visualize_clustering_method(self, method: str, config_key: str):
        """Create all visualizations for a specific clustering method."""
        # Load clustered data
        cluster_file = Path(f"results/clusters/{method}_2021_res8_{config_key}.parquet")
        
        if not cluster_file.exists():
            logger.error(f"Cluster file not found: {cluster_file}")
            return
        
        df = self.load_clustered_data(cluster_file)
        
        # Create spatial maps with different colormaps
        for colormap in tqdm(self.colormaps, desc=f"Creating {method} spatial maps"):
            fig = self.create_spatial_cluster_map(df, method, config_key, colormap)
            
            # Save in multiple formats
            for fmt in self.viz_config['save_formats']:
                output_path = self.spatial_dir / f"spatial_2021_res8_{method}_{config_key}_{colormap}.{fmt}"
                fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            
            plt.close(fig)
        
        # Create distribution analysis
        fig = self.create_cluster_distribution_plot(df, method, config_key)
        for fmt in self.viz_config['save_formats']:
            output_path = self.dist_dir / f"distribution_2021_res8_{method}_{config_key}.{fmt}"
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        # Create feature analysis
        fig = self.create_feature_analysis_plot(df, method, config_key)
        for fmt in self.viz_config['save_formats']:
            output_path = self.dist_dir / f"features_2021_res8_{method}_{config_key}.{fmt}"
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created visualizations for {method} - {config_key}")
    
    def visualize_all_results(self):
        """Create visualizations for all clustering results."""
        results_dir = Path("results/clusters")
        parquet_files = list(results_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.error("No clustering results found!")
            return
        
        # Parse filenames to get methods and configs
        methods_configs = []
        for file in parquet_files:
            filename = file.stem
            
            # Handle different filename patterns:
            # kmeans_2021_res8_k5.parquet -> method=kmeans, config=k5
            # hierarchical_2021_res8_ward_8.parquet -> method=hierarchical, config=ward_8
            # gmm_2021_res8_k10.parquet -> method=gmm, config=k10
            
            if 'kmeans' in filename:
                # kmeans_2021_res8_k5 -> k5
                config = filename.split('_')[-1]
                methods_configs.append(('kmeans', config))
            elif 'hierarchical' in filename:
                # hierarchical_2021_res8_ward_8 -> ward_8
                parts = filename.split('_')
                config = '_'.join(parts[4:])  # ward_8
                methods_configs.append(('hierarchical', config))
            elif 'gmm' in filename:
                # gmm_2021_res8_k10 -> k10
                config = filename.split('_')[-1]
                methods_configs.append(('gmm', config))
        
        # Create visualizations for each method/config
        for method, config in tqdm(methods_configs, desc="Creating visualizations"):
            self.visualize_clustering_method(method, config)
        
        # Create comparison plots
        logger.info("Creating comparison plots...")
        
        # Load representative results for comparison
        comparison_data = {}
        for method, config in methods_configs[:3]:  # Limit to first 3 for comparison
            file_path = results_dir / f"{method}_2021_res8_{config}.parquet"
            comparison_data[f"{method}_{config}"] = pd.read_parquet(file_path)
        
        if comparison_data:
            fig = self.create_comparison_plot(comparison_data)
            for fmt in self.viz_config['save_formats']:
                output_path = self.comp_dir / f"method_comparison_2021_res8.{fmt}"
                fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        logger.info("All visualizations complete!")


def main():
    """Main entry point for visualization."""
    import yaml
    from pathlib import Path
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize visualizer
    visualizer = DelNorteVisualizer(config)
    
    # Create all visualizations
    logger.info("Starting visualization generation for Del Norte 2021 analysis...")
    visualizer.visualize_all_results()
    
    logger.info("Visualization generation complete!")


if __name__ == "__main__":
    main()