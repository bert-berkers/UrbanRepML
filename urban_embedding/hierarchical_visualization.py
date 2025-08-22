#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hierarchical Landscape Visualization System

Creates beautiful multi-resolution plots showing:
- Individual clustering for each H3 resolution layer
- Combined holographic landscape smoothing all resolutions
- Spatial embedding patterns across the hierarchical screen layers
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import RegularPolygon
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import h3
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class HierarchicalLandscapeVisualizer:
    """
    Creates stunning visualizations of hierarchical spatial embeddings across
    multiple H3 resolution layers - the holographic screen layers of space.
    """
    
    def __init__(self, output_dir: str = "hierarchical_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for different resolutions
        self.resolution_colors = {
            8: 'viridis',
            9: 'plasma', 
            10: 'inferno',
            11: 'magma'
        }
        
        # Figure settings for publication quality
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.weight'] = 'bold'
        
        logger.info(f"Hierarchical landscape visualizer initialized: {self.output_dir}")
    
    def create_resolution_cluster_plot(
        self,
        hierarchical_embeddings: Dict[int, pd.DataFrame],
        cluster_results: Dict[int, Dict],
        resolution: int,
        bounds: Dict[str, float]
    ) -> None:
        """
        Create beautiful cluster visualization for a single resolution layer.
        """
        if resolution not in hierarchical_embeddings or resolution not in cluster_results:
            logger.warning(f"Missing data for resolution {resolution}")
            return
        
        logger.info(f"Creating cluster plot for resolution {resolution}")
        
        df = hierarchical_embeddings[resolution]
        
        # Get best clustering result
        best_method = None
        best_score = -1
        for method, result in cluster_results[resolution].items():
            if result.metrics.silhouette_score > best_score:
                best_score = result.metrics.silhouette_score
                best_method = method
        
        if best_method is None:
            logger.warning(f"No clustering results for resolution {resolution}")
            return
        
        cluster_labels = cluster_results[resolution][best_method].cluster_labels
        n_clusters = len(set(cluster_labels))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Resolution {resolution} Hierarchical Analysis\n'
                    f'{len(df)} hexagons, {n_clusters} clusters ({best_method})\n'
                    f'Silhouette Score: {best_score:.3f}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Spatial clusters with hexagons
        ax1 = axes[0, 0]
        self._plot_hexagonal_clusters(df, cluster_labels, ax1, resolution)
        ax1.set_title(f'Spatial Clusters (Resolution {resolution})', fontweight='bold')
        
        # Plot 2: Elevation landscape
        ax2 = axes[0, 1]
        if 'elevation' in df.columns:
            self._plot_elevation_landscape(df, ax2, resolution)
        ax2.set_title('Elevation Landscape', fontweight='bold')
        
        # Plot 3: Embedding space (first 2 PCA components)
        ax3 = axes[1, 0]
        self._plot_embedding_space(df, cluster_labels, ax3, resolution)
        ax3.set_title('Embedding Space (PCA)', fontweight='bold')
        
        # Plot 4: Cluster statistics
        ax4 = axes[1, 1]
        self._plot_cluster_statistics(cluster_labels, cluster_results[resolution], ax4)
        ax4.set_title('Cluster Quality Metrics', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"resolution_{resolution}_landscape.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved resolution {resolution} plot: {filepath}")
    
    def _plot_hexagonal_clusters(
        self, 
        df: pd.DataFrame, 
        cluster_labels: np.ndarray, 
        ax: plt.Axes,
        resolution: int
    ) -> None:
        """Plot hexagonal regions colored by cluster."""
        
        # Get unique clusters and assign colors
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
        
        # Track bounds for axis limits
        all_lats, all_lons = [], []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            if i >= len(cluster_labels):
                break
                
            cluster_id = cluster_labels[i]
            color = colors[np.where(unique_clusters == cluster_id)[0][0]]
            
            # Get hexagon boundary
            try:
                boundary = h3.cell_to_boundary(idx)
                lats, lons = zip(*[(lat, lon) for lat, lon in boundary])
                
                all_lats.extend(lats)
                all_lons.extend(lons)
                
                # Create hexagon patch
                hex_patch = plt.Polygon(list(zip(lons, lats)), 
                                      facecolor=color, edgecolor='white', 
                                      linewidth=0.5, alpha=0.7)
                ax.add_patch(hex_patch)
                
            except Exception as e:
                # Fallback to point
                if 'lat' in row and 'lon' in row:
                    ax.scatter(row['lon'], row['lat'], c=[color], s=10, alpha=0.7)
        
        # Set axis limits and labels
        if all_lats and all_lons:
            ax.set_xlim(min(all_lons), max(all_lons))
            ax.set_ylim(min(all_lats), max(all_lats))
        
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=f'Cluster {cluster_id}') 
                          for i, cluster_id in enumerate(unique_clusters)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    def _plot_elevation_landscape(self, df: pd.DataFrame, ax: plt.Axes, resolution: int) -> None:
        """Plot elevation as a landscape."""
        
        if 'elevation' not in df.columns or 'lat' not in df.columns or 'lon' not in df.columns:
            ax.text(0.5, 0.5, 'No elevation data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create scatter plot with elevation colors
        scatter = ax.scatter(df['lon'], df['lat'], c=df['elevation'], 
                           cmap=self.resolution_colors[resolution], 
                           s=20, alpha=0.8, edgecolors='white', linewidths=0.5)
        
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Elevation (m)', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_embedding_space(
        self, 
        df: pd.DataFrame, 
        cluster_labels: np.ndarray, 
        ax: plt.Axes,
        resolution: int
    ) -> None:
        """Plot embedding space using PCA."""
        
        # Get embedding features
        embedding_cols = [col for col in df.columns if col.startswith('A') or col.startswith('poi_')]
        
        if len(embedding_cols) < 2:
            ax.text(0.5, 0.5, 'Insufficient embedding features', ha='center', va='center', transform=ax.transAxes)
            return
        
        features = df[embedding_cols].fillna(0)
        
        # Apply PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(StandardScaler().fit_transform(features))
        
        # Plot colored by clusters
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                      c=[colors[i]], label=f'Cluster {cluster_id}', 
                      alpha=0.7, s=30)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cluster_statistics(
        self, 
        cluster_labels: np.ndarray, 
        cluster_results: Dict, 
        ax: plt.Axes
    ) -> None:
        """Plot cluster quality statistics."""
        
        methods = list(cluster_results.keys())
        silhouette_scores = [cluster_results[method].metrics.silhouette_score for method in methods]
        spatial_coherences = [cluster_results[method].spatial_coherence for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, silhouette_scores, width, label='Silhouette Score', alpha=0.8)
        bars2 = ax.bar(x + width/2, spatial_coherences, width, label='Spatial Coherence', alpha=0.8)
        
        ax.set_xlabel('Clustering Method', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def create_combined_holographic_landscape(
        self,
        hierarchical_embeddings: Dict[int, pd.DataFrame],
        cluster_results: Dict[int, Dict],
        bounds: Dict[str, float]
    ) -> None:
        """
        Create the ultimate combined holographic landscape showing all resolution layers
        smoothly blended into a comprehensive hierarchical visualization.
        """
        logger.info("Creating COMBINED HOLOGRAPHIC LANDSCAPE - the ultimate visualization!")
        
        # Create massive figure
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('HIERARCHICAL HOLOGRAPHIC LANDSCAPE\n'
                    'Multi-Resolution Spatial Intelligence Layers', 
                    fontsize=20, fontweight='bold')
        
        # Define grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Individual resolution plots (top row)
        for i, resolution in enumerate([8, 9, 10, 11]):
            if resolution not in hierarchical_embeddings:
                continue
            
            ax = fig.add_subplot(gs[0, i])
            self._create_mini_resolution_plot(hierarchical_embeddings[resolution], 
                                            cluster_results.get(resolution, {}), 
                                            ax, resolution)
        
        # Combined landscape (middle - large plot)
        ax_combined = fig.add_subplot(gs[1, :])
        self._create_combined_landscape_plot(hierarchical_embeddings, cluster_results, ax_combined, bounds)
        
        # Statistics and analysis (bottom row)
        ax_stats = fig.add_subplot(gs[2, :2])
        self._plot_hierarchical_statistics(hierarchical_embeddings, cluster_results, ax_stats)
        
        ax_cross = fig.add_subplot(gs[2, 2:])
        self._plot_cross_resolution_analysis(hierarchical_embeddings, ax_cross)
        
        # Save the masterpiece
        filepath = self.output_dir / "HOLOGRAPHIC_LANDSCAPE_MASTERPIECE.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"HOLOGRAPHIC LANDSCAPE MASTERPIECE saved: {filepath}")
    
    def _create_mini_resolution_plot(
        self, 
        df: pd.DataFrame, 
        cluster_results: Dict, 
        ax: plt.Axes, 
        resolution: int
    ) -> None:
        """Create a mini plot for one resolution."""
        
        if not cluster_results:
            ax.text(0.5, 0.5, f'Res {resolution}\nNo clusters', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get best clustering
        best_method = max(cluster_results.keys(), 
                         key=lambda k: cluster_results[k].metrics.silhouette_score)
        cluster_labels = cluster_results[best_method].cluster_labels
        
        # Simple scatter plot colored by clusters
        if 'lat' in df.columns and 'lon' in df.columns:
            unique_clusters = np.unique(cluster_labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = cluster_labels == cluster_id
                ax.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'], 
                          c=[colors[i]], s=2, alpha=0.6)
        
        ax.set_title(f'Resolution {resolution}\n{len(df)} hexagons', fontweight='bold', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_combined_landscape_plot(
        self,
        hierarchical_embeddings: Dict[int, pd.DataFrame],
        cluster_results: Dict[int, Dict],
        ax: plt.Axes,
        bounds: Dict[str, float]
    ) -> None:
        """Create the main combined landscape visualization."""
        
        logger.info("Creating combined landscape with multi-resolution smoothing...")
        
        # Combine all data points from all resolutions
        all_points = []
        all_values = []
        all_weights = []
        
        resolution_weights = {8: 1.0, 9: 0.8, 10: 0.6, 11: 0.4}  # Coarser gets more weight
        
        for resolution, df in hierarchical_embeddings.items():
            if 'lat' not in df.columns or 'lon' not in df.columns:
                continue
            
            weight = resolution_weights.get(resolution, 0.5)
            
            # Use elevation if available, otherwise use embedding features
            if 'elevation' in df.columns:
                values = df['elevation'].fillna(0)
            else:
                # Use first embedding feature
                embedding_cols = [col for col in df.columns if col.startswith('A')]
                if embedding_cols:
                    values = df[embedding_cols[0]].fillna(0)
                else:
                    values = pd.Series(np.random.random(len(df)))
            
            for idx, row in df.iterrows():
                all_points.append([row['lon'], row['lat']])
                all_values.append(values.loc[idx])
                all_weights.append(weight)
        
        if not all_points:
            ax.text(0.5, 0.5, 'No spatial data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        all_points = np.array(all_points)
        all_values = np.array(all_values)
        all_weights = np.array(all_weights)
        
        # Create interpolation grid
        lon_min, lon_max = all_points[:, 0].min(), all_points[:, 0].max()
        lat_min, lat_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        lon_grid = np.linspace(lon_min, lon_max, 200)
        lat_grid = np.linspace(lat_min, lat_max, 200)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Weighted interpolation
        weighted_values = all_values * all_weights
        
        try:
            interpolated = griddata(all_points, weighted_values, 
                                  (lon_mesh, lat_mesh), method='cubic', fill_value=0)
            
            # Create beautiful contour plot
            contour = ax.contourf(lon_mesh, lat_mesh, interpolated, levels=20, 
                                cmap='terrain', alpha=0.8)
            plt.colorbar(contour, ax=ax, label='Hierarchical Landscape Value')
            
            # Overlay points from highest resolution
            if 11 in hierarchical_embeddings:
                df_11 = hierarchical_embeddings[11]
                if 'lat' in df_11.columns and 'lon' in df_11.columns:
                    ax.scatter(df_11['lon'], df_11['lat'], c='white', s=1, alpha=0.3)
            
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}, using simple scatter")
            ax.scatter(all_points[:, 0], all_points[:, 1], c=all_values, 
                      cmap='terrain', s=20, alpha=0.6)
        
        ax.set_title('HOLOGRAPHIC LANDSCAPE\nSmoothed Multi-Resolution Spatial Intelligence', 
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Longitude', fontweight='bold')
        ax.set_ylabel('Latitude', fontweight='bold')
    
    def _plot_hierarchical_statistics(
        self,
        hierarchical_embeddings: Dict[int, pd.DataFrame],
        cluster_results: Dict[int, Dict],
        ax: plt.Axes
    ) -> None:
        """Plot hierarchical statistics across resolutions."""
        
        resolutions = sorted(hierarchical_embeddings.keys())
        cell_counts = [len(hierarchical_embeddings[res]) for res in resolutions]
        
        # Get best silhouette scores
        silhouette_scores = []
        for res in resolutions:
            if res in cluster_results and cluster_results[res]:
                best_score = max(result.metrics.silhouette_score 
                               for result in cluster_results[res].values())
                silhouette_scores.append(best_score)
            else:
                silhouette_scores.append(0)
        
        # Create dual-axis plot
        ax2 = ax.twinx()
        
        bars = ax.bar(resolutions, cell_counts, alpha=0.7, color='skyblue', label='Hexagon Count')
        line = ax2.plot(resolutions, silhouette_scores, 'ro-', linewidth=3, markersize=8, label='Silhouette Score')
        
        ax.set_xlabel('H3 Resolution', fontweight='bold')
        ax.set_ylabel('Number of Hexagons', fontweight='bold', color='blue')
        ax2.set_ylabel('Silhouette Score', fontweight='bold', color='red')
        
        ax.set_title('Hierarchical Scale Analysis', fontweight='bold')
        
        # Add value labels
        for i, (res, count) in enumerate(zip(resolutions, cell_counts)):
            ax.text(res, count + max(cell_counts)*0.01, f'{count:,}', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_cross_resolution_analysis(
        self,
        hierarchical_embeddings: Dict[int, pd.DataFrame],
        ax: plt.Axes
    ) -> None:
        """Plot cross-resolution relationship analysis."""
        
        resolutions = sorted(hierarchical_embeddings.keys())
        
        # Calculate feature complexity across resolutions
        feature_counts = []
        for res in resolutions:
            df = hierarchical_embeddings[res]
            embedding_features = len([col for col in df.columns if col.startswith('A') or col.startswith('poi_')])
            feature_counts.append(embedding_features)
        
        # Plot feature complexity
        ax.plot(resolutions, feature_counts, 'go-', linewidth=3, markersize=8, label='Feature Count')
        ax.fill_between(resolutions, feature_counts, alpha=0.3, color='green')
        
        ax.set_xlabel('H3 Resolution', fontweight='bold')
        ax.set_ylabel('Feature Complexity', fontweight='bold')
        ax.set_title('Cross-Resolution Feature Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def generate_all_visualizations(
        self,
        hierarchical_embeddings: Dict[int, pd.DataFrame],
        cluster_results: Dict[int, Dict],
        bounds: Dict[str, float]
    ) -> None:
        """Generate the complete set of hierarchical landscape visualizations."""
        
        logger.info("Generating COMPLETE hierarchical visualization suite!")
        
        # Individual resolution plots
        for resolution in hierarchical_embeddings.keys():
            self.create_resolution_cluster_plot(hierarchical_embeddings, cluster_results, resolution, bounds)
        
        # The masterpiece: combined holographic landscape
        self.create_combined_holographic_landscape(hierarchical_embeddings, cluster_results, bounds)
        
        logger.info(f"ALL VISUALIZATIONS COMPLETE! Check: {self.output_dir}")


def main():
    """Demo of hierarchical landscape visualization."""
    visualizer = HierarchicalLandscapeVisualizer()
    logger.info("Hierarchical landscape visualizer ready!")
    return visualizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()