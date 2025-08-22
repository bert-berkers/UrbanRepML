#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SRAI-based Visualization System for Del Norte Exploratory Analysis.
Uses proper H3 geometry handling and categorical colormaps for cluster visualization.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import h3
import folium
from folium import plugins
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# SRAI imports for proper spatial visualization
try:
    import srai
    from srai.regionalizers import H3Regionalizer
    from srai.neighbourhoods import H3Neighbourhood
    from srai.plotting import plot_regions, plot_numeric_data, plot_categorical_data
    SRAI_AVAILABLE = True
except ImportError:
    print("Warning: SRAI not fully available, using fallback methods")
    SRAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class SRAIVisualizer:
    """Create professional visualizations using SRAI framework."""
    
    def __init__(self, config: dict):
        """Initialize SRAI visualizer with configuration."""
        self.config = config
        self.viz_config = config['visualization']
        self.colormaps = self.viz_config['colormaps']
        self.figure_size = self.viz_config['figure_size']
        self.dpi = self.viz_config['dpi']
        
        # H3 resolution from config
        self.h3_resolution = config['experiment']['h3_resolution']
        
        # Setup logging
        logging.basicConfig(
            level=config['output']['log_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create output directories
        self.plots_dir = Path("plots")
        self.spatial_srai_dir = self.plots_dir / "spatial_srai"
        self.static_dir = self.spatial_srai_dir / "static"
        self.interactive_dir = self.spatial_srai_dir / "interactive"
        self.comparisons_dir = self.spatial_srai_dir / "comparisons"
        
        for directory in [self.static_dir, self.interactive_dir, self.comparisons_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_clustered_data(self, cluster_file: Path) -> gpd.GeoDataFrame:
        """Load clustering results and convert to GeoDataFrame with H3 geometry."""
        logger.info(f"Loading clustered data from {cluster_file}")
        df = pd.read_parquet(cluster_file)
        
        # Convert H3 indices to geometries
        logger.info("Converting H3 indices to proper geometries...")
        geometries = []
        for h3_idx in tqdm(df['h3_index'], desc="Creating H3 geometries"):
            try:
                # Get H3 boundary as list of (lat, lon) tuples
                boundary = h3.cell_to_boundary(h3_idx)
                # Convert to Shapely Polygon (lon, lat order for Shapely)
                from shapely.geometry import Polygon
                poly = Polygon([(lon, lat) for lat, lon in boundary])
                geometries.append(poly)
            except Exception as e:
                logger.warning(f"Could not create geometry for {h3_idx}: {e}")
                geometries.append(None)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs='EPSG:4326')
        
        # Remove any rows with invalid geometries
        gdf = gdf[gdf.geometry.notna()]
        
        logger.info(f"Created GeoDataFrame with {len(gdf)} hexagons and {gdf['cluster'].nunique()} clusters")
        return gdf
    
    def get_categorical_colors(self, n_clusters: int, colormap_name: str) -> List[str]:
        """Get appropriate categorical colors for the number of clusters."""
        if colormap_name == 'tab10' and n_clusters <= 10:
            cmap = plt.cm.tab10
            colors = [mcolors.to_hex(cmap(i)) for i in range(n_clusters)]
        elif colormap_name == 'tab20' and n_clusters <= 20:
            cmap = plt.cm.tab20
            colors = [mcolors.to_hex(cmap(i)) for i in range(n_clusters)]
        elif colormap_name in ['Set1', 'Set2', 'Set3', 'Paired', 'Accent']:
            cmap = plt.cm.get_cmap(colormap_name)
            colors = [mcolors.to_hex(cmap(i / max(1, n_clusters - 1))) for i in range(n_clusters)]
        else:
            # Fallback to tab20 for any number of clusters
            cmap = plt.cm.tab20
            colors = [mcolors.to_hex(cmap(i % 20 / 19)) for i in range(n_clusters)]
        
        return colors
    
    def create_static_cluster_map(self, gdf: gpd.GeoDataFrame, method: str, 
                                 config_key: str, colormap: str) -> plt.Figure:
        """Create static cluster map using GeoPandas with categorical colors."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Get unique clusters and appropriate colors
        clusters = sorted(gdf['cluster'].unique())
        n_clusters = len(clusters)
        colors = self.get_categorical_colors(n_clusters, colormap)
        
        # Create color mapping
        color_dict = {cluster: colors[i] for i, cluster in enumerate(clusters)}
        gdf['color'] = gdf['cluster'].map(color_dict)
        
        # Plot with GeoPandas
        gdf.plot(
            ax=ax,
            color=gdf['color'],
            edgecolor='white',
            linewidth=0.1,
            alpha=0.9
        )
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Create title with method details
        res = self.h3_resolution
        ax.set_title(f'Del Norte 2021 - {method.upper()} Clustering ({config_key})\n'
                    f'H3 Resolution {res} - {colormap} colormap - {n_clusters} clusters',
                    fontsize=14, fontweight='bold')
        
        # Add legend with cluster IDs
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_dict[c], 
                                edgecolor='black', 
                                label=f'Cluster {c}')
                          for c in clusters]
        ax.legend(handles=legend_elements, 
                 loc='center left', 
                 bbox_to_anchor=(1, 0.5),
                 title='Clusters',
                 frameon=True,
                 fancybox=True,
                 shadow=True)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Remove axes spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_map(self, gdf: gpd.GeoDataFrame, method: str, 
                              config_key: str, colormap: str) -> folium.Map:
        """Create interactive Folium map with hover tooltips."""
        # Get center point for map
        center_lat = gdf.geometry.centroid.y.mean()
        center_lon = gdf.geometry.centroid.x.mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap',
            prefer_canvas=True
        )
        
        # Add other tile layers
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        
        # Get colors for clusters
        clusters = sorted(gdf['cluster'].unique())
        n_clusters = len(clusters)
        colors = self.get_categorical_colors(n_clusters, colormap)
        color_dict = {cluster: colors[i] for i, cluster in enumerate(clusters)}
        
        # Add hexagons to map
        for _, row in gdf.iterrows():
            # Get hexagon boundary
            hex_boundary = [[y, x] for x, y in row.geometry.exterior.coords]
            
            # Calculate feature statistics for this hexagon
            band_cols = [f'band_{i:02d}' for i in range(64)]
            if all(col in row.index for col in band_cols):
                mean_intensity = np.mean([row[col] for col in band_cols if not pd.isna(row[col])])
                std_intensity = np.std([row[col] for col in band_cols if not pd.isna(row[col])])
            else:
                mean_intensity = 0
                std_intensity = 0
            
            # Create popup text
            popup_text = f"""
            <b>Cluster {row['cluster']}</b><br>
            H3 Index: {row['h3_index']}<br>
            Location: ({row['lat']:.4f}, {row['lon']:.4f})<br>
            Mean Intensity: {mean_intensity:.3f}<br>
            Std Intensity: {std_intensity:.3f}
            """
            
            # Add polygon to map
            folium.Polygon(
                locations=hex_boundary,
                color='white',
                weight=0.5,
                fill=True,
                fillColor=color_dict[row['cluster']],
                fillOpacity=0.7,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"Cluster {row['cluster']}"
            ).add_to(m)
        
        # Add cluster centroids as markers
        cluster_centers = gdf.groupby('cluster').agg({
            'lat': 'mean',
            'lon': 'mean',
            'h3_index': 'count'  # Count hexagons per cluster
        }).reset_index()
        
        for _, center in cluster_centers.iterrows():
            folium.CircleMarker(
                location=[center['lat'], center['lon']],
                radius=5,
                popup=f"Cluster {center['cluster']} Center<br>{center['h3_index']} hexagons",
                color='black',
                fill=True,
                fillColor='white',
                weight=2
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add title
        title_html = f'''
        <h3 align="center" style="font-size:20px">
        <b>Del Norte 2021 - {method.upper()} ({config_key}) - H3 Res {self.h3_resolution}</b>
        </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_cluster_comparison(self, gdf_dict: Dict[str, gpd.GeoDataFrame]) -> plt.Figure:
        """Create side-by-side comparison of different clustering methods."""
        n_methods = len(gdf_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8), dpi=self.dpi)
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_key, gdf) in enumerate(gdf_dict.items()):
            ax = axes[idx]
            
            # Get colors
            clusters = sorted(gdf['cluster'].unique())
            n_clusters = len(clusters)
            colors = self.get_categorical_colors(n_clusters, 'tab10')
            color_dict = {cluster: colors[i] for i, cluster in enumerate(clusters)}
            gdf['color'] = gdf['cluster'].map(color_dict)
            
            # Plot
            gdf.plot(
                ax=ax,
                color=gdf['color'],
                edgecolor='white',
                linewidth=0.1,
                alpha=0.9
            )
            
            ax.set_title(f'{method_key}\n{n_clusters} clusters', fontsize=12)
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.suptitle(f'Clustering Method Comparison - Del Norte 2021 (H3 Res {self.h3_resolution})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_cluster_statistics_plot(self, gdf: gpd.GeoDataFrame, method: str, 
                                      config_key: str) -> plt.Figure:
        """Create detailed cluster statistics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        
        # 1. Cluster size distribution (bar plot)
        cluster_sizes = gdf['cluster'].value_counts().sort_index()
        n_clusters = len(cluster_sizes)
        colors = self.get_categorical_colors(n_clusters, 'tab10')
        
        axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values, 
                      color=colors[:len(cluster_sizes)])
        axes[0, 0].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Hexagons')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Spatial spread (convex hull areas)
        from shapely.ops import unary_union
        cluster_areas = []
        for cluster_id in sorted(gdf['cluster'].unique()):
            cluster_geom = gdf[gdf['cluster'] == cluster_id].geometry
            hull = unary_union(cluster_geom.values).convex_hull
            # Convert area to km²
            area_km2 = hull.area * 111.32 * 111.32 * np.cos(np.radians(gdf['lat'].mean()))
            cluster_areas.append(area_km2)
        
        axes[0, 1].bar(range(len(cluster_areas)), cluster_areas, color=colors[:len(cluster_areas)])
        axes[0, 1].set_title('Cluster Spatial Coverage', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Area (km²)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature intensity distribution per cluster
        band_cols = [f'band_{i:02d}' for i in range(64)]
        if all(col in gdf.columns for col in band_cols[:5]):  # Check first 5 bands
            cluster_intensities = []
            for cluster_id in sorted(gdf['cluster'].unique()):
                cluster_data = gdf[gdf['cluster'] == cluster_id]
                # Average across first 10 bands for visualization
                intensities = cluster_data[band_cols[:10]].mean(axis=1).values
                cluster_intensities.append(intensities)
            
            bp = axes[1, 0].boxplot(cluster_intensities, 
                                   labels=[f'C{i}' for i in range(len(cluster_intensities))],
                                   patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(cluster_intensities)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[1, 0].set_title('Feature Intensity Distribution (First 10 bands avg)', 
                               fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Cluster ID')
            axes[1, 0].set_ylabel('Mean Intensity')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cluster quality metrics
        metrics_text = f"""
        Method: {method.upper()}
        Configuration: {config_key}
        H3 Resolution: {self.h3_resolution}
        Total Clusters: {n_clusters}
        Total Hexagons: {len(gdf)}
        
        Cluster Balance:
        Min Size: {cluster_sizes.min()} hexagons
        Max Size: {cluster_sizes.max()} hexagons
        Mean Size: {cluster_sizes.mean():.1f} hexagons
        Std Size: {cluster_sizes.std():.1f} hexagons
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 1].axis('off')
        
        fig.suptitle(f'Cluster Statistics - {method.upper()} ({config_key})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_clustering_result(self, method: str, config_key: str):
        """Create all visualizations for a specific clustering result."""
        # Construct filename based on resolution
        res = self.h3_resolution
        cluster_file = Path(f"results/clusters/{method}_2021_res{res}_{config_key}.parquet")
        
        if not cluster_file.exists():
            logger.error(f"Cluster file not found: {cluster_file}")
            return
        
        # Load data as GeoDataFrame
        gdf = self.load_clustered_data(cluster_file)
        
        # Create static maps with different colormaps
        for colormap in tqdm(self.colormaps, desc=f"Creating {method} static maps"):
            # Skip if colormap not suitable for number of clusters
            n_clusters = gdf['cluster'].nunique()
            if colormap == 'tab10' and n_clusters > 10:
                continue
            if colormap == 'tab20' and n_clusters > 20:
                continue
            
            fig = self.create_static_cluster_map(gdf, method, config_key, colormap)
            output_path = self.static_dir / f"static_2021_res{res}_{method}_{config_key}_{colormap}.png"
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        # Create interactive Folium map (use best colormap for cluster count)
        if n_clusters <= 10:
            best_colormap = 'tab10'
        elif n_clusters <= 20:
            best_colormap = 'tab20'
        else:
            best_colormap = 'Set3'
        
        folium_map = self.create_interactive_map(gdf, method, config_key, best_colormap)
        html_path = self.interactive_dir / f"interactive_2021_res{res}_{method}_{config_key}.html"
        folium_map.save(str(html_path))
        logger.info(f"Saved interactive map to {html_path}")
        
        # Create statistics plot
        stats_fig = self.create_cluster_statistics_plot(gdf, method, config_key)
        stats_path = self.static_dir / f"stats_2021_res{res}_{method}_{config_key}.png"
        stats_fig.savefig(stats_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(stats_fig)
        
        logger.info(f"Created SRAI visualizations for {method} - {config_key}")
    
    def visualize_all_results(self):
        """Create visualizations for all clustering results."""
        res = self.h3_resolution
        results_dir = Path("results/clusters")
        parquet_files = list(results_dir.glob(f"*_2021_res{res}_*.parquet"))
        
        if not parquet_files:
            logger.error(f"No clustering results found for resolution {res}!")
            return
        
        # Parse filenames to get methods and configs
        methods_configs = []
        for file in parquet_files:
            filename = file.stem
            
            if 'kmeans' in filename:
                config = filename.split('_')[-1]
                methods_configs.append(('kmeans', config))
            elif 'hierarchical' in filename:
                parts = filename.split('_')
                config = '_'.join(parts[4:])
                methods_configs.append(('hierarchical', config))
            elif 'gmm' in filename:
                config = filename.split('_')[-1]
                methods_configs.append(('gmm', config))
        
        # Create visualizations for each method/config
        for method, config in tqdm(methods_configs, desc="Creating SRAI visualizations"):
            self.visualize_clustering_result(method, config)
        
        # Create comparison plots
        logger.info("Creating comparison visualizations...")
        
        # Load best results from each method for comparison
        comparison_data = {}
        
        # Select best configuration from each method (e.g., k=10 for kmeans)
        best_configs = {
            'kmeans': 'k10',
            'hierarchical': 'average_8',
            'gmm': 'k10'
        }
        
        for method, config in best_configs.items():
            file_path = results_dir / f"{method}_2021_res{res}_{config}.parquet"
            if file_path.exists():
                comparison_data[f"{method}_{config}"] = self.load_clustered_data(file_path)
        
        if comparison_data:
            comp_fig = self.create_cluster_comparison(comparison_data)
            comp_path = self.comparisons_dir / f"method_comparison_2021_res{res}.png"
            comp_fig.savefig(comp_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(comp_fig)
            logger.info(f"Saved comparison plot to {comp_path}")
        
        logger.info("All SRAI visualizations complete!")


def main():
    """Main entry point for SRAI visualization."""
    import yaml
    from pathlib import Path
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize visualizer
    visualizer = SRAIVisualizer(config)
    
    # Create all visualizations
    logger.info("Starting SRAI visualization generation...")
    visualizer.visualize_all_results()
    
    logger.info("SRAI visualization generation complete!")


if __name__ == "__main__":
    main()