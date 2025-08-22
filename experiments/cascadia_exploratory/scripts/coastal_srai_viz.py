#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SRAI+Folium visualization for Cascadia Coastal Forests clustering results.
Creates clean maps with NO hexagon outlines, just fill colors with transparency.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import h3
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# SRAI imports for H3 operations
try:
    from srai.regionalizers import H3Regionalizer
    SRAI_AVAILABLE = True
except ImportError:
    print("Warning: SRAI not available, using h3 library directly")
    SRAI_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoastalSRAIVisualizer:
    """Create clean cartographic visualizations using SRAI and Folium."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.data_path = Path("data/h3_2021_res8_coastal_forests/cascadia_coastal_forests_2021_res8_final.parquet")
        self.results_dir = Path("results/coastal_2021")
        self.plots_dir = Path("plots/coastal_2021")
        
        # Create output directories
        for subdir in ['static', 'interactive', 'comparisons']:
            (self.plots_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.center_lat = 41.0  # Cascadia coast center
        self.center_lon = -122.0
        
        # Color palettes for clustering
        self.color_palettes = {
            'Set1': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
                     '#ffff33', '#a65628', '#f781bf', '#999999'],
            'Set2': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
                     '#ffd92f', '#e5c494', '#b3b3b3'],
            'Set3': ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
                     '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
                     '#ccebc5', '#ffed6f'],
            'tab10': list(mcolors.TABLEAU_COLORS.values()),
            'tab20': ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'],
            'Paired': ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
                       '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a',
                       '#ffff99', '#b15928']
        }
    
    def load_data(self):
        """Load the coastal forest hexagon data."""
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_parquet(self.data_path)
        
        # Ensure we have h3_index
        if 'h3_index' not in self.data.columns:
            if self.data.index.name:
                self.data['h3_index'] = self.data.index
            else:
                # First column should be h3_index
                self.data['h3_index'] = self.data.iloc[:, 0]
        
        logger.info(f"Loaded {len(self.data)} hexagons")
        return self.data
    
    def load_clustering_results(self, method: str, k: int) -> pd.DataFrame:
        """Load clustering assignments for a specific method and k."""
        assignments_path = self.results_dir / f"assignments/{method}_k{k}.parquet"
        
        if not assignments_path.exists():
            logger.warning(f"Clustering results not found: {assignments_path}")
            return None
        
        logger.info(f"Loading clustering: {method} k={k}")
        clusters_df = pd.read_parquet(assignments_path)
        
        # Merge with main data
        if 'h3_index' in clusters_df.columns:
            result = self.data.merge(clusters_df[['h3_index', 'cluster']], on='h3_index', how='left')
        else:
            # Assume index alignment
            result = self.data.copy()
            result['cluster'] = clusters_df['cluster'].values
        
        return result
    
    def get_cluster_colors(self, n_clusters: int, palette: str = 'Set1') -> Dict[int, str]:
        """Get color mapping for clusters."""
        colors = self.color_palettes.get(palette, self.color_palettes['Set1'])
        
        # Extend palette if needed
        if n_clusters > len(colors):
            # Cycle through colors
            colors = colors * (n_clusters // len(colors) + 1)
        
        return {i: colors[i % len(colors)] for i in range(n_clusters)}
    
    def create_base_map(self, tiles: str = 'OpenStreetMap') -> folium.Map:
        """Create base Folium map centered on Cascadia coast."""
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=8,
            tiles=None,  # We'll add tiles separately for layer control
            prefer_canvas=True  # Better performance for many polygons
        )
        
        # Add multiple tile layers
        tile_options = {
            'OpenStreetMap': {
                'tiles': 'OpenStreetMap',
                'attr': 'OpenStreetMap',
                'name': 'OpenStreetMap (Roads & Towns)',
                'overlay': False,
                'control': True
            },
            'CartoDB Positron': {
                'tiles': 'CartoDB positron',
                'attr': 'CartoDB',
                'name': 'CartoDB Light',
                'overlay': False,
                'control': True
            },
            'CartoDB Dark': {
                'tiles': 'CartoDB dark_matter',
                'attr': 'CartoDB',
                'name': 'CartoDB Dark',
                'overlay': False,
                'control': True
            },
            'Stamen Terrain': {
                'tiles': 'https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
                'attr': 'Stamen',
                'name': 'Terrain',
                'overlay': False,
                'control': True,
                'subdomains': 'abcd'
            },
            'ESRI World Imagery': {
                'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                'attr': 'ESRI',
                'name': 'Satellite',
                'overlay': False,
                'control': True
            }
        }
        
        # Add the selected base tile
        if tiles in tile_options:
            folium.TileLayer(**tile_options[tiles]).add_to(m)
        else:
            folium.TileLayer(tiles='OpenStreetMap').add_to(m)
        
        # Add other tiles to layer control
        for name, opts in tile_options.items():
            if name != tiles:
                folium.TileLayer(**opts).add_to(m)
        
        return m
    
    def add_hexagons_to_map(self, m: folium.Map, data: pd.DataFrame, 
                            palette: str = 'Set1', opacity: float = 0.65):
        """Add H3 hexagons to map with NO outlines, just fill colors."""
        
        # Get unique clusters and colors
        n_clusters = data['cluster'].nunique()
        cluster_colors = self.get_cluster_colors(n_clusters, palette)
        
        logger.info(f"Adding {len(data)} hexagons with {n_clusters} clusters")
        
        # Create feature group for hexagons
        hex_layer = folium.FeatureGroup(name=f'Clusters ({n_clusters} groups)')
        
        # Add hexagons with progress bar
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Adding hexagons"):
            h3_id = row['h3_index']
            cluster = row['cluster']
            
            # Get hexagon boundary
            try:
                coords = h3.h3_to_geo_boundary(h3_id, geo_json=True)
                
                # Create polygon with NO outline
                folium.Polygon(
                    locations=coords,
                    color=None,  # NO outline color
                    weight=0,    # NO outline weight
                    fillColor=cluster_colors[cluster],
                    fillOpacity=opacity,
                    fill=True,
                    popup=f"Cluster: {cluster}<br>H3: {h3_id}",
                    tooltip=f"Cluster {cluster}"
                ).add_to(hex_layer)
                
            except Exception as e:
                logger.debug(f"Could not process hexagon {h3_id}: {e}")
        
        hex_layer.add_to(m)
        
        # Add legend
        self.add_legend(m, cluster_colors, n_clusters)
        
        return m
    
    def add_legend(self, m: folium.Map, cluster_colors: Dict, n_clusters: int):
        """Add a legend to the map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 150px; height: auto;
                    background-color: white; z-index: 1000; 
                    border: 2px solid grey; border-radius: 5px;
                    font-size: 14px; padding: 10px">
        <p style="margin: 0; font-weight: bold;">Clusters</p>
        '''
        
        for i in range(n_clusters):
            legend_html += f'''
            <p style="margin: 2px;">
                <span style="background-color: {cluster_colors[i]}; 
                            width: 20px; height: 15px; 
                            display: inline-block; margin-right: 5px;
                            border: 1px solid #ccc;"></span>
                Cluster {i}
            </p>
            '''
        
        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def create_visualization(self, method: str, k: int, palette: str = 'Set1',
                           tiles: str = 'OpenStreetMap', opacity: float = 0.65):
        """Create a complete visualization for one clustering result."""
        
        # Load clustering results
        data_with_clusters = self.load_clustering_results(method, k)
        
        if data_with_clusters is None:
            logger.error(f"Could not load clustering results for {method} k={k}")
            return None
        
        # Create base map
        m = self.create_base_map(tiles=tiles)
        
        # Add hexagons
        m = self.add_hexagons_to_map(m, data_with_clusters, palette=palette, opacity=opacity)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Save interactive map
        output_path = self.plots_dir / f"interactive/{method}_k{k}_{palette}.html"
        m.save(str(output_path))
        logger.info(f"Saved interactive map to {output_path}")
        
        return m
    
    def create_all_visualizations(self):
        """Create visualizations for all clustering results."""
        
        # Load data
        self.load_data()
        
        # Define methods and k values to visualize
        methods = {
            'kmeans': [5, 8, 10, 12, 15, 20],
            'hierarchical_ward': [8, 10, 12],
            'gmm': [5, 8, 10, 12, 15]
        }
        
        # Define palette options
        palettes = ['Set1', 'Set2', 'tab10', 'tab20']
        
        logger.info("="*80)
        logger.info("CREATING COASTAL FOREST VISUALIZATIONS")
        logger.info("="*80)
        
        for method, k_values in methods.items():
            for k in k_values:
                # Check if clustering exists
                assignments_path = self.results_dir / f"assignments/{method}_k{k}.parquet"
                if not assignments_path.exists():
                    logger.warning(f"Skipping {method} k={k} - no results found")
                    continue
                
                # Create with different palettes and base maps
                for palette in palettes[:2]:  # Use first 2 palettes for each
                    for tiles in ['OpenStreetMap', 'CartoDB positron']:
                        logger.info(f"\nCreating: {method} k={k} with {palette} on {tiles}")
                        
                        try:
                            self.create_visualization(
                                method=method,
                                k=k,
                                palette=palette,
                                tiles=tiles,
                                opacity=0.65
                            )
                        except Exception as e:
                            logger.error(f"Error creating visualization: {e}")
        
        logger.info("\n" + "="*80)
        logger.info("VISUALIZATIONS COMPLETE!")
        logger.info(f"Interactive maps saved to {self.plots_dir / 'interactive'}")
        logger.info("="*80)


def main():
    """Main execution function."""
    visualizer = CoastalSRAIVisualizer()
    visualizer.create_all_visualizations()


if __name__ == "__main__":
    main()