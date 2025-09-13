#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick visualization script for available clustering results.
Creates clean Folium maps with no hexagon outlines.
"""

import numpy as np
import pandas as pd
import folium
from folium import plugins
import h3
from pathlib import Path
import logging
import matplotlib.colors as mcolors
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_coastal_visualization(method='kmeans', k=10):
    """Create a quick visualization for one clustering result."""
    
    # Paths
    data_path = Path("data/h3_2021_res8_coastal_forests/cascadia_coastal_forests_2021_res8_final.parquet")
    assignments_path = Path(f"results/coastal_2021/assignments/{method}_k{k}.parquet")
    output_dir = Path("plots/coastal_2021/interactive")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files exist
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    if not assignments_path.exists():
        logger.error(f"Clustering results not found: {assignments_path}")
        return
    
    # Load data
    logger.info("Loading data...")
    data = pd.read_parquet(data_path)
    clusters = pd.read_parquet(assignments_path)
    
    # Get h3_index column
    if 'h3_index' not in data.columns:
        data['h3_index'] = data.index if data.index.name else data.iloc[:, 0]
    
    # Merge cluster assignments
    if 'h3_index' in clusters.columns:
        data = data.merge(clusters[['h3_index', 'cluster']], on='h3_index', how='left')
    else:
        data['cluster'] = clusters['cluster'].values
    
    # Color palette
    n_clusters = data['cluster'].nunique()
    colors = list(mcolors.TABLEAU_COLORS.values())[:n_clusters]
    if n_clusters > len(colors):
        colors = colors * (n_clusters // len(colors) + 1)
    cluster_colors = {i: colors[i % len(colors)] for i in range(n_clusters)}
    
    logger.info(f"Creating map with {len(data)} hexagons and {n_clusters} clusters")
    
    # Create map centered on Cascadia coast
    m = folium.Map(
        location=[41.0, -122.0],
        zoom_start=8,
        tiles='OpenStreetMap',
        prefer_canvas=True
    )
    
    # Add CartoDB light tiles as an option
    folium.TileLayer(
        tiles='CartoDB positron',
        attr='CartoDB',
        name='Light Background',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add satellite imagery
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='ESRI',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Create feature group for hexagons
    hex_layer = folium.FeatureGroup(name=f'{method.title()} k={k}')
    
    # Add hexagons (sample for quick display)
    sample_size = min(10000, len(data))  # Limit for performance
    data_sample = data.sample(n=sample_size, random_state=42)
    
    logger.info(f"Adding {sample_size} hexagons to map...")
    
    for idx, row in data_sample.iterrows():
        h3_id = row['h3_index']
        cluster = row['cluster']
        
        try:
            # Get hexagon boundary
            coords = h3.h3_to_geo_boundary(h3_id, geo_json=True)
            
            # Add hexagon with NO outline
            folium.Polygon(
                locations=coords,
                color=None,       # NO outline
                weight=0,         # NO weight
                fillColor=cluster_colors[cluster],
                fillOpacity=0.6,  # Semi-transparent
                fill=True,
                tooltip=f"Cluster {cluster}"
            ).add_to(hex_layer)
            
        except Exception as e:
            continue
    
    hex_layer.add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 120px;
                background-color: white; z-index: 1000; 
                border: 2px solid grey; border-radius: 5px;
                font-size: 12px; padding: 10px">
    <p style="margin: 0; font-weight: bold;">{method.title()} k={k}</p>
    '''
    
    for i in range(min(n_clusters, 20)):  # Limit legend size
        legend_html += f'''
        <p style="margin: 2px;">
            <span style="background-color: {cluster_colors[i]}; 
                        width: 15px; height: 12px; 
                        display: inline-block; margin-right: 3px;"></span>
            C{i}
        </p>
        '''
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add controls
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)
    
    # Save map
    output_path = output_dir / f"{method}_k{k}_quick.html"
    m.save(str(output_path))
    logger.info(f"Map saved to {output_path}")
    
    return m


def main():
    """Create visualizations for available results."""
    
    # Check what's available
    assignments_dir = Path("results/coastal_2021/assignments")
    
    if not assignments_dir.exists():
        logger.error("No clustering results found yet")
        return
    
    # Get available files
    available = list(assignments_dir.glob("*.parquet"))
    logger.info(f"Found {len(available)} clustering results")
    
    # Create visualizations for each
    for file in available[:3]:  # Limit to first 3 for quick demo
        # Parse filename
        name = file.stem  # e.g., "kmeans_k10"
        parts = name.split('_')
        
        if 'hierarchical' in name:
            method = '_'.join(parts[:2])  # e.g., "hierarchical_ward"
            k = int(parts[-1].replace('k', ''))
        else:
            method = parts[0]  # e.g., "kmeans"
            k = int(parts[1].replace('k', ''))
        
        logger.info(f"\nCreating visualization for {method} k={k}")
        
        try:
            create_coastal_visualization(method=method, k=k)
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")


if __name__ == "__main__":
    main()