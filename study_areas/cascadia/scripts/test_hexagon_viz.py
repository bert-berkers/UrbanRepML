#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test visualization to ensure hexagons are visible on the map.
"""

import numpy as np
import pandas as pd
import folium
import h3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hexagon_display():
    """Create a test map to verify hexagons are displaying correctly."""
    
    # Load data
    data_path = Path("data/h3_2021_res8_coastal_forests/cascadia_coastal_forests_2021_res8_final.parquet")
    assignments_path = Path("results/coastal_2021/assignments/kmeans_k10.parquet")
    
    logger.info("Loading data...")
    data = pd.read_parquet(data_path)
    clusters = pd.read_parquet(assignments_path)
    
    # Get h3_index
    if 'h3_index' not in data.columns:
        if data.index.name:
            data['h3_index'] = data.index
        else:
            data['h3_index'] = data.iloc[:, 0]
    
    # Merge clusters
    if 'h3_index' in clusters.columns:
        data = data.merge(clusters[['h3_index', 'cluster']], on='h3_index', how='left')
    else:
        data['cluster'] = clusters['cluster'].values
    
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"H3 index sample: {data['h3_index'].iloc[0]}")
    
    # Test if h3_index values are valid
    test_h3 = data['h3_index'].iloc[0]
    try:
        test_coords = h3.h3_to_geo(test_h3)
        logger.info(f"Test H3 center coordinates: {test_coords}")
    except Exception as e:
        logger.error(f"Invalid H3 index format: {e}")
        return
    
    # Get bounds of data
    centroids = []
    for h3_id in data['h3_index'].iloc[:1000]:  # Sample for bounds
        try:
            lat, lon = h3.h3_to_geo(h3_id)
            centroids.append([lat, lon])
        except:
            continue
    
    centroids = np.array(centroids)
    center_lat = centroids[:, 0].mean()
    center_lon = centroids[:, 1].mean()
    
    logger.info(f"Map center: {center_lat:.4f}, {center_lon:.4f}")
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,  # Start zoomed out to see the region
        tiles='OpenStreetMap'
    )
    
    # Color palette - bright, distinct colors
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
              '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A']
    
    n_clusters = data['cluster'].nunique()
    cluster_colors = {i: colors[i % len(colors)] for i in range(n_clusters)}
    
    # Add ALL hexagons (or a larger sample)
    sample_size = min(50000, len(data))  # Increase sample size
    data_sample = data.sample(n=sample_size, random_state=42)
    
    logger.info(f"Adding {sample_size} hexagons to map...")
    
    # Group by cluster for efficiency
    for cluster_id in range(n_clusters):
        cluster_data = data_sample[data_sample['cluster'] == cluster_id]
        logger.info(f"  Cluster {cluster_id}: {len(cluster_data)} hexagons")
        
        for idx, row in cluster_data.iterrows():
            h3_id = row['h3_index']
            
            try:
                # Get hexagon boundary
                coords = h3.h3_to_geo_boundary(h3_id, geo_json=True)
                
                # Add hexagon with bright color and slight border for visibility
                folium.Polygon(
                    locations=coords,
                    color='black',        # Thin black border for visibility
                    weight=0.2,           # Very thin border
                    fillColor=cluster_colors[cluster_id],
                    fillOpacity=0.7,      # More opaque
                    fill=True,
                    tooltip=f"Cluster {cluster_id}<br>H3: {h3_id}"
                ).add_to(m)
                
            except Exception as e:
                logger.debug(f"Could not add hexagon {h3_id}: {e}")
    
    # Add marker at center for reference
    folium.Marker(
        [center_lat, center_lon],
        popup="Data Center",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add circle to show data extent
    folium.Circle(
        location=[center_lat, center_lon],
        radius=100000,  # 100km radius
        color='red',
        fill=False,
        weight=2,
        popup='Approximate data extent'
    ).add_to(m)
    
    # Save map
    output_path = Path("plots/coastal_2021/interactive/test_hexagons_visible.html")
    m.save(str(output_path))
    logger.info(f"Test map saved to {output_path}")
    
    # Also create a simple map with just a few hexagons to verify
    m2 = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,  # Zoom in more
        tiles='OpenStreetMap'
    )
    
    # Add just 100 hexagons with thick borders
    for idx, row in data.iloc[:100].iterrows():
        h3_id = row['h3_index']
        cluster = row['cluster']
        
        try:
            coords = h3.h3_to_geo_boundary(h3_id, geo_json=True)
            
            folium.Polygon(
                locations=coords,
                color='red',          # Red border
                weight=2,             # Thick border
                fillColor='yellow',   # Yellow fill
                fillOpacity=0.5,
                fill=True,
                popup=f"Test Hexagon<br>Cluster: {cluster}<br>H3: {h3_id}"
            ).add_to(m2)
            
            # Also add center point
            center = h3.h3_to_geo(h3_id)
            folium.CircleMarker(
                location=center,
                radius=3,
                color='blue',
                fill=True,
                fillColor='blue'
            ).add_to(m2)
            
        except Exception as e:
            logger.error(f"Error with hexagon {h3_id}: {e}")
    
    output_path2 = Path("plots/coastal_2021/interactive/test_100_hexagons.html")
    m2.save(str(output_path2))
    logger.info(f"Simple test map saved to {output_path2}")
    
    return m, m2


if __name__ == "__main__":
    test_hexagon_display()