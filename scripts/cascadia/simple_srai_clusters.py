#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple SRAI-based clustering visualization.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from srai.regionalizers import H3Regionalizer
from srai.plotting import plot_regions, plot_numeric_data
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_cluster_plot():
    """Create cluster plots using SRAI."""
    
    # Load clustering results
    assignments_path = Path("results/coastal_2021/assignments/kmeans_k10.parquet")
    logger.info("Loading clustering assignments...")
    clusters_df = pd.read_parquet(assignments_path)
    
    # Set h3_index as index
    if 'h3_index' in clusters_df.columns:
        clusters_df = clusters_df.set_index('h3_index')
    
    logger.info(f"Loaded {len(clusters_df)} cluster assignments")
    
    # Create H3 regionalizer
    regionalizer = H3Regionalizer(resolution=8)
    
    # Convert H3 indices to proper GeoDataFrame using SRAI
    logger.info("Creating H3 geometries with SRAI...")
    
    # SRAI needs a dummy area to regionalize - we'll create one covering our H3 cells
    from shapely.geometry import Polygon
    
    # Create bounding box for Cascadia coast
    bbox_polygon = Polygon([
        (-124.7, 38.5),
        (-121.0, 38.5), 
        (-121.0, 43.5),
        (-124.7, 43.5),
        (-124.7, 38.5)
    ])
    
    area_gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs='EPSG:4326')
    
    # Get H3 regions using SRAI
    h3_regions_gdf = regionalizer.transform(area_gdf)
    logger.info(f"SRAI created {len(h3_regions_gdf)} H3 regions")
    
    # Filter to only our clustered hexagons and add cluster data
    clustered_regions = h3_regions_gdf.loc[h3_regions_gdf.index.intersection(clusters_df.index)]
    clustered_regions['cluster'] = clusters_df.loc[clustered_regions.index, 'cluster']
    
    logger.info(f"Found {len(clustered_regions)} regions with cluster assignments")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Plot using SRAI's plot_numeric_data (treat cluster as numeric for coloring)
    plot_numeric_data(
        clustered_regions,
        'cluster',
        ax=ax,
        cmap='tab10',
        alpha=0.7,
        legend=True
    )
    
    ax.set_title('Cascadia Coastal Forests - K-means k=10 Clustering', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    
    # Save
    output_dir = Path("plots/coastal_2021/static")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "srai_kmeans_k10.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    plt.close()
    
    # Create multiple plots
    for method_file in Path("results/coastal_2021/assignments").glob("*.parquet")[:3]:
        method_name = method_file.stem
        logger.info(f"\nCreating plot for {method_name}")
        
        # Load this method's clusters
        method_clusters = pd.read_parquet(method_file)
        if 'h3_index' in method_clusters.columns:
            method_clusters = method_clusters.set_index('h3_index')
        
        # Get regions for this clustering
        method_regions = h3_regions_gdf.loc[h3_regions_gdf.index.intersection(method_clusters.index)]
        method_regions['cluster'] = method_clusters.loc[method_regions.index, 'cluster']
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        
        plot_numeric_data(
            method_regions,
            'cluster',
            ax=ax,
            cmap='Set1' if 'kmeans' in method_name else 'Set2',
            alpha=0.7,
            legend=True
        )
        
        ax.set_title(f'Cascadia Coastal Forests - {method_name.upper().replace("_", " ")}', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        output_path = output_dir / f"srai_{method_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved {output_path}")
        plt.close()
    
    logger.info("All SRAI plots completed!")


if __name__ == "__main__":
    create_simple_cluster_plot()