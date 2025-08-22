#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple SRAI static plots for Cascadia Coastal Forests clustering.
Uses SRAI's built-in plotting functions for proper H3 visualization.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from srai.regionalizers import H3Regionalizer
from srai.plotting import plot_regions, plot_numeric_data
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_srai_plots():
    """Create static plots using SRAI's plotting functions."""
    
    # Load data
    data_path = Path("data/h3_2021_res8_coastal_forests/cascadia_coastal_forests_2021_res8_final.parquet")
    logger.info(f"Loading data from {data_path}")
    data = pd.read_parquet(data_path)
    
    # Get h3_index as the index
    if 'h3_index' not in data.columns:
        if data.index.name:
            data.index.name = None  # Clear index name if it exists
        else:
            data['h3_index'] = data.iloc[:, 0]
            data = data.set_index('h3_index')
    else:
        data = data.set_index('h3_index')
    
    logger.info(f"Loaded {len(data)} hexagons")
    
    # Create H3 regionalizer and get proper geometries
    logger.info("Creating H3 geometries with SRAI...")
    regionalizer = H3Regionalizer(resolution=8)
    
    # Create bounding box for Cascadia coastal area
    from shapely.geometry import Polygon
    bbox_polygon = Polygon([
        (-124.7, 38.5),
        (-121.0, 38.5), 
        (-121.0, 43.5),
        (-124.7, 43.5),
        (-124.7, 38.5)
    ])
    area_gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs='EPSG:4326')
    
    # Get all H3 regions in the area using SRAI
    h3_regions_gdf = regionalizer.transform(area_gdf)
    logger.info(f"SRAI created {len(h3_regions_gdf)} H3 regions")
    
    # Join with our data to get only regions with data
    gdf = h3_regions_gdf.loc[h3_regions_gdf.index.intersection(data.index)].copy()
    logger.info(f"Found {len(gdf)} regions with data")
    
    # Load clustering results
    assignments_dir = Path("results/coastal_2021/assignments")
    output_dir = Path("plots/coastal_2021/static")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available clustering results
    clustering_files = list(assignments_dir.glob("*.parquet"))[:3]  # First 3 for testing
    
    for cluster_file in clustering_files:
        method_name = cluster_file.stem  # e.g., "kmeans_k10"
        logger.info(f"\nCreating plot for {method_name}")
        
        # Load cluster assignments
        clusters = pd.read_parquet(cluster_file)
        
        # Add cluster column to geodataframe
        if 'h3_index' in clusters.columns:
            clusters = clusters.set_index('h3_index')
        
        # Align with main data - only keep regions that have cluster assignments
        gdf_with_clusters = gdf.loc[gdf.index.intersection(clusters.index)].copy()
        gdf_with_clusters['cluster'] = clusters.loc[gdf_with_clusters.index, 'cluster']
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        
        # Plot using matplotlib with proper SRAI geometries
        logger.info("Creating plot with SRAI geometries...")
        
        # Plot each cluster with different color
        n_clusters = gdf_with_clusters['cluster'].nunique()
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for i, cluster_id in enumerate(sorted(gdf_with_clusters['cluster'].unique())):
            cluster_data = gdf_with_clusters[gdf_with_clusters['cluster'] == cluster_id]
            cluster_data.plot(
                ax=ax,
                color=colors[i],
                alpha=0.7,
                edgecolor='none',  # No hexagon outlines
                label=f'Cluster {cluster_id}'
            )
        
        # Customize plot
        ax.set_title(f'Cascadia Coastal Forests - {method_name.upper()}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Set bounds to focus on data
        bounds = gdf_with_clusters.bounds
        ax.set_xlim(bounds.minx.min() - 0.1, bounds.maxx.max() + 0.1)
        ax.set_ylim(bounds.miny.min() - 0.1, bounds.maxy.max() + 0.1)
        
        # Add legend if not too many clusters
        if n_clusters <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"{method_name}_srai_clean.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
        
        plt.close()
    
    # Create a combined comparison plot
    logger.info("\nCreating comparison plot...")
    
    # Load multiple clustering results
    methods = ['kmeans_k5', 'kmeans_k10', 'kmeans_k15']
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for idx, method in enumerate(methods):
        cluster_file = assignments_dir / f"{method}.parquet"
        if not cluster_file.exists():
            continue
        
        # Load clusters
        clusters = pd.read_parquet(cluster_file)
        if 'h3_index' in clusters.columns:
            clusters = clusters.set_index('h3_index')
        
        # Get regions with cluster data
        gdf_with_clusters = gdf.loc[gdf.index.intersection(clusters.index)].copy()
        gdf_with_clusters['cluster'] = clusters.loc[gdf_with_clusters.index, 'cluster']
        
        # Plot each cluster with different color
        n_clusters = gdf_with_clusters['cluster'].nunique()
        colormap = plt.cm.Set1 if idx == 0 else (plt.cm.Set2 if idx == 1 else plt.cm.Set3)
        colors = colormap(np.linspace(0, 1, n_clusters))
        
        for i, cluster_id in enumerate(sorted(gdf_with_clusters['cluster'].unique())):
            cluster_data = gdf_with_clusters[gdf_with_clusters['cluster'] == cluster_id]
            cluster_data.plot(
                ax=axes[idx],
                color=colors[i],
                alpha=0.7,
                edgecolor='none'
            )
        
        axes[idx].set_title(method.upper().replace('_', ' '))
        axes[idx].set_aspect('equal')
        
        # Set consistent bounds
        if idx == 0:
            bounds = gdf_with_clusters.bounds
            xlim = (bounds.minx.min() - 0.1, bounds.maxx.max() + 0.1)
            ylim = (bounds.miny.min() - 0.1, bounds.maxy.max() + 0.1)
        
        axes[idx].set_xlim(xlim)
        axes[idx].set_ylim(ylim)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Cascadia Coastal Forests - K-means Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "kmeans_comparison_srai_clean.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {output_path}")
    
    plt.close()
    
    logger.info("\nAll plots completed!")


if __name__ == "__main__":
    create_srai_plots()