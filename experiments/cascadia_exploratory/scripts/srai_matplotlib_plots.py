#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Use SRAI to get proper H3 geometries, then plot with matplotlib.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from srai.regionalizers import H3Regionalizer
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_matplotlib_cluster_plots():
    """Use SRAI for H3 geometries, matplotlib for plotting."""
    
    # Create output directory
    output_dir = Path("plots/coastal_2021/static")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create H3 regionalizer
    regionalizer = H3Regionalizer(resolution=8)
    
    # Create Cascadia bounding box
    bbox_polygon = Polygon([
        (-124.7, 38.5),
        (-121.0, 38.5), 
        (-121.0, 43.5),
        (-124.7, 43.5),
        (-124.7, 38.5)
    ])
    area_gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs='EPSG:4326')
    
    # Get H3 regions using SRAI
    logger.info("Creating H3 geometries with SRAI...")
    h3_regions_gdf = regionalizer.transform(area_gdf)
    logger.info(f"SRAI created {len(h3_regions_gdf)} H3 regions")
    
    # Get available clustering results
    assignments_dir = Path("results/coastal_2021/assignments")
    clustering_files = list(assignments_dir.glob("*.parquet"))
    
    # Create individual plots
    for cluster_file in clustering_files[:5]:  # First 5 methods
        method_name = cluster_file.stem
        logger.info(f"Creating plot for {method_name}")
        
        # Load clusters
        clusters_df = pd.read_parquet(cluster_file)
        if 'h3_index' in clusters_df.columns:
            clusters_df = clusters_df.set_index('h3_index')
        
        # Get regions with cluster data
        clustered_regions = h3_regions_gdf.loc[h3_regions_gdf.index.intersection(clusters_df.index)].copy()
        clustered_regions['cluster'] = clusters_df.loc[clustered_regions.index, 'cluster']
        
        logger.info(f"  Plotting {len(clustered_regions)} regions")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Plot each cluster with different color
        n_clusters = clustered_regions['cluster'].nunique()
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        
        for i, cluster_id in enumerate(sorted(clustered_regions['cluster'].unique())):
            cluster_data = clustered_regions[clustered_regions['cluster'] == cluster_id]
            cluster_data.plot(
                ax=ax,
                color=colors[i],
                alpha=0.7,
                edgecolor='none',  # No hexagon outlines
                label=f'Cluster {cluster_id}'
            )
        
        # Customize plot
        ax.set_title(f'Cascadia Coastal Forests - {method_name.upper().replace("_", " ")}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legend (limit to first 10 clusters for readability)
        if n_clusters <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set bounds to focus on data
        bounds = clustered_regions.bounds
        ax.set_xlim(bounds.minx.min() - 0.1, bounds.maxx.max() + 0.1)
        ax.set_ylim(bounds.miny.min() - 0.1, bounds.maxy.max() + 0.1)
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f"{method_name}_matplotlib.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved to {output_path}")
        plt.close()
    
    # Create comparison plot
    logger.info("\nCreating comparison plot...")
    methods_to_compare = ['kmeans_k5', 'kmeans_k10', 'kmeans_k15']
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    for idx, method in enumerate(methods_to_compare):
        cluster_file = assignments_dir / f"{method}.parquet"
        if not cluster_file.exists():
            continue
        
        # Load and process
        clusters_df = pd.read_parquet(cluster_file)
        if 'h3_index' in clusters_df.columns:
            clusters_df = clusters_df.set_index('h3_index')
        
        clustered_regions = h3_regions_gdf.loc[h3_regions_gdf.index.intersection(clusters_df.index)].copy()
        clustered_regions['cluster'] = clusters_df.loc[clustered_regions.index, 'cluster']
        
        # Plot
        n_clusters = clustered_regions['cluster'].nunique()
        colormap = plt.cm.Set1 if idx == 0 else (plt.cm.Set2 if idx == 1 else plt.cm.Set3)
        colors = colormap(np.linspace(0, 1, n_clusters))
        
        for i, cluster_id in enumerate(sorted(clustered_regions['cluster'].unique())):
            cluster_data = clustered_regions[clustered_regions['cluster'] == cluster_id]
            cluster_data.plot(
                ax=axes[idx],
                color=colors[i],
                alpha=0.7,
                edgecolor='none'
            )
        
        axes[idx].set_title(f'{method.upper().replace("_", " ")}', fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_aspect('equal')
        
        # Set consistent bounds
        if idx == 0:
            bounds = clustered_regions.bounds
            xlim = (bounds.minx.min() - 0.1, bounds.maxx.max() + 0.1)
            ylim = (bounds.miny.min() - 0.1, bounds.maxy.max() + 0.1)
        
        axes[idx].set_xlim(xlim)
        axes[idx].set_ylim(ylim)
    
    plt.suptitle('Cascadia Coastal Forests - K-means Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "kmeans_comparison_matplotlib.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved comparison to {output_path}")
    plt.close()
    
    logger.info("All plots completed!")


if __name__ == "__main__":
    create_matplotlib_cluster_plots()