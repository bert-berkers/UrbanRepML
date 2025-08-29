#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple visualization of AlphaEarth clustering results.
Uses matplotlib directly without SRAI plotting functions.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main visualization function."""
    
    # Load the final merged parquet file
    data_dir = Path("data/alphaearth_h3_2021")
    parquet_file = data_dir / "alphaearth_h3_res8_20250827_192159.parquet"
    
    logger.info(f"Loading data from {parquet_file}")
    gdf = gpd.read_parquet(parquet_file)
    logger.info(f"Loaded {len(gdf)} H3 hexagons")
    
    # Extract embedding columns
    embedding_cols = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
    logger.info(f"Found {len(embedding_cols)} embedding dimensions")
    
    # Prepare data for clustering
    X = gdf[embedding_cols].values
    
    # Standardize features
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform k-means clustering
    logger.info("Performing k-means clustering with 10 clusters...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster assignments to GeoDataFrame
    gdf['cluster'] = clusters
    
    # Get cluster statistics
    cluster_counts = gdf['cluster'].value_counts().sort_index()
    logger.info("Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        logger.info(f"  Cluster {cluster_id}: {count} hexagons ({count/len(gdf)*100:.1f}%)")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define colors for each cluster (tab10 colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot each cluster
    for cluster_id in range(10):
        cluster_gdf = gdf[gdf['cluster'] == cluster_id]
        if len(cluster_gdf) > 0:
            cluster_gdf.plot(
                ax=ax,
                color=colors[cluster_id],
                edgecolor='none',  # No outlines
                linewidth=0,
                alpha=0.8
            )
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=colors[i], label=f'Cluster {i} ({cluster_counts.get(i, 0):,})')
        for i in range(10)
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fancybox=True,
        shadow=True,
        title='Clusters'
    )
    
    # Customize the plot
    ax.set_title('AlphaEarth Cascadia 2021 - K-means Clustering (k=10)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add grid for reference
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Set axis limits
    ax.set_xlim(gdf.total_bounds[0] - 0.1, gdf.total_bounds[2] + 0.1)
    ax.set_ylim(gdf.total_bounds[1] - 0.1, gdf.total_bounds[3] + 0.1)
    
    # Equal aspect ratio for proper map display
    ax.set_aspect('equal')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("plots/alphaearth_2021")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "alphaearth_kmeans_k10_simple.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved visualization to {output_path}")
    
    # Save high-resolution version
    output_path_hires = output_dir / "alphaearth_kmeans_k10_simple_hires.png"
    plt.savefig(output_path_hires, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved high-resolution version to {output_path_hires}")
    
    # Clean up
    plt.close()
    
    # Save cluster assignments
    cluster_df = gdf[['h3_index', 'cluster']].copy()
    cluster_path = data_dir / "cluster_assignments_k10.parquet"
    cluster_df.to_parquet(cluster_path, index=False)
    logger.info(f"Saved cluster assignments to {cluster_path}")
    
    # Save cluster statistics
    stats = {
        'n_hexagons': len(gdf),
        'n_clusters': 10,
        'cluster_sizes': cluster_counts.to_dict(),
        'bounds': {
            'west': float(gdf.total_bounds[0]),
            'south': float(gdf.total_bounds[1]),
            'east': float(gdf.total_bounds[2]),
            'north': float(gdf.total_bounds[3])
        }
    }
    
    import json
    stats_path = data_dir / "clustering_stats_k10.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")
    
    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()