#!/usr/bin/env python
"""
Simple Multi-Modal Visualization Generator
Creates clean cluster maps from the already processed multi-modal data
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
import time

def create_simple_cluster_maps():
    """Create simple, clean cluster visualizations."""
    print("Creating simple multi-modal cluster visualizations...")
    
    # Load the clustered multi-modal data (should exist from previous run)
    try:
        clustered_path = 'data/processed/multimodal/netherlands_multimodal_res10_pca16_clustered.parquet'
        df = pd.read_parquet(clustered_path)
        print(f"Loaded clustered data: {df.shape}")
    except:
        print("Clustered data not found, using original multi-modal data...")
        df = pd.read_parquet('data/processed/multimodal/netherlands_multimodal_res10.parquet')
        print(f"Loaded data: {df.shape}")
        return
    
    # Load geometry
    geometry_gdf = gpd.read_parquet('data/processed/embeddings/alphaearth/alphaearth_embeddings_res10.parquet')
    geometry_gdf = geometry_gdf[['h3_index', 'geometry']].set_index('h3_index')
    
    # Align geometry
    aligned_geometry = geometry_gdf.loc[df['h3_index']].reset_index()
    viz_gdf = gpd.GeoDataFrame(aligned_geometry, geometry='geometry')
    
    # Add cluster columns to viz_gdf
    cluster_cols = [col for col in df.columns if col.startswith('cluster_k')]
    for col in cluster_cols:
        viz_gdf[col] = df[col].values
    
    print(f"Geometry aligned: {len(viz_gdf):,} hexagons")
    print(f"Cluster columns: {cluster_cols}")
    
    # Convert to Dutch RD
    viz_gdf_rd = viz_gdf.to_crs('EPSG:28992')
    
    # Create output directory
    output_dir = Path('results/plots/multimodal')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate maps for each K value
    for col in cluster_cols:
        k = col.split('_k')[1]
        print(f"Creating map for K={k}...")
        
        fig, ax = plt.subplots(figsize=(16, 20), dpi=150)
        
        # Simple cluster plot
        viz_gdf_rd.plot(
            column=col,
            ax=ax,
            cmap='Set3' if int(k) <= 10 else 'tab20',
            edgecolor='none',
            linewidth=0,
            alpha=0.85
        )
        
        # Clean title and formatting
        ax.set_title(
            f'Netherlands Multi-Modal Urban Clustering (K={k})\n'
            f'AlphaEarth + POI + Roads | {len(viz_gdf):,} H3 hexagons',
            fontsize=16, fontweight='bold', pad=20
        )
        
        ax.set_xlabel('X Coordinate (m) - Dutch RD', fontsize=12)
        ax.set_ylabel('Y Coordinate (m) - Dutch RD', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format coordinates
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000):,}k'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y/1000):,}k'))
        
        # Add north arrow
        ax.annotate('N↑', xy=(0.95, 0.95), xycoords='axes fraction',
                   fontsize=20, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Attribution
        ax.text(0.02, 0.02, 
               f'Multi-Modal Urban Analysis | AlphaEarth + OSM POI + Highway2Vec\n'
               f'Processing: UrbanRepML | Generated: {time.strftime("%Y-%m-%d")}',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save
        output_file = output_dir / f'netherlands_multimodal_k{k}_simple.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved: {output_file}")
    
    print(f"\n🎨 Generated {len(cluster_cols)} visualizations in {output_dir}")

if __name__ == "__main__":
    create_simple_cluster_maps()