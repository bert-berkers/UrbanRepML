#!/usr/bin/env python
"""
Optimized K-means clustering for PCA-reduced resolution 10 AlphaEarth data
Uses MiniBatchKMeans for speed and all available CPU cores
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import time
from joblib import Parallel, delayed
import warnings
import os
warnings.filterwarnings('ignore')

# Set environment variables for maximum performance
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

def main():
    print(f"System has {os.cpu_count()} CPU cores available")
    print("Loading PCA-reduced data...")
    
    # Load PCA data and original geometry
    pca_df = pd.read_parquet('data/processed/embeddings/alphaearth/netherlands_res10_2022_pca16.parquet')
    original_gdf = gpd.read_parquet('data/processed/embeddings/alphaearth/netherlands_res10_2022.parquet')
    
    print(f"Loaded {len(pca_df):,} hexagons")
    
    # Extract PCA embedding columns
    pca_cols = [col for col in pca_df.columns if col.startswith('PC')]
    embeddings = pca_df[pca_cols].values.astype(np.float32)  # Use float32 for speed
    print(f"PCA embedding shape: {embeddings.shape}")
    
    # Function to run MiniBatchKMeans in parallel
    def cluster_and_score(k, embeddings):
        print(f"Starting MiniBatchKMeans with K={k}...")
        start_time = time.time()
        
        # Use MiniBatchKMeans for much faster clustering on large datasets
        # batch_size of 10000 provides good balance between speed and quality
        kmeans = MiniBatchKMeans(
            n_clusters=k, 
            random_state=42, 
            batch_size=10000,
            max_iter=100,
            n_init=3,  # Fewer initializations for speed
            init='k-means++',
            verbose=0
        )
        
        clusters = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score on larger sample for better accuracy
        sample_size = min(100000, len(embeddings))
        sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
        silhouette = silhouette_score(embeddings[sample_idx], clusters[sample_idx])
        
        clustering_time = time.time() - start_time
        print(f"  K={k} completed in {clustering_time:.1f}s, Silhouette: {silhouette:.3f}")
        return k, clusters, silhouette
    
    # Apply K-means clustering in parallel using all cores
    k_values = [8, 10, 12]
    print(f"Running MiniBatchKMeans clustering in parallel on {len(k_values)} K values...")
    print(f"Using {os.cpu_count()-1} parallel jobs")
    
    # Run clustering in parallel using process-based parallelism for better CPU utilization
    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(cluster_and_score)(k, embeddings) for k in k_values
    )
    
    # Store results
    cluster_results = {}
    for k, clusters, silhouette in results:
        cluster_results[k] = {'clusters': clusters, 'silhouette': silhouette}
    
    print("\nCreating visualizations...")
    
    # Create visualizations
    for k in k_values:
        clusters = cluster_results[k]['clusters']
        silhouette = cluster_results[k]['silhouette']
        print(f"Creating visualization for K={k} (Silhouette: {silhouette:.3f})...")
        
        # Create GeoDataFrame with geometry and clusters
        viz_gdf = original_gdf[['h3_index', 'geometry']].copy()
        viz_gdf[f'cluster_k{k}'] = clusters
        
        # Convert to Dutch RD for visualization
        viz_gdf_rd = viz_gdf.to_crs('EPSG:28992')
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 16), dpi=150)  # Reduced DPI for faster rendering
        
        # Plot clusters without hexagon outlines
        viz_gdf_rd.plot(
            column=f'cluster_k{k}',
            ax=ax,
            cmap='tab10' if k <= 10 else 'tab20',  # Better colormaps for clustering
            edgecolor='none',
            linewidth=0,
            alpha=0.9
        )
        
        # Styling
        ax.set_title(f'AlphaEarth Netherlands 2022 - MiniBatchKMeans (K={k})\nH3 Resolution 10 | PCA 16D | {len(viz_gdf):,} hexagons | Silhouette: {silhouette:.3f}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate (m) - Dutch RD', fontsize=12)
        ax.set_ylabel('Y Coordinate (m) - Dutch RD', fontsize=12)
        
        # Add coordinate grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='plain', axis='both')
        
        # Format axis labels with commas
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y):,}'))
        
        # Add north arrow
        ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=16, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction',
                    fontsize=20, ha='center', va='center')
        
        # Add scale bar
        ax.text(0.02, 0.02, '50 km', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                fontsize=10)
        
        # Add attribution
        ax.text(0.98, 0.02, 'AlphaEarth 2022 | Processing: UrbanRepML | MiniBatchKMeans',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path('results/plots/netherlands') / f'alphaearth_clusters_k{k}_res10_pca16_2022.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved: {output_file}")
    
    # Save final clustered data to data folder
    print("\nSaving final clustered data...")
    final_df = pca_df.copy()
    for k in k_values:
        final_df[f'cluster_k{k}'] = cluster_results[k]['clusters']
    
    output_path = 'data/processed/embeddings/alphaearth/netherlands_res10_2022_pca16_clustered.parquet'
    final_df.to_parquet(output_path)
    print(f"Saved clustered data to: {output_path}")
    
    print("\nResolution 10 PCA clustering complete!")
    print(f"Total processing embeddings time: {time.time():.1f} seconds")

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total script runtime: {time.time() - start:.1f} seconds")