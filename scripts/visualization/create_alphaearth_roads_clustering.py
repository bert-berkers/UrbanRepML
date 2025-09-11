#!/usr/bin/env python
"""
Efficient Two-Modal Clustering: AlphaEarth + Roads
PCA-optimized K=12 clustering with multiple colormap perspectives
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import time
import warnings
import os
warnings.filterwarnings('ignore')

# Optimize for multi-core performance
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

def load_and_fuse_data():
    """Load AlphaEarth + Roads and fuse into single dataset."""
    print("Loading AlphaEarth and Roads embeddings...")
    
    # Load AlphaEarth (full 66D - no PCA yet)
    print("  Loading AlphaEarth 66D embeddings...")
    alphaearth_df = pd.read_parquet('data/processed/embeddings/alphaearth/alphaearth_embeddings_res10.parquet')
    alphaearth_df = alphaearth_df.set_index('h3_index')
    
    # Remove metadata columns
    metadata_cols = ['h3_resolution', 'geometry']
    alphaearth_features = alphaearth_df.drop(columns=[col for col in metadata_cols if col in alphaearth_df.columns])
    print(f"    AlphaEarth: {alphaearth_features.shape} features")
    
    # Load Roads (64D Highway2Vec)
    print("  Loading Roads 64D Highway2Vec embeddings...")
    roads_df = pd.read_parquet('data/processed/embeddings/roads/roads_embeddings_res10.parquet')
    roads_df = roads_df.set_index('h3_index')
    
    # Remove metadata columns
    roads_features = roads_df.drop(columns=[col for col in metadata_cols if col in roads_df.columns])
    print(f"    Roads: {roads_features.shape} features")
    
    # Find intersection of hexagons
    print("  Finding common hexagons...")
    common_hexagons = alphaearth_features.index.intersection(roads_features.index)
    print(f"    Common hexagons: {len(common_hexagons):,}")
    
    # Align data
    alphaearth_aligned = alphaearth_features.loc[common_hexagons]
    roads_aligned = roads_features.loc[common_hexagons]
    
    # Add prefixes and concatenate
    alphaearth_aligned = alphaearth_aligned.add_prefix('ae_')
    roads_aligned = roads_aligned.add_prefix('roads_')
    
    # Fuse datasets
    print("  Fusing datasets...")
    fused_df = pd.concat([alphaearth_aligned, roads_aligned], axis=1)
    print(f"    Fused shape: {fused_df.shape} ({fused_df.shape[1]} total features)")
    
    return fused_df

def apply_pca_reduction(df, n_components=32):
    """Apply PCA to reduce dimensionality for efficient clustering."""
    print(f"Applying PCA reduction: {df.shape[1]}D -> {n_components}D...")
    
    # Standardize features
    print("  Standardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df.values.astype(np.float32))
    
    # Apply PCA
    print("  Computing PCA transformation...")
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    
    # Create PCA DataFrame
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(features_pca, columns=pca_cols, index=df.index)
    
    # Report PCA statistics
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"  PCA Results:")
    print(f"    Explained variance: {explained_var.sum():.3f}")
    print(f"    Top 5 components: {explained_var[:5].round(3)}")
    print(f"    Components for 90% variance: {np.where(cumulative_var >= 0.90)[0][0] + 1}")
    
    return pca_df, pca, explained_var

def run_clustering(pca_df, k=12):
    """Run efficient K=12 clustering on PCA features."""
    print(f"Running MiniBatchKMeans clustering (K={k})...")
    
    features = pca_df.values.astype(np.float32)
    
    # Optimized MiniBatchKMeans for large dataset
    print("  Initializing clustering...")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=50000,  # Large batches for 1M+ hexagons
        max_iter=200,      # More iterations for quality
        n_init=10,         # Multiple initializations  
        init='k-means++',
        verbose=1
    )
    
    print("  Fitting clusters...")
    start_time = time.time()
    clusters = kmeans.fit_predict(features)
    clustering_time = time.time() - start_time
    
    # Calculate silhouette score on sample
    print("  Calculating silhouette score...")
    sample_size = min(100000, len(features))
    sample_idx = np.random.choice(len(features), sample_size, replace=False)
    silhouette = silhouette_score(features[sample_idx], clusters[sample_idx])
    
    print(f"  Clustering completed in {clustering_time:.1f}s")
    print(f"  Silhouette score: {silhouette:.3f}")
    
    # Add clusters to dataframe
    result_df = pca_df.copy()
    result_df['cluster_k12'] = clusters
    
    return result_df, kmeans, silhouette

def create_multicolor_visualizations(clustered_df):
    """Generate multiple colormap visualizations of K=12 clustering."""
    print("Creating multiple colormap visualizations...")
    
    # Load geometry for visualization
    print("  Loading H3 geometry...")
    geometry_gdf = gpd.read_parquet('data/processed/embeddings/alphaearth/alphaearth_embeddings_res10.parquet')
    geometry_gdf = geometry_gdf[['h3_index', 'geometry']].set_index('h3_index')
    
    # Align geometry with clustered data
    aligned_geometry = geometry_gdf.loc[clustered_df.index].reset_index()
    viz_gdf = gpd.GeoDataFrame(aligned_geometry, geometry='geometry')
    viz_gdf['cluster_k12'] = clustered_df['cluster_k12'].values
    
    print(f"  Geometry aligned: {len(viz_gdf):,} hexagons")
    
    # Convert to Dutch RD for proper visualization
    print("  Converting to Dutch RD projection...")
    viz_gdf_rd = viz_gdf.to_crs('EPSG:28992')
    
    # Create output directory
    output_dir = Path('results/plots/alphaearth_roads')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define colormap perspectives
    colormaps = {
        'viridis': 'Scientific heat-map style',
        'Set3': 'Categorical distinct colors',  
        'tab20': 'Maximum cluster separation',
        'plasma': 'High contrast urban patterns',
        'Spectral': 'Geographic rainbow spectrum'
    }
    
    print(f"  Generating {len(colormaps)} different colormap visualizations...")
    
    for cmap_name, description in colormaps.items():
        print(f"    Creating {cmap_name} visualization...")
        
        fig, ax = plt.subplots(figsize=(16, 20), dpi=150)
        
        # Plot clusters
        viz_gdf_rd.plot(
            column='cluster_k12',
            ax=ax,
            cmap=cmap_name,
            edgecolor='none',
            linewidth=0,
            alpha=0.8
        )
        
        # Enhanced title
        ax.set_title(
            f'Netherlands AlphaEarth + Roads Clustering (K=12)\n'
            f'{description} | {len(viz_gdf):,} H3 hexagons | 130D->32D PCA',
            fontsize=16, fontweight='bold', pad=20
        )
        
        ax.set_xlabel('X Coordinate (m) - Dutch RD', fontsize=12)
        ax.set_ylabel('Y Coordinate (m) - Dutch RD', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format coordinates  
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000):,}k'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y/1000):,}k'))
        
        # North arrow
        ax.annotate('N↑', xy=(0.95, 0.95), xycoords='axes fraction',
                   fontsize=20, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Attribution
        ax.text(0.02, 0.02, 
               f'Two-Modal Analysis: AlphaEarth 2022 + Highway2Vec Roads\n'
               f'Colormap: {cmap_name} | Generated: {time.strftime("%Y-%m-%d")}',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save visualization
        output_file = output_dir / f'netherlands_alphaearth_roads_k12_{cmap_name.lower()}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"      Saved: {output_file}")
    
    return output_dir

def main():
    """Main execution pipeline."""
    start_time = time.time()
    print("=" * 80)
    print("EFFICIENT TWO-MODAL CLUSTERING: ALPHAEARTH + ROADS")
    print("=" * 80)
    print(f"System: {os.cpu_count()} CPU cores available")
    
    # Phase 1: Load and fuse data
    fused_df = load_and_fuse_data()
    
    # Phase 2: Apply PCA reduction
    pca_df, pca_model, explained_var = apply_pca_reduction(fused_df, n_components=32)
    
    # Phase 3: Run K=12 clustering
    clustered_df, kmeans_model, silhouette = run_clustering(pca_df, k=12)
    
    # Save clustered data immediately (checkpoint!)
    print("Saving clustered dataset...")
    output_path = 'data/processed/multimodal/alphaearth_roads_k12_clustered.parquet'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clustered_df.to_parquet(output_path)
    print(f"  Saved: {output_path}")
    
    # Phase 4: Create visualizations
    output_dir = create_multicolor_visualizations(clustered_df)
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("TWO-MODAL CLUSTERING COMPLETE!")
    print("=" * 80)
    print(f"📊 Processed: {len(fused_df):,} H3 hexagons")
    print(f"🔀 Features: 130D -> 32D PCA ({explained_var.sum():.3f} variance retained)")
    print(f"🎯 Clustering: K=12 (Silhouette: {silhouette:.3f})")
    print(f"🎨 Visualizations: 5 colormap perspectives")  
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"📁 Results: {output_dir}")
    print(f"💾 Data saved: {output_path}")

if __name__ == "__main__":
    main()