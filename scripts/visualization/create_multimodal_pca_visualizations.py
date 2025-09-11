#!/usr/bin/env python
"""
Enhanced Multi-Modal PCA Clustering and Visualization for Netherlands
Combines AlphaEarth + POI + Roads embeddings (108D) with RTX 3090 optimized clustering
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time
from joblib import Parallel, delayed
import warnings
import os
import json
warnings.filterwarnings('ignore')

# Set environment variables for maximum performance
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

def load_multimodal_data():
    """Load the fused multi-modal dataset."""
    print("Loading multi-modal dataset...")
    
    # Load fused multi-modal data
    data_path = Path('data/processed/multimodal/netherlands_multimodal_res10.parquet')
    metadata_path = Path('data/processed/multimodal/netherlands_multimodal_res10.metadata.json')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Multi-modal data not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} hexagons with {df.shape[1]-1} features")
    
    # Load metadata for feature groups
    feature_groups = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            feature_groups = metadata.get('feature_groups', {})
            print(f"Feature groups: {list(feature_groups.keys())}")
    
    return df, feature_groups

def apply_pca_reduction(df, n_components=16):
    """Apply PCA to reduce multi-modal features."""
    print(f"Applying PCA reduction: {df.shape[1]-1} -> {n_components} components...")
    
    # Extract feature columns (exclude h3_index)
    feature_cols = [col for col in df.columns if col != 'h3_index']
    features = df[feature_cols].values.astype(np.float32)
    
    # Additional standardization (data is already normalized per modality)
    print("Applying final standardization...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    print("Computing PCA transformation...")
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    
    # Create PCA DataFrame
    pca_cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(features_pca, columns=pca_cols, index=df.index)
    pca_df['h3_index'] = df['h3_index'].values
    
    # Print PCA statistics
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"PCA Results:")
    print(f"  - Components: {n_components}")
    print(f"  - Explained variance: {explained_var.sum():.3f}")
    print(f"  - Top 5 components: {explained_var[:5]}")
    print(f"  - Cumulative variance: {cumulative_var[-1]:.3f}")
    
    return pca_df, pca, explained_var

def cluster_and_score(k, embeddings):
    """Run MiniBatchKMeans clustering with performance optimization."""
    print(f"Starting MiniBatchKMeans with K={k}...")
    start_time = time.time()
    
    # RTX 3090 optimized parameters
    kmeans = MiniBatchKMeans(
        n_clusters=k, 
        random_state=42, 
        batch_size=20000,  # Larger batch for 1M+ hexagons
        max_iter=150,      # More iterations for better convergence
        n_init=5,          # More initializations for quality
        init='k-means++',
        verbose=0
    )
    
    clusters = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette score on sample for speed
    sample_size = min(200000, len(embeddings))
    sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
    silhouette = silhouette_score(embeddings[sample_idx], clusters[sample_idx])
    
    clustering_time = time.time() - start_time
    print(f"  K={k} completed in {clustering_time:.1f}s, Silhouette: {silhouette:.3f}")
    
    return k, clusters, silhouette, kmeans

def create_visualization(viz_gdf, k, clusters, silhouette, feature_groups, output_dir):
    """Create enhanced visualization with multi-modal attribution."""
    print(f"Creating visualization for K={k}...")
    
    # Create figure with subplots for detailed analysis
    fig = plt.figure(figsize=(20, 12), dpi=150)
    
    # Main cluster map
    ax_main = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    
    # Convert to Dutch RD for visualization
    viz_gdf_rd = viz_gdf.to_crs('EPSG:28992')
    
    # Plot main cluster map
    viz_gdf_rd.plot(
        column=f'cluster_k{k}',
        ax=ax_main,
        cmap='Set3' if k <= 10 else 'tab20',
        edgecolor='none',
        linewidth=0,
        alpha=0.8,
        legend=True
    )
    
    # Enhanced title with multi-modal info
    ax_main.set_title(
        f'Netherlands Multi-Modal Urban Clustering (K={k})\n'
        f'AlphaEarth + POI + Roads | H3 Resolution 10 | {len(viz_gdf):,} hexagons\n'
        f'Silhouette Score: {silhouette:.3f} | Features: 108→16D PCA',
        fontsize=14, fontweight='bold', pad=20
    )
    
    ax_main.set_xlabel('X Coordinate (m) - Dutch RD', fontsize=10)
    ax_main.set_ylabel('Y Coordinate (m) - Dutch RD', fontsize=10)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.ticklabel_format(style='plain', axis='both')
    
    # Format axis labels
    ax_main.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000):,}k'))
    ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y/1000):,}k'))
    
    # Add north arrow and scale
    ax_main.annotate('N↑', xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=16, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Cluster statistics subplot
    ax_stats = plt.subplot2grid((2, 3), (0, 2))
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    
    # Pie chart of cluster sizes
    ax_stats.pie(cluster_counts, labels=[f'C{i}' for i in unique_clusters], autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 8})
    ax_stats.set_title(f'Cluster Distribution\n(K={k})', fontsize=10, fontweight='bold')
    
    # Feature contribution subplot
    ax_features = plt.subplot2grid((2, 3), (1, 2))
    
    # Create feature group contribution chart
    modality_counts = {
        'AlphaEarth': 16,
        'POI': 28, 
        'Roads': 64
    }
    
    bars = ax_features.bar(modality_counts.keys(), modality_counts.values(), 
                          color=['green', 'blue', 'red'], alpha=0.7)
    ax_features.set_title('Feature Contribution\nby Modality', fontsize=10, fontweight='bold')
    ax_features.set_ylabel('Features')
    
    # Add value labels on bars
    for bar, value in zip(bars, modality_counts.values()):
        height = bar.get_height()
        ax_features.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value}D', ha='center', va='bottom', fontsize=8)
    
    # Add comprehensive attribution
    attribution_text = (
        f'Multi-Modal Urban Analysis | Processing: UrbanRepML\n'
        f'Data: AlphaEarth 2022 + OSM POI + Highway2Vec\n'
        f'Algorithm: MiniBatchKMeans | PCA: 108→16D | Generated: {time.strftime("%Y-%m-%d")}'
    )
    
    fig.text(0.02, 0.02, attribution_text, fontsize=8, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    output_file = output_dir / f'netherlands_multimodal_clusters_k{k}_res10.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_file}")
    return output_file

def main():
    """Main execution function."""
    start_time = time.time()
    print("=" * 80)
    print("MULTI-MODAL NETHERLANDS CLUSTERING & VISUALIZATION")
    print("=" * 80)
    print(f"System: {os.cpu_count()} CPU cores available")
    
    # Load multi-modal data
    df, feature_groups = load_multimodal_data()
    
    # Apply PCA reduction
    pca_df, pca_model, explained_variance = apply_pca_reduction(df, n_components=16)
    
    # Extract embeddings for clustering
    pca_cols = [col for col in pca_df.columns if col.startswith('PC')]
    embeddings = pca_df[pca_cols].values.astype(np.float32)
    print(f"Final embedding shape for clustering: {embeddings.shape}")
    
    # Load geometry for visualization
    print("Loading H3 geometry data...")
    # Use one of the original modality files for geometry
    geometry_gdf = gpd.read_parquet('data/processed/embeddings/alphaearth/alphaearth_embeddings_res10.parquet')
    geometry_gdf = geometry_gdf[['h3_index', 'geometry']].set_index('h3_index')
    
    # Align geometry with our multi-modal data
    aligned_geometry = geometry_gdf.loc[pca_df['h3_index']].reset_index()
    viz_gdf = gpd.GeoDataFrame(aligned_geometry, geometry='geometry')
    
    print(f"Geometry aligned: {len(viz_gdf):,} hexagons")
    
    # Run clustering analysis
    k_values = [6, 8, 10, 12, 15]
    print(f"Running clustering with K values: {k_values}")
    
    # Parallel clustering execution
    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(cluster_and_score)(k, embeddings) for k in k_values
    )
    
    # Create output directory
    output_dir = Path('results/plots/multimodal')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process results and create visualizations
    print("\nCreating visualizations...")
    cluster_results = {}
    
    for k, clusters, silhouette, kmeans_model in results:
        cluster_results[k] = {
            'clusters': clusters, 
            'silhouette': silhouette,
            'model': kmeans_model
        }
        
        # Add clusters to visualization dataframe
        viz_gdf[f'cluster_k{k}'] = clusters
        
        # Create visualization
        create_visualization(viz_gdf, k, clusters, silhouette, feature_groups, output_dir)
    
    # Save clustered multi-modal data
    print("\nSaving clustered multi-modal dataset...")
    final_df = pca_df.copy()
    for k in k_values:
        final_df[f'cluster_k{k}'] = cluster_results[k]['clusters']
    
    # Add PCA explained variance info
    final_df.attrs['pca_explained_variance'] = explained_variance.tolist()
    final_df.attrs['pca_cumulative_variance'] = np.cumsum(explained_variance).tolist()
    
    output_path = 'data/processed/multimodal/netherlands_multimodal_res10_pca16_clustered.parquet'
    final_df.to_parquet(output_path)
    print(f"Saved clustered data: {output_path}")
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("MULTI-MODAL CLUSTERING COMPLETE!")
    print("=" * 80)
    print(f"📊 Processed: {len(df):,} H3 hexagons")
    print(f"🔀 Features: 108D → 16D PCA ({explained_variance.sum():.3f} variance retained)")
    print(f"🎯 Clustering: {len(k_values)} K-values with silhouette scoring")
    print(f"🎨 Visualizations: {len(k_values)} high-quality maps generated")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"📁 Results: {output_dir}")
    
    # Best silhouette score
    best_k = max(cluster_results.keys(), key=lambda k: cluster_results[k]['silhouette'])
    best_score = cluster_results[best_k]['silhouette']
    print(f"🏆 Best clustering: K={best_k} (Silhouette: {best_score:.3f})")

if __name__ == "__main__":
    main()