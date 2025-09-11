#!/usr/bin/env python3
"""
Roads Network Embeddings Visualization for Netherlands at Resolution 10
Creates visualizations for Highway2Vec embeddings using SRAI spatial framework.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import argparse
import json
import sys
from typing import Tuple, Optional, Dict

try:
    import srai
    from srai.regionalizers import H3Regionalizer
    import geopandas as gpd
    from shapely.geometry import box
    import h3
    SRAI_AVAILABLE = True
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install srai geopandas h3")
    SRAI_AVAILABLE = False

def load_roads_embeddings(data_path: str = None) -> pd.DataFrame:
    """Load roads network embeddings from parquet file."""
    if data_path is None:
        data_path = "data/processed/embeddings/roads/roads_embeddings_res10.parquet"
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Roads embeddings not found at {data_path}")
    
    print(f"Loading roads embeddings from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Ensure h3_index is the index
    if 'h3_index' in df.columns:
        df = df.set_index('h3_index')
    
    print(f"  Loaded {len(df):,} road embeddings")
    print(f"  Embedding dimensions: {df.shape[1] - 1}")  # Minus h3_resolution column
    print(f"  Columns: {df.columns.tolist()}")
    
    return df

def create_spatial_framework(h3_indices: list, resolution: int = 10) -> gpd.GeoDataFrame:
    """Create spatial framework from H3 indices using SRAI."""
    print(f"Creating spatial framework for {len(h3_indices):,} H3 indices at resolution {resolution}...")
    
    # Create H3Regionalizer
    regionalizer = H3Regionalizer(resolution=resolution)
    
    # Get bounding box from H3 indices
    lats, lngs = [], []
    valid_indices = []
    
    for h3_idx in h3_indices:
        try:
            lat, lng = h3.cell_to_latlng(h3_idx)
            lats.append(lat)
            lngs.append(lng)
            valid_indices.append(h3_idx)
        except Exception as e:
            print(f"  Warning: Invalid H3 index {h3_idx}: {e}")
            continue
    
    if not lats:
        raise ValueError("No valid H3 indices found")
    
    print(f"  Valid H3 indices: {len(valid_indices):,}")
    
    # Create bounding box with small padding
    bounds = box(
        min(lngs) - 0.01,
        min(lats) - 0.01,
        max(lngs) + 0.01,
        max(lats) + 0.01
    )
    
    # Generate spatial framework
    study_area = gpd.GeoDataFrame([1], geometry=[bounds], crs='EPSG:4326')
    regions_gdf = regionalizer.transform(study_area)
    
    print(f"  Generated {len(regions_gdf):,} total regions in bounds")
    
    # Filter to our specific indices
    valid_set = set(valid_indices)
    regions_filtered = regions_gdf[regions_gdf.index.isin(valid_set)]
    
    print(f"  Filtered to {len(regions_filtered):,} matching regions")
    return regions_filtered

def perform_clustering(gdf: gpd.GeoDataFrame, embeddings_df: pd.DataFrame, 
                      n_clusters: int = 8, embedding_cols: list = None) -> gpd.GeoDataFrame:
    """Perform clustering on Highway2Vec embeddings."""
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    # Get embedding columns (Highway2Vec produces numbered columns)
    if embedding_cols is None:
        # Highway2Vec creates columns like 0, 1, 2, ... (numeric indices)
        numeric_cols = [col for col in embeddings_df.columns 
                       if isinstance(col, (int, str)) and str(col).isdigit()]
        
        # If no numeric columns, try to find embedding-like columns
        if not numeric_cols:
            embedding_cols = [col for col in embeddings_df.columns 
                            if col not in ['h3_resolution'] and col != embeddings_df.index.name]
        else:
            embedding_cols = numeric_cols
    
    if not embedding_cols:
        raise ValueError(f"No embedding columns found. Available columns: {embeddings_df.columns.tolist()}")
    
    print(f"  Using embedding columns: {embedding_cols[:10]}{'...' if len(embedding_cols) > 10 else ''}")
    print(f"  Total embedding dimensions: {len(embedding_cols)}")
    
    # Align indices
    common_indices = gdf.index.intersection(embeddings_df.index)
    print(f"  Common indices: {len(common_indices):,}")
    
    if len(common_indices) == 0:
        raise ValueError("No common indices between spatial and embedding data")
    
    # Get embeddings for common indices
    X = embeddings_df.loc[common_indices, embedding_cols].values
    
    # Handle NaN values
    if np.isnan(X).any():
        print("  Handling NaN values...")
        for i in range(X.shape[1]):
            col_median = np.nanmedian(X[:, i])
            X[np.isnan(X[:, i]), i] = col_median
    
    # Standardize and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create result GeoDataFrame
    result_gdf = gdf.loc[common_indices].copy()
    result_gdf['cluster'] = clusters
    
    # Add embedding statistics
    result_gdf['embedding_norm'] = np.linalg.norm(X, axis=1)
    result_gdf['embedding_mean'] = np.mean(X, axis=1)
    result_gdf['embedding_std'] = np.std(X, axis=1)
    
    # Add some key embeddings for analysis
    for i, col in enumerate(embedding_cols[:5]):  # First 5 dimensions
        result_gdf[f'embed_{i}'] = embeddings_df.loc[common_indices, col]
    
    # Print cluster distribution
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("  Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"    Cluster {cluster_id}: {count:,} hexagons ({count/len(clusters)*100:.1f}%)")
    
    return result_gdf

def create_clustering_plot(gdf: gpd.GeoDataFrame, output_path: Path, 
                          title: str = "Roads Network Embeddings Clustering",
                          figsize: Tuple[int, int] = (16, 12)):
    """Create clustering visualization."""
    print(f"Creating clustering plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    
    # Get unique clusters and colors
    unique_clusters = sorted(gdf['cluster'].unique())
    n_clusters = len(unique_clusters)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = gdf[gdf['cluster'] == cluster_id]
        cluster_data.plot(
            ax=ax,
            color=colors[i],
            alpha=0.8,
            edgecolor='white',
            linewidth=0.1,
            label=f'Cluster {cluster_id} ({len(cluster_data):,})'
        )
    
    # Styling
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nHighway2Vec Embeddings (K-means, K={n_clusters})',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    
    # Legend
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.9)
    legend.set_title('Road Clusters', prop={'weight': 'bold'})
    
    # Add metadata
    bounds = gdf.total_bounds
    embedding_stats = f"Avg norm: {gdf['embedding_norm'].mean():.3f} ± {gdf['embedding_norm'].std():.3f}"
    area_info = f"Spatial extent: {bounds[2]-bounds[0]:.3f}° × {bounds[3]-bounds[1]:.3f}°"
    
    metadata_text = f"{len(gdf):,} hexagons | {n_clusters} clusters | {area_info}\n{embedding_stats}"
    ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved clustering plot: {output_path}")

def create_embedding_analysis_plot(gdf: gpd.GeoDataFrame, output_path: Path,
                                  figsize: Tuple[int, int] = (20, 12)):
    """Create multi-panel analysis of embeddings."""
    print("Creating embedding analysis plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=figsize, facecolor='white')
    axes = axes.flatten()
    
    # 1. Embedding norm (magnitude)
    im1 = gdf.plot(column='embedding_norm', ax=axes[0], cmap='viridis',
                   legend=True, legend_kwds={'shrink': 0.8, 'aspect': 20})
    axes[0].set_title('Embedding Magnitude\n(L2 Norm)', fontsize=12, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # 2. Embedding mean
    gdf.plot(column='embedding_mean', ax=axes[1], cmap='RdBu_r',
             legend=True, legend_kwds={'shrink': 0.8, 'aspect': 20})
    axes[1].set_title('Embedding Mean\n(Activation Level)', fontsize=12, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # 3. Embedding std
    gdf.plot(column='embedding_std', ax=axes[2], cmap='plasma',
             legend=True, legend_kwds={'shrink': 0.8, 'aspect': 20})
    axes[2].set_title('Embedding Std\n(Variability)', fontsize=12, fontweight='bold')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    # 4-6. First 3 embedding dimensions
    embed_cols = [col for col in gdf.columns if col.startswith('embed_')][:3]
    cmaps = ['coolwarm', 'seismic', 'bwr']
    
    for i, (col, cmap) in enumerate(zip(embed_cols, cmaps)):
        ax_idx = i + 3
        gdf.plot(column=col, ax=axes[ax_idx], cmap=cmap,
                 legend=True, legend_kwds={'shrink': 0.8, 'aspect': 20})
        axes[ax_idx].set_title(f'Embedding Dim {i}\n(Highway2Vec Feature)', 
                              fontsize=12, fontweight='bold')
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])
    
    # Overall styling
    for ax in axes:
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    fig.suptitle('Roads Network Embeddings Analysis: Highway2Vec Features\n'
                 'Netherlands H3 Resolution 10', fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved analysis plot: {output_path}")

def create_pca_plot(gdf: gpd.GeoDataFrame, embeddings_df: pd.DataFrame, 
                   output_path: Path, n_components: int = 3,
                   figsize: Tuple[int, int] = (20, 6)):
    """Create PCA visualization of embeddings."""
    print(f"Creating PCA plot with {n_components} components...")
    
    # Get embedding columns
    numeric_cols = [col for col in embeddings_df.columns 
                   if isinstance(col, (int, str)) and str(col).isdigit()]
    if not numeric_cols:
        embedding_cols = [col for col in embeddings_df.columns 
                        if col not in ['h3_resolution'] and col != embeddings_df.index.name]
    else:
        embedding_cols = numeric_cols
    
    # Get common indices
    common_indices = gdf.index.intersection(embeddings_df.index)
    X = embeddings_df.loc[common_indices, embedding_cols].values
    
    # Handle NaN
    if np.isnan(X).any():
        for i in range(X.shape[1]):
            col_median = np.nanmedian(X[:, i])
            X[np.isnan(X[:, i]), i] = col_median
    
    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Add PCA components to GeoDataFrame
    pca_gdf = gdf.loc[common_indices].copy()
    for i in range(n_components):
        pca_gdf[f'pca_{i}'] = X_pca[:, i]
    
    # Create subplots
    fig, axes = plt.subplots(1, n_components, figsize=figsize, facecolor='white')
    if n_components == 1:
        axes = [axes]
    
    # Plot each PCA component
    for i in range(n_components):
        pca_gdf.plot(column=f'pca_{i}', ax=axes[i], cmap='RdBu_r',
                     legend=True, legend_kwds={'shrink': 0.8, 'aspect': 20})
        
        var_explained = pca.explained_variance_ratio_[i]
        axes[i].set_title(f'PCA Component {i+1}\n({var_explained:.1%} variance)', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_aspect('equal')
        
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    
    total_variance = pca.explained_variance_ratio_.sum()
    fig.suptitle(f'Roads Embeddings PCA Analysis\n'
                f'Total Variance Explained: {total_variance:.1%}', 
                fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved PCA plot: {output_path}")
    return pca.explained_variance_ratio_

def main():
    """Main visualization pipeline for roads network embeddings."""
    parser = argparse.ArgumentParser(description='Visualize Roads Network Embeddings (Highway2Vec)')
    parser.add_argument('--data-path', type=str, 
                       default='data/processed/embeddings/roads/roads_embeddings_res10.parquet',
                       help='Path to roads embeddings parquet file')
    parser.add_argument('--output-dir', type=str, default='scripts/visualizations/output',
                       help='Output directory for plots')
    parser.add_argument('--clusters', type=int, default=8,
                       help='Number of clusters for K-means (default: 8)')
    parser.add_argument('--pca-components', type=int, default=3,
                       help='Number of PCA components to visualize (default: 3)')
    parser.add_argument('--figsize', type=str, default='16,12',
                       help='Figure size as "width,height" (default: 16,12)')
    
    args = parser.parse_args()
    
    if not SRAI_AVAILABLE:
        print("ERROR: SRAI and required dependencies not available!")
        sys.exit(1)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        figsize = tuple(map(int, args.figsize.split(',')))
    except:
        figsize = (16, 12)
    
    print("Roads Network Embeddings Visualization")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Clusters: {args.clusters}")
    print(f"PCA components: {args.pca_components}")
    print()
    
    try:
        # Load data
        roads_df = load_roads_embeddings(args.data_path)
        
        # Create spatial framework
        spatial_gdf = create_spatial_framework(roads_df.index.tolist(), resolution=10)
        
        # Perform clustering
        clustered_gdf = perform_clustering(spatial_gdf, roads_df, args.clusters)
        
        # Create visualizations
        
        # 1. Clustering plot
        clustering_path = output_dir / "netherlands_roads_res10_clustering.png"
        create_clustering_plot(clustered_gdf, clustering_path, 
                              "Netherlands Roads Network (H3 Res 10)", figsize)
        
        # 2. Embedding analysis plot
        analysis_path = output_dir / "netherlands_roads_res10_analysis.png"
        create_embedding_analysis_plot(clustered_gdf, analysis_path, (20, 12))
        
        # 3. PCA plot
        pca_path = output_dir / "netherlands_roads_res10_pca.png"
        pca_variance = create_pca_plot(clustered_gdf, roads_df, pca_path, 
                                      args.pca_components, (20, 6))
        
        # Save statistics
        stats = {
            'total_hexagons': len(clustered_gdf),
            'embedding_dimensions': roads_df.shape[1] - 1,
            'clusters': args.clusters,
            'pca_components': args.pca_components,
            'pca_variance_explained': pca_variance.tolist(),
            'embedding_statistics': {
                'norm_mean': float(clustered_gdf['embedding_norm'].mean()),
                'norm_std': float(clustered_gdf['embedding_norm'].std()),
                'mean_mean': float(clustered_gdf['embedding_mean'].mean()),
                'mean_std': float(clustered_gdf['embedding_mean'].std()),
            }
        }
        
        stats_path = output_dir / "netherlands_roads_res10_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n" + "=" * 50)
        print("SUCCESS: Roads visualization pipeline completed!")
        print(f"Generated plots:")
        print(f"  - Clustering: {clustering_path}")
        print(f"  - Analysis: {analysis_path}")
        print(f"  - PCA: {pca_path}")
        print(f"  - Statistics: {stats_path}")
        print(f"Total hexagons visualized: {len(clustered_gdf):,}")
        
    except Exception as e:
        print(f"\nERROR in visualization pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()