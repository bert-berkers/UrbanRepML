#!/usr/bin/env python3
"""
Urban Embedding Visualization using SRAI
Creates visualizations for each resolution plus a combined PCA-aligned plot.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import argparse
import json
import sys
from typing import Dict, List, Tuple, Optional

try:
    import srai
    from srai.regionalizers import H3Regionalizer
    import geopandas as gpd
    from shapely.geometry import box
    SRAI_AVAILABLE = True
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install srai geopandas")
    SRAI_AVAILABLE = False

def load_embeddings(experiment_name: str) -> Dict[str, pd.DataFrame]:
    """Load all urban embeddings from results directory."""
    results_dir = Path("results/embeddings") / experiment_name
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {results_dir}")
    
    embeddings = {}
    
    # Resolution to mode mapping (drive=res8, bike=res9, walk=res10)
    mode_mapping = {
        'drive': 8,
        'bike': 9, 
        'walk': 10
    }
    
    for mode, res in mode_mapping.items():
        embedding_file = results_dir / f"urban_embeddings_{mode}_unet.parquet"
        
        if embedding_file.exists():
            print(f"Loading {mode} embeddings (resolution {res})...")
            df = pd.read_parquet(embedding_file)
            
            # Ensure h3_index is the index
            if 'h3_index' in df.columns:
                df = df.set_index('h3_index')
            
            embeddings[mode] = df
            print(f"  Loaded {len(df):,} embeddings with {df.shape[1]} dimensions")
        else:
            print(f"Warning: {embedding_file} not found")
    
    if not embeddings:
        raise FileNotFoundError(f"No embedding files found in {results_dir}")
    
    return embeddings

def create_srai_geodataframe(h3_indices: List[str], resolution: int) -> gpd.GeoDataFrame:
    """Use SRAI to create GeoDataFrame from H3 indices."""
    print(f"Creating SRAI GeoDataFrame for {len(h3_indices):,} H3 indices at resolution {resolution}...")
    
    # Create H3Regionalizer
    regionalizer = H3Regionalizer(resolution=resolution)
    
    # Get bounding box of all H3 indices by converting to lat/lng
    import h3
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
    
    # Create bounding box with padding
    bounds = box(
        min(lngs) - 0.02,
        min(lats) - 0.02,
        max(lngs) + 0.02,
        max(lats) + 0.02
    )
    
    # Use SRAI to generate H3 regions for the bounding area
    study_area = gpd.GeoDataFrame([1], geometry=[bounds], crs='EPSG:4326')
    regions_gdf = regionalizer.transform(study_area)
    
    print(f"  SRAI generated {len(regions_gdf):,} total regions in bounds")
    
    # Filter to only our specific H3 indices
    valid_set = set(valid_indices)
    regions_filtered = regions_gdf[regions_gdf.index.isin(valid_set)]
    
    print(f"  Filtered to {len(regions_filtered):,} matching regions")
    
    return regions_filtered

def perform_clustering(gdf: gpd.GeoDataFrame, embeddings_df: pd.DataFrame, 
                      n_clusters: int = 6) -> gpd.GeoDataFrame:
    """Perform clustering on embeddings and add to GeoDataFrame."""
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    # Get embedding columns
    embed_cols = [col for col in embeddings_df.columns if col.startswith('emb_')]
    
    if not embed_cols:
        raise ValueError("No embedding columns found")
    
    # Align indices between GeoDataFrame and embeddings
    common_indices = gdf.index.intersection(embeddings_df.index)
    print(f"  Common indices: {len(common_indices):,}")
    
    if len(common_indices) == 0:
        raise ValueError("No common indices between spatial and embedding data")
    
    # Get embeddings for common indices
    X = embeddings_df.loc[common_indices, embed_cols].values
    
    # Handle any NaN values
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
    
    # Create result GeoDataFrame with clusters
    result_gdf = gdf.loc[common_indices].copy()
    result_gdf['cluster'] = clusters
    
    # Add embedding data
    for col in embed_cols:
        result_gdf[col] = embeddings_df.loc[common_indices, col]
    
    # Print cluster distribution
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    print("  Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"    Cluster {cluster_id}: {count:,} hexagons ({count/len(clusters)*100:.1f}%)")
    
    return result_gdf

def create_resolution_plot(gdf: gpd.GeoDataFrame, mode: str, resolution: int, 
                          output_path: Path, figsize: Tuple[int, int] = (16, 12)):
    """Create clustering visualization for a specific resolution."""
    print(f"Creating plot for {mode} mode (resolution {resolution})...")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    
    # Get unique clusters and colors
    unique_clusters = sorted(gdf['cluster'].unique())
    n_clusters = len(unique_clusters)
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
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
    ax.set_title(f'Urban Embeddings: {mode.title()} Mode (H3 Resolution {resolution})',
                fontsize=16, fontweight='bold', pad=20)
    
    # Clean axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Subtle grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.3)
    
    # Legend
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.9)
    
    # Add metadata
    bounds = gdf.total_bounds
    area_info = f"Spatial extent: {bounds[2]-bounds[0]:.3f}° × {bounds[3]-bounds[1]:.3f}°"
    ax.text(0.02, 0.98, f"{len(gdf):,} hexagons | {n_clusters} clusters | {area_info}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved: {output_path}")

def create_combined_pca_plot(embeddings_dict: Dict[str, pd.DataFrame], 
                           experiment_name: str, output_path: Path,
                           target_resolution: int = 10, n_clusters: int = 8):
    """Create combined plot with all resolutions aligned via PCA at target resolution."""
    print(f"Creating combined PCA plot at resolution {target_resolution}...")
    
    # Get the target resolution embeddings for spatial reference
    target_mode = None
    for mode, res in [('drive', 8), ('bike', 9), ('walk', 10)]:
        if res == target_resolution and mode in embeddings_dict:
            target_mode = mode
            break
    
    if target_mode is None:
        raise ValueError(f"No embeddings found for target resolution {target_resolution}")
    
    target_df = embeddings_dict[target_mode]
    print(f"  Using {target_mode} mode as spatial reference ({len(target_df):,} hexagons)")
    
    # Create spatial framework at target resolution
    target_gdf = create_srai_geodataframe(target_df.index.tolist(), target_resolution)
    
    # Collect all embeddings for PCA alignment
    all_embeddings = []
    mode_labels = []
    
    for mode, df in embeddings_dict.items():
        embed_cols = [col for col in df.columns if col.startswith('emb_')]
        X = df[embed_cols].values
        
        # Handle NaN
        if np.isnan(X).any():
            for i in range(X.shape[1]):
                col_median = np.nanmedian(X[:, i])
                X[np.isnan(X[:, i]), i] = col_median
        
        all_embeddings.append(X)
        mode_labels.extend([mode] * len(X))
    
    # Combine all embeddings
    X_combined = np.vstack(all_embeddings)
    print(f"  Combined embeddings shape: {X_combined.shape}")
    
    # Fit PCA on combined data to find common latent space
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Use sufficient components to capture most variance
    n_components = min(16, X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"  PCA components: {n_components}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Split back to individual modes and project to target resolution
    start_idx = 0
    mode_pca_embeddings = {}
    
    for mode, df in embeddings_dict.items():
        end_idx = start_idx + len(df)
        mode_pca = X_pca[start_idx:end_idx]
        
        # Create DataFrame with PCA embeddings
        pca_df = pd.DataFrame(
            mode_pca,
            index=df.index,
            columns=[f'pca_{i}' for i in range(n_components)]
        )
        
        mode_pca_embeddings[mode] = pca_df
        start_idx = end_idx
    
    # For spatial visualization, we need to map all modes to target resolution
    # We'll use the target mode's embeddings and add data from other modes
    # that can be spatially co-located (overlapping H3 cells)
    
    combined_data = target_gdf.copy()
    
    # Add PCA embeddings from target mode
    target_pca = mode_pca_embeddings[target_mode]
    common_indices = combined_data.index.intersection(target_pca.index)
    
    for col in target_pca.columns:
        combined_data.loc[common_indices, col] = target_pca.loc[common_indices, col]
    
    # Add resolution/mode identifier
    combined_data['resolution'] = target_resolution
    combined_data['primary_mode'] = target_mode
    
    # For clustering, use the PCA features
    pca_cols = [col for col in combined_data.columns if col.startswith('pca_')]
    X_for_clustering = combined_data[pca_cols].values
    
    # Cluster in PCA space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    combined_data['cluster'] = kmeans.fit_predict(X_for_clustering)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(18, 14), facecolor='white')
    
    # Plot clusters
    unique_clusters = sorted(combined_data['cluster'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = combined_data[combined_data['cluster'] == cluster_id]
        cluster_data.plot(
            ax=ax,
            color=colors[i],
            alpha=0.7,
            edgecolor='white',
            linewidth=0.1,
            label=f'Cluster {cluster_id} ({len(cluster_data):,})'
        )
    
    # Styling
    ax.set_aspect('equal')
    ax.set_title(f'Combined Urban Embeddings: {experiment_name}\n' + 
                f'PCA-Aligned Multi-Resolution Analysis (Spatial Reference: H3 Res {target_resolution})',
                fontsize=18, fontweight='bold', pad=25)
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.3)
    
    # Enhanced legend
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_alpha(0.95)
    legend.set_title('PCA Clusters', prop={'weight': 'bold'})
    
    # Detailed metadata
    bounds = combined_data.total_bounds
    metadata_text = (f"Spatial Framework: H3 Resolution {target_resolution}\n"
                    f"Total Hexagons: {len(combined_data):,}\n"
                    f"PCA Components: {n_components} (Var: {pca.explained_variance_ratio_.sum():.1%})\n"
                    f"Resolutions Combined: {list(embeddings_dict.keys())}\n"
                    f"Spatial Extent: {bounds[2]-bounds[0]:.3f}° × {bounds[3]-bounds[1]:.3f}°")
    
    ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved combined PCA plot: {output_path}")
    
    # Return stats
    return {
        'n_hexagons': len(combined_data),
        'n_clusters': len(unique_clusters),
        'pca_components': n_components,
        'explained_variance': float(pca.explained_variance_ratio_.sum()),
        'resolutions_combined': list(embeddings_dict.keys()),
        'target_resolution': target_resolution
    }

def main():
    """Main visualization pipeline."""
    parser = argparse.ArgumentParser(description='Visualize Urban Embeddings using SRAI')
    parser.add_argument('experiment', type=str,
                       help='Experiment name (e.g., south_holland_fsi99)')
    parser.add_argument('--clusters', type=int, default=6,
                       help='Number of clusters per resolution (default: 6)')
    parser.add_argument('--combined-clusters', type=int, default=8,
                       help='Number of clusters for combined PCA plot (default: 8)')
    parser.add_argument('--target-resolution', type=int, default=10, choices=[8, 9, 10],
                       help='Target resolution for combined plot (default: 10)')
    parser.add_argument('--output-dir', type=str,
                       help='Custom output directory (default: results/plots/EXPERIMENT)')
    parser.add_argument('--figsize', type=str, default='16,12',
                       help='Figure size as "width,height" (default: 16,12)')
    
    args = parser.parse_args()
    
    if not SRAI_AVAILABLE:
        print("ERROR: SRAI and required dependencies not available!")
        sys.exit(1)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/plots") / args.experiment
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Urban Embedding Visualization with SRAI")
    print(f"Experiment: {args.experiment}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    try:
        # Load all embeddings
        embeddings = load_embeddings(args.experiment)
        
        if len(embeddings) == 0:
            print("ERROR: No embeddings found!")
            sys.exit(1)
        
        # Parse figure size
        try:
            figsize = tuple(map(int, args.figsize.split(',')))
        except:
            figsize = (16, 12)
        
        # Create individual resolution plots
        resolution_mapping = {'drive': 8, 'bike': 9, 'walk': 10}
        
        for mode, df in embeddings.items():
            resolution = resolution_mapping[mode]
            print(f"\nCreating plot for {mode} mode (resolution {resolution})...")
            
            # Create SRAI GeoDataFrame
            gdf = create_srai_geodataframe(df.index.tolist(), resolution)
            
            # Perform clustering
            gdf_clustered = perform_clustering(gdf, df, args.clusters)
            
            # Create visualization
            output_path = output_dir / f"{args.experiment}_{mode}_res{resolution}_clusters.png"
            create_resolution_plot(gdf_clustered, mode, resolution, output_path, figsize)
        
        # Create combined PCA plot
        print(f"\nCreating combined PCA plot...")
        combined_output = output_dir / f"{args.experiment}_combined_pca_res{args.target_resolution}.png"
        combined_stats = create_combined_pca_plot(
            embeddings, 
            args.experiment, 
            combined_output,
            args.target_resolution,
            args.combined_clusters
        )
        
        # Save comprehensive statistics
        final_stats = {
            'experiment': args.experiment,
            'individual_plots': len(embeddings),
            'total_hexagons_by_resolution': {
                mode: len(df) for mode, df in embeddings.items()
            },
            'clusters_per_resolution': args.clusters,
            'combined_plot_stats': combined_stats
        }
        
        stats_path = output_dir / f"{args.experiment}_visualization_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        print(f"\n" + "=" * 60)
        print(f"SUCCESS: Visualization pipeline completed!")
        print(f"Generated plots:")
        print(f"  - {len(embeddings)} individual resolution plots")
        print(f"  - 1 combined PCA-aligned plot at resolution {args.target_resolution}")
        print(f"Output directory: {output_dir}")
        print(f"Statistics saved: {stats_path}")
        
    except Exception as e:
        print(f"\nERROR in visualization pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()