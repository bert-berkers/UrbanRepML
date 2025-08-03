"""
Randstad visualization with many fine clusters focused on embedding similarity
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import srai
    from srai.regionalizers import H3Regionalizer
    import geopandas as gpd
    from shapely.geometry import box
    SRAI_AVAILABLE = True
except ImportError:
    SRAI_AVAILABLE = False

def create_randstad_fineclusters():
    """Create Randstad visualization with many fine clusters based on embeddings"""
    
    if not SRAI_AVAILABLE:
        print("SRAI required")
        return
    
    print("Creating Randstad with fine embedding-based clustering...")
    
    # Load embeddings
    filepath = Path("data/alphaearth_processed/netherlands_2023_h3_res10.parquet")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    df = pd.read_parquet(filepath)
    print(f"Total hexagons: {len(df):,}")
    
    # Define Randstad bounds
    randstad_bounds = box(4.0, 51.7, 5.5, 52.6)
    
    print("Processing Randstad area...")
    
    # Use SRAI to get all H3 regions
    regionalizer = H3Regionalizer(resolution=10)
    randstad_gdf = regionalizer.transform(
        gpd.GeoDataFrame([1], geometry=[randstad_bounds], crs='EPSG:4326')
    )
    
    print(f"Generated {len(randstad_gdf):,} H3 regions for Randstad")
    
    # Filter embeddings to Randstad
    randstad_h3_indices = set(randstad_gdf.index.tolist())
    df_randstad = df[df['h3_index'].isin(randstad_h3_indices)].copy()
    
    print(f"Found {len(df_randstad):,} matching hexagons")
    
    # Add geometries
    df_randstad = df_randstad.set_index('h3_index')
    matching_geometries = randstad_gdf.loc[randstad_gdf.index.intersection(df_randstad.index)]
    
    gdf_randstad = gpd.GeoDataFrame(
        df_randstad.loc[matching_geometries.index],
        geometry=matching_geometries.geometry,
        crs='EPSG:4326'
    )
    
    # Filter to land areas
    print("Filtering to land areas...")
    gdf_land = filter_randstad_land(gdf_randstad)
    
    # Fine clustering focused on embeddings NOT spatial proximity
    print(f"Performing fine embedding-based clustering on {len(gdf_land):,} hexagons...")
    gdf_clustered = cluster_fine_embeddings(gdf_land)
    
    # Create visualization
    create_fineclusters_map(gdf_clustered)
    
    return gdf_clustered

def filter_randstad_land(gdf):
    """Filter to land areas in Randstad"""
    
    embedding_cols = [col for col in gdf.columns if col.startswith('embed_')]
    
    gdf['embed_std'] = gdf[embedding_cols].std(axis=1, skipna=True)
    gdf['non_nan_embeds'] = gdf[embedding_cols].notna().sum(axis=1)
    
    gdf['centroid'] = gdf.geometry.centroid
    gdf['lon'] = gdf.centroid.x
    gdf['lat'] = gdf.centroid.y
    
    # Land filter for Randstad
    land_filter = (
        (gdf['lon'] > 4.1) &  # Exclude western ocean
        (gdf['embed_std'].fillna(0) > 0.008) &
        (gdf['non_nan_embeds'] >= 8) &
        # Exclude water bodies
        ~((gdf['lon'] < 4.3) & (gdf['lat'] > 52.3)) &  # Northwestern coast
        ~((gdf['lon'] > 5.1) & (gdf['lon'] < 5.3) & (gdf['lat'] > 52.4))  # IJsselmeer
    )
    
    gdf_land = gdf[land_filter].copy()
    print(f"Land hexagons: {len(gdf_land):,} ({len(gdf_land)/len(gdf)*100:.1f}%)")
    
    return gdf_land

def cluster_fine_embeddings(gdf):
    """Perform fine clustering based PURELY on embedding similarity"""
    
    embedding_cols = [col for col in gdf.columns if col.startswith('embed_')]
    X = gdf[embedding_cols].values
    
    # Handle NaN values
    if np.isnan(X).any():
        for i in range(X.shape[1]):
            col_median = np.nanmedian(X[:, i])
            X[np.isnan(X[:, i]), i] = col_median
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # MANY MORE clusters - focus on embedding patterns not spatial coherence
    n_clusters = 100  # Much finer granularity - smaller clusters
    print(f"Running fine-grained clustering with {n_clusters} clusters...")
    print("Focusing on embedding similarity, NOT spatial proximity")
    
    # Use MiniBatchKMeans for efficiency with large cluster count
    # NO spatial constraints - pure embedding-based clustering
    fine_kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=10000,
        n_init=3,
        max_iter=100
    )
    
    gdf['cluster'] = fine_kmeans.fit_predict(X_scaled)
    
    # Print cluster size statistics
    unique, counts = np.unique(gdf['cluster'], return_counts=True)
    print(f"\nFine Cluster Distribution ({n_clusters} clusters):")
    print(f"Average cluster size: {len(gdf)/n_clusters:.0f} hexagons")
    print(f"Largest cluster: {counts.max():,} hexagons ({counts.max()/len(gdf)*100:.1f}%)")
    print(f"Smallest cluster: {counts.min():,} hexagons ({counts.min()/len(gdf)*100:.1f}%)")
    print(f"Median cluster size: {np.median(counts):.0f} hexagons")
    
    # Show size distribution
    size_ranges = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, float('inf'))]
    for min_size, max_size in size_ranges:
        if max_size == float('inf'):
            count = np.sum(counts >= min_size)
            print(f"Clusters with >={min_size:,} hexagons: {count}")
        else:
            count = np.sum((counts >= min_size) & (counts < max_size))
            print(f"Clusters with {min_size:,}-{max_size:,} hexagons: {count}")
    
    return gdf

def create_fineclusters_map(gdf):
    """Create visualization with many fine clusters"""
    
    print("Creating fine clusters visualization...")
    
    # Full 1440p canvas
    fig = plt.figure(figsize=(25.6, 14.4), facecolor='#fdfdf9')
    
    # Single axis covering entire page
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Enhanced color palette for 100 clusters
    # Use multiple colormaps and cycle through them
    n_clusters = 100
    
    # Generate diverse colors by cycling through multiple colormaps
    cmap_names = ['tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'Dark2', 'Accent']
    colors = []
    
    for i in range(n_clusters):
        cmap_idx = i % len(cmap_names)
        color_idx = (i // len(cmap_names)) % 20  # Cycle through 20 colors per cmap
        cmap = plt.cm.get_cmap(cmap_names[cmap_idx])
        colors.append(cmap(color_idx / 20))
    
    # Plot all clusters
    print("Plotting 100 fine clusters...")
    for cluster_id in range(n_clusters):
        cluster_data = gdf[gdf['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            cluster_data.plot(
                ax=ax,
                color=colors[cluster_id],
                alpha=0.8,
                edgecolor='none',
                linewidth=0
            )
    
    # Set bounds to data
    bounds = gdf.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal')
    
    # Clean styling
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Very subtle grid
    ax.grid(True, alpha=0.01, linestyle='-', linewidth=0.1)
    
    # Minimal text
    fig.text(0.99, 0.01, 'Randstad•Fine•100k', ha='right', va='bottom', 
             fontsize=8, fontweight='200', color='#888888', alpha=0.4)
    
    # Save outputs
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "randstad_fine_100clusters.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Saved fine clusters visualization: {output_path}")
    
    # Ultra high quality version
    output_hq = output_dir / "randstad_fine_print.png"
    plt.savefig(output_hq, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Print version: {output_hq}")
    
    plt.close()
    
    # Save detailed stats
    cluster_stats = gdf['cluster'].value_counts().sort_index()
    
    summary = {
        'method': 'fine_embedding_clustering',
        'algorithm': 'MiniBatchKMeans',
        'spatial_constraint': 'none',
        'total_hexagons': len(gdf),
        'clusters': n_clusters,
        'area': 'randstad',
        'bounds': gdf.total_bounds.tolist(),
        'cluster_size_stats': {
            'mean': float(cluster_stats.mean()),
            'median': float(cluster_stats.median()),
            'min': int(cluster_stats.min()),
            'max': int(cluster_stats.max()),
            'std': float(cluster_stats.std())
        },
        'largest_10_clusters': cluster_stats.head(10).to_dict(),
        'smallest_10_clusters': cluster_stats.tail(10).to_dict()
    }
    
    with open(output_dir / "randstad_fine_stats.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"\nFine clustering visualization complete!")
    print(f"Total hexagons: {len(gdf):,}")
    print(f"Method: Pure embedding-based clustering")
    print(f"Clusters: {n_clusters} (fine granularity)")
    print(f"NO spatial constraints - focuses on land use patterns")

def main():
    create_randstad_fineclusters()

if __name__ == "__main__":
    main()