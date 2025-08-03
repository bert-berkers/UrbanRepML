"""
Randstad visualization with balanced clustering - minimum 10k hexagons per cluster
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

def create_randstad_balanced():
    """Create Randstad visualization with balanced clustering - min 10k per cluster"""
    
    if not SRAI_AVAILABLE:
        print("SRAI required")
        return
    
    print("Creating Randstad with balanced clustering (min 10k hexagons per cluster)...")
    
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
    
    # Balanced clustering with minimum cluster size constraint
    print(f"Performing balanced clustering on {len(gdf_land):,} hexagons...")
    gdf_clustered = cluster_balanced(gdf_land)
    
    # Create visualization
    create_balanced_map(gdf_clustered)
    
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

def cluster_balanced(gdf):
    """Perform balanced clustering ensuring minimum cluster sizes"""
    
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
    
    # Calculate optimal number of clusters for minimum 10k hexagons per cluster
    min_cluster_size = 10000
    total_hexagons = len(gdf)
    max_clusters = total_hexagons // min_cluster_size
    
    # Choose a very conservative number to ensure minimum cluster size
    # With ~578k hexagons, max would be ~57 clusters
    # Let's use even fewer to almost guarantee 10k+ per cluster
    n_clusters = min(25, max_clusters)  # Very conservative for minimum size guarantee
    
    expected_cluster_size = total_hexagons // n_clusters
    
    print(f"Total hexagons: {total_hexagons:,}")
    print(f"Target minimum cluster size: {min_cluster_size:,}")
    print(f"Maximum possible clusters: {max_clusters}")
    print(f"Chosen clusters: {n_clusters}")
    print(f"Expected average cluster size: {expected_cluster_size:,}")
    
    # Use MiniBatchKMeans for efficiency - pure embedding-based clustering
    balanced_kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=15000,
        n_init=5,
        max_iter=150
    )
    
    gdf['cluster'] = balanced_kmeans.fit_predict(X_scaled)
    
    # Check actual cluster sizes
    unique, counts = np.unique(gdf['cluster'], return_counts=True)
    print(f"\nBalanced Cluster Distribution ({n_clusters} clusters):")
    print(f"Actual average cluster size: {counts.mean():.0f} hexagons")
    print(f"Largest cluster: {counts.max():,} hexagons ({counts.max()/len(gdf)*100:.1f}%)")
    print(f"Smallest cluster: {counts.min():,} hexagons ({counts.min()/len(gdf)*100:.1f}%)")
    print(f"Median cluster size: {np.median(counts):.0f} hexagons")
    print(f"Standard deviation: {counts.std():.0f} hexagons")
    
    # Check if minimum size constraint is met
    clusters_below_min = np.sum(counts < min_cluster_size)
    print(f"\nCluster size analysis:")
    print(f"Clusters below {min_cluster_size:,} hexagons: {clusters_below_min}")
    print(f"Clusters at or above {min_cluster_size:,} hexagons: {n_clusters - clusters_below_min}")
    
    if clusters_below_min > 0:
        print(f"WARNING: {clusters_below_min} clusters are below target minimum size")
        smallest_clusters = sorted(enumerate(counts), key=lambda x: x[1])[:5]
        print("5 smallest clusters:")
        for cluster_id, size in smallest_clusters:
            print(f"  Cluster {cluster_id}: {size:,} hexagons")
    else:
        print("SUCCESS: All clusters meet minimum size requirement!")
    
    return gdf

def create_balanced_map(gdf):
    """Create visualization with balanced clusters"""
    
    print("Creating balanced clusters visualization...")
    
    # Full 1440p canvas
    fig = plt.figure(figsize=(25.6, 14.4), facecolor='#fdfdf9')
    
    # Single axis covering entire page
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Get number of clusters
    n_clusters = len(gdf['cluster'].unique())
    
    # Color palette for balanced clusters (around 50)
    if n_clusters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    else:
        # For more clusters, combine multiple colormaps
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.tab20b(np.linspace(0, 1, min(20, n_clusters-20)))
        if n_clusters > 40:
            colors3 = plt.cm.tab20c(np.linspace(0, 1, n_clusters-40))
            colors = np.vstack([colors1, colors2, colors3])
        else:
            colors = np.vstack([colors1, colors2])
    
    # Plot all clusters
    print(f"Plotting {n_clusters} balanced clusters...")
    for cluster_id in range(n_clusters):
        cluster_data = gdf[gdf['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            cluster_data.plot(
                ax=ax,
                color=colors[cluster_id],
                alpha=0.85,
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
    fig.text(0.99, 0.01, f'Randstad•Balanced•{n_clusters}k', ha='right', va='bottom', 
             fontsize=8, fontweight='200', color='#888888', alpha=0.4)
    
    # Save outputs
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"randstad_balanced_{n_clusters}clusters.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Saved balanced clusters visualization: {output_path}")
    
    # Ultra high quality version
    output_hq = output_dir / "randstad_balanced_print.png"
    plt.savefig(output_hq, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Print version: {output_hq}")
    
    plt.close()
    
    # Save detailed stats
    cluster_stats = gdf['cluster'].value_counts().sort_index()
    
    summary = {
        'method': 'balanced_embedding_clustering',
        'algorithm': 'MiniBatchKMeans',
        'spatial_constraint': 'none',
        'min_cluster_size_target': 10000,
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
        'clusters_below_10k': int(np.sum(cluster_stats < 10000)),
        'clusters_above_10k': int(np.sum(cluster_stats >= 10000)),
        'largest_clusters': cluster_stats.head(10).to_dict(),
        'smallest_clusters': cluster_stats.tail(10).to_dict()
    }
    
    with open(output_dir / "randstad_balanced_stats.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"\nBalanced clustering visualization complete!")
    print(f"Total hexagons: {len(gdf):,}")
    print(f"Method: Balanced embedding-based clustering")
    print(f"Clusters: {n_clusters} (balanced detail)")
    print(f"Target: minimum 10k hexagons per cluster")

def main():
    create_randstad_balanced()

if __name__ == "__main__":
    main()