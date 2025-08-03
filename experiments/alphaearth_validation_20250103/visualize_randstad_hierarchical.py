"""
Randstad visualization with hierarchical clustering - more clusters, better quality
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
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

def create_randstad_hierarchical():
    """Create Randstad visualization with hierarchical clustering"""
    
    if not SRAI_AVAILABLE:
        print("SRAI required")
        return
    
    print("Creating Randstad with hierarchical clustering...")
    
    # Load embeddings
    filepath = Path("data/alphaearth_processed/netherlands_2023_h3_res10.parquet")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    df = pd.read_parquet(filepath)
    print(f"Total hexagons: {len(df):,}")
    
    # Define Randstad bounds - comprehensive area
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
    
    # Hierarchical clustering with MORE clusters
    print(f"Performing hierarchical clustering on {len(gdf_land):,} hexagons...")
    gdf_clustered = cluster_hierarchical(gdf_land)
    
    # Create visualization
    create_hierarchical_map(gdf_clustered)
    
    return gdf_clustered

def filter_randstad_land(gdf):
    """Filter to land areas in Randstad"""
    
    embedding_cols = [col for col in gdf.columns if col.startswith('embed_')]
    
    gdf['embed_std'] = gdf[embedding_cols].std(axis=1, skipna=True)
    gdf['non_nan_embeds'] = gdf[embedding_cols].notna().sum(axis=1)
    
    gdf['centroid'] = gdf.geometry.centroid
    gdf['lon'] = gdf.centroid.x
    gdf['lat'] = gdf.centroid.y
    
    # Enhanced land filter for Randstad
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

def cluster_hierarchical(gdf):
    """Perform memory-efficient hierarchical clustering with more clusters"""
    
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
    
    # Memory-efficient approach: Use connectivity-based hierarchical clustering
    # This avoids computing full pairwise distance matrix
    n_clusters = 30  # Even more clusters for finest detail
    print(f"Running memory-efficient hierarchical clustering with {n_clusters} clusters...")
    
    # Use connectivity to make it memory efficient
    # Create sparse connectivity matrix based on spatial neighbors (more efficient)
    from sklearn.neighbors import kneighbors_graph
    
    # Create spatial coordinates for connectivity
    coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    connectivity = kneighbors_graph(coords, n_neighbors=20, include_self=False)
    
    # AgglomerativeClustering with connectivity (much more memory efficient)
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        connectivity=connectivity,
        linkage='ward'
    )
    
    gdf['cluster'] = hierarchical.fit_predict(X_scaled)
    
    # Print detailed cluster distribution
    unique, counts = np.unique(gdf['cluster'], return_counts=True)
    print(f"\nMemory-Efficient Hierarchical Cluster Distribution ({n_clusters} clusters):")
    for i, count in enumerate(counts):
        print(f"  Cluster {i:2d}: {count:7,} hexagons ({count/len(gdf)*100:5.1f}%)")
    
    return gdf

def create_hierarchical_map(gdf):
    """Create full-page hierarchical clustering visualization"""
    
    print("Creating hierarchical clustering map...")
    
    # Full 1440p canvas
    fig = plt.figure(figsize=(25.6, 14.4), facecolor='#fdfdf9')
    
    # Single axis covering entire page
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Enhanced color palette for 30 clusters
    # Use multiple colormaps for better distinction
    colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors2 = plt.cm.Set3(np.linspace(0, 1, 10))
    colors = np.vstack([colors1, colors2])
    
    # Plot all 30 clusters
    for cluster_id in range(30):
        cluster_data = gdf[gdf['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            cluster_data.plot(
                ax=ax,
                color=colors[cluster_id],
                alpha=0.9,
                edgecolor='none',
                linewidth=0
            )
    
    # Set bounds to data
    bounds = gdf.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal')
    
    # Completely clean styling
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Ultra-subtle grid
    ax.grid(True, alpha=0.015, linestyle='-', linewidth=0.1)
    
    # Minimal corner text
    fig.text(0.99, 0.01, 'Randstad•Hierarchical•30k', ha='right', va='bottom', 
             fontsize=8, fontweight='200', color='#888888', alpha=0.4)
    
    # Save outputs
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "randstad_hierarchical_30clusters.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Saved hierarchical visualization: {output_path}")
    
    # Ultra high quality version
    output_hq = output_dir / "randstad_hierarchical_print.png"
    plt.savefig(output_hq, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Print version: {output_hq}")
    
    plt.close()
    
    # Save detailed stats
    cluster_stats = gdf['cluster'].value_counts().sort_index()
    
    summary = {
        'method': 'hierarchical_clustering',
        'algorithm': 'AgglomerativeClustering',
        'linkage': 'ward',
        'total_hexagons': len(gdf),
        'clusters': 30,
        'area': 'randstad',
        'bounds': gdf.total_bounds.tolist(),
        'cluster_sizes': cluster_stats.to_dict(),
        'largest_clusters': cluster_stats.head(10).to_dict(),
        'smallest_clusters': cluster_stats.tail(5).to_dict()
    }
    
    with open(output_dir / "randstad_hierarchical_stats.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"\nHierarchical clustering visualization complete!")
    print(f"Total hexagons: {len(gdf):,}")
    print(f"Method: Hierarchical (Ward linkage)")
    print(f"Clusters: 30 (finest detail)")
    print(f"Enhanced color distinction")

def main():
    create_randstad_hierarchical()

if __name__ == "__main__":
    main()