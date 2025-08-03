"""
Single large Randstad visualization with more clusters - full page usage
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

def create_randstad_fullpage():
    """Create single large Randstad visualization using entire page"""
    
    if not SRAI_AVAILABLE:
        print("SRAI required")
        return
    
    print("Creating full-page Randstad visualization...")
    
    # Load embeddings from experiments folder
    filepath = Path("experiments/alphaearth_validation_20250103/data/alphaearth_processed/netherlands_2023_h3_res10.parquet")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    df = pd.read_parquet(filepath)
    print(f"Total hexagons: {len(df):,}")
    
    # Define Randstad bounds - larger area covering the megalopolis
    randstad_bounds = box(4.0, 51.7, 5.5, 52.6)  # Large Randstad area
    
    print("Processing Randstad area...")
    
    # Use SRAI to get all H3 regions for Randstad
    regionalizer = H3Regionalizer(resolution=10)
    randstad_gdf = regionalizer.transform(
        gpd.GeoDataFrame([1], geometry=[randstad_bounds], crs='EPSG:4326')
    )
    
    print(f"Generated {len(randstad_gdf):,} H3 regions for Randstad")
    
    # Filter embeddings to Randstad area
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
    
    # Clustering with MORE clusters for better detail
    print(f"Performing clustering on {len(gdf_land):,} hexagons...")
    gdf_clustered = cluster_randstad(gdf_land)
    
    # Create full-page visualization
    create_single_fullpage_map(gdf_clustered)
    
    return gdf_clustered

def filter_randstad_land(gdf):
    """Filter Randstad area to land"""
    
    embedding_cols = [col for col in gdf.columns if col.startswith('embed_')]
    
    gdf['embed_std'] = gdf[embedding_cols].std(axis=1, skipna=True)
    gdf['non_nan_embeds'] = gdf[embedding_cols].notna().sum(axis=1)
    
    gdf['centroid'] = gdf.geometry.centroid
    gdf['lon'] = gdf.centroid.x
    gdf['lat'] = gdf.centroid.y
    
    # Randstad-specific land filter
    land_filter = (
        (gdf['lon'] > 4.1) &  # Exclude far western ocean
        (gdf['embed_std'].fillna(0) > 0.008) &
        (gdf['non_nan_embeds'] >= 8) &
        # Exclude major water bodies in Randstad
        ~((gdf['lon'] < 4.3) & (gdf['lat'] > 52.3)) &  # Northwestern coastal water
        ~((gdf['lon'] > 5.1) & (gdf['lon'] < 5.3) & (gdf['lat'] > 52.4))  # IJsselmeer area
    )
    
    gdf_land = gdf[land_filter].copy()
    print(f"Land hexagons: {len(gdf_land):,} ({len(gdf_land)/len(gdf)*100:.1f}%)")
    
    return gdf_land

def cluster_randstad(gdf):
    """Perform clustering with more clusters for detail"""
    
    embedding_cols = [col for col in gdf.columns if col.startswith('embed_')]
    X = gdf[embedding_cols].values
    
    # Handle NaN
    if np.isnan(X).any():
        for i in range(X.shape[1]):
            col_median = np.nanmedian(X[:, i])
            X[np.isnan(X[:, i]), i] = col_median
    
    # More clusters for better detail
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = 20  # More clusters for finer detail
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    gdf['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Print cluster distribution
    unique, counts = np.unique(gdf['cluster'], return_counts=True)
    print(f"\nCluster Distribution ({n_clusters} clusters):")
    for i, count in enumerate(counts):
        print(f"  Cluster {i:2d}: {count:7,} hexagons ({count/len(gdf)*100:5.1f}%)")
    
    return gdf

def create_single_fullpage_map(gdf):
    """Create single full-page visualization using entire canvas"""
    
    print("Creating full-page single map...")
    
    # Full 1440p canvas - 16:9 aspect ratio
    fig = plt.figure(figsize=(25.6, 14.4), facecolor='#fdfdf9')
    
    # Single axis covering THE ENTIRE PAGE - no margins!
    ax = fig.add_axes([0, 0, 1, 1])  # Full page coverage
    
    # Color palette for 20 clusters
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Plot all clusters
    for cluster_id in range(20):
        cluster_data = gdf[gdf['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            cluster_data.plot(
                ax=ax,
                color=colors[cluster_id],
                alpha=0.85,
                edgecolor='none',
                linewidth=0
            )
    
    # Set bounds to Randstad area
    bounds = gdf.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal')
    
    # NO styling - completely clean
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Very subtle grid
    ax.grid(True, alpha=0.02, linestyle='-', linewidth=0.1)
    
    # MINIMAL text - tiny corner info only
    fig.text(0.99, 0.01, 'Randstad•H3-10', ha='right', va='bottom', 
             fontsize=8, fontweight='200', color='#888888', alpha=0.5)
    
    # Save in experiments folder
    output_dir = Path("experiments/alphaearth_validation_20250103/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "randstad_fullpage_20clusters.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Saved full-page Randstad: {output_path}")
    
    # Ultra high quality version
    output_hq = output_dir / "randstad_fullpage_print.png"
    plt.savefig(output_hq, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Print version: {output_hq}")
    
    plt.close()
    
    # Save stats
    cluster_stats = gdf['cluster'].value_counts().sort_index()
    
    summary = {
        'area': 'randstad',
        'total_hexagons': len(gdf),
        'clusters': 20,
        'bounds': gdf.total_bounds.tolist(),
        'dominant_clusters': cluster_stats.head(5).to_dict()
    }
    
    with open(output_dir / "randstad_fullpage_stats.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"\nRandstad full-page visualization complete!")
    print(f"Total hexagons: {len(gdf):,}")
    print(f"Clusters: 20 (more detail)")
    print(f"Full page coverage achieved")

def main():
    create_randstad_fullpage()

if __name__ == "__main__":
    main()