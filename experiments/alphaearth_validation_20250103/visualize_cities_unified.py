"""
Unified clustering across all Dutch cities - larger areas, minimal text, full page imagery
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

def create_unified_cities():
    """Create unified clustering across all cities with larger contexts"""
    
    if not SRAI_AVAILABLE:
        print("SRAI required")
        return
    
    print("Creating unified city visualization...")
    
    # Load embeddings from experiments folder
    filepath = Path("experiments/alphaearth_validation_20250103/data/alphaearth_processed/netherlands_2023_h3_res10.parquet")
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    df = pd.read_parquet(filepath)
    print(f"Total hexagons: {len(df):,}")
    
    # Define cities with LARGER bounds including hinterland
    cities = {
        'Amsterdam': {
            'bounds': box(4.65, 52.15, 5.15, 52.55),  # Much larger - includes suburbs
            'center': (4.9, 52.37)
        },
        'Rotterdam': {
            'bounds': box(4.20, 51.75, 4.75, 52.15),  # Includes port areas and surroundings
            'center': (4.47, 51.92)
        },
        'The Hague': {
            'bounds': box(4.05, 51.85, 4.55, 52.25),  # Includes coastal areas and suburbs
            'center': (4.31, 52.08)
        },
        'Utrecht': {
            'bounds': box(4.85, 51.85, 5.35, 52.25),  # Includes surrounding towns
            'center': (5.12, 52.09)
        },
        'Eindhoven': {
            'bounds': box(5.25, 51.25, 5.75, 51.65),  # Includes tech corridor
            'center': (5.47, 51.44)
        },
        'Groningen': {
            'bounds': box(6.35, 53.05, 6.85, 53.45),  # Includes northern countryside
            'center': (6.57, 53.22)
        }
    }
    
    # Combine all city bounds into one large area
    print("Processing all cities together...")
    all_bounds = []
    for city_info in cities.values():
        all_bounds.append(city_info['bounds'])
    
    # Create combined region
    combined_bounds = box(
        min(b.bounds[0] for b in all_bounds) - 0.05,
        min(b.bounds[1] for b in all_bounds) - 0.05,
        max(b.bounds[2] for b in all_bounds) + 0.05,
        max(b.bounds[3] for b in all_bounds) + 0.05
    )
    
    # Use SRAI to get all H3 regions
    regionalizer = H3Regionalizer(resolution=10)
    combined_gdf = regionalizer.transform(
        gpd.GeoDataFrame([1], geometry=[combined_bounds], crs='EPSG:4326')
    )
    
    print(f"Generated {len(combined_gdf):,} H3 regions for combined area")
    
    # Filter embeddings to combined area
    combined_h3_indices = set(combined_gdf.index.tolist())
    df_combined = df[df['h3_index'].isin(combined_h3_indices)].copy()
    
    print(f"Found {len(df_combined):,} matching hexagons")
    
    # Add geometries
    df_combined = df_combined.set_index('h3_index')
    matching_geometries = combined_gdf.loc[combined_gdf.index.intersection(df_combined.index)]
    
    gdf_combined = gpd.GeoDataFrame(
        df_combined.loc[matching_geometries.index],
        geometry=matching_geometries.geometry,
        crs='EPSG:4326'
    )
    
    # Filter to land areas
    print("Filtering to land areas...")
    gdf_land = filter_combined_land(gdf_combined)
    
    # UNIFIED CLUSTERING across ALL cities
    print(f"Performing UNIFIED clustering on {len(gdf_land):,} hexagons...")
    gdf_clustered = cluster_unified(gdf_land)
    
    # Create visualization
    create_full_page_map(gdf_clustered, cities)
    
    return gdf_clustered

def filter_combined_land(gdf):
    """Filter combined area to land"""
    
    embedding_cols = [col for col in gdf.columns if col.startswith('embed_')]
    
    gdf['embed_std'] = gdf[embedding_cols].std(axis=1, skipna=True)
    gdf['non_nan_embeds'] = gdf[embedding_cols].notna().sum(axis=1)
    
    gdf['centroid'] = gdf.geometry.centroid
    gdf['lon'] = gdf.centroid.x
    gdf['lat'] = gdf.centroid.y
    
    # General land filter for combined area
    land_filter = (
        (gdf['lon'] > 4.1) &  # Exclude far western ocean
        (gdf['embed_std'].fillna(0) > 0.008) &
        (gdf['non_nan_embeds'] >= 8) &
        # Exclude major water bodies
        ~((gdf['lon'] < 4.3) & (gdf['lat'] > 52.3))  # Northwestern coastal water
    )
    
    gdf_land = gdf[land_filter].copy()
    print(f"Land hexagons: {len(gdf_land):,} ({len(gdf_land)/len(gdf)*100:.1f}%)")
    
    return gdf_land

def cluster_unified(gdf):
    """Perform unified clustering across all data"""
    
    embedding_cols = [col for col in gdf.columns if col.startswith('embed_')]
    X = gdf[embedding_cols].values
    
    # Handle NaN
    if np.isnan(X).any():
        for i in range(X.shape[1]):
            col_median = np.nanmedian(X[:, i])
            X[np.isnan(X[:, i]), i] = col_median
    
    # Unified clustering - same colors across all cities
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = 14  # Consistent number for all cities
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    gdf['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Print unified distribution
    unique, counts = np.unique(gdf['cluster'], return_counts=True)
    print(f"\nUnified Cluster Distribution:")
    for i, count in enumerate(counts):
        print(f"  Cluster {i:2d}: {count:7,} hexagons ({count/len(gdf)*100:5.1f}%)")
    
    return gdf

def create_full_page_map(gdf, cities):
    """Create full-page visualization with minimal text"""
    
    print("Creating full-page visualization...")
    
    # Full 1440p canvas
    fig = plt.figure(figsize=(25.6, 14.4), facecolor='#fdfdf9')
    
    # 2x3 grid filling entire page - no margins!
    city_names = ['Amsterdam', 'Utrecht', 'Rotterdam', 'The Hague', 'Eindhoven', 'Groningen']
    
    # Full page positions - edge to edge
    positions = [
        [0.005, 0.67, 0.328, 0.33],   # Amsterdam - top left
        [0.336, 0.67, 0.328, 0.33],  # Utrecht - top middle  
        [0.667, 0.67, 0.328, 0.33],  # Rotterdam - top right
        [0.005, 0.34, 0.328, 0.33],  # The Hague - middle left
        [0.336, 0.34, 0.328, 0.33],  # Eindhoven - middle middle
        [0.667, 0.34, 0.328, 0.33],  # Groningen - middle right
    ]
    
    # UNIFIED color palette - same across all cities
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    
    for i, city_name in enumerate(city_names):
        if city_name not in cities:
            continue
        
        city_info = cities[city_name]
        
        # Create axis
        ax = fig.add_axes(positions[i])
        
        # Get city data from unified clustering
        bounds = city_info['bounds']
        city_data = gdf.cx[bounds.bounds[0]:bounds.bounds[2], bounds.bounds[1]:bounds.bounds[3]]
        
        if len(city_data) == 0:
            continue
        
        # Plot ALL clusters with SAME color scheme
        for cluster_id in range(14):
            cluster_data = city_data[city_data['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                cluster_data.plot(
                    ax=ax,
                    color=colors[cluster_id],
                    alpha=0.9,
                    edgecolor='none',
                    linewidth=0
                )
        
        # Set bounds - larger areas with hinterland
        ax.set_xlim(bounds.bounds[0], bounds.bounds[2])
        ax.set_ylim(bounds.bounds[1], bounds.bounds[3])
        ax.set_aspect('equal')
        
        # MINIMAL styling - let imagery dominate
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Very subtle grid
        ax.grid(True, alpha=0.03, linestyle='-', linewidth=0.2)
        
        # MINIMAL text - tiny city label only
        center = city_info['center']
        ax.text(center[0], bounds.bounds[3] - 0.02, city_name,
               ha='center', va='top', fontsize=10, fontweight='200',
               color='#555555', alpha=0.7)
        
        # Tiny center dot
        ax.plot(center[0], center[1], 'o', markersize=1.5, 
               color='white', markeredgecolor='#444444', 
               markeredgewidth=0.3, zorder=1000, alpha=0.8)
    
    # MINIMAL bottom info - very small
    fig.text(0.99, 0.01, 'H3•10', ha='right', va='bottom', 
             fontsize=8, fontweight='200', color='#888888', alpha=0.6)
    
    # Tiny scale in corner
    fig.add_artist(plt.Line2D([0.96, 0.98], [0.05, 0.05], 
                             transform=fig.transFigure, 
                             color='#888888', linewidth=0.8, alpha=0.6))
    fig.text(0.97, 0.07, '10km', ha='center', va='bottom', 
             fontsize=7, color='#888888', alpha=0.6)
    
    # Save - full page, high quality in experiments folder
    output_dir = Path("experiments/alphaearth_validation_20250103/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "dutch_cities_unified_fullpage.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Saved full-page visualization: {output_path}")
    
    # Ultra high quality version
    output_hq = output_dir / "dutch_cities_unified_print.png"
    plt.savefig(output_hq, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='#fdfdf9', edgecolor='none')
    print(f"Print version: {output_hq}")
    
    plt.close()
    
    # Save minimal stats
    cluster_stats = gdf['cluster'].value_counts().sort_index()
    
    summary = {
        'unified_clustering': True,
        'total_hexagons': len(gdf),
        'clusters': 14,
        'cities': len(cities),
        'dominant_clusters': cluster_stats.head(5).to_dict()
    }
    
    with open(output_dir / "unified_cities_stats.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"\nUnified visualization complete!")
    print(f"Total hexagons: {len(gdf):,}")
    print(f"Unified clusters: 14")
    print(f"Colors consistent across all cities")
    print(f"Larger areas with hinterland context")

def main():
    create_unified_cities()

if __name__ == "__main__":
    main()