"""
Process raw AlphaEarth 2022 TIFFs for Netherlands to H3 resolution 8
with K-means kmeans_clustering_1layer and static cartographic visualization
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv('keys/.env')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from stage1_modalities.alphaearth.processor import AlphaEarthProcessor

# Set up for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'


def process_alphaearth_2022():
    """Process AlphaEarth 2022 TIFFs to H3 resolution 8."""
    
    print("="*60)
    print("Processing AlphaEarth 2022 Data for Netherlands")
    print("="*60)
    
    # Configuration
    config = {
        'source_dir': os.getenv('ALPHAEARTH_NETHERLANDS_PATH', 'G:/My Drive/AlphaEarth_Netherlands/'),  # Use env var
        'subtile_size': 512,
        'min_pixels_per_hex': 5,
        'max_workers': 10
    }
    
    # Initialize processor
    processor = AlphaEarthProcessor(config)
    
    # Process 2022 data to resolution 8
    print("\n1. Processing raw TIFFs to H3 resolution 8...")
    print(f"   Source: {config['source_dir']}")
    print(f"   Year filter: 2022")
    print(f"   H3 resolution: 8")
    
    gdf = processor.process(
        raw_data_path=Path(config['source_dir']),
        year_filter='2022',
        h3_resolution=8
    )
    
    print(f"   Processed {len(gdf)} H3 hexagons")
    
    # Save processed embeddings
    output_dir = Path('data/study_areas/netherlands/embeddings/alphaearth')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'netherlands_res8_2022.parquet'
    gdf.to_parquet(output_file)
    print(f"   Saved to: {output_file}")
    
    return gdf


def apply_clustering(gdf, k_values=[8, 10, 12]):
    """Apply K-means kmeans_clustering_1layer to embeddings."""
    
    print("\n2. Applying K-means kmeans_clustering_1layer...")
    
    # Extract embedding columns
    embedding_cols = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
    X = gdf[embedding_cols].values
    
    print(f"   Embedding dimensions: {X.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for k in k_values:
        print(f"\n   K={k}:")
        
        # Apply K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, labels)
        print(f"     Silhouette score: {sil_score:.3f}")
        
        # Add cluster labels to GDF
        gdf[f'cluster_k{k}'] = labels
        
        # Get cluster centers (in original space)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        results[k] = {
            'labels': labels,
            'centers': centers,
            'silhouette': sil_score,
            'inertia': kmeans.inertia_
        }
    
    return gdf, results


def create_cartographic_map(gdf, k, output_path):
    """Create professional cartographic visualization with proper styling."""
    
    print(f"\n   Creating map for K={k}...")
    
    # Transform to Dutch RD for accurate representation
    gdf_rd = gdf.to_crs('EPSG:28992')
    
    # Create figure with proper size for A4 landscape
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Get bounds
    bounds = gdf_rd.total_bounds
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]
    
    # Add padding (5%)
    padding = 0.05
    ax.set_xlim(bounds[0] - x_range * padding, bounds[2] + x_range * padding)
    ax.set_ylim(bounds[1] - y_range * padding, bounds[3] + y_range * padding)
    
    # Use ColorBrewer qualitative palette
    if k <= 8:
        colors = plt.cm.Set2(np.linspace(0, 1, k))
    elif k <= 12:
        colors = plt.cm.Set3(np.linspace(0, 1, k))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, k))
    
    # Plot clusters
    for cluster_id in range(k):
        cluster_data = gdf_rd[gdf_rd[f'cluster_k{k}'] == cluster_id]
        if len(cluster_data) > 0:
            cluster_data.plot(
                ax=ax,
                color=colors[cluster_id],
                edgecolor='white',
                linewidth=0.1,
                alpha=0.9,
                label=f'Cluster {cluster_id + 1}'
            )
    
    # Add Netherlands boundary (if available)
    try:
        nl_boundary = gpd.read_file('data/study_areas/netherlands/area_gdf/netherlands_boundary.geojson')
        nl_boundary_rd = nl_boundary.to_crs('EPSG:28992')
        nl_boundary_rd.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.7, linestyle='--')
    except:
        pass
    
    # Add grid
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')
    
    # Format axes
    ax.set_xlabel('X Coordinate (m) - Dutch RD', fontsize=11)
    ax.set_ylabel('Y Coordinate (m) - Dutch RD', fontsize=11)
    
    # Format tick labels
    ax.ticklabel_format(style='plain', axis='both')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y):,}'))
    
    # Add title
    title = f'AlphaEarth Netherlands 2022 - K-means Clustering (K={k})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add subtitle with metadata
    subtitle = f'H3 Resolution 8 | {len(gdf)} hexagons | EPSG:28992 (Dutch RD)'
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, 
            fontsize=10, ha='center', alpha=0.7)
    
    # Add north arrow
    x_pos = bounds[0] + x_range * 0.9
    y_pos = bounds[1] + y_range * 0.85
    arrow = mpatches.FancyArrowPatch(
        (x_pos, y_pos), (x_pos, y_pos + y_range * 0.05),
        arrowstyle='->,head_width=0.4,head_length=0.8',
        lw=2, color='black'
    )
    ax.add_patch(arrow)
    ax.text(x_pos, y_pos + y_range * 0.07, 'N', fontsize=12, 
            fontweight='bold', ha='center')
    
    # Add scale bar
    scale_length = 10000  # 10 km
    scale_x = bounds[0] + x_range * 0.7
    scale_y = bounds[1] + y_range * 0.05
    
    ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
            'k-', linewidth=3)
    ax.plot([scale_x, scale_x], [scale_y - y_range * 0.005, scale_y + y_range * 0.005], 
            'k-', linewidth=2)
    ax.plot([scale_x + scale_length, scale_x + scale_length], 
            [scale_y - y_range * 0.005, scale_y + y_range * 0.005], 
            'k-', linewidth=2)
    ax.text(scale_x + scale_length/2, scale_y - y_range * 0.02, '10 km', 
            fontsize=9, ha='center')
    
    # Add legend
    handles = []
    for i in range(k):
        count = (gdf[f'cluster_k{k}'] == i).sum()
        patch = mpatches.Patch(color=colors[i], label=f'Cluster {i+1} (n={count})')
        handles.append(patch)
    
    legend = ax.legend(handles=handles, loc='lower left', 
                      frameon=True, fancybox=True, shadow=True,
                      ncol=2 if k > 6 else 1, fontsize=9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add data source annotation
    ax.text(0.99, 0.01, 'Data: AlphaEarth 2022 | Processing: UrbanRepML', 
            transform=ax.transAxes, fontsize=8, ha='right', 
            alpha=0.5, style='italic')
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"     Saved to: {output_path}")


def main():
    """Main execution pipeline."""
    
    try:
        # Process AlphaEarth data
        gdf = process_alphaearth_2022()
        
        # Apply kmeans_clustering_1layer
        k_values = [8, 10, 12]
        gdf_clustered, clustering_results = apply_clustering(gdf, k_values)
        
        # Create output directories
        results_dir = Path('results [old 2024]/plots/netherlands')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings_dir = Path('results [old 2024]/embeddings/netherlands')
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        print("\n3. Creating cartographic visualizations...")
        for k in k_values:
            output_path = results_dir / f'alphaearth_clusters_k{k}_2022.png'
            create_cartographic_map(gdf_clustered, k, output_path)
        
        # Save clustered data
        print("\n4. Saving clustered embeddings...")
        clustered_file = embeddings_dir / 'alphaearth_res8_clustered_2022.parquet'
        gdf_clustered.to_parquet(clustered_file)
        print(f"   Saved to: {clustered_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("Processing Complete!")
        print("="*60)
        print(f"\nSummary:")
        print(f"  - Processed hexagons: {len(gdf)}")
        print(f"  - H3 resolution: 8")
        print(f"  - Embedding dimensions: 64")
        print(f"  - Clustering results [old 2024]:")
        for k, results in clustering_results.items():
            print(f"    K={k}: Silhouette={results['silhouette']:.3f}")
        print(f"\nOutput files:")
        print(f"  - Embeddings: {clustered_file}")
        print(f"  - Visualizations: {results_dir}/")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())