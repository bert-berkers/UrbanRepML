"""
Create K-means clustering visualizations for AlphaEarth resolution 10 data
without hexagon outlines for cleaner appearance
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Load AlphaEarth resolution 10 data."""
    
    logger.info("Loading AlphaEarth resolution 10 data...")
    
    data_file = Path('data/processed/embeddings/alphaearth/netherlands_res10_2022.parquet')
    
    if not data_file.exists():
        raise FileNotFoundError(f"Resolution 10 data not found: {data_file}")
    
    gdf = gpd.read_parquet(data_file)
    logger.info(f"  Loaded {len(gdf):,} hexagons")
    logger.info(f"  Columns: {gdf.columns.tolist()[:10]}...")
    
    # Check for embedding columns
    embedding_cols = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
    logger.info(f"  Embedding dimensions: {len(embedding_cols)}")
    
    return gdf, embedding_cols


def apply_clustering(gdf, embedding_cols, k_values=[8, 10, 12]):
    """Apply K-means clustering to resolution 10 embeddings."""
    
    logger.info(f"Applying K-means clustering to {len(gdf):,} hexagons...")
    
    # Extract embeddings
    X = gdf[embedding_cols].values
    logger.info(f"  Embedding matrix shape: {X.shape}")
    
    # Standardize features
    logger.info("  Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for k in k_values:
        logger.info(f"  Clustering with K={k}...")
        
        # Apply K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, labels)
        logger.info(f"    Silhouette score: {sil_score:.3f}")
        
        # Add cluster labels to GDF
        gdf[f'cluster_k{k}'] = labels
        
        # Log cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"    Cluster sizes: {dict(zip(unique, counts))}")
        
        results[k] = {
            'labels': labels,
            'silhouette': sil_score,
            'inertia': kmeans.inertia_
        }
    
    return gdf, results


def create_cartographic_plot(gdf, k, output_path):
    """Create professional cartographic visualization without hexagon outlines."""
    
    logger.info(f"Creating cartographic plot for K={k}...")
    
    # Transform to Dutch RD for accurate representation
    gdf_rd = gdf.to_crs('EPSG:28992')
    
    # Create figure
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
    
    # Plot clusters without outlines for cleaner appearance
    logger.info(f"  Plotting {k} clusters...")
    for cluster_id in range(k):
        cluster_data = gdf_rd[gdf_rd[f'cluster_k{k}'] == cluster_id]
        if len(cluster_data) > 0:
            cluster_data.plot(
                ax=ax,
                color=colors[cluster_id],
                edgecolor='none',  # No hexagon outlines
                linewidth=0,
                alpha=0.9,
                label=f'Cluster {cluster_id + 1}'
            )
            logger.info(f"    Cluster {cluster_id + 1}: {len(cluster_data):,} hexagons")
    
    # Add Netherlands boundary (if available)
    try:
        nl_boundary = gpd.read_file('data/processed/h3_regions/netherlands/netherlands_boundary.geojson')
        nl_boundary_rd = nl_boundary.to_crs('EPSG:28992')
        nl_boundary_rd.boundary.plot(ax=ax, color='black', linewidth=1.5, alpha=0.7, linestyle='--')
        logger.info("  Added Netherlands boundary")
    except Exception as e:
        logger.warning(f"  Could not add boundary: {e}")
    
    # Add subtle grid
    ax.grid(True, linestyle=':', alpha=0.2, color='gray', linewidth=0.5)
    
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
    subtitle = f'H3 Resolution 10 | {len(gdf):,} hexagons | EPSG:28992 (Dutch RD)'
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
    
    # Add legend with cluster counts
    handles = []
    for i in range(k):
        count = (gdf[f'cluster_k{k}'] == i).sum()
        patch = mpatches.Patch(color=colors[i], label=f'Cluster {i+1} (n={count:,})')
        handles.append(patch)
    
    legend = ax.legend(handles=handles, loc='lower left', 
                      frameon=True, fancybox=True, shadow=True,
                      ncol=2 if k > 6 else 1, fontsize=9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add data source annotation
    ax.text(0.99, 0.01, 'Data: AlphaEarth 2022 | Processing: UrbanRepML | Resolution: H3-10', 
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
    
    logger.info(f"  Saved plot to: {output_path}")


def main():
    """Main execution pipeline."""
    
    try:
        logger.info("="*60)
        logger.info("RESOLUTION 10 CLUSTERING VISUALIZATION")
        logger.info("="*60)
        
        # Load data
        gdf, embedding_cols = load_data()
        
        # Apply clustering
        k_values = [8, 10, 12]
        gdf_clustered, clustering_results = apply_clustering(gdf, embedding_cols, k_values)
        
        # Create output directory
        results_dir = Path('results/plots/netherlands')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        logger.info("Creating cartographic visualizations...")
        for k in k_values:
            output_path = results_dir / f'alphaearth_clusters_k{k}_res10_2022.png'
            create_cartographic_plot(gdf_clustered, k, output_path)
        
        # Save clustered data
        logger.info("Saving clustered embeddings...")
        embeddings_dir = Path('results/embeddings/netherlands')
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        clustered_file = embeddings_dir / 'alphaearth_res10_clustered_2022.parquet'
        gdf_clustered.to_parquet(clustered_file)
        logger.info(f"  Saved clustered data to: {clustered_file}")
        
        # Print summary
        logger.info("="*60)
        logger.info("VISUALIZATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Processed hexagons: {len(gdf):,}")
        logger.info(f"H3 resolution: 10")
        logger.info(f"Embedding dimensions: {len(embedding_cols)}")
        logger.info("Clustering results:")
        for k, results in clustering_results.items():
            logger.info(f"  K={k}: Silhouette={results['silhouette']:.3f}")
        
        logger.info("Output files:")
        logger.info(f"  Embeddings: {clustered_file}")
        logger.info(f"  Visualizations: {results_dir}/alphaearth_clusters_k*_res10_2022.png")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())