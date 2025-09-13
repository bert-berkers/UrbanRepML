"""
Full Resolution 10 Visualization with SRAI
==========================================
Visualize ALL 198K+ hexagons using existing SRAI infrastructure.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys

# Add current directory to path to import local modules
sys.path.append('.')
from scripts.srai_visualizations import SRAIVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_cluster_res10(n_clusters: int = 12):
    """Load res 10 data and create clusters for visualization."""
    
    # Load the resolution 10 data
    data_path = Path("data/h3_2021_res11/del_norte_2021_res10.parquet")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} hexagons at resolution 10")
    
    # Get band columns for clustering
    band_cols = [col for col in df.columns if col.startswith('band_')]
    logger.info(f"Found {len(band_cols)} feature bands")
    
    # Clean data - handle NaN values
    for col in band_cols:
        df[col] = df[col].fillna(df[col].mean())
        df[col] = np.clip(df[col], -1e10, 1e10)
    
    # Apply PCA for dimensionality reduction
    logger.info("Applying PCA for dimensionality reduction...")
    X = df[band_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=8, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Cluster the data
    logger.info(f"Clustering into {n_clusters} clusters...")
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(X_reduced)
    
    # Add cluster labels to dataframe
    df['cluster'] = labels
    
    logger.info(f"Created {len(np.unique(labels))} clusters")
    logger.info(f"Cluster distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    
    return df


def create_full_resolution_visualization():
    """Create visualization of all resolution 10 hexagons."""
    
    # Load and cluster the data
    df_clustered = load_and_cluster_res10(n_clusters=15)
    
    # Save the clustered data
    output_path = Path("results/clusters/full_res10_k15.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clustered.to_parquet(output_path)
    logger.info(f"Saved clustered data to {output_path}")
    
    # Create SRAI visualization config (matching your existing setup)
    config = {
        'experiment': {
            'h3_resolution': 10
        },
        'visualization': {
            'colormaps': ['tab20', 'Set1', 'Paired'],
            'figure_size': (20, 16),  # Large figure for detail
            'dpi': 150
        },
        'output': {
            'log_level': 'INFO'
        }
    }
    
    # Initialize SRAI visualizer
    logger.info("Initializing SRAI visualizer...")
    visualizer = SRAIVisualizer(config)
    
    # Load the clustered data as GeoDataFrame
    logger.info("Converting to GeoDataFrame (this may take a few minutes)...")
    gdf = visualizer.load_clustered_data(output_path)
    
    # Create static visualizations
    logger.info("Creating static visualizations...")
    for colormap in ['tab20', 'Set1', 'Paired']:
        fig = visualizer.create_static_cluster_map(
            gdf=gdf,
            method='kmeans',
            config_key='k15',
            colormap=colormap
        )
        
        # Save the figure
        plot_path = visualizer.static_dir / f"full_res10_kmeans_k15_{colormap}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved static plot: {plot_path}")
    
    # Create interactive map
    logger.info("Creating interactive map (this will take several minutes)...")
    interactive_map = visualizer.create_interactive_map(
        gdf=gdf,
        method='kmeans', 
        config_key='k15',
        colormap='tab20'
    )
    
    # Save interactive map
    map_path = visualizer.interactive_dir / "full_res10_kmeans_k15.html"
    interactive_map.save(str(map_path))
    logger.info(f"Saved interactive map: {map_path}")
    
    print(f"\\n✅ FULL RESOLUTION 10 VISUALIZATION COMPLETE")
    print(f"📊 Total hexagons visualized: {len(gdf):,}")
    print(f"🎨 Clusters created: {gdf['cluster'].nunique()}")
    print(f"📁 Static plots: {visualizer.static_dir}")
    print(f"🌐 Interactive map: {map_path}")


if __name__ == "__main__":
    create_full_resolution_visualization()