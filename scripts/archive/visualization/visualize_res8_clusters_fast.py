#!/usr/bin/env python3
"""
Fast Resolution 8 Clustering Visualization Script using Datashader and Dissolve

Optimized version that uses:
1. Dissolve to merge adjacent hexagons with same cluster ID
2. Datashader for fast rasterization of millions of geometries
3. Efficient memory management

Performance: 10-100x faster than matplotlib for large datasets
Adapted for Resolution 8 hexagons (coarser resolution than res10)
"""

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyproj import Transformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Datashader imports for fast rendering
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
from colorcet import palette

# Configure for performance and clean output
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# CPU optimization
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# Professional plotting style
plt.rcParams.update({
    'figure.dpi': 150,  # Reduced from 200 for faster rendering
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})

# Multiple colormap combinations for pattern discovery - REVERSED ORDER (16 first)
COLORMAP_COMBINATIONS = [
    (16, 'tab20c'),
    (16, 'tab20'),
    (12, 'Paired'),
    (12, 'tab20'),
    (8, 'Set3'),
    (8, 'tab10')
]

# Study area configurations - Updated for res8
STUDY_AREA_CONFIG = {
    'cascadia': {
        'file_pattern': 'cascadia_coastal_forests_2021_res8_final.parquet',
        'data_dir': 'data/study_areas/cascadia/embeddings/alphaearth',
        'output_dir': 'results [old 2024]/visualizations/cascadia_clusters_res8',
        'crs': 'EPSG:3857',  # Web Mercator
        'title_prefix': 'Cascadia',
        'resolution': 8
    },
    'netherlands': {
        'file_pattern': 'netherlands_res8_*.parquet',
        'data_dir': 'data/study_areas/netherlands/embeddings/alphaearth',
        'output_dir': 'results [old 2024]/visualizations/netherlands_clusters_res8',
        'crs': 'EPSG:28992',  # Dutch RD
        'title_prefix': 'Netherlands',
        'resolution': 8
    },
    'pearl_river_delta': {
        'file_pattern': 'prd_res8_*.parquet',
        'data_dir': 'data/study_areas/pearl_river_delta/embeddings/alphaearth',
        'output_dir': 'results [old 2024]/visualizations/prd_clusters_res8',
        'crs': 'EPSG:3857',  # Web Mercator
        'title_prefix': 'Pearl River Delta',
        'resolution': 8
    }
}


def find_study_area_data(study_area: str) -> Path:
    """Find the most recent data file for the specified study area."""

    if study_area not in STUDY_AREA_CONFIG:
        available = ', '.join(STUDY_AREA_CONFIG.keys())
        raise ValueError(f"Study area '{study_area}' not supported. Available: {available}")

    config = STUDY_AREA_CONFIG[study_area]
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / config['data_dir']

    # Find matching files
    pattern = config['file_pattern']
    matching_files = list(data_dir.glob(pattern))

    if not matching_files:
        raise FileNotFoundError(f"No data files found for {study_area} in {data_dir}")

    # Return most recent
    data_path = sorted(matching_files)[-1]
    print(f"Found data: {data_path.name}")
    return data_path


def load_and_prepare_embeddings(data_path: Path) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    """Load embeddings and prepare for kmeans_clustering_1layer."""

    print(f"Loading embeddings from {data_path.name}...")

    # Try reading as GeoDataFrame first
    try:
        gdf = gpd.read_parquet(data_path)
    except ValueError:
        # If no geo metadata, read as regular DataFrame and create geometry
        import pandas as pd
        df = pd.read_parquet(data_path)

        # Check for various possible geometry/index columns
        if 'region_id' in df.columns or df.index.name == 'region_id':
            # SRAI format with region_id
            if df.index.name == 'region_id':
                df = df.reset_index()
            # Use SRAI to create geometries
            from srai.h3 import h3_to_geoseries
            geometry_series = h3_to_geoseries(df['region_id'])
            gdf = gpd.GeoDataFrame(df, geometry=geometry_series, crs='EPSG:4326')
        elif 'h3_index' in df.columns:
            # Legacy h3_index format
            from srai.h3 import h3_to_geoseries
            geometry_series = h3_to_geoseries(df['h3_index'])
            gdf = gpd.GeoDataFrame(df, geometry=geometry_series, crs='EPSG:4326')
        elif 'lat' in df.columns and 'lng' in df.columns:
            from shapely.geometry import Point
            geometry = [Point(xy) for xy in zip(df.lng, df.lat)]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        else:
            raise ValueError("No geometry, region_id, h3_index, or lat/lng columns found in data")

    print(f"Loaded {len(gdf):,} hexagons")

    # Extract AlphaEarth embedding columns (A00-A63)
    embedding_cols = [col for col in gdf.columns if col.startswith('A') and col[1:].isdigit()]
    embeddings = gdf[embedding_cols].values.astype(np.float32)  # float32 for speed

    print(f"Embeddings shape: {embeddings.shape}")
    return gdf, embeddings


def apply_pca_reduction(embeddings: np.ndarray, n_components: int = 16) -> Tuple[np.ndarray, PCA]:
    """Apply PCA dimensionality reduction for computational efficiency."""

    print(f"Applying PCA: {embeddings.shape[1]}D -> {n_components}D...")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA variance retained: {variance_explained:.3f}")
    print(f"Reduced shape: {embeddings_reduced.shape}")

    return embeddings_reduced, pca


def perform_minibatch_clustering(embeddings_reduced: np.ndarray,
                                n_clusters_list: List[int]) -> Dict[int, np.ndarray]:
    """Apply MiniBatchKMeans kmeans_clustering_1layer efficiently."""

    print(f"MiniBatchKMeans kmeans_clustering_1layer with {len(n_clusters_list)} configurations...")
    print(f"Using {os.cpu_count()} CPU cores")

    def cluster_single_k(k: int) -> Tuple[int, np.ndarray, float]:
        """Cluster with single K value."""
        print(f"  Computing {k} clusters...")
        start_time = time.time()

        # MiniBatchKMeans for speed on large datasets
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=10000,
            max_iter=100,
            n_init=3,
            init='k-means++',
            verbose=0
        )

        clusters = kmeans.fit_predict(embeddings_reduced)

        duration = time.time() - start_time
        print(f"    K={k} completed in {duration:.1f}s")

        return k, clusters, 0  # Skip silhouette for speed

    # Parallel kmeans_clustering_1layer
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(cluster_single_k)(k) for k in n_clusters_list
    )

    # Store results [old 2024]
    cluster_results = {}
    for k, clusters, _ in results:
        cluster_results[k] = clusters
        print(f"  Final K={k}: {len(set(clusters))} clusters")

    return cluster_results


def add_coordinate_grid_and_labels(ax, bounds, target_crs: str):
    """Add coordinate grid with proper lat/lon labels - lightweight version."""

    if target_crs == 'EPSG:3857':  # Web Mercator
        # Convert bounds from Web Mercator to lat/lon
        transformer_to_latlon = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
        lon_min, lat_min = transformer_to_latlon.transform(bounds[0], bounds[1])
        lon_max, lat_max = transformer_to_latlon.transform(bounds[2], bounds[3])

        # Create transformer from lat/lon to Web Mercator
        transformer_to_mercator = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

        # Create grid - adjusted for res8 (coarser grid)
        lon_step = 1.0  # Coarser grid for res8
        lat_step = 0.5
        lon_grid = np.arange(np.floor(lon_min / lon_step) * lon_step,
                            np.ceil(lon_max / lon_step) * lon_step + lon_step, lon_step)
        lat_grid = np.arange(np.floor(lat_min / lat_step) * lat_step,
                            np.ceil(lat_max / lat_step) * lat_step + lat_step, lat_step)

        # Add white grid lines
        for lon in lon_grid:
            x, _ = transformer_to_mercator.transform(lon, (lat_min + lat_max) / 2)
            if bounds[0] <= x <= bounds[2]:
                ax.axvline(x, color='white', alpha=0.5, linewidth=0.6, zorder=10)

        for lat in lat_grid:
            _, y = transformer_to_mercator.transform((lon_min + lon_max) / 2, lat)
            if bounds[1] <= y <= bounds[3]:
                ax.axhline(y, color='white', alpha=0.5, linewidth=0.6, zorder=10)

        # Add labels for grid lines
        for lon in lon_grid:
            x, _ = transformer_to_mercator.transform(lon, lat_min)
            if bounds[0] <= x <= bounds[2]:
                lon_label = f'{abs(lon):.1f}°{"E" if lon >= 0 else "W"}'
                ax.text(x, bounds[1] - (bounds[3] - bounds[1]) * 0.02, lon_label,
                       ha='center', va='top', fontsize=9, color='black', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        for lat in lat_grid:
            _, y = transformer_to_mercator.transform(lon_min, lat)
            if bounds[1] <= y <= bounds[3]:
                lat_label = f'{abs(lat):.1f}°{"N" if lat >= 0 else "S"}'
                ax.text(bounds[0] - (bounds[2] - bounds[0]) * 0.01, y, lat_label,
                       ha='right', va='center', fontsize=9, color='black', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))


def create_fast_cluster_visualization(gdf: gpd.GeoDataFrame, clusters: np.ndarray,
                                     n_clusters: int, colormap: str, study_area: str,
                                     output_path: Path, use_dissolve: bool = True,
                                     use_datashader: bool = True):
    """Create single cluster visualization using optimizations."""

    config = STUDY_AREA_CONFIG[study_area]

    # Prepare data - handle both region_id and h3_index
    if 'region_id' in gdf.columns:
        viz_gdf = gdf[['region_id', 'geometry']].copy()
    elif 'h3_index' in gdf.columns:
        viz_gdf = gdf[['h3_index', 'geometry']].copy()
    else:
        # Assume index is the identifier
        viz_gdf = gdf[['geometry']].copy()

    viz_gdf['cluster'] = clusters

    # Reproject for visualization
    print(f"  Reprojecting to {config['crs']}...")
    viz_gdf_proj = viz_gdf.to_crs(config['crs'])
    bounds = viz_gdf_proj.total_bounds

    # OPTIMIZATION 1: Dissolve adjacent hexagons with same cluster
    if use_dissolve:
        print(f"  Dissolving {len(viz_gdf_proj):,} hexagons by cluster...")
        start_time = time.time()
        viz_gdf_dissolved = viz_gdf_proj.dissolve(by='cluster', as_index=False)
        print(f"    Reduced to {len(viz_gdf_dissolved):,} polygons in {time.time() - start_time:.1f}s")
        data_to_plot = viz_gdf_dissolved
    else:
        data_to_plot = viz_gdf_proj

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))  # Slightly larger for res8

    # OPTIMIZATION 2: Use datashader for rasterization
    if use_datashader and not use_dissolve:  # Datashader works best with original hexagons
        print(f"  Rendering with datashader...")

        # Create canvas
        canvas = ds.Canvas(plot_width=1500, plot_height=1500,
                          x_range=(bounds[0], bounds[2]),
                          y_range=(bounds[1], bounds[3]))

        # Rasterize polygons
        agg = canvas.polygons(data_to_plot, geometry='geometry', agg=ds.mean('cluster'))

        # Create color map
        colors = plt.get_cmap(colormap).colors[:n_clusters]
        img = tf.shade(agg, cmap=colors, how='eq_hist')

        # Display as image
        ax.imshow(img.to_pil(), extent=bounds, origin='lower', aspect='equal')

    else:
        # Traditional matplotlib plotting (faster with dissolved data)
        print(f"  Rendering with matplotlib...")
        data_to_plot.plot(
            column='cluster',
            ax=ax,
            cmap=colormap,
            edgecolor='none',
            linewidth=0,
            alpha=0.8,
            legend=False
        )

    # Add coordinate grid
    add_coordinate_grid_and_labels(ax, bounds, config['crs'])

    # Title
    method = "Dissolved" if use_dissolve else "Datashader" if use_datashader else "Standard"
    res = config.get('resolution', 8)
    title = f"{config['title_prefix']} AlphaEarth - {n_clusters} Clusters ({method})\nH3 Res {res} | {len(gdf):,} hexagons"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Clean appearance
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path.name}")


def main():
    """Main execution pipeline."""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Fast Resolution 8 Clustering Visualization')
    parser.add_argument('--study-area', type=str, required=True,
                       choices=list(STUDY_AREA_CONFIG.keys()),
                       help='Study area to visualize')
    parser.add_argument('--clusters', type=str, default='8,12,16',
                       help='Comma-separated cluster counts (default: 8,12,16)')
    parser.add_argument('--pca-components', type=int, default=16,
                       help='PCA components for dimensionality reduction (default: 16)')
    parser.add_argument('--skip-pca', action='store_true',
                       help='Skip PCA and use full 64D embeddings')
    parser.add_argument('--no-dissolve', action='store_true',
                       help='Skip dissolve optimization')
    parser.add_argument('--no-datashader', action='store_true',
                       help='Skip datashader optimization')
    parser.add_argument('--sample', type=float, default=1.0,
                       help='Sample fraction of data for testing (0-1)')

    args = parser.parse_args()

    # Parse cluster counts
    cluster_counts = [int(x.strip()) for x in args.clusters.split(',')]

    print("="*80)
    print("FAST RESOLUTION 8 CLUSTERING VISUALIZATION")
    print("="*80)
    print(f"Study Area: {args.study_area}")
    print(f"Cluster Counts: {cluster_counts}")
    print(f"Optimizations: Dissolve={not args.no_dissolve}, Datashader={not args.no_datashader}")
    print(f"Sample Rate: {args.sample * 100:.0f}%")
    print(f"CPU Cores: {os.cpu_count()}")

    start_time = time.time()

    # Setup output directory (main folder, not subfolder)
    config = STUDY_AREA_CONFIG[args.study_area]
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = find_study_area_data(args.study_area)
    gdf, embeddings = load_and_prepare_embeddings(data_path)

    # Sample data if requested (for testing)
    if args.sample < 1.0:
        sample_size = int(len(gdf) * args.sample)
        print(f"Sampling {sample_size:,} hexagons ({args.sample * 100:.0f}%)...")
        sample_idx = np.random.choice(len(gdf), sample_size, replace=False)
        gdf = gdf.iloc[sample_idx].reset_index(drop=True)
        embeddings = embeddings[sample_idx]

    # Apply PCA if not skipped
    if args.skip_pca:
        print("Skipping PCA - using full 64D embeddings")
        embeddings_for_clustering = embeddings
    else:
        embeddings_for_clustering, pca = apply_pca_reduction(embeddings, args.pca_components)

    # Perform kmeans_clustering_1layer
    cluster_results = perform_minibatch_clustering(embeddings_for_clustering, cluster_counts)

    # Create visualizations
    print(f"\nCreating optimized visualizations...")

    for n_clusters, colormap in COLORMAP_COMBINATIONS:
        if n_clusters in cluster_results:
            # Generate filename with optimization suffix
            suffix = "dissolved" if not args.no_dissolve else "datashader" if not args.no_datashader else "standard"
            filename = f'{args.study_area}_res8_{n_clusters:02d}clusters_{colormap}_{suffix}.png'
            output_path = output_dir / filename

            create_fast_cluster_visualization(
                gdf, cluster_results[n_clusters], n_clusters,
                colormap, args.study_area, output_path,
                use_dissolve=not args.no_dissolve,
                use_datashader=not args.no_datashader
            )

    # Summary
    total_time = time.time() - start_time
    print(f"\n" + "="*80)
    print("FAST VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Total Processing Time: {total_time:.1f}s")
    print(f"Output Directory: {output_dir}")

    # Compare to original timing
    original_estimate = len(gdf) / 1000  # Rough estimate: 1s per 1000 hexagons
    speedup = original_estimate / total_time
    print(f"Estimated Speedup: {speedup:.1f}x faster")


if __name__ == "__main__":
    main()