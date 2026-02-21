"""
Resolution-Agnostic Cluster Visualization

Consolidated from:
- visualize_res10_clusters_fast.py
- visualize_res8_clusters_fast.py
- visualize_hierarchical_embeddings_fast.py

Golden combo: dissolve + MiniBatchKMeans + datashader
"""

import logging
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

# Datashader imports for fast rendering (optional)
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from colorcet import palette
    HAS_DATASHADER = True
except ImportError:
    HAS_DATASHADER = False

from utils import StudyAreaPaths
from utils.spatial_db import SpatialDB

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# CPU optimization
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())

# Default colormap combinations (n_clusters, colormap_name)
COLORMAP_COMBINATIONS = [
    (16, 'tab20c'),
    (16, 'tab20'),
    (12, 'Paired'),
    (12, 'tab20'),
    (8, 'Set3'),
    (8, 'tab10'),
]

# Study area configurations — merged from all three original scripts
# data_dir and output_dir are now derived from StudyAreaPaths at runtime.
STUDY_AREA_CONFIG = {
    'netherlands': {
        'file_pattern_by_res': {
            8: 'netherlands_res8_*.parquet',
            9: 'netherlands_res9_*.parquet',
            10: 'netherlands_res10_*.parquet',
        },
        'crs': 'EPSG:28992',
        'title_prefix': 'Netherlands',
    },
    'cascadia': {
        'file_pattern_by_res': {
            8: 'cascadia_coastal_forests_2021_res8_final.parquet',
            10: 'cascadia_res10_*.parquet',
        },
        'crs': 'EPSG:3857',
        'title_prefix': 'Cascadia',
    },
    'pearl_river_delta': {
        'file_pattern_by_res': {
            8: 'prd_res8_*.parquet',
            10: 'prd_res10_2023_FIXED_*.parquet',
        },
        'crs': 'EPSG:3857',
        'title_prefix': 'Pearl River Delta',
    },
}

# Scale names for hierarchical labels
SCALE_NAMES = {
    5: "Metro", 6: "City", 7: "District",
    8: "Neighborhood", 9: "Block", 10: "Building",
}

# Default cluster counts for hierarchical mode (auto-scale by resolution)
DEFAULT_HIERARCHICAL_CLUSTERS = {
    5: 8, 6: 10, 7: 12, 8: 14, 9: 16, 10: 16,
}


def find_study_area_data(study_area: str, resolution: int, modality: str = "alphaearth") -> Path:
    """Find the most recent data file for a study area at a given resolution."""
    if study_area not in STUDY_AREA_CONFIG:
        available = ', '.join(STUDY_AREA_CONFIG.keys())
        raise ValueError(f"Study area '{study_area}' not supported. Available: {available}")

    config = STUDY_AREA_CONFIG[study_area]
    paths = StudyAreaPaths(study_area)
    data_dir = paths.stage1(modality)

    # Try resolution-specific pattern first
    patterns = config.get('file_pattern_by_res', {})
    pattern = patterns.get(resolution)

    if pattern is None:
        # Fallback: generic pattern
        pattern = f'{study_area}_res{resolution}_*.parquet'

    matching_files = list(data_dir.glob(pattern))
    if not matching_files:
        raise FileNotFoundError(
            f"No data files found for {study_area} res{resolution} in {data_dir} "
            f"(pattern: {pattern})"
        )

    data_path = sorted(matching_files)[-1]
    print(f"Found data: {data_path.name}")
    return data_path


def load_and_prepare_embeddings(
    data_path: Path,
    study_area: str = "netherlands",
    resolution: int = 10,
) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    """
    Load embeddings and prepare for kmeans_clustering_1layer.

    Supports both SRAI region_id format and legacy h3_index format.
    Uses SpatialDB for geometry lookup from pre-computed region files.
    Returns (GeoDataFrame with geometry, embedding numpy array).
    """
    print(f"Loading embeddings from {data_path.name}...")

    db = SpatialDB.for_study_area(study_area)

    try:
        gdf = gpd.read_parquet(data_path)
    except ValueError:
        df = pd.read_parquet(data_path)

        # Handle both region_id (SRAI standard) and h3_index (legacy)
        if 'region_id' in df.columns or df.index.name == 'region_id':
            if df.index.name == 'region_id':
                df = df.reset_index()
            geom_gdf = db.geometry(df['region_id'], resolution=resolution, crs=4326)
            gdf = gpd.GeoDataFrame(df, geometry=geom_gdf.geometry.values, crs='EPSG:4326')
        elif 'h3_index' in df.columns:
            geom_gdf = db.geometry(df['h3_index'], resolution=resolution, crs=4326)
            gdf = gpd.GeoDataFrame(df, geometry=geom_gdf.geometry.values, crs='EPSG:4326')
        elif 'lat' in df.columns and 'lng' in df.columns:
            from shapely.geometry import Point
            geometry = [Point(xy) for xy in zip(df.lng, df.lat)]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        else:
            raise ValueError("No geometry, region_id, h3_index, or lat/lng columns found")

    print(f"Loaded {len(gdf):,} hexagons")

    # Extract embedding columns
    # Try known prefix patterns, fall back to all numeric columns
    embedding_cols = [col for col in gdf.columns if col.startswith(('A', 'P', 'R', 'S', 'G')) and len(col) >= 2 and col[1:].isdigit()]
    if not embedding_cols:
        exclude = {"pixel_count", "tile_count", "geometry", "h3_index", "region_id"}
        embedding_cols = [col for col in gdf.columns if col not in exclude and pd.api.types.is_numeric_dtype(gdf[col])]
    embeddings = gdf[embedding_cols].values.astype(np.float32)

    print(f"Embeddings shape: {embeddings.shape}")
    return gdf, embeddings


def apply_pca_reduction(embeddings: np.ndarray, n_components: int = 16) -> Tuple[np.ndarray, PCA]:
    """Apply PCA dimensionality reduction for computational efficiency."""
    print(f"Applying PCA: {embeddings.shape[1]}D -> {n_components}D...")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA variance retained: {variance_explained:.3f}")
    return embeddings_reduced, pca


def perform_minibatch_clustering(
    embeddings_reduced: np.ndarray,
    n_clusters_list: List[int],
    standardize: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Apply MiniBatchKMeans kmeans_clustering_1layer efficiently.

    Args:
        embeddings_reduced: Pre-processed embedding matrix
        n_clusters_list: List of cluster counts to compute
        standardize: Whether to standardize before kmeans_clustering_1layer

    Returns:
        Dict mapping n_clusters -> cluster label array
    """
    print(f"MiniBatchKMeans kmeans_clustering_1layer with {len(n_clusters_list)} configurations...")

    data = embeddings_reduced
    if standardize:
        data = StandardScaler().fit_transform(data)

    def cluster_single_k(k: int) -> Tuple[int, np.ndarray]:
        start_time = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=min(10000, len(data)),
            max_iter=100,
            n_init=3,
            init='k-means++',
            verbose=0,
        )
        clusters = kmeans.fit_predict(data)
        duration = time.time() - start_time
        print(f"  K={k} completed in {duration:.1f}s")
        return k, clusters

    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(cluster_single_k)(k) for k in n_clusters_list
    )

    cluster_results = {}
    for k, clusters in results:
        cluster_results[k] = clusters
        print(f"  Final K={k}: {len(set(clusters))} clusters")

    return cluster_results


def add_coordinate_grid_and_labels(ax, bounds, target_crs: str):
    """Add coordinate grid with proper lat/lon labels."""
    if target_crs == 'EPSG:3857':
        transformer_to_latlon = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
        lon_min, lat_min = transformer_to_latlon.transform(bounds[0], bounds[1])
        lon_max, lat_max = transformer_to_latlon.transform(bounds[2], bounds[3])

        transformer_to_mercator = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

        lon_step = 0.5
        lat_step = 0.5
        lon_grid = np.arange(
            np.floor(lon_min / lon_step) * lon_step,
            np.ceil(lon_max / lon_step) * lon_step + lon_step,
            lon_step,
        )
        lat_grid = np.arange(
            np.floor(lat_min / lat_step) * lat_step,
            np.ceil(lat_max / lat_step) * lat_step + lat_step,
            lat_step,
        )

        for lon in lon_grid:
            x, _ = transformer_to_mercator.transform(lon, (lat_min + lat_max) / 2)
            if bounds[0] <= x <= bounds[2]:
                ax.axvline(x, color='white', alpha=0.5, linewidth=0.6, zorder=10)

        for lat in lat_grid:
            _, y = transformer_to_mercator.transform((lon_min + lon_max) / 2, lat)
            if bounds[1] <= y <= bounds[3]:
                ax.axhline(y, color='white', alpha=0.5, linewidth=0.6, zorder=10)

        for lon in lon_grid:
            x, _ = transformer_to_mercator.transform(lon, lat_min)
            if bounds[0] <= x <= bounds[2]:
                lon_label = f'{abs(lon):.1f}°{"E" if lon >= 0 else "W"}'
                ax.text(
                    x, bounds[1] - (bounds[3] - bounds[1]) * 0.02, lon_label,
                    ha='center', va='top', fontsize=9, color='black', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                )

        for lat in lat_grid:
            _, y = transformer_to_mercator.transform(lon_min, lat)
            if bounds[1] <= y <= bounds[3]:
                lat_label = f'{abs(lat):.1f}°{"N" if lat >= 0 else "S"}'
                ax.text(
                    bounds[0] - (bounds[2] - bounds[0]) * 0.01, y, lat_label,
                    ha='right', va='center', fontsize=9, color='black', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                )


def create_cluster_visualization(
    gdf: gpd.GeoDataFrame,
    clusters: np.ndarray,
    n_clusters: int,
    colormap: str,
    output_path: Path,
    target_crs: str = 'EPSG:28992',
    title_prefix: str = '',
    resolution: int = 10,
    use_dissolve: bool = True,
    use_datashader: bool = True,
    figsize: Tuple[int, int] = (12, 12),
) -> None:
    """
    Create a single cluster visualization using dissolve + matplotlib/datashader.

    Args:
        gdf: GeoDataFrame with hexagon geometries
        clusters: Cluster label array
        n_clusters: Number of clusters
        colormap: Matplotlib colormap name
        output_path: Path to save the figure
        target_crs: Target CRS for reprojection
        title_prefix: Study area name for title
        resolution: H3 resolution for title
        use_dissolve: Merge adjacent same-cluster hexagons
        use_datashader: Use datashader for rasterization (only without dissolve)
        figsize: Figure size
    """
    # Identify the hex ID column
    id_cols = [c for c in ['region_id', 'h3_index'] if c in gdf.columns]
    keep_cols = id_cols + ['geometry']
    viz_gdf = gdf[[c for c in keep_cols if c in gdf.columns]].copy()
    viz_gdf['cluster'] = clusters

    print(f"  Reprojecting to {target_crs}...")
    viz_gdf_proj = viz_gdf.to_crs(target_crs)
    bounds = viz_gdf_proj.total_bounds

    if use_dissolve:
        print(f"  Dissolving {len(viz_gdf_proj):,} hexagons by cluster...")
        start_time = time.time()
        viz_dissolved = viz_gdf_proj.dissolve(by='cluster', as_index=False)
        print(f"    Reduced to {len(viz_dissolved):,} polygons in {time.time() - start_time:.1f}s")
        data_to_plot = viz_dissolved
    else:
        data_to_plot = viz_gdf_proj

    fig, ax = plt.subplots(figsize=figsize)

    if use_datashader and not use_dissolve and HAS_DATASHADER:
        print(f"  Rendering with datashader...")
        canvas = ds.Canvas(
            plot_width=1500, plot_height=1500,
            x_range=(bounds[0], bounds[2]),
            y_range=(bounds[1], bounds[3]),
        )
        agg = canvas.polygons(data_to_plot, geometry='geometry', agg=ds.mean('cluster'))
        colors = plt.get_cmap(colormap).colors[:n_clusters]
        img = tf.shade(agg, cmap=colors, how='eq_hist')
        ax.imshow(img.to_pil(), extent=bounds, origin='lower', aspect='equal')
    else:
        print(f"  Rendering with matplotlib...")
        data_to_plot.plot(
            column='cluster', ax=ax, cmap=colormap,
            edgecolor='none', linewidth=0, alpha=0.8, legend=False,
        )

    add_coordinate_grid_and_labels(ax, bounds, target_crs)

    method = "Dissolved" if use_dissolve else "Datashader" if use_datashader else "Standard"
    title = (
        f"{title_prefix} Embeddings - {n_clusters} Clusters ({method})\n"
        f"H3 Res {resolution} | {len(gdf):,} hexagons"
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path.name}")


def create_hierarchical_subplot(
    study_area: str,
    resolutions: List[int],
    n_clusters_dict: Optional[Dict[int, int]] = None,
    colormap: str = 'tab20',
    output_path: Optional[Path] = None,
    year: str = "2022",
    use_pca: bool = True,
    pca_components: int = 16,
    crs: str = 'EPSG:28992',
) -> Path:
    """
    Create multi-resolution hierarchical subplot grid.

    Args:
        study_area: Study area name
        resolutions: List of H3 resolutions to visualize
        n_clusters_dict: Mapping resolution -> n_clusters (defaults to auto-scale)
        colormap: Matplotlib colormap
        output_path: Where to save the combined figure
        year: Data year
        use_pca: Apply PCA reduction
        pca_components: Number of PCA components
        crs: Target CRS

    Returns:
        Path to saved figure
    """
    if n_clusters_dict is None:
        n_clusters_dict = {r: DEFAULT_HIERARCHICAL_CLUSTERS.get(r, 10) for r in resolutions}

    if output_path is None:
        paths = StudyAreaPaths(study_area)
        output_path = paths.stage3("kmeans_clustering_1layer") / "hierarchical_fast"
    output_path.mkdir(parents=True, exist_ok=True)

    n_res = len(resolutions)
    ncols = 3
    nrows = int(np.ceil(n_res / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes = axes.flatten() if n_res > 1 else [axes]

    for idx, resolution in enumerate(resolutions):
        logger.info(f"Processing Resolution {resolution}")

        try:
            data_path = find_study_area_data(study_area, resolution)
            gdf, embeddings = load_and_prepare_embeddings(data_path)

            n_clusters = n_clusters_dict.get(resolution, 10)

            # PCA + standardize + cluster
            if use_pca and embeddings.shape[1] > pca_components:
                pca = PCA(n_components=pca_components, random_state=42)
                embeddings_reduced = pca.fit_transform(embeddings)
            else:
                embeddings_reduced = embeddings

            embeddings_scaled = StandardScaler().fit_transform(embeddings_reduced)

            start_time = time.time()
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=42,
                batch_size=min(10000, len(embeddings)),
                max_iter=100, n_init=3, init='k-means++', verbose=0,
            )
            clusters = kmeans.fit_predict(embeddings_scaled)
            cluster_time = time.time() - start_time

            # Dissolved visualization into subplot
            viz_gdf = gdf[['geometry']].copy()
            viz_gdf['cluster'] = clusters
            viz_gdf_proj = viz_gdf.to_crs(crs)

            viz_dissolved = viz_gdf_proj.dissolve(by='cluster', as_index=False)
            viz_dissolved.plot(
                column='cluster', ax=axes[idx], cmap=colormap,
                edgecolor='none', linewidth=0, alpha=0.85, legend=False, aspect='equal',
            )

            bounds = viz_gdf_proj.total_bounds
            axes[idx].set_xlim(bounds[0], bounds[2])
            axes[idx].set_ylim(bounds[1], bounds[3])
            axes[idx].set_aspect('equal', adjustable='box')

            scale_name = SCALE_NAMES.get(resolution, 'Unknown')
            title = f"Res {resolution} ({scale_name})\n{len(gdf):,} hexagons -> {n_clusters} clusters"
            axes[idx].set_title(title, fontsize=11, fontweight='bold')

            # North arrow
            x, y, arrow_length = 0.95, 0.95, 0.08
            axes[idx].annotate(
                'N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=3, headwidth=8),
                ha='center', va='center', fontsize=12, fontweight='bold',
                xycoords=axes[idx].transAxes,
            )
            axes[idx].set_axis_off()

        except Exception as e:
            logger.error(f"Error processing res{resolution}: {e}")
            axes[idx].text(
                0.5, 0.5, f"Res {resolution}\nError: {str(e)}",
                ha='center', va='center', transform=axes[idx].transAxes,
            )
            axes[idx].set_axis_off()

    # Hide unused subplots
    for idx in range(n_res, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"{study_area.title()} - Hierarchical Multi-Resolution Clustering\n"
        f"Embeddings {year} | MiniBatch K-Means + Dissolve",
        fontsize=16, fontweight='bold', y=0.98,
    )

    output_file = output_path / f"{study_area}_hierarchical_{colormap}_fast.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved hierarchical plot: {output_file}")
    return output_file
