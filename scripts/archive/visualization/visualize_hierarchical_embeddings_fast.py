#!/usr/bin/env python3
"""
Fast Hierarchical Multi-Resolution Clustering Visualization
============================================================

Combines the hierarchical approach with fast MiniBatch K-Means and dissolve optimization.

Applies to aggregated AlphaEarth embeddings at resolutions 5-10:
- Res 5: Metro (239 hexagons)
- Res 6: City (1,384 hexagons)
- Res 7: District (8,748 hexagons)
- Res 8: Neighborhood (58,127 hexagons)
- Res 9: Block (398,931 hexagons)
- Res 10: Building (2,769,174 hexagons)

Uses:
- MiniBatch K-Means for speed
- Dissolve to merge adjacent hexagons
- Multi-resolution subplot layout
"""

import argparse
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import h3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from shapely.geometry import Polygon

# Configure
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())

# Professional plotting
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'font.family': 'sans-serif'
})


def load_resolution_embeddings(study_area: str, resolution: int, year: str = "2022") -> Tuple[gpd.GeoDataFrame, np.ndarray]:
    """Load embeddings for a specific resolution."""

    embeddings_path = Path(f"data/study_areas/{study_area}/embeddings/alphaearth/{study_area}_res{resolution}_{year}.parquet")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    # Load data
    df = pd.read_parquet(embeddings_path)

    # Get embedding columns
    embedding_cols = [col for col in df.columns if col.startswith('A')]
    embeddings = df[embedding_cols].values.astype(np.float32)

    # Create geometries from h3_index
    if 'h3_index' in df.columns:
        # IMPORTANT: h3.cell_to_boundary returns (lat, lon), but Polygon expects (lon, lat)
        geometries = [
            Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h3_idx)])
            for h3_idx in df['h3_index']
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs='EPSG:4326')
    else:
        gdf = gpd.GeoDataFrame(df, crs='EPSG:4326')

    logger.info(f"Loaded res{resolution}: {len(gdf):,} hexagons, {embeddings.shape[1]} bands")

    return gdf, embeddings


def perform_fast_clustering(embeddings: np.ndarray, n_clusters: int, use_pca: bool = True,
                           pca_components: int = 16) -> Tuple[np.ndarray, float]:
    """Perform fast MiniBatch K-Means clustering."""

    # Optional PCA for speed
    if use_pca and embeddings.shape[1] > pca_components:
        pca = PCA(n_components=pca_components, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)
        variance = pca.explained_variance_ratio_.sum()
        logger.info(f"  PCA: {embeddings.shape[1]}D -> {pca_components}D (variance: {variance:.3f})")
    else:
        embeddings_reduced = embeddings

    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_reduced)

    # MiniBatch K-Means
    start_time = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=min(10000, len(embeddings)),
        max_iter=100,
        n_init=3,
        init='k-means++',
        verbose=0
    )

    clusters = kmeans.fit_predict(embeddings_scaled)
    duration = time.time() - start_time

    logger.info(f"  Clustered into {n_clusters} groups in {duration:.1f}s")

    return clusters, duration


def create_dissolved_visualization(gdf: gpd.GeoDataFrame, clusters: np.ndarray,
                                   resolution: int, n_clusters: int, ax, crs: str = 'EPSG:28992',
                                   colormap: str = 'tab20') -> float:
    """Create dissolved cluster visualization for a single resolution."""

    start_time = time.time()

    # Add cluster labels
    viz_gdf = gdf[['geometry']].copy()
    viz_gdf['cluster'] = clusters

    # Reproject
    viz_gdf_proj = viz_gdf.to_crs(crs)

    # Dissolve by cluster - key optimization!
    logger.info(f"  Dissolving {len(viz_gdf_proj):,} hexagons...")
    viz_dissolved = viz_gdf_proj.dissolve(by='cluster', as_index=False)
    logger.info(f"  Reduced to {len(viz_dissolved):,} polygons")

    # Plot with proper cartographic settings - NO EDGES
    viz_dissolved.plot(
        column='cluster',
        ax=ax,
        cmap=colormap,
        edgecolor='none',  # No edge outlines
        linewidth=0,       # No edge lines
        alpha=0.85,
        legend=False,
        aspect='equal'  # Ensure proper aspect ratio
    )

    # Set proper bounds and aspect
    bounds = viz_gdf_proj.total_bounds
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect('equal', adjustable='box')

    # Title with scale names
    scale_names = {
        5: "Metro", 6: "City", 7: "District",
        8: "Neighborhood", 9: "Block", 10: "Building"
    }

    title = f"Res {resolution} ({scale_names.get(resolution, 'Unknown')})\n"
    title += f"{len(gdf):,} hexagons → {n_clusters} clusters"
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Add north arrow
    x, y, arrow_length = 0.95, 0.95, 0.08
    ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=3, headwidth=8),
                ha='center', va='center', fontsize=12, fontweight='bold',
                xycoords=ax.transAxes)

    # Clean axes
    ax.set_axis_off()

    duration = time.time() - start_time
    return duration


def visualize_hierarchical_fast(
    study_area: str = "netherlands",
    resolutions: List[int] = [5, 6, 7, 8, 9, 10],
    n_clusters_dict: Dict[int, int] = None,
    colormap: str = 'tab20',
    output_dir: str = 'results [old 2024]/visualizations/hierarchical_fast',
    year: str = "2022",
    use_pca: bool = True,
    pca_components: int = 16,
    crs: str = 'EPSG:28992'
):
    """
    Main visualization pipeline for hierarchical multi-resolution clustering.

    Args:
        study_area: Study area name (e.g., 'netherlands')
        resolutions: List of resolutions to visualize
        n_clusters_dict: Dict mapping resolution to number of clusters
        colormap: Matplotlib colormap
        output_dir: Output directory for visualizations
        year: Data year
        use_pca: Whether to use PCA reduction
        pca_components: Number of PCA components
        crs: Target CRS for visualization
    """

    logger.info("="*80)
    logger.info("FAST HIERARCHICAL MULTI-RESOLUTION CLUSTERING")
    logger.info("="*80)
    logger.info(f"Study Area: {study_area}")
    logger.info(f"Resolutions: {resolutions}")
    logger.info(f"Colormap: {colormap}")
    logger.info(f"PCA: {use_pca} ({pca_components} components)")

    start_time = time.time()

    # Default cluster counts (auto-scale by resolution)
    if n_clusters_dict is None:
        n_clusters_dict = {
            5: 8,    # Metro - few large regions
            6: 10,   # City
            7: 12,   # District
            8: 14,   # Neighborhood
            9: 16,   # Block
            10: 16   # Building
        }

    # Setup output directory
    output_path = Path(output_dir) / study_area
    output_path.mkdir(parents=True, exist_ok=True)

    # Create multi-resolution figure
    n_res = len(resolutions)
    ncols = 3
    nrows = int(np.ceil(n_res / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    axes = axes.flatten() if n_res > 1 else [axes]

    # Process each resolution
    total_cluster_time = 0
    total_vis_time = 0

    for idx, resolution in enumerate(resolutions):
        logger.info(f"\nProcessing Resolution {resolution}")
        logger.info("-" * 40)

        try:
            # Load embeddings
            gdf, embeddings = load_resolution_embeddings(study_area, resolution, year)

            # Cluster
            n_clusters = n_clusters_dict.get(resolution, 10)
            clusters, cluster_time = perform_fast_clustering(
                embeddings, n_clusters, use_pca, pca_components
            )
            total_cluster_time += cluster_time

            # Visualize
            vis_time = create_dissolved_visualization(
                gdf, clusters, resolution, n_clusters,
                axes[idx], crs, colormap
            )
            total_vis_time += vis_time

        except Exception as e:
            logger.error(f"Error processing res{resolution}: {e}")
            axes[idx].text(0.5, 0.5, f"Res {resolution}\nError: {str(e)}",
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_axis_off()

    # Hide unused subplots
    for idx in range(n_res, len(axes)):
        axes[idx].set_visible(False)

    # Main title
    fig.suptitle(
        f"{study_area.title()} - Hierarchical Multi-Resolution Clustering\n"
        f"AlphaEarth {year} | MiniBatch K-Means + Dissolve",
        fontsize=16, fontweight='bold', y=0.98
    )

    # Save
    output_file = output_path / f"{study_area}_hierarchical_{colormap}_fast.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # Summary
    total_time = time.time() - start_time

    logger.info("\n" + "="*80)
    logger.info("HIERARCHICAL VISUALIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total Time: {total_time:.1f}s")
    logger.info(f"  Clustering: {total_cluster_time:.1f}s")
    logger.info(f"  Visualization: {total_vis_time:.1f}s")
    logger.info(f"Output: {output_file}")
    logger.info(f"Resolutions: {len(resolutions)}")

    return output_file


def create_individual_resolution_maps(
    study_area: str = "netherlands",
    resolutions: List[int] = [5, 6, 7, 8, 9, 10],
    n_clusters_dict: Dict[int, int] = None,
    colormap: str = 'tab20',
    output_dir: str = 'results [old 2024]/visualizations/hierarchical_fast',
    year: str = "2022",
    use_pca: bool = True,
    pca_components: int = 16,
    crs: str = 'EPSG:28992'
):
    """Create individual high-resolution maps for each resolution."""

    logger.info("\n" + "="*80)
    logger.info("CREATING INDIVIDUAL RESOLUTION MAPS")
    logger.info("="*80)

    # Default cluster counts
    if n_clusters_dict is None:
        n_clusters_dict = {5: 8, 6: 10, 7: 12, 8: 14, 9: 16, 10: 16}

    output_path = Path(output_dir) / study_area
    output_path.mkdir(parents=True, exist_ok=True)

    for resolution in resolutions:
        logger.info(f"\nCreating individual map for res{resolution}")

        try:
            # Load embeddings
            gdf, embeddings = load_resolution_embeddings(study_area, resolution, year)

            # Cluster
            n_clusters = n_clusters_dict.get(resolution, 10)
            clusters, _ = perform_fast_clustering(embeddings, n_clusters, use_pca, pca_components)

            # Create figure with proper aspect
            fig, ax = plt.subplots(figsize=(12, 14))

            # Visualize
            create_dissolved_visualization(
                gdf, clusters, resolution, n_clusters,
                ax, crs, colormap
            )

            # Override title for individual maps
            scale_names = {
                5: "Metro (~8.5 km)", 6: "City (~3.2 km)", 7: "District (~1.2 km)",
                8: "Neighborhood (~460 m)", 9: "Block (~174 m)", 10: "Building (~66 m)"
            }
            title = f"{study_area.title()} - Res {resolution} ({scale_names.get(resolution, '')})\n"
            title += f"{len(gdf):,} hexagons | {n_clusters} clusters | MiniBatch K-Means + Dissolve"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            # Save
            output_file = output_path / f"{study_area}_res{resolution:02d}_{colormap}_fast.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"  Saved: {output_file.name}")

        except Exception as e:
            logger.error(f"Error creating map for res{resolution}: {e}")


def main():
    """Main execution pipeline."""

    parser = argparse.ArgumentParser(
        description='Fast Hierarchical Multi-Resolution Clustering Visualization'
    )
    parser.add_argument('--study-area', type=str, default='netherlands',
                       help='Study area name')
    parser.add_argument('--resolutions', type=str, default='5,6,7,8,9,10',
                       help='Comma-separated resolutions')
    parser.add_argument('--clusters', type=str, default='8,10,12,14,16,16',
                       help='Comma-separated cluster counts per resolution')
    parser.add_argument('--colormap', type=str, default='tab20',
                       help='Matplotlib colormap')
    parser.add_argument('--output-dir', type=str, default='results [old 2024]/visualizations/hierarchical_fast',
                       help='Output directory')
    parser.add_argument('--year', type=str, default='2022',
                       help='Data year')
    parser.add_argument('--no-pca', action='store_true',
                       help='Skip PCA reduction')
    parser.add_argument('--pca-components', type=int, default=16,
                       help='Number of PCA components')
    parser.add_argument('--crs', type=str, default='EPSG:28992',
                       help='Target CRS for visualization')
    parser.add_argument('--individual', action='store_true',
                       help='Also create individual high-res maps per resolution')

    args = parser.parse_args()

    # Parse arguments
    resolutions = [int(r) for r in args.resolutions.split(',')]
    cluster_counts = [int(c) for c in args.clusters.split(',')]

    if len(cluster_counts) != len(resolutions):
        logger.warning(f"Cluster counts ({len(cluster_counts)}) != resolutions ({len(resolutions)})")
        cluster_counts = cluster_counts + [cluster_counts[-1]] * (len(resolutions) - len(cluster_counts))

    n_clusters_dict = dict(zip(resolutions, cluster_counts))

    # Create combined visualization
    output_file = visualize_hierarchical_fast(
        study_area=args.study_area,
        resolutions=resolutions,
        n_clusters_dict=n_clusters_dict,
        colormap=args.colormap,
        output_dir=args.output_dir,
        year=args.year,
        use_pca=not args.no_pca,
        pca_components=args.pca_components,
        crs=args.crs
    )

    # Create individual maps if requested
    if args.individual:
        create_individual_resolution_maps(
            study_area=args.study_area,
            resolutions=resolutions,
            n_clusters_dict=n_clusters_dict,
            colormap=args.colormap,
            output_dir=args.output_dir,
            year=args.year,
            use_pca=not args.no_pca,
            pca_components=args.pca_components,
            crs=args.crs
        )

    logger.info("\n" + "="*80)
    logger.info("ALL VISUALIZATIONS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()