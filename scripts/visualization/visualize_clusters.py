#!/usr/bin/env python3
"""
Unified Cluster Visualization CLI

Replaces the three separate visualization scripts:
- visualize_res10_clusters_fast.py
- visualize_res8_clusters_fast.py
- visualize_hierarchical_embeddings_fast.py

Usage:
    # Single resolution
    python scripts/visualization/visualize_clusters.py --study-area netherlands --resolution 10
    python scripts/visualization/visualize_clusters.py --study-area cascadia --resolution 8

    # Multi-resolution hierarchical
    python scripts/visualization/visualize_clusters.py --study-area netherlands --resolution 5,6,7,8,9,10 --hierarchical

    # With options
    python scripts/visualization/visualize_clusters.py --study-area netherlands --resolution 10 --clusters 8,12,16 --no-dissolve --sample 0.1
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# Professional plotting style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})


def main():
    parser = argparse.ArgumentParser(
        description='Unified Cluster Visualization (dissolve + MiniBatchKMeans + datashader)'
    )
    parser.add_argument(
        '--study-area', type=str, required=True,
        help='Study area to visualize (e.g., netherlands, cascadia, pearl_river_delta)',
    )
    parser.add_argument(
        '--resolution', type=str, required=True,
        help='H3 resolution(s). Single int or comma-separated (e.g., 10 or 5,6,7,8,9,10)',
    )
    parser.add_argument(
        '--clusters', type=str, default='8,12,16',
        help='Comma-separated cluster counts (default: 8,12,16). '
             'For hierarchical mode, one per resolution (e.g., 8,10,12,14,16,16)',
    )
    parser.add_argument(
        '--hierarchical', action='store_true',
        help='Create multi-resolution hierarchical subplot grid',
    )
    parser.add_argument(
        '--colormap', type=str, default='tab20',
        help='Matplotlib colormap (default: tab20)',
    )
    parser.add_argument(
        '--pca-components', type=int, default=16,
        help='PCA components for dimensionality reduction (default: 16)',
    )
    parser.add_argument(
        '--skip-pca', action='store_true',
        help='Skip PCA and use full embeddings',
    )
    parser.add_argument(
        '--no-dissolve', action='store_true',
        help='Skip dissolve optimization',
    )
    parser.add_argument(
        '--no-datashader', action='store_true',
        help='Skip datashader optimization',
    )
    parser.add_argument(
        '--sample', type=float, default=1.0,
        help='Sample fraction of data for testing (0-1)',
    )
    parser.add_argument(
        '--year', type=str, default='2022',
        help='Data year (default: 2022)',
    )
    parser.add_argument(
        '--crs', type=str, default=None,
        help='Target CRS override (default: from study area config)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override output directory',
    )

    args = parser.parse_args()

    # Import here to defer heavy imports
    from stage3_analysis.visualization.cluster_viz import (
        STUDY_AREA_CONFIG,
        COLORMAP_COMBINATIONS,
        find_study_area_data,
        load_and_prepare_embeddings,
        apply_pca_reduction,
        perform_minibatch_clustering,
        create_cluster_visualization,
        create_hierarchical_subplot,
    )

    # Parse resolutions
    resolutions = [int(r.strip()) for r in args.resolution.split(',')]

    # Get study area config
    if args.study_area not in STUDY_AREA_CONFIG:
        available = ', '.join(STUDY_AREA_CONFIG.keys())
        print(f"Error: Study area '{args.study_area}' not supported. Available: {available}")
        sys.exit(1)

    config = STUDY_AREA_CONFIG[args.study_area]
    target_crs = args.crs or config['crs']

    start_time = time.time()

    if args.hierarchical or len(resolutions) > 1:
        # ── Hierarchical mode ──
        cluster_counts = [int(c.strip()) for c in args.clusters.split(',')]

        if len(cluster_counts) == 1:
            # Single count — apply to all resolutions
            n_clusters_dict = {r: cluster_counts[0] for r in resolutions}
        elif len(cluster_counts) == len(resolutions):
            n_clusters_dict = dict(zip(resolutions, cluster_counts))
        else:
            # Pad with last value
            cluster_counts = cluster_counts + [cluster_counts[-1]] * (len(resolutions) - len(cluster_counts))
            n_clusters_dict = dict(zip(resolutions, cluster_counts[:len(resolutions)]))

        output_dir = Path(args.output_dir) if args.output_dir else Path(
            f'results/visualizations/hierarchical_fast/{args.study_area}'
        )

        print("=" * 80)
        print("HIERARCHICAL MULTI-RESOLUTION CLUSTERING")
        print("=" * 80)
        print(f"Study Area: {args.study_area}")
        print(f"Resolutions: {resolutions}")
        print(f"Clusters: {n_clusters_dict}")

        output_file = create_hierarchical_subplot(
            study_area=args.study_area,
            resolutions=resolutions,
            n_clusters_dict=n_clusters_dict,
            colormap=args.colormap,
            output_path=output_dir,
            year=args.year,
            use_pca=not args.skip_pca,
            pca_components=args.pca_components,
            crs=target_crs,
        )
        print(f"\nOutput: {output_file}")

    else:
        # ── Single resolution mode ──
        resolution = resolutions[0]
        cluster_counts = [int(c.strip()) for c in args.clusters.split(',')]

        print("=" * 80)
        print(f"CLUSTER VISUALIZATION - Resolution {resolution}")
        print("=" * 80)
        print(f"Study Area: {args.study_area}")
        print(f"Cluster Counts: {cluster_counts}")
        print(f"Optimizations: Dissolve={not args.no_dissolve}, Datashader={not args.no_datashader}")

        # Setup output directory
        project_root = Path(__file__).parent.parent.parent
        output_dir = Path(args.output_dir) if args.output_dir else (
            project_root / config['output_dir']
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        data_path = find_study_area_data(args.study_area, resolution)
        gdf, embeddings = load_and_prepare_embeddings(data_path)

        # Sample if requested
        if args.sample < 1.0:
            sample_size = int(len(gdf) * args.sample)
            print(f"Sampling {sample_size:,} hexagons ({args.sample * 100:.0f}%)...")
            sample_idx = np.random.choice(len(gdf), sample_size, replace=False)
            gdf = gdf.iloc[sample_idx].reset_index(drop=True)
            embeddings = embeddings[sample_idx]

        # PCA
        if args.skip_pca:
            embeddings_for_clustering = embeddings
        else:
            embeddings_for_clustering, _ = apply_pca_reduction(embeddings, args.pca_components)

        # Cluster
        cluster_results = perform_minibatch_clustering(embeddings_for_clustering, cluster_counts)

        # Visualize
        print(f"\nCreating visualizations...")
        for n_clusters, colormap in COLORMAP_COMBINATIONS:
            if n_clusters in cluster_results:
                suffix = "dissolved" if not args.no_dissolve else "datashader" if not args.no_datashader else "standard"
                filename = f'{args.study_area}_res{resolution}_{n_clusters:02d}clusters_{colormap}_{suffix}.png'
                output_path = output_dir / filename

                create_cluster_visualization(
                    gdf, cluster_results[n_clusters], n_clusters,
                    colormap, output_path,
                    target_crs=target_crs,
                    title_prefix=config['title_prefix'],
                    resolution=resolution,
                    use_dissolve=not args.no_dissolve,
                    use_datashader=not args.no_datashader,
                )

        print(f"\nOutput Directory: {output_dir}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"COMPLETE - Total Time: {total_time:.1f}s")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
