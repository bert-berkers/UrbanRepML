"""
Rasterized KMeans Cluster Maps for Urban Embeddings.

Produces cluster map PNGs for any embedding parquet file using rasterized
centroid rendering (not dissolve). Accepts arbitrary parquet paths for stage1
unimodal or stage2 multimodal embeddings.

Output is organized in date-stamped subdirectories: {output_dir}/YYYY-MM-DD/

Lifetime: durable
Stage: 3 (post-training analysis)

Usage:
    python scripts/stage3/plot_cluster_maps.py \
        --study-area netherlands \
        --embedding-path data/study_areas/netherlands/stage2_multimodal/concat/embeddings/netherlands_res9_2022_raw.parquet \
        --output-dir data/study_areas/netherlands/stage2_multimodal/concat/plots/res9/clusters \
        --resolution 9 \
        --label "3-Modality Concat (781D)"

    # With custom date and k values
    python scripts/stage3/plot_cluster_maps.py \
        --embedding-path path/to/embeddings.parquet \
        --output-dir path/to/output \
        --resolution 10 \
        --k-values 8 16 \
        --date 2026-03-09
"""

import argparse
import logging
import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import (
    RASTER_H,
    RASTER_W,
    load_boundary,
    plot_spatial_map,
    rasterize_categorical,
)
from stage3_analysis.visualization.clustering_utils import (
    apply_pca_reduction,
    perform_minibatch_clustering,
)

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DPI = 300


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------


def load_embeddings(embedding_path: Path) -> pd.DataFrame:
    """Load embedding parquet, return DataFrame indexed by region_id.

    Auto-detects embedding columns by prefix pattern (A, P, R, S, G, emb_)
    or falls back to all numeric columns.
    """
    try:
        df = gpd.read_parquet(embedding_path)
        # Drop geometry for embedding extraction
        df = pd.DataFrame(df.drop(columns=["geometry"], errors="ignore"))
    except (ValueError, TypeError):
        df = pd.read_parquet(embedding_path)

    # Ensure region_id is the index
    if df.index.name != "region_id" and "region_id" in df.columns:
        df = df.set_index("region_id")

    # Detect embedding columns
    emb_cols = [
        col for col in df.columns
        if (col.startswith(("A", "P", "R", "S", "G")) and len(col) >= 2 and col[1:].isdigit())
        or col.startswith("emb_")
        or col.startswith("gtfs2vec_")
    ]
    if not emb_cols:
        exclude = {"pixel_count", "tile_count", "geometry", "region_id", "h3_resolution"}
        emb_cols = [
            col for col in df.columns
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not emb_cols:
        raise ValueError(f"No embedding columns found in {embedding_path}")

    return df[emb_cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate rasterized KMeans cluster maps for urban embeddings."
    )
    parser.add_argument(
        "--study-area", type=str, default="netherlands",
        help="Study area name (default: netherlands)",
    )
    parser.add_argument(
        "--embedding-path", type=str, required=True,
        help="Path to the embedding parquet file",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Base directory for output; a YYYY-MM-DD subdir will be created",
    )
    parser.add_argument(
        "--resolution", type=int, default=9,
        help="H3 resolution of the embeddings (default: 9)",
    )
    parser.add_argument(
        "--label", type=str, default="",
        help="Descriptive label for map titles (e.g. '3-Modality UNet (128D)')",
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[8, 12, 16],
        help="List of cluster counts (default: 8 12 16)",
    )
    parser.add_argument(
        "--pca-components", type=int, default=16,
        help="Number of PCA components before clustering (default: 16)",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date for output subdirectory (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--no-boundary", action="store_true",
        help="Skip the study area boundary underlay (useful for sparse modalities like GTFS)",
    )
    parser.add_argument(
        "--filter-empty", action="store_true",
        help="Remove all-zero and low-variance hexagons before clustering (for sparse modalities)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    embedding_path = Path(args.embedding_path)
    if not embedding_path.exists():
        print(f"ERROR: Embedding file not found: {embedding_path}")
        sys.exit(1)

    # Date-stamped output directory
    date_str = args.date or date.today().isoformat()
    output_dir = Path(args.output_dir) / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = args.label or embedding_path.stem
    resolution = args.resolution
    stamp = max(1, 11 - resolution)  # res9=2, res10=1, res8=3

    print(f"\n{'=' * 60}")
    print(f"Rasterized Cluster Maps: {title_prefix}")
    print(f"{'=' * 60}")

    # 1. Load embeddings
    emb_df = load_embeddings(embedding_path)
    print(f"Loaded {len(emb_df):,} hexagons, {emb_df.shape[1]}D embeddings")

    # 2. Get centroids via SpatialDB
    paths = StudyAreaPaths(args.study_area)
    db = SpatialDB.for_study_area(args.study_area)
    cx, cy = db.centroids(emb_df.index, resolution=resolution, crs=28992)
    print(f"Centroids loaded: {len(cx):,} points")

    # 3. Load boundary for underlay (skip if --no-boundary)
    boundary_gdf = None if args.no_boundary else load_boundary(paths)

    # 4. Compute extent
    if boundary_gdf is not None:
        ext = boundary_gdf.total_bounds
    else:
        ext = db.extent(emb_df.index, resolution=resolution, crs=28992)
    minx, miny, maxx, maxy = ext
    pad = (maxx - minx) * 0.03
    extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    # 4b. Filter empty/background hexagons if requested
    if args.filter_empty:
        vals = emb_df.values.astype(np.float64)
        n_before = len(emb_df)

        # Strategy 1: detect dominant repeated vector (>50% identical rows).
        # Catches encoders that assign a learned default embedding to
        # hexagons without data (e.g. GTFS2Vec non-transit hexagons).
        sample_idx = np.linspace(0, n_before - 1, min(10000, n_before), dtype=int)
        sample = np.round(vals[sample_idx], 6)
        unique_vecs, _, counts = np.unique(sample, axis=0, return_inverse=True, return_counts=True)
        dominant_idx = np.argmax(counts)
        dominant_frac = counts[dominant_idx] / len(sample)

        if dominant_frac > 0.5:
            default_vec = unique_vecs[dominant_idx]
            diffs = np.abs(vals - default_vec)
            keep = diffs.max(axis=1) >= 1e-5
        else:
            # Strategy 2: remove all-zero rows only
            keep = ~(vals == 0.0).all(axis=1)

        emb_df = emb_df[keep]
        cx, cy = cx[keep], cy[keep]
        print(f"Filtered {n_before:,} -> {len(emb_df):,} hexagons (removed {n_before - len(emb_df):,} background)")

    # 5. PCA reduction
    embeddings = emb_df.values.astype(np.float32)
    if embeddings.shape[1] > args.pca_components:
        reduced, pca = apply_pca_reduction(embeddings, n_components=args.pca_components)
    else:
        print(f"Skipping PCA: {embeddings.shape[1]}D <= {args.pca_components} components")
        reduced = embeddings

    # 6. Clustering
    cluster_results = perform_minibatch_clustering(
        reduced, args.k_values, standardize=True,
    )

    # 7. Render maps
    cmap = "tab20"
    print(f"\nRendering {len(cluster_results)} cluster maps (stamp={stamp})...")

    for k, labels in sorted(cluster_results.items()):
        fig, ax = plt.subplots(figsize=(12, 14), dpi=DPI)
        fig.set_facecolor("white")

        image = rasterize_categorical(
            cx, cy, labels, extent, n_clusters=k, cmap=cmap, stamp=stamp,
        )
        plot_spatial_map(ax, image, extent, boundary_gdf)

        ax.set_title(
            f"{title_prefix} -- MiniBatchKMeans k={k}\n"
            f"H3 res{resolution} | {len(emb_df):,} hexagons",
            fontsize=12, fontweight="bold", pad=10,
        )

        # Cluster legend
        cmap_obj = plt.get_cmap(cmap)
        legend_elements = [
            Patch(facecolor=cmap_obj(i / max(k - 1, 1)), label=f"Cluster {i}")
            for i in range(k)
        ]
        ncol = 4 if k > 8 else 3
        ax.legend(
            handles=legend_elements, loc="lower left", ncol=ncol,
            fontsize=7, framealpha=0.9, title="Clusters",
        )

        plt.tight_layout()
        out_path = output_dir / f"clusters_k{k}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone. {len(cluster_results)} maps saved to {output_dir}")


if __name__ == "__main__":
    main()
