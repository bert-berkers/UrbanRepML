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
RASTER_W = 2000
RASTER_H = 2400


# ---------------------------------------------------------------------------
# Rasterization helpers (from plot_embeddings.py)
# ---------------------------------------------------------------------------


def _stamp_pixels(image, py, px, rgb, stamp, height, width):
    """Write RGB values to image with a square stamp of given radius."""
    if stamp <= 1:
        image[py, px, :3] = rgb
        image[py, px, 3] = 1.0
    else:
        for dy in range(-stamp + 1, stamp):
            for dx in range(-stamp + 1, stamp):
                sy = np.clip(py + dy, 0, height - 1)
                sx = np.clip(px + dx, 0, width - 1)
                image[sy, sx, :3] = rgb
                image[sy, sx, 3] = 1.0


def rasterize_categorical(
    cx: np.ndarray,
    cy: np.ndarray,
    labels: np.ndarray,
    extent: tuple,
    n_clusters: int,
    width: int = RASTER_W,
    height: int = RASTER_H,
    cmap: str = "tab20",
    stamp: int = 1,
) -> np.ndarray:
    """Rasterize integer cluster labels to an RGBA image.

    Args:
        cx, cy: centroid coordinates in EPSG:28992.
        labels: integer cluster assignment array.
        extent: (minx, miny, maxx, maxy) in EPSG:28992.
        n_clusters: total number of clusters (for colormap scaling).
        width, height: output image dimensions.
        cmap: matplotlib colormap name.
        stamp: pixel radius per point (1=single pixel, 2+=fills gaps at coarser res).

    Returns:
        (height, width, 4) RGBA float32 array with transparent background.
    """
    minx, miny, maxx, maxy = extent
    mask = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    cx_m, cy_m, lab_m = cx[mask], cy[mask], labels[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    colormap_obj = plt.get_cmap(cmap)
    norm_vals = lab_m.astype(float) / max(n_clusters - 1, 1)
    rgb = colormap_obj(norm_vals)[:, :3].astype(np.float32)

    image = np.zeros((height, width, 4), dtype=np.float32)
    _stamp_pixels(image, py, px, rgb, stamp, height, width)
    return image


# ---------------------------------------------------------------------------
# Map rendering helpers (from plot_embeddings.py)
# ---------------------------------------------------------------------------


def _clean_map_axes(ax):
    """Remove ticks and labels for a clean map look."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def _add_rd_grid(ax, extent):
    """Add RD New (EPSG:28992) coordinate grid lines and labels."""
    minx, miny, maxx, maxy = extent
    step = 50_000  # 50 km grid
    x_grid = np.arange(
        np.floor(minx / step) * step,
        np.ceil(maxx / step) * step + step,
        step,
    )
    y_grid = np.arange(
        np.floor(miny / step) * step,
        np.ceil(maxy / step) * step + step,
        step,
    )
    for x in x_grid:
        if minx <= x <= maxx:
            ax.axvline(x, color='grey', alpha=0.3, linewidth=0.5, zorder=10)
    for y in y_grid:
        if miny <= y <= maxy:
            ax.axhline(y, color='grey', alpha=0.3, linewidth=0.5, zorder=10)
    for x in x_grid:
        if minx <= x <= maxx:
            ax.text(
                x, miny - (maxy - miny) * 0.015, f'{x:.0f}',
                ha='center', va='top', fontsize=7, color='#555555',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
            )
    for y in y_grid:
        if miny <= y <= maxy:
            ax.text(
                minx - (maxx - minx) * 0.01, y, f'{y:.0f}',
                ha='right', va='center', fontsize=7, color='#555555',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
            )


def load_boundary(paths: StudyAreaPaths) -> gpd.GeoDataFrame | None:
    """Load study area boundary, filter to European NL, reproject to 28992."""
    from shapely import get_geometry, get_num_geometries

    boundary_path = paths.area_gdf_file()
    if not boundary_path.exists():
        logger.warning("Boundary file not found: %s", boundary_path)
        return None

    boundary_gdf = gpd.read_file(boundary_path)
    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    boundary_gdf = boundary_gdf.to_crs(epsg=28992)

    geom = boundary_gdf.geometry.iloc[0]
    n_parts = get_num_geometries(geom)
    if n_parts > 1:
        euro_geom = max(
            (get_geometry(geom, i) for i in range(n_parts)),
            key=lambda g: g.area,
        )
        boundary_gdf = gpd.GeoDataFrame(
            geometry=[euro_geom], crs=boundary_gdf.crs
        )

    return boundary_gdf


def plot_spatial_map(ax, image, extent, boundary_gdf, title=""):
    """Render a rasterized image on an axes with boundary underlay and RD grid."""
    if boundary_gdf is not None:
        boundary_gdf.plot(
            ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5,
            zorder=1,
        )
    minx, miny, maxx, maxy = extent
    ax.imshow(
        image,
        extent=[minx, maxx, miny, maxy],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        zorder=2,
    )
    _add_rd_grid(ax, extent)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    _clean_map_axes(ax)
    if title:
        ax.set_title(title, fontsize=11)


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
        vals = emb_df.values
        n_before = len(emb_df)

        # Strategy: detect dominant background mode via row-wise std clustering
        # Many embedders produce identical vectors for "no data" hexagons
        row_std = vals.std(axis=1)
        median_std = np.median(row_std)

        # Hexagons whose row_std matches the median exactly (within tolerance)
        # are likely background/default vectors
        background = np.abs(row_std - median_std) < 0.001

        # Only filter if the "background" is >50% of data (clear bimodal split)
        if background.sum() > 0.5 * n_before:
            keep = ~background
        else:
            # Fallback: remove all-zero rows only
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
