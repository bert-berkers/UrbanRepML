#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spatial maps for causal emergence analysis.

Produces two maps:
1. Prediction improvement map (vrz): where multi-scale concat embeddings
   outperform res9-only embeddings (blue) vs. underperform (red).
2. Embedding similarity divergence: cosine similarity between each res9
   hexagon's embedding and its res8 parent's embedding. Low similarity
   signals scale disagreement -- hexagons where micro and macro views
   of urban character diverge.

Lifetime: durable
Stage: 3

Reuses rasterization infrastructure from stage3_analysis.linear_probe_viz
(centroid rasterization via SpatialDB) without instantiating the full
visualizer class.
"""

import logging
from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, Normalize
from shapely import get_geometry, get_num_geometries
from sklearn.metrics.pairwise import cosine_similarity

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB

logger = logging.getLogger(__name__)

STUDY_AREA = "netherlands"
H3_RESOLUTION = 9


# ------------------------------------------------------------------
# Rasterization helpers (extracted from LinearProbeVisualizer)
# ------------------------------------------------------------------

def rasterize_centroids(
    hex_ids: pd.Index,
    rgb_array: np.ndarray,
    extent: tuple,
    study_area: str = STUDY_AREA,
    resolution: int = H3_RESOLUTION,
    width: int = 2000,
    height: int = 2400,
) -> np.ndarray:
    """
    Rasterize H3 centroids to an RGBA image via SpatialDB.

    Reimplements LinearProbeVisualizer._rasterize_centroids as a standalone
    function to avoid instantiating the full visualizer class.

    Args:
        hex_ids: Index of H3 hex ID strings.
        rgb_array: (N, 3) float array with R, G, B in [0, 1].
        extent: (minx, miny, maxx, maxy) in EPSG:28992.
        study_area: Study area name for SpatialDB lookup.
        resolution: H3 resolution level.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        (height, width, 4) RGBA float32 array with white background.
    """
    db = SpatialDB.for_study_area(study_area)
    all_cx, all_cy = db.centroids(hex_ids, resolution=resolution, crs=28992)

    minx, miny, maxx, maxy = extent
    mask = (
        (all_cx >= minx) & (all_cx <= maxx)
        & (all_cy >= miny) & (all_cy <= maxy)
    )

    cx = all_cx[mask]
    cy = all_cy[mask]
    rgb_masked = rgb_array[mask]

    px = ((cx - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy - miny) / (maxy - miny) * (height - 1)).astype(int)

    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    image = np.ones((height, width, 4), dtype=np.float32)
    image[py, px, :3] = rgb_masked
    image[py, px, 3] = 1.0

    return image


def load_boundary(paths: StudyAreaPaths) -> gpd.GeoDataFrame | None:
    """Load study area boundary in EPSG:28992, filtered to European NL."""
    boundary_path = paths.area_gdf_file()
    if not boundary_path.exists():
        return None

    boundary_gdf = gpd.read_file(boundary_path)
    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    boundary_gdf = boundary_gdf.to_crs(epsg=28992)

    # Filter to European Netherlands (exclude Caribbean)
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


def compute_extent(boundary_gdf: gpd.GeoDataFrame | None, hex_ids: pd.Index) -> tuple:
    """Compute render extent with 3% padding."""
    if boundary_gdf is not None:
        extent = boundary_gdf.total_bounds
    else:
        db = SpatialDB.for_study_area(STUDY_AREA)
        extent = db.extent(hex_ids, resolution=H3_RESOLUTION, crs=28992)

    minx, miny, maxx, maxy = extent
    pad = (maxx - minx) * 0.03
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


def add_scale_bar(ax, length_km: int = 50):
    """Add a scale bar to a map axis in EPSG:28992 coordinate space."""
    bar_length = length_km * 1000
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    y0 = ylim[0] + (ylim[1] - ylim[0]) * 0.04
    tick_height = (ylim[1] - ylim[0]) * 0.012

    ax.plot([x0, x0 + bar_length], [y0, y0], color="black", linewidth=2,
            solid_capstyle="butt", transform=ax.transData)
    ax.plot([x0, x0], [y0 - tick_height, y0 + tick_height],
            color="black", linewidth=1.5, transform=ax.transData)
    ax.plot([x0 + bar_length, x0 + bar_length],
            [y0 - tick_height, y0 + tick_height],
            color="black", linewidth=1.5, transform=ax.transData)
    ax.text(x0 + bar_length / 2, y0 + tick_height * 1.8,
            f"{length_km} km", ha="center", va="bottom",
            fontsize=9, fontweight="bold", transform=ax.transData)


def add_north_arrow(ax):
    """Add a north arrow to upper-right corner."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[1] - (xlim[1] - xlim[0]) * 0.06
    y_pos = ylim[1] - (ylim[1] - ylim[0]) * 0.06
    arrow_length = (ylim[1] - ylim[0]) * 0.06

    ax.annotate(
        "", xy=(x_pos, y_pos), xytext=(x_pos, y_pos - arrow_length),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
        transform=ax.transData,
    )
    ax.text(x_pos, y_pos + arrow_length * 0.2, "N",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            transform=ax.transData)


def render_map(
    hex_ids: pd.Index,
    rgb_array: np.ndarray,
    boundary_gdf: gpd.GeoDataFrame | None,
    render_extent: tuple,
    cmap,
    norm,
    colorbar_label: str,
    title_lines: list[str],
    output_path: Path,
    dpi: int = 300,
):
    """Render a spatial map with rasterized centroids and save as PNG + PDF."""
    plt.style.use("default")

    raster_image = rasterize_centroids(
        hex_ids=hex_ids,
        rgb_array=rgb_array,
        extent=render_extent,
    )

    fig, ax = plt.subplots(figsize=(12, 14))
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if boundary_gdf is not None:
        boundary_gdf.plot(
            ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5,
        )

    ax.imshow(
        raster_image,
        extent=[render_extent[0], render_extent[2],
                render_extent[1], render_extent[3]],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        zorder=2,
    )

    ax.set_xlim(render_extent[0], render_extent[2])
    ax.set_ylim(render_extent[1], render_extent[3])
    ax.grid(True, linewidth=0.5, alpha=0.5, color="gray")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    add_scale_bar(ax, length_km=50)
    add_north_arrow(ax)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, label=colorbar_label)

    ax.set_title("\n".join(title_lines), fontsize=11, pad=10)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    # Also save PDF
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Saved map: {output_path}")
    logger.info(f"Saved map: {pdf_path}")


# ------------------------------------------------------------------
# Map 1: Prediction Improvement (vrz)
# ------------------------------------------------------------------

def map_prediction_improvement(paths: StudyAreaPaths, output_dir: Path):
    """
    Map showing where multi-scale concat outperforms res9-only for vrz.

    Blue = concat is better (positive improvement).
    Red = res9-only is better (negative improvement).
    """
    target_col = "vrz"

    # Load OOF predictions
    res9_only_dir = paths.stage3("dnn_probe") / "2026-03-07_multiscale_res9_only"
    concat_dir = paths.stage3("dnn_probe") / "2026-03-07_multiscale_multiscale_concat"

    res9_preds = pd.read_parquet(res9_only_dir / f"predictions_{target_col}.parquet")
    concat_preds = pd.read_parquet(concat_dir / f"predictions_{target_col}.parquet")

    logger.info(f"Loaded res9-only predictions: {len(res9_preds):,} hexagons")
    logger.info(f"Loaded concat predictions: {len(concat_preds):,} hexagons")

    # Compute improvement: |residual_res9_only| - |residual_concat|
    # Positive = concat is better (smaller absolute error)
    merged = res9_preds[["residual"]].rename(columns={"residual": "res9_residual"}).join(
        concat_preds[["residual"]].rename(columns={"residual": "concat_residual"}),
        how="inner",
    )
    merged["improvement"] = merged["res9_residual"].abs() - merged["concat_residual"].abs()

    n_hexagons = len(merged)
    logger.info(f"Merged predictions: {n_hexagons:,} hexagons")

    hex_ids = pd.Index(merged.index, name="region_id")

    # Boundary and extent
    boundary_gdf = load_boundary(paths)
    render_extent = compute_extent(boundary_gdf, hex_ids)

    # Map improvement to RGB (RdBu diverging, symmetric around 0)
    improvement = merged["improvement"]
    vmax = float(improvement.abs().quantile(0.98))
    if vmax == 0:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.colormaps["RdBu"]
    rgba = cmap(norm(improvement.values))
    rgb = rgba[:, :3].astype(np.float32)

    # Statistics
    n_concat_better = int((improvement > 0).sum())
    n_res9_better = int((improvement < 0).sum())
    mean_improvement = float(improvement.mean())

    title_lines = [
        "Prediction Improvement: Multi-Scale vs Res9-Only (Amenities)",
        f"Blue = multi-scale better ({n_concat_better:,}) | "
        f"Red = res9-only better ({n_res9_better:,}) | "
        f"mean={mean_improvement:+.4f} | n={n_hexagons:,}",
    ]

    render_map(
        hex_ids=hex_ids,
        rgb_array=rgb,
        boundary_gdf=boundary_gdf,
        render_extent=render_extent,
        cmap=cmap,
        norm=norm,
        colorbar_label="Prediction improvement\n(+ = multi-scale better)",
        title_lines=title_lines,
        output_path=output_dir / "spatial_improvement_vrz.png",
    )


# ------------------------------------------------------------------
# Map 2: Embedding Similarity Divergence (res9 vs res8 parent)
# ------------------------------------------------------------------

def map_embedding_divergence(paths: StudyAreaPaths, output_dir: Path):
    """
    Map cosine similarity between each res9 hexagon's embedding and
    its res8 parent's embedding.

    Low similarity = scale disagreement = hexagons where micro and
    macro views diverge.
    """
    # Load embeddings
    res9_path = paths.fused_embedding_file("unet", 9)
    res8_path = paths.fused_embedding_file("unet", 8)

    res9_emb = pd.read_parquet(res9_path)
    res8_emb = pd.read_parquet(res8_path)

    logger.info(f"Loaded res9 embeddings: {res9_emb.shape}")
    logger.info(f"Loaded res8 embeddings: {res8_emb.shape}")

    # Map each res9 hex to its res8 parent (h3-py OK for hierarchy traversal)
    res9_ids = res9_emb.index.to_numpy()
    parent_ids = np.array([h3.cell_to_parent(h, 8) for h in res9_ids])

    # Build aligned arrays for cosine similarity
    # Only keep res9 hexagons whose parent exists in res8 embeddings
    parent_series = pd.Series(parent_ids, index=res9_emb.index, name="parent_id")
    valid_mask = parent_series.isin(res8_emb.index)
    logger.info(f"Res9 hexagons with valid res8 parent: {valid_mask.sum():,} / {len(valid_mask):,}")

    valid_res9 = res9_emb.loc[valid_mask]
    valid_parents = parent_series.loc[valid_mask]

    # Get parent embeddings aligned to child order
    parent_emb_aligned = res8_emb.loc[valid_parents.values].values

    # Compute row-wise cosine similarity
    # Normalize both sets of vectors
    res9_norms = np.linalg.norm(valid_res9.values, axis=1, keepdims=True)
    parent_norms = np.linalg.norm(parent_emb_aligned, axis=1, keepdims=True)

    # Avoid division by zero
    res9_norms = np.where(res9_norms == 0, 1, res9_norms)
    parent_norms = np.where(parent_norms == 0, 1, parent_norms)

    cos_sim = np.sum(
        (valid_res9.values / res9_norms) * (parent_emb_aligned / parent_norms),
        axis=1,
    )

    similarity_df = pd.DataFrame(
        {"cosine_similarity": cos_sim},
        index=valid_res9.index,
    )

    n_hexagons = len(similarity_df)
    hex_ids = pd.Index(similarity_df.index, name="region_id")

    logger.info(
        f"Cosine similarity stats: "
        f"mean={cos_sim.mean():.4f}, "
        f"std={cos_sim.std():.4f}, "
        f"min={cos_sim.min():.4f}, "
        f"max={cos_sim.max():.4f}"
    )

    # Cosine distance = 1 - cos_sim. Use rank-based (percentile) normalization
    # to amplify the narrow range and reveal spatial structure.
    cos_dist = 1.0 - cos_sim
    ranks = pd.Series(cos_dist).rank(pct=True).values  # 0..1 percentile

    logger.info(
        f"Cosine distance stats: "
        f"mean={cos_dist.mean():.6f}, "
        f"std={cos_dist.std():.6f}, "
        f"min={cos_dist.min():.6f}, "
        f"max={cos_dist.max():.6f}"
    )

    # Boundary and extent
    boundary_gdf = load_boundary(paths)
    render_extent = compute_extent(boundary_gdf, hex_ids)

    # Map to RGB using inferno (high divergence = bright/yellow, low = dark/purple)
    # Ranks are already in [0, 1] so use identity normalization
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.colormaps["inferno"]
    rgba = cmap(norm(ranks))
    rgb = rgba[:, :3].astype(np.float32)

    # Statistics
    low_thresh = float(np.quantile(cos_sim, 0.1))
    n_low = int((cos_sim < low_thresh).sum())

    title_lines = [
        "Embedding Divergence: Res9 vs Res8 Parent",
        f"Bright = high divergence | Dark = coherent | "
        f"cos_sim mean={cos_sim.mean():.4f} | n={n_hexagons:,}",
    ]

    # For the colorbar, show actual cosine distance quantiles
    # Create a custom norm that maps to the original distance range
    dist_norm = Normalize(
        vmin=float(np.quantile(cos_dist, 0.01)),
        vmax=float(np.quantile(cos_dist, 0.99)),
    )

    render_map(
        hex_ids=hex_ids,
        rgb_array=rgb,
        boundary_gdf=boundary_gdf,
        render_extent=render_extent,
        cmap=cmap,
        norm=dist_norm,
        colorbar_label="Cosine distance (rank-normalized)\n1 - cos_sim, percentile-scaled",
        title_lines=title_lines,
        output_path=output_dir / "embedding_divergence_res8_res9.png",
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    paths = StudyAreaPaths(STUDY_AREA)
    output_dir = paths.project_root / "reports" / "figures" / "causal-emergence"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Map 1: Prediction Improvement (vrz) ===")
    map_prediction_improvement(paths, output_dir)

    logger.info("=== Map 2: Embedding Similarity Divergence ===")
    map_embedding_divergence(paths, output_dir)

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
