"""Shared rasterization and spatial visualization helpers.

Provides centroid-based rasterization of H3-indexed data onto matplotlib
axes. Originally developed in ``scripts/plot_embeddings.py``; extracted
here so every stage-3 visualization script can import a single canonical
copy instead of copy-pasting.

All coordinate arguments are expected in EPSG:28992 (RD New) unless noted
otherwise.

Typical usage::

    from utils.visualization import (
        load_boundary,
        rasterize_continuous,
        rasterize_categorical,
        plot_spatial_map,
    )

    boundary = load_boundary(paths)
    img = rasterize_continuous(cx, cy, values, extent)
    fig, ax = plt.subplots()
    plot_spatial_map(ax, img, extent, boundary, title="My Map")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default raster canvas dimensions (pixels)
# ---------------------------------------------------------------------------

RASTER_W = 2000
RASTER_H = 2400


# ---------------------------------------------------------------------------
# Boundary loading
# ---------------------------------------------------------------------------


def load_boundary(
    paths: StudyAreaPaths,
    crs: int = 28992,
) -> gpd.GeoDataFrame | None:
    """Load study area boundary, filter to largest part, reproject.

    For the Netherlands this filters to European NL (dropping Caribbean
    islands). Returns ``None`` when the boundary file is missing.

    Args:
        paths: ``StudyAreaPaths`` instance pointing at the study area.
        crs: Target CRS as EPSG code.  Defaults to 28992 (RD New).

    Returns:
        GeoDataFrame with a single (Multi)Polygon in the target CRS, or
        ``None`` if the source file does not exist.
    """
    from shapely import get_geometry, get_num_geometries

    boundary_path = paths.area_gdf_file()
    if not boundary_path.exists():
        logger.warning("Boundary file not found: %s", boundary_path)
        return None

    boundary_gdf = gpd.read_file(boundary_path)
    if boundary_gdf.crs is None:
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    boundary_gdf = boundary_gdf.to_crs(epsg=crs)

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


# ---------------------------------------------------------------------------
# Rasterization helpers
# ---------------------------------------------------------------------------


def _stamp_pixels(
    image: np.ndarray,
    py: np.ndarray,
    px: np.ndarray,
    rgb: np.ndarray,
    stamp: int,
    height: int,
    width: int,
) -> None:
    """Write RGB values to *image* with a square stamp of given radius.

    When *stamp* is 1, each point is a single pixel.  Larger values fill a
    square of side ``2*stamp - 1`` centred on each point.
    """
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


def rasterize_continuous(
    cx: np.ndarray,
    cy: np.ndarray,
    values: np.ndarray,
    extent: tuple,
    width: int = RASTER_W,
    height: int = RASTER_H,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    stamp: int = 1,
) -> np.ndarray:
    """Rasterize continuous values to an RGBA image.

    Args:
        cx, cy: Centroid coordinates in EPSG:28992.
        values: Float array of same length as *cx*/*cy*.
        extent: ``(minx, miny, maxx, maxy)`` in EPSG:28992.
        width, height: Output image dimensions in pixels.
        cmap: Matplotlib colormap name.
        vmin, vmax: Value range for colormap normalization.  When ``None``,
            the 2nd/98th percentiles of the data are used.
        stamp: Pixel radius per point (1 = single pixel, 2+ fills gaps at
            coarser resolutions).

    Returns:
        ``(height, width, 4)`` RGBA float32 array with transparent background.
    """
    minx, miny, maxx, maxy = extent
    mask = (
        (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
        & np.isfinite(values)
    )
    cx_m, cy_m, val_m = cx[mask], cy[mask], values[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    if vmin is None:
        vmin = float(np.nanpercentile(val_m, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(val_m, 98))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap_obj = plt.get_cmap(cmap)
    rgb = colormap_obj(norm(val_m))[:, :3].astype(np.float32)

    image = np.zeros((height, width, 4), dtype=np.float32)
    _stamp_pixels(image, py, px, rgb, stamp, height, width)
    return image


def rasterize_rgb(
    cx: np.ndarray,
    cy: np.ndarray,
    rgb_array: np.ndarray,
    extent: tuple,
    width: int = RASTER_W,
    height: int = RASTER_H,
    stamp: int = 1,
) -> np.ndarray:
    """Rasterize pre-computed RGB values to an RGBA image.

    Args:
        cx, cy: Centroid coordinates in EPSG:28992.
        rgb_array: ``(N, 3)`` float array with R, G, B in ``[0, 1]``.
        extent: ``(minx, miny, maxx, maxy)`` in EPSG:28992.
        width, height: Output image dimensions in pixels.
        stamp: Pixel radius per point.

    Returns:
        ``(height, width, 4)`` RGBA float32 array with transparent background.
    """
    minx, miny, maxx, maxy = extent
    mask = (
        (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    )

    cx_m = cx[mask]
    cy_m = cy[mask]
    rgb_masked = rgb_array[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    image = np.zeros((height, width, 4), dtype=np.float32)
    _stamp_pixels(image, py, px, rgb_masked, stamp, height, width)
    return image


def rasterize_binary(
    cx: np.ndarray,
    cy: np.ndarray,
    extent: tuple,
    width: int = RASTER_W,
    height: int = RASTER_H,
    color: tuple = (0.2, 0.5, 0.8),
    stamp: int = 1,
) -> np.ndarray:
    """Rasterize binary presence to an RGBA image.

    Args:
        cx, cy: Centroid coordinates in EPSG:28992.
        extent: ``(minx, miny, maxx, maxy)`` in EPSG:28992.
        width, height: Output image dimensions in pixels.
        color: RGB tuple for presence pixels.
        stamp: Pixel radius per point.

    Returns:
        ``(height, width, 4)`` RGBA float32 array with transparent background.
    """
    minx, miny, maxx, maxy = extent
    mask = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    cx_m, cy_m = cx[mask], cy[mask]

    px = ((cx_m - minx) / (maxx - minx) * (width - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (height - 1)).astype(int)
    np.clip(px, 0, width - 1, out=px)
    np.clip(py, 0, height - 1, out=py)

    n = len(px)
    rgb = np.broadcast_to(np.array(color, dtype=np.float32), (n, 3)).copy()
    image = np.zeros((height, width, 4), dtype=np.float32)
    _stamp_pixels(image, py, px, rgb, stamp, height, width)
    return image


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
        cx, cy: Centroid coordinates in EPSG:28992.
        labels: Integer cluster assignment array.
        extent: ``(minx, miny, maxx, maxy)`` in EPSG:28992.
        n_clusters: Total number of clusters (for colormap scaling).
        width, height: Output image dimensions in pixels.
        cmap: Matplotlib colormap name.
        stamp: Pixel radius per point (1 = single pixel, 2+ fills gaps at
            coarser resolutions).

    Returns:
        ``(height, width, 4)`` RGBA float32 array with transparent background.
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
# Shared plot helpers
# ---------------------------------------------------------------------------


def _clean_map_axes(ax) -> None:
    """Remove ticks and labels for a clean map look."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def _add_rd_grid(ax, extent: tuple, *, show_labels: bool = True) -> None:
    """Add RD New (EPSG:28992) coordinate grid lines and optional labels.

    Args:
        ax: Matplotlib axes.
        extent: ``(minx, miny, maxx, maxy)`` in EPSG:28992.
        show_labels: Whether to add coordinate labels alongside grid lines.
    """
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

    if show_labels:
        for x in x_grid:
            if minx <= x <= maxx:
                ax.text(
                    x, miny - (maxy - miny) * 0.015, f'{x:.0f}',
                    ha='center', va='top', fontsize=7, color='#555555',
                    bbox=dict(
                        boxstyle='round,pad=0.2', facecolor='white', alpha=0.7,
                    ),
                )
        for y in y_grid:
            if miny <= y <= maxy:
                ax.text(
                    minx - (maxx - minx) * 0.01, y, f'{y:.0f}',
                    ha='right', va='center', fontsize=7, color='#555555',
                    bbox=dict(
                        boxstyle='round,pad=0.2', facecolor='white', alpha=0.7,
                    ),
                )


def plot_spatial_map(
    ax,
    image: np.ndarray,
    extent: tuple,
    boundary_gdf: gpd.GeoDataFrame | None,
    title: str = "",
    *,
    show_rd_grid: bool = True,
    title_fontsize: int = 11,
) -> None:
    """Render a rasterized image on axes with boundary underlay and RD grid.

    Args:
        ax: Matplotlib axes.
        image: RGBA raster from one of the ``rasterize_*`` functions.
        extent: ``(minx, miny, maxx, maxy)`` in EPSG:28992.
        boundary_gdf: Study area boundary for underlay, or ``None`` to skip.
        title: Axes title string.
        show_rd_grid: Whether to overlay the 50-km RD grid.
        title_fontsize: Font size for the title.
    """
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
    if show_rd_grid:
        _add_rd_grid(ax, extent)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    _clean_map_axes(ax)
    if title:
        ax.set_title(title, fontsize=title_fontsize)


def _add_colorbar(
    fig,
    ax,
    cmap: str,
    vmin: float,
    vmax: float,
    label: str = "",
    *,
    label_fontsize: int = 10,
    tick_fontsize: int = 8,
):
    """Add a vertical colorbar to the right of an axes.

    Args:
        fig: Matplotlib figure.
        ax: Matplotlib axes.
        cmap: Colormap name.
        vmin, vmax: Value range for normalization.
        label: Colorbar label.
        label_fontsize: Font size for the label text.
        tick_fontsize: Font size for tick labels.

    Returns:
        Matplotlib colorbar instance.
    """
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    return cbar


def detect_embedding_columns(df) -> list[str]:
    """Detect embedding columns from a DataFrame.

    Tries known prefix patterns (A00, emb_0, gtfs2vec_0, etc.), then falls
    back to all numeric non-metadata columns.

    Args:
        df: pandas DataFrame.

    Returns:
        Sorted list of embedding column names.
    """
    import pandas as pd

    # Pattern 1: A00, A01, ... (AlphaEarth style)
    cols = [c for c in df.columns if len(c) >= 3 and c[0] == "A" and c[1:].isdigit()]
    if cols:
        return sorted(cols, key=lambda c: int(c[1:]))

    # Pattern 2: emb_0, emb_1, ... (SRAI style)
    cols = [c for c in df.columns if c.startswith("emb_")]
    if cols:
        return sorted(cols, key=lambda c: int(c.split("_")[1]))

    # Pattern 2b: gtfs2vec_0, gtfs2vec_1, ... (GTFS encoder style)
    cols = [c for c in df.columns if c.startswith("gtfs2vec_")]
    if cols:
        return sorted(cols, key=lambda c: int(c.split("_")[1]))

    # Pattern 3: Single-letter prefix with digits (P00, R00, etc.)
    for prefix in ("P", "R", "S", "G", "D"):
        cols = [c for c in df.columns if c.startswith(prefix) and len(c) >= 2 and c[1:].isdigit()]
        if cols:
            return sorted(cols, key=lambda c: int(c[1:]))

    # Fallback: all numeric columns except known metadata
    exclude = {
        "pixel_count", "tile_count", "geometry", "region_id",
        "cluster_id", "h3_resolution",
    }
    cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols
