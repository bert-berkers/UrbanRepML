"""Shared rasterization and spatial visualization helpers.

Provides centroid-based rasterization of H3-indexed data onto matplotlib
axes. Originally developed in ``scripts/plot_embeddings.py``; extracted
here so every stage-3 visualization script can import a single canonical
copy instead of copy-pasting.

All coordinate arguments are expected in EPSG:28992 (RD New) unless noted
otherwise.

Typical usage::

    from utils.visualization import (
        filter_empty_hexagons,
        load_boundary,
        rasterize_continuous,
        rasterize_categorical,
        plot_spatial_map,
    )

    boundary = load_boundary(paths)
    emb_df = filter_empty_hexagons(emb_df, display_name="AlphaEarth")
    img = rasterize_continuous(cx, cy, values, extent)
    fig, ax = plt.subplots()
    plot_spatial_map(ax, img, extent, boundary, title="My Map")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    .. deprecated:: 2026-05-02
       Centroid-splat with degree-units ``stamp`` carries directional bleed,
       speckle holes, and lat-lon aspect distortion. Migrate to
       :func:`rasterize_continuous_voronoi` (KDTree-Voronoi in metric CRS,
       ``max_dist_m`` in metres). See ``specs/rasterize_voronoi.md`` and
       ``.claude/plans/2026-05-02-rasterize-voronoi-toolkit.md`` (W3).
       Removed in W6 of the toolkit migration.

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

    .. deprecated:: 2026-05-02
       Migrate to :func:`rasterize_rgb_voronoi` (KDTree-Voronoi, metric CRS,
       ``max_dist_m`` cutoff). Removed in W6 of the toolkit migration.

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

    .. deprecated:: 2026-05-02
       Migrate to :func:`rasterize_binary_voronoi` (KDTree-Voronoi, metric
       CRS, ``max_dist_m`` cutoff). Removed in W6.

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

    .. deprecated:: 2026-05-02
       Migrate to :func:`rasterize_categorical_voronoi` (KDTree-Voronoi,
       metric CRS, ``max_dist_m`` cutoff; supports ``color_map`` dict
       override). Removed in W6.

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
    disable_rd_grid: bool = False,
) -> None:
    """Render a rasterized image on axes with boundary underlay and RD grid.

    Args:
        ax: Matplotlib axes.
        image: RGBA raster from one of the ``rasterize_*`` functions.
        extent: ``(minx, miny, maxx, maxy)`` in EPSG:28992 (legacy
            centroid-splat form). The new Voronoi rasterizers return
            ``(minx, maxx, miny, maxy)``; callers using them must
            re-order before passing here, or set ``disable_rd_grid=True``
            and pass the matplotlib-order tuple verbatim (the imshow path
            doesn't care about Y order).
        boundary_gdf: Study area boundary for underlay, or ``None`` to skip.
        title: Axes title string.
        show_rd_grid: Whether to overlay the 50-km RD grid.
        title_fontsize: Font size for the title.
        disable_rd_grid: Hard suppress the RD grid overlay. Takes precedence
            over ``show_rd_grid``. Absorbs the ``plot_targets.py`` shadow
            override (W3 case 3) — used by callers that don't want the RD
            grid even when other panels in the same figure do.
    """
    if boundary_gdf is not None:
        boundary_gdf.plot(
            ax=ax, facecolor="none", edgecolor="#cccccc", linewidth=0.5,
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
    if show_rd_grid and not disable_rd_grid:
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


# ---------------------------------------------------------------------------
# Empty/background hexagon filter
# ---------------------------------------------------------------------------


def filter_empty_hexagons(
    emb_df: pd.DataFrame,
    display_name: str,
    constant_threshold: float = 0.10,
) -> pd.DataFrame:
    """Filter out hexagons with no meaningful embedding signal.

    Three-pass strategy:

    Pass 0: Remove rows matching a dominant repeated vector (>50% of rows share
        the same embedding). Catches encoders like GTFS2Vec that assign a
        learned default embedding to hexagons without data.
    Pass 1: Remove rows where ALL embedding dims are exactly zero.
    Pass 2: Remove low-variance rows (water/empty hexagons that the encoder
        maps to near-identical output vectors). Uses row-wise std with a
        bimodal gap detection between "background" and "real" embeddings.
        Skipped when Pass 0 already found a dominant vector.

    Args:
        emb_df: DataFrame of embedding values indexed by H3 cell ID.
        display_name: Human-readable modality name for log messages.
        constant_threshold: Minimum fraction of rows that must be flagged
            as low-variance before Pass 2 applies (default 0.10 = 10%).

    Returns:
        Filtered DataFrame with background/empty rows removed.
    """
    n_original = len(emb_df)
    vals = emb_df.values
    to_drop = np.zeros(n_original, dtype=bool)

    # Pass 0: dominant repeated vector (non-zero default embeddings)
    # Some encoders (e.g. GTFS2Vec) produce a shared learned embedding for
    # hexagons without data. These are not all-zero but are all identical.
    # Detect by measuring distance from the most common row.
    found_dominant = False
    if n_original > 100:
        # Sample to find candidate default vector efficiently
        sample_idx = np.linspace(0, n_original - 1, min(10000, n_original), dtype=int)
        sample = vals[sample_idx].astype(np.float64)

        # Find the most common vector: round to 6 decimals, count unique rows
        rounded = np.round(sample, 6)
        unique_vecs, inverse, counts = np.unique(
            rounded, axis=0, return_inverse=True, return_counts=True,
        )
        dominant_idx = np.argmax(counts)
        dominant_frac = counts[dominant_idx] / len(sample)

        if dominant_frac > 0.5:
            # More than 50% of the sample matches one vector -- likely a default
            default_vec = unique_vecs[dominant_idx]
            # Check all rows against this vector (L-inf distance)
            diffs = np.abs(vals.astype(np.float64) - default_vec)
            is_default = diffs.max(axis=1) < 1e-5
            n_default = int(is_default.sum())
            if n_default > n_original * 0.5:
                to_drop |= is_default
                found_dominant = True
                logger.info(
                    "  Pass 0: %d dominant-vector rows (%.1f%%) -- "
                    "shared default embedding detected",
                    n_default, 100.0 * n_default / n_original,
                )

    # Pass 1: all-zero rows
    all_zero = (vals == 0.0).all(axis=1)
    n_zero = int(all_zero.sum())
    if n_zero > 0:
        new_zeros = all_zero & ~to_drop
        n_new = int(new_zeros.sum())
        if n_new > 0:
            to_drop |= all_zero
            logger.info(
                "  Pass 1: %d all-zero rows (%.1f%%)",
                n_zero, 100.0 * n_zero / n_original,
            )

    # Pass 2: low-variance rows (near-constant embeddings)
    # Skip if Pass 0 already found a dominant vector -- the remaining rows
    # are real data with naturally varied variance. Pass 2 is designed for
    # modalities without a clear default embedding (e.g. AlphaEarth water
    # hexagons that produce near-constant encoder output).
    if n_original > 100 and not found_dominant:
        row_std = vals.std(axis=1)
        p10_std = np.percentile(row_std, 10)
        p50_std = np.percentile(row_std, 50)
        if p50_std > 0 and p10_std / p50_std < 0.8:
            cutoff = (p10_std + p50_std) / 2
        else:
            overall_std = vals.std()
            cutoff = overall_std * 0.01 if overall_std > 0 else 0

        low_var = row_std < cutoff
        n_low = int(low_var.sum())
        n_new_low = int((low_var & ~to_drop).sum())
        if n_new_low > 0 and n_low / n_original >= constant_threshold:
            to_drop |= low_var
            logger.info(
                "  Pass 2: %d low-variance rows (%.1f%%, std < %.4f)",
                n_low, 100.0 * n_low / n_original, cutoff,
            )

    n_drop = int(to_drop.sum())
    if n_drop == 0:
        logger.info("  No empty/constant hexagons to filter for %s", display_name)
        return emb_df

    filtered = emb_df[~to_drop]
    logger.info(
        "  Filtered %d -> %d hexagons (dropped %d, %.1f%%) for %s",
        n_original, len(filtered), n_drop, 100.0 * n_drop / n_original, display_name,
    )
    return filtered


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


# ---------------------------------------------------------------------------
# KDTree-Voronoi rasterization (new standard; replaces centroid-splat above).
#
# Contract: ``specs/rasterize_voronoi.md`` (frozen 2026-05-02 by W1).
# Lifted from: ``scripts/one_off/viz_ring_agg_res9_grid.py:74-167`` (visually
# validated by the human across the three-embeddings study + LBM probe overlay).
# Plan: ``.claude/plans/2026-05-02-rasterize-voronoi-toolkit.md`` (W2a).
#
# Three structural wins over centroid-splat:
#   1. No directional south-east bleed (was caused by asymmetric splat offsets).
#   2. No density-dependent speckle holes (was hexagonal-vs-rectangular packing).
#   3. No lat-lon aspect distortion (1 deg lon != 1 deg lat at 52 N -- handled
#      by working in metric CRS, default EPSG:28992 for NL).
#
# One operational win: gallery reuse. The KDTree query is the dominant cost
# (~5s for NL res9 at 250 m/px). Splitting it from the colour-mapping step
# lets a gallery render 8 panels for the price of one Voronoi by re-using
# ``nearest_idx``. Per-panel cost is trivial fancy-indexing.
# ---------------------------------------------------------------------------


def voronoi_indices(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Build the per-pixel nearest-hex index array + alpha mask once.

    The KDTree query is the dominant cost; this function isolates it so a
    gallery of N panels can re-use ``nearest_idx`` via N cheap
    :func:`gather_rgba` calls instead of N full Voronoi queries.

    Args:
        cx_m: Hex centroid x-coords (1-D float, length N) in a metric CRS.
        cy_m: Hex centroid y-coords (1-D float, length N) in the same CRS.
        extent_m: ``(minx, miny, maxx, maxy)`` bounding box in the same
            metric CRS. Note the input order matches
            ``geopandas.total_bounds`` and shapely conventions.
        pixel_m: Output pixel size (metres). Keyword-only.
        max_dist_m: Voronoi cutoff (metres). Pixels farther than this from
            their nearest centroid are masked out. Keyword-only.

    Returns:
        ``(nearest_idx, inside, extent_xy)`` where:
            * ``nearest_idx``: ``(H, W)`` int64, index into the centroid
              array. Dtype is load-bearing -- :func:`gather_rgba` assumes
              int64 indexing into ``(N, 3)`` colour arrays.
            * ``inside``: ``(H, W)`` bool, True where pixel is within
              ``max_dist_m`` of its nearest centroid (alpha mask).
            * ``extent_xy``: ``(minx, maxx, miny, maxy)`` -- the **Y order
              is swapped from input** because matplotlib
              ``imshow(..., extent=...)`` expects ``(left, right, bottom,
              top)``. This is intentional, not a bug.

    Notes:
        Pixel-centre convention: pixel ``(0, 0)`` covers
        ``[minx, minx+pixel_m] x [miny, miny+pixel_m]`` and is queried at
        its centre. Matches ``imshow(origin='lower')``.

        Determinism: ``cKDTree.query(k=1, eps=0)`` is deterministic. Same
        inputs + same ``pixel_m`` + same ``max_dist_m`` produce byte-
        identical ``nearest_idx`` and ``inside``. Ties are broken by input
        order.
    """
    from scipy.spatial import cKDTree  # local import: optional dep at module top

    cx_m = np.asarray(cx_m, dtype=np.float64)
    cy_m = np.asarray(cy_m, dtype=np.float64)
    if cx_m.ndim != 1 or cy_m.ndim != 1 or cx_m.shape != cy_m.shape:
        raise ValueError(
            f"cx_m and cy_m must be 1-D arrays of identical length; got "
            f"shapes {cx_m.shape} and {cy_m.shape}"
        )

    minx, miny, maxx, maxy = extent_m
    width = max(1, int(np.ceil((maxx - minx) / pixel_m)))
    height = max(1, int(np.ceil((maxy - miny) / pixel_m)))

    # Pixel-centre coordinates in metric CRS (origin='lower' convention).
    xs = minx + (np.arange(width) + 0.5) * pixel_m
    ys = miny + (np.arange(height) + 0.5) * pixel_m
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    tree = cKDTree(np.column_stack([cx_m, cy_m]))
    dist, idx = tree.query(pts, k=1)

    nearest_idx = idx.reshape(height, width).astype(np.int64)
    inside = (dist <= max_dist_m).reshape(height, width)
    return nearest_idx, inside, (minx, maxx, miny, maxy)


def gather_rgba(
    nearest_idx: np.ndarray,
    inside: np.ndarray,
    rgb_per_hex: np.ndarray,
) -> np.ndarray:
    """Project a per-hex colour table onto the precomputed index grid.

    Cheap fancy-index gather. Use this for gallery panels after a single
    :func:`voronoi_indices` call.

    Args:
        nearest_idx: ``(H, W)`` int from :func:`voronoi_indices`.
        inside: ``(H, W)`` bool from :func:`voronoi_indices`.
        rgb_per_hex: ``(N, 3)`` float in ``[0, 1]``. ``N`` must match the
            centroid array length passed to :func:`voronoi_indices`.

    Returns:
        ``(H, W, 4)`` float32 RGBA. RGB is ``rgb_per_hex[nearest_idx]``;
        alpha is hard-edged (1.0 inside, 0.0 outside) by design.
    """
    rgb_per_hex = np.asarray(rgb_per_hex)
    if rgb_per_hex.ndim != 2 or rgb_per_hex.shape[1] != 3:
        raise ValueError(
            f"rgb_per_hex must be (N, 3); got shape {rgb_per_hex.shape}"
        )
    h, w = nearest_idx.shape
    img = np.zeros((h, w, 4), dtype=np.float32)
    img[..., :3] = rgb_per_hex[nearest_idx]
    img[..., 3] = inside.astype(np.float32)
    return img


def rasterize_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    rgb_per_hex: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """One-shot KDTree-Voronoi rasterization in a metric CRS.

    Thin wrapper around :func:`voronoi_indices` + :func:`gather_rgba` for
    callers that only need a single colour mapping. For galleries, call
    :func:`voronoi_indices` once and :func:`gather_rgba` per panel.

    Args:
        cx_m: Hex centroid x-coords in a metric CRS.
        cy_m: Hex centroid y-coords in the same CRS.
        rgb_per_hex: ``(N, 3)`` float in ``[0, 1]``.
        extent_m: ``(minx, miny, maxx, maxy)`` in the same metric CRS.
        pixel_m: Output pixel size (metres). Keyword-only.
        max_dist_m: Voronoi cutoff (metres). Keyword-only.

    Returns:
        ``(image, extent_xy)``. ``image`` is ``(H, W, 4)`` float32 RGBA.
        ``extent_xy`` is ``(minx, maxx, miny, maxy)`` -- directly usable as
        ``imshow(image, extent=extent_xy, origin='lower',
        interpolation='nearest', aspect='equal')``.
    """
    nearest_idx, inside, extent_xy = voronoi_indices(
        cx_m, cy_m, extent_m, pixel_m=pixel_m, max_dist_m=max_dist_m,
    )
    return gather_rgba(nearest_idx, inside, rgb_per_hex), extent_xy


def voronoi_params_for_resolution(
    resolution: int,
) -> tuple[float, float]:
    """Suggested ``(pixel_m, max_dist_m)`` for an H3 resolution.

    Per ``specs/rasterize_voronoi.md`` §"Defaults" guidance table. Used
    by W3-migrated callers that previously derived a pixel-radius
    ``stamp`` from ``max(1, 11 - resolution)``; this is the metric-CRS
    equivalent for the Voronoi rasterizer.

    Args:
        resolution: H3 resolution (5..11).

    Returns:
        ``(pixel_m, max_dist_m)`` in metres. Falls back to res9 baseline
        for unknown resolutions.
    """
    table = {
        7: (1500.0, 2000.0),
        8: (600.0, 800.0),
        9: (250.0, 300.0),
        10: (100.0, 120.0),
        11: (40.0, 50.0),
    }
    return table.get(resolution, (250.0, 300.0))


def _apply_bg_color(
    image: np.ndarray,
    bg_color: tuple[float, float, float, float] | tuple[float, float, float],
) -> np.ndarray:
    """Paint outside-cutoff pixels (alpha=0) with ``bg_color`` (alpha=1).

    Used by per-mode wrappers when ``bg_color`` kwarg is set, to absorb
    the ``plot_targets.py`` shadow's white-background semantics. A 3-tuple
    is treated as opaque (alpha=1); a 4-tuple uses its alpha component.

    Args:
        image: ``(H, W, 4)`` float32 RGBA from :func:`rasterize_voronoi`.
        bg_color: RGB (3-tuple) or RGBA (4-tuple) in ``[0, 1]``.

    Returns:
        New ``(H, W, 4)`` float32 RGBA with outside-cutoff pixels painted.
    """
    if len(bg_color) not in (3, 4):
        raise ValueError(
            f"bg_color must be RGB or RGBA tuple; got length {len(bg_color)}"
        )
    out = image.copy()
    outside = out[..., 3] <= 0.0
    rgb = np.array(bg_color[:3], dtype=np.float32)
    alpha = float(bg_color[3]) if len(bg_color) == 4 else 1.0
    out[outside, :3] = rgb
    out[outside, 3] = alpha
    return out


def rasterize_labels(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    labels: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
    fill_value: int = -1,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Integer labels -> int64 label grid (peer to Voronoi RGBA API).

    Returns the *integer label* of the nearest hex per pixel (or
    ``fill_value`` outside ``max_dist_m``). Used downstream by edge
    detection and label-boundary visualisation; an RGBA gather would
    erase the integer-label structure those algorithms need.

    Internally a thin wrapper over :func:`voronoi_indices` +
    ``np.where(inside, labels[nearest_idx], fill_value)``. Lifted from
    ``scripts/stage3/plot_targets.py:rasterize_labels_to_grid`` per
    ``specs/rasterize_voronoi.md`` §"Peer function: rasterize_labels"
    (W3 case 2).

    Args:
        cx_m, cy_m: Hex centroid coordinates in a metric CRS.
        labels: Integer label array of length N.
        extent_m: ``(minx, miny, maxx, maxy)`` in the same metric CRS.
        pixel_m: Output pixel size (metres). Keyword-only.
        max_dist_m: Voronoi cutoff (metres). Keyword-only.
        fill_value: Value for pixels outside the cutoff. Default -1.
            Keyword-only.

    Returns:
        ``(label_grid, extent_xy)`` where:
            * ``label_grid``: ``(H, W)`` int64; integer labels inside,
              ``fill_value`` outside.
            * ``extent_xy``: ``(minx, maxx, miny, maxy)`` -- matplotlib
              ``imshow`` order (Y swapped from input).
    """
    labels = np.asarray(labels)
    nearest_idx, inside, extent_xy = voronoi_indices(
        cx_m, cy_m, extent_m, pixel_m=pixel_m, max_dist_m=max_dist_m,
    )
    # int64 throughout; fill_value cast to int64 for type stability.
    gathered = labels[nearest_idx].astype(np.int64, copy=False)
    label_grid = np.where(inside, gathered, np.int64(fill_value))
    return label_grid, extent_xy


# ---------------------------------------------------------------------------
# Per-mode wrappers: replace the four deprecated `rasterize_*` functions.
# `stamp` is GONE -- the new API uses `max_dist_m` (geometric meaning).
# ---------------------------------------------------------------------------


def rasterize_continuous_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    values: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
    bg_color: tuple[float, float, float, float] | tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Continuous scalar values -> colormap -> RGBA Voronoi raster.

    Args:
        cx_m, cy_m: Hex centroid coordinates in a metric CRS.
        values: Float array of same length as cx_m/cy_m.
        extent_m: ``(minx, miny, maxx, maxy)`` in the same metric CRS.
        cmap: Matplotlib colormap name. Keyword-only.
        vmin, vmax: Value range for colormap normalization. When ``None``,
            the 2nd/98th percentiles of ``values`` are used (preserves
            ``rasterize_continuous`` behaviour).
        pixel_m: Output pixel size (metres). Keyword-only.
        max_dist_m: Voronoi cutoff (metres). Keyword-only.
        bg_color: Optional RGB or RGBA tuple in [0, 1] for outside-cutoff
            pixels. When ``None`` (default), outside-cutoff pixels are
            transparent (alpha=0). When set, outside-cutoff pixels are
            painted with ``bg_color`` and alpha=1 (3-tuple is treated as
            opaque). Absorbs the ``plot_targets.py`` shadow's
            white-background semantics (W3 case 1). Keyword-only.

    Returns:
        ``(image, extent_xy)``. See :func:`rasterize_voronoi`.
    """
    values = np.asarray(values)
    finite = np.isfinite(values)
    # Compute percentiles over finite values only; preserves legacy behaviour.
    if vmin is None:
        vmin = float(np.nanpercentile(values[finite], 2)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(values[finite], 98)) if finite.any() else 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap_obj = plt.get_cmap(cmap)
    rgb_per_hex = colormap_obj(norm(values))[:, :3].astype(np.float32)
    # NaN values would propagate as colormap's "bad" colour (often transparent
    # in newer matplotlib); force them to the colormap's vmin colour for
    # parity with the deprecated centroid-splat behaviour where they were
    # silently dropped by the finite-mask filter. Hex remains in the centroid
    # array, so the Voronoi cell still paints.
    if (~finite).any():
        bad_rgb = colormap_obj(0.0)[:3]
        rgb_per_hex[~finite] = np.array(bad_rgb, dtype=np.float32)

    image, extent_xy = rasterize_voronoi(
        cx_m, cy_m, rgb_per_hex, extent_m,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )
    if bg_color is not None:
        image = _apply_bg_color(image, bg_color)
    return image, extent_xy


def rasterize_categorical_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    labels: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    n_clusters: int | None = None,
    cmap: str = "tab20",
    color_map: dict | None = None,
    fallback_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
    bg_color: tuple[float, float, float, float] | tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Integer or string cluster labels -> categorical RGBA Voronoi raster.

    Args:
        cx_m, cy_m: Hex centroid coordinates in a metric CRS.
        labels: Cluster assignment array (length N). Integer or string.
        extent_m: ``(minx, miny, maxx, maxy)`` in the same metric CRS.
        n_clusters: Total number of integer clusters (for colormap
            scaling). Required for the ``cmap`` path; ignored when
            ``color_map`` provides full coverage. Keyword-only.
        cmap: Matplotlib colormap name. Used when ``color_map`` is None
            (integer labels only).
        color_map: Optional ``dict[label -> (r, g, b)]`` mapping. When
            provided, takes precedence over ``cmap`` for labels present
            in the dict. Supports both integer and string labels.
            Absorbs the ``plot_targets.py`` shadow's distinguishing
            parameter (W3 case 1).
        fallback_color: RGB used for labels not in ``color_map`` when
            no ``cmap`` path applies (e.g. string labels missing from
            the dict). Default ``(0.8, 0.8, 0.8)`` matches the
            ``plot_targets.py`` shadow's gray fallback.
        pixel_m: Output pixel size (metres). Keyword-only.
        max_dist_m: Voronoi cutoff (metres). Keyword-only.
        bg_color: Optional RGB or RGBA tuple in [0, 1] for outside-cutoff
            pixels. When ``None`` (default), outside-cutoff pixels are
            transparent. When set, painted with ``bg_color`` (alpha=1).
            Keyword-only.

    Returns:
        ``(image, extent_xy)``. See :func:`rasterize_voronoi`.

    Notes:
        Does NOT validate ``labels.max() < n_clusters``. ``n_clusters``
        is used only for ``cmap`` scaling; out-of-range labels are
        gracefully mod-wrapped by matplotlib's colormap-clip semantics.
        See W2 scratchpad for the validation-vs-defensive choice.
    """
    labels = np.asarray(labels)
    n = len(labels)

    # Detect whether we can take the cmap-scaling path: requires integer
    # labels and an n_clusters value. Strings or n_clusters=None mean
    # color_map must cover every label (fallback_color for missing keys).
    is_integer = np.issubdtype(labels.dtype, np.integer)

    if is_integer and n_clusters is not None:
        colormap_obj = plt.get_cmap(cmap)
        norm_vals = labels.astype(float) / max(n_clusters - 1, 1)
        rgb_per_hex = colormap_obj(norm_vals)[:, :3].astype(np.float32)
    else:
        rgb_per_hex = np.broadcast_to(
            np.array(fallback_color, dtype=np.float32), (n, 3),
        ).copy()

    if color_map is not None:
        # Apply explicit overrides for labels present in the dict.
        for lbl, rgb in color_map.items():
            mask = labels == lbl
            if mask.any():
                rgb_per_hex[mask] = np.array(rgb[:3], dtype=np.float32)

    image, extent_xy = rasterize_voronoi(
        cx_m, cy_m, rgb_per_hex, extent_m,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )
    if bg_color is not None:
        image = _apply_bg_color(image, bg_color)
    return image, extent_xy


def rasterize_binary_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    color: tuple[float, float, float] = (0.2, 0.5, 0.8),
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Presence-only Voronoi raster -- all hexes painted in ``color``.

    Args:
        cx_m, cy_m: Hex centroid coordinates in a metric CRS.
        extent_m: ``(minx, miny, maxx, maxy)`` in the same metric CRS.
        color: RGB tuple for presence pixels. Keyword-only.
        pixel_m: Output pixel size (metres). Keyword-only.
        max_dist_m: Voronoi cutoff (metres). Keyword-only.

    Returns:
        ``(image, extent_xy)``. See :func:`rasterize_voronoi`.
    """
    cx_m = np.asarray(cx_m, dtype=np.float64)
    n = cx_m.shape[0]
    rgb_per_hex = np.broadcast_to(
        np.array(color, dtype=np.float32), (n, 3),
    ).copy()
    return rasterize_voronoi(
        cx_m, cy_m, rgb_per_hex, extent_m,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )


def rasterize_rgb_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    rgb_array: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Pre-computed (N, 3) RGB -> RGBA Voronoi raster.

    Identity wrapper around :func:`rasterize_voronoi` -- kept for naming
    parity with the deprecated :func:`rasterize_rgb`.

    Args:
        cx_m, cy_m: Hex centroid coordinates in a metric CRS.
        rgb_array: ``(N, 3)`` float in ``[0, 1]``.
        extent_m: ``(minx, miny, maxx, maxy)`` in the same metric CRS.
        pixel_m: Output pixel size (metres). Keyword-only.
        max_dist_m: Voronoi cutoff (metres). Keyword-only.

    Returns:
        ``(image, extent_xy)``. See :func:`rasterize_voronoi`.
    """
    return rasterize_voronoi(
        cx_m, cy_m, rgb_array, extent_m,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )


# ---------------------------------------------------------------------------
# CRS adapter: lat/lon -> metric.
# ---------------------------------------------------------------------------


def latlon_to_metric(
    lats: np.ndarray,
    lons: np.ndarray,
    target_crs: int = 28992,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject lat/lon (EPSG:4326) arrays to a metric CRS.

    Required for callers whose centroid arrays are in EPSG:4326 (e.g. SRAI
    defaults). Uses ``pyproj.Transformer`` with ``always_xy=True``. Note
    the argument order: lat (Y) first to match the SRAI / GeoPandas
    convention of returning ``(latitude, longitude)`` from many APIs;
    internally swapped to ``(lon, lat)`` for ``always_xy=True`` semantics.

    Args:
        lats: Latitude array (degrees, EPSG:4326).
        lons: Longitude array (degrees, EPSG:4326).
        target_crs: Target metric CRS as EPSG code. Defaults to 28992
            (RD New) for NL.

    Returns:
        ``(cx_m, cy_m)``: x and y arrays in the target CRS, same shape
        as the inputs.
    """
    import pyproj

    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    transformer = pyproj.Transformer.from_crs(
        4326, target_crs, always_xy=True,
    )
    cx_m, cy_m = transformer.transform(lons, lats)
    return np.asarray(cx_m), np.asarray(cy_m)


# ---------------------------------------------------------------------------
# GeoDataFrame overload wrappers: SRAI-indexed (region_id) GDF -> rasterizer.
#
# Pattern choice (see W2 scratchpad): separate `*_gdf` functions, NOT
# @singledispatch. Rationale: explicit dispatch is more discoverable,
# matches SRAI conventions where overloading is rare, and avoids hiding
# the (cx_m, cy_m) -> GDF conversion behind a single name. W3 callers
# can pick the form that matches their data plumbing.
# ---------------------------------------------------------------------------


def _gdf_to_metric_centroids(
    gdf: gpd.GeoDataFrame,
    target_crs: int = 28992,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Extract metric-CRS centroids + total_bounds from a SRAI-indexed GDF.

    Args:
        gdf: GeoDataFrame indexed by ``region_id`` (SRAI convention; NOT
            ``h3_index``). Must have a ``geometry`` column. CRS metadata
            must be set; if ``gdf.crs is None`` we assume EPSG:4326 as
            the SRAI H3Regionalizer default.
        target_crs: Target metric CRS as EPSG code.

    Returns:
        ``(cx_m, cy_m, extent_m)`` -- x, y arrays + ``(minx, miny, maxx,
        maxy)`` total_bounds in the target CRS.

    Raises:
        ValueError: if the GDF is empty or has no geometry column.
    """
    if "geometry" not in gdf.columns:
        raise ValueError("GeoDataFrame has no 'geometry' column")
    if len(gdf) == 0:
        raise ValueError("GeoDataFrame is empty")

    if gdf.index.name not in ("region_id", None):
        # Soft warning: project standard is `region_id`. Not enforced
        # because some callers may legitimately pass nameless indices.
        logger.warning(
            "GDF index name is %r; project standard is 'region_id'",
            gdf.index.name,
        )

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    if gdf.crs.to_epsg() != target_crs:
        gdf = gdf.to_crs(target_crs)

    centroids = gdf.geometry.centroid
    cx_m = np.asarray(centroids.x.to_numpy(), dtype=np.float64)
    cy_m = np.asarray(centroids.y.to_numpy(), dtype=np.float64)
    minx, miny, maxx, maxy = gdf.total_bounds
    return cx_m, cy_m, (float(minx), float(miny), float(maxx), float(maxy))


def rasterize_continuous_voronoi_gdf(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    *,
    target_crs: int = 28992,
    extent_m: tuple[float, float, float, float] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
    bg_color: tuple[float, float, float, float] | tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """GDF overload of :func:`rasterize_continuous_voronoi`.

    Args:
        gdf: SRAI-indexed (``region_id``) GeoDataFrame with a ``value_col``
            column carrying the continuous values.
        value_col: Column name holding the scalar values.
        target_crs: Metric CRS to reproject to. Default 28992 (RD New).
        extent_m: Override total_bounds; defaults to ``gdf.total_bounds``
            in the target CRS.
        cmap, vmin, vmax, pixel_m, max_dist_m, bg_color: See
            :func:`rasterize_continuous_voronoi`.
    """
    cx_m, cy_m, ext = _gdf_to_metric_centroids(gdf, target_crs=target_crs)
    extent_m = extent_m or ext
    values = gdf[value_col].to_numpy()
    return rasterize_continuous_voronoi(
        cx_m, cy_m, values, extent_m,
        cmap=cmap, vmin=vmin, vmax=vmax,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
        bg_color=bg_color,
    )


def rasterize_categorical_voronoi_gdf(
    gdf: gpd.GeoDataFrame,
    label_col: str,
    *,
    n_clusters: int | None = None,
    target_crs: int = 28992,
    extent_m: tuple[float, float, float, float] | None = None,
    cmap: str = "tab20",
    color_map: dict | None = None,
    fallback_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
    bg_color: tuple[float, float, float, float] | tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """GDF overload of :func:`rasterize_categorical_voronoi`."""
    cx_m, cy_m, ext = _gdf_to_metric_centroids(gdf, target_crs=target_crs)
    extent_m = extent_m or ext
    labels = gdf[label_col].to_numpy()
    return rasterize_categorical_voronoi(
        cx_m, cy_m, labels, extent_m,
        n_clusters=n_clusters, cmap=cmap, color_map=color_map,
        fallback_color=fallback_color,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
        bg_color=bg_color,
    )


def rasterize_binary_voronoi_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    target_crs: int = 28992,
    extent_m: tuple[float, float, float, float] | None = None,
    color: tuple[float, float, float] = (0.2, 0.5, 0.8),
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """GDF overload of :func:`rasterize_binary_voronoi`."""
    cx_m, cy_m, ext = _gdf_to_metric_centroids(gdf, target_crs=target_crs)
    extent_m = extent_m or ext
    return rasterize_binary_voronoi(
        cx_m, cy_m, extent_m,
        color=color, pixel_m=pixel_m, max_dist_m=max_dist_m,
    )


def rasterize_rgb_voronoi_gdf(
    gdf: gpd.GeoDataFrame,
    rgb_cols: tuple[str, str, str] | list[str],
    *,
    target_crs: int = 28992,
    extent_m: tuple[float, float, float, float] | None = None,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """GDF overload of :func:`rasterize_rgb_voronoi`.

    Args:
        gdf: SRAI-indexed GeoDataFrame.
        rgb_cols: Three column names holding R, G, B in [0, 1].
        Other args: See :func:`rasterize_rgb_voronoi`.
    """
    if len(rgb_cols) != 3:
        raise ValueError(
            f"rgb_cols must be 3 column names; got {len(rgb_cols)}"
        )
    cx_m, cy_m, ext = _gdf_to_metric_centroids(gdf, target_crs=target_crs)
    extent_m = extent_m or ext
    rgb_array = gdf[list(rgb_cols)].to_numpy().astype(np.float32)
    return rasterize_rgb_voronoi(
        cx_m, cy_m, rgb_array, extent_m,
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )


# ---------------------------------------------------------------------------
# Figure-provenance wrapper (W4 of rasterize-voronoi-toolkit plan)
# ---------------------------------------------------------------------------
#
# ``save_voronoi_figure`` writes ``fig.savefig(path)`` and a sibling
# ``{path}.provenance.yaml`` audit-trail file matching the schema in
# ``specs/artifact_provenance.md`` §"Figure-provenance specialisation".
#
# Differs from the existing ``stage3_analysis/save_figure.py`` in three ways:
#   1. Lives in ``utils/`` (broader scope — any caller, not just stage3 viz).
#   2. Records ``plot_config_hash`` (SHA-256-truncated, via
#      ``utils.provenance.compute_config_hash``) so two figures with identical
#      config can be detected without a deep-equality walk.
#   3. Auto-populates ``parent_run_id`` from the active ``SidecarWriter`` via
#      ``utils.provenance.get_active_sidecar()`` — no manual threading.
#
# Failure semantics: if the provenance yaml emission fails (disk error, yaml
# error, anything), the figure save still succeeds.  Provenance is best-effort,
# not a blocker.  Matches the cluster-2 ``save_figure`` pattern.

def save_voronoi_figure(
    fig: "plt.Figure",
    path: "Path | str",
    *,
    source_runs: list[str] | None = None,
    source_artifacts: list[str] | list["Path"] | None = None,
    plot_config: dict | None = None,
    provenance: bool = True,
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: str | None = None,
    producer_script: str | None = None,
):
    """Save *fig* and (by default) emit a sibling ``*.provenance.yaml``.

    Per the W4 contract in ``.claude/plans/2026-05-02-rasterize-voronoi-toolkit.md``
    and ``specs/rasterize_voronoi.md`` §"Provenance integration hook".

    Args:
        fig: matplotlib Figure.
        path: target file path (PNG/SVG/PDF).
        source_runs: list of upstream ``run_id`` strings whose data backs this
            figure.  Empty list is valid (logged but does not raise).
        source_artifacts: list of input file paths that contributed to the
            figure (parquet, csv, etc.).  Recorded relative to project root
            in the provenance yaml.
        plot_config: free-form dict of plot settings (pixel_m, max_dist_m,
            cmap, dpi, ...).  Hashed via ``compute_config_hash`` so the
            figure's plot-config can be deduped without a deep-equality walk.
        provenance: if False, skip the provenance yaml entirely (escape hatch
            for ad-hoc exploratory plotting).  Default True.
        dpi/bbox_inches/facecolor: passed to ``fig.savefig``.
        producer_script: override for the script path (default: caller's
            ``sys.argv[0]`` relativized to project root).

    Returns:
        The resolved figure path (``Path`` instance).

    Notes:
        ``parent_run_id`` is auto-populated when ``save_voronoi_figure`` runs
        inside an active :class:`utils.provenance.SidecarWriter` block.  This
        is read via :func:`utils.provenance.get_active_sidecar` — a contextvar
        registry that the writer manages on enter/exit.

        Provenance emission failures are caught and logged to stderr; the
        figure save itself is the load-bearing operation and never blocked
        by audit-trail issues.
    """
    # Resolve and ensure parent dir.  This must succeed for the figure to save.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Always-load-bearing: write the figure first.
    savefig_kwargs = {"dpi": dpi, "bbox_inches": bbox_inches}
    if facecolor is not None:
        savefig_kwargs["facecolor"] = facecolor
    fig.savefig(path, **savefig_kwargs)

    if not provenance:
        return path

    # Provenance emission is best-effort.  Any failure here logs and returns
    # the figure path unchanged — the figure save is the contract; the audit
    # trail is a courtesy.
    try:
        _emit_voronoi_figure_provenance(
            path=path,
            source_runs=source_runs,
            source_artifacts=source_artifacts,
            plot_config=plot_config,
            producer_script=producer_script,
        )
    except Exception as err:  # noqa: BLE001 — best-effort by design
        import sys
        print(
            f"[save_voronoi_figure WARN] provenance yaml write failed for "
            f"{path.name!r}: {type(err).__name__}: {err}",
            file=sys.stderr,
        )

    return path


def _emit_voronoi_figure_provenance(
    *,
    path: Path,
    source_runs: list[str] | None,
    source_artifacts: list[str] | list[Path] | None,
    plot_config: dict | None,
    producer_script: str | None,
) -> Path:
    """Write the ``{path}.provenance.yaml`` sibling.  Internal helper.

    Schema matches ``specs/artifact_provenance.md`` §"Figure-provenance
    specialisation" plus two extensions specified in W4 of the rasterize-
    voronoi-toolkit plan: ``plot_config_hash`` and ``parent_run_id``.
    """
    import sys
    from datetime import datetime, timezone

    import yaml

    from utils.provenance import (
        _git_info,
        _project_root,
        _relativise,
        compute_config_hash,
        get_active_sidecar,
    )

    # --- normalize inputs --------------------------------------------------
    source_runs = list(source_runs) if source_runs else []
    source_artifacts_rel: list[str] = []
    if source_artifacts:
        source_artifacts_rel = [_relativise(p) for p in source_artifacts]

    plot_config = dict(plot_config) if plot_config else {}
    # Coerce non-JSON-serialisable plot_config values to repr() with a
    # warning, matching ``compute_config_hash``'s behaviour for non-leaf types.
    plot_config_safe: dict = {}
    import json
    for k, v in plot_config.items():
        try:
            json.dumps(v)
            plot_config_safe[k] = v
        except TypeError:
            import warnings
            warnings.warn(
                f"save_voronoi_figure: plot_config[{k!r}] of type "
                f"{type(v).__name__!r} is not JSON-serialisable; coercing "
                f"via repr().",
                UserWarning,
                stacklevel=4,
            )
            plot_config_safe[k] = repr(v)

    # Stable hash regardless of insertion order (compute_config_hash
    # canonicalises with sort_keys=True).
    plot_config_hash = compute_config_hash(plot_config_safe)

    # --- producer_script -----------------------------------------------------
    if producer_script is None:
        try:
            producer_script_rel = _relativise(sys.argv[0]) if sys.argv else None
        except Exception:
            producer_script_rel = sys.argv[0] if sys.argv else None
    else:
        producer_script_rel = producer_script

    # --- parent_run_id (auto-populate from active SidecarWriter) -----------
    active = get_active_sidecar()
    parent_run_id: str | None = None
    if active is not None:
        try:
            parent_run_id = active.run_id  # property; raises if pre-enter
        except RuntimeError:
            parent_run_id = None  # writer constructed but not entered

    git_commit, git_dirty = _git_info()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    provenance_doc = {
        "figure_path": path.name,
        "created_at": now,
        "source_runs": source_runs,
        "source_artifacts": source_artifacts_rel,
        "plot_config": plot_config_safe,
        "plot_config_hash": plot_config_hash,
        "parent_run_id": parent_run_id,
        # 5 base figure-provenance fields from specs/artifact_provenance.md
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "started_at": now,
        "ended_at": now,
        "producer_script": producer_script_rel,
        "schema_version": "1.0",
    }

    sidecar_path = path.with_suffix(path.suffix + ".provenance.yaml")
    with sidecar_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            provenance_doc,
            fh,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    return sidecar_path
