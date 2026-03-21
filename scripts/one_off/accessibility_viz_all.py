"""Comprehensive accessibility graph visualizations.

Generates 5 plot groups (21 figures total):
  1. Overview maps: degree + gravity for walk/bike/drive (6 figs)
  2. City network maps: walk edges over hex polygons (4 figs)
  3. Gravity floodfill: 7 origins grid + 2 individual (3 figs)
  4. Isochrone maps: Dijkstra 10-min walk (3 figs)
  5. Lattice comparison: degree deviation for all modes (3 figs)

Overview and lattice maps use auto-sized disk stamps derived from
h3.average_hexagon_edge_length(). Isochrone and network maps use hex
polygons with OSM basemaps.

Lifetime: temporary
Stage: 3 (analysis/visualization)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import contextily as cx
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from shapely.geometry import Polygon

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import RASTER_H, RASTER_W, _add_colorbar

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

STUDY_AREA = "netherlands"
DPI = 150

paths = StudyAreaPaths(STUDY_AREA)
db = SpatialDB.for_study_area(STUDY_AREA)
FIGURES_DIR = paths.root / "accessibility" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Mode configs: (parquet_name, resolution)
MODES = [
    ("walk_res9", 9),
    ("bike_res8", 8),
    ("drive_res7", 7),
]


# ===================================================================
# Hex polygon renderer — SRAI regions_gdf with geopandas .plot()
# ===================================================================

def load_regions_rd(resolution: int) -> gpd.GeoDataFrame:
    """Load regions_gdf for a resolution, reprojected to EPSG:28992."""
    gdf = gpd.read_parquet(paths.region_file(resolution))
    return gdf.to_crs(epsg=28992)


# ===================================================================
# Helpers
# ===================================================================

def hex_to_polygon_wgs84(hex_id: str) -> Polygon:
    """Convert a single H3 hex ID to a Shapely polygon in WGS84."""
    boundary = h3.cell_to_boundary(hex_id)
    coords = [(lng, lat) for lat, lng in boundary]
    return Polygon(coords)


def find_nearest_hex(
    rd_x: float, rd_y: float, hex_ids: np.ndarray,
    cx_arr: np.ndarray, cy_arr: np.ndarray,
) -> str:
    """Find the hex nearest to the given RD coordinate."""
    dists = (cx_arr - rd_x) ** 2 + (cy_arr - rd_y) ** 2
    idx = int(np.nanargmin(dists))
    return hex_ids[idx]



def plot_hex_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    cmap: str,
    title: str,
    colorbar_label: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot hex polygons filled by column value. No borders, white bg."""
    vals = gdf[column].dropna()
    if vmin is None:
        vmin = float(vals.quantile(0.02))
    if vmax is None:
        vmax = float(vals.quantile(0.98))

    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_facecolor("white")

    gdf.plot(
        ax=ax, column=column, cmap=cmap,
        edgecolor="none", linewidth=0,
        vmin=vmin, vmax=vmax,
        missing_kwds={"color": "none"},  # skip NaN hexes
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    _add_colorbar(fig, ax, cmap, vmin, vmax, label=colorbar_label)

    n_valid = int(vals.notna().sum())
    n_total = len(gdf)
    ax.text(
        0.02, 0.02, f"{n_valid:,} / {n_total:,} hexes",
        transform=ax.transAxes, fontsize=9, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved %s", out_path)


# ===================================================================
# PLOT GROUP 1: Overview maps (hex polygons, white background)
# ===================================================================

def plot_group_1():
    """6 overview maps: degree + gravity for walk/bike/drive."""
    logger.info("=" * 60)
    logger.info("PLOT GROUP 1: Overview maps (hex polygons)")
    logger.info("=" * 60)

    for mode_name, res in MODES:
        mode_label = mode_name.split("_")[0].title()
        logger.info("Loading %s...", mode_name)

        edges_df = pd.read_parquet(
            paths.root / "accessibility" / f"{mode_name}.parquet"
        )

        # Degree per origin hex
        degree = edges_df.groupby("origin_hex").size().rename("degree")
        # Mean incoming gravity weight per dest hex
        gravity = edges_df.groupby("dest_hex")["gravity_weight"].mean().rename("gravity")

        # Load ALL hexes at this resolution — gives full NL shape
        logger.info("  Loading regions_gdf for res%d...", res)
        regions = load_regions_rd(res)
        logger.info("  %d hexes total", len(regions))

        # Join values
        regions = regions.join(degree).join(gravity)

        # -- Degree map --
        n_deg = int(regions["degree"].notna().sum())
        logger.info("  Rendering degree map (%d hexes with values)...", n_deg)
        plot_hex_map(
            regions, "degree", "viridis",
            title=f"{mode_label} Accessibility -- Degree (res{res})",
            colorbar_label="Edges per hex",
            out_path=FIGURES_DIR / f"{mode_name}_degree.png",
        )

        # -- Gravity map --
        n_grav = int(regions["gravity"].notna().sum())
        logger.info("  Rendering gravity map (%d hexes with values)...", n_grav)
        plot_hex_map(
            regions, "gravity", "magma",
            title=f"{mode_label} Accessibility -- Mean Gravity Weight (res{res})",
            colorbar_label="Mean gravity weight",
            out_path=FIGURES_DIR / f"{mode_name}_gravity.png",
        )

        del edges_df, regions


# ===================================================================
# PLOT GROUP 2: City network maps (walk_res9, hex polygon mode)
# ===================================================================

CITIES = {
    "amsterdam": {"center": (121000, 487000), "zoom": 5000, "name": "Amsterdam"},
    "rotterdam": {"center": (92000, 437000), "zoom": 5000, "name": "Rotterdam"},
    "denhaag": {"center": (81000, 454000), "zoom": 5000, "name": "Den Haag"},
    "eindhoven": {"center": (162000, 383000), "zoom": 5000, "name": "Eindhoven"},
}


def plot_group_2():
    """4 city network maps with hex polygons and edge lines."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PLOT GROUP 2: City network maps")
    logger.info("=" * 60)

    walk_edges = pd.read_parquet(
        paths.root / "accessibility" / "walk_res9.parquet"
    )

    # Load res9 regions_gdf for hex polygons
    regions_gdf = gpd.read_parquet(paths.region_file(9))
    regions_rd = regions_gdf.to_crs(epsg=28992)

    # Build centroid lookup for all walk hexes
    all_walk_hexes = sorted(
        set(walk_edges["origin_hex"]) | set(walk_edges["dest_hex"])
    )
    logger.info("Computing centroids for %d walk hexes...", len(all_walk_hexes))
    walk_cx, walk_cy = db.centroids(all_walk_hexes, 9, crs=28992)
    hex_to_x = {h: float(walk_cx[i]) for i, h in enumerate(all_walk_hexes)}
    hex_to_y = {h: float(walk_cy[i]) for i, h in enumerate(all_walk_hexes)}

    # Pre-compute coordinates on edges
    walk_edges["ox"] = walk_edges["origin_hex"].map(hex_to_x)
    walk_edges["oy"] = walk_edges["origin_hex"].map(hex_to_y)
    walk_edges["dx"] = walk_edges["dest_hex"].map(hex_to_x)
    walk_edges["dy"] = walk_edges["dest_hex"].map(hex_to_y)

    # Compute edge distance
    walk_edges["dist_m"] = np.sqrt(
        (walk_edges["dx"] - walk_edges["ox"]) ** 2
        + (walk_edges["dy"] - walk_edges["oy"]) ** 2
    )

    # Mean incoming gravity per dest hex
    gravity_all = walk_edges.groupby("dest_hex")["gravity_weight"].mean()

    for city_slug, city_cfg in CITIES.items():
        logger.info("  %s...", city_cfg["name"])
        cx_c, cy_c = city_cfg["center"]
        zoom = city_cfg["zoom"]
        xmin, xmax = cx_c - zoom, cx_c + zoom
        ymin, ymax = cy_c - zoom, cy_c + zoom

        # Filter edges where BOTH endpoints are in bbox AND distance < 1000m
        in_origin = (
            (walk_edges["ox"] >= xmin) & (walk_edges["ox"] <= xmax)
            & (walk_edges["oy"] >= ymin) & (walk_edges["oy"] <= ymax)
        )
        in_dest = (
            (walk_edges["dx"] >= xmin) & (walk_edges["dx"] <= xmax)
            & (walk_edges["dy"] >= ymin) & (walk_edges["dy"] <= ymax)
        )
        local = walk_edges[in_origin & in_dest].copy()
        local = local[local["dist_m"] <= 1000].copy()

        logger.info("    %d edges in bbox (both endpoints + <1000m)", len(local))
        if len(local) == 0:
            logger.warning("    No edges found, skipping %s", city_slug)
            continue

        local_hexes = sorted(
            set(local["origin_hex"]) | set(local["dest_hex"])
        )

        # Build hex polygon GeoDataFrame
        local_regions = regions_rd.loc[regions_rd.index.isin(local_hexes)].copy()
        local_regions["gravity"] = local_regions.index.map(
            lambda h: gravity_all.get(h, 0.0)
        )

        # Build LineCollection for edges
        segments = np.stack(
            [
                np.column_stack([local["ox"].values, local["oy"].values]),
                np.column_stack([local["dx"].values, local["dy"].values]),
            ],
            axis=1,
        )
        grav_vals = local["gravity_weight"].values

        fig, ax = plt.subplots(figsize=(12, 12))

        # 1. Hex polygons colored by gravity weight -- faint, no outlines
        gmin = local_regions["gravity"].quantile(0.02)
        gmax = local_regions["gravity"].quantile(0.98)
        local_regions.plot(
            ax=ax,
            column="gravity",
            cmap="YlOrRd",
            edgecolor="none",
            linewidth=0,
            alpha=0.3,
            legend=False,
            vmin=gmin,
            vmax=gmax,
            zorder=2,
        )

        # 2. Edge lines -- thick with black outline for contrast
        # Black outline pass (slightly wider)
        lc_bg = LineCollection(segments, linewidths=2.0, alpha=0.5, zorder=3)
        lc_bg.set_color("black")
        ax.add_collection(lc_bg)
        # Colored pass on top
        lc = LineCollection(segments, linewidths=1.2, alpha=0.8, zorder=4)
        grav_p2 = float(np.nanpercentile(grav_vals, 2))
        grav_p98 = float(np.nanpercentile(grav_vals, 98))
        lc.set_array(grav_vals)
        lc.set_clim(grav_p2, grav_p98)
        lc.set_cmap("YlOrRd")
        ax.add_collection(lc)

        # 3. Set limits and add basemap
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        try:
            cx.add_basemap(
                ax, crs="EPSG:28992",
                source=cx.providers.CartoDB.Positron,
                zoom=13,
            )
        except Exception as e:
            logger.warning("    Basemap failed: %s", e)

        # Single colorbar for gravity
        _add_colorbar(fig, ax, "YlOrRd", gmin, gmax, label="Mean gravity weight")

        ax.set_title(
            f"{city_cfg['name']} -- Walk Network (res9)",
            fontsize=14, fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        out = FIGURES_DIR / f"network_{city_slug}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("    Saved %s", out)

    del walk_edges


# (Group 3 — gravity floodfill — removed)


# ===================================================================
# PLOT GROUP 4: Isochrone maps (hex polygons + OSM basemap)
# ===================================================================

ISOCHRONE_ORIGINS = [
    ("Amsterdam Centraal", 121000, 487000),
    ("Rotterdam Centraal", 92000, 437000),
    ("Den Haag Centraal", 81000, 454000),
    ("Utrecht Centraal", 136000, 456000),
    ("Eindhoven Centraal", 162000, 383000),
    ("Rural Drenthe", 240000, 545000),
    ("Rural Zeeland", 45000, 390000),
]


def plot_group_4():
    """Isochrone maps: Dijkstra 10-min reachability from 4 origins."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PLOT GROUP 4: Isochrone maps (hex polygons + basemap)")
    logger.info("=" * 60)

    edges = pd.read_parquet(paths.root / "accessibility" / "walk_res9.parquet")

    # Build NetworkX DiGraph
    logger.info("Building NetworkX DiGraph from %d edges...", len(edges))
    G = nx.DiGraph()
    edge_tuples = list(zip(
        edges["origin_hex"].values,
        edges["dest_hex"].values,
        edges["travel_time_s"].values,
    ))
    G.add_weighted_edges_from(edge_tuples, weight="travel_time_s")
    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # Get centroids for nearest-hex lookup
    all_nodes = np.array(list(G.nodes()), dtype=str)
    logger.info("Computing centroids for %d graph nodes...", len(all_nodes))
    node_cx, node_cy = db.centroids(list(all_nodes), 9, crs=28992)

    # -- 4x2 grid (7 origins + 1 empty) --
    fig, axes = plt.subplots(4, 2, figsize=(20, 48))
    axes_flat = axes.flatten()

    individual_data = {}  # store for individual plots

    for i, (name, rd_x, rd_y) in enumerate(ISOCHRONE_ORIGINS):
        ax = axes_flat[i]
        logger.info("  %s...", name)

        origin = find_nearest_hex(rd_x, rd_y, all_nodes, node_cx, node_cy)
        logger.info("    Origin hex: %s", origin)

        # Dijkstra with 600s cutoff
        try:
            lengths = nx.single_source_dijkstra_path_length(
                G, origin, cutoff=600, weight="travel_time_s",
            )
        except nx.NetworkXError:
            logger.warning("    Origin %s not in graph, skipping", origin)
            ax.set_title(f"{name}\n(origin not in graph)", fontsize=11)
            ax.set_axis_off()
            continue

        reachable_hexes = list(lengths.keys())
        travel_times = np.array(list(lengths.values()), dtype=np.float64)
        logger.info("    %d hexes reachable in 10 min", len(reachable_hexes))

        # Convert hex IDs to polygons
        polys = [hex_to_polygon_wgs84(h) for h in reachable_hexes]
        reach_gdf = gpd.GeoDataFrame(
            {"travel_time_s": travel_times},
            geometry=polys,
            crs="EPSG:4326",
        ).to_crs(epsg=28992)

        # Find origin centroid in RD
        origin_idx = reachable_hexes.index(origin)
        o_cx_rd, o_cy_rd = db.centroids([origin], 9, crs=28992)
        o_cx_val, o_cy_val = float(o_cx_rd[0]), float(o_cy_rd[0])

        # Store for individual plots
        if name in ("Amsterdam Centraal", "Eindhoven Centraal"):
            individual_data[name] = (reach_gdf, origin, o_cx_val, o_cy_val, len(reachable_hexes))

        # Plot hex polygons
        window = 8000
        reach_gdf.plot(
            ax=ax,
            column="travel_time_s",
            cmap="plasma",
            vmin=0,
            vmax=600,
            edgecolor="none",
            alpha=0.7,
            legend=False,
        )

        # White star for origin
        ax.plot(
            o_cx_val, o_cy_val, marker="*", color="white", markersize=18,
            markeredgecolor="black", markeredgewidth=1.0, zorder=10,
        )

        ax.set_xlim(o_cx_val - window, o_cx_val + window)
        ax.set_ylim(o_cy_val - window, o_cy_val + window)

        # OSM basemap
        try:
            cx.add_basemap(
                ax, crs="EPSG:28992",
                source=cx.providers.CartoDB.Positron,
                zoom=12,
            )
        except Exception as e:
            logger.warning("    Basemap failed: %s", e)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{name}\n{len(reachable_hexes)} hexes reachable in 10 min",
            fontsize=12, fontweight="bold",
        )

    # Hide unused axes
    for idx in range(len(ISOCHRONE_ORIGINS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap="plasma", norm=Normalize(vmin=0, vmax=600)
    )
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes_flat.tolist(), fraction=0.02, pad=0.02,
        label="Travel time (seconds)",
    )

    fig.suptitle(
        "10-Minute Walk Isochrones (Dijkstra on walk_res9)",
        fontsize=16, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.95, 0.99])
    out = FIGURES_DIR / "walk_res9_isochrones.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", out)

    # -- Individual isochrone maps --
    for name, (reach_gdf, origin, o_cx_val, o_cy_val, n_reach) in individual_data.items():
        slug = name.split()[0].lower()
        logger.info("Rendering individual isochrone: %s...", name)
        fig_ind, ax_ind = plt.subplots(figsize=(14, 14))

        window = 8000
        reach_gdf.plot(
            ax=ax_ind,
            column="travel_time_s",
            cmap="plasma",
            vmin=0,
            vmax=600,
            edgecolor="none",
            alpha=0.7,
            legend=False,
        )
        ax_ind.plot(
            o_cx_val, o_cy_val, marker="*", color="white", markersize=22,
            markeredgecolor="black", markeredgewidth=1.0, zorder=10,
        )
        ax_ind.set_xlim(o_cx_val - window, o_cx_val + window)
        ax_ind.set_ylim(o_cy_val - window, o_cy_val + window)

        try:
            cx.add_basemap(
                ax_ind, crs="EPSG:28992",
                source=cx.providers.CartoDB.Positron,
                zoom=12,
            )
        except Exception as e:
            logger.warning("    Basemap failed: %s", e)

        sm = plt.cm.ScalarMappable(
            cmap="plasma", norm=Normalize(vmin=0, vmax=600)
        )
        sm.set_array([])
        fig_ind.colorbar(sm, ax=ax_ind, fraction=0.035, pad=0.02, label="Travel time (s)")

        ax_ind.set_xticks([])
        ax_ind.set_yticks([])
        ax_ind.set_title(
            f"{name} -- 10-min Walk Isochrone\n{n_reach} hexes reachable",
            fontsize=14, fontweight="bold",
        )
        fig_ind.tight_layout()
        out = FIGURES_DIR / f"isochrone_{slug}.png"
        fig_ind.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig_ind)
        logger.info("  Saved %s", out)

    del edges, G


# ===================================================================
# PLOT GROUP 5: Lattice comparison (degree deviation, all modes)
# ===================================================================

def plot_group_5():
    """Degree deviation from uniform 6-ring lattice for all modes."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PLOT GROUP 5: Lattice comparison (all modes)")
    logger.info("=" * 60)

    for mode_name, res in MODES:
        mode_label = mode_name.split("_")[0].title()
        logger.info("Loading %s...", mode_name)

        edges = pd.read_parquet(
            paths.root / "accessibility" / f"{mode_name}.parquet"
        )
        degree = edges.groupby("origin_hex").size()
        deviation = (degree - 6).rename("deviation")

        regions = load_regions_rd(res)
        regions = regions.join(deviation)
        n_dev = int(regions["deviation"].notna().sum())
        logger.info("  Rendering lattice comparison (%d hexes, res%d)...", n_dev, res)

        plot_hex_map(
            regions, "deviation", "RdBu_r",
            title=f"{mode_label} Accessibility -- Degree Deviation from H3 Lattice (degree - 6)",
            colorbar_label="Degree - 6",
            out_path=FIGURES_DIR / f"{mode_name}_vs_lattice.png",
            vmin=-6, vmax=6,
        )

        del edges, regions


# ===================================================================
# Main
# ===================================================================

def main():
    logger.info("Output directory: %s", FIGURES_DIR)
    logger.info("Canvas: %d x %d px, DPI=%d", RASTER_W, RASTER_H, DPI)
    plot_group_1()
    plot_group_2()
    plot_group_4()
    plot_group_5()
    logger.info("")
    logger.info("All done. All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
