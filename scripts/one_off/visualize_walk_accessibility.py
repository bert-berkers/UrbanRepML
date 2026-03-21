"""Visualize walk accessibility graph: overview maps + isochrone maps.

Lifetime: temporary (one-off visualization redo)
Stage: 3 (post-training analysis / visualization)

Produces:
  - walk_res9_degree.png: degree per hex, viridis, stamp=2, grey background
  - walk_res9_gravity.png: gravity sum per hex, magma, stamp=2, grey background
  - walk_res9_isochrones.png: 2x2 Dijkstra isochrones (10min walk), plasma
  - isochrone_amsterdam.png, isochrone_rotterdam.png: individual zoomed isochrones
  - walk_res9_vs_lattice.png: degree deviation from lattice (degree - 6), RdBu_r
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import (
    _add_colorbar,
    load_boundary,
    plot_spatial_map,
    rasterize_binary,
    rasterize_continuous,
)

STUDY_AREA = "netherlands"
RESOLUTION = 9
STAMP = 2
DPI = 150
BG_COLOR = (0.85, 0.85, 0.85)

paths = StudyAreaPaths(STUDY_AREA)
db = SpatialDB.for_study_area(STUDY_AREA)
boundary = load_boundary(paths)

FIGURES_DIR = paths.root / "accessibility" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
print("Loading walk_res9.parquet...")
edges_df = pd.read_parquet(paths.root / "accessibility" / "walk_res9.parquet")

# Connected hexes
connected_hexes = sorted(set(edges_df["origin_hex"]) | set(edges_df["dest_hex"]))
print(f"Connected hexes: {len(connected_hexes):,}")

# Degree per origin hex
degree = edges_df.groupby("origin_hex").size().rename("degree")
# Gravity sum per origin hex
gravity = edges_df.groupby("origin_hex")["gravity_weight"].sum().rename("gravity")

# ---------------------------------------------------------------
# Load ALL res9 hex centroids for background
# ---------------------------------------------------------------
print("Loading all res9 hex centroids for background...")
import geopandas as gpd

regions_gdf = gpd.read_parquet(paths.region_file(RESOLUTION))
all_hex_ids = list(regions_gdf.index)
print(f"Total res9 hexes: {len(all_hex_ids):,}")

# Get full extent from all hexes
all_cx, all_cy = db.centroids(all_hex_ids, RESOLUTION, crs=28992)
extent = db.extent(all_hex_ids, RESOLUTION, crs=28992)
minx, miny, maxx, maxy = extent
print(f"Extent: {minx:.0f}, {miny:.0f} to {maxx:.0f}, {maxy:.0f}")

# Background image (all hexes in grey)
print("Rasterizing background...")
bg_img = rasterize_binary(all_cx, all_cy, extent, color=BG_COLOR, stamp=STAMP)

# Get connected hex centroids
conn_cx, conn_cy = db.centroids(list(degree.index), RESOLUTION, crs=28992)


# ---------------------------------------------------------------
# Helper: composite plot with background
# ---------------------------------------------------------------
def make_overview_map(values, hex_ids, cmap, title, label, out_path,
                      vmin=None, vmax=None):
    """Create a single overview map with grey background + foreground values."""
    cx, cy = db.centroids(list(hex_ids), RESOLUTION, crs=28992)
    vals = np.array([values[h] for h in hex_ids])

    fg_img = rasterize_continuous(cx, cy, vals, extent, cmap=cmap, stamp=STAMP,
                                  vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 14))
    # Background: all hexes in grey
    ax.imshow(bg_img, extent=[minx, maxx, miny, maxy], origin="lower",
              aspect="equal", interpolation="nearest", zorder=1)
    # Boundary
    if boundary is not None:
        boundary.plot(ax=ax, facecolor="none", edgecolor="#cccccc",
                      linewidth=0.5, zorder=2)
    # Foreground: connected hexes with values
    ax.imshow(fg_img, extent=[minx, maxx, miny, maxy], origin="lower",
              aspect="equal", interpolation="nearest", zorder=3)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Colorbar
    v0 = vmin if vmin is not None else float(np.nanpercentile(vals, 2))
    v1 = vmax if vmax is not None else float(np.nanpercentile(vals, 98))
    _add_colorbar(fig, ax, cmap, v0, v1, label=label)

    n_conn = len(hex_ids)
    n_total = len(all_hex_ids)
    pct = 100 * n_conn / n_total
    ax.text(0.02, 0.02,
            f"{n_conn:,} / {n_total:,} hexes connected ({pct:.1f}%)",
            transform=ax.transAxes, fontsize=9, color="#444444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------
# 1. Degree map
# ---------------------------------------------------------------
print("\n--- Degree map ---")
make_overview_map(
    degree, list(degree.index), "viridis",
    "Walk Accessibility Graph -- Degree (res9)",
    "Edges per hex",
    FIGURES_DIR / "walk_res9_degree.png",
)

# ---------------------------------------------------------------
# 2. Gravity map
# ---------------------------------------------------------------
print("\n--- Gravity map ---")
make_overview_map(
    gravity, list(gravity.index), "magma",
    "Walk Accessibility Graph -- Gravity Weight Sum (res9)",
    "Sum of gravity weights",
    FIGURES_DIR / "walk_res9_gravity.png",
)

# ---------------------------------------------------------------
# 3. Isochrone maps (2x2 + individual)
# ---------------------------------------------------------------
print("\n--- Building graph for isochrones ---")
G = nx.from_pandas_edgelist(
    edges_df, "origin_hex", "dest_hex",
    edge_attr="travel_time_s", create_using=nx.DiGraph,
)
print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Origins in RD coords (approx)
ORIGINS = {
    "Amsterdam Centraal": (121000, 487000),
    "Rotterdam Centraal": (92000, 437000),
    "Zoetermeer (suburban)": (97000, 453000),
    "Drenthe (rural)": (240000, 545000),
}

# Find nearest graph node for each origin
print("Finding nearest graph nodes...")
node_list = list(G.nodes())
node_cx, node_cy = db.centroids(node_list, RESOLUTION, crs=28992)
node_coords = np.column_stack([node_cx, node_cy])

origin_hexes = {}
for name, (ox, oy) in ORIGINS.items():
    dists = np.sqrt((node_cx - ox) ** 2 + (node_cy - oy) ** 2)
    idx = np.argmin(dists)
    origin_hexes[name] = node_list[idx]
    print(f"  {name}: {node_list[idx]} (dist={dists[idx]:.0f}m)")

CUTOFF = 600  # 10 minutes
ZOOM_RADIUS = 8000  # +/- 8km
# Res9 hex edge ~174m. At 16km window on 2000px canvas, that's ~22px between
# hex centers. Stamp=8 gives 15px diameter dots with slight overlap -- good fill.
ISOCHRONE_STAMP = 8


def make_isochrone_ax(ax, name, origin_hex, show_title=True):
    """Draw a single isochrone on the given axes."""
    # Dijkstra
    lengths = nx.single_source_dijkstra_path_length(
        G, origin_hex, cutoff=CUTOFF, weight="travel_time_s"
    )
    reachable = list(lengths.keys())
    times = np.array([lengths[h] for h in reachable])

    # Origin centroid for zoom
    ocx, ocy = db.centroids([origin_hex], RESOLUTION, crs=28992)
    ox, oy = float(ocx[0]), float(ocy[0])

    # Zoom extent
    zoom_extent = (ox - ZOOM_RADIUS, oy - ZOOM_RADIUS,
                   ox + ZOOM_RADIUS, oy + ZOOM_RADIUS)
    zminx, zminy, zmaxx, zmaxy = zoom_extent

    # Background: all res9 hexes in zoom window (lighter grey, smaller stamp)
    zoom_mask = ((all_cx >= zminx) & (all_cx <= zmaxx) &
                 (all_cy >= zminy) & (all_cy <= zmaxy))
    zoom_bg = rasterize_binary(
        all_cx[zoom_mask], all_cy[zoom_mask], zoom_extent,
        color=(0.80, 0.80, 0.80), stamp=ISOCHRONE_STAMP - 2,
    )

    # Foreground: reachable hexes colored by travel time
    r_cx, r_cy = db.centroids(reachable, RESOLUTION, crs=28992)
    fg_img = rasterize_continuous(
        r_cx, r_cy, times, zoom_extent, cmap="plasma", stamp=ISOCHRONE_STAMP,
        vmin=0, vmax=CUTOFF,
    )

    # Draw
    ax.set_facecolor("#f0f0f0")
    ax.imshow(zoom_bg, extent=[zminx, zmaxx, zminy, zmaxy], origin="lower",
              aspect="equal", interpolation="nearest", zorder=1)
    if boundary is not None:
        boundary.plot(ax=ax, facecolor="none", edgecolor="#aaaaaa",
                      linewidth=0.8, zorder=2)
    ax.imshow(fg_img, extent=[zminx, zmaxx, zminy, zmaxy], origin="lower",
              aspect="equal", interpolation="nearest", zorder=3)

    # Mark origin
    ax.plot(ox, oy, "w*", markersize=14, markeredgecolor="black",
            markeredgewidth=1.0, zorder=5)

    ax.set_xlim(zminx, zmaxx)
    ax.set_ylim(zminy, zmaxy)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title(f"{name}\n{len(reachable)} hexes reachable in 10 min",
                      fontsize=11)

    return len(reachable), times


# 2x2 composite
print("\nGenerating 2x2 isochrone plot...")
fig, axes = plt.subplots(2, 2, figsize=(20, 24))
for ax_item, (name, origin_hex) in zip(axes.flat, origin_hexes.items()):
    n_reach, _ = make_isochrone_ax(ax_item, name, origin_hex)
    print(f"  {name}: {n_reach} hexes reachable")

# Shared colorbar
_add_colorbar(fig, axes[0, 1], "plasma", 0, CUTOFF,
              label="Travel time (seconds)")

fig.suptitle("Walk Isochrones (10 min cutoff, res9)", fontsize=16,
             fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGURES_DIR / "walk_res9_isochrones.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {FIGURES_DIR / 'walk_res9_isochrones.png'}")

# Individual zoomed maps
for slug, name in [("amsterdam", "Amsterdam Centraal"),
                   ("rotterdam", "Rotterdam Centraal")]:
    print(f"\nGenerating individual isochrone: {slug}...")
    fig, ax = plt.subplots(figsize=(12, 14))
    n_reach, times = make_isochrone_ax(ax, name, origin_hexes[name])
    _add_colorbar(fig, ax, "plasma", 0, CUTOFF, label="Travel time (seconds)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"isochrone_{slug}.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / f'isochrone_{slug}.png'}")

# ---------------------------------------------------------------
# 4. Lattice comparison map (degree - 6)
# ---------------------------------------------------------------
print("\n--- Lattice comparison map ---")
deviation = degree - 6  # lattice has 6 neighbors
make_overview_map(
    deviation, list(deviation.index), "RdBu_r",
    "Walk Graph vs Lattice -- Degree Deviation (degree - 6)",
    "Degree deviation from lattice (6)",
    FIGURES_DIR / "walk_res9_vs_lattice.png",
    vmin=-6, vmax=6,
)

print("\nDone! All maps saved to:", FIGURES_DIR)
