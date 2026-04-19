"""Accessibility graph visualization: overview maps + isochrone analysis.

Lifetime: durable
Stage: 3 (analysis / visualization)

Produces:
  - 6 overview maps (degree + gravity for walk/bike/drive)
  - 1 walk-vs-lattice divergence map
  - 1 combined 2x2 isochrone figure
  - 2 individual zoomed isochrone maps (Amsterdam, Rotterdam)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import (
    _add_colorbar,
    load_boundary,
    plot_spatial_map,
    rasterize_continuous,
)

STUDY_AREA = "netherlands"
ACC_DIR = StudyAreaPaths(STUDY_AREA).accessibility()
FIG_DIR = ACC_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_edges(name: str) -> pd.DataFrame:
    return pd.read_parquet(ACC_DIR / f"{name}.parquet")


def get_all_hex_ids(df: pd.DataFrame) -> np.ndarray:
    """Union of origin and dest hex IDs."""
    return np.array(
        sorted(set(df["origin_hex"].unique()) | set(df["dest_hex"].unique()))
    )


def get_centroids(hex_ids: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    db = SpatialDB.for_study_area(STUDY_AREA)
    cx, cy = db.centroids(hex_ids, resolution, crs=28992)
    return cx, cy


def compute_extent(cx: np.ndarray, cy: np.ndarray, pad_frac: float = 0.02):
    minx, maxx = cx.min(), cx.max()
    miny, maxy = cy.min(), cy.max()
    dx = (maxx - minx) * pad_frac
    dy = (maxy - miny) * pad_frac
    return (minx - dx, miny - dy, maxx + dx, maxy + dy)


def make_single_map(
    cx, cy, values, extent, boundary,
    cmap, title, cb_label, filepath,
    vmin=None, vmax=None, stamp=1,
):
    """Render and save a single full-NL map."""
    fig, ax = plt.subplots(figsize=(12, 14))
    img = rasterize_continuous(cx, cy, values, extent, cmap=cmap, vmin=vmin, vmax=vmax, stamp=stamp)
    plot_spatial_map(ax, img, extent, boundary, title=title)

    v0 = vmin if vmin is not None else float(np.nanpercentile(values[np.isfinite(values)], 2))
    v1 = vmax if vmax is not None else float(np.nanpercentile(values[np.isfinite(values)], 98))
    _add_colorbar(fig, ax, cmap, v0, v1, label=cb_label)

    fig.savefig(filepath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {filepath.name}")


# -----------------------------------------------------------------------
# PART 1: Overview Maps
# -----------------------------------------------------------------------

def plot_overview_maps():
    paths = StudyAreaPaths(STUDY_AREA)
    boundary = load_boundary(paths)

    # NOTE: bike_res8 and drive_res7 both contain res7 hex IDs.
    # The file name refers to the coarsening strategy, not the hex resolution.
    # (name, actual_h3_resolution, stamp_size)
    configs = [
        ("walk_res9", 9, 1),
        ("bike_res8", 7, 7),
        ("drive_res7", 7, 7),
    ]

    for name, res, stamp in configs:
        print(f"\n--- {name} ---")
        df = load_edges(name)
        hex_ids = get_all_hex_ids(df)
        cx, cy = get_centroids(hex_ids, res)
        extent = compute_extent(cx, cy)

        # Build lookup: hex_id -> index
        hex_to_idx = {h: i for i, h in enumerate(hex_ids)}

        # Degree map
        degree_series = df.groupby("origin_hex").size()
        degree_vals = np.zeros(len(hex_ids), dtype=float)
        for h, d in degree_series.items():
            if h in hex_to_idx:
                degree_vals[hex_to_idx[h]] = d
        make_single_map(
            cx, cy, degree_vals, extent, boundary,
            cmap="viridis",
            title=f"{name} — Edge Degree per Hex",
            cb_label="Edges (count)",
            filepath=FIG_DIR / f"{name}_degree.png",
            stamp=stamp,
        )

        # Gravity map
        gravity_series = df.groupby("dest_hex")["gravity_weight"].mean()
        gravity_vals = np.zeros(len(hex_ids), dtype=float)
        for h, g in gravity_series.items():
            if h in hex_to_idx:
                gravity_vals[hex_to_idx[h]] = g
        make_single_map(
            cx, cy, gravity_vals, extent, boundary,
            cmap="magma",
            title=f"{name} — Mean Incoming Gravity Weight",
            cb_label="Mean gravity weight",
            filepath=FIG_DIR / f"{name}_gravity.png",
            stamp=stamp,
        )

    # Walk degree vs lattice divergence
    print("\n--- walk_res9 vs lattice ---")
    df = load_edges("walk_res9")
    hex_ids = get_all_hex_ids(df)
    cx, cy = get_centroids(hex_ids, 9)
    extent = compute_extent(cx, cy)

    hex_to_idx = {h: i for i, h in enumerate(hex_ids)}
    degree_series = df.groupby("origin_hex").size()
    divergence = np.full(len(hex_ids), np.nan, dtype=float)
    for h, d in degree_series.items():
        if h in hex_to_idx:
            divergence[hex_to_idx[h]] = d - 6  # uniform 1-ring = 6

    make_single_map(
        cx, cy, divergence, extent, boundary,
        cmap="RdBu_r",
        title="Walk res9 — Degree Divergence from Uniform Lattice (degree - 6)",
        cb_label="Edges above/below 6",
        filepath=FIG_DIR / "walk_res9_vs_lattice.png",
        vmin=-6, vmax=20,
        stamp=1,
    )


# -----------------------------------------------------------------------
# PART 2: Isochrone Analysis
# -----------------------------------------------------------------------

ORIGINS = {
    "Amsterdam Centraal": (121000, 487000),
    "Rotterdam Centraal": (92000, 437000),
    "Zoetermeer (suburban)": (97000, 453000),
    "Drenthe (rural)": (240000, 545000),
}

CUTOFF_S = 600  # 10 minutes walking


def find_nearest_hex(cx, cy, hex_ids, target_x, target_y):
    dists = np.sqrt((cx - target_x) ** 2 + (cy - target_y) ** 2)
    idx = np.argmin(dists)
    return hex_ids[idx], cx[idx], cy[idx]


def build_walk_graph(df: pd.DataFrame) -> nx.DiGraph:
    print("  Building NetworkX graph from 2.3M edges...")
    G = nx.from_pandas_edgelist(
        df, "origin_hex", "dest_hex",
        edge_attr="travel_time_s",
        create_using=nx.DiGraph,
    )
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_isochrone(G, origin_hex, cutoff_s):
    distances = nx.single_source_dijkstra_path_length(
        G, source=origin_hex, weight="travel_time_s", cutoff=cutoff_s,
    )
    return distances


def plot_isochrones_combined():
    print("\n=== PART 2: Isochrone Analysis ===")

    df = load_edges("walk_res9")
    hex_ids = get_all_hex_ids(df)
    cx, cy = get_centroids(hex_ids, 9)
    hex_to_idx = {h: i for i, h in enumerate(hex_ids)}

    G = build_walk_graph(df)

    paths = StudyAreaPaths(STUDY_AREA)
    boundary = load_boundary(paths)

    # Combined 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 24))
    axes_flat = axes.flatten()

    for i, (label, (tx, ty)) in enumerate(ORIGINS.items()):
        origin_hex, ox, oy = find_nearest_hex(cx, cy, hex_ids, tx, ty)
        print(f"\n  {label}: origin={origin_hex} at ({ox:.0f}, {oy:.0f})")

        distances = compute_isochrone(G, origin_hex, CUTOFF_S)
        n_reach = len(distances)
        max_t = max(distances.values()) if distances else 0
        print(f"    Reachable: {n_reach} hexes, max travel time: {max_t:.0f}s")

        # Build arrays for reachable hexes
        reach_hexes = list(distances.keys())
        reach_times = np.array([distances[h] for h in reach_hexes])
        reach_cx = np.array([cx[hex_to_idx[h]] for h in reach_hexes if h in hex_to_idx])
        reach_cy = np.array([cy[hex_to_idx[h]] for h in reach_hexes if h in hex_to_idx])
        # Filter to only hexes we have coordinates for
        valid = [h for h in reach_hexes if h in hex_to_idx]
        reach_times = np.array([distances[h] for h in valid])
        reach_cx = np.array([cx[hex_to_idx[h]] for h in valid])
        reach_cy = np.array([cy[hex_to_idx[h]] for h in valid])

        # Zoomed extent: compute from reachable hexes with 30% padding
        if len(reach_cx) > 1:
            rminx, rmaxx = reach_cx.min(), reach_cx.max()
            rminy, rmaxy = reach_cy.min(), reach_cy.max()
            rdx = max((rmaxx - rminx) * 0.3, 200)
            rdy = max((rmaxy - rminy) * 0.3, 200)
            local_extent = (rminx - rdx, rminy - rdy, rmaxx + rdx, rmaxy + rdy)
        else:
            window = 2000
            local_extent = (ox - window, oy - window, ox + window, oy + window)

        ax = axes_flat[i]
        # Compute stamp size based on zoom level: at ~2km window, hex spacing
        # is ~175m (res9), canvas is 2000px, so pixels per hex ~ 2000/(window_m/175)
        window_x = local_extent[2] - local_extent[0]
        iso_stamp = max(1, int(2000 / (window_x / 175) * 0.6))
        iso_stamp = min(iso_stamp, 12)  # cap
        img = rasterize_continuous(
            reach_cx, reach_cy, reach_times, local_extent,
            cmap="plasma", vmin=0, vmax=CUTOFF_S, stamp=iso_stamp,
        )
        plot_spatial_map(
            ax, img, local_extent, boundary,
            title=f"{label}: {n_reach:,} reachable, max {max_t:.0f}s",
            show_rd_grid=False,
        )
        # Mark origin
        ax.plot(ox, oy, marker="*", color="white", markersize=15,
                markeredgecolor="black", markeredgewidth=1.0, zorder=10)

        _add_colorbar(fig, ax, "plasma", 0, CUTOFF_S, label="Travel time (s)")

    fig.suptitle(
        f"Walk Isochrones (res9, {CUTOFF_S}s cutoff)",
        fontsize=16, fontweight="bold", y=0.92,
    )
    fig.savefig(FIG_DIR / "walk_res9_isochrones.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Saved walk_res9_isochrones.png")

    # Individual zoomed maps for Amsterdam and Rotterdam
    for label in ["Amsterdam Centraal", "Rotterdam Centraal"]:
        tx, ty = ORIGINS[label]
        origin_hex, ox, oy = find_nearest_hex(cx, cy, hex_ids, tx, ty)
        distances = compute_isochrone(G, origin_hex, CUTOFF_S)
        n_reach = len(distances)
        max_t = max(distances.values()) if distances else 0

        valid = [h for h in distances.keys() if h in hex_to_idx]
        reach_times = np.array([distances[h] for h in valid])
        reach_cx = np.array([cx[hex_to_idx[h]] for h in valid])
        reach_cy = np.array([cy[hex_to_idx[h]] for h in valid])

        # Tight zoom from reachable hexes with 20% padding
        if len(reach_cx) > 1:
            rminx, rmaxx = reach_cx.min(), reach_cx.max()
            rminy, rmaxy = reach_cy.min(), reach_cy.max()
            rdx = max((rmaxx - rminx) * 0.2, 200)
            rdy = max((rmaxy - rminy) * 0.2, 200)
            local_extent = (rminx - rdx, rminy - rdy, rmaxx + rdx, rmaxy + rdy)
        else:
            local_extent = (ox - 2000, oy - 2000, ox + 2000, oy + 2000)

        window_x = local_extent[2] - local_extent[0]
        ind_stamp = max(1, int(2000 / (window_x / 175) * 0.6))
        ind_stamp = min(ind_stamp, 12)

        fig, ax = plt.subplots(figsize=(12, 14))
        img = rasterize_continuous(
            reach_cx, reach_cy, reach_times, local_extent,
            cmap="plasma", vmin=0, vmax=CUTOFF_S, stamp=ind_stamp,
        )
        plot_spatial_map(
            ax, img, local_extent, boundary,
            title=f"{label} — Walk Isochrone ({CUTOFF_S}s cutoff, {n_reach:,} reachable)",
            show_rd_grid=False,
        )
        ax.plot(ox, oy, marker="*", color="white", markersize=20,
                markeredgecolor="black", markeredgewidth=1.5, zorder=10)
        _add_colorbar(fig, ax, "plasma", 0, CUTOFF_S, label="Travel time (s)")

        slug = label.split()[0].lower()
        filepath = FIG_DIR / f"isochrone_{slug}.png"
        fig.savefig(filepath, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {filepath.name}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PART 1: Overview Maps ===")
    plot_overview_maps()
    plot_isochrones_combined()
    print("\nDone. All figures saved to:", FIG_DIR)
