"""
Accessibility graph generator for stage 2 fusion models.

Builds travel-time and gravity-weighted edges between H3 hexagons using
OSM road network data and RUDIFUN building density (FSI).

The pipeline is split into 4 independent steps, each saving an intermediate
parquet file. You can run them one at a time or all at once.

    Step 1: step_sjoin       — Load roads + hexes, spatial join, save road-hex assignments
    Step 2: step_edges       — Build edges from shared roads, compute travel times
    Step 3: step_rudifun     — Overlay RUDIFUN bouwblok FSI onto H3 hexagons
    Step 4: step_gravity     — Combine edges + FSI into gravity-weighted final output

Usage (run all at once):
    python -m stage2_fusion.graphs.accessibility_graph --study-area netherlands --resolution 9 --mode walk

Usage (run one step at a time):
    python -m stage2_fusion.graphs.accessibility_graph --study-area netherlands --resolution 9 --mode walk --step 1
    python -m stage2_fusion.graphs.accessibility_graph --study-area netherlands --resolution 9 --mode walk --step 2
    python -m stage2_fusion.graphs.accessibility_graph --study-area netherlands --resolution 9 --mode walk --step 3
    python -m stage2_fusion.graphs.accessibility_graph --study-area netherlands --resolution 9 --mode walk --step 4

Intermediate files (saved to data/study_areas/{area}/accessibility/intermediates/):
    - step1_road_hex_assignments.parquet  (~big, road_idx + region_id + highway)
    - step2_raw_edges.parquet             (origin_hex, dest_hex, travel_time_s, fastest_road_class, mode)
    - step3_fsi_per_hex.parquet           (region_id, FSI_24)
    - walk_res9.parquet                   (final output with gravity_weight)

Lifetime: durable
Stage: 2 (fusion)
"""

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.paths import StudyAreaPaths

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Road speed lookup (m/s) by highway class
# Source: legacy graph_construction.py, validated against OSM wiki
# ---------------------------------------------------------------------------

ROAD_SPEEDS: dict[str, float] = {
    "motorway": 33.3,       # ~120 km/h
    "motorway_link": 22.2,  # ~80 km/h
    "trunk": 27.8,          # ~100 km/h
    "trunk_link": 16.7,     # ~60 km/h
    "primary": 22.2,        # ~80 km/h
    "primary_link": 13.9,   # ~50 km/h
    "secondary": 16.7,      # ~60 km/h
    "secondary_link": 11.1, # ~40 km/h
    "tertiary": 11.1,       # ~40 km/h
    "tertiary_link": 8.3,   # ~30 km/h
    "residential": 8.3,     # ~30 km/h
    "living_street": 4.2,   # ~15 km/h
    "unclassified": 8.3,    # ~30 km/h
    "service": 4.2,         # ~15 km/h
    "cycleway": 4.17,       # ~15 km/h
    "footway": 1.4,         # ~5 km/h
    "path": 1.4,            # ~5 km/h
    "pedestrian": 1.4,      # ~5 km/h
    "track": 2.8,           # ~10 km/h
    "steps": 0.7,           # ~2.5 km/h
}

# Mode-specific road type exclusions
MODE_EXCLUSIONS: dict[str, set[str]] = {
    "walk": set(),  # pedestrians can use any road
    "bike": {"motorway", "motorway_link", "trunk", "trunk_link"},
    "drive": {"footway", "path", "pedestrian", "cycleway", "steps"},
}

# Gravity decay constants (beta) per mode — from legacy graph_construction.py
GRAVITY_BETA: dict[str, float] = {
    "walk": 0.002,
    "bike": 0.0012,
    "drive": 0.0008,
}


def _intermediates_dir(paths: StudyAreaPaths) -> Path:
    """Return the directory for intermediate files, creating it if needed."""
    d = paths.accessibility() / "intermediates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prefix(mode: str, resolution: int) -> str:
    """Return filename prefix for intermediates, scoped by mode + resolution."""
    return f"{mode}_res{resolution}_"


# ===================================================================
# STEP 1: Load roads + hexes, spatial join, save road-hex assignments
# ===================================================================
#
# This is the heaviest I/O step. It loads the full OSM PBF, extracts
# road LineStrings, loads all H3 hex polygons, projects both to
# EPSG:28992 (RD New, metric), and runs a spatial join to find which
# roads intersect which hexagons.
#
# Output: step1_road_hex_assignments.parquet
#   Columns: road_idx (int), region_id (str), highway (str)
#   Each row = "road X passes through hexagon Y, road type is Z"
#
# Also saves step1_hex_centroids.parquet for use in step 2.
# ===================================================================

def step_sjoin(
    study_area: str,
    resolution: int,
    mode: str = "walk",
    osm_date: str = "latest",
) -> pd.DataFrame:
    """Step 1: Spatial join roads to hexagons.

    Loads OSM roads from PBF, H3 hex polygons from regions parquet,
    and runs gpd.sjoin to assign each road to all hexagons it touches.

    Returns DataFrame of road-hex assignments.
    Saves intermediate to step1_road_hex_assignments.parquet.
    """
    paths = StudyAreaPaths(study_area)
    inter_dir = _intermediates_dir(paths)

    # --- Load study area boundary (needed by OSMPbfLoader) ---
    logger.info("Step 1: Loading study area boundary...")
    area_gdf_path = paths.area_gdf_file("geojson")
    if not area_gdf_path.exists():
        area_gdf_path = paths.area_gdf_file("parquet")
    area_gdf = gpd.read_file(area_gdf_path)
    if area_gdf.crs != "EPSG:4326":
        area_gdf = area_gdf.to_crs("EPSG:4326")

    # --- Load roads from OSM PBF ---
    # OSMPbfLoader comes from SRAI, uses quackosm under the hood.
    # tags={"highway": True} loads all road types.
    logger.info("Step 1: Loading roads from OSM PBF (this may take a few minutes)...")
    pbf_path = paths.osm_snapshot_pbf(osm_date)
    if not pbf_path.exists():
        raise FileNotFoundError(f"OSM PBF not found: {pbf_path}")

    from srai.loaders import OSMPbfLoader
    loader = OSMPbfLoader(pbf_file=pbf_path)
    roads_gdf = loader.load(area_gdf, tags={"highway": True})

    # Keep only line geometries (no points/polygons)
    roads_gdf = roads_gdf[roads_gdf.geometry.type.isin(["LineString", "MultiLineString"])]
    logger.info(f"  Loaded {len(roads_gdf):,} road segments")

    if "highway" not in roads_gdf.columns:
        raise ValueError("OSMPbfLoader did not return a 'highway' column.")

    # --- Filter by travel mode ---
    # e.g. walk mode keeps all roads, bike excludes motorways, etc.
    excluded = MODE_EXCLUSIONS[mode]
    if excluded:
        before = len(roads_gdf)
        roads_gdf = roads_gdf[~roads_gdf["highway"].isin(excluded)]
        logger.info(f"  Mode '{mode}': removed {before - len(roads_gdf):,} excluded road types, {len(roads_gdf):,} remaining")

    # --- Load hex polygons ---
    logger.info("Step 1: Loading hex polygons...")
    region_path = paths.region_file(resolution)
    if not region_path.exists():
        raise FileNotFoundError(f"Region file not found: {region_path}")
    hexes_gdf = gpd.read_parquet(region_path)
    if hexes_gdf.index.name != "region_id":
        hexes_gdf.index.name = "region_id"
    logger.info(f"  {len(hexes_gdf):,} hexagons at res{resolution}")

    # --- Project to RD New (EPSG:28992) for metric distances ---
    logger.info("Step 1: Projecting to EPSG:28992...")
    hexes_proj = hexes_gdf.to_crs(epsg=28992)
    roads_proj = roads_gdf.to_crs(epsg=28992)

    # --- Save hex centroids for step 2 ---
    # Centroids are needed to compute edge distances, but we don't want
    # to reload all hex polygons in step 2.
    hex_centroids = hexes_proj.geometry.centroid
    centroids_df = pd.DataFrame({
        "region_id": hexes_proj.index,
        "cx": hex_centroids.x.values,
        "cy": hex_centroids.y.values,
    })
    centroids_path = inter_dir / f"{_prefix(mode, resolution)}step1_hex_centroids.parquet"
    centroids_df.to_parquet(centroids_path, index=False)
    logger.info(f"  Saved {len(centroids_df):,} hex centroids to {centroids_path}")

    # --- Spatial join: which roads pass through which hexagons? ---
    # This is the core operation. gpd.sjoin with predicate='intersects'
    # finds all (road, hex) pairs where the road geometry touches the hex polygon.
    # A single road can touch multiple hexagons — that's how we detect edges.
    logger.info("Step 1: Spatial join — roads to hexagons (this is the slow part)...")
    hexes_for_join = hexes_proj.reset_index()[["region_id", "geometry"]]
    joined = gpd.sjoin(
        roads_proj[["highway", "geometry"]],
        hexes_for_join,
        how="inner",
        predicate="intersects",
    )
    logger.info(f"  {len(joined):,} road-hex assignments")

    # --- Save result ---
    # We only need road_idx, region_id, highway — drop geometry to save space.
    result_df = pd.DataFrame({
        "road_idx": joined.index,
        "region_id": joined["region_id"].values,
        "highway": joined["highway"].values,
    })
    out_path = inter_dir / f"{_prefix(mode, resolution)}step1_road_hex_assignments.parquet"
    result_df.to_parquet(out_path, index=False)
    logger.info(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return result_df


# ===================================================================
# STEP 2: Build edges from shared roads, compute travel times
# ===================================================================
#
# Reads step1 output. For each road that appears in 2+ hexagons,
# creates an edge between those hex pairs. The fastest road class
# among shared roads determines the edge speed.
#
# Edge weight = centroid_distance / speed = travel time in seconds.
# Edges are symmetric: (A,B) also produces (B,A).
#
# Output: step2_raw_edges.parquet
#   Columns: origin_hex, dest_hex, travel_time_s, fastest_road_class, mode
# ===================================================================

def step_edges(
    study_area: str,
    resolution: int,
    mode: str = "walk",
) -> pd.DataFrame:
    """Step 2: Build edges from shared roads.

    Reads step1_road_hex_assignments.parquet and step1_hex_centroids.parquet.
    For each road touching 2+ hexagons, creates edges between those hex pairs.

    Returns DataFrame of raw edges with travel times.
    Saves to step2_raw_edges.parquet.
    """
    paths = StudyAreaPaths(study_area)
    inter_dir = _intermediates_dir(paths)

    # --- Load step 1 outputs ---
    assignments_path = inter_dir / f"{_prefix(mode, resolution)}step1_road_hex_assignments.parquet"
    centroids_path = inter_dir / f"{_prefix(mode, resolution)}step1_hex_centroids.parquet"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Run step 1 first. Missing: {assignments_path}")
    if not centroids_path.exists():
        raise FileNotFoundError(f"Run step 1 first. Missing: {centroids_path}")

    logger.info("Step 2: Loading step 1 outputs...")
    assignments_df = pd.read_parquet(assignments_path)
    centroids_df = pd.read_parquet(centroids_path)
    logger.info(f"  {len(assignments_df):,} road-hex assignments, {len(centroids_df):,} centroids")

    # --- Filter assignments by mode ---
    # Remove road types excluded for this mode (e.g., footways for drive mode)
    excluded = MODE_EXCLUSIONS[mode]
    if excluded:
        before = len(assignments_df)
        assignments_df = assignments_df[~assignments_df["highway"].isin(excluded)]
        logger.info(f"  Mode '{mode}': filtered to {len(assignments_df):,} assignments (removed {before - len(assignments_df):,})")

    # Build centroid lookup dicts (hex_id -> x, hex_id -> y)
    cx = dict(zip(centroids_df["region_id"], centroids_df["cx"]))
    cy = dict(zip(centroids_df["region_id"], centroids_df["cy"]))
    hex_id_set = set(centroids_df["region_id"])

    # --- Group assignments by road ---
    # road_hex_map: road_idx -> [(region_id, highway), ...]
    # A road in 2+ hexagons = potential edge between those hexagons.
    logger.info("Step 2: Grouping roads by hexagon...")
    road_hex_map: dict[int, list[tuple[str, str]]] = {}
    for road_idx, region_id, highway in zip(
        assignments_df["road_idx"], assignments_df["region_id"], assignments_df["highway"]
    ):
        road_hex_map.setdefault(road_idx, []).append((region_id, highway))

    # --- Find hex pairs connected by shared roads ---
    # For each road in 2+ hexes, generate all hex pairs.
    # Track the fastest road class per pair (highest speed wins).
    logger.info("Step 2: Building edges from shared roads...")
    pair_best_speed: dict[tuple[str, str], tuple[float, str]] = {}

    for road_idx, hex_entries in tqdm(
        road_hex_map.items(), desc="Building edges", unit="road"
    ):
        # Skip roads that only touch one hexagon — no edge to create
        if len(hex_entries) < 2:
            continue

        # Deduplicate: a road can intersect a hex via multiple sub-geometries.
        # Keep the fastest road class per hex for this road.
        hex_ids_for_road: dict[str, tuple[float, str]] = {}
        for region_id, highway in hex_entries:
            speed = ROAD_SPEEDS.get(highway, 1.4)  # default to walking speed
            if region_id not in hex_ids_for_road or speed > hex_ids_for_road[region_id][0]:
                hex_ids_for_road[region_id] = (speed, highway)

        hex_list = list(hex_ids_for_road.keys())

        # Generate all pairs of hexagons this road connects.
        # NOTE: we do NOT filter by H3 adjacency — a highway crossing
        # non-adjacent hexes creates a valid long-range edge.
        for i in range(len(hex_list)):
            for j in range(i + 1, len(hex_list)):
                h_a, h_b = hex_list[i], hex_list[j]

                # Take the fastest road class between the two hexes
                speed_a, cls_a = hex_ids_for_road[h_a]
                speed_b, cls_b = hex_ids_for_road[h_b]
                best_speed = max(speed_a, speed_b)
                best_cls = cls_a if speed_a >= speed_b else cls_b

                # Canonical pair key (sorted) to avoid duplicates
                pair_key = (min(h_a, h_b), max(h_a, h_b))
                if pair_key not in pair_best_speed or best_speed > pair_best_speed[pair_key][0]:
                    pair_best_speed[pair_key] = (best_speed, best_cls)

    logger.info(f"  {len(pair_best_speed):,} unique hex pairs with shared roads")

    # --- Compute travel times ---
    # travel_time = centroid_distance(A, B) / speed
    # Symmetric: A->B and B->A get the same travel time.
    logger.info("Step 2: Computing travel times...")
    edge_records = []
    for (h_a, h_b), (speed, road_cls) in tqdm(
        pair_best_speed.items(), desc="Travel times", unit="pair"
    ):
        if h_a not in hex_id_set or h_b not in hex_id_set:
            continue

        # Euclidean distance between hex centroids (EPSG:28992, metres)
        dx = cx[h_a] - cx[h_b]
        dy = cy[h_a] - cy[h_b]
        dist_m = np.sqrt(dx * dx + dy * dy)

        travel_time_s = dist_m / speed if speed > 0 else float("inf")

        # Add both directions (symmetric graph)
        edge_records.append((h_a, h_b, travel_time_s, road_cls))
        edge_records.append((h_b, h_a, travel_time_s, road_cls))

    edges_df = pd.DataFrame(
        edge_records,
        columns=["origin_hex", "dest_hex", "travel_time_s", "fastest_road_class"],
    )
    edges_df["mode"] = mode
    logger.info(f"  {len(edges_df):,} directed edges (symmetric)")

    # --- Save ---
    out_path = inter_dir / f"{_prefix(mode, resolution)}step2_raw_edges.parquet"
    edges_df.to_parquet(out_path, index=False)
    logger.info(f"  Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    return edges_df


# ===================================================================
# STEP 3: RUDIFUN overlay — compute FSI per H3 hexagon
# ===================================================================
#
# Loads RUDIFUN 2024 bouwblok (building block) polygons with FSI_24,
# overlays them against H3 hex polygons, and computes area-weighted
# mean FSI per hexagon.
#
# This step is independent of steps 1-2. It only needs the hex polygons.
# You can run it in parallel with step 2 if you want.
#
# Output: step3_fsi_per_hex.parquet
#   Columns: region_id (str), FSI_24 (float)
# ===================================================================

def step_rudifun(
    study_area: str,
    resolution: int,
    rudifun_layer: str = "nl2_00_Basis_Bouwblok",
) -> Optional[pd.DataFrame]:
    """Step 3: Overlay RUDIFUN building density onto H3 hexagons.

    Loads RUDIFUN GDB, overlays bouwblok polygons against hex polygons,
    computes area-weighted mean FSI_24 per hexagon.

    Returns DataFrame with (region_id, FSI_24), or None if RUDIFUN not found.
    Saves to step3_fsi_per_hex.parquet.
    """
    paths = StudyAreaPaths(study_area)
    inter_dir = _intermediates_dir(paths)

    # --- Check RUDIFUN GDB exists ---
    rudifun_gdb_path = paths.accessibility() / "Rudifun_2024_nl.gdb"
    if not rudifun_gdb_path.exists():
        logger.warning(f"  RUDIFUN GDB not found at {rudifun_gdb_path}. Skipping.")
        return None

    # --- Load RUDIFUN bouwblok ---
    # nl2_00_Basis_Bouwblok has ~300K+ building block polygons for all of NL.
    # Each has FSI_24 (Floor Space Index, 2024 vintage).
    logger.info(f"Step 3: Loading RUDIFUN layer '{rudifun_layer}'...")
    rudifun_gdf = gpd.read_file(rudifun_gdb_path, layer=rudifun_layer)
    logger.info(f"  {len(rudifun_gdf):,} bouwblok features loaded")

    if rudifun_gdf.crs is None or rudifun_gdf.crs.to_epsg() != 28992:
        rudifun_gdf = rudifun_gdf.to_crs(epsg=28992)

    # Keep only what we need — FSI_24 and geometry. Drop rows with no FSI.
    rudifun_gdf = rudifun_gdf[["FSI_24", "geometry"]].dropna(subset=["FSI_24"])
    logger.info(f"  {len(rudifun_gdf):,} features with valid FSI_24")

    # --- Load hex polygons ---
    logger.info("Step 3: Loading hex polygons...")
    region_path = paths.region_file(resolution)
    hexes_gdf = gpd.read_parquet(region_path)
    if hexes_gdf.index.name != "region_id":
        hexes_gdf.index.name = "region_id"
    hexes_proj = hexes_gdf.to_crs(epsg=28992)
    hex_reset = hexes_proj.reset_index()[["region_id", "geometry"]]

    # --- Area-weighted overlay ---
    # gpd.overlay(bouwblok, hexes, how='intersection') clips each bouwblok
    # polygon to each hex polygon it overlaps. We then compute:
    #   FSI_per_hex = sum(FSI_bouwblok * clipped_area) / sum(clipped_area)
    #
    # Chunked to avoid memory explosion (bouwblok × hex = huge).
    logger.info("Step 3: Area-weighted overlay (chunked, this takes a while)...")
    chunk_size = 50_000
    overlay_parts = []
    n_chunks = (len(rudifun_gdf) + chunk_size - 1) // chunk_size

    for i in tqdm(range(n_chunks), desc="RUDIFUN overlay", unit="chunk"):
        chunk = rudifun_gdf.iloc[i * chunk_size : (i + 1) * chunk_size]
        part = gpd.overlay(chunk, hex_reset, how="intersection")
        if len(part) > 0:
            overlay_parts.append(part)

    if not overlay_parts:
        logger.warning("  No overlay results. RUDIFUN may not overlap study area.")
        return None

    overlay = pd.concat(overlay_parts, ignore_index=True)

    # --- Aggregate: area-weighted mean FSI per hexagon ---
    overlay["intersection_area"] = overlay.geometry.area
    overlay["weighted_fsi"] = overlay["FSI_24"] * overlay["intersection_area"]

    fsi_agg = overlay.groupby("region_id").agg(
        total_weighted_fsi=("weighted_fsi", "sum"),
        total_area=("intersection_area", "sum"),
    )
    fsi_agg["FSI_24"] = fsi_agg["total_weighted_fsi"] / fsi_agg["total_area"]

    result_df = fsi_agg[["FSI_24"]].reset_index()
    logger.info(
        f"  FSI computed for {len(result_df):,} hexagons "
        f"(mean={result_df['FSI_24'].mean():.3f}, max={result_df['FSI_24'].max():.3f})"
    )

    # --- Save ---
    out_path = inter_dir / f"res{resolution}_step3_fsi_per_hex.parquet"
    result_df.to_parquet(out_path, index=False)
    logger.info(f"  Saved to {out_path}")

    return result_df


# ===================================================================
# STEP 4: Combine edges + FSI into gravity-weighted final output
# ===================================================================
#
# Reads step2 (raw edges) and step3 (FSI per hex).
# Applies gravity formula: weight = FSI_dest * exp(-beta * travel_time)
# If step3 output doesn't exist, falls back to pure distance decay.
#
# Output: {mode}_res{resolution}.parquet (final output)
#   Columns: origin_hex, dest_hex, travel_time_s, gravity_weight, mode, fastest_road_class
# ===================================================================

def step_gravity(
    study_area: str,
    resolution: int,
    mode: str = "walk",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Step 4: Apply gravity weighting and produce final output.

    Reads step2_raw_edges.parquet and step3_fsi_per_hex.parquet.
    Applies gravity formula and saves the final accessibility graph.

    Returns the final DataFrame.
    """
    paths = StudyAreaPaths(study_area)
    inter_dir = _intermediates_dir(paths)

    # --- Load step 2 edges ---
    edges_path = inter_dir / f"{_prefix(mode, resolution)}step2_raw_edges.parquet"
    if not edges_path.exists():
        raise FileNotFoundError(f"Run step 2 first. Missing: {edges_path}")

    logger.info("Step 4: Loading raw edges...")
    edges_df = pd.read_parquet(edges_path)
    logger.info(f"  {len(edges_df):,} edges loaded")

    # --- Load step 3 FSI (optional) ---
    fsi_path = inter_dir / f"res{resolution}_step3_fsi_per_hex.parquet"
    fsi_per_hex: Optional[pd.Series] = None

    if fsi_path.exists():
        logger.info("Step 4: Loading FSI per hexagon...")
        fsi_df = pd.read_parquet(fsi_path)
        fsi_per_hex = fsi_df.set_index("region_id")["FSI_24"]
        logger.info(f"  FSI loaded for {len(fsi_per_hex):,} hexagons")
    else:
        logger.warning(
            f"  step3_fsi_per_hex.parquet not found. "
            f"Using pure distance decay (no gravity weighting)."
        )

    # --- Apply gravity formula ---
    # gravity_weight = FSI_destination * exp(-beta * travel_time_seconds)
    # If no FSI data, just use exp(-beta * travel_time) (distance decay only).
    beta = GRAVITY_BETA[mode]
    logger.info(f"Step 4: Applying gravity formula (beta={beta}, mode={mode})...")

    if fsi_per_hex is not None:
        dest_fsi = edges_df["dest_hex"].map(fsi_per_hex).fillna(0.0)
        edges_df["gravity_weight"] = dest_fsi * np.exp(-beta * edges_df["travel_time_s"])
    else:
        edges_df["gravity_weight"] = np.exp(-beta * edges_df["travel_time_s"])

    # --- Save final output ---
    if output_path is None:
        out_dir = paths.accessibility()
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{mode}_res{resolution}.parquet"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    edges_df = edges_df[
        ["origin_hex", "dest_hex", "travel_time_s", "gravity_weight", "mode", "fastest_road_class"]
    ]
    edges_df.to_parquet(output_path, index=False)
    logger.info(
        f"  DONE! Saved {len(edges_df):,} edges to {output_path} "
        f"({output_path.stat().st_size / 1e6:.1f} MB)"
    )

    return edges_df


# ===================================================================
# Convenience: run all steps
# ===================================================================

def build_accessibility_graph(
    study_area: str,
    resolution: int,
    mode: str = "walk",
    rudifun_layer: str = "nl2_00_Basis_Bouwblok",
    osm_date: str = "latest",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run the full 4-step pipeline.

    Convenience wrapper that calls step_sjoin -> step_edges -> step_rudifun -> step_gravity.
    Each step saves intermediates, so if it crashes partway you can resume from the last step.
    """
    if mode not in MODE_EXCLUSIONS:
        raise ValueError(f"Unknown mode {mode!r}. Choose from: {list(MODE_EXCLUSIONS)}")

    step_sjoin(study_area, resolution, mode, osm_date)
    step_edges(study_area, resolution, mode)
    step_rudifun(study_area, resolution, rudifun_layer)
    return step_gravity(study_area, resolution, mode, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build accessibility graph from OSM roads + RUDIFUN density.\n\n"
        "Run all steps:  --step all (default)\n"
        "Run one step:   --step 1  (or 2, 3, 4)\n"
        "Steps 1-2 are sequential. Step 3 is independent. Step 4 needs 2+3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument("--resolution", type=int, default=9)
    parser.add_argument("--mode", default="walk", choices=["walk", "bike", "drive"])
    parser.add_argument("--rudifun-layer", default="nl2_00_Basis_Bouwblok")
    parser.add_argument("--osm-date", default="latest")
    parser.add_argument("--output", default=None, help="Output parquet path (step 4 only)")
    parser.add_argument(
        "--step", default="all",
        help="Which step to run: 1, 2, 3, 4, or 'all' (default: all)",
    )

    args = parser.parse_args()

    if args.step == "all":
        build_accessibility_graph(
            study_area=args.study_area,
            resolution=args.resolution,
            mode=args.mode,
            rudifun_layer=args.rudifun_layer,
            osm_date=args.osm_date,
            output_path=Path(args.output) if args.output else None,
        )
    elif args.step == "1":
        step_sjoin(args.study_area, args.resolution, args.mode, args.osm_date)
    elif args.step == "2":
        step_edges(args.study_area, args.resolution, args.mode)
    elif args.step == "3":
        step_rudifun(args.study_area, args.resolution, args.rudifun_layer)
    elif args.step == "4":
        step_gravity(
            args.study_area, args.resolution, args.mode,
            Path(args.output) if args.output else None,
        )
    else:
        parser.error(f"Unknown step: {args.step}. Use 1, 2, 3, 4, or 'all'.")
