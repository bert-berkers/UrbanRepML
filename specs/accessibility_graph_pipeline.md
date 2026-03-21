# Accessibility Graph Pipeline

## Status: Implemented

## Context

The FullAreaUNet's message-passing operates on graph edges. The original implementation used a uniform H3 1-ring lattice: every hexagon connects to its 6 geometric neighbors with equal weight. This is a topological prior that says "all directions matter equally, all neighbors are equally reachable." In reality, urban connectivity is shaped by road networks and land use intensity. A hexagon bordering a canal has no road connection to its geometric neighbor across the water. A hexagon adjacent to a dense commercial district should receive stronger signal than one adjacent to empty farmland.

The accessibility graph replaces the uniform lattice with a weighted graph derived from two real-world data sources:

1. **OSM road network** -- which hexagons are actually connected by roads, and how fast you can travel between them
2. **RUDIFUN building density (FSI)** -- how much built floor area exists at the destination, serving as a gravity attraction term

This gives the UNet's GCN layers something meaningful to propagate: travel-time weighted edges with gravity attraction, rather than uniform topology.

## Pipeline Architecture

The pipeline runs in 4 steps, each producing an intermediate Parquet file. Steps can be run independently (step 3 is fully parallel with steps 1-2), and the pipeline is resumable from any step.

```
Step 1: OSM PBF --> spatial join roads to H3 hexes
Step 2: Shared-road edges with H3 adjacency filter + travel time
Step 3: RUDIFUN bouwblok FSI overlay (independent of steps 1-2)
Step 4: Gravity weighting (FSI x distance decay) --> final output
```

### Step 1: Spatial Join (roads to hexagons)

Loads roads from OSM PBF via SRAI's `OSMPbfLoader`, filters by mode-specific whitelist, projects both roads and H3 hex polygons to EPSG:28992 (RD New, metric), and runs `gpd.sjoin(predicate='intersects')`. Each row in the output says "road X passes through hexagon Y with highway type Z."

Also saves hex centroids (in EPSG:28992 metres) for distance computation in step 2.

**Output**: `step1_road_hex_assignments.parquet` (road_idx, region_id, highway)

### Step 2: Edge Construction

Groups step 1 output by road. For each road touching 2+ hexagons, generates candidate edges between those hex pairs. Applies the H3 adjacency filter (see Design Decisions below) to keep only edges between geometrically adjacent hexagons. For each valid pair, selects the fastest road class among shared roads and computes travel time as centroid distance divided by speed. Edges are symmetric (A->B and B->A).

**Output**: `step2_raw_edges.parquet` (origin_hex, dest_hex, travel_time_s, fastest_road_class, mode)

### Step 3: RUDIFUN Overlay

Loads RUDIFUN 2024 bouwblok (building block) polygons from GDB, overlays them against H3 hex polygons using `gpd.overlay(how='intersection')`, and computes area-weighted mean FSI per hexagon. Chunked (50K features per chunk) to avoid memory explosion.

This step is independent of steps 1-2 and can run in parallel. FSI is resolution-specific but mode-independent: the same FSI values apply to walk, bike, and drive at a given resolution.

**Output**: `step3_fsi_per_hex.parquet` (region_id, FSI_24)

### Step 4: Gravity Weighting

Combines step 2 edges with step 3 FSI. Applies the gravity formula:

```
gravity_weight = FSI_destination * exp(-beta * travel_time_seconds)
```

Where beta varies by mode: walk=0.002, bike=0.0012, drive=0.0008. Higher beta means faster distance decay -- walking penalizes distance more than driving.

Falls back to pure distance decay (`exp(-beta * t)`) if RUDIFUN data is not available.

**Output**: `{mode}_res{resolution}.parquet` (origin_hex, dest_hex, travel_time_s, gravity_weight, mode, fastest_road_class)

## Design Decisions

### 1. Inclusion-based road type filtering (whitelist over blacklist)

Each travel mode has a whitelist of included OSM highway types. The alternative -- an exclusion list -- was tried first and failed: it missed too many inappropriate types. Forest tracks (`track`) inflated connectivity in the Veluwe, motorways appeared in the walk graph, and every new edge type OSM introduced would silently leak through.

| Mode | Includes | Excludes (by omission) |
|------|----------|----------------------|
| Walk (res9) | residential, footway, path, pedestrian, steps, service, unclassified, tertiary through primary | motorway/trunk (and links), track, cycleway |
| Bike (res8) | cycleway, residential, path, service, unclassified, tertiary through primary | motorway/trunk (and links), track, footway, steps |
| Drive (res7) | motorway through tertiary (and links), unclassified, residential, service | footway, path, pedestrian, steps, track, cycleway, living_street |

### 2. H3 adjacency filter (the critical fix)

**Problem**: A road crossing N hexagons creates all-pairs edges between those N hexagons -- O(N^2) edges per road. Long linear infrastructure (Afsluitdijk causeway, Houtribdijk, airport perimeters, coastal roads) produced massive degree inflation. Some hexagons had hundreds of edges instead of the expected 6-12.

**Solution**: Build a valid-pair set from H3 k=1 neighbours using `SpatialDB.neighbours()` (which wraps SRAI's `H3Neighbourhood`). Before creating an edge between hex A and hex B, check that `frozenset((A, B))` is in the valid-pair set. This reduces a road through hexes A->B->C->D from 6 edges (A-B, A-C, A-D, B-C, B-D, C-D) to 3 sequential edges (A-B, B-C, C-D).

**Result**: Walk graph dropped from 1.66M to 1.0M edges (~40% reduction). All lattice deviation maps became exclusively blue/white with zero hotspots.

### 3. Mode-resolution coupling

Each mode operates at a single H3 resolution matched to its travel scale:

| Mode | Resolution | Hex edge length | Rationale |
|------|-----------|----------------|-----------|
| Walk | 9 | ~175m | Walking trips are short; fine resolution captures block-level connectivity |
| Bike | 8 | ~660m | Cycling range is 3-4x walking; coarser resolution appropriate |
| Drive | 7 | ~1.8km | Driving covers large distances; coarsest resolution avoids noise |

### 4. Fastest-road-class-wins for multi-road hex pairs

When multiple roads connect the same two hexagons, the fastest road class determines the edge speed. This is a simplification -- in reality, a pedestrian won't use a secondary road at 60 km/h -- but it captures the key signal: if a fast road connects two hexagons, they are well-connected. The mode-specific whitelist already filters out inappropriate road types before this point.

### 5. Centroid distance for travel time

Travel time = Euclidean distance between hex centroids / road speed. This is an approximation (real road paths curve), but at H3 res9 (~175m edge length), centroid-to-centroid distance is a reasonable proxy for the straight-line road segment connecting two adjacent hexagons.

### 6. Graceful FSI fallback

If RUDIFUN data is not available (e.g., for a non-Netherlands study area), the pipeline falls back to pure distance decay without the FSI attraction term. This keeps the pipeline portable.

## Integration with FullAreaUNet

The accessibility graph integrates through `MultiResolutionLoader`:

1. `MultiResolutionLoader.__init__()` accepts an `accessibility_graph` parameter (path to the final Parquet)
2. `_build_adjacency_graphs()` checks: if accessibility graph is provided AND this is the finest resolution, call `_load_accessibility_edges()` instead of `_build_uniform_adjacency()`
3. `_load_accessibility_edges()` reads the Parquet, maps hex string IDs to integer indices, and returns `edge_index` (COO format) with `gravity_weight` as `edge_weight`
4. Coarser resolutions always use uniform 1-ring adjacency (edge_weight=1.0)
5. `FullAreaUNet` passes `edge_weight` through all `EncoderBlock` and `DecoderBlock` layers, which forward it to `GCNConv(edge_weight=edge_weight)`

The integration is backward-compatible: `accessibility_graph=None` (default) produces identical behavior to the pre-accessibility codebase.

**CLI**: `python scripts/stage2/train_full_area_unet.py --accessibility-graph data/study_areas/netherlands/accessibility/walk_res9.parquet`

**GCNConv note**: The model was originally SAGEConv, but SAGEConv does not support `edge_weight`. The swap to GCNConv was required specifically for accessibility graph integration.

## Data Layout

```
data/study_areas/netherlands/accessibility/
    walk_res9.parquet              # Final output: 1.0M directed edges
    bike_res8.parquet              # Final output: 241K directed edges
    drive_res7.parquet             # Final output: 43K directed edges
    Rudifun_2024_nl.gdb/           # Source: RUDIFUN building density
    figures/                       # Validation visualizations
        adjacency_filtered/        # Post-filter degree maps
    intermediates/
        walk_res9_step1_road_hex_assignments.parquet
        walk_res9_step1_hex_centroids.parquet
        walk_res9_step2_raw_edges.parquet
        res9_step3_fsi_per_hex.parquet   # Resolution-scoped, mode-independent
        bike_res8_step1_...
        bike_res8_step2_...
        res8_step3_fsi_per_hex.parquet
        drive_res7_step1_...
        drive_res7_step2_...
        res7_step3_fsi_per_hex.parquet
```

## Validation Results

Validation focused on the adjacency filter's effect on degree distribution:

- **Before filter**: Houtribdijk (causeway connecting Lelystad to Enkhuizen), Maasvlakte (port area), Afsluitdijk, Veluwe forest roads, and coastal areas all showed unrealistically high degree in lattice deviation maps (red hotspots)
- **After filter**: All deviation maps exclusively blue/white, zero hotspots
- **Edge counts** (directed, post-filter): walk 1.0M, bike 241K, drive 43K
- **Validation figures**: `data/study_areas/netherlands/accessibility/figures/adjacency_filtered/`

## Lessons Learned

1. **All-pairs was the original bug.** The first implementation created edges between every pair of hexagons sharing a road. This is correct for roads spanning 2 hexagons but catastrophically wrong for roads spanning 10+. The Afsluitdijk (a 32km causeway) produced hundreds of cross-hex edges. The fix (H3 adjacency filter) was simple in retrospect but required the degree deviation maps to diagnose.

2. **Exclusion lists fail silently.** The first mode filtering used a blacklist (exclude motorways from walk). This missed `track` (forest/agricultural roads that inflated Veluwe connectivity) and several link road types. Switching to a whitelist made the filtering explicit and self-documenting.

3. **Step 3 independence was a deliberate design choice.** FSI overlay is computationally expensive (~20 min for NL) and resolution-dependent but mode-independent. Decoupling it from steps 1-2 means you compute FSI once per resolution, then reuse it across walk/bike/drive modes at that resolution.

## Open Items

| Item | Status |
|------|--------|
| Option B: aggregate gravity weights to coarser resolutions | Deferred -- coarser resolutions use uniform 1-ring for now |
| NL-scale spatial join performance for step 1 | Works but slow (~15 min); placeholder for optimization |
| GTFS transit accessibility layer | Planned but not started; would add a 4th mode |

## CLI Reference

```bash
# Run full pipeline for one mode
python -m stage2_fusion.graphs.accessibility_graph \
    --study-area netherlands --resolution 9 --mode walk

# Run individual steps
python -m stage2_fusion.graphs.accessibility_graph \
    --study-area netherlands --resolution 9 --mode walk --step 1

# Train UNet with accessibility graph
python scripts/stage2/train_full_area_unet.py \
    --accessibility-graph data/study_areas/netherlands/accessibility/walk_res9.parquet
```
