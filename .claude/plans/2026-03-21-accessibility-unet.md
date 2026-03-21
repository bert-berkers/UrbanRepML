# AccessibilityUNet Pipeline

**Goal**: Build the accessibility graph from OSM road data and train FullAreaUNet on real weighted graphs. Beat ring agg R²=0.556.

**Strategic context**: The UNet has been training on uniform 1-ring lattice (geometry only). Ring agg wins because geometric distance is a decent proxy for accessibility — but it's the wrong distance. The AccessibilityUNet uses real travel-time weighted graphs, giving message-passing something meaningful to propagate. This is the original vision from the active inference symposium.

## Wave 1: Understand current state (parallel)

1. **stage2-fusion-architect**: Read the legacy `stage2_fusion/graphs/graph_construction.py` (SpatialGraphConstructor) — extract the Dijkstra + gravity model logic, road mode speeds, FSI weighting. Also read `stage2_fusion/models/full_area_unet.py` to confirm exactly how it consumes `edge_indices` and `edge_weights`. Report: what's reusable, what needs rewriting, what the UNet expects as graph input tensors.

2. **srai-spatial**: Read `utils/spatial_db.py` (SpatialDB) and determine how to do a bulk spatial join of OSM road segments against H3 hex boundaries at res9. The goal: for each pair of adjacent hexagons, find the fastest OSM road class crossing the boundary. Report: which SpatialDB methods to use, what the query looks like, expected performance.

3. **Explore**: Find where FSI / building density data lives in this codebase. Check `data/study_areas/netherlands/` for any FSI, building density, or population data. Also check `scripts/archive/preprocessing/setup_fsi_filter.py` for how FSI was previously computed. Report: what mass data exists and where.

## Wave 2: Build the lattice weight generator (sequential after Wave 1)

4. **stage2-fusion-architect**: Build `stage2_fusion/graphs/accessibility_graph.py` — the "cheat" approach:
   - Input: OSM road network (from PBF or pre-loaded), H3 regions_gdf at target resolution
   - For each pair of adjacent hexagons: SpatialDB spatial join to find road segments crossing the boundary
   - Take the fastest road class → lookup speed (walk: 5 km/h, cycle: 15 km/h, drive: 50 km/h or road-class specific)
   - Edge weight = centroid distance / speed = travel time
   - Output: Parquet with `(origin_region_id, dest_region_id, travel_time_seconds, mode)`
   - Per-mode filtering: walking only uses pedestrian-compatible roads, cycling uses cyclable, driving uses all
   - Use SpatialDB for the spatial join, SRAI H3Neighbourhood for adjacency

## Wave 3: Integrate with UNet data loader

5. **stage2-fusion-architect**: Modify `stage2_fusion/data/multi_resolution_loader.py`:
   - Add option to load accessibility graph from Parquet instead of building uniform 1-ring
   - Convert `(origin_region_id, dest_region_id, travel_time_seconds)` to `edge_indices` + `edge_weights` tensors
   - Fallback to uniform 1-ring if no accessibility graph exists (backward compatible)
   - Handle multi-resolution: if accessibility graph exists for res9, derive coarser resolutions by aggregation

## Wave 4: Generate accessibility graph for Netherlands

6. **execution**: Run the accessibility graph generator for Netherlands at res9 (walking mode first as proof of concept). Use existing OSM PBF data from `data/study_areas/netherlands/stage1_unimodal/osm/`. Save output to `data/study_areas/netherlands/accessibility/walking_res9.parquet`.

## Wave 5: Train and probe

7. **execution**: Train FullAreaUNet with accessibility graph (1000 epochs, patience=100, LR=0.001). Save checkpoint with versioning.

8. **stage3-analyst**: Probe the AccessibilityUNet embeddings against leefbaarometer. Write results to `probe_results/unet_accessibility/results.parquet` using the standardized format (if Terminal B's writer is available, otherwise raw format and convert later).

## Wave 6: Verification

9. **qaqc**: Verify the accessibility graph (edge count, weight distribution, connectivity), UNet training convergence, and probe results. Compare R² against ring agg baseline (0.556).

## Final Wave: Close-out

- Coordinator scratchpad
- `/librarian-update`
- `/ego-check`

## Execution
Invoke: `/niche .claude/plans/2026-03-21-accessibility-unet.md`
