# AccessibilityUNet Pipeline

**Goal**: Build the accessibility graph from OSM road data and train FullAreaUNet on real weighted graphs. Beat ring agg R²=0.556.

**Strategic context**: The UNet has been training on uniform 1-ring lattice (geometry only). Ring agg wins because geometric distance is a decent proxy for accessibility — but it's the wrong distance. The AccessibilityUNet uses real travel-time weighted graphs, giving message-passing something meaningful to propagate. This is the original vision from the active inference symposium.

---

## Wave 1: Understand current state (parallel) — DONE

### Findings
- **UNet interface**: `edge_indices: Dict[int, Tensor[2,E]]`, `edge_weights: Dict[int, Tensor[E]]` per resolution
- **SAGEConv bug**: `conv(out, edge_index)` never passes `edge_weight` — two-line fix needed in EncoderBlock + DecoderBlock
- **SpatialDB**: geometry retrieval only, no spatial join. Use `OSMPbfLoader` (reuses roads processor) + `gpd.sjoin`
- **Road-in-both-hexes approach**: assign each road segment to all hexes it intersects, then for adjacent hex pairs check which roads appear in both → fastest road class = edge speed. Simpler than shared-boundary geometry.
- **Density data**: RUDIFUN 2024 GDB at `data/study_areas/netherlands/accessibility/Rudifun_2024_nl.gdb`, layer `nl2_00_Basis_Bouwblok` — FSI_24 per building block polygon, Netherlands-wide, EPSG:28992. Spatial join to H3 res9 for per-hex gravity mass.
- **Legacy speeds**: walk 1.4 m/s, bike 4.17 m/s, drive 11.11 m/s. Beta: walk 0.002, bike 0.0012, drive 0.0008. Mode-per-resolution: {8: drive, 9: bike, 10: walk}.

### Gate 1: PASSED
All three sub-questions answered. Proceeding with gravity weighting (RUDIFUN bouwblok FSI), road-in-both-hexes spatial join, and SAGEConv fix.

---

## Wave 2: Build the accessibility graph generator (parallel)

4a. **stage2-fusion-architect**: Build `stage2_fusion/graphs/accessibility_graph.py`:
   - Input: OSM PBF path, H3 regions_gdf, RUDIFUN GDB path, target resolution
   - Step 1: Load roads via `OSMPbfLoader` (reuse roads processor pattern), get GeoDataFrame of LineStrings with `highway` column
   - Step 2: Assign roads to hexes via `gpd.sjoin(roads, hex_polygons, predicate='intersects')` — a road in two adjacent hexes = edge
   - Step 3: For each adjacent hex pair with shared roads, take fastest road class → lookup speed → edge travel_time = centroid_distance / speed
   - Step 4: Load RUDIFUN bouwblok (`nl2_00_Basis_Bouwblok`), spatial join to H3 res9, aggregate FSI_24 per hex (area-weighted mean)
   - Step 5: Gravity weight = `FSI_destination * exp(-beta * travel_time)`
   - Output: Parquet with `(origin_hex, dest_hex, travel_time_s, gravity_weight, mode)`
   - Tile by res5 parent for memory (~200 tiles, ~13K hexes each)

4b. **stage2-fusion-architect**: Fix SAGEConv in `full_area_unet.py` — pass `edge_weight` to conv calls in EncoderBlock and DecoderBlock (two-line fix).

### Gate 2: Code review
> **Go/no-go**: Does the module structure look right? Does the SAGEConv fix compile?
> - If spatial join approach is too slow at Netherlands scale → simplify: drop road-type lookup, use uniform centroid distance with gravity only.

---

## Wave 3: Integrate with UNet data loader

5. **stage2-fusion-architect**: Modify `stage2_fusion/data/multi_resolution_loader.py`:
   - Add option to load accessibility graph from Parquet instead of building uniform 1-ring
   - Convert `(origin_region_id, dest_region_id, travel_time_seconds)` to `edge_indices` + `edge_weights` tensors
   - Fallback to uniform 1-ring if no accessibility graph exists (backward compatible)
   - Handle multi-resolution: if accessibility graph exists for res9, derive coarser resolutions by aggregation

### Gate 3: Integration check
> **Go/no-go**: Does the loader produce valid PyG Data objects with the accessibility weights? Quick smoke test.

---

## Wave 4: Generate accessibility graph for Netherlands

6. **execution**: Run the accessibility graph generator for Netherlands at res9 (walking mode first as proof of concept). Use existing OSM PBF data from `data/study_areas/netherlands/stage1_unimodal/osm/`. Save output to `data/study_areas/netherlands/accessibility/walking_res9.parquet`.

### Gate 4: Generation succeeded?
> **Go/no-go**: Did it complete? Is the output file non-empty? Quick row count + file size check.

---

## Wave 5: Network analysis and visualization — PARTLY DONE

### 5a. Stats analysis — DONE
Basic stats computed, analysis saved to `data/study_areas/netherlands/accessibility/accessibility_graph_analysis.md`.
Walk: 2.3M edges, 325K nodes, 188 components (98.7% in largest). Bike: 335K edges, 57K nodes. Drive: 49K edges, 8.8K nodes.

### 5b. Visualization — REDO IN FRESH CONTEXT

Save all to `data/study_areas/netherlands/accessibility/figures/`.

#### Two rendering modes

**1. Stamping (full NL overview only):**
- Use `utils/visualization.py` (`rasterize_continuous`, `plot_spatial_map`, etc.)
- Full `RASTER_W=2000, RASTER_H=2400` per panel — NEVER divide by subplot count
- `stamp = max(1, 11 - h3_resolution)` → res9=2, res8=3, res7=4
- White background — NO grey hex underlay, NO boundary outline
- Just show the data points on white
- The grid (full raster resolution per panel) is the key insight, not just stamp size

**2. Hex polygons with dissolve (zoomed/city-scale views):**
- For smaller areas, render actual hexagon polygons — stamping can't show hex shapes at zoom
- Use `h3.cells_to_geo()` or load from regions parquet, dissolve adjacent same-value hexes to reduce geometry count
- OSM basemap via `contextily.add_basemap(ax, crs='EPSG:28992', source=contextily.providers.CartoDB.Positron)`
- This gives proper hex shapes + street context

#### Plot group 1: Overview maps (full NL, stamping) — DONE, minor redo

6 maps exist but need: white background (remove grey underlay), no boundary outline, just data on white.
Keep stamp sizing as-is. Keep colorbars. Remove the grey.

For each of walk_res9, bike_res8, drive_res7:
- `{mode}_res{r}_degree.png` — degree per hex, viridis
- `{mode}_res{r}_gravity.png` — mean incoming gravity weight, magma

#### Plot group 2: Speed/graph network maps (zoomed, hex polygons + OSM basemap) — NEW

Pick 4 cities (Amsterdam, Rotterdam, Den Haag, Eindhoven). For each, auto-fit zoom to local graph:
- Show graph edges as lines colored by speed (fast=red, slow=blue) via `LineCollection`
- Show hex nodes as actual hex polygons colored by gravity weight
- OSM basemap underneath via contextily
- Goal: visually confirm edges follow real roads and speeds make sense

Files: `network_amsterdam.png`, `network_rotterdam.png`, `network_denhaag.png`, `network_eindhoven.png`

#### Plot group 3: Gravity floodfill maps (zoomed, hex polygons + OSM basemap) — NEW

**Origins** (5 cities + 2 rural):
- Amsterdam Centraal (~121000, 487000 RD)
- Rotterdam Centraal (~92000, 437000 RD)
- Den Haag Centraal (~81000, 454000 RD)
- Utrecht Centraal (~136000, 456000 RD)
- Eindhoven Centraal (~162000, 383000 RD)
- Rural Drenthe (~240000, 545000 RD)
- Rural Zeeland (~45000, 390000 RD)

For each origin, show **gravity weight** of all edges FROM that origin:
```python
origin_edges = edges_df[edges_df['origin_hex'] == origin]
# dest hexes colored by gravity_weight, origin hex highlighted
```

This shows "how much does each neighbor attract from this origin" — the direct gravity signal.

**Rendering**: hex polygons (not stamps) + OSM basemap via contextily. Auto-fit zoom to extent of reachable hexes (±padding).

**Layout**:
- 4×2 subplot grid (7 origins + 1 legend), full raster per panel
- Individual zoomed maps for Amsterdam and Eindhoven

Files: `gravity_floodfill_grid.png`, `gravity_amsterdam.png`, `gravity_eindhoven.png`

#### Plot group 4: Histograms — DONE ✓

All 4 histogram figures acceptable as-is:
- `hist_travel_times.png` — good, multimodal overlay
- `hist_gravity_weights.png`
- `hist_degree.png`
- `bar_road_classes.png` — great cross-modal comparison

#### Plot group 5: Cross-modal (future/stretch)

Comparing walk/bike/drive requires parent-child resolution mapping (res9→res8→res7). Defer.

#### Delegation plan for 5b (fresh context)

**Wave 5b-redo** (parallel):
- **stage3-analyst (A)**: Plot group 1 redo (quick — just remove grey background + boundary, keep everything else) + Plot group 2 (network maps, new)
- **stage3-analyst (B)**: Plot group 3 (gravity floodfill maps with hex polygons + contextily)

### 5b Results — DONE

Visualization script: `scripts/one_off/accessibility_viz_all.py`
Output: `data/study_areas/netherlands/accessibility/figures/`

**Rendering approach**: SRAI hex polygons (`regions_gdf.plot()`) for NL-wide overview maps — no stamps, no raster math. Hex polygons + contextily for zoomed city views. `edgecolor="none"` everywhere.

**20 figures total**:
- Group 1: 6 overview maps (degree + gravity × walk/bike/drive) — hex polygon fill, no borders
- Group 2: 4 network maps (Amsterdam/Rotterdam/Den Haag/Eindhoven) — edges with black outline for contrast
- Group 4: 3 isochrone maps (7-city grid + Amsterdam + Eindhoven individuals) — hex polygons + OSM basemap
- Group 5: 3 lattice comparison (degree-6 deviation for all modes) — RdBu_r hex fill
- Histograms: 4 kept from earlier

### Gate 5b: Do the maps make physical sense? — FAILED

**Critical finding**: Degree maps show too-high accessibility in:
1. **Forests** (Veluwe, Biesbosch) — OSM includes `highway=track`, `path`, `footway` in forests
2. **Highways** — motorway/trunk edges shouldn't be walkable
3. **Industrial areas** — dense internal road networks inflate degree

Gravity maps are better (forests are dark = low FSI) but degree is wrong. The isochrone maps confirm: Drenthe rural isochrone spreads along forest tracks unrealistically.

**Conclusion**: The accessibility graph generator needs road-type filtering before we can train on it. Wave 6 is BLOCKED until this is fixed.

---

## Wave 5c: Fix accessibility graph edge selection — DONE

Two fixes applied and validated:

### Road type filtering per mode

**Walk** (res9): Include only roads where walking makes physical sense
- Include: `residential`, `living_street`, `pedestrian`, `footway`, `path`, `steps`, `service`, `unclassified`, `tertiary`, `secondary`, `primary`
- Exclude: `motorway`, `motorway_link`, `trunk`, `trunk_link`, `track` (forest/field), `cycleway` (walk on cycleway debatable)
- Key insight: `track` is the forest/agricultural culprit — removing it should clean up Veluwe/forests

**Bike** (res8): Include roads with cycling infrastructure
- Include: `cycleway`, `residential`, `living_street`, `service`, `unclassified`, `tertiary`, `secondary`, `primary`, `path` (when bicycle=yes)
- Exclude: `motorway`, `motorway_link`, `trunk`, `trunk_link`, `track`, `footway`, `steps`

**Drive** (res7): Motorized vehicles only
- Include: `motorway`, `motorway_link`, `trunk`, `trunk_link`, `primary`, `secondary`, `tertiary`, `unclassified`, `residential`, `service`
- Exclude: `footway`, `path`, `steps`, `cycleway`, `pedestrian`, `track`, `living_street`

### Tasks

1. **stage2-fusion-architect**: Read `stage2_fusion/graphs/accessibility_graph.py`, understand current road type handling, add mode-specific road type filters
2. **execution**: Regenerate all 3 parquets: `walk_res9.parquet`, `bike_res8.parquet`, `drive_res7.parquet`
3. **execution**: Re-run `scripts/one_off/accessibility_viz_all.py` to regenerate all plots
4. **stage3-analyst**: Visual QA — do forests/highways/industrial now show lower degree?

### Gate 5c: PASSED
> Lattice deviation maps exclusively blue/white, zero red hotspots. Veluwe clean, causeways clean, coastal clean.
> Walk: 2.3M → 1.0M edges (-57%). Bike: 295K → 241K (-18%). Drive: 45K → 43K (-3%).
> Spec: `specs/accessibility_graph_pipeline.md`. Figures: `accessibility/figures/adjacency_filtered/`.
> Old results preserved in `accessibility/pre_adjacency_filter/`.

---

## Wave 6: Train and probe (sequential) — DONE

Executed across OODA 1+2 on 2026-03-22. Evolved beyond original scope: supervised decoder head with Kendall uncertainty weighting (not just accessibility graph training).

**Results**: Supervised multiscale R²=0.574 beats ring_agg 0.535. Per-target: lbm 0.562, vrz 0.826 (best), won 0.483 (raw_concat still wins at 0.618). Full comparison: `data/study_areas/netherlands/stage3_analysis/comparison/2026-03-22/summary.md`.

### Gate 6: PASSED (with nuance)
> R² 0.574 > 0.556 target. But no single approach dominates all targets — local-vs-spatial tradeoff is real. Next: gated architecture to learn when to smooth vs preserve.

---

## Wave 7: Verification — SKIPPED

Verification folded into OODA 2's inline QAQC (14/14 checks PASS in OODA 1, extraction bugs caught and fixed in OODA 2). Full verification not separately executed — the supervised decoder head plan (`2026-03-22-supervised-decoder-head.md`) superseded this wave.

---

## Final Wave: Close-out — DONE 2026-03-22

Executed as OODA 3/3 weekend close-off.

## Status: COMPLETE

This plan is finished. The accessibility UNet pipeline (Waves 1-6) delivered:
- Accessibility graph generator with mode-specific road filtering
- FullAreaUNet with SAGEConv edge_weight support
- Supervised decoder head with Kendall uncertainty weighting
- Mean R²=0.574 (vs ring_agg 0.535 baseline)
- Full comparison suite with per-target analysis

**Next direction**: Gated architecture to address local-vs-spatial tradeoff. See ego forward-look 2026-03-22.
