# POI Pipeline: pyosmium Extraction + Sedona Spatial Join

## Status: Draft

## Context

The POI modality pipeline has a provenance problem. When `data_source='osm_online'` (Overpass API), OSM returns **current** data, but the pipeline stamps output filenames with `--year 2022`. This produced 13 mislabeled files on 2026-03-06 (documented in stage1-modality-encoder scratchpad). The fix requires extracting POI data from **date-specific PBF snapshots** so year labels are truthful.

We have three PBF snapshots (`2020-01-01`, `2022-01-01`, `2024-01-01`) plus a full history file (`netherlands-internal.osh.pbf`, 3.1 GB), all at `data/study_areas/netherlands/stage1_unimodal/osm/`. A pyosmium-based extractor already exists in `scripts/one_off/extract_pois_from_history.py` and was partially integrated into the processor via a stashed commit. This spec promotes that work from one-off to durable pipeline component and adds Sedona-based spatial joining.

SRAI's `OSMPbfLoader` (backed by quackosm/DuckDB) **cannot read PBF files on Windows** due to a zlib corruption error. pyosmium works reliably on all platforms.

## Decision

### Architecture

```
PBF snapshot (year-specific .osm.pbf)
  -> pyosmium extraction (POI nodes with HEX2VEC_FILTER tags)
  -> GeoParquet (intermediate, year-tagged)
  -> SRAI IntersectionJoiner (point-in-H3-hexagon spatial join)
  -> count matrix per hex (year-tagged)
  -> hex2vec / geovex / count embeddings (year-tagged)
```

### Key design decisions

**1. Extract from date-specific snapshots, not full history**

Use the pre-extracted `.osm.pbf` snapshots (e.g., `netherlands-2022-01-01.osm.pbf`) rather than the full `.osh.pbf` history file. Rationale:

- Snapshots are simpler -- single pass, no version tracking, no cutoff logic
- Snapshots already exist for all three target years (2020, 2022, 2024)
- The full-history extractor (`extract_pois_from_history.py`) is more complex and slower (must track per-node version chains)
- If a new year is needed, `osmium time-filter` produces a snapshot in ~2 minutes

The existing history extractor remains useful for ad-hoc dates but is not the primary path.

**2. Nodes only -- this is sufficient for hex2vec**

HEX2VEC_FILTER has 15 OSM keys: `aeroway, amenity, building, healthcare, historic, landuse, leisure, military, natural, office, shop, sport, tourism, water, waterway`.

The current SRAI pipeline extracts 96% ways and 4% nodes. But for hex2vec's categorical counting, the question is: **do nodes carry enough signal?**

Analysis by OSM key:
- **Primarily node-tagged** (~60% of HEX2VEC_FILTER value count): `amenity` (114 values -- restaurants, ATMs, schools), `shop` (164 values), `office` (40), `healthcare` (20), `tourism` (22), `historic` (34), `sport` (105)
- **Primarily way-tagged**: `building` (75 values), `landuse` (32), `natural` (32), `water` (14), `waterway` (18)
- **Mixed**: `leisure` (33), `aeroway` (11), `military` (11)

For the **amenity/shop/office** categories that drive urban function differentiation, nodes carry the primary signal. A restaurant is tagged on a node; the building it occupies is a way with `building=commercial`. Hex2vec counts *presence* of functional categories per hexagon -- losing building/landuse *polygons* means we lose area-based features, but nodes still give us the point locations of functional POIs.

**Recommendation**: Start with nodes-only extraction. Compare hex2vec embedding quality (via leefbaarometer probe R2) against the mislabeled-but-complete Overpass data. If quality drops significantly, add way centroid extraction as a follow-up (pyosmium can extract way centroids with a node location handler in a second pass).

**3. Keep SRAI IntersectionJoiner, do not replace with Sedona**

`IntersectionJoiner` is a 1-line call that produces the exact `joint_gdf` format expected by `CountEmbedder`, `Hex2VecEmbedder`, and `GeoVexEmbedder`. Replacing it with Sedona would require:
- Manually constructing H3 polygon geometries for the join
- Producing output in SRAI's `(region_id, feature_id)` MultiIndex format
- Testing compatibility with all three embedders

The benefit (speed) is marginal: `IntersectionJoiner` on 500K-800K points against 900K hexagons takes ~2 minutes with GeoPandas spatial index. Sedona would be faster but the join is not the bottleneck (pyosmium extraction and hex2vec training are).

**Keep IntersectionJoiner. Use Sedona only if join time exceeds 10 minutes on production data.**

**4. Shared vs per-year artifacts**

| Artifact | Shared across years? | Rationale |
|----------|---------------------|-----------|
| `regions_gdf` | Yes | H3 tessellation depends only on study area boundary, not data |
| `H3Neighbourhood` | Yes | Topology is boundary-dependent, not data-dependent |
| `area_gdf` | Yes | Study area boundary is fixed |
| POI extraction (parquet) | **Per-year** | Different POIs exist in different years |
| `joint_gdf` | **Per-year** | Spatial join depends on POI locations |
| `features_gdf` | **Per-year** | POI feature matrix depends on extraction |
| Embeddings | **Per-year** | Final output is year-specific |

The processor should compute `regions_gdf` and `neighbourhood` once and reuse them across years. This is already partially implemented (neighbourhood caching exists).

**5. Pop the stash and extend**

The stashed code (`stash@{0}`) adds a `pyosmium-history` data source to the processor. The integration is well-structured:
- `_resolve_pyosmium_pois_path()` for path resolution
- `load_data()` extended with the new branch
- `__main__.py` updated with the new `--data-source` choice

**Pop the stash**, then make these modifications:
- Rename `pyosmium-history` to `pyosmium` (it works with snapshots too, not just history)
- Add a `pyosmium` data source that runs extraction inline rather than requiring a separate pre-extraction step
- Keep the pre-extracted parquet path as a fast-path cache

## What Exists vs What Needs Building

### Exists (working)
- `scripts/one_off/extract_pois_from_history.py` -- pyosmium POI extractor (nodes only, from `.osh.pbf`)
- Stashed processor integration (`stash@{0}`) -- `pyosmium-history` data source in `processor.py` and `__main__.py`
- `SpatialDB` -- Sedona-backed spatial engine (not needed for this pipeline but available)
- PBF snapshots for 2020, 2022, 2024
- Full history `.osh.pbf`

### Needs building
1. **Snapshot extractor** -- adapt `extract_pois_from_history.py` to also handle plain `.osm.pbf` snapshots (simpler: no version tracking needed). Currently the handler is history-specific (tracks `_cur_id`, `_best`, `_past_cutoff`).
2. **Promote to durable script** -- move from `scripts/one_off/` to `scripts/poi/extract_pois_pyosmium.py` with proper docstring.
3. **Inline extraction in processor** -- option to run pyosmium extraction directly from `load_data()` instead of requiring a separate pre-extraction step.
4. **Multi-year batch script** -- a wrapper that extracts + embeds for all three years in sequence, reusing `regions_gdf` and `neighbourhood`.

### Path bug to fix
`StudyAreaPaths.osm_dir()` returns `root / "osm"` but actual PBF files live at `root / "stage1_unimodal" / "osm"`. Either:
- (a) Move OSM files to `root/osm/` (matches CLAUDE.md data layout docs), or
- (b) Fix `osm_dir()` to point to `stage1_unimodal/osm/`

Option (a) is correct per CLAUDE.md. The OSM PBF files are raw data shared by POI and roads, not stage1 output. They should not live under `stage1_unimodal/`.

## Implementation Steps

### Phase 1: Pop stash and fix paths (30 min)
1. `git stash pop` to recover the pyosmium integration
2. Resolve any merge conflicts with current `processor.py`
3. Fix `StudyAreaPaths.osm_dir()` or move OSM files to correct location
4. Verify `extract_pois_from_history.py` runs on the `.osh.pbf` for 2022 cutoff

### Phase 2: Snapshot extractor (1-2 hours)
1. Write `scripts/poi/extract_pois_pyosmium.py` -- durable script that:
   - Accepts `--source snapshot` (default) or `--source history`
   - For snapshots: simplified handler (no version tracking, single pass)
   - For history: existing handler with cutoff logic
   - Both produce identical output format (GeoParquet with `feature_id` index)
2. Add `--years 2020,2022,2024` batch mode
3. Output: `intermediate/pois_gdf/{area}_res{res}_{year}_pois.parquet`

### Phase 3: Processor integration (1 hour)
1. Rename `pyosmium-history` data source to `pyosmium`
2. Add inline extraction: if pre-extracted parquet missing, run pyosmium extraction automatically
3. Update `__main__.py` CLI help text
4. Test: `python -m stage1_modalities.poi --study-area netherlands --resolution 9 --data-source pyosmium --osm-date 2022-01-01 --save-intermediate`

### Phase 4: Multi-year pipeline (1 hour)
1. Write `scripts/poi/run_multi_year.py` -- orchestrates extraction + embedding for all years
2. Computes `regions_gdf` and `neighbourhood` once, passes to each year's pipeline
3. Outputs year-tagged embeddings and intermediates

### Phase 5: Validation (30 min)
1. Run leefbaarometer DNN probe on pyosmium-extracted 2022 POI embeddings
2. Compare R2 against current (mislabeled) Overpass-sourced embeddings
3. If R2 drops >10%, investigate adding way centroids

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Nodes-only misses important POI signal | Medium | Medium | Validate via probe R2 comparison; way centroid extraction is a known follow-up |
| pyosmium snapshot extraction is slow on 1.2 GB PBF | Low | Low | The history extractor processes 3.1 GB in ~15 min; snapshots are simpler/faster |
| Stash pop has merge conflicts | Low | Low | Stash diff is clean and well-scoped; current `processor.py` unchanged since stash creation |
| `building` key in HEX2VEC_FILTER produces sparse counts with nodes-only | High | Low | Building=* on nodes is rare (mainly `building=entrance`). Hex2vec handles sparse features gracefully; the 75 building subtypes will mostly be zero, which is informative (absence of buildings = natural/water area) |

## Consequences

- **Positive**: Truthful year labels on all POI data. Multi-year comparison becomes possible. No more Overpass API dependency. Reproducible from static PBF files.
- **Negative**: Nodes-only extraction captures fewer features (~500K vs ~14M). Some HEX2VEC_FILTER categories (`building`, `landuse`) will be very sparse. Requires validation.
- **Neutral**: Adds another data source option to an already 3-source processor. Complexity is manageable because the sources share the same output format.
