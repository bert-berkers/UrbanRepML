# Plan: Stage 1 Multi-Year Foundation + Stage 3 Plotting

## Goal

Build a solid, honest Stage 1 foundation: fix data provenance mislabeling, design the multi-year POI pipeline (pyosmium+Sedona), generate proper per-modality res9 plots using Stage 3 tools, and clean up plot organization. This is pre-work for Stage 2 fusion — we need trustworthy, well-visualized unimodal embeddings before fusing them.

## Overarching Objectives

1. **Data honesty**: No file should claim a year it doesn't represent. Guard rails in code to prevent future mislabeling.
2. **Multi-year readiness**: Efficient pipeline to process POI for 2020, 2022, 2024 from PBF snapshots. Roads uses SRAI/quackosm (lighter data). POI uses pyosmium+Sedona (handles 12.5M+ features reliably).
3. **Validate embeddings visually**: Per-modality res9 plots using Stage 3 infrastructure, not one-off scripts. Clear separation between unimodal, multimodal, and analysis plots.
4. **Clean plot organization**: Plots live where their data lives — unimodal with modalities, fused with stage2, probes with stage3, targets with targets.

## Current State

### Data on disk
- **AlphaEarth res9**: `data/study_areas/netherlands/stage1_unimodal/alphaearth/netherlands_res9_2022.parquet` — VALID (from Google Earth Engine 2022 composites)
- **POI res9**: `data/study_areas/netherlands/stage1_unimodal/poi/netherlands_res9_2022.parquet` — MISLABELED (Overpass/current data, not 2022)
- **POI hex2vec res9**: `data/study_areas/netherlands/stage1_unimodal/poi/hex2vec/netherlands_res9_2022.parquet` — MISLABELED (trained on mislabeled POI counts)
- **Roads res9**: `data/study_areas/netherlands/stage1_unimodal/roads/netherlands_res9_2022.parquet` — MISLABELED (Overpass/current data)
- **Leefbaarometer**: `data/study_areas/netherlands/target/leefbaarometer/leefbaarometer_h3res9_2022.parquet` — VALID (from 2022 survey)
- **PBF snapshots**: `data/study_areas/netherlands/osm/netherlands-{2020,2022,2024}-01-01.osm.pbf` — all valid, confirmed readable
- **Full history**: `data/study_areas/netherlands/osm/netherlands-internal.osh.pbf` — for extracting additional year snapshots via osmium CLI

### Existing plots (to clean up)
- `stage1_unimodal/plots/res9_diagnostics/` — **DELETE**: orphan dir from one-off script, cross-cutting diagnostics that don't belong here
- `stage1_unimodal/alphaearth/plots/` — res10 plots (8 files, Mar 1). Keep, but these are res10 not res9.
- `stage1_unimodal/poi/hex2vec_27feat/plots/` — res10 plots (9 files, Mar 1). Keep.
- `stage1_unimodal/roads/plots/` — res10 plots (9 files, Feb 28). Keep.
- `target/leefbaarometer/plots/` — target distribution plots (8 files). Keep.

### Plot organization (target state)
```
data/study_areas/netherlands/
  stage1_unimodal/
    alphaearth/plots/res9/    <-- per-modality unimodal diagnostics
    alphaearth/plots/res10/   <-- (existing plots could be moved here)
    poi/hex2vec/plots/res9/
    roads/plots/res9/
  stage2_multimodal/
    plots/                    <-- fused results (concat, learned fusion)
  stage3_analysis/
    {probe_type}/{run_id}/plots/   <-- probe results (existing structure, keep)
  target/
    leefbaarometer/plots/     <-- target distributions (existing, keep + extend)
```

### Stashed code
- `stash@{0}`: "pine-branching-bay: pyosmium-history support + settings.json fix"
  - Contains: `pyosmium-history` data source in POI processor, pyosmium POI extractor, settings.json change
  - Status: 3+ sessions undecided. The pyosmium-history code extracts POI nodes from `.osh.pbf`. Needs evaluation as part of the pyosmium+Sedona pipeline design.

### Key code locations
- **POI processor**: `stage1_modalities/poi/processor.py` — `load_data()`, `process_to_h3()`, HEX2VEC_FILTER categories
- **POI CLI**: `stage1_modalities/poi/__main__.py` — `--data-source`, `--year`, `--osm-date` args (year defaults to 2022 regardless of source!)
- **Roads processor**: `stage1_modalities/roads/processor.py`
- **Roads CLI**: `stage1_modalities/roads/__main__.py` — `--year` hardcoded default 2022
- **SpatialDB/Sedona**: `utils/spatial_db.py` — SedonaDB spatial engine, bulk H3 geometry queries
- **StudyAreaPaths**: `utils/paths.py` — `embedding_file()`, `osm_snapshot_pbf()`, `stage1()`
- **Cluster viz**: `stage3_analysis/visualization/cluster_viz.py` — MiniBatchKMeans, dissolve, datashader, hierarchical subplots
- **Linear probe viz**: `stage3_analysis/linear_probe_viz.py` — `_rasterize_centroids()` (best spatial renderer), PCA RGB maps, spatial residuals
- **DNN probe viz**: `stage3_analysis/dnn_probe_viz.py` — inherits linear_probe_viz
- **One-off diagnostics (to replace)**: `scripts/one_off/res9_2022_diagnostics.py` — produced the plots we're deleting

### Scratchpads with relevant context
- `.claude/scratchpad/stage1-modality-encoder/2026-03-07.md` — mislabeling audit, all 10 files listed, guard rail proposal, multi-year design
- `.claude/scratchpad/devops/2026-03-07.md` — PBF loading confirmed working, quackosm 0.17.0, pyosmium 4.3.0
- `.claude/scratchpad/stage3-analyst/2026-03-07.md` — full Stage 3 tool audit, gap analysis, proposed new modules
- `.claude/scratchpad/execution/2026-03-06.md` — documents the Overpass fallback that caused mislabeling
- `.claude/scratchpad/coordinator/2026-03-06.md` — session 3 DNN probe R2 table, highway2vec 1D collapse

### DNN Probe results (for reference — this is what the embeddings predict)
| Modality | mean R2 | best target |
|---|---|---|
| AlphaEarth 64D | 0.507 | vrz=0.751 |
| POI hex2vec 50D | 0.420 | soc=0.606 |
| Roads h2v 30D | 0.239 | soc=0.443 |
| Concat 144D | **0.544** | vrz=0.779 |

Roads 1D collapse (PC1=96.5%) is a valid result — road networks follow standard hierarchy. Not a bug.

---

## Wave 1 (parallel — 3 agents)

### 1a. spec-writer: Design pyosmium+Sedona POI pipeline

**Read these first:**
- `git stash show -p stash@{0}` — the pyosmium-history stash (POI extractor + processor integration)
- `stage1_modalities/poi/processor.py` — current pipeline: `load_data()` flow, `HEX2VEC_FILTER`, `IntersectionJoiner` usage
- `stage1_modalities/poi/__main__.py` — CLI args, year resolution logic
- `utils/spatial_db.py` — what Sedona can do today (spatial joins?)
- `scripts/one_off/extract_pois_from_history.py` — existing pyosmium POI extractor
- `.claude/scratchpad/stage1-modality-encoder/2026-03-02.md` — notes on pyosmium-history integration

**Design decisions to make:**
- pyosmium extracts nodes only. Current pipeline has 96% ways, 4% nodes. For hex2vec categorical counting, are nodes sufficient? Check what HEX2VEC_FILTER categories actually need (amenity, shop, etc. are tagged on nodes; buildings are ways — but buildings aren't POI categories).
- Can Sedona replace SRAI's `IntersectionJoiner` for point-in-H3-hexagon spatial join?
- What's shared across years (regions_gdf, neighbourhood graph) vs per-year (POI extraction, counting)?
- Should the stashed pyosmium-history code be popped and extended, or rewritten?

**Architecture to design:**
```
PBF file (year-specific)
  -> pyosmium (extract POI nodes with tags, fast, reliable)
  -> GeoDataFrame or Parquet (intermediate, year-tagged)
  -> Sedona spatial join (points to H3 hexagons)
  -> count matrix per hex (year-tagged)
  -> hex2vec training (year-tagged embeddings)
```

**Output:** `specs/poi-pipeline-pyosmium-sedona.md`
**Scratchpad:** `.claude/scratchpad/spec-writer/2026-03-07.md`
**Acceptance:** Spec covers architecture, what exists vs needs building, multi-year efficiency, implementation steps, risks.

### 1b. stage1-modality-encoder: Fix mislabeling + guard rail

**Actions:**
1. **Delete** `data/study_areas/netherlands/stage1_unimodal/plots/` (the entire orphan directory — 8 files in `res9_diagnostics/`)
2. **Rename** these 10 files from `_2022` to `_latest`:
   - `poi/netherlands_res9_2022.parquet` -> `poi/netherlands_res9_latest.parquet`
   - `poi/hex2vec/netherlands_res9_2022.parquet` -> `poi/hex2vec/netherlands_res9_latest.parquet`
   - `poi/intermediate/features_gdf/netherlands_res9_2022_features.parquet` -> `..._latest_features.parquet`
   - `poi/intermediate/joint_gdf/netherlands_res9_2022_joint.parquet` -> `..._latest_joint.parquet`
   - `poi/intermediate/regions_gdf/netherlands_res9_2022_regions.parquet` -> `..._latest_regions.parquet`
   - `poi/intermediate/neighbourhood/netherlands_res9_2022_neighbourhood.pkl` -> `..._latest_neighbourhood.pkl`
   - `roads/netherlands_res9_2022.parquet` -> `roads/netherlands_res9_latest.parquet`
   - `roads/intermediate/features_gdf/netherlands_res9_2022_features.parquet` -> `..._latest_features.parquet`
   - `roads/intermediate/joint_gdf/netherlands_res9_2022_joint.parquet` -> `..._latest_joint.parquet`
   - `roads/intermediate/regions_gdf/netherlands_res9_2022_regions.parquet` -> `..._latest_regions.parquet`
3. **Add guard rail** to both processors: when `data_source='osm_online'` and `year` is not `'latest'`, raise `ValueError` explaining that Overpass returns current data, not historical snapshots.
   - POI: `stage1_modalities/poi/processor.py` (in `__init__` or `load_data`)
   - POI CLI: `stage1_modalities/poi/__main__.py` (change `--year` default from `2022` to `latest`)
   - Roads: `stage1_modalities/roads/processor.py`
   - Roads CLI: `stage1_modalities/roads/__main__.py` (change `--year` default from `2022` to `latest`)
4. **Update `StudyAreaPaths.embedding_file()`** to accept `year: Union[int, str]` so `'latest'` works as a year value.

**Scratchpad:** `.claude/scratchpad/stage1-modality-encoder/2026-03-07.md` (update existing)
**Acceptance:** Guard rail raises on mismatched source+year. Files renamed. Orphan plots deleted. No code references hardcoded year 2022 as default.

### 1c. ego: Assess supra system design flaw

**Issue:** `characteristic_states.yaml` is a single file with no coordinator scoping. When multiple coordinators run concurrently on different tasks (e.g., Stage 1 foundation vs Stage 2 UNet audit), each overwrites the supra state. This caused our coordinator to absorb Stage 2 focus dimensions that didn't apply to our Stage 1 work.

**Assess:**
- How should supra state work with multiple concurrent coordinators?
- Options: per-coordinator state files, coordinator-scoped sections in one file, read-only supra with coordinator-local overrides
- Is the supra system bouncing too much within sessions? The state changed 3 times today.
- Read `.claude/supra/schema.yaml` and `.claude/supra/characteristic_states.yaml` for current design

**Scratchpad:** `.claude/scratchpad/ego/2026-03-07.md`

---

## Wave 2 (after Wave 1 — 2 agents, parallel)

### 2a. stage3-analyst: Build per-modality res9 diagnostic plots

**Objective:** Generate proper per-modality res9 plots using existing Stage 3 infrastructure. Do NOT write one-off scripts. Extend or compose existing tools.

**Existing tools to use/extend:**
- `stage3_analysis/visualization/cluster_viz.py` — `load_and_prepare_embeddings()`, `perform_minibatch_clustering()`, `create_cluster_visualization()`. Already supports res9 via `STUDY_AREA_CONFIG`.
- `stage3_analysis/linear_probe_viz.py` — `_rasterize_centroids()` is the fastest spatial renderer. Extract it into a shared utility if needed.

**Plots to generate per modality (alphaearth, poi/hex2vec, roads):**
1. PCA RGB spatial map (top 3 PCs as RGB on Netherlands map)
2. Embedding distributions (KDE of top PCA components)
3. Cluster maps (k=8, k=12)
4. Coverage map
5. PCA explained variance curve

**Output locations:**
- `data/study_areas/netherlands/stage1_unimodal/alphaearth/plots/res9/`
- `data/study_areas/netherlands/stage1_unimodal/poi/hex2vec/plots/res9/`
- `data/study_areas/netherlands/stage1_unimodal/roads/plots/res9/`

**Leefbaarometer target distributions (separate from embeddings):**
- Load: `data/study_areas/netherlands/target/leefbaarometer/leefbaarometer_h3res9_2022.parquet`
- Plot: 2x3 grid, one subplot per score (lbm, fys, onv, soc, vrz, won). KDE + histogram. Median/mean lines. N hexagons annotated.
- Output: `data/study_areas/netherlands/target/leefbaarometer/plots/res9_distributions.png`
- This is TARGET data — visually distinct from embedding distributions. Different color palette, clearly titled "Leefbaarometer Target Distributions".

**Data to use (current "latest" res9 — still useful for pipeline validation):**
- `stage1_unimodal/alphaearth/netherlands_res9_2022.parquet` (this one IS actually 2022)
- `stage1_unimodal/poi/hex2vec/netherlands_res9_latest.parquet` (renamed in Wave 1)
- `stage1_unimodal/roads/netherlands_res9_latest.parquet` (renamed in Wave 1)

**Create a durable script** at `scripts/stage1/plot_unimodal_diagnostics.py` that:
- Takes `--study-area`, `--resolution`, `--modality` (or `all`), `--year`
- Uses Stage 3 tools internally
- Outputs to the correct per-modality plot directory
- Has module docstring with lifetime=durable, stage=1+3

**Scratchpad:** `.claude/scratchpad/stage3-analyst/2026-03-07.md` (update existing)
**Acceptance:** Plots generated for all 3 modalities + leefbaarometer. Script is durable, not one-off. Uses Stage 3 infrastructure.

### 2b. execution: Run the diagnostic plots

**Run the script created by 2a** on all three modalities + leefbaarometer target.
Capture output, report any failures.

**Scratchpad:** `.claude/scratchpad/execution/2026-03-07.md`

---

## Wave 3: QAQC (light verification)

### 3. qaqc: Verify Wave 1 + Wave 2 outputs

**Check:**
- [ ] `stage1_unimodal/plots/` directory is gone
- [ ] No files named `_2022` exist for POI/roads res9 (only `_latest`)
- [ ] Guard rail triggers: simulate `--data-source osm_online --year 2022` and confirm it errors
- [ ] AlphaEarth res9 file still named `_2022` (correct — it IS 2022 data)
- [ ] Per-modality plots exist in `{modality}/plots/res9/`
- [ ] Leefbaarometer distributions exist in `target/leefbaarometer/plots/`
- [ ] Durable script exists at `scripts/stage1/plot_unimodal_diagnostics.py` with docstring
- [ ] No plot code lives in `scripts/one_off/` (the old diagnostics script should be deleted or archived)

**Scratchpad:** `.claude/scratchpad/qaqc/2026-03-07.md`

---

## Final Wave (mandatory close-out)

1. Write coordinator scratchpad at `.claude/scratchpad/coordinator/2026-03-07.md` (update existing)
2. `/librarian-update` — sync codebase graph with today's changes
3. `/ego-check` — process health assessment + forward-look

Steps 2 and 3 can run in parallel after step 1.

---

## Post-Plan: Next Session Work

Once this plan completes, the foundation is set for:

1. **Implement pyosmium+Sedona POI pipeline** (from the spec written in Wave 1a)
2. **Run POI pipeline for 2020, 2022, 2024** from PBF snapshots
3. **Run roads pipeline for 2020, 2022, 2024** using SRAI/quackosm (confirmed working)
4. **Re-run hex2vec training** on actual 2022 POI data
5. **Re-run DNN probes** with correctly-dated embeddings
6. **Stage 2 fusion** with trustworthy, multi-year unimodal embeddings

## Execution
Invoke: `/coordinate .claude/plans/stage1-foundation-and-plotting.md`
