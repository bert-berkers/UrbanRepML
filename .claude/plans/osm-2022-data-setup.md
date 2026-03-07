# Plan: Historical OSM 2022 Data Setup for POI + Roads

## Goal
Set up 2022 OSM data for both POI and roads processors, ensuring all output files and intermediates are properly year-tagged to coexist with other years.

## Context
- AlphaEarth embeddings are from 2022, leefbaarometer targets are from 2022
- Current POI/roads data was loaded from latest OSM (2026) — temporally misaligned
- User will download `netherlands-internal.osh.pbf` from Geofabrik internal archive
- osmium time-filter extracts a 2022-01-01 snapshot PBF
- The PBF goes in `data/study_areas/netherlands/osm/`
- Output embeddings already include year in filename (`netherlands_res10_2022.parquet`)
- **Problem**: Intermediates (regions_gdf, features_gdf, joint_gdf, neighbourhood) do NOT include year — they'll overwrite each other

## Pre-requisites (user does manually)
1. Download `netherlands-internal.osh.pbf` from `https://osm-internal.download.geofabrik.de/europe/netherlands/`
2. Place it at `data/study_areas/netherlands/osm/netherlands-internal.osh.pbf`
3. Install osmium-tool: `pip install osmium` or `conda install -c conda-forge osmium-tool`
4. Extract 2022 snapshot:
   ```bash
   osmium time-filter data/study_areas/netherlands/osm/netherlands-internal.osh.pbf 2022-01-01T00:00:00Z -o data/study_areas/netherlands/osm/netherlands-2022-01-01.osm.pbf
   ```

## Wave 1: Year-tag intermediates (parallel)

### 1a. stage1-modality-encoder: Add year to POI intermediate file paths
**Files**: `stage1_modalities/poi/processor.py`, `stage1_modalities/poi/__main__.py`
**What**:
- Intermediate filenames currently: `netherlands_res10_regions.parquet`, `netherlands_res10_features.parquet`, `netherlands_res10_joint.parquet`
- Change to: `netherlands_res10_2022_regions.parquet`, etc.
- The `--year` CLI arg already exists (default 2022). Pass it through to `_save_intermediate_data()` and `load_intermediates()`
- Neighbourhood cache: `netherlands_res10_neighbourhood.pkl` → `netherlands_res10_2022_neighbourhood.pkl`
- `osm_snapshot_pbf(date)` already maps `--osm-date` to the right PBF. Wire `--osm-date` to also set `--year` if not explicitly provided (e.g., `--osm-date 2022-01-01` → year=2022)
- Acceptance: intermediates include year, old yearless intermediates still loadable (fallback)

### 1b. stage1-modality-encoder: Add year to roads intermediate/output paths
**Files**: `stage1_modalities/roads/processor.py`
**What**:
- Roads processor needs the same year-tagging treatment
- Add `--osm-date` and `--year` CLI support (roads has no `__main__.py` — check how it's invoked)
- Intermediate and output filenames must include year
- Acceptance: roads output is year-tagged, auto-resolves PBF from osm/ dir

## Wave 2: QAQC verification

### 2. qaqc: Verify year-tagging
- All intermediate filenames include year
- `load_intermediates()` with year parameter works
- Output embedding filenames include year (already do, verify)
- `StudyAreaPaths.embedding_file()` year parameter works correctly
- POI and roads processor tests pass
- No hardcoded paths without year

## Wave 3: devops commit

### 3. devops: Commit changes
- Commit message: `feat: add year-tagging to POI/roads intermediates for temporal alignment`
- Push all unpushed commits (currently 2 ahead + this one)

## Final Wave
- Coordinator scratchpad
- `/librarian-update`
- `/ego-check`

## Post-plan: User runs (manual)

After the plan executes and the user has the 2022 PBF:

```bash
# POI: full pipeline with 2022 OSM data
python -m stage1_modalities.poi --study-area netherlands --resolution 10 --data-source pbf --osm-date 2022-01-01 --save-intermediate

# POI: hex2vec embeddings from 2022 intermediates
python -m stage1_modalities.poi --study-area netherlands --resolution 10 --embedder hex2vec --year 2022 --batch-size 65536 --initial-batch-size 16384 --early-stopping-patience 2 --device cpu

# Roads: with 2022 data
python -m stage1_modalities.roads --study-area netherlands --resolution 10 --data-source pbf --osm-date 2022-01-01
```

## Execution
Invoke: `/coordinate .claude/plans/osm-2022-data-setup.md`
