# h3_index vs region_id Naming Convention

## Status: Draft

## Context

The codebase uses two names for the same thing: H3 hexagonal cell identifiers (hex strings like `"891f1d48163ffff"`).

- `region_id` -- the SRAI convention. SRAI's `H3Regionalizer` produces GeoDataFrames indexed by `region_id`. All SRAI tools expect this name.
- `h3_index` -- the legacy name, used before SRAI adoption. Still present in Stage 1 output and many scripts.

The current state is a partial migration. Stage 1 processors write `h3_index` to parquet files on disk. Stage 2+ code works with `region_id` internally. Bridge code exists in at least 5 locations to translate between the two. This dual naming creates cognitive overhead and has already been flagged as "decision debt" by both the librarian (codebase graph) and the ego (forward-look for 2026-02-08).

### Where each name appears today

**`h3_index` writers (produce `h3_index` in output):**
- `stage1_modalities/alphaearth/processor.py` -- builds records with `{'h3_index': h3_index_str}` (line 367)
- `stage1_modalities/poi/processor.py` -- renames `region_id` -> `h3_index` before return (line 253-254)
- `stage1_modalities/roads/processor.py` -- renames `region_id` -> `h3_index` in two code paths (lines 202-205, 326-329)
- `base.py` `save_embeddings()` -- writes with `index=False`, so column name comes from caller

**Bridge code (translates `h3_index` <-> `region_id`):**
- `stage2_fusion/data/multimodal_loader.py` -- reads `h3_index`, sets as index (lines 47-54)
- `stage2_fusion/data/cone_dataset.py` -- reads `h3_index`, renames to `region_id` (lines 175-177)
- `stage2_fusion/inference/hierarchical_cone_inference.py` -- reads `h3_index`, renames to `region_id` (lines 129-131)
- `scripts/netherlands/train_lattice_unet_res10_cones.py` -- reads `h3_index`, renames to `region_id` (lines 247-249)
- `stage3_analysis/visualization/cluster_viz.py` -- handles both names with fallback logic (lines 153-167, 323)
- `stage3_analysis/validation.py` -- checks for `h3_index` column (lines 55-56, 63-64, 109-110)
- `scripts/processing_modalities/alphaearth/process_tiff_to_h3.py` -- renames `region_id` -> `h3_index` (line 477)

**`region_id` native (no translation needed):**
- `stage2_fusion/data/hierarchical_cone_masking.py` -- uses `region_id` via `regions_gdf` throughout
- `stage2_fusion/graphs/hexagonal_graph_constructor.py` -- expects `region_id` index (line 88)
- `stage2_fusion/data/study_area_filter.py` -- uses `region_id` (line 445)
- All SRAI calls (`H3Regionalizer`, `H3Neighbourhood`)
- `scripts/preprocessing_auxiliary_data/setup_regions.py`, `setup_density.py`, `setup_fsi_filter.py`
- `scripts/accessibility/` scripts

**On-disk parquet files (12 embedding files across 3 study areas):**
- All embedding parquets use `h3_index` as column name
- All regions_gdf parquets use `region_id` as index (produced by SRAI)

## Decision

**Recommendation: Option (b) -- Migrate to `region_id` throughout.**

## Option (a): Commit to the Bridge Pattern Permanently

Keep `h3_index` in Stage 1 parquet output, keep the bridge code, document it as the standard.

### Pros
- Zero risk to existing data files (12 embedding parquets remain valid as-is)
- No re-processing required
- No data migration script needed
- Already working today

### Cons
- The bridge code is not centralized. It exists in at least 5 separate locations with slightly different implementations: `MultiModalLoader` checks for `h3_index` column, `ConeDataset` renames `h3_index` to `region_id`, `cluster_viz.py` handles both with fallback, `validation.py` checks for `h3_index` specifically. Each location is a subtly different translation and a place where a future developer (or agent) could get it wrong.
- Stage 1 processors (POI, roads) actively rename FROM `region_id` TO `h3_index` at their output boundary (e.g., `poi/processor.py` line 253-254: `if embeddings_df.index.name == 'region_id': embeddings_df.index.name = 'h3_index'`). This is the SRAI convention being intentionally overwritten. The processors receive `region_id` from SRAI and convert it to `h3_index` for output, then Stage 2 converts it back. This is a round-trip through a legacy name.
- Every new consumer of Stage 1 output must know to look for `h3_index` and every new Stage 2+ component must know to expect `region_id`. This is a persistent source of confusion.
- The CLAUDE.md "Stage Boundary Convention" section already acknowledges this is a compromise, not a design. It reads as an apology for technical debt rather than an intentional architecture.
- As new modalities are added (GTFS, aerial imagery), each must replicate the `region_id` -> `h3_index` conversion at output and each consumer must replicate the `h3_index` -> `region_id` conversion at intake.

## Option (b): Migrate to `region_id` Throughout

Change Stage 1 output to use `region_id`. Remove bridge code. Update existing parquet files.

### Pros
- Single naming convention across the entire codebase. No translation needed anywhere.
- Aligns with SRAI convention, which is a project principle ("SRAI-First").
- Removes ~30 lines of bridge/translation code across 7 files.
- Removes the round-trip conversion in POI and roads processors (they currently receive `region_id` from SRAI, rename it to `h3_index`, only for downstream code to rename it back).
- New modalities and new consumers "just work" without needing to know about a legacy convention.
- Eliminates an entire class of potential bugs (mismatched column names at stage boundaries).

### Cons
- 12 existing embedding parquet files must be updated (column rename from `h3_index` to `region_id`). This is a simple pandas operation per file: `df.rename(columns={'h3_index': 'region_id'})` or setting the index name. Estimated effort: a short script, under 30 lines.
- The ~85 tile_regions parquet files (cascadia, pearl_river_delta) may also use `h3_index` and would need checking.
- Any external notebooks, one-off scripts, or analysis code that reads these parquets by column name will break. This is a single-developer project so the blast radius is limited, but it is real.
- Requires coordinated changes across Stage 1, Stage 2, Stage 3, and scripts. This is not a single-file change.

### Affected Files for Migration

**Stage 1 (stop writing `h3_index`, use `region_id` natively):**
1. `stage1_modalities/alphaearth/processor.py` -- change `{'h3_index': h3_index_str}` to `{'region_id': h3_index_str}` at line 367
2. `stage1_modalities/poi/processor.py` -- remove the `region_id` -> `h3_index` rename at lines 253-254
3. `stage1_modalities/roads/processor.py` -- remove the `region_id` -> `h3_index` rename at lines 202-205 and 326-329

**Stage 2 (remove bridge code):**
4. `stage2_fusion/data/multimodal_loader.py` -- change `h3_index` references to `region_id` (lines 30, 46-54, 157, 233, 245)
5. `stage2_fusion/data/cone_dataset.py` -- remove `h3_index` -> `region_id` rename (lines 175-177)
6. `stage2_fusion/pipeline.py` -- change `h3_index` reference (lines 306-308)
7. `stage2_fusion/inference/hierarchical_cone_inference.py` -- remove `h3_index` -> `region_id` rename (lines 129-131)

**Stage 3 (simplify dual-handling):**
8. `stage3_analysis/visualization/cluster_viz.py` -- remove `h3_index` fallback logic, keep `region_id` only (lines 153-167, 323)
9. `stage3_analysis/validation.py` -- change `h3_index` references to `region_id` (lines 55-56, 63-64, 109-110, 116)

**Scripts (update references):**
10. `scripts/netherlands/train_lattice_unet_res10_cones.py` -- remove rename (lines 247-249)
11. `scripts/netherlands/infer_cone_alphaearth.py` -- change `h3_index` references (lines 141, 241, 273)
12. `scripts/processing_modalities/alphaearth/process_tiff_to_h3.py` -- remove `region_id` -> `h3_index` rename (lines 470-477)
13. Various scripts in `scripts/processing_modalities/` that reference `h3_index`

**Data migration (one-time script):**
14. Rename column in 12 embedding parquet files
15. Check and update ~85 tile_regions parquet files if applicable

**Documentation:**
16. `CLAUDE.md` -- update "Stage Boundary Convention" section, remove bridge pattern description
17. Librarian's `codebase_graph.md` -- update "Index Contracts" table

## Consequences

### Positive
- One name, one convention, everywhere. Developer confusion eliminated.
- SRAI alignment is complete rather than partial.
- Bridge code removed, reducing maintenance surface.
- New modalities and consumers require no special knowledge of legacy naming.

### Negative
- One-time migration effort across ~13 source files and ~12 data files.
- Risk of breaking uncommitted notebooks or ad-hoc analysis scripts that reference `h3_index`.
- Requires coordinated multi-file change (should be done as a single commit).

### Neutral
- The data files themselves are unchanged in content (same hex strings, same embeddings). Only the column/index name changes. Any downstream analysis that reads by position rather than by name is unaffected.
- The migration script is trivially reversible if problems emerge.

## Implementation Notes

### Ordering
1. Write the parquet migration script and run it (update all data files first)
2. Update Stage 1 processors (stop producing `h3_index`)
3. Update Stage 2 consumers (remove bridge code)
4. Update Stage 3 and scripts
5. Update CLAUDE.md and codebase graph
6. Run import smoke tests and verify end-to-end loading
7. Commit as a single atomic change

### Dependencies
- This should happen AFTER the qaqc import smoke tests confirm current code works (do not change naming on top of untested SRAI migration)
- This should happen BEFORE any new modality processors are written (avoid encoding the legacy convention in new code)

### Migration Script Sketch
```python
# One-time data migration (do not commit this script)
import pandas as pd
from pathlib import Path

parquets = Path("data/study_areas").rglob("*.parquet")
for p in parquets:
    df = pd.read_parquet(p)
    if 'h3_index' in df.columns:
        df = df.rename(columns={'h3_index': 'region_id'})
        df.to_parquet(p)
        print(f"Updated: {p}")
```
