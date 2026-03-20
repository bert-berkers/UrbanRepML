# Stage 1: GeoVeX POI Embeddings + Yearly OSM Data

**Intent**: Expand POI coverage with GeoVeX encoder and fix temporal provenance by wiring all modalities through dated PBF snapshots.

## Valuate guidance

This is a focused Stage 1 modality session. Suggested weights:
- spatial_correctness=5 (temporal provenance is the core issue)
- code_quality=4 (touching processors that other stages depend on)
- model_architecture=3 (GeoVeX is an existing SRAI embedder, not new architecture)
- test_coverage=3 (standard)
- exploration_vs_exploitation=3 (some investigation needed for GeoVeX setup)

Intent: "Wire POI and Roads through 2022 PBF snapshots at res9, generate GeoVeX embeddings as complementary POI representation"

## Context

Current state:
- **POI**: hex2vec is the standard (50D). GeoVeX is registered in `MODALITY_REGISTRY` but embeddings don't exist yet. GeoVeX uses a VAE approach vs hex2vec's skip-gram — should provide complementary signal.
- **Temporal mismatch**: res9 POI+Roads use "latest" (2026 Overpass API). res10 uses 2022 PBF snapshots (correct). The `osm/` directory has PBF infrastructure (`{area}-2022-01-01.osm.pbf`) but res9 processors aren't wired to use it.
- **GTFS**: Uses 2026 OVapi feed. Temporal alignment with 2022 AlphaEarth is a known issue but not addressable today.

Key files:
- `stage1_modalities/poi/` — POI processors, hex2vec encoder
- `stage1_modalities/roads/` — Roads highway2vec processor
- `utils/paths.py` — `StudyAreaPaths.osm_dir()`, `.osm_snapshot_pbf(date)`
- CLAUDE.md documents: "POI and roads processors auto-resolve PBF paths from `osm/` when `data_source='pbf'`"

## Wave 0: Clean State
- `git status`, commit any dirty state
- Start `/loop 5m /sync` for lateral coordination

## Wave 1: Investigate GeoVeX + OSM state (parallel)

1. **stage1-modality-encoder**: Investigate GeoVeX integration. Check what SRAI's GeoVeX embedder expects (input format, training requirements, hyperparameters). Check if `stage1_modalities/poi/` already has GeoVeX code or if it needs writing from scratch. Report: what exists, what's missing, estimated complexity.

2. **stage1-modality-encoder**: Investigate OSM temporal state. Check which PBF files exist in `data/study_areas/netherlands/osm/`. Check if POI and Roads processors have a `data_source='pbf'` path that works for res9. Check what's needed to generate `netherlands-2022-01-01.osm.pbf` if it doesn't exist. Report: what exists, what's blocking, what's needed.

## Wave 2: Generate 2022-dated res9 POI + Roads (if PBFs exist)

3. **stage1-modality-encoder**: Run POI hex2vec processor with `data_source='pbf'` pointing at the 2022 snapshot for res9. Output: `poi/hex2vec/netherlands_res9_2022.parquet`. This replaces the "latest" version with temporally correct data.

4. **stage1-modality-encoder**: Same for Roads: generate `roads/netherlands_res9_2022.parquet` from 2022 PBF.

## Wave 3: GeoVeX POI embeddings

5. **stage1-modality-encoder**: Generate GeoVeX embeddings for res9 using the 2022 PBF data. Follow SRAI's GeoVeX API. Output: `poi/geovex/netherlands_res9_2022.parquet`.

## Wave 4: Verify + Compare

6. **execution**: Run `plot_embeddings.py` for the new GeoVeX embeddings to visually verify they look reasonable.
7. **qaqc**: Compare hex2vec vs GeoVeX — dimensionality, coverage, variance structure. Run correlation between the two to see if they're capturing different signal.
8. **devops**: Commit.

## Final Wave: Close-out
- Coordinator scratchpad
- `/ego-check`

## Execution

```
/valuate
```
Then:
```
/niche .claude/plans/2026-03-15-stage1-geovex-and-osm-dates.md
```
