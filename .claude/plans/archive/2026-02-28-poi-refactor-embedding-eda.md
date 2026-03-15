# Plan: POI Processor Refactor + Hex2Vec/GeoVex Generation + Embedding EDA

## Context

We have POI count embeddings (28 dims, 6.46M hexes) and Roads Highway2Vec embeddings (64 dims, 1.02M hexes) at res10 for Netherlands. **Hex2vec and geovex haven't been generated yet** — the POI processor has them implemented but they default to off. The processor is monolithic: `process_to_h3()` bundles regionalize → join → count → diversity → hex2vec → geovex → rename into one method. You can't re-run just one embedder without re-downloading from OSM.

Intermediates exist on disk (5.1M regions_gdf, 2.6M features_gdf, 9.8M joint_gdf) but there's no method to load them.

**Goal**: Refactor POI processor for independent embedder runs, generate hex2vec + geovex (separate GPU runs), then visualize all embeddings.

## Part 1: POI Processor Refactor

**File**: `stage1_modalities/poi/processor.py`

### Extract sub-methods from `process_to_h3`

| Method | Extracted from | Returns |
|---|---|---|
| `load_intermediates(h3_resolution, study_area_name)` | New | `(regions_gdf, features_gdf, joint_gdf)` |
| `run_count_embeddings(regions_gdf, features_gdf, joint_gdf)` | Lines 162-184 | DataFrame, original col names, `region_id` index |
| `run_hex2vec(regions_gdf, features_gdf, joint_gdf)` | Lines 186-219 | DataFrame, `hex2vec_0..N` cols, `region_id` index |
| `run_geovex(regions_gdf, features_gdf, joint_gdf)` | Lines 223-261 | DataFrame, `geovex_0..N` cols, `region_id` index |

`process_to_h3` calls these internally — no external behavior change. Base class (`ModalityProcessor`) has no abstract methods to match.

### `StudyAreaPaths.embedding_file` change

**File**: `utils/paths.py`

Add optional `sub_embedder` param:
```python
def embedding_file(self, modality, resolution, year=2022, sub_embedder=None):
    base = self.stage1(modality)
    if sub_embedder:
        base = base / sub_embedder
    return base / f"{self.study_area}_res{resolution}_{year}.parquet"
```

**17 existing callers all omit `sub_embedder` → zero breakage** (librarian verified).

### CLI entry point

Create `stage1_modalities/poi/__main__.py`:
```bash
python -m stage1_modalities.poi --study-area netherlands --resolution 10 --embedder hex2vec
python -m stage1_modalities.poi --study-area netherlands --resolution 10 --embedder geovex
python -m stage1_modalities.poi --study-area netherlands --resolution 10  # full pipeline (existing)
```

When `--embedder` is set: loads intermediates → runs just that embedder → saves to sub-embedder path.

## Part 2: Generate Hex2Vec + GeoVex

Two separate GPU runs, full Netherlands (5.1M regions):
1. `python -m stage1_modalities.poi --study-area netherlands --resolution 10 --embedder hex2vec`
2. `python -m stage1_modalities.poi --study-area netherlands --resolution 10 --embedder geovex`

**Output files:**
- `stage1_unimodal/poi/hex2vec/netherlands_res10_2022.parquet`
- `stage1_unimodal/poi/geovex/netherlands_res10_2022.parquet`

## Part 3: Embedding EDA Visualization

**New file**: `scripts/plot_embeddings.py`

### Plots per modality (×4 modalities = ~28 figures)

| # | Plot | Method |
|---|---|---|
| 1 | Dimension grid (4×3, first 12 dims) | `rasterize_continuous` per dim |
| 2 | Summary stats (mean + std across dims) | Per-hex aggregation |
| 3 | PCA top-3 components (3-panel) | `apply_pca_reduction` → rasterize each |
| 4 | PCA RGB composite (PC1→R, PC2→G, PC3→B) | p2/p98 normalize → `_rasterize_centroids` |
| 5 | MiniBatchKMeans clusters (k=8,12,16) | `perform_minibatch_clustering` |
| 6 | Correlation heatmap | `seaborn.heatmap` (non-spatial) |
| 7 | Coverage map | Binary rasterize |

### Modalities
- POI count (`stage1_unimodal/poi/netherlands_res10_2022.parquet`, 28 dims)
- POI hex2vec (`stage1_unimodal/poi/hex2vec/netherlands_res10_2022.parquet`)
- POI geovex (`stage1_unimodal/poi/geovex/netherlands_res10_2022.parquet`)
- Roads highway2vec (`stage1_unimodal/roads/netherlands_res10_2022.parquet`, 64 dims)

### Plot output directories (next to data, per project convention)
- `stage1_unimodal/poi/plots/`
- `stage1_unimodal/poi/hex2vec/plots/`
- `stage1_unimodal/poi/geovex/plots/`
- `stage1_unimodal/roads/plots/`

### Reuse from existing code

| Component | Source |
|---|---|
| `rasterize_continuous`, `plot_spatial_map`, `load_boundary` | `scripts/plot_targets.py:90-248` |
| `_rasterize_centroids` (RGB array → RGBA image) | `stage3_analysis/linear_probe_viz.py:701-756` |
| RGB normalization (p2/p98 percentile) | `stage3_analysis/linear_probe_viz.py:762-1049` |
| `apply_pca_reduction(embeddings, n_components)` | `stage3_analysis/visualization/cluster_viz.py:185-193` |
| `perform_minibatch_clustering(emb, k_list)` | `stage3_analysis/visualization/cluster_viz.py:196-243` |
| `SpatialDB.centroids(hex_ids, res, crs=28992)` | `utils/spatial_db.py` |
| `StudyAreaPaths.embedding_file(mod, res, year, sub_embedder)` | `utils/paths.py` |

## Execution Waves

### Wave 0: Clean state ✓

### Wave 1: Refactor (parallel)
- **stage1-modality-encoder**: Refactor `processor.py` — extract methods, add `load_intermediates`, create `__main__.py`
- **srai-spatial**: Add `sub_embedder` param to `StudyAreaPaths.embedding_file` in `utils/paths.py`

### Wave 2: Verify refactor
- **qaqc**: Run full test suite, verify no regressions

### Wave 3: Generate embeddings (sequential, user-monitored)
- **execution**: Run hex2vec → user confirms output → run geovex

### Wave 4: EDA Visualization
- **stage3-analyst**: Create `scripts/plot_embeddings.py`
- **execution**: Run it for all 4 modalities

### Wave 5: Commit
- **devops**: Commit all changes

### Final Wave: Close-out
- Coordinator scratchpad, `/librarian-update`, `/ego-check`

## Critical files

| File | Action |
|---|---|
| `stage1_modalities/poi/processor.py` | Refactor: extract 4 methods |
| `stage1_modalities/poi/__main__.py` | New: CLI entry point |
| `utils/paths.py` | Add `sub_embedder` param |
| `scripts/plot_embeddings.py` | New: EDA visualization script |

## Verification

1. `pytest` — all existing tests pass
2. `python -m stage1_modalities.poi --study-area netherlands --resolution 10 --embedder hex2vec` — loads intermediates, runs hex2vec, saves parquet
3. Same for `--embedder geovex`
4. Output parquets have `region_id` index, correct column prefixes
5. `python scripts/plot_embeddings.py --study-area netherlands` — generates ~28 PNGs
6. Visual: maps show Netherlands shape, PCA RGB has distinct spatial structure
