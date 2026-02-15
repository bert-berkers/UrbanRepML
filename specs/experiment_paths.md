# Experiment Path Manager

## Status: Approved

## Context

Hardcoded paths are scattered across all three stage packages. At least 14 source
files in `stage1_modalities/`, `stage2_fusion/`, and `stage3_analysis/` construct
paths using string interpolation, relative `Path("data/...")` literals, or absolute
Windows paths (`D:\Projects`, `C:\Users\...`). This creates several problems:

1. **Fragility**: Moving the project directory or renaming study areas breaks paths.
2. **Inconsistency**: Some files use `embeddings/alphaearth/`, others use
   `stage1_unimodal/alphaearth/`. Old directory names like `results [old 2024]` and
   `data/preprocessed [TODO SORT & CLEAN UP]` persist in code.
3. **Duplication**: The same path pattern (e.g. embedding file for a modality at a
   resolution) is reconstructed independently in 5+ files.
4. **Portability**: Absolute Windows paths in `pipeline.py` configs prevent
   cross-machine use.

`utils/paths.py` provides a single `StudyAreaPaths` class that all stage packages
import. Old directories remain on disk untouched; only code references change.

## Directory Convention

New runs and all code references use:

```
data/study_areas/{study_area}/
  stage1_unimodal/{modality}/               # Unimodal embeddings
    {study_area}_res{res}_{year}.parquet
    {study_area}_res{res}_pca{n}_{year}.parquet
    intermediate/                           # Per-tile processing artifacts
  stage2_multimodal/{model_name}/           # Fusion model outputs
    checkpoints/
    embeddings/
    plots/
    training_logs/
  stage3_analysis/{analysis_type}/          # Analysis outputs
    e.g. linear_probe/, dnn_probe/, clustering/
  cones/                                    # Hierarchical cone cache
    cone_cache_res{p}_to_{t}/
    parent_to_children_res{p}_to_{t}.pkl
  regions_gdf/                              # H3 tessellation
    {study_area}_res{res}.parquet
  boundaries/                               # Study area geometry
    {study_area}_boundary.geojson
  target/{target_name}/                     # Ground truth targets
    {target_name}_h3res{res}_{year}.parquet
  accessibility/                            # Accessibility graphs
```

Old directories (`embeddings/`, `results [old 2024]/`, `analysis/`,
`data/preprocessed [TODO SORT & CLEAN UP]`) remain on disk. Migration is
data-side only (rename/symlink) -- done manually, not by code.

## API Reference

```python
from utils import StudyAreaPaths

paths = StudyAreaPaths("netherlands")

# Stage 1
paths.stage1("alphaearth")                          # -> .../stage1_unimodal/alphaearth/
paths.embedding_file("alphaearth", 10)               # -> ...alphaearth/netherlands_res10_2022.parquet
paths.embedding_file("alphaearth", 10, year=2023)    # -> ...alphaearth/netherlands_res10_2023.parquet
paths.pca_embedding_file("alphaearth", 10, 16)       # -> ...alphaearth/netherlands_res10_pca16_2022.parquet
paths.intermediate("alphaearth")                     # -> ...alphaearth/intermediate/

# Stage 2
paths.stage2("lattice_unet_cones")                   # -> .../stage2_multimodal/lattice_unet_cones/
paths.checkpoints("lattice_unet_cones")              # -> .../checkpoints/
paths.model_embeddings("lattice_unet_cones")         # -> .../embeddings/
paths.plots("lattice_unet_cones")                    # -> .../plots/
paths.training_logs("lattice_unet_cones")            # -> .../training_logs/

# Stage 3
paths.stage3("linear_probe")                         # -> .../stage3_analysis/linear_probe/
paths.stage3("dnn_probe")                            # -> .../stage3_analysis/dnn_probe/
paths.stage3("clustering")                           # -> .../stage3_analysis/clustering/

# Shared / stage-independent
paths.regions()                                      # -> .../regions_gdf/
paths.region_file(10)                                # -> .../regions_gdf/netherlands_res10.parquet
paths.boundaries()                                   # -> .../boundaries/
paths.boundary_file()                                # -> .../boundaries/netherlands_boundary.geojson
paths.target("leefbaarometer")                       # -> .../target/leefbaarometer/
paths.target_file("leefbaarometer", 10, 2022)        # -> leefbaarometer_h3res10_2022.parquet
paths.cones()                                        # -> .../cones/
paths.cone_cache(5, 10)                              # -> .../cones/cone_cache_res5_to_10/
paths.cone_lookup(5, 10)                             # -> .../cones/parent_to_children_res5_to_10.pkl
paths.accessibility()                                # -> .../accessibility/
```

## Refactor Table

Every file in the three stage packages that has hardcoded paths, what it currently
does, and what it should do instead.

### Stage 1: `stage1_modalities/`

| File | Current pattern | Replacement |
|------|----------------|-------------|
| `alphaearth/processor.py:240` | `Path('data/study_areas/default/embeddings/intermediate/alphaearth')` | `paths.intermediate("alphaearth")` |
| `poi/processor.py:85` | `Path(config.get('intermediate_dir', 'data/study_areas/default/embeddings/intermediate/poi'))` | `paths.intermediate("poi")` |
| `poi/processor.py:336` | `config.get('output_dir', 'data/study_areas/default/embeddings/poi')` | `str(paths.stage1("poi"))` |
| `roads/processor.py:75` | `Path(config.get('intermediate_dir', 'data/study_areas/default/embeddings/intermediate/roads'))` | `paths.intermediate("roads")` |
| `roads/processor.py:395` | `config.get('output_dir', 'data/study_areas/default/embeddings/roads')` | `str(paths.stage1("roads"))` |

### Stage 2: `stage2_fusion/`

| File | Current pattern | Replacement |
|------|----------------|-------------|
| `pipeline.py:59` | `self.project_dir / 'results [old 2024]'` | `paths.stage2(model_name)` |
| `pipeline.py:60` | `self.project_dir / 'data' / 'preprocessed [TODO SORT & CLEAN UP]'` | Remove or migrate to study-area-specific paths |
| `pipeline.py:61` | `self.project_dir / 'data' / 'embeddings'` | `paths.stage1(modality)` per modality |
| `pipeline.py:297-300` | `self.embeddings_dir / f'{filename}.parquet'` (4 hardcoded subpaths) | `paths.embedding_file(modality, res)` per modality |
| `pipeline.py:534` | `r"D:\Projects\UrbanRepML"` | `_find_project_root()` or remove |
| `pipeline.py:617,703` | `r"C:\Users\Bert Berkers\PycharmProjects\UrbanRepML"` | `_find_project_root()` or remove |
| `data/study_area_loader.py:30` | `base_path: str = "data/study_areas"` | Accept `StudyAreaPaths` or derive from it |
| `data/study_area_loader.py:50` | `'results [old 2024]': self.base_path / 'results [old 2024]'` | Remove; use `paths.stage2(model_name)` |
| `data/study_area_loader.py:49` | `'embeddings': self.base_path / 'embeddings'` | `paths.stage1(modality)` |
| `data/cone_dataset.py:98` | `data_dir = f"data/study_areas/{study_area}"` | `paths.root` |
| `data/hierarchical_cone_masking.py:906` | `f"data/study_areas/{study_area}/regions_gdf/{study_area}_res{res}.parquet"` | `paths.region_file(res)` |
| `data/multimodal_loader.py:263` | `output_dir: str = 'data/study_areas/default/stage2_fusion/multimodal'` | `paths.stage2("multimodal")` |
| `inference/hierarchical_cone_inference.py:121` | `f"data/study_areas/{self.study_area}/embeddings/alphaearth/..."` | `paths.embedding_file("alphaearth", res, year)` |
| `inference/hierarchical_cone_inference.py:552` | `"data/study_areas/netherlands/results [old 2024]/lattice_unet_cones/checkpoints/best.pth"` | `paths.checkpoints("lattice_unet_cones") / "best.pth"` |
| `inference/hierarchical_cone_inference.py:575` | `Path("data/study_areas/netherlands/results [old 2024]/lattice_unet_cones/inference")` | `paths.stage2("lattice_unet_cones") / "inference"` |

### Stage 3: `stage3_analysis/`

| File | Current pattern | Replacement |
|------|----------------|-------------|
| `linear_probe.py:82-83` | `f"data/study_areas/{self.study_area}/embeddings/alphaearth/..."` | `paths.embedding_file("alphaearth", self.h3_resolution, self.year)` |
| `linear_probe.py:87-88` | `f"data/study_areas/{self.study_area}/embeddings/alphaearth/..._pca16_..."` | `paths.pca_embedding_file("alphaearth", self.h3_resolution, 16, self.year)` |
| `linear_probe.py:92-93` | `f"data/study_areas/{self.study_area}/target/leefbaarometer/..."` | `paths.target_file("leefbaarometer", self.h3_resolution, self.year)` |
| `linear_probe.py:97` | `f"data/study_areas/{self.study_area}/analysis/linear_probe"` | `paths.stage3("linear_probe")` |
| `dnn_probe.py:113-114` | `f"data/study_areas/{self.study_area}/embeddings/alphaearth/..."` | `paths.embedding_file("alphaearth", self.h3_resolution, self.year)` |
| `dnn_probe.py:118-119` | `f"data/study_areas/{self.study_area}/target/leefbaarometer/..."` | `paths.target_file("leefbaarometer", self.h3_resolution, self.year)` |
| `dnn_probe.py:123` | `f"data/study_areas/{self.study_area}/analysis/dnn_probe"` | `paths.stage3("dnn_probe")` |
| `analytics.py:65` | `self.output_dir / 'embeddings' / self.city_name` | `paths.model_embeddings(model_name)` |
| `analytics.py:273` | `self.output_dir / 'analysis' / self.city_name / 'cluster_statistics.parquet'` | `paths.stage3("clustering") / 'cluster_statistics.parquet'` |
| `linear_probe_viz.py:760` | `self.output_dir.parent.parent / "embeddings" / "alphaearth" / "netherlands_res10_2022.parquet"` | `paths.embedding_file("alphaearth", 10, 2022)` |
| `linear_probe_viz.py:820` | `self.output_dir.parent.parent.parent / "boundaries" / "netherlands_boundary.geojson"` | `paths.boundary_file()` |
| `leefbaarometer_target.py:55` | `f"data/study_areas/{self.study_area}/target/leefbaarometer/..."` (scores CSV) | `paths.target("leefbaarometer") / "open-data-.../...csv"` |
| `leefbaarometer_target.py:60` | `f"data/study_areas/{self.study_area}/target/leefbaarometer/..."` (grid gpkg) | `paths.target("leefbaarometer") / "geometrie-.../...gpkg"` |
| `leefbaarometer_target.py:66` | `f"data/study_areas/{self.study_area}/target/leefbaarometer/..."` (output) | `paths.target_file("leefbaarometer", res, year)` |
| `leefbaarometer_target.py:71` | `f"data/study_areas/{self.study_area}/regions_gdf/..."` | `paths.region_file(self.h3_resolution)` |
| `validation.py:30` | `Path("data/processed")` | `paths.root` (or remove; this file may be stale) |
| `visualization/cluster_viz.py:70-91` | Hardcoded per-study-area `data_dir` and `output_dir` dicts | `paths.stage1("alphaearth")` and `paths.stage3("clustering")` |

## Alternatives Considered

1. **Config-file-only approach** (YAML with all paths): Rejected because it still
   requires every file to parse the same config, and path construction logic
   (interpolating study_area, resolution, year) would be duplicated in YAML templates.

2. **Environment variables** (`URBANREPML_DATA_DIR`): Complementary, not sufficient.
   The class can respect an env var for `project_root`, but the directory structure
   convention must be encoded in code.

3. **Keep paths in each stage's own module**: Violates DRY. The same pattern
   (`data/study_areas/{sa}/embeddings/alphaearth/{sa}_res{r}_{y}.parquet`) appears
   in stage1, stage2, and stage3 code.

## Consequences

- **Positive**: Single source of truth for all data paths. New study areas
  automatically get correct directory structure. Path changes require editing one file.
- **Positive**: Eliminates absolute Windows paths from source code.
- **Negative**: All three stage packages gain a dependency on `utils.paths`.
  This is acceptable because `utils/` is a leaf dependency with no imports from
  stage packages.
- **Neutral**: Old data directories on disk are not renamed. A manual migration
  (symlinks or renames) is needed for existing study areas. The spec does not
  prescribe when this migration happens.

## Implementation Notes

1. Create `utils/__init__.py` and `utils/paths.py` (this spec's Step 3).
2. Wave 2 agents refactor their owned files per the refactor table:
   - **stage1-modality-encoder**: `alphaearth/processor.py`, `poi/processor.py`,
     `roads/processor.py`
   - **stage2-fusion-architect**: `pipeline.py`, `study_area_loader.py`,
     `cone_dataset.py`, `hierarchical_cone_masking.py`, `multimodal_loader.py`,
     `hierarchical_cone_inference.py`
   - **stage3-analyst**: `linear_probe.py`, `dnn_probe.py`, `analytics.py`,
     `linear_probe_viz.py`, `leefbaarometer_target.py`, `cluster_viz.py`,
     `validation.py`
3. Each refactored file adds `from utils import StudyAreaPaths` and constructs
   a `StudyAreaPaths` instance from the study area name (which is always available
   in config or as a parameter).
4. The `StudyAreaPaths` class does NOT create directories. Callers use
   `path.mkdir(parents=True, exist_ok=True)` as needed. This keeps the class
   pure (no side effects on import or construction).