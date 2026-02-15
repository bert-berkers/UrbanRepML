# Restructure into Explicit 3-Stage Pipeline

## Context

The project has a natural 3-stage pipeline (modality encoding -> fusion -> analysis), but the current package structure doesn't reflect it cleanly:

- **Stage 3 logic scattered**: `urban_embedding/analysis/` (3 files), `modalities/alphaearth/visualize_clusters_simple.py`, `scripts/visualization/` (3 files), `scripts/analysis/` (1 file)
- **Stage 2 package overstuffed**: `urban_embedding/` contains both fusion logic AND downstream analysis
- **Scripts organized by study area** (`scripts/netherlands/`) rather than by pipeline stage
- Package names don't communicate the pipeline order

Goal: rename packages to make the 3-stage pipeline explicit, move misplaced code, and reorganize scripts by stage.

---

## Package Renames

### Current -> New

| Current | New | Role |
|---------|-----|------|
| `modalities/` | `stage1_modalities/` | Individual modality encoders -> H3 embeddings |
| `urban_embedding/` (minus analysis/) | `stage2_fusion/` | Multi-modal fusion -> urban embeddings |
| *(new)* | `stage3_analysis/` | Clustering, visualization, regression, evaluation |
| `study_areas/` | *(keep as-is)* | Config package, not a pipeline stage |

### Import change examples

```python
# Before
from stage1_modalities.alphaearth.processor import AlphaEarthProcessor
from stage2_fusion.models.cone_batching_unet import ConeBatchingUNet
from stage2_fusion.analysis.analytics import UrbanEmbeddingAnalyzer

# After
from stage1_modalities.alphaearth.processor import AlphaEarthProcessor
from stage2_fusion.models.cone_batching_unet import ConeBatchingUNet
from stage3_analysis.analytics import UrbanEmbeddingAnalyzer
```

---

## File Moves

### stage1_modalities/ (rename modalities/)
- Rename directory `modalities/` -> `stage1_modalities/`
- **Delete** `stage1_modalities/alphaearth/visualize_clusters_simple.py` (hardcoded paths, superseded by scripts/visualization/ and stage3_analysis)
- Update internal imports: `from modalities.base` -> `from stage1_modalities.base` in:
  - `stage1_modalities/__init__.py`
  - `stage1_modalities/poi/processor.py:48`
  - `stage1_modalities/roads/processor.py:37`
  - `stage1_modalities/aerial_imagery/processor.py:28`

### stage2_fusion/ (rename urban_embedding/, remove analysis/)
- Rename directory `urban_embedding/` -> `stage2_fusion/`
- **Remove** `stage2_fusion/analysis/` (moved to stage3_analysis)
- Update `stage2_fusion/__init__.py`: remove `UrbanEmbeddingAnalyzer` from `__all__`
- No internal cross-references to `urban_embedding.analysis` exist (verified via grep)

Subpackages that stay:
```
stage2_fusion/
├── __init__.py
├── pipeline.py
├── models/          (full_area_unet, cone_batching_unet, accessibility_unet)
├── data/            (cone_dataset, multimodal_loader, hierarchical_cone_masking,
│                     feature_processing, study_area_loader, study_area_filter, spatial_batching)
├── graphs/          (graph_construction, hexagonal_graph_constructor)
├── geometry/        (h3_geometry)
├── training/        (unified_trainer)
├── inference/       (hierarchical_cone_inference)
├── losses/          (cone_losses)
└── utils/           (validation)
```

### stage3_analysis/ (new package)
Create from scattered analysis code:
```
stage3_analysis/
├── __init__.py                          # NEW
├── analytics.py                         # FROM urban_embedding/analysis/analytics.py
├── hierarchical_cluster_analysis.py     # FROM urban_embedding/analysis/hierarchical_cluster_analysis.py
├── hierarchical_visualization.py        # FROM urban_embedding/analysis/hierarchical_visualization.py
```

Update internal imports in moved files: none needed (they don't import from `urban_embedding.*`)

---

## Scripts Reorganization

### Current -> New

```
scripts/
├── alphaearth_earthengine_retrieval/    ->  stage0_data_acquisition/earthengine/
├── processing_modalities/              ->  stage1_encoding/
│   ├── alphaearth/                         ├── alphaearth/
│   ├── POI/                                ├── poi/  (lowercase)
│   └── multimodal integration/             └── (delete or move to stage2)
├── preprocessing_auxiliary_data/        ->  stage1_encoding/preprocessing/
├── accessibility/                      ->  stage1_encoding/accessibility/
├── netherlands/                        ->  stage2_training/
├── analysis/                           ->  stage3_evaluation/
├── visualization/                      ->  stage3_evaluation/visualization/
├── tools/                              ->  tools/  (keep)
├── archive/                            ->  archive/  (keep)
└── examples/ (top-level)               ->  examples/  (keep, update imports)
```

### Import updates in scripts (~25 files)

All `from modalities.*` -> `from stage1_modalities.*` (14 occurrences in scripts/)
All `from urban_embedding.*` -> `from stage2_fusion.*` (26 occurrences in scripts/)

---

## Config Updates

### pyproject.toml
```toml
[tool.hatch.build.targets.wheel]
packages = [
    "stage1_modalities",   # was "stage1_modalities"
    "stage2_fusion",       # was "stage2_fusion"
    "stage3_analysis",     # NEW
    "study_areas",
]

[project.scripts]
urbanrepml = "stage2_fusion.cli:main"   # was stage2_fusion.cli
```

### CLAUDE.md
- Update all code examples and import paths
- Update the architecture section to reflect 3-stage naming
- Update "Key Commands" section

### configs/netherlands_pipeline.yaml
- Check for any hardcoded package references

---

## Tests

### tests/test_geometry/test_h3_geometry.py
- Update: `from urban_embedding.geometry` -> `from stage2_fusion.geometry`

---

## Execution Order

1. **Rename `modalities/` -> `stage1_modalities/`** (git mv)
2. **Rename `urban_embedding/` -> `stage2_fusion/`** (git mv)
3. **Create `stage3_analysis/`** and move files from `stage2_fusion/analysis/`
4. **Delete** `stage1_modalities/alphaearth/visualize_clusters_simple.py`
5. **Update all internal imports** in the 3 packages
6. **Reorganize `scripts/`** directories (git mv)
7. **Update all script imports** (~40 import lines across ~25 files)
8. **Update `pyproject.toml`** package list + entry point
9. **Update `CLAUDE.md`** and `README.md` references
10. **Update `examples/`** imports
11. **Update `tests/`** imports
12. **Run `uv sync`** to rebuild package
13. **Verify** all imports work

---

## Verification

```bash
# Rebuild package
uv sync

# Stage 1 imports
uv run python -c "from stage1_modalities.base import ModalityProcessor; print('Stage 1 OK')"
uv run python -c "from stage1_modalities.alphaearth.processor import AlphaEarthProcessor; print('AlphaEarth OK')"

# Stage 2 imports
uv run python -c "from stage2_fusion.models.cone_batching_unet import ConeBatchingUNet; print('ConeBatchingUNet OK')"
uv run python -c "from stage2_fusion.models.accessibility_unet import AccessibilityUNet; print('AccessibilityUNet OK')"
uv run python -c "from stage2_fusion.data.cone_dataset import ConeDataset; print('ConeDataset OK')"

# Stage 3 imports
uv run python -c "from stage3_analysis.analytics import UrbanEmbeddingAnalyzer; print('Analyzer OK')"
uv run python -c "from stage3_analysis.hierarchical_cluster_analysis import *; print('HierCluster OK')"

# Tests
uv run python -m pytest stage3_analysis/ -v

# Verify no stale references remain
grep -r "from modalities" --include="*.py" .
grep -r "from urban_embedding" --include="*.py" .
# Both should return 0 results [old 2024]
```

---

## Risk Assessment

- **Blast radius**: ~40 import lines across ~25 Python files + docs
- **Git history**: `git mv` preserves file history
- **Reversibility**: Fully reversible via `git mv` back
- **Breaking**: Any external consumers of `modalities` or `urban_embedding` package names will break (but this is a single-developer project)
- **PyCharm**: Will need to re-index after renames; interpreter already set to uv .venv
