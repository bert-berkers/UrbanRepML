---
paths:
  - "stage1_modalities/**"
  - "stage2_fusion/**"
  - "stage3_analysis/**"
  - "scripts/**"
  - "utils/**"
---

# Data-Code Separation

## Absolute Boundary

- Code lives in `stage1_modalities/`, `stage2_fusion/`, `stage3_analysis/`, `scripts/`
- Data lives in `data/` (gitignored)
- NEVER write data files (parquet, pkl, csv, tiff) outside `data/`
- NEVER write code files inside `data/`

## Study-Area Based Processing

All processing is organized by study area. Each area at `data/study_areas/{area_name}/` contains:
- `area_gdf/` -- study area boundary (pairs with regions_gdf/)
- `regions_gdf/` -- H3 tessellation (produced by SRAI)
- `stage1_unimodal/` -- per-modality embeddings
- `stage2_multimodal/` -- fused results
- `stage3_analysis/` -- analysis outputs
- `target/` -- ground truth targets

## Rules

- Every processing script requires a `--study-area` parameter or equivalent
- Never hardcode paths to specific study areas in library code
- Config/paths should be parameterized, not embedded
