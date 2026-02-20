---
paths:
  - "stage1_modalities/**"
  - "stage2_fusion/**"
  - "stage3_analysis/**"
  - "scripts/**"
  - "utils/**"
---

# Index Naming Contracts

## The Standard

- **`region_id`** is the SRAI convention and the project standard
- H3Regionalizer produces `GeoDataFrame` with `region_id` as index containing H3 hex strings
- ALWAYS work with `region_id` in Stage 2+ code

## Stage Boundary Convention

- **Stage 1 parquet outputs** use `h3_index` as column name (backwards compatibility with existing saved data)
- **Stage 2+** uses `region_id` internally (SRAI convention)
- The `MultiModalLoader` bridges this by normalizing column names at the stage1->stage2 boundary

## When Writing New Code

- New Stage 1 processors: output `h3_index` column (until migration to `region_id` is complete)
- New Stage 2+ code: expect `region_id` as index name
- Never create a third naming convention
- If you encounter both names in a file, the bridge code in `MultiModalLoader` is the canonical translation point
