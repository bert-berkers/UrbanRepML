# Archive Index

This file indexes archived material in the UrbanRepML repo.
Archived = superseded but preserved. Data is never deleted, only organized
(per `memory/feedback_no_delete_data.md`).

---

## Result Artifacts (data — preserve only)

### results/archive/2024/embeddings/
Moved 2026-04-19 from repo-root `results [old 2024]/` (gitignored, 125 MB).
Pre-hex2vec, pre-StudyAreaPaths era. AlphaEarth res8/res10 embeddings only.

- `netherlands/alphaearth_res10_clustered_2022.parquet` — Stage 3 clustered embeddings — 2026-04-19 — pre-hex2vec era
- `netherlands/alphaearth_res8_clustered_2022.parquet` — Stage 3 clustered embeddings — 2026-04-19 — pre-hex2vec era
- `south_holland/` — regional variant — 2026-04-19 — (see directory for contents)
- `south_holland_fsi99/` — regional variant — 2026-04-19 — (see directory for contents)

---

## Scripts (code — grouped by origin)

All under `scripts/archive/`. Git-tracked. Read-only.

### scripts/archive/legacy/ — 9 files, archived 2026-03-01
Pre-refactor pipeline scripts (generate_netherlands variants, run_experiment variants).
Superseded by study-area-based pipeline with `StudyAreaPaths`.

### scripts/archive/probes/ — 12 files, archived 2026-03-22
Stage 3 probe scripts superseded by `DNNProbeRegressor` and `DNNClassificationProber` refactor.
Old one-off probe runners replaced by `scripts/stage3/` durable scripts.

### scripts/archive/visualization/ — 8 files, archived 2026-03-08
Early cluster viz scripts (dissolve-based). Superseded by rasterized maps in
`scripts/plot_embeddings.py` and `scripts/stage3/plot_cluster_maps.py`.

### scripts/archive/roads/ — 5 files, archived 2026-03-01
Road processing scripts predating `RoadsProcessor` class and OSM PBF pipeline.

### scripts/archive/diagnostics/ — 1 file, archived 2026-03-15
Diagnostic/inspection scripts superseded by structured stage3 analysis.

### scripts/archive/debug/ — 2 files, archived 2026-03-01
Ad-hoc debug scripts. No longer referenced.

### scripts/archive/preprocessing/ — 3 files, archived 2026-03-01
Data preprocessing utilities predating `StudyAreaPaths` and SRAI-first conventions.

### scripts/archive/utilities/ — 2 files, archived 2026-02-14
General utility scripts superseded by `utils/` package.

### scripts/archive/benchmarks/ — 2 files, archived 2026-02-14
Early benchmarking scripts. (see directory for contents)

### scripts/archive/cone_alphaearth/ — 4 files, archived 2026-03-22
Cone-based AlphaEarth experiments predating `LazyConeBatcher` and `HierarchicalConeMaskingSystem`.

### scripts/archive/migration/ — 2 files, archived 2026-03-22
One-off migration scripts (data layout changes, path refactors). Run-once; preserved for reference.

### scripts/archive/one_off/ — 3 files, archived 2026-03-15
Temporary scripts that aged past the 30-day shelf life in `scripts/one_off/`.

---

*Future additions: append new entries here as more material is archived.*
