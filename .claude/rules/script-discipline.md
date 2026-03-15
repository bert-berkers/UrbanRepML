---
paths:
  - "scripts/**"
---

# Script Discipline

## No Duplication

Before creating a new script, search for existing scripts and `utils/` functions that overlap.
Extend or call existing code -- do not copy-paste. The rasterization helpers were duplicated 4+ times;
that pattern is banned.

## Shared Visualization

Use `utils/visualization.py` (`rasterize_continuous`, `plot_spatial_map`, `load_boundary`, etc.)
for all spatial plotting. Never inline these functions into scripts.

## Script Placement

| Directory | Purpose | Lifetime |
|-----------|---------|----------|
| `scripts/{domain}/` | Durable workflow scripts | Permanent |
| `scripts/one_off/` | Temporary debug/exploration | 30 days |
| `scripts/archive/{category}/` | Historical reference | Read-only |

One-off scripts MUST have a module docstring declaring `Lifetime: temporary` and an approximate
expiry date. The coordinator flags stale one-offs for archive or deletion.

## Every Script Must

1. Have a module docstring stating purpose and lifetime (`durable` or `temporary`)
2. Use `StudyAreaPaths` for all data paths -- no hardcoded `data/study_areas/...` strings
3. Write outputs to date-keyed subdirs (`YYYY-MM-DD/`) where applicable
4. NOT contain test code -- tests go in `tests/`
