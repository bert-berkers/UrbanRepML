# Script Hygiene Conventions

## Status: Draft

## Context

The `scripts/` directory has grown to 65 Python files across 12 subdirectories with significant bloat:

- **7 roads scripts** across `one_off/` and `processing_modalities/POI/` doing variations of the same thing (e.g., `run_roads_processor.py`, `run_roads_processor_online.py`, `run_roads_full_netherlands.py`, `run_roads_full_netherlands_no_progress.py`, `run_roads_duckdb_direct.py`, `generate_roads_netherlands.py`, `generate_roads_from_intermediate.py`).
- **6 plot scripts** scattered between top-level `scripts/`, `one_off/`, `archive/visualization/`, and `visualization/` with unclear promotion/demotion status.
- **6 superseded training scripts** in `scripts/netherlands/` (e.g., `run_experiment.py` dates to 2024, `run_hexagonal.py` and `run_netherlands_lattice_unet.py` are pre-cone-batching).
- **1 test file in scripts/** (`scripts/processing_modalities/multimodal integration/test_multimodal_pipeline.py`) and at least 2 debug/test scripts in `scripts/netherlands/` (`test_cone_forward_pass.py`, `debug_cone_data.py`).
- **A directory with a space** (`scripts/processing_modalities/multimodal integration/`) which breaks shell commands and imports.

An `archive/` directory already exists with a reasonable subcategory structure (`benchmarks/`, `legacy/`, `utilities/`, `visualization/`), and `one_off/` exists but has no expiry enforcement. The conventions below formalize what is partially in place and add the missing guardrails.

## Decision

### Three-Tier Organization

```
scripts/
  {domain}/              # Durable scripts, organized by domain
  one_off/               # Temporary scripts, 30-day shelf life
  archive/               # Historical scripts, read-only reference
    benchmarks/
    legacy/
    visualization/
    utilities/
    roads/               # (new, for the roads script consolidation)
```

**Tier 1 -- Durable (`scripts/{domain}/`)**: Scripts that are part of the active project workflow. Referenced from CLAUDE.md Key Commands or used regularly. These must use `StudyAreaPaths` for path construction and have a module docstring.

**Tier 2 -- One-off (`scripts/one_off/`)**: Temporary scripts for one-time tasks (data migration, debug sessions, exploratory plotting). Expected to be archived or deleted within 30 days. Must have a module docstring that states `Lifetime: temporary`.

**Tier 3 -- Archive (`scripts/archive/{category}/`)**: Scripts that were once useful but are no longer active. Kept for reference only. Subcategorized by purpose. These are read-only in spirit: do not modify archived scripts, copy them out if you need a variant.

### New-Script Gate

Every new script must include a module docstring with at minimum:

```python
"""
Brief description of what this script does.

Lifetime: durable | temporary
Stage: stage1 | stage2 | stage3 | preprocessing | tools
"""
```

- `Lifetime` declares the script's expected tier. `temporary` scripts go in `one_off/`. `durable` scripts go in the appropriate domain directory.
- `Stage` indicates which part of the pipeline this script exercises. This helps future cleanup: a Stage 2 training script that references a deleted model class is clearly dead.

All new scripts must use `utils.paths.StudyAreaPaths` for path construction. Hardcoded paths like `Path("data/study_areas/netherlands/...")` are prohibited. This is an extension of the existing DATA-CODE SEPARATION principle to scripts.

### Periodic Cleanup Rule

The coordinator's Final Wave should check `scripts/one_off/` for scripts older than 30 days (by git commit date, not filesystem mtime). Candidates are either:

1. **Promoted** to a durable domain directory (if still actively used).
2. **Archived** to `scripts/archive/{category}/` (if historically useful).
3. **Deleted** (if purely ephemeral and no longer informative).

The coordinator reports the count and action taken in the OODA report. If no scripts are stale, no action needed.

### No Tests in scripts/

Test files belong in `tests/`. The existing `scripts/processing_modalities/multimodal integration/test_multimodal_pipeline.py` is the canonical anti-pattern. Debug scripts (like `debug_cone_data.py`) that are really one-time diagnostic sessions belong in `one_off/`, not in domain directories alongside production scripts.

### Directory Naming

No spaces in directory names. `scripts/processing_modalities/multimodal integration/` should be renamed to `scripts/processing_modalities/multimodal_integration/` or removed entirely if the sole occupant is the misplaced test file.

## Alternatives Considered

1. **Flat `scripts/` with naming prefixes** (e.g., `oneoff_roads_debug.py`, `archive_old_trainer.py`): Rejected. Name-based organization does not scale and makes `ls` output noisy. Directory-based tiers are clearer and allow `.gitignore` or cleanup tooling to target `one_off/` specifically.

2. **Automatic archival via git hooks**: Rejected. A git pre-commit hook that checks file ages adds complexity and would need to handle exceptions (e.g., a script in `one_off/` that has been touched recently but is still temporary). Manual coordinator review during Final Wave is lower-overhead and more accurate.

3. **Deleting all archived scripts**: Rejected. Some archived scripts contain useful patterns (e.g., the non-cone training script shows the pre-cone architecture). Keeping them in `archive/` costs nothing and aids understanding.

4. **Strict enforcement via linter**: Rejected at this stage. A custom lint rule to check for module docstrings and `StudyAreaPaths` usage would be ideal long-term but is over-engineered for a single-developer research project. The convention is enforced socially (agent instructions, CLAUDE.md).

## Consequences

### Positive

- Clear expectation for every script's lifecycle. New contributors (and agents) know where to put things.
- The 30-day one_off rule prevents the unbounded accumulation that led to the current 65-script state.
- `StudyAreaPaths` enforcement eliminates a class of bugs where scripts break when study area directory structure changes.
- Tests in `tests/` means `pytest` discovers them automatically. Tests hidden in `scripts/` are invisible to CI.

### Negative

- Initial cleanup effort: ~20 scripts need to be moved or deleted. This is a one-time cost.
- The docstring gate adds friction to creating quick throwaway scripts. This is intentional: the friction is the point. If a script is too small to deserve a 3-line docstring, it should be a REPL session, not a committed file.

### Neutral

- The `archive/` directory will grow over time. This is acceptable -- archived scripts are inert and do not affect import paths or tooling.
- The 30-day threshold is arbitrary. Some temporary scripts may deserve longer; the coordinator can exercise judgment during cleanup.

## Implementation Notes

### Ordering

1. **Write CLAUDE.md section** (this spec provides the exact text below).
2. **Move misplaced test**: `scripts/processing_modalities/multimodal integration/test_multimodal_pipeline.py` to `tests/` (or delete if broken beyond repair).
3. **Rename spaced directory**: `multimodal integration/` to `multimodal_integration/` or remove if empty after step 2.
4. **Triage current scripts** (this is the bulk of the work):
   - Roads scripts: keep the best one in `one_off/` if still needed, archive the rest to `archive/roads/`.
   - Superseded training scripts: `run_experiment.py`, `run_experiment_fsi95.py`, `run_hexagonal.py`, `run_netherlands_lattice_unet.py` to `archive/legacy/`.
   - Debug scripts: `test_cone_forward_pass.py`, `debug_cone_data.py` to `one_off/` (or `tests/` if they have assertions).
   - Plot one-offs in top-level `scripts/`: `regenerate_rgb_top3.py` to `one_off/` or `archive/visualization/`.
5. **Add docstrings** to any promoted durable scripts that lack them.
6. **Update CLAUDE.md Key Commands** if any referenced scripts moved.

### Dependencies

- Depends on nothing. This is a pure organizational change with no code behavior impact.
- Should happen BEFORE new modality scripts are written (POI embeddings, GTFS) to prevent new scripts from following the old unstructured pattern.

### Initial Triage Classification

Based on the current 65 scripts, here is a preliminary classification:

**Keep as durable (scripts/{domain}/):**
- `scripts/tools/create_study_area.py` -- active CLI tool
- `scripts/tools/list_study_areas.py` -- active CLI tool
- `scripts/accessibility/*.py` (3 scripts) -- active pipeline step
- `scripts/preprocessing_auxiliary_data/*.py` (6 scripts) -- active preprocessing
- `scripts/plot_embeddings.py` -- active EDA tool (recently refactored)
- `scripts/plot_linear_probe.py` -- active analysis plotting
- `scripts/plot_targets.py` -- active target visualization
- `scripts/compare_probes.py` -- active analysis
- `scripts/netherlands/train_lattice_unet_res10_cones.py` -- active training
- `scripts/netherlands/train_cone_alphaearth.py` -- active training
- `scripts/netherlands/infer_cone_alphaearth.py` -- active inference
- `scripts/netherlands/apply_pca_alphaearth.py` -- active preprocessing
- `scripts/processing_modalities/alphaearth/*.py` (7 scripts) -- active stage1
- `scripts/processing_modalities/POI/generate_netherlands_embeddings.py` -- active stage1
- `scripts/alphaearth_earthengine_retrieval/*.py` (3 scripts) -- active data retrieval
- `scripts/visualization/visualize_clusters.py` -- active analysis
- `scripts/analysis/validate_embeddings.py` -- active analysis

**Archive (scripts/archive/):**
- `scripts/netherlands/run_experiment.py` -> `archive/legacy/` (2024-era, superseded)
- `scripts/netherlands/run_experiment_fsi95.py` -> `archive/legacy/` (superseded)
- `scripts/netherlands/run_hexagonal.py` -> `archive/legacy/` (pre-cone)
- `scripts/netherlands/run_netherlands_lattice_unet.py` -> `archive/legacy/` (pre-cone)
- `scripts/processing_modalities/POI/generate_roads_netherlands.py` -> `archive/roads/`
- `scripts/processing_modalities/POI/generate_roads_from_intermediate.py` -> `archive/roads/`
- `scripts/regenerate_rgb_top3.py` -> `archive/visualization/`

**Move to one_off/ (already there or should be):**
- 5 roads scripts already in `one_off/` -- archive all but `run_roads_full_netherlands.py` (the most complete)
- 3 plot scripts already in `one_off/` -- keep, subject to 30-day rule
- `scripts/one_off/dnn_probe_sweep.py` -- keep (actively deferred, will be re-run)
- `scripts/one_off/bridge_archive_embeddings.py` -- keep (data migration utility)
- `scripts/one_off/run_poi_full_netherlands.py` -- keep (active POI work)

**Move to tests/ or delete:**
- `scripts/processing_modalities/multimodal integration/test_multimodal_pipeline.py` -> `tests/` or delete
- `scripts/netherlands/test_cone_forward_pass.py` -> `tests/` or `one_off/`
- `scripts/netherlands/debug_cone_data.py` -> `one_off/`
- `scripts/processing_modalities/alphaearth/test_res10_setup.py` -> `tests/` or `one_off/`

## CLAUDE.md Addition

The following text should be inserted as a new section in CLAUDE.md, after "Utility Infrastructure" and before "Key Commands". It matches the existing concise style.

```markdown
## Script Organization

Three tiers in `scripts/`:

- **`scripts/{domain}/`** — Durable scripts that are part of the active workflow. Must have a module docstring and use `StudyAreaPaths` for all paths.
- **`scripts/one_off/`** — Temporary scripts (debug, migration, one-time plots). 30-day shelf life; coordinator flags stale ones for archive or deletion.
- **`scripts/archive/{category}/`** — Historical scripts kept for reference. Read-only.

**Every new script** requires a module docstring stating its purpose, lifetime (`durable`/`temporary`), and which stage it exercises. No hardcoded `data/study_areas/...` paths.

**No tests in scripts/**. Tests go in `tests/`. Debug scripts go in `one_off/`.
```
