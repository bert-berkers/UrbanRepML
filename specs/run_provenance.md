# Run-Level Provenance

## Status: Draft

## Context

Every training run, embedding computation, or analysis execution currently writes
outputs to flat directories (e.g. `stage3_analysis/linear_probe/`). When an
experiment is re-run with different parameters, old outputs are silently
overwritten. There is no record of which git commit, config, or upstream data
produced a given result. This makes it impossible to reproduce or compare runs.

Run-level provenance adds a thin layer: each execution gets a dated run
directory containing a `run_info.json` manifest. No existing files move.

## Decision

### Run naming convention

Format: `YYYY-MM-DD_descriptor`

- Date prefix enables chronological sorting via `sorted()`.
- Descriptor is user-provided and describes the run (e.g. `res10`,
  `alphaearth_pca16`, `baseline_lr0001`).
- If descriptor is empty, the run ID is just `YYYY-MM-DD`.
- On collision, the caller appends `_2`, `_3`, etc. Collision handling is the
  caller's responsibility, not the `create_run_id` function's.

### `run_info.json` schema

```json
{
  "run_id": "2026-02-14_alphaearth_pca16",
  "stage": "stage3",
  "created_at": "2026-02-14T15:30:00",
  "git_hash": "3971847",
  "study_area": "netherlands",
  "config": {},
  "upstream_runs": {
    "stage1/alphaearth": "2026-02-14_res10",
    "stage2/cone_alphaearth": "2026-02-14_cones"
  }
}
```

Fields: `run_id` (string, matches directory name), `stage` (stage1|stage2|stage3),
`created_at` (ISO 8601), `git_hash` (short hash or null), `study_area` (string),
`config` (arbitrary dict -- caller decides what to capture), `upstream_runs`
(optional dict mapping stage/name keys to run IDs consumed by this run).

### Target directory structure

```
data/study_areas/{study_area}/
  stage1_unimodal/{modality}/{run_id}/
    run_info.json
    {sa}_res{res}_{year}.parquet
    intermediate/

  stage2_multimodal/{model_name}/{run_id}/
    run_info.json
    checkpoints/
    embeddings/
    training_logs/

  stage3_analysis/{analysis_type}/{run_id}/
    run_info.json
    metrics_summary.csv
    predictions_*.parquet
    plots/
```

Run directories nest inside the existing stage directories returned by
`StudyAreaPaths.stage1()`, `.stage2()`, `.stage3()`.

### Migration

No existing files are moved. Old flat outputs remain where they are. New runs
create dated subdirectories beside them. `latest_run()` scans only for
directories matching `YYYY-MM-DD*`, so old flat files are ignored.

### API reference (utils/paths.py additions)

```python
# New methods on StudyAreaPaths:
paths.stage1_run("alphaearth", "2026-02-14_res10")  # -> .../stage1_unimodal/alphaearth/2026-02-14_res10/
paths.stage2_run("lattice_unet", "2026-02-14_cones") # -> .../stage2_multimodal/lattice_unet/2026-02-14_cones/
paths.stage3_run("linear_probe", "2026-02-14_baseline") # -> .../stage3_analysis/linear_probe/2026-02-14_baseline/
paths.latest_run(paths.stage3("linear_probe"))       # -> Path to most recent run dir, or None
paths.create_run_id("alphaearth_pca16")              # -> "2026-02-14_alphaearth_pca16"
paths.create_run_id()                                # -> "2026-02-14"

# New standalone function:
from utils.paths import write_run_info
write_run_info(
    run_dir,
    stage="stage3",
    study_area="netherlands",
    config={"lr": 0.001, "epochs": 50},
    upstream_runs={"stage1/alphaearth": "2026-02-14_res10"},
)
# Creates run_dir, writes run_info.json, returns path to JSON file.
```

## Alternatives Considered

1. **MLflow / W&B**: External tracking servers add infrastructure complexity
   disproportionate to a single-developer research project. A JSON file per run
   captures enough provenance without dependencies.

2. **Database (SQLite)**: More queryable but harder to inspect, version-control,
   or move between machines. JSON files are human-readable and git-friendly.

3. **Collision handling in `create_run_id`**: Rejected. The function would need
   to know the target directory, coupling ID generation to filesystem state.
   Callers that need uniqueness can check and append suffixes.

## Consequences

- **Positive**: Every run is self-documenting. Git hash links results to code.
  Upstream run references enable lineage tracing across stages.
- **Positive**: `latest_run()` enables scripts to auto-discover the most recent
  output without hardcoding run IDs.
- **Negative**: `write_run_info` is the one place that creates directories,
  breaking the "class does NOT create directories" convention of StudyAreaPaths.
  This is acceptable because it is a standalone function, not a class method.
- **Neutral**: Old flat outputs and new run directories coexist. This is messy
  but avoids a risky migration step.

## Implementation Notes

1. Add methods and function to `utils/paths.py` (see API reference above).
2. Add imports: `json`, `subprocess`, `re`, `datetime.date`, `datetime.datetime`.
3. Adoption is incremental: scripts opt in by calling `create_run_id()` +
   `write_run_info()` when they want provenance. No existing code breaks.
4. First adopters should be `stage3_analysis/linear_probe.py` and
   `stage3_analysis/dnn_probe.py` since those are the most actively iterated.