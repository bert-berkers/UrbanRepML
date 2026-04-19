# Run-Level Provenance

## Status: Active (partial)
> The forward-looking design (directory layout, `run_info.json` schema, `utils/paths.py` API) remains **Draft** — not yet implemented. This is the already-designed answer to MEMORY.md P0 #7 (checkpoint versioning); intersects with Theme B (experiment ledger) in the 2026-04-18 organizational flywheel audit.
>
> The retro-filled **Checkpoint Index** at the bottom of this doc is **Active**: it is the authoritative pointer table for existing Stage 2 UNet checkpoints, added 2026-04-19 as part of the Q8 probe-confound resolution (Terminal C, `twilight-branching-tide`). Until the full `run_info.json` mechanism ships, this hand-curated index is the single source of truth for disambiguating which checkpoint backs which report.

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

## Checkpoint Index (Stage 2 UNet, retro-filled 2026-04-19)

Hand-curated index of checkpoints currently on disk at
`data/study_areas/netherlands/stage2_multimodal/unet/checkpoints/`. Added to
resolve the 74D-checkpoint ambiguity flagged in the 2026-03-29 probe-confound
report (Q8 ternary; Terminal C Wave 3). Rows capture only what is verifiable
from filename, directory listing, and citing reports — no fields are guessed
from model internals. When the forward-looking `run_info.json` mechanism
ships, each row should migrate into a per-run manifest and this table becomes
a redirect list.

| Filename | Date | Input dims | Training data (year) | Citing reports | Notes |
|----------|------|-----------|---------------------|----------------|-------|
| `best_model_2022_74D_2026-03-21.pt` | 2026-03-21 | 74D (AE 64 + Roads 10) | 2022 | `reports/2026-03-21-accessibility-unet-probe-results.md` (§Data line 87) | Accessibility UNet (walk edges at res9, uniform 1-ring at res8/7); 1000 ep; wandb run `mt10aget`. Decreasing dim pyramid [74→37→18]. |
| `best_model_2022_74D_2026-03-22.pt` | 2026-03-22 | 74D (AE 64 + Roads 10) | 2022 | `reports/2026-03-29-ring-agg-plus-unet-probe-comparison.md` (§Data, "unet_supervised_multiscale") | Supervised multiscale variant referenced as "UNet-MS 192D" in the 03-29 report (mean R²=0.574). Relation to the 03-21 checkpoint (re-train vs. later epoch vs. different head) not recorded — no wandb link captured in reports. |
| `best_model_2022_64D_2026-03-21.pt` | 2026-03-21 | 64D (AE only) | 2022 | — | AE-only 64D variant; not cited in the current probe-confound reports but kept for baseline comparisons. |
| `best_model_20mix_64D_2026-03-14.pt` | 2026-03-14 | 64D | 20mix (multi-year blend, see MEMORY "Data Temporal Mismatch") | `reports/2026-03-14-unet-vs-concat-probe-comparison.md` | Pre-74D-era 208D/130K baseline run; retained for historical comparison. |
| `best_model.pt` | unknown | unknown | unknown | `reports/2026-03-08-causal-emergence-phase1.md` (§Method, "247K × 128" shape) | **Unversioned — do not rely on.** Per MEMORY.md P0 #3 this file has been overwritten at least once; the 03-08 CE phase 1 report references the 128D 3-modality era weights which are no longer recoverable from this path. |

### Known provenance gaps

- The 2026-03-22 74D checkpoint has no `run_info.json` sidecar and no wandb
  link in the 03-29 report. Disambiguating its relation to the 2026-03-21 74D
  checkpoint (re-train from scratch? resumed epoch? different supervised
  head?) currently requires a human who was present at training time.
- `best_model.pt` is an unversioned artifact that has been overwritten without
  backup; the 03-08 CE story cannot be reproduced from the current tree and
  is a candidate for either a 74D rerun (Q8 item 5a, deferred) or a
  "superseded by future rerun" footer on the 03-08 report.
- This table should be regenerated (not edited) once `run_info.json` manifests
  exist for each checkpoint.