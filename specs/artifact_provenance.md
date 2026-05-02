# Artifact Provenance — Sidecars, Figure Provenance, and the Append-Only Ledger

## Status: Draft (frozen for cluster-2 Wave-2 implementation)

**Frame**: Plan `.claude/plans/2026-04-18-cluster2-ledger-sidecars.md` (W0 schema-freeze).
**Complements**: `specs/run_provenance.md` (`run_info.json` per-run-directory manifest). That spec describes the *coarser* per-run-dir layer; this one specifies the *finer* per-artifact sibling layer plus the cross-run ledger. Both layers coexist — a run directory can hold one `run_info.json` and many `*.run.yaml` artifact sidecars.

## Purpose

Every stage3 artifact leaves a machine-readable engram — a salience-preserving note that a future analyst-Selflet (human or agent) can re-interpret without archaeological dig through git, configs, and `latest_run()` tie-breakers. The ledger (`data/ledger/runs.jsonl`) is the stigmergic trace layer: past runs deposit signals, a future daemon folds them into views on demand. This moves cross-run analysis from **event-driven** (user opens a dir, grep-browses PNGs) to **rate-possible** (daemon-foldable any time) — outer-loop rate-enablement in the gyroscopic two-timescale sense, Levin-style mnemonic improvisation at data scale.

## Scope

**In scope (this plan, W2):**
- Stage3 probe outputs: `LinearProbeRegressor.save_results()`, `DNNProbeRegressor.save_results()`, `ClassificationProber.save_results()`, `DNNClassificationProber.save_results()`.
- Stage3 figure outputs: the four `*_viz.py` classes and `scripts/stage3/plot_cluster_maps.py`.

**Explicitly NOT in scope:**
- Stage1 retrofit (alphaearth/poi/roads/gtfs processor sidecars) — separate future plan.
- Stage2 retrofit (training-run + checkpoint provenance) — separate future plan, intersects with the `[open|42d]` checkpoint-versioning item from 2026-04-19 close-out.
- Backfill of legacy artifacts with minimal sidecars — separate future migration plan.
- Figure write sites outside `stage3_analysis/` and `scripts/stage3/` (accessibility plots, target plots, concat EDA, etc.) — large surface; bundle into stage1/2 future plans as they are retrofit.

## Sidecar file — `*.run.yaml`

For every data artifact written by an in-scope probe save, a sibling sidecar `{artifact_name}.run.yaml` is written alongside it. The sidecar is YAML (human-readable, git-diffable, survives binary-file opacity).

### Minimum fields (15)

| Field | Type | How produced | Example |
|-------|------|--------------|---------|
| `run_id` | string | `{stage}-{producer}-{started_at_compact}-{config_hash_short}` — see §run_id format | `stage3-linear_probe-20260424T154032-a3f1b2c7` |
| `git_commit` | string (40-char hex) | `git rev-parse HEAD` at `SidecarWriter.__enter__`. Null if not in a git repo. | `4fb0aa5c89f3b2a1d8e7...` |
| `git_dirty` | bool | `True` if `git status --porcelain` non-empty at enter-time, else `False`. Null if not in a git repo. | `false` |
| `config_hash` | string (16-char hex) | SHA-256 of canonical-JSON-serialised config dict; first 16 chars. See §config_hash algorithm. | `a3f1b2c78d4e5f60` |
| `config_path` | string (relative path) or null | Path to the on-disk config file that backed this run, if any; relative to project root. Null if config was constructed inline. | `configs/linear_probe_netherlands.yaml` |
| `input_paths` | list[string] | Relative paths of every input artifact the producer read. Producer declares these to the writer. | `["data/.../alphaearth/.../res10_2022.parquet", "data/.../target/leefbaarometer_res10_2022.parquet"]` |
| `output_paths` | list[string] | Relative paths of every output artifact the producer wrote. The sidecar is the sibling of output_paths[0] by convention. | `["data/.../stage3_analysis/linear_probe/.../metrics_summary.csv", "..."]` |
| `seed` | int or null | The random seed used (e.g. `LinearProbeConfig.random_state`). Null if the producer does not seed. | `42` |
| `wall_time_seconds` | float | `ended_at − started_at`. Measured by `SidecarWriter`, not by the caller. | `87.42` |
| `started_at` | string (ISO 8601 UTC) | `datetime.now(timezone.utc).isoformat(timespec='seconds')` at `__enter__`. | `2026-04-24T15:40:32+00:00` |
| `ended_at` | string (ISO 8601 UTC) | Same format, at `__exit__`. | `2026-04-24T15:41:59+00:00` |
| `producer_script` | string (relative path) | `sys.argv[0]` relativized to project root, or the `__file__` of the calling module if not invoked as a script. | `stage3_analysis/linear_probe.py` |
| `study_area` | string | The `study_area` field from the producer's config. | `netherlands` |
| `stage` | string (enum) | One of `stage1`, `stage2`, `stage3`, `figure`. For this plan always `stage3` (probes) or `figure` (viz). | `stage3` |
| `schema_version` | string (semver) | Schema version this sidecar was written against. Current: `"1.0"`. | `1.0` |

### `extra:` conventions

All domain-specific fields go under `extra:` so the 15-field minimum schema stays stable under evolution.

```yaml
extra:
  # Domain-specific, free-form. Examples:
  overall_r2: 0.535
  overall_rmse: 0.082
  n_folds: 5
  block_width_m: 10000
  target_cols: ["lbm", "fys", "onv", "soc", "vrz", "won"]
  n_features: 64
  n_regions: 131472
  status: success   # or: failed (see Fail-mode decisions §2)
  exception_class: null  # populated to str(type(e).__name__) on failure
```

Keys under `extra` are stable across versions *within* a producer — a probe's sidecar at schema_version 1.0 can add `extra.new_metric` without a schema bump. Producers document their own `extra.*` keys in their docstrings. Cross-producer queries should rely only on the 15 minimum fields; `extra.*` is advisory.

## Figure-provenance specialisation — `*.provenance.yaml`

Figures are derived artifacts — they consume one or more probe runs and a plot config. They need a narrower sidecar that points *back* at the source runs rather than re-declaring them.

### Fields

| Field | Type | Source | Example |
|-------|------|--------|---------|
| `source_runs` | list[string] | List of `run_id`s of every upstream probe run that contributed data to this figure. Required. | `["stage3-linear_probe-20260424T154032-a3f1b2c7"]` |
| `source_artifacts` | list[string] | Relative paths to the exact files read (e.g. `predictions_lbm.parquet`, `metrics_summary.csv`). Redundant with `source_runs` but useful for content-hash checks. | `["data/.../2026-04-24_default/metrics_summary.csv"]` |
| `plot_config` | dict | Free-form dict of plot settings (dpi, top_n, colormap, etc.). Stringify non-JSON-serialisable values with a warning. | `{dpi: 300, top_n: 20, colormap: "RdBu_r"}` |
| `git_commit` | string | Same as sidecar. | `4fb0aa5c89f3b2a1d8e7...` |
| `git_dirty` | bool | Same as sidecar. | `false` |
| `started_at` | string (ISO 8601 UTC) | Wrapping-context enter time. | `2026-04-24T15:42:10+00:00` |
| `ended_at` | string (ISO 8601 UTC) | Wrapping-context exit time. | `2026-04-24T15:42:14+00:00` |
| `producer_script` | string | Relative path of the `*_viz.py` or plotting script. | `stage3_analysis/linear_probe_viz.py` |
| `schema_version` | string | `"1.0"`. Minor bumps are compatible; major bumps require migration notes. | `1.0` |

**Which of the 15 base fields apply, which don't**: `git_commit`, `git_dirty`, `started_at`, `ended_at`, `producer_script`, `schema_version` are all retained. The following are intentionally *dropped*: `run_id` (figures don't get their own run_id — they live inside a producer's run_id via `source_runs`), `config_hash` / `config_path` (the plot_config dict replaces this), `input_paths` / `output_paths` (subsumed by `source_artifacts` + the sibling file), `seed` (figures are deterministic given sources), `wall_time_seconds` (rarely interesting for plots but MAY be added under `extra:` if desired), `study_area` / `stage` (inferable from source_runs; putting them here would risk drift).

## Ledger — `data/ledger/runs.jsonl`

### Format

JSON Lines. One row per completed run. Append-only. UTF-8. One flat object per line (no nested arrays at the top level). Source of truth.

### Why JSONL, not parquet or SQLite

Human-readable, git-diffable (though the file itself is gitignored — see "Location" below), trivial to `tail -f`, no schema migration on append, no lock contention under simple advisory file-locks for multi-terminal concurrent writes. A parquet view (`data/ledger/runs.parquet`) MAY be materialised on-demand as a derived artifact for pandas-fast reads over millions of rows — JSONL remains canonical. This matches the Levin-style split: fidelity-preserving JSONL source, salience-preserving parquet view.

### Location

`data/ledger/runs.jsonl` at project root. The `data/` directory is gitignored (data-code separation rule); the ledger is ephemeral runtime state per-machine. Cross-machine reproducibility flows through commit+sidecar pairs, not the ledger. Do **not** commit the ledger.

### Row schema

Each row is a flat JSON object with these fields projected from the sidecar:

```json
{
  "run_id": "...",
  "git_commit": "...",
  "git_dirty": false,
  "config_hash": "...",
  "config_path": "...",
  "seed": 42,
  "wall_time_seconds": 87.42,
  "started_at": "...",
  "ended_at": "...",
  "producer_script": "...",
  "study_area": "netherlands",
  "stage": "stage3",
  "schema_version": "1.0",
  "sidecar_path": "data/.../metrics_summary.csv.run.yaml"
}
```

**Projected from the 15 sidecar fields, minus `input_paths` and `output_paths`** (which stay in the sidecar — they can be arbitrarily long and would bloat the JSONL). Plus one added field: `sidecar_path` (relative path), so readers can chase back to full provenance. `extra.*` fields are **NOT** projected into the ledger — the ledger is a cross-run index, not a cross-run metric store. Queries that need `extra.*` open the pointed-at sidecar.

### Concurrency

`ledger_append` uses an advisory file-lock (`fcntl.flock` on POSIX; `msvcrt.locking` on Windows; wrap in a single helper that fails-open if the lock module is unavailable — log warn, proceed without lock, accept occasional line interleave over hard failure). Append-only means lock scope is minimal. No readers need the lock — JSONL tolerates partial trailing lines (the `read_ledger` reader skips malformed rows, see §Fail-mode decisions).

## Fail-mode decisions (LOAD-BEARING)

Three fail-mode questions must be resolved explicitly. Silent defaults would break invariants downstream.

### 1. Read side — `read_ledger` on malformed row

**Decision: skip + stderr warn, fail-open.**

The reader iterates the JSONL file, attempts `json.loads` per line, and on `JSONDecodeError` or missing-required-field writes a warning to `stderr` with the line number and skips that row. Partial trailing line at EOF (common if a writer was interrupted) is also skipped. Returns a DataFrame of the rows that parsed. Rationale: read paths are advisory — an analyst exploring "what happened this week" is better served by 99% of the data with a visible warning than by a hard crash. This is consistent with plan §W1.

### 2. Write side — `ledger_append` failure (disk full, lock contention, permission)

**Decision: RAISE.**

If `ledger_append` cannot write — disk full, lock contention beyond a short retry budget, permission denied, the `data/ledger/` directory not creatable — the function raises the underlying `OSError` / `IOError`. The probe run fails visibly.

*Rationale*: The coordinator's W4 invariant is `len(read_ledger()) == count(stage3 *.run.yaml from this session)`. A swallowed write would silently break that invariant and create ghost sidecars (on-disk `.run.yaml` files that the ledger doesn't know about). Raising forces the probe run to fail visibly; the sidecar itself is written *before* the ledger-append (write-sidecar-first, then ledger-append-or-raise), so a raised ledger-append leaves a sidecar-without-ledger-row on disk — detectable and fixable by the `scripts/one_off/audit_sidecar_coverage.py` script from W4.

The alternative (swallow) prioritises run completion over provenance integrity, which directly contradicts the plan's thesis that sidecars are mnemonic-improvisation infrastructure. If your run is worth the compute, it is worth a provenance row; if the ledger won't accept a row, something is deeply wrong and you want to know now, not at analysis time.

*Operational note for W1 implementation*: `ledger_append` should retry the file-lock a bounded number of times (say 3 retries at 100ms each) before raising, to tolerate benign multi-terminal contention. Beyond that, raise.

### 3. `SidecarWriter.__exit__` during exception

**Decision: YES — write the sidecar with `extra.status: "failed"` and `extra.exception_class: "<class name>"`, then re-raise.**

If the wrapped block raises, `__exit__` catches (via normal context-manager `exc_type` inspection), captures `ended_at` / `wall_time_seconds`, writes the sidecar with:
- `output_paths` set to whatever the producer had declared so far (may be empty list),
- `extra.status: "failed"`,
- `extra.exception_class: exc_type.__name__`,
- `extra.exception_message: str(exc_val)[:500]` (truncated to keep sidecars small),

then returns `False` to propagate the exception.

**Errata (2026-05-02)**: ledger-append does **NOT** fire on failed runs — the sidecar is written but the JSONL row is suppressed. The detectable signal is therefore *sidecar-without-row*, surfaced by the W4 audit script `scripts/one_off/audit_sidecar_coverage.py`. (Earlier draft language said "the ledger-append also fires" — that was incorrect; the implementation in `utils/provenance.py:SidecarWriter.__exit__` skips `ledger_append` when `exc_type is not None`. A failed run leaves a sidecar with `extra.status: "failed"` on disk but does not pollute the cross-run JSONL index.)

*Rationale*: A failed run is also a run whose provenance is worth capturing. Failed runs are the most interesting cases for outer-loop adjustment (rate-enablement of "figure out why this keeps crashing" is exactly the loop this plan enables). The alternative (skip sidecar on failure) loses the signal that a run was attempted; later analyses can't distinguish "never tried" from "tried and crashed". The small cost (one sidecar write during exception unwinding) is worth the completeness.

*Edge case*: if `__exit__` itself raises during sidecar write (e.g. disk full mid-failure), the original exception takes precedence — wrap the sidecar write in `try/except`, log a stderr warn, and re-raise the original. Do not suppress the user's real exception with a provenance IOError.

## `config_hash` algorithm

```python
# Canonical JSON serialisation:
#   - keys sorted (recursive)
#   - no whitespace other than the default item separator
#   - ensure_ascii=True (UTF-8 bytes are deterministic without escaping nuance)
#   - default=_coerce handles non-JSON-serialisable leaves
import hashlib, json

def compute_config_hash(cfg: dict) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, default=_coerce)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
```

`_coerce` handles non-JSON-serialisable leaves (`np.ndarray`, `Path`, `datetime`, custom dataclasses). The caller chooses between two strategies:
- **Stringify + warn** (default): call `repr(value)` on unknown types (NOT `str(value)` — see implementation note below), emit a `UserWarning` once per type seen. Pragmatic but non-deterministic across Python versions that change `repr()`.
- **Raise** (strict): raise `TypeError("config_hash: non-serialisable leaf")`. Recommended for production probes where determinism matters. W2 implementation should default to **raise** for the stage3 probes — a config that includes a non-serialisable leaf is a bug.

**Implementation note (2026-05-02 errata)**: `utils/provenance.py:_stringify_with_warn` uses `repr(value)`, not `str(value)`. This is the deliberate choice: `repr()` preserves type information and quote-marks-around-strings in the canonical JSON, so `'1'` (string) and `1` (int) hash differently even after coercion. `str()` would have collapsed them. The brittleness across Python versions that change `repr()` for a given type is accepted; if it bites, the strict-mode raise path is the correct response, not silently re-mapping to `str()`. `UserWarning` (not `print` to stderr) so callers can filter it via `warnings.filterwarnings`.

The 16-hex-char truncation is a brevity choice — collision probability is ~1 in 2⁶⁴, sufficient for a single-project research ledger. Callers that need a longer hash can re-compute the full 64-char SHA-256 from the same canonical form.

**Nested dicts/lists** are normalised recursively by `sort_keys=True` at every level. Lists retain order (order is semantic — `[1,2]` ≠ `[2,1]`). Tuples are coerced to lists by the JSON encoder.

## `run_id` format

```
{stage}-{producer}-{started_at_compact}-{config_hash_short}
```

- `stage`: `stage1|stage2|stage3|figure`
- `producer`: the producer script's module name without path or extension, e.g. `linear_probe`, `dnn_probe`, `classification_probe`, `dnn_classification_probe`. (`figure` stage may use the viz module name.)
- `started_at_compact`: `YYYYMMDDTHHMMSS` UTC — compact ISO 8601 without separators, 15 chars.
- `config_hash_short`: first 8 chars of `config_hash` (16-char field truncated further for run_id brevity).

Example: `stage3-linear_probe-20260424T154032-a3f1b2c7`

*Why this shape*: self-describing (you can tell the stage and producer from the ID alone), sortable (the timestamp is lexicographically orderable), collision-resistant (the hash prefix distinguishes two runs in the same second), and compatible with the existing `run_info.json` `run_id` which uses `YYYY-MM-DD_descriptor` — the two IDs can coexist because `run_info.json` is per-directory and `*.run.yaml` is per-artifact; a run directory's `run_info.json` can point at the per-artifact `run_id`s as a manifest.

## Schema versioning

Current: `schema_version: "1.0"`.

- **Minor bump (1.0 → 1.1)**: purely additive — a new optional field added to the minimum-15 set. Readers that only understand 1.0 can still parse 1.1 sidecars by ignoring the new field. Reader behaviour on unknown fields: ignore, do not warn (to avoid warning floods on old readers).
- **Major bump (1.0 → 2.0)**: semantics change — a field renamed, removed, or its type changed. Requires a migration note appended to this spec under a new `## Schema migration: 1.0 → 2.0` section. Readers SHOULD refuse to parse a schema they don't understand with a clear error pointing at this spec. The reader MAY be extended with version-specific parse branches if backward compatibility is needed.

`extra.*` is not version-scoped; producers are free to evolve their extra keys freely at any version.

## qaqc invariants (for W4)

Three invariants the W4 qaqc gate must verify:

1. **Artifact-sidecar sibling**: every in-scope stage3-probe-produced artifact file has a sibling `{artifact_name}.run.yaml` in the same directory. Every figure file has a sibling `{figure_name}.provenance.yaml`. Pre-W2 legacy artifacts are expected to fail this and that is fine (backfill is out of scope).

2. **Ledger-sidecar count equality**: `len(read_ledger())` equals the count of stage3 `*.run.yaml` files on disk for artifacts produced during the current session. (Session scoping is to avoid re-counting old ledger rows when computing this invariant at session-end.)

3. **Determinism**: same `config_hash` + same input files (by path + content hash, checked via mtime comparison as a cheap proxy) ⇒ same sidecar, modulo `wall_time_seconds`, `started_at`, `ended_at`, `run_id`, `git_dirty`, `git_commit`. If the caller re-runs with the same config and inputs, the `extra.*` metric values must match exactly (to the float precision the producer writes them at).

---

## Appendix A — Audit findings (W0 read-only)

### A.1 Probe save-path conventions (existing)

All four probes already use `StudyAreaPaths.stage3_run()` with a date-prefixed `run_id` (via `create_run_id(run_descriptor)`) and all four already call `write_run_info()` when `run_id is not None`. The flat-directory fallback (`paths.stage3(probe_name)`) is retained when `run_descriptor == ""`.

Per-probe output surfaces (inside the run directory):

| Probe | `save_results()` output files | File:line |
|-------|-------------------------------|-----------|
| `LinearProbeRegressor` | `metrics_summary.csv`, `coefficients.csv`, `predictions_{target}.parquet` (×6), `config.json`, `run_info.json` | `stage3_analysis/linear_probe.py:548-635` |
| `DNNProbeRegressor` | `metrics_summary.csv`, `predictions_{target}.parquet` (×6), `config.json`, `training_curves/{target}_fold{k}.json` (×~30), `run_info.json` | `stage3_analysis/dnn_probe.py:755-865` |
| `ClassificationProber` | `metrics_summary.csv`, `predictions_{target}.parquet` (×7 taxonomy levels), `config.json`, `run_info.json` | `stage3_analysis/classification_probe.py:423-493` |
| `DNNClassificationProber` | `metrics_summary.csv`, `predictions_{target}.parquet` (×7), `config.json`, `training_curves/{target}_fold{k}.json`, `run_info.json` | `stage3_analysis/dnn_classification_probe.py:690-802` |

*Inconsistency the W2 retrofit should normalise*: the two DNN probes save `training_curves/` as a subdirectory of training JSON files. Every file in that directory is a separate output artifact that should get its own `*.run.yaml` sibling *or* the entire directory should be treated as one bundled artifact with one sidecar at `training_curves.run.yaml`. **Recommendation for W2**: one sidecar per top-level file, and one sidecar for the `training_curves/` directory-as-artifact (named `training_curves.run.yaml` at the parent level). This matches user intent — `training_curves/` is semantically one thing.

*Pre-existing provenance*: `run_info.json` already captures `git_hash`, `created_at`, `study_area`, `config`, `upstream_runs`. The new `*.run.yaml` sidecars *do not replace* `run_info.json` — they add a finer per-artifact layer. The two coexist. `run_info.json` is the run-directory manifest; `*.run.yaml` is the per-artifact engram. A future consolidation could unify them at a major schema bump (2.0) but is out of scope here.

### A.2 Figure-write call sites (in scope for W2 retrofit)

**Stage3 viz modules** (the 4 that W2 retrofits):

| File | Line | Context |
|------|------|---------|
| `stage3_analysis/linear_probe_viz.py` | 130 | `plot_coefficient_bars` → `coefficients_{target}.png` |
| `stage3_analysis/linear_probe_viz.py` | 211 | `plot_coefficient_bars_faceted` |
| `stage3_analysis/linear_probe_viz.py` | 271 | `plot_coefficient_heatmap` |
| `stage3_analysis/linear_probe_viz.py` | 339 | `plot_scatter_predicted_vs_actual` |
| `stage3_analysis/linear_probe_viz.py` | 399 | `plot_spatial_residuals` |
| `stage3_analysis/linear_probe_viz.py` | 448 | (another per-target plot) |
| `stage3_analysis/linear_probe_viz.py` | 631 | `plot_rgb_top3_map` |
| `stage3_analysis/linear_probe_viz.py` | 1112 | `plot_cross_target_correlation` |
| `stage3_analysis/linear_probe_viz.py` | 1168 | `plot_metrics_comparison` |
| `stage3_analysis/dnn_probe_viz.py` | 184 | `plot_training_curves` |
| `stage3_analysis/dnn_probe_viz.py` | 204 | training-curves facet |
| `stage3_analysis/dnn_probe_viz.py` | 414 | scatter / residuals |
| `stage3_analysis/dnn_probe_viz.py` | 537 | spatial maps |
| `stage3_analysis/dnn_probe_viz.py` | 739 | comparison bars |
| `stage3_analysis/classification_viz.py` | 146 | per-target classification plot |
| `stage3_analysis/classification_viz.py` | 364 | confusion / accuracy degradation |
| `stage3_analysis/classification_viz.py` | 444 | spatial map |
| `stage3_analysis/classification_viz.py` | 540 | cross-level comparison |
| `stage3_analysis/classification_viz.py` | 696 | F1 per class |
| `stage3_analysis/dnn_classification_viz.py` | 131 | training curves |
| `stage3_analysis/dnn_classification_viz.py` | 151 | comparison |
| `scripts/stage3/plot_cluster_maps.py` | 346 | `clusters_k{k}.png` per K |

**Count**: 22 in-scope figure-write sites across the four `*_viz.py` modules + `plot_cluster_maps.py`. All use the same `fig.savefig(path, dpi=..., bbox_inches="tight", ...)` pattern — good news for W2, which can replace each with `save_figure(fig, path, sources=..., plot_config=...)` mechanically.

**Save-path convention inconsistency** the retrofit must handle: `linear_probe_viz.py` / `dnn_probe_viz.py` / `classification_viz.py` / `dnn_classification_viz.py` all write into `self.output_dir` (set by the caller to `{run_dir}/plots/`). `plot_cluster_maps.py` writes to a date-subdir under the user's `--output-dir`. Both fit the sibling-sidecar model (`{figure}.provenance.yaml` beside the PNG) without normalisation needed.

**Out-of-scope figure sites** (not retrofit in this plan, kept here as an index for future plans):

- `scripts/stage3/plot_accessibility_maps.py` (3 sites), `plot_accessibility_overview.py` (5), `plot_causal_emergence*.py` (3), `plot_cluster_comparison.py` (2), `plot_concat_embeddings.py` (4), `plot_gtfs_embeddings.py` (4), `plot_probe_comparison.py` (3), `plot_supervised_probe_viz.py` (3), `plot_targets.py` (20), `probe_multiscale_embeddings.py` (2), `run_probe_comparison.py` (2), `map_causal_emergence.py` (2), `dnn_probe_sweep.py` (2)
- `stage3_analysis/cluster_comparison_plotter.py` (1), `comparison_plotter.py` (1), `analytics.py` (1), `hierarchical_visualization.py` (2)

Total out-of-scope figure sites: ~60, for a future plan.

### A.3 Surprises / notes for implementers

- **`write_run_info` already captures `git_hash`** (short form, 7-char). The new sidecar uses 40-char `git_commit` for full reproducibility — the two are compatible (short is a prefix of long). W1 `SidecarWriter` should call `git rev-parse HEAD` (full hash) and ALSO call `git status --porcelain` for the `git_dirty` bool.
- **Probes already compute a `run_id`** via `create_run_id(run_descriptor)` — format `YYYY-MM-DD_descriptor`. Our new `run_id` format is richer (stage + producer + compact timestamp + config hash). The two can coexist: the old `run_id` names the run *directory*, the new `run_id` names the *run within a sidecar*. W1 should not attempt to unify them.
- **All 4 probes use identical data-loading + spatial-blocking code** (same `load_and_join_data` + `create_spatial_blocks` pattern, nearly line-for-line). A W2 retrofit touching all 4 save paths is a natural moment to consider extraction of shared base logic — but that is out of scope for this cluster; flag as `[open|0d]` for a future refactor plan.
- **No existing config-hashing**: the probes' `config.json` output records the config dict but does not hash it. W1's `compute_config_hash` is new functionality; W2 wires it in.
- **`specs/run_provenance.md` describes `run_info.json`** as the Draft forward design for per-run-dir manifests. It is listed as "Active (partial)" because the retro-filled Checkpoint Index section IS live. The new per-artifact sidecar layer in this spec is a separate, orthogonal mechanism — not a replacement — and W3b of the plan should reference BOTH specs in CLAUDE.md §Utility Infrastructure when `utils/provenance.py` lands.
