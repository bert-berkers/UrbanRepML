# Cluster 2 — Ledger + Sidecars + Figure Provenance (IN PROGRESS, 2026-04-24)

| Field | Value |
|---|---|
| **Status** | IN PROGRESS — W0 shipped; executing under session `sunlit-blooming-ridge` |
| **Source** | `reports/2026-04-18-organizational-flywheel-audit.md` §5 roadmap items #9, #10, #11, #12 |
| **Parent plan** | `.claude/plans/2026-04-18-flywheel-gyroscope-fix.md` (cluster 1, executed `99099c0`) |
| **Cluster** | 2 — Territory (touches `utils/` + `stage3_analysis/` + `data/ledger/` + `.claude/rules/`) |
| **Depends on** | `2026-04-18-cluster2-paths-consolidation.md` ✅ shipped `dcac83d` |
| **Est** | ~4–6h (bounded single-session scope) |
| **Progress** | W0 ✅ (`specs/artifact_provenance.md` frozen, 22 in-scope figure sites catalogued, fail-modes resolved); W1 pending |

## Coordinator adjustments at execution start (2026-04-24 sunlit-blooming-ridge)

Plan structure held, with three surgical adjustments made after the Wave-0 `/niche` OODA read:

1. **W2 split into W2a ‖ W2b** (parallel). Probe `.save()` retrofit and figure-save retrofit share only the `utils/provenance.py` API and are otherwise independent; bundling them into a single sequential wave was sequential-disguised-as-one. Split matches the plan's own rate-enablement thesis.
2. **Write-side fail-mode promoted from "implementer's discretion" to schema decision** — now resolved in W0's spec (see `## Fail-mode decisions` below).
3. **Commit-per-wave-on-green** rather than end-of-session batch commit. A plan whose thesis is "enable the outer loop to run at a rate" should not itself ship as one monolithic commit — the commit log is part of the rate-enabled record.

Plus one pre-flight gate for W3: verify the `scratchpad/coordinator/notes.md` identity-drift failure-mode entry exists before W3a writes its "2026-04-24 partially resolved" subsection. If the anchor is missing, W3a writes the parent entry first.

## Frame (why a single Selflet should read the whole plan before diving in)

Sidecars and the ledger are **mnemonic-improvisation infrastructure at the data scale**. Each experiment run leaves a machine-readable note (`*.run.yaml` sidecar) for a future analyst-Selflet to re-interpret. The append-only ledger (`data/ledger/runs.jsonl`) is the stigmergic trace layer — a past run leaves a signal for a future analyst. Figure provenance (`save_figure(sources=)`) closes the loop: every plot knows which runs it depends on.

Gyroscopically: this moves cross-run analysis from **event-driven** (user opens a dir, grep-browses PNGs) to **rate-possible** (daemon folds `runs.jsonl` into a view any time). The outer loop's goal-reconfiguration over experimental results no longer requires archaeological digs. That is the load-bearing justification — not reproducibility per se, but *enabling the outer loop to run at a rate*.

## Scope decisions (resolving DRAFT's Next-session TODO)

1. **One plan with 5 waves, not four plans.** The four audit items (#9 sidecars, #10 ledger, #11 figure provenance, #12 config hashing) are semantically one system — config hashing underlies sidecars, ledger rows are sidecar summaries, figure provenance is a sidecar specialisation. Splitting would fork shared schema decisions across plans. Each wave is individually shippable (rate-driven within the cluster).
2. **Ledger format: JSONL.** `data/ledger/runs.jsonl`, append-only, one row per completed run. Human-readable, git-diffable, no lock contention for multi-terminal appends, trivial to tail/filter. SQLite is premature optimisation — revisit only if cross-run reads become slow. An optional `data/ledger/runs.parquet` can be materialised on-demand as a *view* (fidelity-preserving JSONL source; salience-preserving parquet view — matches Levin's salience/fidelity split).
3. **Prototype stage: stage3 probes first.** `LinearProbeRegressor` + `DNNProbeRegressor` have clean save boundaries, deterministic outputs, multiple instances to test the schema, immediate analytical payoff, and low blast radius. Stage1 (heterogeneous modality encoders) and stage2 (long training runs, checkpoint complexity) are retrofits for a later plan.
4. **Sidecar minimum fields:** `run_id`, `git_commit`, `git_dirty`, `config_hash`, `config_path`, `input_paths`, `output_paths`, `seed`, `wall_time_seconds`, `started_at`, `ended_at`, `producer_script`, `study_area`, `stage`, `schema_version`. Optional fields (Dirichlet, metric values, any domain-specific keys) go under `extra:` so the minimum schema stays stable.
5. **qaqc gates (W5):** (a) every new artifact path has a `*.run.yaml` sibling; (b) `len(read_ledger())` equals the count of on-disk `.run.yaml` files; (c) same `config_hash` + same inputs → same sidecar modulo `wall_time_seconds`.
6. **Complementary to `specs/run_provenance.md`** (discovered during W0 audit). The existing spec covers the coarser `run_info.json` per-run-directory manifest. Our per-artifact `*.run.yaml` sibling + cross-run ledger is a finer layer; they coexist at the same run. All 4 probes already call `write_run_info()` via `create_run_id(run_descriptor)` — our new `run_id` format is richer; old names the run dir, new names the sidecar. No unification attempted (would require a 2.0 schema bump).

## Fail-mode decisions (resolved in W0, documented in `specs/artifact_provenance.md`)

1. **Read side, `read_ledger` on malformed row** → skip + stderr warn, fail-open.
2. **Write side, `ledger_append` failure** → **raise** after bounded retry (3× at 100ms for benign multi-terminal lock contention). Rationale: swallowing breaks the W4 invariant `len(read_ledger()) == count(stage3 *.run.yaml)` and creates ghost sidecars. Write-sidecar-first-then-ledger-append means a raised `ledger_append` leaves a detectable sidecar-without-row for W4's audit script to flag — prioritises provenance integrity over run completion.
3. **`SidecarWriter.__exit__` during wrapped-code exception** → write partial sidecar with `extra.status: "failed"` + `extra.exception_class`, then re-raise the original exception. Rationale: failed runs are the highest-signal cases for outer-loop adjustment; losing their provenance contradicts the plan thesis. Edge case: if the sidecar write itself fails during exception, the original exception takes precedence.

## Wave structure

### W0 — Read-only audit + schema finalisation (spec-writer) ✅ SHIPPED 2026-04-24

- ✅ Audited `LinearProbeRegressor.save_results()` (linear_probe.py:548-635), `DNNProbeRegressor.save_results()` (dnn_probe.py:755-865), `ClassificationProber.save_results()` (classification_probe.py:423-493), `DNNClassificationProber.save_results()` (dnn_classification_probe.py:690-802).
- ✅ Audited 22 in-scope figure-write sites across 4 viz files + `plot_cluster_maps.py`; catalogued ~60 out-of-scope sites as a future-plan index.
- ✅ `specs/artifact_provenance.md` frozen (~280 lines): 15 minimum fields + `extra.*` conventions + figure-provenance `*.provenance.yaml` + JSONL ledger format + three fail-mode decisions + `config_hash` algorithm + `run_id` format + schema versioning + Appendix A audit findings.
- ✅ Librarian specs index (`.claude/scratchpad/librarian/codebase_graph/infrastructure.md`) updated with cross-link to complementary `specs/run_provenance.md`.
- **Surprises (for W1/W2 implementers):**
  - Pre-existing `specs/run_provenance.md` is complementary, not superseded — coarser per-run-dir layer coexists with our per-artifact sibling layer.
  - DNN probes write a `training_curves/` subdirectory of ~30 JSON files → produce ONE sidecar (`training_curves.run.yaml`) for the subdirectory, not 30 per-file sidecars.
  - All 4 probes have ~95% duplicated `load_and_join_data` + `create_spatial_blocks` code (~250 lines each). Out of scope here; flagged `[open|0d]` for a future `BaseProbe` refactor plan.
- **W0 scratchpad**: `.claude/scratchpad/spec-writer/2026-04-24-sunlit-blooming-ridge.md`.

### W1 — Core utility module (devops + qaqc in parallel)

Parallel: implementation (devops) and tests (qaqc) can run simultaneously once the spec is frozen. Test-writer pairs contract-first against the spec, not against the implementation.

- **devops — `utils/provenance.py`:**
  - `compute_config_hash(cfg: dict) -> str` — canonical JSON (sorted keys, UTF-8) + SHA-256 → first 16 hex chars (per spec §config_hash algorithm).
  - `SidecarWriter(artifact_path, config, inputs, producer_script=None, study_area=None, stage='stage3', extra=None)` — context manager. `__enter__` captures `started_at`, `git_commit`, `git_dirty`, computes `config_hash` and `run_id`. `__exit__` captures `ended_at`, `wall_time_seconds`, writes `{artifact_path}.run.yaml` sibling. On exception: writes partial sidecar with `extra.status="failed"` + `extra.exception_class`, then re-raises.
  - `ledger_append(sidecar_path: Path)` — reads the sidecar, projects 14 fields (15 minimum minus `input_paths` + minus `output_paths`, plus `sidecar_path`) to a JSONL row, appends to `data/ledger/runs.jsonl` with advisory file-lock. **Bounded retry** (3× at 100ms) on lock contention, then raises.
  - `read_ledger(path: Path = None) -> pd.DataFrame` — convenience reader; malformed row → skip + stderr warn (fail-open).
- **qaqc — `tests/utils/test_provenance.py`:** config-hash stability (dict ordering invariance, nested dict/list normalisation, non-serialisable coercion), sidecar round-trip (write → read YAML → field equality), `__exit__`-on-exception writes failed sidecar then re-raises, ledger append idempotence, read_ledger fail-open on malformed rows, run_id format assertion.
- **Acceptance:** all tests pass; `python -c "from utils.provenance import SidecarWriter, compute_config_hash, ledger_append, read_ledger"` imports clean; commit on green.

### W2 — Stage3 probe retrofit (split W2a ‖ W2b, both stage3-analyst)

Split into parallel sub-waves at coordinator-execution time. They share only the `utils/provenance.py` API (frozen in W1) and are otherwise independent.

**W2a — Probe save() sidecar wiring:**
- Wire `SidecarWriter` into `LinearProbeRegressor.save_results()`, `DNNProbeRegressor.save_results()`, `ClassificationProber.save_results()`, `DNNClassificationProber.save_results()`.
- DNN probes: produce ONE `training_curves.run.yaml` sidecar for the subdirectory-as-artifact (not 30 per-file sidecars).
- Call `ledger_append()` after sidecar write. Coexists with the existing `write_run_info()` calls — do not remove those.
- Acceptance: run one probe (e.g. `python -m stage3_analysis.linear_probe --study-area netherlands` if CLI exists, else ad-hoc); verify `.run.yaml` sibling + new row in `data/ledger/runs.jsonl`.

**W2b — Figure save wrapper + viz retrofit:**
- Add `stage3_analysis/save_figure.py`: `save_figure(fig, path: Path, sources: list[str], plot_config: dict = None)` — calls `fig.savefig` then writes `*.provenance.yaml` sibling per spec.
- Retrofit the 22 in-scope figure-write sites in `linear_probe_viz.py`, `dnn_probe_viz.py`, `classification_probe_viz.py` (confirmed filename: `classification_viz.py` per W0 audit), `dnn_classification_viz.py`, and `scripts/stage3/plot_cluster_maps.py` to call `save_figure()` instead of raw `fig.savefig()`. Source run_ids come from the probe sidecars loaded at plot time (probes must expose their `run_id` in results structure).
- Acceptance: re-run a probe's viz step; verify `.provenance.yaml` sibling exists next to each figure with `source_runs` populated.

**Integration acceptance (after both):** one full probe-plus-viz cycle produces sidecar + ledger row + figure provenance; commit on green.

### W3 — Gyroscopic governance updates (spec-writer, docs-only, small)

**Pre-flight gate:** Verify `scratchpad/coordinator/notes.md` "Failure Mode: Identity Tagging Drift" entry exists before writing the "2026-04-24 partially resolved" subsection. If the parent entry is missing, W3a writes the parent entry first (date-stamped 2026-04-19 per the plan's cross-link in W3a step 1) — otherwise the subsection has no anchor and the cross-link from the rule file dangles.


This wave is docs-only and semantically adjacent: the sidecar/ledger work *is* outer-loop rate-enablement, and the identity/supra-ghost items are outer-loop coherence. Bundling keeps the governance layer in sync with the data layer.

- **W3a — `coordinator-coordination.md` additions:**
  1. New section `## Identity: SessionStart Is Canonical` after the three-scale architecture section, naming the three identity components (SessionStart session_id, terminals yaml, supra yaml) and their priority order. Cross-link `scratchpad/coordinator/notes.md` §"2026-04-19 — Failure Mode: Identity Tagging Drift" and the `d077c25` fix.
  2. Add to Anti-Patterns: "Skills writing identity-bearing files from a forked subagent context."
  3. New subsection `## Supra-Ghost Recovery Protocol` — when `/valuate` targets the wrong supra (desktop-app PID-walk failure or subagent-context regression): (a) write the correct supra yaml directly, (b) note the pivot in the coordinator close-out's prior-entries index, (c) do not rewrite history on already-written files with the wrong id — fix forward, preserve joint files as historical record.
- **W3b — CLAUDE.md §Utility Infrastructure:** add a three-line entry for `utils/provenance.py` (single source of truth for sidecar + ledger + figure provenance; see `specs/artifact_provenance.md`).
- Acceptance: rule file + CLAUDE.md render cleanly; spec cross-links resolve.

> **Scope note on session-identity-hardening:** W3a here is the W3 wave of `.claude/plans/2026-04-19-session-identity-hardening.md`. W1 (runtime identity check in `/niche`) and W2 (subagent guard on identity-bearing writes) are deferred — they require hook changes + tests, too much for a bundled docs wave. Mark the session-identity-hardening plan as "W3 absorbed into cluster-2; W1/W2 remain OPEN" in its status block at close-out.

### W4 — qaqc gate + README pass (qaqc)

- qaqc script: `scripts/one_off/audit_sidecar_coverage.py` (temporary, 30-day shelf life) — walks `data/` + `reports/` + `stage3_analysis/` output dirs, finds artifact files without sibling `.run.yaml`, prints a count + a sample. Acceptance target: all stage3-probe-produced artifacts from W2 onward have sidecars; pre-W2 legacy artifacts are expected to be uncovered and that is fine (backfill is out of scope for this plan).
- Also detects the write-side failure signature: sidecar-without-ledger-row (the raised `ledger_append` leaves this pattern — audit script flags it as a distinct bucket from missing-sidecar).
- Verify: `len(read_ledger()) == count(*.run.yaml from stage3)` for artifacts produced during the session.
- qaqc produces commit-readiness verdict: whether the working tree is committable.

### Final Wave — close-out

- Coordinator scratchpad with all 7 Markov-completeness items.
- Update `.claude/scratchpad/coordinator/notes.md` identity-drift failure-mode entry with a "2026-04-24 partially resolved: W3a codified canonical identity + supra-ghost recovery in rule file" subsection.
- `done` message to any peers naming the commit(s).
- Close the session-identity-hardening plan's status: W3 absorbed; W1/W2 still OPEN.

## Out of scope (explicit)

- **Stage1 retrofit** — alphaearth/poi/roads processor sidecars. Separate future plan. Heterogeneous save paths; bigger refactor.
- **Stage2 retrofit** — training runs + checkpoint provenance. Separate future plan. Interacts with checkpoint versioning (`[open|42d]` from 2026-04-19 close-out).
- **Backfill of legacy artifacts** with minimal sidecars. One-shot migration script; separate future plan.
- **Session-identity-hardening W1 (`/niche` Wave-0 identity check) and W2 (subagent guard hook)** — require fresh context with hook-testing focus; Friday-evening docs-only bundle cannot carry them.
- **Extensive theory-doc writing** on Levin/sheaves/gyroscope applied to `.claude/`. The frame here is internalised into the plan structure; writing a long essay is research-paper work, not infra work.

## Execution

```
/clear
/niche follow plan .claude/plans/2026-04-18-cluster2-ledger-sidecars.md
```

The plan is self-contained. A fresh context reading this file should know: the frame (gyroscopic outer-loop rate-enablement, salience-preserving engrams), the five scope decisions, the wave dependencies, and the explicit out-of-scope list. No need to reload Levin's 2024 paper or the user's gyroscope concept note to execute — those are internalised into the plan's structure.

**Source concept docs (for reference only, not required reading):**
- `deepresearch/levin_self_improvising_memory_synthesis.md` — the full Levin paper synthesis + sheaf/formal-gap review
- `deepresearch/gyroscopic_two_timescale_pomdp.md` — the user's two-timescale POMDP formalism
