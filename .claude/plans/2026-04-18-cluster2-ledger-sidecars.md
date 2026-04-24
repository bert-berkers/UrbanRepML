# Cluster 2 — Ledger + Sidecars + Figure Provenance (READY, re-scoped 2026-04-24)

| Field | Value |
|---|---|
| **Status** | READY — executable next `/niche` |
| **Source** | `reports/2026-04-18-organizational-flywheel-audit.md` §5 roadmap items #9, #10, #11, #12 |
| **Parent plan** | `.claude/plans/2026-04-18-flywheel-gyroscope-fix.md` (cluster 1, executed `99099c0`) |
| **Cluster** | 2 — Territory (touches `utils/` + `stage3_analysis/` + `data/ledger/` + `.claude/rules/`) |
| **Depends on** | `2026-04-18-cluster2-paths-consolidation.md` ✅ shipped `dcac83d` |
| **Est** | ~4–6h (bounded single-session scope) |

## Frame (why a single Selflet should read the whole plan before diving in)

Sidecars and the ledger are **mnemonic-improvisation infrastructure at the data scale**. Each experiment run leaves a machine-readable note (`*.run.yaml` sidecar) for a future analyst-Selflet to re-interpret. The append-only ledger (`data/ledger/runs.jsonl`) is the stigmergic trace layer — a past run leaves a signal for a future analyst. Figure provenance (`save_figure(sources=)`) closes the loop: every plot knows which runs it depends on.

Gyroscopically: this moves cross-run analysis from **event-driven** (user opens a dir, grep-browses PNGs) to **rate-possible** (daemon folds `runs.jsonl` into a view any time). The outer loop's goal-reconfiguration over experimental results no longer requires archaeological digs. That is the load-bearing justification — not reproducibility per se, but *enabling the outer loop to run at a rate*.

## Scope decisions (resolving DRAFT's Next-session TODO)

1. **One plan with 5 waves, not four plans.** The four audit items (#9 sidecars, #10 ledger, #11 figure provenance, #12 config hashing) are semantically one system — config hashing underlies sidecars, ledger rows are sidecar summaries, figure provenance is a sidecar specialisation. Splitting would fork shared schema decisions across plans. Each wave is individually shippable (rate-driven within the cluster).
2. **Ledger format: JSONL.** `data/ledger/runs.jsonl`, append-only, one row per completed run. Human-readable, git-diffable, no lock contention for multi-terminal appends, trivial to tail/filter. SQLite is premature optimisation — revisit only if cross-run reads become slow. An optional `data/ledger/runs.parquet` can be materialised on-demand as a *view* (fidelity-preserving JSONL source; salience-preserving parquet view — matches Levin's salience/fidelity split).
3. **Prototype stage: stage3 probes first.** `LinearProbeRegressor` + `DNNProbeRegressor` have clean save boundaries, deterministic outputs, multiple instances to test the schema, immediate analytical payoff, and low blast radius. Stage1 (heterogeneous modality encoders) and stage2 (long training runs, checkpoint complexity) are retrofits for a later plan.
4. **Sidecar minimum fields:** `run_id`, `git_commit`, `git_dirty`, `config_hash`, `config_path`, `input_paths`, `output_paths`, `seed`, `wall_time_seconds`, `started_at`, `ended_at`, `producer_script`, `study_area`, `stage`, `schema_version`. Optional fields (Dirichlet, metric values, any domain-specific keys) go under `extra:` so the minimum schema stays stable.
5. **qaqc gates (W5):** (a) every new artifact path has a `*.run.yaml` sibling; (b) `len(read_ledger())` equals the count of on-disk `.run.yaml` files; (c) same `config_hash` + same inputs → same sidecar modulo `wall_time_seconds`.

## Wave structure

### W0 — Read-only audit + schema finalisation (spec-writer)

- Read `stage3_analysis/linear_probe.py`, `dnn_probe.py`, `classification_probe.py` save paths.
- Read `scripts/stage3/plot_cluster_maps.py` + any `stage3_analysis/*viz*.py` figure-write sites.
- Freeze the sidecar schema as a spec file: `specs/artifact_provenance.md`. Include the 15 minimum fields + `extra:` conventions + figure-provenance specialisation (`source_runs`, `source_artifacts`, `plot_config`).
- Acceptance: spec written, no code changes, librarian's `codebase_graph.md` updated with the new spec entry.

### W1 — Core utility module (devops + stage3-analyst)

- Write `utils/provenance.py`:
  - `compute_config_hash(cfg: dict) -> str` (canonical JSON + sha256, stable across dict ordering)
  - `SidecarWriter` context manager — captures `started_at`, on `__exit__` captures `ended_at`, `wall_time_seconds`, writes `*.run.yaml` next to the artifact
  - `ledger_append(sidecar_path: Path)` — reads sidecar, appends a minimal row to `data/ledger/runs.jsonl` (uses advisory file-lock for cross-terminal append safety)
  - `read_ledger(path: Path = None) -> pd.DataFrame` — convenience reader that handles malformed rows fail-open (skip + stderr warn)
- Tests: `tests/utils/test_provenance.py` — config-hash stability, sidecar round-trip, ledger append idempotence, reader fail-open on malformed rows.
- Acceptance: all tests pass; `python -c "from utils.provenance import SidecarWriter, compute_config_hash, ledger_append, read_ledger"` imports clean.

### W2 — Stage3 probe retrofit (stage3-analyst)

- Wire `SidecarWriter` into `LinearProbeRegressor.save()`, `DNNProbeRegressor.save()`, `ClassificationProber.save()`, `DNNClassificationProber.save()`.
- Add `stage3_analysis/save_figure.py`: `save_figure(fig, path: Path, sources: list[str], plot_config: dict = None)` — writes figure + `*.provenance.yaml` sibling.
- Retrofit `linear_probe_viz.py`, `dnn_probe_viz.py`, `classification_probe_viz.py`, `dnn_classification_probe_viz.py` to call `save_figure()` instead of `fig.savefig()` directly. Sources = list of `run_id`s from the source probe sidecars.
- Acceptance: run one probe end-to-end (e.g. `python -m stage3_analysis.linear_probe --study-area netherlands` if CLI exists, else an ad-hoc script); verify `.run.yaml` sits next to the probe output, `runs.jsonl` has a new row, figure has a `.provenance.yaml` sibling pointing at the probe's `run_id`.

### W3 — Gyroscopic governance updates (spec-writer, docs-only, small)

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
