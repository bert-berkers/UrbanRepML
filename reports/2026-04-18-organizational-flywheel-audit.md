# Organizational Flywheel Audit — 2026-04-18

**Session**: `golden-standing-creek-2026-04-18`
**Mode**: audit-only (no code changes, no specs/scripts edits)
**Inputs**: six Wave 1 surface audits (librarian, qaqc, spec-writer, stage3-analyst, devops, ego) + the plan at `.claude/plans/2026-04-18-organizational-flywheel-audit.md`
**Output**: this report + one Wave 2 scratchpad entry
**Status**: Draft (Wave 3 ego self-audit pending)

---

## 1. TL;DR

The UrbanRepML project is alive — code, data, and results continue to accrete. The machinery that keeps the coordinator + subagents + human from getting lost in that accretion — call it the **ORIENT substrate** — has rotted in parallel with the territory. The audit found disorder in both: eight distinct themes, affecting five ORIENT artifacts (scratchpads, `codebase_graph.md`, `specs/`, `.claude/plans/`, session-state YAMLs) and three OBSERVE surfaces (repo root, `stage3_analysis/` layouts, experiment ledger). The rot is not catastrophic — no lost data, no broken pipelines — but the cost compounds: every new wave pays more context tax to orient, and each `/clear` boundary leaks more state forward than necessary.

Reframing the flywheel as a **gyroscope** (the plan's central metaphor): the flywheel is the persistence of motion across `/clear` and calendar boundaries; the gyroscope is that motion made directional by a reference frame. Today both are weak: the motion persists but is undirected at session boundaries (intent null, `[open]` items aged invisibly, claim-squatting universal), and the map is crowded with tombstones that make the current territory harder to see.

**The top three highest-leverage recommendations, across all themes:**

1. **Enforce a Markov-complete handoff packet on the coordinator's final scratchpad entry of every session** (not just plan-driven ones). This is Contract 1 in §4 — the reference frame, aged `[open]` items, active plan pointer, peer-terminal pointer, and a single "if you only read this" paragraph. **One hook change + one rule update.** Single highest-leverage fix because it closes the primary state leak without touching the territory.

2. **Consolidate `utils/paths.py` onto one stage3-result convention** (`{type}/{date}/{run_id}/` with `config.yaml`+`run.yaml` sidecars, per stage3-analyst §2.2). Five layout conventions coexist in `netherlands/stage3_analysis/` precisely because three inconsistent path-building methods in `utils/paths.py` allow them. **This is a territory fix that stops the map from drifting** — the downstream benefits (figure sidecars, reports↔figures coupling, experiment ledger) all presuppose it.

3. **Evict the changelog from `codebase_graph.md` and split into per-stage child files** (librarian §P1+P2). The graph is 3228 lines, 338KB, 13 confirmed drift items, and is never read in full. A thin TOC + per-stage children restores its function as "where does X live?" without any information loss. **The changelog paragraph IS git log in prose form — deleting it is a gain, not a loss.**

These three interlock: #1 stops state from leaking forward across sessions, #2 stops the territory from generating new drift faster than the map can absorb it, #3 makes the map queryable again. In that order, each enables the next.

A tension worth naming upfront: recommendations #2 and #3 both require a future wave of actual editing work, which contradicts today's audit-only constraint. The report documents what to do; a subsequent session executes. The hard part is **scheduling that execution without it becoming another avoidance-accumulating `[open]` item** — precisely the class-6 failure mode (calendar-day state loss) the audit identified.

---

## 2. Four-Surface Model — OODA Mapping of Artifacts

The plan's four-surface model (OBSERVE / ORIENT / DECIDE / ACT) is the organizing lens. Every disorder the audit found sits on one of these surfaces; every fix targets one. Here is the full mapping.

| Surface | What it is | Artifacts in UrbanRepML | Health today |
|---|---|---|---|
| **OBSERVE** — territory | The state that just IS. Code, data outputs, training artifacts. | `stage*_*/` code; `data/study_areas/.../*`; `wandb/`, `lightning_logs/`, `checkpoints/`; repo-root strays. | Cluttered but not broken. Repo root has ~2MB of ignored-but-present logs + 158KB–205KB training logs that ARE tracked. `stage3_analysis/` runs five coexisting layout conventions. 125MB `results [old 2024]/` is superseded. |
| **ORIENT** — maps | The five documentation artifacts that exist so the organization doesn't get lost as the territory grows. | **(1)** `.claude/scratchpad/{agent}/` • **(2)** `.claude/scratchpad/librarian/codebase_graph.md` • **(3)** `specs/` • **(4)** `.claude/plans/` • **(5)** session-state YAMLs (`supra/sessions/`, `coordinators/terminals/`, `coordinators/session-*.yaml`, `coordinators/messages/`, `scratchpad/valuate/`). | All five partially rotted. codebase_graph 3228 lines / 13 drift items. 17 specs / no index / 6 overlap. `coordinator-coordination.md` references two directories that no longer exist. Scratchpad contract specifies format but not completeness. |
| **DECIDE** — human-in-the-loop | The human's chat thread, translated into new plans or revisions of existing ones. | `.claude/plans/*.md`; user chat; `/valuate` sessions. | Working. Plan structure is good; when a plan exists and is followed, the system behaves well. Weakness is **when no plan is active**: session opens with null intent, no reference frame. |
| **ACT** — subagent work | Specialist agents execute; coordinator orchestrates. Modifies territory or produces new maps. | Task dispatches → specialist scratchpads → code/data/spec changes. Reports and figures are the crystallized ACT products. | Working within sessions. Weakness at the boundary: ACT produces reports + figures + runs, but these rarely feed back into ORIENT (no experiment ledger, no figure sidecars, no report→run provenance). Crystallization gap. |

**Key structural observation — "Crystallization gap"**: the ACT→ORIENT loop is broken. Every specialist ACT produces outputs that should update the map (new spec → `specs/index.md` update; new training run → experiment ledger row; new figure → sidecar YAML; new code → `codebase_graph.md` entry). In practice, these updates happen sporadically or not at all. Themes B, F, and parts of C all name this crystallization gap from different angles. The Markov contracts in §4 are specifically designed to close it.

---

## 3. Per-Theme Findings (A–H)

Priority is assigned **across all themes**, not per theme. P0 = fix in the next 1–2 sessions; P1 = fix in the next 2 weeks; P2 = track and fix opportunistically.

### Theme A — Repo-root hygiene

**Current state (from devops Wave 1 §A):** the repo root contains 47 entries including 5 git-tracked log files that should NOT be tracked, a `nul` Windows artifact, `img.png` (521KB stray), `results [old 2024]/` (125MB, literal brackets in name), and 6 ignored-but-present log files (`stage2_fusion.log` is 2.4MB alone). Three specific tracked training/probe logs:
- `probe_log_concat_74d.txt` (14.8KB) — tracked
- `probe_log_ring_agg_k10.txt` (14.9KB) — tracked
- `probe_log_unet_2022.txt` (14.8KB) — tracked
- `training_log_1000ep.txt` (158KB) — tracked
- `training_log_unet_1337ep.txt` (205KB) — tracked
- `training_log_unet_1000ep.txt` (0 bytes) — tracked

**Classification**: TERRITORY (OBSERVE) problem. No ORIENT artifact damaged directly; but indirectly, the clutter raises the cost of any fresh-context observe-pass.
**Recommended fix**: (a) Add to `.gitignore`: `probe_log_*.txt` and `training_log_*.txt`. (b) `git rm --cached` the five tracked logs. (c) Delete `nul`, `img.png` (confirm latter with user — modified 2026-04-17, possibly from green-branching-kelp session), `logs/*.log` (0-byte), and the ignored `.log`/`.txt` strays. (d) Flag `results [old 2024]/` for deletion with explicit human confirmation per the "never delete data results" memory — but note that the dir IS explicitly labeled "old", spec `experiment_paths.md` is migrating AWAY from it, and no reports cite it. This is a "special case" deletion; a one-line note in the eventual commit suffices per the policy's spirit.
**Priority**: **P2** for the strays (cosmetic); **P1** for the tracked logs (actual git history pollution). Not P0 because nothing breaks until the repo is shared externally.

### Theme B — Experiment ledger missing

**Current state (from devops Wave 1 §B):** no `reports/experiments.csv`. The connection `wandb run_id → training script → checkpoint → probe results → report` must be reconstructed from logs. Five wandb runs identified; two completed, three killed. Checkpoint versioning IS partially implemented in `data/study_areas/netherlands/stage2_multimodal/unet/checkpoints/` but schema is inconsistent (some files `{year}_{dimD}_{date}`, one anomalous 44MB file `best_model_20mix_64D_2026-03-14.pt` — likely includes optimizer state). MEMORY.md P0 #7 (checkpoint versioning, dated 2026-03-09) is partially resolved but not closed.

**Cross-theme note**: `specs/run_provenance.md` (5.3KB, 2026-02-14, status "Draft Approved") literally IS the answer to this theme — spec-written, never implemented (spec-writer Wave 1 §1, `run_provenance.md` row).
**Classification**: OBSERVE/ACT feedback — results produced by ACT don't crystallize into an ORIENT-consumable ledger.
**Recommended fix**: Promote `specs/run_provenance.md` from Draft to Active; implement its `run_info.json` manifest + dated run dirs. Create `reports/experiments.csv` with the schema devops proposed (§A4): `run_id, date, script, model_class, modalities, year, dims, epochs_planned, epochs_ran, lr, best_epoch, best_loss, checkpoint_path, embeddings_path, probe_report, notes`. Populate retroactively from the 2 fully-logged wandb runs + 3 known checkpoint files + LatticeGCN lightning_logs.
**Priority**: **P1**. The spec is already written; this is execution. Holds up closing MEMORY.md P0 #7.

### Theme C — Specs sprawl

**Current state (from spec-writer Wave 1):** 17 specs, no `specs/README.md` index. Classification: **7 Active** (load-bearing: `accessibility_graph_pipeline`, `dnn_probe`, `experiment_paths`, `script-hygiene`, `session-identity-architecture`, `temporal-supra-profiles`, `claude_code_multi_agent_setup` — though the last is Feb-13 and partially stale), **4 Historical** (`3_stage_pipeline_restructure`, `coordinator-hello`, `hex2vec-poi-recovery`, `skip-connection-fix`), **1 Superseded** (`coordinator_to_coordinator` — the 27KB spec is superseded by the 5KB `.claude/rules/coordinator-coordination.md` runtime rule), **1 Redundant** (`hooks_architecture` overlaps `claude_code_multi_agent_setup` Phase 1), **4 Draft/Partial** (`between-wave-pause-redesign`, `h3_index_vs_region_id`, `poi-pipeline-pyosmium-sedona`, `run_provenance`).

Six of these are Claude-infra specs with substantial overlap. The strongest pattern observed: **specs are rationale, rules are runtime.** `coordinator-coordination.md` (the rule) is 5KB and load-bearing at runtime; `coordinator_to_coordinator.md` (the spec) is 27KB and documents the rationale. Both have a place — but only if the spec's supersession is visible.

**Classification**: ORIENT problem — map-level, directly damages the "which spec is load-bearing right now?" query.
**Recommended fix**: (a) Create `specs/README.md` per spec-writer §4 (index with status + one-line summary, grouped by functional area). (b) Add status headers to each spec (`## Status: Active | Implemented | Superseded | Draft`), with Superseded specs pointing to their successor. (c) Do NOT rewrite or merge files today — audit-only. The index + headers are the MINIMAL change that restores Markov-completeness; actual consolidation is a separate wave. (d) For `between-wave-pause-redesign.md`: diff against `niche/SKILL.md` wave structure to determine absorption status; this is the single open research item.
**Priority**: **P1**. High-leverage, low-cost (no content rewrites, just metadata).

### Theme D — Archive opacity

**Current state (from qaqc Wave 1 §D):** `scripts/archive/` has 53 files in 12 subdirs with NO README, NO index. CLAUDE.md describes it as "read-only reference" — but unsearchable is worse than deleted. Detailed assessment (qaqc §Sample assessment per subdir):
- KEEP+INDEX: `benchmarks/` (2), `diagnostics/` (1), `cone_alphaearth/` (partial), `legacy/` (6 of 9), `preprocessing/` (3 conditional), `probes/` (8 of 12 — **the most valuable subdir**, documents research trail including the "ring agg k=10 wins" finding), `roads/` (2 of 5), `visualization/` (4 of 8).
- DELETE: `debug/` (both files — problem solved), `migration/` (both — explicitly self-expired, one dated 2026-04-15), `utilities/` (both — pre-SRAI anti-pattern, `netherlands_h3_regionalizer.py` uses `import h3` directly), 4 redundant variants in `roads/`, 4 in `visualization/`, 4 in `probes/`.
- Net: ~15–17 DELETE candidates, ~30–32 KEEP+INDEX, no files need relocation.

**Classification**: ORIENT problem — map-level. Archive without index is invisible.
**Recommended fix**: Single flat `scripts/archive/README.md` as TOC with columns `subdir/filename | archived_date | what | why | reference_value`. Per qaqc §"Index proposal": flat > per-subdir, because small subdirs (benchmarks: 2 files) don't justify per-subdir READMEs. After the index is written, a separate PR can delete the ~15–17 expired items.
**Priority**: **P2**. The archive is currently low-query; the index is durable hygiene.

### Theme E — Stage3 layout drift

**Current state (from stage3-analyst Wave 1 §1):** Netherlands `stage3_analysis/` has **five coexisting layout conventions**:
- **A**: bare-date-only (`stage3_analysis/2026-02-20/*.png`) — one orphan
- **B**: `{type}/{YYYY-MM-DD}_{suffix}/` — flat-dated (most linear_probe runs)
- **B'**: `{type}/{YYYY-MM-DD}/{YYYY-MM-DD}_{suffix}/` — nested-dated (the `utils/paths.py::stage3_run()` contract)
- **C**: `{type}/{approach}/` — no dates, overwrites in place (the `cluster_results/` subtree)
- **D**: `{type}/{YYYY-MM-DD}/{category}/` — comparison rollups
- **E**: `{type}/2025-mm-dd/` — **literal placeholder string**, bug in an early cluster script; affects `cascadia`, `pearl_river_delta`, and `netherlands/kmeans_clustering_1layer/`.

**Worst instance**: `dnn_probe/2026-03-21/` contains BOTH `2026-03-21_custom_concat_74d/` (Convention B') AND `concat_74d/` (suffix-only) as separate siblings with potentially different contents. Two layout conventions, same date, same approach, no way to tell which run a downstream consumer read.

**Root cause** (stage3-analyst §6): `utils/paths.py` itself contains three inconsistent path-building methods: `stage3_run()` uses Convention B', `probe_results_root()` uses flat-with-today-date, `cluster_results_root()` uses flat-without-date. The drift in the territory is a direct reflection of drift in the path library.

Also: `data/study_areas/netherlands/cluster_results/` exists as empty peer of `stage3_analysis/` — dead scaffold from a prior layout attempt.

**Classification**: OBSERVE (territory) problem with its root cause in `utils/paths.py` code — which per data-code separation is code not map.
**Recommended fix**: (a) Consolidate `utils/paths.py` onto Convention B' everywhere, strip `{date}_` from run_id (date already in parent path). (b) Add `config.yaml` + `run.yaml` sidecars to every run output (required; the "rerun + hope" → "rerun + verify" flip). (c) Promote `cluster_results/` from Convention C to dated — preserve existing four-approach flat layer under `cluster_results/{last_known_date}/` per "never delete data results" memory, new writes go to dated. (d) Add `stage3_analysis/_runs.parquet` manifest written by ProbeResultsWriter / ClusterResultsWriter. (e) Fix the Convention E placeholder bug in the generator script; leave existing on-disk `2025-mm-dd` dirs in place (per "never delete data results"; mark historical via manifest).
**Priority**: **P0 for the paths.py consolidation** (blocks most other stage3 work). **P1 for the migration** (new writes to canonical; historical untouched).

### Theme F — Reports ↔ figures decoupled

**Current state (from stage3-analyst Wave 1 §3):** 6 reports, 43 path citations audited. **34 OK, 3 DEAD, 6 AMBIGUOUS.**

The 3 dead citations all in `2026-03-08-causal-emergence-phase1.md`: `multiscale_probe_results.csv`, `multiscale_comparison.png`, `multiscale_delta.png` — files never existed on disk, OR were generated and deleted (if the latter, the "never delete data" memory was violated; needs investigation).

The 6 ambiguous citations all stem from the Convention B-vs-B' mismatch from theme E: reports cite flat `dnn_probe/2026-03-07_multiscale_*` paths but real files live nested at `dnn_probe/2026-03-07/2026-03-07_multiscale_*`. Files exist, report text is misleading.

No figure has a `.yaml` sidecar with git-SHA, source-script, source-runs metadata. No report has a provenance block. `reports/figures/accessibility/2026-03-21/` is an empty scaffold (directory created, no files written). Zero reports reference `cluster_results/` — stage3 clustering is disconnected from the reports layer.

**Classification**: OBSERVE/ACT feedback — crystallization gap. Reports name figures, figures don't name reports, and neither names runs.
**Recommended fix**: (a) Figure sidecar contract per stage3-analyst §4.1 — every PNG/PDF under `reports/figures/` gets a `same-stem.yaml` sidecar with `figure_id, generated_at, generated_by (script path), git_sha, source_runs, data_versions, config_hash, referenced_by`. (b) `reports/figures/{report-slug}/` nested layout per §5 — one dir per report, `_shared/` for cross-report, `_manifest.parquet` for discovery. (c) Plot scripts that don't emit sidecars fail lint (enforce at write time). (d) Investigate the 3 dead citations — if files were deleted, the policy was violated; if never generated, add a report-text fix note.
**Priority**: **P1**. Downstream of theme E (path consolidation) but can run in parallel on the sidecar-generation side.

### Theme G — `codebase_graph.md` rot

**Current state (from librarian Wave 1):** **3228 lines / 338KB**, 23 update rounds, `.bak-2026-03-29` sibling. The ENTIRE header is a single 1200-word paragraph summarizing rounds 17–52. **13 confirmed drift items** including three top-level script relocations (D1: `compare_probes.py`/`plot_linear_probe.py`/`plot_targets.py` all moved to `scripts/stage3/` — graph still says `scripts/`), three one-off relocations (D2: `osmium_time_filter.py`/`extract_pois_from_history.py` moved to `scripts/processing_modalities/`, `dnn_probe_res9_modality_comparison.py` gone), a documented-but-nonexistent script (D3: `probe_20mix_multiscale.py`), 8 specs missing from the graph's specs section (D4 — graph lists 7 of 17 total), `.claude/plans/` undercounted (D5 — 3 listed, 18 exist), `utils/visualization.py` absent entirely despite being imported by 5+ scripts (D10).

**Classification**: ORIENT problem — THE observe-map artifact, currently broken.
**Recommended fix**: (a) Evict the changelog paragraph (librarian §P2) — git log has it; the map is not a history. (b) Split into thin TOC root (~150 lines: interface contracts + TOC + H3 audit + region_id flow) + per-stage child files under `codebase_graph/` subdir (§P1: `utils.md`, `stage1.md`, `stage2.md`, `stage3.md`, `scripts.md`, `infrastructure.md`, `data_artifacts.md`, `known_issues.md`). (c) Per-file `<!-- last-verified: YYYY-MM-DD by librarian -->` stamp; files >30 days unverified are flagged stale. (d) Delete resolved bugs and tombstones (§P3); migrate the `.claude/ Infrastructure` section (lines 2096–2376) into `specs/hooks_architecture.md` per §P4. Post-rework estimate: ~2100 lines across 9 files vs 3228 in one.
**Priority**: **P1** for the split; **P0** for the eviction of the changelog (single highest-impact single line-range edit; ~1200 words deleted).

### Theme H — OODA loop persistence (META)

**Current state (from ego Wave 1):** The close-out protocol specifies scratchpad **format** but not **completeness**. Seven classes of state loss identified across `/clear` / terminal-switch / calendar-day boundaries:

1. **Unresolved-list decay across `/clear`** — 5 `[open]` items from 2026-03-29 persisted in files but weren't surfaced in 2026-04-17's chat (the 19-day-stale probe-confound backlog).
2. **Intent fidelity across `/clear`** — today's supra YAML has `intent: null`; the plan file substitutes but nothing in the niche skill formalizes "plan = frozen intent."
3. **Cross-wave scope drift within a session** — 2026-04-17 had three post-Final-Wave dispatches that weren't in the plan; no "session-closed" lock.
4. **Protocol rule re-learning** — three separate memory files for the same "coordinator must delegate" lesson; Pre-Edit Gate exists on paper but is self-invoked.
5. **Terminal-switch state** — `terminals/{pid}.yaml` carries only `started_at` + `supra_session_id`; no active-plan pointer, no peer-terminal summary.
6. **Calendar-day boundary state** — 19-day silent gap between 2026-03-29 and 2026-04-17; `[open]` items have no age annotation; 3-week-stale P0 reads identical to today's P0.
7. **Mid-session model-swap signals** — the "Opus 4.7 weak at viz" pattern was legible by Wave 3a in green-branching-kelp, only surfaced post-hoc as a memory file.

Session-state machinery audit found:
- `.claude/coordinators/sessions/` and `.claude/coordinators/supra/` referenced in `coordinator-coordination.md` lines 85–86 **do not exist on disk** — replaced by `terminals/{pid}.yaml` during the 2026-03-14 PPID→terminal-PID refactor, rule file wasn't updated.
- **All 4 recent claim files have `claimed_paths: ['*']`** — claim-squatting is universal despite the rule forbidding it after the first OODA cycle.
- Messages channel carries only HELLO broadcasts; `request`/`warning`/`done` messages predicted by the protocol are essentially absent.
- 7-day message archive is not running (2026-03-07 messages still present on disk, 42 days old).
- `/niche` reads the supra reference frame only at Wave 0 by design (explicit line in SKILL.md: "Do NOT re-read supra states between waves") — acceptable as context-budget tradeoff IF Wave 0 echoes the frame into the coordinator scratchpad, but no rule mandates that echo.

**Classification**: ORIENT + protocol surface. This is the meta-leverage theme — if H's contracts are honored, the other themes become local cleanups that stay fixed. Without them, every cleanup starts rotting again.

**Recommended fix**: The five Markov-completeness contracts in §4 below. Plus: update `coordinator-coordination.md` to point at `terminals/{pid}.yaml` (delete references to nonexistent `sessions/` and `supra/`). Plus: either enforce claim-narrowing or remove the rule — don't keep an aspirational protocol that's universally violated.
**Priority**: **P0**. This is the single highest-leverage theme. Contract 1 (scratchpads) alone closes the primary state leak.

---

## 4. Per-Artifact Markov-Completeness Contracts

**This is the load-bearing core of the report.** Each contract specifies what one of the five ORIENT artifacts must guarantee for the next context window — what it must contain for a cold-start resume, what invariants must hold across writes, and what mechanism keeps it from rotting. Contracts are written as implementable (hook / rule / skill protocol), not as discipline.

Draft material in ego's Wave 1 §Synthesis. What follows is refined and hook/rule-addressable.

### Contract 1 — Scratchpads (coordinator + agent)

**Purpose**: the stigmergic memory layer. The final entry of each agent-type's session-keyed scratchpad must be a self-contained context packet for the next fresh window.

**What the final entry of every session's coordinator scratchpad must contain:**

1. **Summary comment** — `<!-- SUMMARY: one-line ... -->` (already required, keep).
2. **Prior-entries index** — cumulative, one line per entry (already required, keep).
3. **Current reference frame** — a block echoing the active supra values: `mode=... speed=... explore=... quality=... tests=... spatial=... model=... urgency=... data_eng=... intent=... focus=[...] suppress=[...]`. If `/valuate` wasn't run, write `intent=null (plan-driven: {path})` or `intent=null (no plan, no valuation — under-valuated session)`.
4. **Aged `[open]` items** — each tagged `[open|Nd]` where N = days since first opened. Carry forward from prior entries; do NOT re-open already-carried items, just increment the age. `[stale|Nd]` when N >= 14. Escalate to `[blocked:human-decision]` when N >= 21 AND unresolved across two sessions.
5. **Active plan pointer** — `Plan: .claude/plans/{file}.md` if any. Plans function as frozen intent; their presence is load-bearing.
6. **Peer-terminal pointer** — read from `.claude/coordinators/` session-*.yaml files: list of other currently-active or recently-ended terminals with their supra_session_id.
7. **"If you only read this entry, here's the gist" paragraph** — single paragraph at bottom. Must subsume 1–6 in prose; must be sufficient to orient a cold-start session without reading anything else. **This paragraph is the Markov-complete summary.**

**Per-agent (non-coordinator) scratchpad final entry must contain:** items 1, 2, 7. Items 3–6 are optional for specialists — they inherit the reference frame from the coordinator.

**Invariants across writes:**
- Append-only — never rewrite earlier entries.
- Session-keyed filename (`{date}-{session_id}.md`) — no cross-terminal clobbering.
- Signal vocabulary (BLOCKED/URGENT/SHAPE_CHANGED/etc.) consistent.
- The 30-line-per-entry guideline is soft; override with `<!-- OVERRIDE: Markov-completeness requires ~N lines -->` at the top of the entry. **Length is not the cost; state loss is.** (This audit's Wave 1 entries all used the override successfully — the mechanism works.)

**Enforcement:**
- Extend existing SubagentStop hook (`subagent-stop.py`) with a `check_markov_completeness(scratchpad_path)` function that:
  - For coordinator scratchpads at session-end (detectable via `/niche` Final Wave invocation or `/clear`): refuse completion if items 1–7 missing from the latest entry.
  - For specialist scratchpads: log warning if 1/2/7 missing, accept write (fail-open for specialists).
- Extend `multi-agent-protocol.md` Scratchpad Format section to list items 1–7 explicitly.
- The "if you only read this" paragraph is the key enforceable invariant — a simple regex check (`^## If you only read`) suffices.

**Fail mode:**
- Specialist entries: fail-open (log, accept).
- Coordinator session-end entries: fail-closed (refuse /niche Final Wave completion; surface missing items to human).

### Contract 2 — `codebase_graph.md`

**Purpose**: the structural map of the territory. Must answer "where does X live?" in ≤5 seconds for any X in the active codebase.

**What the root file must contain (for cold-start resume):**

1. **Last-verified datestamp** — top of file: `<!-- last-verified: YYYY-MM-DD by librarian -->`. If >30 days old, the root file must emit a "STALE" banner when read.
2. **Cross-stage interface contracts** — the most-queried content: what shape flows stage1→stage2→stage3, what `region_id` looks like everywhere, what H3 resolution boundaries are respected.
3. **TOC of child files** — one line per child, one-line topic summary.
4. **H3 compliance audit** — single table, no prose.
5. **Index contracts (region_id flow)** — single table.

**Per-child-file contract**: each of `utils.md`, `stage1.md`, `stage2.md`, `stage3.md`, `scripts.md`, `infrastructure.md`, `data_artifacts.md`, `known_issues.md` must:
- Have a `<!-- last-verified: YYYY-MM-DD by librarian -->` stamp.
- List every module in its domain with path + one-line contract. Zero-line modules (empty stubs) still listed.
- Fit in ≤500 lines. Exceeded → split further.
- Contain only CURRENT state. Renamed/deleted modules removed, not annotated-in-place. Exception: one-line tombstones for commonly-confused deletions, auto-removed after 6 months.

**Invariants across writes:**
- Only the librarian agent writes. Other agents read-only.
- Committed changes only. Aspirational items go in `.claude/plans/`, not here.
- Root ceiling: 200 lines. Each child ceiling: 500 lines.
- No changelog; no round numbers; no update history. git log is the history.

**Enforcement:**
- `/librarian-update` skill: runs at every session's Final Wave; re-verifies at least one child file's claims against disk (spot check).
- SubagentStop on librarian invocation: rejects writes that leave root >200 lines or any child >500 lines.
- Monthly scheduled librarian-update (CronCreate candidate) refreshes verification stamps.

**Fail mode**: fail-open on reads — a stale graph is still better than no graph, but staleness banner must be visible. Fail-closed on writes exceeding size ceilings.

### Contract 3 — `specs/`

**Purpose**: architectural decision trail. Spec = rationale (why we chose X); rule = runtime (what the system does). This distinction is load-bearing.

**What `specs/` must contain:**

1. **`README.md`** — index grouped by functional area (pipeline / scripts / Claude-infra). One row per spec: filename | status | one-line summary. Statuses: **Active** | **Implemented** | **Superseded (→ successor)** | **Draft** | **Historical**.
2. **Every spec** has a `## Status:` header on line 2–3. Superseded specs have a top-line pointer block: `> Superseded by: {path}. See that spec/rule for current behavior.` Content is preserved (rationale persists); superseded specs are NOT deleted.
3. **No two Active specs overlap in scope.** If they do, one is consolidated or downgraded to Superseded.

**Spec decision tree (when to write which artifact):**
- Architectural choice with alternatives considered AND cross-session consequence → **Spec**.
- Result/finding/figure tied to a specific run → **Report** (`reports/`).
- Work that only the next context window needs to know about → **Scratchpad**.

**Mandatory spec sections**: `# Title`, `## Status`, `## Context`, `## Decision`, `## Alternatives Considered`, `## Consequences` (Positive/Negative/Neutral), `## Implementation Notes`.

**Invariants across writes:**
- `spec-writer` is primary author. Domain specialists may co-author deep technical specs (`skip-connection-fix.md` is the canonical example).
- Coordinator does not write specs directly — delegates.
- Status transitions are explicit: Draft→Active requires at least one code reference or rule pointing here. Active→Implemented when behavior is realized and further changes unlikely. Active/Implemented→Superseded requires a successor pointer.
- Drafts older than 30 days with no uptake: promote to `.claude/plans/` or move to `specs/archive/`.
- Size guidance: 3–12 KB typical; >20 KB should split (`temporal-supra-profiles.md` at 38KB is the outstanding split candidate — theory half extracts to `memory/` or `deepresearch/`).

**Enforcement:**
- `spec-writer` Pre-Write Gate: before writing a new spec, must read `specs/README.md` and declare which (if any) existing specs are superseded.
- Update `README.md` same-commit as any spec status change.
- Monthly audit (during organizational-flywheel waves) checks draft staleness.

**Fail mode**:
- Write-time: fail-closed. Spec-writer must declare status before commit.
- Read-time: fail-open. Missing Status header reads as "Active" (optimistic default).

### Contract 4 — `.claude/plans/`

**Purpose**: the DECIDE substrate's output; queued actions the human has approved. A plan is frozen intent — when followed, it substitutes for a fresh `/valuate`.

**What every plan must contain:**

1. **Date-stamped filename** — `YYYY-MM-DD-{slug}.md`. Header date matches.
2. **Frontmatter block** — `Status: ready | in-progress | completed | superseded`; `Mode: {audit-only | code-change | hybrid}`; `Deliverable: {path}`; `Session: {supra_session_id}` if plan-driven.
3. **Framing** — one section explaining why this plan exists.
4. **Wave structure** — waves declared up front (Wave 0 / 1 / 2 / ... / Final). Each wave names: agents dispatched, expected outputs, success criteria.
5. **Markov-completeness handoff checklist** — in the Final Wave section. Lists what the closing coordinator scratchpad entry must contain (items 1–7 of Contract 1, specialized as needed).
6. **Execution section** — how to invoke: `/niche follow plan {path}` or equivalent.

**Invariants across writes:**
- Completed/superseded plans move to `.claude/plans/archive/` with a terminal header noting outcome ("Completed 2026-04-18, see reports/...").
- Filename date matches header date.
- A plan CANNOT declare Wave Final without the Markov-completeness handoff checklist.

**Enforcement:**
- `/niche` Wave 0 step 4 (plan-read step): if plan lacks Execution section OR wave structure OR Final-Wave checklist, refuse to auto-follow; ask user to confirm "treat as informal" OR upgrade the plan.
- Monthly plan-cleanup sweep moves completed plans to archive.

**Fail mode**:
- Read-time: fail-open. A malformed plan still informs.
- Execution-time: fail-closed. A malformed plan is not auto-dispatched into waves.

### Contract 5 — Session states (the gyroscope's current-reading layer)

**Purpose**: the five-component ecosystem that makes "where are we pointed right now?" answerable. This is the load-bearing contract for the whole gyroscope framing.

**Components and contracts:**

1. **`.claude/supra/sessions/{supra_session_id}.yaml`** — must contain:
   - `supra_session_id, date, mode, dimensions{...}, intent, focus[], suppress[], last_attuned, active_graph`
   - If `intent == null AND no plan file is active`, session is under-valuated — niche Wave 0 must surface this to the user as "Run `/valuate` to set intent, or supply a task in chat."
   - Archived to `supra/sessions/archive/` after 30 days (currently never archived — 22 files on disk, some from 2026-03-08).

2. **`.claude/coordinators/session-{id}.yaml`** — must contain:
   - `session_id, claimed_paths[], status, heartbeat_at, task_summary, ended_reason (if ended)`
   - `claimed_paths MUST NOT remain ['*']` after first OODA cycle (rule already says this; all 4 recent files violate it — enforce via hook OR remove the rule).
   - `task_summary` MUST be populated with real content after first OODA cycle — not "Starting up".
   - Born-dead sessions (like pine-waiting-dust's 1-min lifetime) must be detected and logged as anomalies, not left as debris.

3. **`.claude/coordinators/terminals/{pid}.yaml`** — must contain:
   - `started_at, supra_session_id` (current fields), PLUS:
   - `active_plan: {path | null}` — extends the file with plan-awareness
   - `last_wave_completed_at: {ts | null}` — proof-of-life per wave, not per cron heartbeat
   - This file is the "am I alive? what am I doing?" proof-of-life for the terminal; currently too thin to answer either question.

4. **`.claude/scratchpad/valuate/{date}.md`** — must contain:
   - Per-supra-session-id entry with intent + mode + key dims + focus + suppress
   - Written by every `/valuate` invocation — if not, valuate failed silently. (Today: 2026-04-18 has no valuate entry; plan substitutes but this is the edge-case not-the-norm.)

5. **`.claude/coordinators/messages/{date}/`** — must contain:
   - At minimum a HELLO message from every coordinator's Wave 0 (currently correct)
   - Rule-level transitions: path-claim narrowing, task-done, inter-coordinator requests SHOULD emit messages (currently rarely do)
   - **Messages older than 7 days must be archived** (currently NOT running — 2026-03-07 messages still present 42 days later).

**Invariants across writes:**
- If a claim file has `status: active` and `heartbeat_at > 30 min ago` → cleanup marks ended.
- If a claim file has `claimed_paths: ['*']` and scratchpad shows >1 OODA cycle → claim-squatting violation → log to ego.
- Supra session files never deleted; archived after 30 days.
- Messages archived after 7 days.

**Enforcement:**
- **New SessionStart hook** (or extension of existing): at session start, read all five components and surface violations to the coordinator's first observation:
  - Stale heartbeats (auto-end them)
  - Squatting claims from prior sessions (auto-end)
  - Null supra intent AND no active plan → prompt user to `/valuate`
  - Stale messages (>7 days) → archive
  - Session-state integrity report (five-line summary) injected into coordinator Wave 0 context.
- **Update `coordinator-coordination.md`**: delete references to `.claude/coordinators/sessions/` and `.claude/coordinators/supra/` (nonexistent dirs); point at `terminals/{pid}.yaml` as the canonical identity-read target.
- **Scheduled archive job** (CronCreate candidate): daily run archiving supra-sessions >30d and messages >7d.

**Fail mode**:
- Read-time: fail-open (missing file → neutral defaults).
- Wave 0: **fail-closed** — if supra intent is null AND no plan file is active AND no task description in `$ARGUMENTS`, refuse to proceed without asking user to `/valuate` or supply a task. This is the critical fail-closed gate: it prevents born-dead sessions.
- Claim-squatting: **fail-closed** — coordinator that has run >1 OODA cycle with `claimed_paths: ['*']` gets subsequent wave dispatches gated by "narrow your claims first."

---

## 5. Implementation Roadmap

Ordered by dependency + leverage. Each item: target surface, effort (S/M/L), payoff (S/M/L), owner.

| # | Action | Surface | Effort | Payoff | Owner | Priority | Depends on |
|---|---|---|---|---|---|---|---|
| 1 | Write `specs/README.md` index (per spec-writer §4) | ORIENT/specs | S | M | spec-writer | P0 | — |
| 2 | Add `## Status:` headers to each of 17 specs | ORIENT/specs | S | M | spec-writer | P0 | 1 |
| 3 | Update `coordinator-coordination.md` to reference `terminals/{pid}.yaml` (delete stale `sessions/`+`supra/` refs) | ORIENT/rules | S | M | coordinator | P0 | — |
| 4 | Extend `multi-agent-protocol.md` with Markov-completeness contract (items 1–7 for coordinator close-out) | ORIENT/rules | S | L | coordinator | P0 | — |
| 5 | Extend SubagentStop hook with `check_markov_completeness()` for coordinator close-out scratchpads | ORIENT/hooks | M | L | devops | P0 | 4 |
| 6 | Evict the 1200-word changelog paragraph from `codebase_graph.md` | ORIENT/codebase_graph | S | L | librarian | P0 | — |
| 7 | Split `codebase_graph.md` into thin TOC + per-stage child files (per librarian §P1) | ORIENT/codebase_graph | L | L | librarian | P1 | 6 |
| 8 | Consolidate `utils/paths.py` to one stage3-result convention (Convention B') | OBSERVE/code | M | L | stage3-analyst + devops | P0 | — |
| 9 | Add `config.yaml`+`run.yaml` sidecars to ProbeResultsWriter + build ClusterResultsWriter | OBSERVE/code | M | L | stage3-analyst | P1 | 8 |
| 10 | Promote `specs/run_provenance.md` Draft→Active; create `reports/experiments.csv`; populate retroactively | ORIENT/specs + OBSERVE/reports | M | L | devops | P1 | 2 |
| 11 | Figure sidecar contract — `.yaml` per figure; per-report nested `reports/figures/{slug}/` layout | OBSERVE/reports + ORIENT | M | M | stage3-analyst | P1 | 9 |
| 12 | Investigate 3 dead citations in `2026-03-08-causal-emergence-phase1.md` | OBSERVE | S | S | stage3-analyst | P1 | — |
| 13 | Write `scripts/archive/README.md` TOC (per qaqc §Index proposal) | ORIENT/scripts | M | M | qaqc | P1 | — |
| 14 | `git rm --cached` the 5 tracked training/probe logs; add `probe_log_*.txt`+`training_log_*.txt` to `.gitignore` | OBSERVE | S | S | devops | P1 | — |
| 15 | Delete strays: `nul`, ignored `.log`/`.txt` root files, `logs/*` 0-byte | OBSERVE | S | S | devops | P2 | — |
| 16 | Delete `img.png` (confirm with user first — modified 2026-04-17) | OBSERVE | S | S | devops + human | P2 | — |
| 17 | Delete `results [old 2024]/` (125MB; confirm with user per data-policy override) | OBSERVE | S | M | devops + human | P2 | — |
| 18 | After #13 written: delete ~15–17 DELETE-candidate archive files per qaqc assessment | OBSERVE | S | S | qaqc | P2 | 13 |
| 19 | Move `scripts/processing_modalities/extract_pois_from_history.py` + `osmium_time_filter.py` to `one_off/` (both declare Lifetime:temporary) | OBSERVE | S | S | qaqc | P2 | — |
| 20 | Archive `one_off/` scripts at 30-day shelf limit (accessibility_viz_all, analyze_accessibility_graphs, visualize_walk_accessibility — all 28d today) | OBSERVE | S | S | qaqc | P2 | — |
| 21 | Fix data-code violations: 2 `.log` files in `scripts/processing_modalities/alphaearth/`, test file in scripts, `scripts/preprocessing_auxiliary_data/cache/` | OBSERVE | S | S | qaqc | P1 | — |
| 22 | Decide: enforce `claimed_paths` narrowing via hook, OR remove the rule. (All 4 recent sessions violate it.) | ORIENT/rules | S | M | human | P1 | — |
| 23 | Extend `terminals/{pid}.yaml` with `active_plan` + `last_wave_completed_at` fields | ORIENT/session-state | S | M | devops | P1 | — |
| 24 | Scheduled archive job for `supra/sessions/` (>30d) + `coordinators/messages/` (>7d) | ORIENT/session-state | S | M | devops + CronCreate | P1 | — |
| 25 | Diff `specs/between-wave-pause-redesign.md` against `niche/SKILL.md`; mark historical or keep draft | ORIENT/specs | S | S | spec-writer | P2 | 2 |
| 26 | Split `specs/temporal-supra-profiles.md` (38KB) — theory→`memory/`, implementation→own spec | ORIENT/specs | M | M | spec-writer | P2 | 2 |

**Recommended execution order** (the "first pull" of leverage):
1. Items **1, 2, 3, 4, 6** — all small, ORIENT-side, unblock the rest. One focused session.
2. Item **5** (hook) — requires 1–4 to land first so its rules exist.
3. Item **8** (paths.py consolidation) — unblocks 9, 11, 12.
4. Items **7, 9, 10, 11, 13** in parallel — each owned by different agents, low cross-dependency.
5. P2 cleanup (14–21) as opportunistic weekend work.

**Implementation constraint**: per today's audit-only rule, none of these is executed today. Scheduling them to actually happen is itself an open problem (Class-6 calendar-day state loss); the Markov contracts exist precisely to prevent this roadmap from silently aging like the probe-confound backlog.

---

## 6. Anti-Patterns Observed

What the audit found we keep doing that we shouldn't:

1. **Append-only growth without retention contract.** Applies to `codebase_graph.md` (3228 lines, 23 rounds), `specs/` (17 files no index), `scripts/archive/` (53 files no index), `supra/sessions/` (22 files never archived), `coordinators/messages/` (42-day-old messages present). The pattern: something gets created with good intent, and the discipline to delete / compact / archive never materializes. **Fix shape**: every accreting artifact needs an explicit retention contract (ceiling + cadence). The Markov contracts in §4 supply these.

2. **Specs that should be rules lingering as 27KB narrative.** `coordinator_to_coordinator.md` (27KB spec) is superseded by `coordinator-coordination.md` (5KB rule) with 80% content overlap. The rule is load-bearing at runtime; the spec is rationale. Keeping both is fine IF the supersession is visible; currently it isn't. **Fix shape**: "spec = rationale, rule = runtime" dichotomy + explicit Superseded status pointers.

3. **Aspirational protocols universally violated.** All 4 recent claim files have `claimed_paths: ['*']` despite the rule requiring narrowing after first OODA cycle. Either the rule should be enforced or removed — don't keep it aspirational. Same pattern: `request`/`warning`/`done` messages predicted by coord-coord protocol almost never appear in the channel. **Fix shape**: rules that are 100% violated should be either enforced by a hook OR deleted. Keeping them as "discipline we should honor someday" is lie-by-omission.

4. **Claude-infra rule files referencing dead directories.** `coordinator-coordination.md` lines 85–86 point at `.claude/coordinators/sessions/` and `.claude/coordinators/supra/` which don't exist on disk (replaced by `terminals/{pid}.yaml` during the 2026-03-14 refactor). Classic doc rot. **Fix shape**: when refactoring an implementation, update the rule file same-commit.

5. **Crystallization gap — ACT products don't update ORIENT maps.** Training runs don't land in an experiment ledger. Figures don't have sidecars. Specs status doesn't update. This is the most expensive anti-pattern because the cost is invisible per-event but compounds: after 6 months you have 5 wandb runs you can't link to reports, 9 figures without provenance, 4 draft specs older than 30 days. **Fix shape**: make the crystallization write mandatory at the moment the ACT output is produced (hooks, writer classes).

6. **Avoidance as retention mechanism.** The 21-day-stale probe-confound backlog rolled silently across the 19-day gap between 2026-03-29 and 2026-04-17, then across the 1-day gap into today. Ego flagged it twice. Nobody surfaced it in chat. The `[open]` tag lacks age — a 21-day P0 reads identical to a same-session P0. **Fix shape**: aged `[open|Nd]` tags (Contract 1 item 4); `[open]` items >14d automatically promoted to session-start visibility.

7. **Post-Final-Wave dispatches.** 2026-04-17 green-branching-kelp had three additional waves after Final, each compounding the session's aesthetic failure. The protocol doesn't forbid post-Final work, but it also doesn't call it out as a scope drift signal. **Fix shape**: Final Wave commits a session-closed lock; further dispatches require explicit user override.

8. **Model-weak-at-task signals surfacing only post-hoc as memory files.** `memory/feedback_opus_47_weak_at_viz.md` was written after four waves of 2026-04-17 viz work failed. No mechanism exists for mid-session detection. **Fix shape**: out of current audit scope but worth noting — a "task-model fit" check at dispatch time.

---

## 7. Open Questions for Human

The audit identified decisions it cannot make unilaterally:

1. **`img.png` at repo root (521KB, modified 2026-04-17)** — possibly a viz output from yesterday's green-branching-kelp session. Delete OR move to `reports/figures/`? Confirm before action.

2. **`results [old 2024]/` deletion (125MB)** — no reports cite it, `specs/experiment_paths.md` migrates away from it, dir is literally labeled "old". But MEMORY.md policy says "never delete data results." This feels like the policy's intended exception (explicit old + superseded + uncited), but an exception still needs approval. Confirm: delete entirely, or move somewhere explicit like `data/_archive_2025/`?

3. **3 dead citations in `2026-03-08-causal-emergence-phase1.md`** (`multiscale_probe_results.csv`, `multiscale_comparison.png`, `multiscale_delta.png`) — if files were generated and deleted, the "never delete data" policy was violated. If never generated, the report has always had dead refs. Investigation needed; decision on whether to patch report text or attempt to regenerate the figures.

4. **Claim-squatting rule — enforce or remove?** All 4 recent claim files have `claimed_paths: ['*']`. The rule in `coordinator-coordination.md` requires narrowing after first OODA cycle. The rule is 100% violated. Option A: enforce via hook (coordinator dispatch gated on narrow claim). Option B: remove the rule as aspirational. Cannot keep both the rule and its universal violation.

5. **`specs/between-wave-pause-redesign.md` absorption status** — did `niche/SKILL.md` absorb the Wave Results format? Requires a short research task (diff spec against skill) before the spec can be marked Historical or kept as Draft.

6. **Canonical "see here first" Claude-infra spec pointer.** CLAUDE.md line 205 currently points at `specs/claude_code_multi_agent_setup.md` (Feb 13, partially stale — pre-PPID, pre-supra, pre-niche rename). Two options: (a) update CLAUDE.md to point at the newer `session-identity-architecture.md` + `temporal-supra-profiles.md`, OR (b) update `claude_code_multi_agent_setup.md` with a "current state" preamble. Spec-writer Wave 1 recommended (b) — human confirm.

7. **Execution scheduling for this roadmap.** The roadmap has 26 items. Without a commitment mechanism, it will silently age like the probe-confound backlog. Proposal: schedule items 1–6 (the "first pull") as the next focused weekend session. Human consent?

8. **21-day probe-confound backlog from 2026-03-29.** Five `[open]` items carried silently across two sessions. Ego has flagged this twice. Needs explicit human decision: (a) prioritize and work through, (b) explicitly defer with a new target date, OR (c) declare abandoned and close out. "Quiet persistence" is the current default and the worst option.

---

## Appendix: Source scratchpads

All Wave 1 surface audits at:
- `.claude/scratchpad/librarian/2026-04-18-golden-standing-creek.md` (Theme G, 259 lines)
- `.claude/scratchpad/qaqc/2026-04-18-golden-standing-creek.md` (Themes A-scripts + D, 219 lines)
- `.claude/scratchpad/spec-writer/2026-04-18-golden-standing-creek.md` (Theme C, 253 lines)
- `.claude/scratchpad/stage3-analyst/2026-04-18-golden-standing-creek.md` (Themes E+F, 335 lines)
- `.claude/scratchpad/devops/2026-04-18-golden-standing-creek.md` (Themes A-repo + B, 207 lines)
- `.claude/scratchpad/ego/2026-04-18-golden-standing-creek.md` (Theme H + Markov contract drafts, 258 lines)

Plan: `.claude/plans/2026-04-18-organizational-flywheel-audit.md` (173 lines)

---

*This report is itself a map artifact. It is the DECIDE substrate's output for the golden-standing-creek session. Future sessions consume it via `specs/` (after human approves promoting its contracts to specs).*
