# Cluster 2 — Ledger + Sidecars + Figure Provenance (placeholder, 2026-04-18)

| Field | Value |
|---|---|
| **Status** | DRAFT — to be perfected next session |
| **Source** | `reports/2026-04-18-organizational-flywheel-audit.md` §5 roadmap items #9, #10, #11, #12 |
| **Parent plan** | `.claude/plans/2026-04-18-flywheel-gyroscope-fix.md` (cluster 1, executed `99099c0`) |
| **Cluster** | 2 — Territory (touches `stage*/` + `scripts/` + `data/study_areas/` sidecar layout) |
| **Depends on** | `2026-04-18-cluster2-paths-consolidation.md` (all sidecar paths go through `StudyAreoPaths`) |

## Why this plan exists

The audit's ACT/ORIENT surfaces rot because experiments leave no machine-readable trace of what produced them. Per-experiment sidecars (a `run.yaml` alongside every generated artifact) + a central ledger (one append-only row per experiment run) + figure provenance (every PNG knows its source config) close that gap.

## Scope (to be tightened next session)

- **Item #9 — Per-experiment sidecars**: every generated artifact (embedding, model checkpoint, probe result, figure) gets a companion `*.run.yaml` with: git commit, config hash, input paths, seed, wall time, producing script.
- **Item #10 — Experiment ledger**: append-only `data/ledger/runs.jsonl` (one row per completed run). Consumed by stage3 analysis for cross-run comparison.
- **Item #11 — Figure provenance**: every figure written by `stage3_analysis/*` calls a shared `save_figure(fig, path, sources=[...])` helper that writes a `*.provenance.yaml` sidecar.
- **Item #12 — Config hashing convention**: reproducible hash over a canonicalised config dict; stable across reruns with identical inputs.

## Out of scope

- Paths consolidation — sibling plan, must land first
- Archive index, repo-root cleanup — sibling housekeeping plan

## Next-session TODO

1. Decide: one big plan or split into four (#9, #10, #11, #12 each its own)?
2. Sidecar schema — draft the minimum fields, leave optional ones open for iteration
3. Ledger format — JSONL vs SQLite? JSONL is simpler; SQLite wins when reading cross-run
4. Pick a prototype stage first (stage1? stage3 probes?) and materialize end-to-end before touching others
5. qaqc gate: every new artifact has a sidecar; ledger row count == number of completed runs

## Execution

```
/clear
/niche follow plan .claude/plans/2026-04-18-cluster2-ledger-sidecars.md
```

(Not ready to execute — needs paths consolidation first + scope split decision.)
