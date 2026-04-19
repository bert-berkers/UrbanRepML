# Terminal A — Paths Consolidation (warmstart, 2026-04-19)

| Field | Value |
|---|---|
| **Status** | CLAIMED by `calm-lifting-dew-2026-04-19` at 14:39 |
| **Terminal ID** | A (PyCharm tab name: **A - paths**) |
| **Session suffix** | `calm-lifting-dew` (this terminal's coordinator session) |
| **Source** | `.claude/scratchpad/valuate/2026-04-19.md` §Terminal A + `reports/2026-04-18-organizational-flywheel-audit.md` §5 roadmap item #8 |
| **Parent plan** | `.claude/plans/2026-04-18-cluster2-paths-consolidation.md` (placeholder; this plan supersedes with wave structure) |
| **Cluster** | 2 — Territory refactor (touches `utils/` + `scripts/`; can break pipelines; qaqc gated) |
| **Peer terminals** | B (housekeeping), C (probe), D (spec-governance) |
| **Est** | 2–3h |
| **Priority** | FIRST of cluster 2 — unblocks ledger-sidecars (deferred to next weekend) |

## Characteristic-state calibration

```
mode=focused speed=3 explore=2 quality=5 tests=4 spatial=3 model=2 urgency=4 data_eng=5
intent="Migrate hardcoded data/study_areas/ paths in scripts/** to StudyAreaPaths"
focus=[paths-consolidation, utils/paths.py, scripts-audit, qaqc-gate]
suppress=[ledger-sidecars, probe-confound, housekeeping, spec-governance]
```

Shorthand: `/valuate focused, quality 5, tests 4, urgency 4, data_eng 5, focus "paths consolidation"`

## Claimed paths (narrowed from `['*']` at first OODA cycle)

- `scripts/**` (migration target)
- `utils/paths.py` (may need new methods)
- `utils/__init__.py` (re-exports only if needed)

**Explicitly not claimed**: `stage*/` (read-only reference), `data/` (the tree itself — unchanged), `.claude/hooks/**` (B), `stage3_analysis/**` (C), `specs/**` (D), `reports/**` (B+C+D split).

## Wave structure

### W1 — Audit (librarian + Explore in parallel)

- **librarian**: grep `scripts/**/*.py` for hardcoded `data/study_areas/`, `data\study_areas\`, or f-string constructions of same. Dump a list: `{file}:{line} — {the literal path string}`.
- **Explore**: read `utils/paths.py` — enumerate the `StudyAreaPaths` methods already exposed. Note gaps that the audit grep will likely need.

Acceptance: concrete call-site list in scratchpad; `StudyAreaPaths` inventory documented.

### W2 — Expand StudyAreaPaths if needed (stage1-modality-encoder or devops)

If W1 audit reveals call sites that need paths `StudyAreaPaths` doesn't expose:
- Add the missing methods with clear names + docstrings
- Keep the existing API backward-compatible; append-only

Acceptance: all audit call-sites have a corresponding `StudyAreaPaths` method.

### W3 — Migrate (Explore or relevant stageN agent per script)

- Batch by script domain: POI scripts, road scripts, GTFS scripts, analysis scripts, etc.
- Replace each hardcoded path with the `StudyAreaPaths` method call
- Preserve script behavior exactly — no surprise refactors

Acceptance: every audited call-site replaced; scripts still runnable (no import errors).

### W4 — qaqc verify (qaqc)

- Grep verify: `rg 'data[/\\]study_areas' scripts/` returns empty (or only intentional exceptions)
- Import smoke test: `python -c "import scripts.{module}"` for touched modules
- Any test suites covering touched scripts still pass

Acceptance: qaqc produces commit-readiness verdict; zero residual hardcodes in `scripts/` outside justified exceptions.

### Final Wave — close-out

- Coordinator scratchpad entry with all 7 Markov-completeness items
- `done` messages to B, C, D: "paths consolidation landed in commit {sha}; ledger-sidecars now unblocked for next weekend"
- Commit in one or a small number of coherent commits with clear messages

## Out of scope

- Sidecars, ledger, archive index (sibling cluster-2 plans, must land after this)
- Any `.claude/` or `specs/` work (cluster 1 done; D handles lingering spec questions)
- Moving scripts between `scripts/` tiers (`one_off/`, `archive/`) — that's B's housekeeping scope

## Execution

```
(already claimed; Terminal A is live)
/niche follow plan .claude/plans/2026-04-19-terminal-a-paths-calm-lifting-dew.md
```
