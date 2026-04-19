# Cluster 2 — Paths Consolidation (placeholder, 2026-04-18)

| Field | Value |
|---|---|
| **Status** | DRAFT — to be perfected next session |
| **Source** | `reports/2026-04-18-organizational-flywheel-audit.md` §5 roadmap item #8 |
| **Parent plan** | `.claude/plans/2026-04-18-flywheel-gyroscope-fix.md` (cluster 1, executed `99099c0`) |
| **Cluster** | 2 — Territory (touches `utils/` + `scripts/`; can break pipelines; gets qaqc gate) |
| **Priority order** | FIRST of cluster 2 — unblocks #9, #11, #12 |

## Why this plan exists

Audit Theme G named `utils/paths.py` as the single source of truth for study-area paths, but many `scripts/*.py` still hardcode `data/study_areas/{area}/...` strings. Those hardcodes drift: when the layout changes, grep-and-replace is incomplete, and the `codebase_graph` stops matching reality.

## Scope (to be tightened next session)

- Audit every `scripts/**/*.py` for hardcoded `data/study_areas/...` strings
- Migrate all to `StudyAreaPaths` methods from `utils/paths.py`
- Add missing methods to `StudyAreaPaths` if scripts need paths the helper doesn't yet expose
- qaqc gate: grep should return zero hardcoded study-area paths outside `utils/paths.py` after migration

## Out of scope

- Sidecars, ledger, archive index — those are sibling cluster-2 plans (see `2026-04-18-cluster2-ledger-sidecars.md` and `2026-04-18-cluster2-housekeeping.md`)
- Any `.claude/` or `specs/` work — cluster 1 already done

## Next-session TODO

1. Flesh out wave structure (probably: W1 audit, W2 migrate, W3 qaqc verify, Final)
2. Name the specific agents (`librarian` for audit, `stage1/2/3-*` or `Explore` for migration, `qaqc` for verification)
3. Write acceptance criteria per wave
4. Narrow claimed_paths (`scripts/**`, `utils/paths.py`, relevant `stage*/` only if a script wraps a module)

## Execution

```
/clear
/niche follow plan .claude/plans/2026-04-18-cluster2-paths-consolidation.md
```

(Not ready to execute until the wave structure is filled in.)
