# Cluster 2 — Housekeeping (placeholder, 2026-04-18)

| Field | Value |
|---|---|
| **Status** | DRAFT — to be perfected next session |
| **Source** | `reports/2026-04-18-organizational-flywheel-audit.md` §5 roadmap items #14–21, #24 + cluster 1 deferrals (R4, item 24) + repo-root cleanup |
| **Parent plan** | `.claude/plans/2026-04-18-flywheel-gyroscope-fix.md` (cluster 1, executed `99099c0`) |
| **Cluster** | 2 — Territory + infrastructure (low-risk cleanup, but touches `data/` and repo root) |
| **Priority order** | LAST of cluster 2 — do after paths + sidecars land |

## Why this plan exists

A grab-bag of low-risk cleanup items that the audit flagged and that the cluster-1 fix chose to defer rather than mix into the Markov-contract work. None of these is urgent individually; together they keep the repo root and `data/` layout matching the documented conventions.

## Scope (to be tightened next session)

- **Items #14–17 — Archive index + data-code violation sweep**: grep for residual data paths inside `stage*/` and `scripts/**`; add an `archive/INDEX.md` listing what's parked there and why.
- **Items #18–21 — Repo-root cleanup**:
  - `img.png` at repo root — delete (audit §7 flagged)
  - `results [old 2024]/` dir at repo root — **move to `results/archive/2024/` per `feedback_no_delete_data.md` (never delete data)**
  - Any other stray top-level files that don't belong
- **Item #24 + R4 — Daily archive cron** (deferred from cluster 1):
  - Add to `.claude/hooks/session-start.py` a ≥24h-since-last-sweep gate
  - Archive `supra/sessions/*.yaml` with `last_attuned` > 30d → `supra/sessions/archive/`
  - Archive `coordinators/messages/{date}/` directories older than 7d → `coordinators/messages/archive/`
  - Needs new `.last_archive_sweep` state file + `last_attuned`-aware YAML scanning (devops flagged as non-trivial in cluster-1 W3)

## Out of scope

- Paths consolidation and sidecars — sibling cluster-2 plans, must land first
- Any `.claude/rules/` or `specs/` work — cluster 1 done

## Next-session TODO

1. Split into two if wave structure gets messy: (a) repo-root + data-violation cleanup, (b) hook-level archive cron
2. Confirm `results [old 2024]/` → `results/archive/2024/` move matches the `feedback_no_delete_data.md` preservation rule
3. Decide `.last_archive_sweep` format: plain text timestamp, or structured yaml?
4. qaqc gate: after sweep, `git status` shows no accidental data loss; archive dirs populated with expected counts

## Execution

```
/clear
/niche follow plan .claude/plans/2026-04-18-cluster2-housekeeping.md
```

(Not ready to execute — do paths + sidecars first.)
