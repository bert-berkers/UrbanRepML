# Terminal B — Housekeeping + Hook Dev (warmstart, 2026-04-19)

| Field | Value |
|---|---|
| **Status** | READY — warmstart for next `/valuate` in an unclaimed terminal |
| **Terminal ID** | B (PyCharm tab name: **B - housekeeping**) |
| **Source** | `.claude/scratchpad/valuate/2026-04-19.md` §Terminal B + `reports/2026-04-18-organizational-flywheel-audit.md` §7 Q1/Q2/Q4 + §5 roadmap items #14–21, #24 + R4 |
| **Parent plan** | `.claude/plans/2026-04-18-cluster2-housekeeping.md` (placeholder; this plan supersedes) |
| **Cluster** | 2 — Repo hygiene + hook dev (low-risk, no empirical edge, no forensics) |
| **Peer terminals** | A (paths, claimed), C (probe), D (spec-governance) |
| **Est** | 1–2h |

## Characteristic-state calibration

```
mode=focused speed=3 explore=2 quality=4 tests=3 spatial=2 model=2 urgency=3 data_eng=3
intent="Repo hygiene + claim-narrowing hook — Q1/Q2/Q4 + #14-21/#24/R4"
focus=[housekeeping, Q4-hook, repo-root-cleanup, archive-cron]
suppress=[paths-consolidation, probe-confound, spec-governance, Q3-forensics]
```

Shorthand: `/valuate focused, quality 4, tests 3, urgency 3, focus "housekeeping + Q4 hook"`

## Claimed paths (narrow at first OODA cycle — do NOT leave as `['*']`)

- `.claude/hooks/**` (Q4 enforce-via-hook + R4 archive cron)
- `archive/**` (item #14 archive index)
- Repo-root stray binaries: `img.png`, `results [old 2024]/` (Q1, Q2)
- `data/` repo-root stray files only (NOT `data/study_areas/**` — that's A)

**Explicitly not claimed**: `scripts/**` (A), `stage3_analysis/**` (C), `reports/**` (C+D), `specs/**` (D), `CLAUDE.md` (D for Q6).

## Wave structure

### W1 — Audit (librarian + devops in parallel)

- **librarian**: scan `archive/` — list contents, group by origin, note which are git-tracked vs untracked. Draft `archive/INDEX.md` structure.
- **devops**: confirm `img.png` and `results [old 2024]/` are on disk, get sizes + last-modified, check git-tracked status. Dump one-liner summary.

Acceptance: `archive/INDEX.md` draft outline written to scratchpad; `img.png` + `results [old 2024]/` metadata captured.

### W2 — Q1/Q2 repo-root cleanup (devops)

- **Q1 — `img.png`**: delete. (Audit §7 Q1 flagged; it's a stray viz output. 521KB binary, modified 2026-04-17.)
- **Q2 — `results [old 2024]/` (125MB)**: **move** to `results/archive/2024/` per `memory/feedback_no_delete_data.md` ("never delete data results — only organize"). Confirm with `git status` that nothing inside was tracked (should be gitignored — verify).
- Commit after each move with a dedicated message.

Acceptance: repo root lists only expected entries; `git status` clean after both moves; no data files lost.

### W3 — Q4 claim-narrowing hook (devops)

- **Target hook**: `.claude/hooks/stop.py` or a new `subagent-mid-ooda-check.py` — whichever fires after first OODA cycle.
- **Detection logic**: read the active terminal's `.claude/coordinators/session-*.yaml`, check `claimed_paths`. If `['*']` AND `ooda_cycle >= 2`, raise `[needs:human] Claim still `['*']` past first OODA — narrow per coordinator-coordination.md rule`.
- **Fail mode**: warning only (not blocking) — surfaces the issue without breaking sessions that have legitimate reason.
- Unit test: fake session YAML with `['*']` + cycle=2 → warning; cycle=1 → silent; narrowed claims → silent.

Acceptance: hook fires on the fake YAML test case; `.claude/rules/coordinator-coordination.md` updated with "enforced via hook" note.

### W4 — R4 daily archive cron (devops)

- **Gate**: `.last_archive_sweep` file at `.claude/coordinators/.last_archive_sweep` (plain ISO-8601 timestamp).
- **Check in `.claude/hooks/session-start.py`**: if `now - last_sweep > 24h`, run sweep.
- **Sweep rules**:
  - `.claude/supra/sessions/*.yaml` with `last_attuned > 30d` → move to `.claude/supra/sessions/archive/`
  - `.claude/coordinators/messages/{date}/` where `date > 7d ago` → move to `.claude/coordinators/messages/archive/{date}/`
- **Preserve-don't-delete**: move, never delete. Per `feedback_no_delete_data.md`.

Acceptance: sweep runs on first SessionStart >24h after sweep timestamp; archive dirs populated with expected counts; `git status` shows no accidental data loss.

### W5 — Item #14 archive index (librarian)

- Write `archive/INDEX.md` from W1 audit. One line per top-level archive entry: `- {path} — {origin} — {date archived} — {why}`.
- Commit alongside Q1/Q2 or as its own commit.

Acceptance: `archive/INDEX.md` exists, covers all `archive/` top-level entries.

### Final Wave — close-out

- Coordinator scratchpad entry with all 7 Markov-completeness items (see `.claude/rules/multi-agent-protocol.md`).
- `done` messages to peer terminals A, C, D summarizing: what landed, any new open items.
- Check: `git status` committable; all edits in claimed paths; no collateral damage to peers' domains.

## Out of scope

- Q3 dead-figure citations (D owns — spec-governance theme)
- Paths consolidation (A)
- Any probe/model work (C)
- Spec editing (D)
- Q7 roadmap scheduling (dissolved per 2026-04-19 valuate)

## Execution

```
/clear
/valuate focused, quality 4, tests 3, urgency 3, focus "housekeeping + Q4 hook"
/niche follow plan .claude/plans/2026-04-19-terminal-b-housekeeping.md
```
