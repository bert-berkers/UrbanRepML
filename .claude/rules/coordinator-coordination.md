---
paths:
  - ".claude/**"
  - "skills/**"
---

# Coordinator-to-Coordinator Coordination Protocol

When multiple Claude Code sessions run concurrently on this codebase, each coordinator
registers a claim file in `.claude/coordinators/`. This rule applies to all coordinator
sessions and to the `/coordinate` skill.

## Three-Scale Cognitive Architecture

This protocol operates at the **lateral** scale of a three-level cognitive system:

| Scale | Entity | Temporal Reach | Communication |
|-------|--------|---------------|---------------|
| **Supra** | Human user | Longest — across all workstreams and sessions | Direct chat, task delegation |
| **Lateral** | Coordinators (peers) | Mid — session-scoped | This protocol: claims, messages, heartbeats |
| **Vertical** | Specialist agents | Shortest — task-scoped | Scratchpads, SubagentStart/Stop hooks |

The human is the **supra-coordinator** — the apex of the cognitive light cone. They see across all concurrent coordinator sessions. When lateral coordination fails (conflicting claims, deadlock, cross-workstream dependency), **escalate to the human** rather than attempting autonomous resolution.

**P0 escalation rule**: Each coordinator handles P0 items within its claimed paths autonomously. P0 items that span multiple coordinator domains escalate to the human for assignment — the human decides which coordinator owns it.

Coordinator-to-coordinator messages are "bioelectric signals" between cells. They should be:
- **Sparse**: one message per state transition, not a running commentary
- **Actionable**: include what changed and what the recipient should do about it
- **Gradient-like**: signal strength (info < warning < request) indicates priority

## Identity: One Per Terminal (since 2026-05-03)

A terminal has one identity for its lifetime. `/clear` is a context flush, not a lifecycle event — same terminal, same identity, fresh context. Two components compose it:

1. **SessionStart-minted poetic name** — the identity (e.g. `pale-listening-dew`) issued when the hook first fires for a fresh terminal. Source of truth.
2. **`coordinators/terminals/{pid}.yaml`** — PID-keyed record carrying the canonical field `identity_id`. SessionStart re-injects the same identity on every fire (after `/clear`, after compact, etc.); `stop.py` does NOT clear it.

Legacy fields `session_id` and `supra_session_id` are accepted on read for backward compat (live terminal files written before the collapse may carry one or both). Writers always emit `identity_id`. The supra valuation file lives at `.claude/supra/sessions/{identity_id}.yaml` — identity IS the supra layer, no separate concept.

If `terminals.yaml` and SessionStart-injected name disagree, SessionStart wins — rewrite the terminals file, don't rewrite SessionStart. See `scratchpad/coordinator/notes.md` §"2026-04-19 — Failure Mode: Identity Tagging Drift" and `specs/session-identity-architecture.md`.

## Protocol Obligations

### On Session Start
1. Read all existing `session-*.yaml` files in `.claude/coordinators/`.
2. Note any active (non-stale) coordinators and their claimed paths.
3. Surface active coordinator summaries to the user at the start of the first OODA cycle.
4. Narrow your own claimed_paths from the initial `["*"]` to your actual working domain
   within the first OODA cycle. Leaving `["*"]` is claim squatting.

### Before Modifying Any File
1. Check active claims using `coordinator_registry.check_conflict()`.
2. If a conflict with an active session is found, warn the user. Do not silently proceed.
3. If the conflicting session is stale (heartbeat > 30 min), proceed but log a message.
4. If the user says proceed despite a live conflict, log a `warning` message to the
   other coordinator's domain.

### During Work
- Update `heartbeat_at` via `coordinator_registry.update_heartbeat()` at each OODA cycle.
- Update `active_agents` in your claim file when dispatching or completing subagents.
- Read new messages addressed to your session_id or "all" at each OODA OBSERVE phase.

### On State Transitions
Leave messages for events that matter to other coordinators:
- `info`: "Starting refactor of utils/paths.py -- adding study_area_root() method"
- `warning`: "Modified stage2_fusion/models/full_area_unet.py -- embedding dim changed to 128"
- `request`: "Need utils/paths.py stable until 15:00 -- can you defer your changes?"
- `done`: "Refactor complete. New method: study_area_root(area_name) -> Path"

Keep messages sparse. Routine status belongs in your scratchpad, not the message log.

### On Session End
1. Mark your claim as ended via `coordinator_registry.delete_claim()` (sets `status: ended`, does not delete the file).
2. Optionally write a `done` message summarizing what was accomplished and what changed.
3. Run `coordinator_registry.cleanup_stale()` to archive stale sessions' artifacts.

## Anti-Patterns (Do Not Do These)

- **Claim squatting**: Leaving `claimed_paths: ["*"]` beyond the first OODA cycle.
  > Enforced via hook: a warning is emitted by `subagent-stop.py` when a session's
  > `claimed_paths` remain `['*']` more than 15 minutes after `started_at`. See
  > `.claude/hooks/subagent-stop.py` (`check_claim_narrowing`).
- **Hijacking**: Modifying files another active coordinator claims without checking or warning.
- **Inferential imperialism**: Writing specs or assertions about paths outside your claimed_paths.
  If you need cross-domain information, read the code (read-only) or leave a `request` message.
- **Message spam**: Writing more than one message per state transition per OODA cycle.
- **Skills writing identity-bearing files from a forked subagent context**: `/valuate` and `/niche` must run inline in the coordinator's context, not as dispatched subagents. A forked context has the wrong PID parentage and writes identity to the wrong terminal file, producing supra-ghosts (coordinators that have valuation but no session, or vice versa).

## Supra-Ghost Recovery Protocol

Supra-ghost = valuation/session identity mismatch between `coordinators/terminals/{pid}.yaml` and the SessionStart-issued `session_id`. Symptoms: `/valuate` targets a supra that no live terminal claims; two terminals share a supra file; `supra_reader` returns an empty state despite an active session.

**Recovery steps (fix-forward, never rewrite history):**

1. **Write the correct supra yaml directly** at `supra/sessions/{correct_supra_session_id}.yaml`, inferring values from SessionStart + the most recent valuate invocation. Do NOT delete joint files; they are historical record.
2. **Note the pivot in the coordinator close-out** as a prior-entries line — e.g. `HH:MM — supra-ghost pivot: {wrong_supra_id} → {correct_supra_id}`. The pivot is Markov-completeness-relevant.
3. **Do NOT rewrite existing files with the wrong id**. They belong to the timeline that produced them. Fix forward; preserve as evidence.
4. **If the skill ran as a forked subagent** (root cause of the 2026-04-19 incident), fix the skill's invocation path first (`.claude/skills/{name}/SKILL.md` should specify inline execution) — otherwise the ghost will recur next session.

Cross-reference: commit `d077c25` (skills inline fix), `scratchpad/coordinator/notes.md` §"Failure Mode: Identity Tagging Drift", `specs/session-identity-architecture.md`.

## Staleness Thresholds

| Threshold | Meaning |
|-----------|---------|
| 30 minutes | Claim treated as stale -- proceed with info log, do not block |
| 2 hours | Claim marked as ended by next session's cleanup |
| 7 days | Ended claims archived to `coordinators/archive/` |

## Daily Archive Sweep

> Archive sweep: a daily cron in `.claude/hooks/session-start.py` moves supra
> sessions (last_attuned > 30d) to `supra/sessions/archive/` and message dirs
> (date > 7d ago) to `messages/archive/{date}/`. Gated by `.last_archive_sweep`
> timestamp. Preserve-don't-delete — nothing is removed.

Implemented in `.claude/hooks/archive_sweep.py` (`maybe_run_sweep()`). Key properties:
- Gate file: `.claude/coordinators/.last_archive_sweep` (plain ISO-8601 UTC timestamp)
- Runs at most once per 24 hours; skipped silently if gate is recent
- Fail-open: per-item errors logged to stderr, sweep continues; gate always rewritten
- Live terminal protection: supra sessions referenced in `coordinators/terminals/*.yaml` are never moved, even if stale
- Sessions with missing/malformed `last_attuned` are skipped (don't guess)

## Files

| Path | Purpose |
|------|---------|
| `.claude/coordinators/session-{id}.yaml` | Claim per session (active or `status: ended`) |
| `.claude/coordinators/terminals/{pid}.yaml` | Terminal-PID-keyed identity carrying the canonical `identity_id` (written by `coordinator_registry.write_ppid_identity()`; archived to `terminals/archive/` when the terminal is reaped). One identity per terminal for the terminal's lifetime — `/clear` does not rotate it. Legacy fields `session_id` / `supra_session_id` are accepted on read. |
| `.claude/coordinators/messages/{date}/{ts}-{id}.yaml` | Per-day message files |
| `.claude/hooks/coordinator_registry.py` | Shared library (I/O, PID helpers) |

The `.claude/coordinators/` directory is gitignored -- ephemeral runtime state only.

> **Note on the terminal-PID scheme**: earlier drafts of this protocol referenced `.claude/coordinators/sessions/{id}.{ppid}` and `.claude/coordinators/supra/{id}.{ppid}`. Both were consolidated into the single `terminals/{pid}.yaml` file in March 2026 (see auto-memory `project_desktop_app_quirks.md`). On 2026-05-03 the file was further simplified to a single `identity_id` field — the prior session/supra split is gone because `/clear` no longer rotates anything (see `specs/session-identity-architecture.md`).
