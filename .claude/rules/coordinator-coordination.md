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
1. Delete your claim file via `coordinator_registry.delete_claim()`.
2. Optionally write a `done` message summarizing what was accomplished and what changed.
3. Run `coordinator_registry.cleanup_stale()` to remove crashed sessions' artifacts.

## Anti-Patterns (Do Not Do These)

- **Claim squatting**: Leaving `claimed_paths: ["*"]` beyond the first OODA cycle.
- **Hijacking**: Modifying files another active coordinator claims without checking or warning.
- **Inferential imperialism**: Writing specs or assertions about paths outside your claimed_paths.
  If you need cross-domain information, read the code (read-only) or leave a `request` message.
- **Message spam**: Writing more than one message per state transition per OODA cycle.

## Staleness Thresholds

| Threshold | Meaning |
|-----------|---------|
| 30 minutes | Claim treated as stale -- proceed with info log, do not block |
| 2 hours | Claim deleted by next session's cleanup |
| 7 days | Messages purged by cleanup |

## Files

| Path | Purpose |
|------|---------|
| `.claude/coordinators/session-{id}.yaml` | Active claim per session |
| `.claude/coordinators/messages/{ts}-{id}.yaml` | Individual message files |
| `.claude/hooks/coordinator_registry.py` | Shared library (I/O functions) |

The `.claude/coordinators/` directory is gitignored -- ephemeral runtime state only.
