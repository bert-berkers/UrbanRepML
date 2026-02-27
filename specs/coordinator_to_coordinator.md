# Coordinator-to-Coordinator Communication Protocol

## Status: v1.2 — Implemented (2026-02-27)

## Context

UrbanRepML uses a single-coordinator model today: one Claude Code session runs as coordinator, delegates to specialists, and writes its decisions to `.claude/scratchpad/coordinator/YYYY-MM-DD.md`. The hooks architecture (`specs/hooks_architecture.md`) and multi-agent setup (`specs/claude_code_multi_agent_setup.md`) assume a single coordinator per session.

When a user opens multiple Claude Code terminals on the same codebase -- one doing multi-modality implementation in `stage1_modalities/`, another doing SRAI terminology fixes across `stage2_fusion/` -- four problems emerge:

1. **Invisible overlap.** Neither coordinator knows the other exists. Both might edit `utils/paths.py` or a shared `__init__.py`.
2. **Accidental hijacking.** Coordinator A renames a function in `utils/spatial_db.py`. Coordinator B, unaware, writes code calling the old name. The result compiles for B's session but breaks on merge.
3. **Inferential imperialism.** Coordinator A, working on Stage 1, assumes Stage 2 embeddings have shape `(N, 64)` and writes a spec saying so. Coordinator B, who owns Stage 2, has just changed the shape to `(N, 128)`. A's spec is wrong but looks authoritative.
4. **No message passing.** If A needs B to hold off on a directory while A refactors, there is no channel to say so. The only option is the user manually telling B.

These problems are not theoretical. The ego assessment flagged duplicate script confusion (`scripts/plot_linear_vs_dnn_comparison.py` vs `scripts/one_off/` version) -- exactly the kind of overlap that happens when two sessions do not coordinate.

All four problems are now addressed by the advisory file-based protocol implemented in commits `52a7836` (initial protocol), `d176a37` (Levin enhancements), and `75dc879` (continuous lateral awareness).

## Decision

Introduce an advisory file-based protocol using `.claude/coordinators/` as the communication directory. Each active coordinator session registers a **claim file** declaring its identity, domain, and claimed paths. A **message log** allows coordinators to leave notes for each other. Hooks are extended to register on start and deregister on stop.

The protocol is advisory, not enforcing: coordinators check claims before modifying files and warn the user on conflicts, but do not hard-block. Hard filesystem locks are brittle (crash leaves stale lock, no built-in timeout), and the cost of a false lock-out exceeds the cost of an occasional overlap that the user resolves.

## Protocol Overview

```
Session A starts                    Session B starts
    |                                    |
    v                                    v
Register claim file                 Register claim file
(.claude/coordinators/              (.claude/coordinators/
 session-A.yaml)                     session-B.yaml)
    |                                    |
    v                                    v
Work: before modifying file F,      Work: before modifying file F,
  scan all claim files for F.         scan all claim files for F.
  If claimed by another session:      If claimed by another session:
    -> warn user                        -> warn user
    -> optionally leave message         -> optionally leave message
    |                                    |
    v                                    v
Session A ends                      Session B ends
    |                                    |
    v                                    v
Delete claim file                   Delete claim file
+ Leave farewell message            + Leave farewell message
  (optional)                           (optional)
```

## Data Structures

### Claim File

**Path**: `.claude/coordinators/session-{id}.yaml`

The `{id}` is a human-memorable word combination in the format `{adjective}-{participle/adjective}-{noun}` (e.g., `gentle-amber-tide`, `swift-drifting-maple`). Generated randomly from curated word lists at session start. The names are pleasant, surprising, and distinguishable at a glance -- designed so a user glancing at `.claude/coordinators/` can instantly tell sessions apart without decoding timestamps.

```yaml
# .claude/coordinators/session-gentle-amber-tide.yaml
session_id: "gentle-amber-tide"
started_at: "2026-02-27T14:30:22"
heartbeat_at: "2026-02-27T14:45:10"     # Updated periodically
task_summary: "Multi-modality Stage 1 implementation"  # Human-readable
domain:
  primary: "stage1_modalities"            # Top-level domain name
  description: "Adding POI and Roads modality encoders"
claimed_paths:
  - "stage1_modalities/**"
  - "scripts/processing_modalities/**"
  - "utils/paths.py"                       # Specific shared files if needed
read_only_paths:                           # Paths this session reads but does not write
  - "stage2_fusion/models/**"
  - "utils/spatial_db.py"
active_agents:                             # Currently running subagents
  - agent_type: "stage1-modality-encoder"
    task: "Implement POI processor"
    started_at: "2026-02-27T14:35:00"
```

**Field semantics:**

| Field | Required | Purpose |
|-------|----------|---------|
| `session_id` | Yes | Unique session identifier |
| `started_at` | Yes | ISO timestamp of session start |
| `heartbeat_at` | Yes | Last-updated timestamp; used for staleness detection |
| `task_summary` | Yes | Human-readable description for the user |
| `domain.primary` | Yes | Short domain name (e.g., `stage1_modalities`, `stage3_analysis`) |
| `domain.description` | Yes | What the coordinator is doing, in a sentence |
| `claimed_paths` | Yes | Glob patterns for directories/files this session may write |
| `read_only_paths` | No | Paths this session reads but will not modify |
| `active_agents` | No | Currently running subagents, updated during work |

### Message Log

**Path**: `.claude/coordinators/messages/{timestamp}-{session_id}.yaml`

Each message is a separate YAML file in the `messages/` subdirectory. This eliminates write contention entirely -- no two coordinators ever append to the same file.

```yaml
# .claude/coordinators/messages/20260227-144000-gentle-amber-tide.yaml
from: "gentle-amber-tide"
to: "all"                                # or a specific session_id
at: "2026-02-27T14:40:00"
level: "info"                            # info | warning | request | done
body: "Refactoring utils/paths.py -- adding study_area_root() method. Will be done by 15:00."
```

```yaml
# .claude/coordinators/messages/20260227-145000-swift-drifting-maple.yaml
from: "swift-drifting-maple"
to: "gentle-amber-tide"
at: "2026-02-27T14:50:00"
level: "request"
body: "Need utils/paths.py stable -- can you notify when refactor is complete?"
```

```yaml
# .claude/coordinators/messages/20260227-150200-gentle-amber-tide.yaml
from: "gentle-amber-tide"
to: "swift-drifting-maple"
at: "2026-02-27T15:02:00"
level: "done"
body: "utils/paths.py refactor complete. New method: study_area_root(area_name) -> Path."
```

**Message levels:**

| Level | Meaning |
|-------|---------|
| `info` | Informational -- no action expected |
| `warning` | Something is about to change or has changed that others should know |
| `request` | Asking another coordinator to do or avoid something |
| `done` | A previously announced task is complete |

### Directory Layout

```
.claude/
  coordinators/
    session-gentle-amber-tide.yaml      # Active claim (Coordinator A)
    session-swift-drifting-maple.yaml    # Active claim (Coordinator B)
    messages/                            # Per-message files (no write contention)
      20260227-144000-gentle-amber-tide.yaml
      20260227-145000-swift-drifting-maple.yaml
      20260227-150200-gentle-amber-tide.yaml
```

The `.claude/coordinators/` directory should be gitignored -- these are ephemeral runtime artifacts, not project state. Add to `.gitignore`:

```
.claude/coordinators/
```

## Lifecycle

### Session Start

1. **SessionStart hook** (`session-start.py`) is extended with a new function: `register_coordinator()`.
2. Generate a session ID: pick one word from each of three curated lists (adjective, participle/adjective, noun) and join with hyphens (e.g., `quiet-falling-ember`). If the name collides with an existing claim file, regenerate.
3. Create `.claude/coordinators/` directory if it does not exist.
4. Scan existing claim files in `.claude/coordinators/session-*.yaml`.
5. For each existing claim file:
   - Check `heartbeat_at`. If older than 30 minutes, mark as **stale** (session likely crashed).
   - If not stale, include in the injected context: "Active coordinator: {task_summary}, claiming {claimed_paths}".
6. Write this session's claim file with initial `claimed_paths` set to `["*"]` (unconstrained). The coordinator will narrow this after the user states the task.
7. Inject active coordinator summaries into `additionalContext` so the coordinator sees them immediately on session start.

**The coordinator's first action** after reading the SessionStart context should be to update its claim file with the actual domain and paths based on the user's task. This is done by writing a narrower `claimed_paths` list.

### During Work

Before any file modification (either by the coordinator directly or by delegating to a specialist):

1. **Read all claim files** in `.claude/coordinators/`.
2. **Match the target file path** against every other session's `claimed_paths` globs.
3. If a match is found:
   - **If the other session is stale** (heartbeat > 30 min): proceed with a warning logged to the message file. Do not block.
   - **If the other session is active**: warn the user with a message like:
     ```
     Coordinator session-swift-drifting-maple ("SRAI terminology fixes")
     claims stage2_fusion/**. You are about to modify
     stage2_fusion/models/full_area_unet.py. Proceed anyway?
     ```
   - If the user says yes, proceed and log a `warning` message.
   - If the user says no, back off and find an alternative approach.
4. **Update heartbeat_at** in this session's claim file periodically (every 10 minutes or on each OODA cycle, whichever is more frequent).
5. **Update active_agents** when spawning or completing subagents.

### Reading Messages

At each OODA cycle's OBSERVE phase, the coordinator should:

1. Read all files in `.claude/coordinators/messages/`.
2. Filter for messages addressed to this session's ID or to `"all"`.
3. Filter for messages newer than the last check.
4. Surface relevant messages in the OODA report.

### Session End

1. **Stop hook** (`stop.py`) is extended with `deregister_coordinator()`.
2. Delete this session's claim file from `.claude/coordinators/`.
3. Optionally write a `done` message to `.claude/coordinators/messages/` summarizing what was accomplished: "Session A completed. Modified: stage1_modalities/poi/, utils/paths.py. Committed as abc1234."
4. **Stale claim cleanup**: if any other claim files have `heartbeat_at` older than 2 hours, delete them (they are from crashed sessions).

## Conflict Resolution

### Overlap Detected Before Modification

The default behavior is: **warn the user, let the user decide**. The coordinator does not autonomously resolve conflicts because the user has the most context about what both sessions are doing.

Escalation ladder:

1. **No overlap**: proceed silently.
2. **Overlap with stale session** (heartbeat > 30 min): proceed with info log. The other session is likely dead.
3. **Overlap with active session, different file**: proceed with warning. Two coordinators can claim the same directory but work on different files within it.
4. **Overlap with active session, same file**: stop and ask the user. This is a genuine conflict.
5. **Overlap with active session, same file, user says proceed**: proceed, log a `warning` message, and accept the risk of merge conflicts.

### Overlap Detected After Modification (retrospective)

If a coordinator modifies a file and only then discovers (via message or updated claim) that another session also modified it:

1. Log a `warning` message with the specific file and what was changed.
2. Alert the user: "Both sessions modified {file}. You may need to reconcile changes."
3. Do not attempt automatic reconciliation -- git diff and manual review are the right tools.

## Levin's Design Principles Integration (v1.1)

The protocol was enhanced with concepts from Levin's cognitive light cone framework. These additions make the system self-improving rather than merely structured.

### Three-Scale Cognitive Architecture

The coordinator-to-coordinator protocol operates at the **lateral** scale of a three-level system:

| Scale | Entity | Temporal Reach | Communication Channel |
|-------|--------|---------------|----------------------|
| **Supra** | Human user | Longest -- across all workstreams | Direct chat |
| **Lateral** | Coordinators | Mid -- session-scoped | Claims + messages (this protocol) |
| **Vertical** | Specialist agents | Shortest -- task-scoped | Scratchpads + hooks |

The human is the supra-coordinator. When lateral coordination fails (deadlock, conflicting claims), escalate to the human.

### Pervasive Signaling

Agents propagate signals along a pipeline adjacency graph defined in `subagent-context.py`:

```python
PIPELINE_ADJACENCY = {
    "stage1-modality-encoder": ["stage2-fusion-architect", "srai-spatial"],
    "stage2-fusion-architect": ["stage1-modality-encoder", "stage3-analyst", "srai-spatial"],
    "stage3-analyst": ["stage2-fusion-architect"],
    "srai-spatial": ["stage1-modality-encoder", "stage2-fusion-architect", "stage3-analyst"],
    "qaqc": ["*"],  # sees all
    "librarian": ["*"],
    "ego": ["*"],
}
```

Eight signal keywords propagate automatically: `BLOCKED`, `URGENT`, `CRITICAL`, `BROKEN`, `SHAPE_CHANGED`, `INTERFACE_CHANGED`, `DEPRECATED`, `NEEDS_TEST`. These are matched with word-boundary regex to prevent false positives from identifier references.

### Continuous Lateral Awareness

Coordinator messages are injected at two points:
1. **SessionStart** -- full orientation including active coordinator claims (fires on ALL sources including post-`/clear`)
2. **SubagentStart** -- unread messages injected every wave (every agent spawn), giving continuous awareness without requiring manual message checking

### Cognitive Light Cone Metrics

Each session starts with a one-line summary quantifying the system's cognitive reach:
```
Light cone: 12d memory, 15 agents, ~5 unresolved, 1 active coordinators
```
This provides immediate situational awareness of temporal depth, agent breadth, forward projection quality, and lateral reach.

### Autonomy Contracts

Each agent type has a defined scope of autonomous decisions documented in `.claude/agents/coordinator.md`. Agents within scope do not need coordinator approval -- the scratchpad is the accountability mechanism. This prevents the coordinator from becoming a bottleneck while maintaining traceability.

### Ego as Network Attention

The ego agent serves as the **attention mechanism** across the entire network topology:

| What Ego Watches | Scale | Examples |
|-----------------|-------|----------|
| **Nodes** (coordinator<->agent) | Vertical | Delegation quality, scratchpad freshness, agent autonomy calibration |
| **Edges** (coordinator<->coordinator) | Lateral | Claim respect, message responsiveness, signal propagation, heartbeat regularity |
| **Supra interface** (coordinator<->human) | Upward | Escalation calibration, OODA report quality, compression quality, intent fidelity |

The system forms a **quadruple**: Human + Ego + Coordinators + Agents. The ego monitors connection quality at every scale, enabling the human to progressively delegate more oversight as the system demonstrates reliable self-monitoring.

This maps to Levin's principle that competent subsystems can be trusted with increasing autonomy as they demonstrate goal-directedness at their own level. Ego is the mechanism that verifies goal-directedness.

## Anti-Patterns

### Hijacking

**Definition**: Coordinator A modifies files that Coordinator B has claimed, without checking claims or informing B.

**Prevention**: The claim-check step before every file modification. The hook-injected context shows active coordinators on session start. The OODA OBSERVE phase reads messages.

**Detection**: If a coordinator does not check claims (bug or oversight), the other coordinator will notice at its next OBSERVE phase when it reads the git diff or message log.

### Inferential Imperialism

**Definition**: Coordinator A makes assertions or writes specs about Coordinator B's domain, treating its assumptions as facts.

**Prevention**:
- Claim files declare `read_only_paths`. A coordinator that lists a path as read-only is signaling: "I consume this but do not own it. Do not rely on my assumptions about its internals."
- The protocol rule: **a coordinator MUST NOT write specs, modify code, or make architectural assertions about paths outside its `claimed_paths`**. If it needs cross-domain information, it should:
  1. Read the other coordinator's claim file for context.
  2. Read (not write) the relevant code.
  3. Leave a `request` message asking the owning coordinator to confirm or provide the information.
  4. Wait for a response or ask the user to relay.

**Example**: Coordinator A (Stage 1) needs to know the embedding dimension that Stage 2 expects. Instead of assuming 64 and writing a spec, A should:
- Read B's claim file: "SRAI terminology fixes in stage2_fusion/".
- Leave a message: "What embedding dimension does stage2_fusion expect at the input boundary?"
- Or read `stage2_fusion/models/full_area_unet.py` directly (read-only) and confirm the dimension from code.

### Claim Squatting

**Definition**: A coordinator claims `**` (all paths) and never narrows its claims.

**Prevention**: The initial claim of `["*"]` is a placeholder. The coordinator protocol should require narrowing claims within the first OODA cycle. If a coordinator has not narrowed its claims after its first OBSERVE-ORIENT, the protocol is being violated.

**Detection**: Other coordinators reading the claim file will see the broad claim and can escalate to the user.

### Message Spam

**Definition**: A coordinator writes dozens of messages per cycle, burying important signals in noise.

**Prevention**: Messages should be limited to state transitions (started, changed, done) and explicit requests. Routine status belongs in the coordinator's scratchpad, not in the shared message log.

## Failure Modes and Recovery

### Session Crash (Stale Claim)

**Problem**: A session crashes without running the Stop hook. Its claim file remains, blocking other sessions from modifying claimed paths.

**Detection**: The `heartbeat_at` timestamp goes stale. Any coordinator checking claims will notice the staleness.

**Recovery**:
- Claims older than 30 minutes are treated as stale (warn but do not block).
- Claims older than 2 hours are deleted by any session that encounters them during cleanup.
- The user can manually delete stale claim files: `rm .claude/coordinators/session-*.yaml`.

### Message File Corruption

**Problem**: Two sessions write to the message directory simultaneously.

**Mitigation**: The per-message file approach eliminates write contention entirely -- each message is a separate file (`messages/{timestamp}-{session_id}.yaml`). No two coordinators ever write to the same file. Individual corrupted message files can be deleted without affecting others.

**Recovery**: If a message file is corrupted, delete it. Messages are informational, not transactional. No data loss beyond the individual message.

### Heartbeat Drift

**Problem**: A session is alive but idle (user reading documentation). Heartbeat goes stale. Another session treats its claims as expired.

**Mitigation**: Update heartbeat on every tool call, not just OODA cycles. The SubagentStart and SubagentStop hooks naturally touch the coordinator's claim file. Additionally, the SessionStart hook can update heartbeat on resume/compact events.

### Race Condition on Claim Check

**Problem**: Two coordinators check claims simultaneously, both find no conflict, and both proceed to modify the same file.

**Mitigation**: This is a known limitation of advisory locking. The window is small (milliseconds between check and write). The consequences are a git conflict, not data loss. The message log and git history provide full auditability. This tradeoff is acceptable: the alternative (file-level OS locks) introduces deadlock risk and crash-recovery complexity that outweighs the benefit.

## Hook Integration

### Modified Files

| File | Implementation |
|------|---------------|
| `.claude/hooks/session-start.py` | `register_coordinator()`, cognitive light cone metrics, extended 8-keyword signal vocabulary, full context injection on ALL sources (including post-`/clear`) |
| `.claude/hooks/stop.py` | `deregister_coordinator()`, stale claim cleanup |
| `.claude/hooks/subagent-context.py` | Claim awareness injection, pipeline adjacency signal propagation (`get_sibling_signals()`), coordinator message injection (`get_coordinator_messages()`) |
| `.claude/hooks/subagent-stop.py` | Heartbeat update on every agent completion |
| `.claude/settings.json` | No change needed -- hooks were already registered |

### New Files

| File | Purpose |
|------|---------|
| `.claude/hooks/coordinator_registry.py` | Shared library for claim file I/O: `read_claims()`, `write_claim()`, `delete_claim()`, `check_conflict()`, `read_messages()`, `write_message()` |
| `.claude/rules/coordinator-coordination.md` | Path-scoped rule (paths: `.claude/**`) documenting the protocol for coordinators |

### .gitignore Addition

```
.claude/coordinators/
```

## Alternatives Considered

### Hard Filesystem Locks (fcntl/msvcrt)

**Rejected.** OS-level file locks are fragile across platforms (fcntl on Unix, msvcrt on Windows). A crashed process leaves a permanent lock with no built-in expiry. Recovery requires manual intervention or a separate lock-breaking tool. The complexity is disproportionate to the problem: occasional merge conflicts are cheaper than a lock management system.

### Git Branch Isolation

**Rejected as primary mechanism.** Each coordinator could work on a separate git branch. This eliminates file conflicts but introduces merge complexity. More importantly, it breaks the current workflow where all coordinators see the same HEAD and the same scratchpad state. Branch isolation would require a merge-on-close protocol that is significantly more complex than advisory claims.

However, git branches remain a valid **complementary** strategy. A user who knows two coordinators will work for hours on overlapping domains can manually create branches. The advisory protocol handles the common case where overlap is partial and short-lived.

### Centralized Lock Server

**Rejected.** Requires an external process running alongside Claude Code sessions. Violates the design requirement of file-I/O-only communication. Adds operational complexity (what if the server crashes?). Overkill for 2-3 concurrent sessions.

### Scratchpad-Only Communication

**Rejected as sufficient.** The current scratchpad system is designed for cross-session (sequential) communication, not cross-coordinator (parallel) communication. Coordinator scratchpads are written at session end, not during work. By the time Coordinator A writes "I modified utils/paths.py", Coordinator B has already made conflicting changes. The protocol needs real-time claim visibility, which scratchpads do not provide.

## Consequences

### Positive

- Coordinators gain visibility into each other's active domains before making changes, preventing the most common class of parallel-session conflicts.
- The user can read `.claude/coordinators/` at any time to understand what each session is doing -- pure human-readable YAML.
- Message passing provides a lightweight channel for cross-session notes without requiring the user to relay information manually.
- Advisory (not enforcing) design means a crashed session never permanently blocks work.
- Builds on existing hook infrastructure with minimal new code (~200 lines in `coordinator_registry.py`, ~30 lines of additions to each existing hook).

### Negative

- Advisory protocol depends on coordinators checking claims. If a coordinator session has a bug or the hook fails, overlap is not prevented. The protocol is only as strong as its adoption.
- Heartbeat-based staleness is a heuristic. A session that is alive but idle for 31 minutes will have its claims treated as stale, potentially allowing another session to enter its domain.
- The message file (or message directory) adds filesystem artifacts that must be cleaned up. Stale messages from weeks ago will accumulate unless periodic cleanup is implemented.
- Adds ~50ms per file modification for claim checking (reading and glob-matching all claim files). For sessions with many small edits, this adds up, though it remains negligible compared to model inference time.

### Neutral

- The `.claude/coordinators/` directory is gitignored, so it has zero impact on the repository itself. It exists only at runtime on the developer's machine.
- The protocol does not change the single-coordinator case at all. If only one session is active, the claim file is written and never checked against anything. The overhead is a single file write on start and delete on stop.
- Claim granularity is glob-based, same as the existing rules system. This is a known tradeoff: too broad and everything conflicts, too narrow and overlaps are missed. The escalation ladder (directory-level to file-level) mitigates this.

## Implementation History

| Commit | Date | Description |
|--------|------|-------------|
| `52a7836` | 2026-02-27 | Initial protocol: coordinator_registry.py, claim files, message log, 4 hook integrations, coordinator-coordination.md rule |
| `d176a37` | 2026-02-27 | Levin principles: pipeline adjacency graph, signal vocabulary, cognitive light cone metrics, delegation checkpoint, QAQC response protocol, autonomy contracts |
| `32a630f` | 2026-02-27 | Self-assemblage: agent gap detection in ego-check, convention discovery in librarian-update |
| `187c496` | 2026-02-27 | Fix: word-boundary signal matching, stale agent definitions |
| `75dc879` | 2026-02-27 | Continuous lateral awareness: post-/clear context injection, per-wave coordinator message injection |

Total implementation: ~600 lines across 10 files. All hooks verified functional by QAQC (7/7 checks pass).
