# Post-Vibecoding Follow-Up Plan

*Created: 2026-02-28 by coordinator silver-shining-wind*
*Updated: 2026-02-28 — added Wave 1c (OODA session identification)*
*Source: ego forward-look 2026-03-01 + coordinator session 2026-02-28 findings*

## Context

Yesterday (2026-02-27) was a 5-session vibecoding day: 18 commits, 47 files, +3889 lines.
Today's audit session confirmed everything is sound (157/157 tests, no regressions).
Four housekeeping commits are unpushed. Several carried-forward items need attention.

## Wave 0: Clean State + Push

1. **Push to origin** — 4 commits ahead (`ebab973`, `753785d`, `28e6143`, `ec12d3e`). Push them.
2. Verify `git status` is clean and branch is up to date.

## Wave 1: P0 Fix + Protocol Updates (parallel)

### 1a. Fix execution.md lateral awareness (P0 — 5 sessions overdue)

**Agent**: `spec-writer`
**File**: `.claude/agents/execution.md`
**What**: Add lateral awareness instructions. Currently line 67 only instructs reading the coordinator's scratchpad. The execution agent has shown zero independent cross-agent observations across 5 sessions (2026-02-08 through 2026-02-27).
**Change**: Add instruction to read adjacent agent scratchpads (from pipeline adjacency) before executing, and to note in Cross-agent observations which agent flagged any issue being fixed.
**Acceptance criteria**: execution.md includes lateral reading instructions; existing content preserved.

### 1b. Codify scratchpad self-reconciliation (systemic fix for meta-staleness)

**Agent**: `spec-writer`
**File**: `.claude/rules/multi-agent-protocol.md`
**What**: Add a reconciliation step to the Scratchpad Discipline section. QAQC did this ad-hoc today; it should be standard protocol.
**Change**: Add rule: "Before adding new Unresolved items, verify existing ones against reality. Remove items that have been resolved. Stale unresolved items are false signals."
**Acceptance criteria**: multi-agent-protocol.md Scratchpad Discipline section includes reconciliation instruction.

### 1c. Add session self-identification to OODA report (user can't tell coordinators apart)

**Agent**: `spec-writer`
**File**: `.claude/skills/coordinate/skill.md`
**What**: The OODA report template has no session identification. When multiple coordinators are active, the user can't tell which terminal is which. The session ID is already available (injected by SessionStart hook, stored in `.claude/coordinators/.current_session_id`), but the coordinator never surfaces it to the user.
**Change**: Add `**Session**: [session-name from SessionStart hook]` as the first field in the OODA report template (before `**State**`). Also add a note in Wave 0 that the coordinator should announce its session name.
**Acceptance criteria**: OODA report template in skill.md starts with `**Session**` field; Wave 0 section mentions session identification.

## Wave 2: QAQC Verification

**Agent**: `qaqc`
- Verify execution.md changes are consistent with the autonomy contracts table in coordinator.md
- Verify multi-agent-protocol.md changes don't contradict existing scratchpad format requirements
- Verify skill.md OODA template change is consistent with how session IDs are generated/stored
- Quick check: are there any other agent `.md` files that reference scratchpad protocol and need updating?

## Wave 3: Commit + Push

**Agent**: `devops`
- Commit Wave 1 changes in one logical commit
- Push to origin
- Verify clean state

## Wave 4: DNN Probe Hyperparameter Sweep (research)

**Agent**: `execution`
**Script**: `scripts/one_off/dnn_probe_sweep.py`
**What**: Run the sweep to test hidden_dim 32/128/256/512 on 3-modality concat (AE+POI+Roads, 156 features). Yesterday's Session 2 got R² 0.534 with hidden_dim=32 — the 32-unit bottleneck on 156 input features may be limiting.
**Note**: This is a long-running GPU task (~10-30 min depending on epochs). Run with `--dry-run` first to verify config, then full run.
**Acceptance criteria**: R² matrix printed for all hidden_dim values across 6 leefbaarometer targets.

## Wave 5: Interpret Results

**Agent**: `stage3-analyst`
- Analyze the sweep results: does wider architecture significantly outperform hidden_dim=32?
- If yes, re-run the multimodal vs AE-only comparison with the best architecture for a fair comparison
- Update the stage3-analyst scratchpad with findings

## Final Wave (mandatory)

- Write coordinator scratchpad
- Dispatch librarian agent for codebase graph update
- Dispatch ego agent for process health assessment

## Execution

Invoke: `/coordinate .claude/plans/post-vibecoding-followup.md`
