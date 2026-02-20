---
name: ego-check
description: "Run the ego agent to produce a process health assessment (coherent/strained/drifting) and forward-look for the next session."
user-invocable: true
disable-model-invocation: true
context: fork
agent: ego
argument-hint: "[optional: specific concern to investigate]"
---

## Task

Perform a full ego process health assessment for today's development session.

$ARGUMENTS

## Protocol

Follow the ego agent's standard protocol (defined in `.claude/agents/ego.md`):

1. **Read ALL agent scratchpads** for today — you have full visibility across cognitive light cones
2. **Read recent git diffs** — `git diff HEAD~5..HEAD --stat` and targeted file diffs
3. **Check specs vs implementation** — compare `specs/` goals with actual changes
4. **Look for agent output patterns** — confusion, repeated failures, conflicting edits

## Required Output

Write to `.claude/scratchpad/ego/YYYY-MM-DD.md` (today's date) with:
- **Working**: what's going well
- **Strained**: friction, confusion, conflict
- **Health Metrics** (lightweight ASI-inspired):
  - *Scratchpad freshness*: which agents wrote today? Which are >3 days stale?
  - *Cross-referencing*: are agents reading each other's scratchpads? (look for mentions)
  - *Recommendation closure*: are previously flagged issues being addressed?
- **Forward-Look**: recommendations for next session

Then write the forward-look to `.claude/scratchpad/coordinator/YYYY-MM-DD+1.md` (tomorrow's date) to seed the next OODA cycle.
