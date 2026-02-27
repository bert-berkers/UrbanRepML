---
paths:
  - ".claude/**"
---

# Multi-Agent Stigmergic Protocol

## Scratchpad Format (MANDATORY)

Every agent that does work MUST write a dated entry to `.claude/scratchpad/{agent_type}/YYYY-MM-DD.md` containing:

- **What I did**: actions taken, files modified, decisions made
- **Cross-agent observations**: what I read from other agents' scratchpads, what was useful, what confused me, what I disagree with or would do differently
- **Unresolved**: open questions, things that need follow-up

## Scratchpad Discipline

- If an entry for today already exists, **update it in place** -- consolidate into a single coherent daily log
- Do NOT append-only. Scratchpad bloat degrades signal quality (ego flagged this 2026-02-08)
- Final entry should be a single coherent daily log, not an append-only stream
- Keep entries under 80 lines. If you need more, you're writing too much detail

## Coordination Architecture

- The main agent IS the coordinator -- never spawn a coordinator sub-agent
- The coordinator runs OODA (observe-orient-decide-act)
- The `/coordinate` skill activates coordinator mode
- Specialists are dispatched via the Task tool
- Always foreground agents (`run_in_background: false`) so the user sees activity
- Task descriptions: `"[Agent]: [task]"` format (e.g. `"Librarian: update codebase graph"`)

## The Human Layer

The human user is the supra-coordinator — they sit above the coordinator in the cognitive hierarchy:

```
Human (supra) → sets goals, resolves conflicts, approves irreversible actions
  ↓
Coordinator (lateral) → translates intent, delegates, synthesizes, reports
  ↓
Specialists (vertical) → execute domain work, write scratchpads
```

**Information flows upward as compression**: specialists write 80-line scratchpads, the coordinator compresses to a 5-line OODA report, the human sees a 1-2 sentence summary.

**Intent flows downward as expansion**: the human says "fix the probe pipeline", the coordinator identifies 3 sub-tasks and assigns agents, each agent gets a detailed prompt with file paths and acceptance criteria.

This asymmetry is by design — the human's attention is the scarcest resource, so the system compresses toward them and expands away from them.

## Cross-Agent Communication

- Scratchpads are the primary cross-session communication mechanism (stigmergy)
- The ego monitors scratchpads for process health
- The librarian's `codebase_graph.md` is the shared map
- Read other agents' scratchpads before starting work to avoid duplication

## Signal Vocabulary

Agents SHOULD use these keywords in their scratchpads when relevant. The SessionStart and SubagentStart hooks scan for these automatically and propagate them to adjacent agents.

| Signal | Meaning | Use When... |
|--------|---------|-------------|
| `BLOCKED` | Work cannot proceed | Missing dependency, broken upstream, waiting on external input |
| `URGENT` | Needs attention this session | Time-sensitive issue, regression, data corruption risk |
| `CRITICAL` | System-level concern | Pipeline broken, data loss risk, architectural violation |
| `BROKEN` | Something that previously worked now fails | Test regression, import error, runtime crash |
| `SHAPE_CHANGED` | Data shape or interface contract modified | Column added/removed, tensor dim changed, return type changed |
| `INTERFACE_CHANGED` | Function signature or API modified | Parameters added/removed, renamed, new required args |
| `DEPRECATED` | Code or pattern being phased out | Old API still works but should not be used in new code |
| `NEEDS_TEST` | New or changed code lacks test coverage | Feature added without tests, bugfix without regression test |

These signals propagate along the pipeline adjacency graph (defined in `subagent-context.py`), so a `SHAPE_CHANGED` from `stage2-fusion-architect` is automatically surfaced to `stage3-analyst`.

## Autonomy Scope

Agents make autonomous decisions within their domain. They do NOT need coordinator approval for in-scope choices — the scratchpad is the accountability mechanism. See `.claude/agents/coordinator.md` for the full autonomy contracts table.

**General principle**: If a decision is reversible and stays within your domain's output contract, make it and document it. If it changes an interface or crosses domain boundaries, escalate to the coordinator.

## Multi-File Creation Protocol

When creating multiple new files in a single wave:
1. **Assign __init__.py ownership** to ONE agent per package — prevents merge conflicts
2. **Use filesystem grep for scope audits** — `rg PATTERN dir/` not `git grep` (untracked files are invisible to git grep)
3. **Include Plan agent's recommendation** in each delegation prompt when applicable
4. **QAQC produces commit-readiness verdict** — after verification, explicitly state whether working tree is committable
