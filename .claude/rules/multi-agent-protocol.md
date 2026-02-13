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

## Cross-Agent Communication

- Scratchpads are the primary cross-session communication mechanism (stigmergy)
- The ego monitors scratchpads for process health
- The librarian's `codebase_graph.md` is the shared map
- Read other agents' scratchpads before starting work to avoid duplication
