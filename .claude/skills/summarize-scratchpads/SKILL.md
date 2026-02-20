---
name: summarize-scratchpads
description: "Compress today's and yesterday's scratchpad entries across all agents into a structured 20-30 line summary. Use this to quickly orient on multi-agent state without reading every scratchpad."
user-invocable: true
context: fork
agent: Explore
allowed-tools: Read, Glob, Grep
argument-hint: "[optional: specific agent name to focus on]"
---

## Task

Read all scratchpad entries from today and yesterday across all agents in `.claude/scratchpad/*/`, then produce a compressed cross-agent summary.

$ARGUMENTS

## Steps

1. Use Glob to find all `.claude/scratchpad/*/????-??-??.md` files
2. Filter to today's and yesterday's dates (check the two most recent dates present)
3. Read each file
4. Synthesize into the output format below

## Output Format

Return a 20-30 line structured summary:

```
## Scratchpad Summary — [date range]

### In Progress
- [Agent]: [what they're working on, key decisions made]

### Blocked
- [Agent]: [what's blocking them, who/what could unblock]

### Tensions
- [Between which agents, about what — naming conflicts, interface mismatches, disagreements]

### Unresolved
- [Open questions that no agent has answered yet]

### Notable
- [Key cross-agent observations, important corrections, process insights]
```

## Rules

- Keep it under 30 lines. This is a compression tool, not a verbatim copy.
- Flag agents that should have written scratchpads but didn't (based on recent git activity).
- Flag stale agents: if an agent has a definition in `.claude/agents/` but its most recent scratchpad is >5 days old, note it. Stale signals decay 2.4x faster (drift research).
- If an agent's scratchpad is over 80 lines, flag it as bloat (per multi-agent protocol).
- Prioritize tensions and unresolved items — those drive the next OODA cycle.
