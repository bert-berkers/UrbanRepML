---
paths:
  - ".claude/**"
---

# Multi-Agent Stigmergic Protocol

## Scratchpad Format (MANDATORY)

Every agent that does work MUST write a dated entry to `.claude/scratchpad/{agent_type}/YYYY-MM-DD.md` containing:

- `<!-- SUMMARY: one-line summary of what you did -->` (first line, machine-extractable)
- **What I did**: actions taken, files modified, decisions made
- **Cross-agent observations**: what I read from other agents' scratchpads, what was useful, what confused me, what I disagree with or would do differently
- **Unresolved**: open questions, things that need follow-up — each tagged `[open]`, `[stale]`, or `[blocked:reason]`

## Scratchpad Discipline

- **Session-keyed files**: Scratchpads are `{agent_type}/{date}-{session_id}.md`. Each terminal writes its own file — no cross-terminal clobbering.
- **Append, don't overwrite**: Each agent invocation APPENDS a new timestamped section (`## HH:MM — summary`). Never rewrite earlier entries — they belong to earlier agent invocations.
- **Prior entries index**: Start each new entry with `**Prior entries**: 10:15 — built X | 10:45 — added Y`. This makes every entry a self-contained context packet — the hook injects the tail, so your entry must carry forward the gist of earlier work.
- Each entry should be self-contained and under 30 lines. Multiple short entries > one bloated rewrite.
- **`<!-- OVERRIDE: {rationale} -->` escape** (formal): the ≤30-line guideline is soft. Starting an entry with `<!-- OVERRIDE: {rationale} -->` on line 1 is a recognized escape from the line limit. The override is triggered when:
  - The entry must satisfy **Contract 1** (coordinator close-out with all 7 items — see Markov-Completeness Contract section below), OR
  - Preserving handoff completeness for the next context window requires more than 30 lines (e.g. a Wave-final audit synthesis with aged `[open|Nd]` items, peer-terminal pointers, and a full reference frame block).

  **Length is not the cost; loss-of-state is.** A 60-line entry that a cold-start session can resume from is strictly better than a 25-line entry that leaks state forward. The override makes the cost explicit so reviewers can see "yes, this entry needed that length" rather than treating every long entry as discipline rot.
- **Reconciliation-first**: Before writing new Unresolved items, check if earlier entries in the same file already flagged them. Don't duplicate — reference by time.
- Items tagged `[stale]` for 2+ sessions should be removed or escalated to the coordinator.
- **Output references**: Reference large outputs by path, don't paste inline. Scratchpads are index + reasoning, not data.
- The hook injects the last 100 lines of the most recent scratchpad — write your summary/unresolved at the bottom so it's what the next agent sees.

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

## Signal Vocabulary (v2 — Structured Tags)

> **Migration note**: The earlier single-word signals (`BLOCKED`, `URGENT`, `CRITICAL`, `BROKEN`, `SHAPE_CHANGED`, `INTERFACE_CHANGED`, `DEPRECATED`, `NEEDS_TEST`) are **deprecated**. They flagged *attention* but not *actionability, routing, or contract coupling*. They are replaced by the structured tags below. Old signals still parse as legacy literals but new scratchpad entries MUST use v2 tags.

Agents use structured tags in scratchpads. The SessionStart, SubagentStart, and SubagentStop hooks scan for these and propagate relevant tags to adjacent agents along the pipeline adjacency graph (defined in `subagent-context.py`). Tags are machine-parseable; their bracket form is load-bearing.

### State markers (carried with age)

| Tag | Meaning |
|---|---|
| `[open\|Nd]` | Open item; N = days since first raised. Increment, don't reset, across sessions. |
| `[stale\|Nd]` | Open ≥ 14 days; needs attention this session. |
| `[escalated\|Nd]` | Open ≥ 21 days; requires human decision. Surface at Final Wave. |
| `[done\|YYYY-MM-DD]` | Resolved. Date records when, not who. |
| `[wontfix:reason]` | Explicit abandonment with rationale. Not silence. |

### Frequency-escalation rule

The day-age thresholds above (14d → `[stale]`, 21d → `[escalated]`) capture *elapsed time* but not *confirmation density*. An item that 5+ agents independently flag in a single session, or 3+ agents flag across 2+ sessions, warrants attention even at N=5d. Codified rule:

| Confirmation pattern | Promotion |
|---|---|
| 3+ agents × 2+ sessions | `[open\|Nd]` → `[stale\|Nd]` regardless of N |
| 5+ agents × 1 session | `[open\|Nd]` → `[stale\|Nd]` regardless of N |
| 6+ agents × 3+ sessions | promote to `[escalated\|Nd]` regardless of N |

**Mechanics:** the day-age N is preserved (don't reset); only the bucket changes. The promoting coordinator MUST cite the confirmation evidence in the scratchpad close-out (e.g. "frequency-escalated: 5 agents across qaqc-W1, devops-W1, stage3-W2a, stage3-W2b, ego-2026-04-19"). Coordinator judgment had been making this call correctly by hand (2026-04-24 hook-filename-drift case); this rule makes it enforceable across sessions.

**Why this exists:** the strict day-age model implicitly assumes confirmation is uniform-in-time. In practice, multiple specialists touching the same area in the same week generate dense confirmation that should be acted on, not aged through. Without this rule, frequency-escalated items wait 9d for promotion past their natural attention threshold.

### Block reasons

| Tag | Meaning |
|---|---|
| `[blocked:human-decision]` | Waiting on the supra-coordinator. |
| `[blocked:upstream:agent-or-file]` | Waiting on a specific producer (agent type or file path). |
| `[blocked:data-missing:path]` | Required data artifact not present on disk. |
| `[blocked:design:question]` | Open design question — can't proceed without resolution. |

### Change records

| Tag | Meaning |
|---|---|
| `[shape-changed:path:before→after]` | Tensor/column/schema shape modification. |
| `[interface-changed:func:old→new]` | Function signature or API change. |
| `[file-moved:old→new]` | Relocation — preserve for librarian. |
| `[file-deleted:path:reason]` | Removal with rationale. |

### Routing

| Tag | Meaning |
|---|---|
| `[→agent-type]` | Suggested next owner. |
| `[needs:agent-type\|human]` | Required participant before this can close. |

### Decision records

| Tag | Meaning |
|---|---|
| `[decided:option:rationale]` | Resolved alternative, with why-X-not-Y compressed. |
| `[deferred:topic:to-when-or-plan]` | Explicitly kicked forward, with target (date or plan path). Not silent drift. |

### Contract coupling

| Tag | Meaning |
|---|---|
| `[contract:N]` | Audit §4 Contract N this item couples to (1=scratchpad, 2=codebase_graph, 3=specs, 4=plans, 5=session-states). |
| `[plan:path]` | Pointer to active plan file. |
| `[audit:report-path#section]` | Pointer into an audit report (e.g. `reports/2026-04-18-organizational-flywheel-audit.md#theme-H`). |

The hook-side Markov-completeness check (see `markov_check.py`) recognizes these tags explicitly. The regex that satisfies Contract 1 item 4 (aged open items in the coordinator close-out) is `\[(open|stale|escalated)\|\d+d\]` — any of the three aged-item tag forms counts. This is broader than `\[open\|\d+d\]` alone because a correctly-aged session may have escalated every `[open|Nd]` item past 14d (→ `[stale]`) or 21d (→ `[escalated]`), leaving no literal `[open]` tags; the check must not false-fail that case.

## Markov-Completeness Contract (Coordinator Close-Out)

The final entry of each agent-type's session-keyed scratchpad must be a self-contained context packet for the next fresh window. This is **Contract 1** from the 2026-04-18 organizational flywheel audit (`reports/2026-04-18-organizational-flywheel-audit.md §4`). It closes the primary state leak across `/clear`, terminal-switch, and calendar-day boundaries.

### The seven items (coordinator close-out)

The final coordinator scratchpad entry of every session MUST contain all seven:

1. **`SUMMARY` comment** — `<!-- SUMMARY: one-line ... -->` on the first line (machine-extractable).
2. **Prior-entries index** — cumulative for the day across session files. One line per prior entry: `HH:MM — one-line summary`. Carries forward even across session-keyed file boundaries within the same calendar day.
3. **Reference frame block** — a block echoing the active supra values: `mode=... speed=N explore=N quality=N tests=N spatial=N model=N urgency=N data_eng=N intent="..." focus=[...] suppress=[...]`. If `/valuate` wasn't run, write `intent=null (plan-driven: {path})` or `intent=null (no plan, no valuation — under-valuated session)`.
4. **Aged `[open|Nd]` / `[stale|Nd]` / `[escalated|Nd]` items** — every carried unresolved item, tagged with age in days. Do NOT re-open already-carried items; increment N across sessions. Escalate to `[stale|Nd]` at N ≥ 14 and `[escalated|Nd]` at N ≥ 21. The hook-enforceable marker is `\[(open|stale|escalated)\|\d+d\]` — any of the three forms satisfies the contract (see Signal Vocabulary section for rationale).
5. **Active plan pointer** — `Plan: .claude/plans/{file}.md` if any. Plans function as frozen intent; their presence is load-bearing. If no plan, write `Plan: none (ad-hoc session)`.
6. **Peer-terminal pointer** — read from `.claude/coordinators/terminals/*.yaml`: list other active or recently-ended terminals with their `supra_session_id`. `Peers: none` is an acceptable value but must be stated.
7. **"If you only read this entry, here's the gist" paragraph** — single paragraph at the bottom, preceded by a heading matching `^## If you only read`. Must subsume items 1–6 in prose; must be sufficient to orient a cold-start session without reading anything else. **This paragraph is the Markov-complete summary.**

### Two enforcement tiers

**Specialist scratchpads** (all non-coordinator agent types): items **1, 2, 7** required. Items 3–6 inherit from the coordinator's reference frame.
- Fail mode: **fail-OPEN**. `subagent-stop.py` logs a warning and accepts the write. The specialist's job is domain work; the contract is softest here because context is inherited.

**Coordinator close-out scratchpads** (final entry of a `/niche` session, or at `/clear` / terminal close): all **seven** items required.
- Fail mode: **fail-CLOSED**. `stop.py` refuses completion and lists the missing items to stderr. The coordinator is the handoff node; missing items here leak state forward silently.

The single shared implementation lives in `markov_check.py` and is called by both `subagent-stop.py` (specialist, fail-open) and `stop.py` (coordinator, fail-closed). One helper, two callers, two fail modes.

## Autonomy Scope

Agents make autonomous decisions within their domain. They do NOT need coordinator approval for in-scope choices — the scratchpad is the accountability mechanism. See `.claude/agents/coordinator.md` for the full autonomy contracts table.

**General principle**: If a decision is reversible and stays within your domain's output contract, make it and document it. If it changes an interface or crosses domain boundaries, escalate to the coordinator.

## Multi-File Creation Protocol

When creating multiple new files in a single wave:
1. **Assign __init__.py ownership** to ONE agent per package — prevents merge conflicts
2. **Use filesystem grep for scope audits** — `rg PATTERN dir/` not `git grep` (untracked files are invisible to git grep)
3. **Include Plan agent's recommendation** in each delegation prompt when applicable
4. **QAQC produces commit-readiness verdict** — after verification, explicitly state whether working tree is committable

## Memory Strands

Scratchpads serve two fundamentally different purposes depending on graph mode:

- **Static mode** (`/valuate`): Scratchpads crystallize governance — the attractor basin that channels future flow. These are natura naturata: not the infrastructure itself but the *pattern* development took around it, like deltas or lung branching. Write for permanence.
- **Dynamic mode** (`/niche`): Scratchpads are the fossil record of natura naturans — traces of a generative process that vanishes when the context window dies. Each context window is a hula hoop keeping a section of flexible netting alive. Write to make the invisible visible.

The `subagent-context.py` hook injects the current graph mode into every agent's context. The mode shapes what you write: governance crystallization vs process traces.
