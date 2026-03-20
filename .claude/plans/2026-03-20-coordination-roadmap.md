# Coordination Roadmap — Static → Dynamic → Autonomous

## The Complementary Duality

| | Plan 1: Scratchpad Protocol | Plan 2: Strand Coordination | ~~Plan 3: Agent Teams~~ |
|---|---|---|---|
| **Column** | Static (natura naturata) | Dynamic (natura naturans) | ~~Dynamic (autonomous)~~ |
| **What** | How scratchpads crystallize governance | How strands coordinate during overlapping lifetimes | ~~PARKED — architectural mismatch~~ |
| **Scope** | Cross-session memory | Within/across strand coupling | ~~N/A~~ |
| **Human role** | Sets characteristic states via `/valuate` | Steers via chat, stays in the loop | ~~N/A~~ |
| **Status** | **Implemented** (2026-03-20) | Research phase (open questions) | **Parked** — see below |
| **File** | `2026-03-20-scratchpad-best-practices.md` | `2026-03-20-niche-strand-coordination.md` | `2026-03-20-agent-teams-future.md` |

## Dependency Chain

```
Plan 1 (scratchpad protocol)
  │
  │  Static governance must be solid before dynamic coordination
  │  can rely on it. The attractor basin must exist before flow
  │  can be channeled through it.
  │
  ▼
Plan 2 (strand coordination)
  │
  │  Strand-to-strand coupling via /sync must work before
  │  autonomous strands can self-coordinate. The local roads
  │  must exist before traffic can flow without a guide.
  │
  ▼
Plan 3 (agent teams) — PARKED
     Anthropic's teammates are subprocesses, not terminals.
     No PPID = no supra identity = no /valuate column.
     Cells without nuclei. Revisit only if Anthropic ships
     terminal-level teammate isolation.
```

## The Biological Analogy

| Stage | Biological Equivalent | Coordination Mechanism |
|---|---|---|
| Plan 1 | DNA (governance, persists across cell deaths) | Scratchpads crystallize characteristic states |
| Plan 2 | Cell signaling (paracrine, between neighbors) | `/sync` between overlapping strands |
| Plan 3 | Organism (autonomous multi-cellular coordination) | Agent teams with shared task lists |

Each stage doesn't replace the previous — it builds on it. Organisms still have DNA. Cells still signal. The layers accumulate.

## Implementation Order

### Now: Plan 1 (this session)
- 3 files, ~50 lines
- Mode-aware injection, consolidation directives, delegation template
- Verification: spawn test subagent, check injection

### Next session: Plan 2 research
- Answer the 5 open questions
- Map current `/loop /sync` behavior
- Determine minimal viable change (task state tags on Unresolved?)
- Design strand identity model

### ~~Plan 3~~ — PARKED
Anthropic's agent teams are subprocesses of the lead agent, not independent terminals. The shard model requires terminal = PPID = one `/valuate` → `/niche` column. Teammates can't hold supra identity or run `/valuate` — they're cells without nuclei. Revisit only if Anthropic ships terminal-level teammate isolation. See `2026-03-20-agent-teams-future.md` for full analysis.
