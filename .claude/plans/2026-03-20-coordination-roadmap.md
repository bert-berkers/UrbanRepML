# Coordination Roadmap — Static → Dynamic → Autonomous

## The Complementary Duality

| | Plan 1: Scratchpad Protocol | Plan 2: Strand Coordination | Plan 3: Agent Teams |
|---|---|---|---|
| **Column** | Static (natura naturata) | Dynamic (natura naturans) | Dynamic (autonomous) |
| **What** | How scratchpads crystallize governance | How strands coordinate during overlapping lifetimes | Push specialists one layer down: subagents → teammates with own subagents |
| **Scope** | Cross-session memory | Within/across strand coupling | Within-shard delegation depth |
| **Human role** | Sets characteristic states via `/valuate` | Steers via chat, stays in the loop | Same — team lead IS the coordinator, `/valuate` unchanged |
| **Status** | **Implemented** (2026-03-20) | Research phase (open questions) | **Parked** — framing is wrong, needs human correction |
| **File** | `2026-03-20-scratchpad-best-practices.md` | `2026-03-20-niche-strand-coordination.md` | `2026-03-20-agent-teams-future.md` |

## Dependency Chain

```
Plan 1 (scratchpad protocol) ✓ DONE
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
Plan 3 (agent teams) — FUTURE
     Not a new architecture — everything shifts one layer down.
     Specialists become teammates (internal nodes with subagents)
     instead of subagents (leaf nodes). Trigger: when specialist
     tasks regularly outgrow single-agent context capacity.
```

## The Layer Shift

```
Current:                          Plan 3:

Human (supra)                     Human (supra)
  │                                 │
  Coordinator (main agent)          Team Lead (same coordinator)
  ├── qaqc (subagent, leaf)        ├── qaqc (teammate, internal)
  ├── stage2 (subagent, leaf)      │   └── subagents
  └── librarian (subagent, leaf)   ├── stage2 (teammate, internal)
                                   │   └── subagents
                                   └── librarian (teammate, internal)
                                       └── subagents
```

## The Biological Analogy

| Stage | Biological Equivalent | Coordination Mechanism |
|---|---|---|
| Plan 1 | DNA (governance, persists across cell deaths) | Scratchpads crystallize characteristic states |
| Plan 2 | Cell signaling (paracrine, between neighbors) | `/sync` between overlapping strands |
| Plan 3 | Tissue differentiation (cells with specialized organelles) | Teammates with own subagents, lateral communication |

Each stage doesn't replace the previous — it builds on it. Organisms still have DNA. Cells still signal. The layers accumulate.

## Implementation Order

### Done: Plan 1 (2026-03-20)
- 3 files, +69/-9 lines, committed as `996935f`
- Graph mode injection, consolidation directives, delegation template, strand history

### Next: Plan 2 research
- Answer the 5 open questions
- Map current `/loop /sync` behavior
- Determine minimal viable change (task state tags on Unresolved?)
- Design strand identity model

### Future: Plan 3
Trigger: specialist tasks regularly exceed single-agent context capacity. Not a new architecture — just pushing existing specialists from leaf nodes to internal nodes. Team lead = coordinator, teammates = current specialists, teammates' subagents = the new leaf nodes. `/valuate` stays at the main agent level. See `2026-03-20-agent-teams-future.md` for full analysis.
