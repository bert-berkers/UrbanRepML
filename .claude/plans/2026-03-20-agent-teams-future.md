# Plan: Agent Teams — PARKED (Needs Human Correction)

> **DO NOT IMPLEMENT.** The framing below is wrong — the human flagged it but was too tired to correct it (2026-03-20). The layer-shift table and architectural reasoning need rework before this plan is actionable. Do not act on this document until the human has reviewed and corrected it.

## The Insight

Agent teams don't replace the shard model — they push everything **one abstraction layer down**. Each session's main agent remains the coordinator (`/valuate` + `/niche`). What changes is that specialists (qaqc, stage2, librarian, etc.) become **teammates** with their own context windows and lateral communication, and those teammates can spawn their own subagents for granular work.

## Layer Shift Table

| Role | Current (subagents) | Plan 3 (agent teams) |
|------|---------------------|----------------------|
| **Human** | Supra — sets characteristic states via `/valuate` | Supra — unchanged |
| **Main Agent** | Coordinator (`/valuate` + `/niche`), spawns subagents | **Team Lead** (same coordinator), spawns teammates via shared task list |
| **Specialists** | Subagents — spawn, work, report back. Leaf nodes. No lateral communication. | **Teammates** — qaqc, stage2, librarian as independent agents with own context windows. Lateral communication between them (= `/sync`). Internal nodes. |
| *N/A* | *Doesn't exist* | **Teammates' subagents** — granular work within each domain. The leaf nodes move here. |

```
Current:                          Plan 3:

Human (supra)                     Human (supra)
  │                                 │
  Main Agent (coordinator)          Main Agent (team lead)
  ├── qaqc (subagent)              ├── Shared Task List
  ├── stage2 (subagent)            ├── qaqc (teammate)
  ├── librarian (subagent)         │   ├── subagent: run tests
  └── ...                          │   └── subagent: review viz
                                   ├── stage2 (teammate)
  Each specialist is a leaf.       │   ├── subagent: fix loss fn
                                   │   └── subagent: profile memory
                                   ├── librarian (teammate)
                                   │   └── subagent: audit imports
                                   └── ...

                                   Each specialist is an internal node.
                                   Lateral arrows between teammates = /sync.
```

## What Already Maps

| Agent teams concept | Our existing equivalent |
|--------------------|-----------------------|
| Shared Task List | Coordinator's delegation plan (OODA waves) |
| Teammate lateral communication | `/sync` between strands |
| Teammate claiming tasks | Scratchpad path claims |
| Team Lead assigning tasks | Coordinator's ACT phase |

## Why Not Now

The trigger for Plan 3 is: **specialist tasks regularly outgrow single-agent capacity**. Currently, qaqc runs a test suite in one context window. stage2 modifies a model in one context window. They don't need sub-delegation.

Plan 3 becomes worth it when:
- A single specialist task (e.g. "audit all data contracts") is too large for one context window
- Multiple specialists need to coordinate laterally without coordinator mediation
- The coordinator's context window is consumed by orchestration overhead rather than synthesis

## Earlier Concern: Terminal Identity (Resolved)

Earlier analysis flagged that teammates are subprocesses, not terminals — they can't hold supra identity. This is correct but irrelevant under the layer-shift model: **teammates don't need `/valuate`**. The team lead (main agent) holds the supra identity and runs `/valuate`. Teammates inherit the lead's characteristic states, which is fine because they're specialists within a single shard, not independent shards.

Option 2 from the earlier analysis ("shared valuation — all teammates inheriting the lead's states") was dismissed as "subagents with extra steps." But with the layer shift, it's not extra steps — it's the right architecture. The difference is that teammates get **lateral communication** and **persistent context windows**, which current subagents don't have.

## Prerequisites

- Plan 1 implemented: scratchpad protocol with mode-aware injection (**done**, 2026-03-20)
- Plan 2 implemented: strand-aware `/sync` with task states (research phase)
- Specialist tasks regularly exceeding single-agent context capacity
- Agent teams stable in Claude Code (currently experimental)

## Historical Context

Originally parked 2026-03-20 as architectural mismatch. Reconsidered same day after realizing the layer-shift model preserves terminal identity. Not a mismatch — just premature.
