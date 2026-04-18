# Session Identity Architecture

## Status: Active

How multi-terminal session isolation works, and why there are two identity layers.

## The Problem

One human, multiple terminals. Each terminal runs `/valuate` once, then multiple `/niche` → `/clear` → `/niche` cycles. Without isolation, terminal B overwrites terminal A's identity files and the stop hook deregisters the wrong coordinator.

## Identity Hierarchy

```
Terminal (PPID)
  └── Supra identity (persists across /clear)
        ├── Coordinator 1 (born at terminal open or /clear)
        ├── Coordinator 2 (born after /clear, joins same supra)
        └── Coordinator 3 (born after another /clear, same supra)
```

Two layers, two lifetimes:

| Layer | Lifetime | Changes on /clear? | Purpose |
|-------|----------|-------------------|---------|
| **Supra** | Terminal open → close | No | Valuation state, attentional weights, temporal priors, narrator identity for /sync |
| **Coordinator** | /niche → /clear | Yes | Claim file ownership, heartbeat, task summary, active agents |

The **supra** is the user-facing identity. It answers: "which terminal is this, and what did the human set their weights to?" The coordinator session ID is internal plumbing for claim file management.

## PPID as Isolation Key

`os.getppid()` returns the Claude Code process PID — stable across all hook invocations within one terminal, unique per terminal. Both session and supra files use the **same PPID** (same terminal = same PPID). They live in separate subdirectories only for clean `ls` and simple glob patterns. The difference is purely lifecycle: session files get archived on `/clear`, supra files survive until the terminal closes.

```
.claude/coordinators/
├── sessions/                          # Coordinator identity (rotates on /clear)
│   ├── hushed-spinning-glen.18196     # Terminal 1's current coordinator
│   └── calm-drifting-ash.24668        # Terminal 2's current coordinator
├── supra/                             # Supra identity (persists)
│   ├── hushed-spinning-glen-2026-03-14.18196   # Terminal 1's supra
│   └── calm-drifting-ash-2026-03-14.24668      # Terminal 2's supra
├── messages/{date}/                   # Per-day message files
├── session-hushed-spinning-glen.yaml  # Claim file (current coordinator)
└── session-calm-drifting-ash.yaml     # Claim file (current coordinator)
```

On `/clear` (which fires stop → session-start):
1. stop.py **archives** `sessions/*.{ppid}` and marks claim as `status: ended`
2. session-start.py generates a new coordinator name, writes new `sessions/*.{ppid}`
3. `read_ppid_supra()` finds the **existing** `supra/*.{ppid}` — same supra persists
4. New coordinator joins the existing supra session file in `.claude/supra/sessions/`

On terminal close (stop hook, no session-start after):
1. `sessions/*.{ppid}` archived, claim marked ended
2. Temporal prior updated (if session was valuated)
3. Supra PPID file stays — archived on next startup by `cleanup_stale_ppid_files` (process is dead)

Nothing is ever deleted. Claims get `status: ended`. Old ended claims (>7 days) move to `coordinators/archive/`. Dead-process PPID files move to `{subdir}/archive/`.

## Session Lifecycle

```
Terminal opens (PPID = 18196)
  │
  ├─ session-start.py fires
  │   ├─ Generate coordinator ID: "hushed-spinning-glen"
  │   ├─ Write sessions/hushed-spinning-glen.18196
  │   ├─ No existing supra for PPID 18196 → create supra
  │   ├─ Write supra/hushed-spinning-glen-2026-03-14.18196
  │   └─ Write claim file session-hushed-spinning-glen.yaml
  │
  ├─ User runs /valuate → sets weights in supra session file
  │
  ├─ User runs /niche → OODA cycles with agents
  │
  ├─ User runs /clear (fires stop, then session-start)
  │   │
  │   ├─ stop.py fires
  │   │   ├─ Read sessions/*.18196 → "hushed-spinning-glen"
  │   │   ├─ Mark claim as status: ended
  │   │   └─ Archive sessions/hushed-spinning-glen.18196 → sessions/archive/
  │   │       (supra file untouched)
  │   │
  │   └─ session-start.py fires again
  │       ├─ Generate NEW coordinator: "jade-rising-brook"
  │       ├─ Write sessions/jade-rising-brook.18196
  │       ├─ read_ppid_supra → finds supra/hushed-spinning-glen-2026-03-14.18196
  │       ├─ Joins existing supra (appends to coordinators list)
  │       └─ Write claim file session-jade-rising-brook.yaml
  │
  ├─ User runs /niche again → OODA cycles (same supra, new coordinator)
  │
  └─ Terminal closes
      └─ stop.py fires (no session-start after)
          ├─ Update temporal prior
          ├─ Mark claim as status: ended
          └─ Archive sessions/jade-rising-brook.18196
              (supra file stays — archived on next startup when process is dead)
```

## Cognitive Light Cone

The light cone metaphor (from Levin's morphogenetic fields) maps three scales:

| Biological | System | What it does |
|-----------|--------|-------------|
| Cell | Agent (subagent) | Specialist work, writes scratchpad |
| Organism | Coordinator (session) | OODA orchestration, claim files |
| Colony | Human (supra) | Attentional weights, narrative identity |

The `cognitive_light_cone_summary()` in session-start.py quantifies reach:
- **Temporal depth**: days of scratchpad history
- **Agent reach**: how many agents have contributed
- **Unresolved items**: forward projection quality
- **Active coordinators**: lateral reach (multi-terminal)

## Graph Modes & Shard Coupling

A **shard** is a terminal — one PPID, one `/valuate` → `/niche` pipeline, one full vertical column through the three-layer graph: indicators (bottom) — percepts (middle) — needs/desires (top). Indicators are the files, code paths, and data this terminal works with. Percepts are the agents (context windows) it spawns. Needs/desires are the intent and characteristic states set during `/valuate`. Each terminal is one shard. The edge directions within a shard change by mode (see below). (cf. [shard theory](https://www.lesswrong.com/w/shard-theory))

The two graph modes differ in *where cross-shard coupling happens* and *which direction the edges flow*:

**Static graph (`/valuate`)** — rational preference formation.
- Indicators → percepts: **one-way**. Indicators are passive inputs (you read the state of the world).
- Percepts ↔ needs/desires: **bidirectional**. Mutual assessment — "given what I see, what do I want? Given what I want, what matters?"
- Cross-shard coupling is **at the needs/desires level** (dotted). The human negotiates between shards via the valuate scratchpad: what is each terminal's intent, and what path constraints apply? Coordinator claims are constraints on indicators but set at this level.
- `/sync` is off. No lateral percept communication — you're deciding intent, not executing.

**Dynamic graph (`/niche`)** — active inference.
- Needs/desires → percepts: **one-way down**. Intent is locked in from valuation. The valuated weights drive what percepts look for.
- Indicators ↔ percepts: **bidirectional**. Active inference — percepts don't just read the filesystem, they write code, run tests, modify the world to match the preferred state. Every intermediate step (file read, search, agent dispatch) is directed by the objective set during valuation.
- Cross-shard coupling is **at the percept level** (dotted). Lateral coordination between context windows via `/sync` narrative messaging.
- No cross-shard coupling at needs/desires during niche. Preferences were set during valuation and don't renegotiate during execution.

**The full loop**: `/valuate` sets characteristic states (the preferred observations). `/niche` then self-evidences for those states — the coordinator dispatches agents in waves, each wave modifying the codebase (indicators) to better match the preferred state. This is active inference: the system acts to make its observations confirm its priors. Every file read, search, and agent dispatch during `/niche` is directed by the intent set during `/valuate`. The more precisely you valuate, the more directed niche execution becomes.

See `deepresearch/liveability_approaches_graph.json` for the full edge topology.

Stored as `active_graph` key in the supra session YAML (PPID-isolated, not a singleton file).

## Key Files

| File | Purpose |
|------|---------|
| `coordinator_registry.py` | PPID helpers, claim I/O, message I/O |
| `supra_reader.py` | Supra state reading, temporal priors, graph mode |
| `session-start.py` | Registration, cleanup, context injection |
| `stop.py` | Deregistration, temporal prior update |
| `subagent-context.py` | Agent context injection with PPID reads |

## Related Specs

- `temporal-supra-profiles.md` — Temporal prior EMA, segment keys, profile migration
- `claude_code_multi_agent_setup.md` — Hook lifecycle, scratchpad protocol
- `coordinator-coordination.md` — Claim files, conflict detection, message protocol
