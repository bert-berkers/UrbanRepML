# Session Identity Architecture

## Status: Active (simplified 2026-05-03)

How identity works in a multi-terminal PyCharm + Claude Code setup. **One terminal = one identity, for the lifetime of the terminal.**

## The Model

One human, multiple PyCharm terminals. Each terminal opens, gets a poetic name (e.g. `pale-listening-dew`), and keeps that name until the terminal window closes. That's the whole identity model.

- **`/clear` is a context flush, not a lifecycle event.** Same terminal, same name, fresh context. Multiple `/niche` runs in the same terminal share the same identity and the same valuation state.
- **Bash subshell PIDs are noise.** What matters is the stable PyCharm shell ancestor (`powershell.exe`, `bash`, etc.) found by walking the process tree past the Claude binary. `coordinator_registry.get_terminal_pid()` does this walk; the marker list extension on 2026-05-03 (commit `12ad893`) made it recognise `claude.exe` alongside `node.exe` so the walk terminates at the correct ancestor on Windows + PyCharm.
- **No session/supra distinction.** Earlier drafts of this spec carried two identity layers (a session ID that rotated on `/clear`, and a supra ID that survived it). With `/clear` demoted to a context flush, both layers collapse into one: `identity_id`.

## Files (records layer)

| Path | Purpose | Lifetime |
|------|---------|----------|
| `.claude/coordinators/terminals/{pid}.yaml` | Carries the single `identity_id` for the shell PID | Terminal open → close |
| `.claude/coordinators/session-{identity_id}.yaml` | Claim file: working domain, heartbeat, active agents | Terminal open → close (`status: ended` on close) |
| `.claude/supra/sessions/{identity_id}.yaml` | Valuation state: weights, intent, focus/suppress | Terminal open → close |
| `.claude/scratchpad/coordinator/{date}-{identity_id}.md` | Per-day, per-terminal scratchpad. Multiple `/niche` runs append. | Per calendar day |
| `.claude/coordinators/messages/{date}/{ts}-{identity_id}.yaml` | Cross-terminal messages | Per day |

The terminal file has one identity field (`identity_id`) — no `session_id` / `supra_session_id` split. The valuation file is `{identity_id}.yaml` directly; the date that used to suffix supra IDs (`pale-listening-dew-2026-05-03`) is dropped from identity. A date can still appear as a *label* inside the YAML (e.g. `valuated_at: 2026-05-03T19:30:00Z`) for temporal-prior bookkeeping — that is metadata, not identity.

## Terminal Lifecycle

```
Terminal opens (PID = 29556)
  │
  ├─ session-start.py fires
  │   ├─ get_terminal_pid() walks past claude.exe → 29556 (powershell.exe)
  │   ├─ Generate poetic name: "pale-listening-dew"
  │   ├─ Write terminals/29556.yaml         { identity_id: pale-listening-dew }
  │   ├─ Write session-pale-listening-dew.yaml (claim)
  │   └─ SessionStart prompt-injects the name into the coordinator's context
  │
  ├─ /valuate → writes supra/sessions/pale-listening-dew.yaml
  │
  ├─ /niche → OODA cycles, agents append to scratchpad/{type}/2026-05-03-pale-listening-dew.md
  │
  ├─ /clear → context flush ONLY
  │   - identity does not rotate
  │   - terminals/29556.yaml unchanged
  │   - claim unchanged
  │   - supra unchanged
  │   - SessionStart re-injects the SAME name into the new context
  │
  ├─ /niche again → same identity, fresh context, same scratchpad file (appended)
  │
  └─ Terminal closes
      └─ stop.py fires (no session-start after)
          ├─ Mark claim as status: ended
          ├─ Update temporal prior from supra
          └─ Archive terminals/29556.yaml → terminals/archive/
              (supra and scratchpad files stay in place — never deleted)
```

Nothing is ever deleted. Claim files get `status: ended`. Old ended claims (>7 days) move to `coordinators/archive/`. Dead-process terminal files move to `terminals/archive/` on the next session-start that finds the PID gone.

## Cognitive Light Cone

The light cone metaphor (from Levin's morphogenetic fields) maps three scales:

| Biological | System | What it does |
|-----------|--------|-------------|
| Cell | Agent (subagent) | Specialist work, writes scratchpad |
| Organism | Coordinator (terminal) | OODA orchestration, claim file, valuation state |
| Colony | Human (across terminals) | Sets intent per terminal, resolves cross-terminal conflicts |

The `cognitive_light_cone_summary()` in session-start.py quantifies reach: temporal depth (days of scratchpad history), agent reach (how many agents have contributed), unresolved items (forward projection quality), active coordinators (lateral reach across terminals).

## Graph Modes & Shard Coupling

A **shard** is a terminal — one PID, one `identity_id`, one full vertical column through the three-layer graph: indicators (bottom) — percepts (middle) — needs/desires (top). Indicators are the files, code paths, and data this terminal works with. Percepts are the agents (context windows) it spawns. Needs/desires are the intent and characteristic states set during `/valuate`. Each terminal is one shard. (cf. [shard theory](https://www.lesswrong.com/w/shard-theory))

The two graph modes differ in *where cross-shard coupling happens* and *which direction the edges flow*:

**Static graph (`/valuate`)** — rational preference formation.
- Indicators → percepts: **one-way**. Indicators are passive inputs.
- Percepts ↔ needs/desires: **bidirectional**. Mutual assessment.
- Cross-shard coupling at the **needs/desires level** (dotted) — human negotiates between terminals via the valuate scratchpad and path claims.
- `/sync` is off.

**Dynamic graph (`/niche`)** — active inference.
- Needs/desires → percepts: **one-way down**. Intent locked in from valuation.
- Indicators ↔ percepts: **bidirectional**. Percepts read and write the world to match preferred state.
- Cross-shard coupling at the **percept level** (dotted) — lateral coordination between context windows via `/sync`.

The `active_graph` key lives in the supra session YAML (terminal-isolated, not a singleton).

## Identity-as-record vs. identity-as-prompt

The harness carries identity in two layers, and they do different work. The first is the **records layer**: the terminal file at `.claude/coordinators/terminals/{pid}.yaml`, the claim file `session-{identity_id}.yaml`, the supra YAML at `.claude/supra/sessions/{identity_id}.yaml`. These exist to be looked up — by `coordinator_registry.read_ppid_identity()`, by peer-terminal pointer scans during Markov-7 close-out, by `/sync` when one terminal addresses another, by claim conflict detection. Records-mode is correct here: lateral coordination needs a stable file lookup keyed by something the addressing system can resolve. When this layer breaks (terminal file missing, PID unstable across hook invocations, ghost supra), lateral coordination genuinely degrades — peers can't find each other, claims drift, messages mis-route.

The second is the **prompt layer**: the SessionStart hook injects the poetic name (e.g., `pale-listening-dew`) directly into the coordinator's context as a system reminder the moment its window opens. No file I/O. Whatever future agent's interpretive competencies show up, the name is already there as a prompt — it doesn't decode into structured fields, it *cues* the agent into self-recognition. In Levin's frame (see `memory/feedback_stigmergic_traces_are_generative_model.md`), this is identity-as-message-between-agents-across-time: the name is interpretable by whoever wakes up, with whatever competencies they happen to have. The records layer tries to faithfully record which shell is which; the prompt layer just hands the agent its own name and trusts it to mean something.

Practical guidance for maintainers: **the two layers are not redundant — they fail differently.** When the records layer breaks (PID drift, missing terminal file, hook silenced), self-identity does not break — the prompt layer still delivers the agent's own name in context, and the agent can still write to its own scratchpad keyed by that name. What breaks is *lateral* identity: peers can't find this terminal, claim conflicts go undetected, peer-terminal pointers in Markov-7 read empty. So when the records layer breaks, fix it (it's load-bearing for cross-terminal coordination AND for `/clear` durability — the prompt layer fires once per context, the records layer is what survives the flush) — but do not panic. The agent's own self-identity within a single context is robust because it arrived as a prompt, not a fetch. And do not reflexively reach for "write the terminal file from the skill" as a fix when the upstream cause is `get_terminal_pid()` resolving inconsistently across hook invocations; writing under a drifted PID just produces a record that future readers also can't find.

## Migration note

This spec was simplified from a two-layer (session/supra) identity model on 2026-05-03. The earlier model had a coordinator-session that rotated on `/clear` distinct from a supra-session that survived it; both lived in PID-keyed files under `coordinators/sessions/{id}.{ppid}` and `coordinators/supra/{id}.{ppid}`. That scheme was already partly consolidated into `terminals/{pid}.yaml` (see auto-memory `project_desktop_app_quirks.md`), and after the 2026-05-03 fix to `get_terminal_pid()` made identity stable across `/clear`, the rotation lost its purpose — `/clear` became a context flush, not a lifecycle event. The hooks (`session-start.py`, `stop.py`, `coordinator_registry.py`, `subagent-context.py`, `supra_reader.py`, `sync_run.py`) were brought into compliance in the same session via `read/write_ppid_identity()` (replacing the old `_session` and `_supra` pairs) and `generate_identity_id()`. Live `terminals/{pid}.yaml` files written before the collapse may still carry `session_id` and/or `supra_session_id` instead of `identity_id`; the read path falls back through both legacy field names, and the next legitimate write upgrades the file forward. Cross-reference: `.claude/scratchpad/coordinator/2026-05-03-pale-listening-dew.md` for the close-out that motivated this simplification, and `.claude/scratchpad/spec-writer/2026-05-03-pale-listening-dew-W2.md` for the implementation summary.

## Related Specs

- `temporal-supra-profiles.md` — Temporal prior EMA, segment keys, profile migration
- `claude_code_multi_agent_setup.md` — Hook lifecycle, scratchpad protocol
- `coordinator-coordination.md` — Claim files, conflict detection, message protocol
