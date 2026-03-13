# Valuation & Niche Construction: Temporal Supra Sessions

## Status: Draft (ontology corrected 2026-03-13)

## Theoretical Grounding

Two approaches to resident-environment fit, grounded in the user's research presentation "Towards Autonomous Urban Digital Twins" (see `deepresearch/liveability_approaches_graph.json` for the formal graph definition, `deepresearch/render_liveability_graph.py` to render).

### The Cognitive Light Cone (Levin)

Cell → Organism → Colony maps to Agent → Coordinator → Human. Each scale has its own temporal reach and communication bandwidth. The human sits at the apex — longest temporal horizon, most compressed information.

### The Three Nodes

| Graph Node | Cognitive Light Cone | Implementation | Nature |
|---|---|---|---|
| **Indicators** | Objective environment | Codebase, filesystem, git state, data files | Always available, shared across all percepts |
| **Percepts** | Sequential context windows | Each `/niche` run = one percept. `/clear` + restart = new percept, same identity. Scratchpads = memory traces of dead percepts. | Subjective, mortal, lineage-connected |
| **Needs/Desires** | The human (supra) | Needs = hard constraints (focus, "don't touch X"). Desires = soft constraints (speed vs quality tradeoff). | Rational choice: maximize utility subject to budget constraints |

**Key distinction**: Percepts are NOT individual context windows of specialist agents. They are the sequential context windows of **coordinator runs within one terminal**. A terminal's identity persists across percept death — scratchpads carry the memory forward. Specialist agents are sub-percept processes — they exist within a percept's lifetime.

### Needs vs Desires (Rational Choice Theory)

The N/D node splits along economic lines:
- **Needs** = hard constraints (the budget constraint). Non-negotiable: "maintain spatial correctness", "don't delete data", focus directives. These bound the feasible set.
- **Desires** = soft constraints (the utility function). Tradeable: speed vs quality, exploration vs exploitation. Dimension weights encode the human's current utility preferences.
- **Compound states** = pre-resolved constraint satisfaction. "creative-prototyping" means "I need exploration AND desire speed" — the tradeoff is already resolved. This is the N/D ↔ N/D coupling: needs constraining desires, desires constraining needs, until an internally consistent state emerges.

### Static = Valuation (`/valuate`)

- **Indicators → Percepts** (one-way): The codebase feeds into the context window. Morning inread: git log, scratchpads, ego assessment. Read-only.
- **Percepts ↔ Needs/Desires** (bidirectional): The human's needs shape perception (focus/suppress directives push down), AND the percept's observations inform the human's needs (questionnaire surfaces what matters). Mutual assessment.
- **N/D ↔ N/D coupling** (dotted): The human's needs constrain each other. Compound states encode these tradeoffs. The human resolves this internally during `/valuate`.
- **Per ↔ Per**: NONE. No lateral coupling during valuation — there is one context window doing the assessment.

Sets the characteristic states. Happens once per terminal start.

### Dynamic = Niche Construction (`/niche`)

- **Indicators ↔ Percepts** (bidirectional): The coordinator both reads AND writes the codebase. OBSERVE reads; ACT writes. This is the niche construction — the organism modifies its own environment.
- **Needs/Desires → Percepts** (one-way DOWN): The resolved constraint set pushes down once. The human does not continuously re-negotiate — they set the budget during `/valuate` and the percepts execute within it. OODA checkpoints are for course correction, not re-valuation.
- **Percept ↔ Percept coupling** (dotted): Lateral message passing between concurrent terminals via `/sync`. This is **homo narrans** — each terminal IS a narrator, telling its story to other terminals. `/loop 5m /sync` is the heartbeat of this narrative exchange. The session_id is the identity constant that makes the narrator recognizable across context window deaths.
- **N/D ↔ N/D**: NONE. Needs are already resolved during `/valuate`.

The structural flip: in valuation, coupling lives at the Needs/Desires level (human resolving internal tradeoffs). In niche construction, coupling moves to the Percept level (terminals narrating to each other). The human steps back.

### What Scratchpads Actually Are

Scratchpads are NOT the lateral coupling mechanism. They are **memory traces of dead percepts** — how the next context window inherits knowledge from the last within the same terminal lineage. They are intra-lineage inheritance, not inter-lineage communication.

The lateral coupling (Per ↔ Per) is `/sync` message passing between concurrent terminals. Different mechanism, different purpose:
- **Scratchpads**: vertical inheritance within a percept lineage (how Process 2 knows what Process 1 did)
- **`/sync` messages**: lateral narrative between concurrent terminals (homo narrans)

### The Lifecycle

```
Terminal (one workflow, one identity, one supra session_id)
  ├── Percept 1: /valuate → /niche OODA → context full → scratchpad written → die
  ├── Percept 2: read scratchpad → /niche OODA → context full → scratchpad written → die
  └── Percept 3: read scratchpad → /niche OODA → ...
      All share the same characteristic states set in /valuate (Percept 1)
      All narrate laterally to other terminals via /sync
```

One terminal = one identity = one workflow. Valuation is slow (once). Niche construction is fast (many OODA cycles). You don't re-evaluate what matters every context refresh — you re-evaluate what to DO next.

### Renaming

| Old | New | Rationale |
|---|---|---|
| `/attune` | `/valuate` | Sets precision weights = valuation of what matters. Static graph. |
| `/coordinate` | `/niche` | Niche construction within the valued landscape. Dynamic graph. |
| "supra session" | "session" | Supra IS the coordinator — same entity, two phases |
| "characteristic states" | kept | Already correct terminology |

## Context

The system currently uses manually-named profiles (`friday-evening.yaml`, `sunday-wrapup.yaml`, etc.) as quick-start presets during valuation. These work, but they are static snapshots that never evolve. Meanwhile, session data (`supra/sessions/*.yaml`) is never analyzed or fed back.

The human's attentional preferences follow temporal rhythms. Friday evenings are creative infrastructure sessions. Saturday mornings shift to focused production. Sunday afternoons become sprint-shipping bow-tying. These patterns are observable in the session history but not exploited by the system.

### What exists today

| Artifact | Location | Contents |
|---|---|---|
| Manual profiles | `supra/profiles/{name}.yaml` | mode, dimensions, focus, suppress, saved_at |
| Session states | `supra/sessions/{session_id}.yaml` | mode, dimensions, focus, suppress, last_attuned, last_attuned_by |
| Global prior | `supra/characteristic_states.yaml` | Current default state |
| Schema | `supra/schema.yaml` | 8 dimensions, 4 modes, 6 compound states |

Session files contain valuation timestamps (`last_attuned`) but no temporal metadata beyond that. The session ID is a random name (e.g., `swift-branching-isle`) with no temporal signal.

### The Bayesian mechanics constraint

From MEMORY.md: "Temporal weight decay toward a fixed mean is WRONG. Precision weights relax toward moving attractor trajectories per Bayesian mechanics." This means the update rule must not decay toward a global average. Each temporal segment has its own attractor, and that attractor itself drifts over time as the human's workflow evolves. The prior for "friday-evening" in March 2026 may differ from "friday-evening" in June 2026.

## Decision

Add a **temporal prior store** that records the dimension values chosen during each valuation, indexed by a temporal segment key (day-of-week + time-of-day bucket). Each new valuation updates the prior for that segment using an exponential moving average (EMA) with configurable learning rate. The valuate skill uses the temporal prior as the default starting point when no shorthand is provided, replacing or augmenting the current morning inread profile suggestion.

Manual profiles are preserved as-is. They serve a different purpose: named snapshots for specific work contexts (e.g., `stage2-unet-setup`) that are task-driven, not time-driven. Temporal priors are the time-driven complement.

### Design: thin layer, not rewrite

The temporal prior system is a new file (`supra/temporal_priors.yaml`) and ~60 lines of new functions in `supra_reader.py`. The valuate skill gets a small addition to Step 1 (read temporal prior) and Step 5 (write observation). No existing data structures change. No existing behavior breaks.

## Data Model

### Temporal segment keys

A segment key combines day-of-week and time-of-day bucket:

```
{day}-{time_bucket}
```

**Days**: `monday`, `tuesday`, `wednesday`, `thursday`, `friday`, `saturday`, `sunday`

**Time buckets** (4 buckets, aligned to the human's actual session patterns):

| Bucket | Hours (local time) | Rationale |
|---|---|---|
| `morning` | 06:00 - 11:59 | Morning orientation sessions |
| `afternoon` | 12:00 - 16:59 | Focused production work |
| `evening` | 17:00 - 21:59 | Creative/infrastructure sessions |
| `night` | 22:00 - 05:59 | Late sessions, unusual |

Examples: `friday-evening`, `saturday-morning`, `sunday-afternoon`.

These are the same naming convention as the existing manual profiles, which is intentional -- it makes the mental model continuous.

### Storage format: `supra/temporal_priors.yaml`

```yaml
# Auto-maintained by /valuate. Do not edit manually.
version: 1
learning_rate: 0.3    # EMA alpha
min_observations: 2   # Need at least N observations before suggesting as prior

segments:
  friday-evening:
    observations: 4
    first_seen: "2026-03-06T17:27:22Z"
    last_seen: "2026-03-13T18:30:00Z"
    prior:
      mode: focused           # Most recent EMA-weighted mode (majority vote)
      dimensions:
        execution_speed: 3.7
        exploration_vs_exploitation: 3.2
        code_quality: 3.1
        test_coverage: 2.3
        spatial_correctness: 2.8
        model_architecture: 4.6
        urgency: 3.0
        data_engineering_diligence: 3.0
    mode_history:             # Sliding window of last 5 modes for majority vote
      - focused
      - focused
      - creative
      - focused

  saturday-morning:
    observations: 2
    first_seen: "2026-03-07T09:00:00Z"
    last_seen: "2026-03-08T10:00:00Z"
    prior:
      mode: sprint
      dimensions:
        execution_speed: 4.5
        # ...
    mode_history:
      - sprint
      - focused

  # ... other segments as observed
```

Key design choices:

1. **Dimensions are floats (EMA outputs)**, rounded to integers only at read-time when used as defaults. The fractional part carries information about trend direction.
2. **Mode is categorical** -- cannot be averaged. Use a majority vote over a sliding window of the last 5 observations for that segment.
3. **Focus/suppress are NOT stored in temporal priors.** They are task-specific, not time-specific. The temporal prior suggests dimension weights and mode only.
4. **`min_observations: 2`** prevents the system from suggesting a prior based on a single data point. The first session in a new time slot records data but does not become a prior yet.
5. **Version field** for future schema migrations.

### Why not store this in the session files?

Session files are ephemeral (gitignored, runtime state). Temporal priors are accumulated knowledge that should persist across sessions and even across git branches. They go in `supra/` alongside the schema and global states, and they are gittracked.

## Update Rule

### Exponential Moving Average (EMA) per dimension

When `/valuate` writes a session state with dimension values `v_new`, the temporal prior for the current segment is updated:

```
prior[dim] = (1 - alpha) * prior[dim] + alpha * v_new
```

Where `alpha` is the learning rate (default 0.3).

**Why EMA over alternatives:**

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| Simple mean | Easy, stable | Equal weight to ancient observations; violates moving-attractor constraint | Reject |
| Exponential moving average | Recent observations weighted more; attractor naturally drifts; one parameter | Smooths over rare sessions; needs tuning | **Selected** |
| Full Bayesian with precision weighting | Most principled; matches the Bayesian mechanics framing | Requires variance tracking, conjugate prior design, more code complexity | Defer to v2 |
| Raw last-observation | Zero-lag; always reflects latest preference | No smoothing; single bad session corrupts the prior | Reject |

**Alpha = 0.3 rationale**: With weekend-warrior sessions (roughly 3 sessions per weekend, 12 per month), alpha=0.3 gives an effective memory of ~3 observations (half-life = ln(2)/ln(1/0.7) ~ 1.9 observations). This means the prior adapts within 2-3 sessions of a sustained preference change, which feels right for a human whose workflow evolves month-to-month.

### Mode update (majority vote)

Mode cannot be EMA'd. Instead, maintain a sliding window of the last 5 mode choices for each segment. The prior's mode is the majority. On ties, prefer the most recent choice.

### First observation (cold start)

When a segment has zero prior observations, the valuation proceeds as today (global prior or manual profile). The observation is recorded but not used as a prior until `min_observations` is reached.

### Respecting the moving-attractor constraint

EMA inherently satisfies the Bayesian mechanics constraint: the attractor for each segment moves with the human's choices. There is no decay toward a fixed global mean. Each segment has its own independent trajectory. If the human stops having Friday evening sessions, the Friday evening prior freezes at its last value rather than decaying to some neutral state -- it waits for new evidence.

A future v2 could replace EMA with a proper Bayesian update that tracks per-dimension precision (inverse variance), where high-variance dimensions get updated more aggressively and low-variance dimensions resist change. This would be the full Bayesian mechanics treatment. But EMA is the correct first approximation: it captures recency weighting and per-segment attractors without requiring variance estimation from sparse data.

## Integration with /valuate

### Changes to the valuate skill (SKILL.md)

**Step 1 (Read Current State) -- add temporal prior lookup:**

After reading the global/session state, also read the temporal prior for the current segment:
1. Determine current temporal segment from local time (e.g., `friday-evening`).
2. Look up that segment in `temporal_priors.yaml`.
3. If it exists and has `observations >= min_observations`, it becomes the "suggested prior."

**Step 3.5 (Morning Inread) -- surface temporal prior:**

Currently, the inread surfaces manual profiles by matching day-of-week names. Replace this with temporal prior awareness:

```
Good morning. Here's your inread before we valuate:

[reading list as before]

Temporal prior for friday-evening (4 observations):
  Mode: focused | Speed: 3.7->4 | Explore: 3.2->3 | Quality: 3.1->3 |
  Tests: 2.3->2 | Spatial: 2.8->3 | Model: 4.6->5
  (Your Friday evenings tend toward focused mode with high model attention)

Manual profile also available: "friday-evening" (saved 2026-03-13)

Say "use prior" to start from the learned prior, "load friday-evening" for the
manual profile, or answer the questions to set custom values.
```

The temporal prior shows both the raw EMA value and the rounded integer that would be used. The one-line natural language summary ("Your Friday evenings tend toward...") highlights the distinctive features of that segment -- dimensions that differ from the global default by more than 1 point.

**Step 5 (Apply Changes) -- record observation:**

After writing the session state, also call `record_temporal_observation()` to update the prior for the current segment. This happens regardless of whether the human used shorthand, the questionnaire, or "use prior" -- every valuation is an observation.

**New shorthand: `use prior`:**

Recognized in Step 3 of `/valuate` as a shorthand that loads the temporal prior's rounded values as the starting state. Equivalent to shorthand with each dimension set explicitly, but with less typing.

### Changes to supra_reader.py

Add these functions:

```python
TEMPORAL_PRIORS_PATH = SUPRA_DIR / "temporal_priors.yaml"

def _temporal_segment_key() -> str:
    """Return current temporal segment key like 'friday-evening'."""

def read_temporal_priors() -> dict:
    """Read temporal_priors.yaml."""

def get_temporal_prior(segment: str | None = None) -> dict | None:
    """Get the prior for a segment. Returns None if < min_observations."""

def record_temporal_observation(states: dict, segment: str | None = None) -> bool:
    """Record a valuation observation, updating the EMA prior."""

def temporal_prior_to_states(prior: dict) -> dict:
    """Round a temporal prior's float dimensions to integers for use as states."""
```

Total: ~60 lines of new code. All pure functions except the I/O wrappers.

## Evolution Visualization

The human should be able to see how their temporal preferences have evolved. Two levels:

### Level 1: Inline during /valuate (v1, implement now)

When surfacing the temporal prior in Step 3.5, show a trend indicator for dimensions that have moved significantly (more than 0.5 in the last 3 observations):

```
Temporal prior for saturday-morning (6 observations):
  Mode: sprint | Speed: 4.8 | Model: 3.2 (trending down from 4.1)
```

This is free -- it just requires comparing the current EMA against the EMA from 3 observations ago, which can be stored as a secondary field or computed from the observation count and current value.

### Level 2: `/valuate history` command (v2, defer)

A dedicated command that prints a compact table of all segments with their priors:

```
Temporal priors (7 segments, 23 total observations):

| Segment           | N  | Mode    | Speed | Explore | Quality | Tests | Model |
|-------------------|----|---------|-------|---------|---------|-------|-------|
| friday-evening    | 7  | focused | 3.7   | 3.2     | 3.1     | 2.3   | 4.6   |
| saturday-morning  | 6  | sprint  | 4.8   | 2.1     | 3.5     | 3.8   | 3.2   |
| saturday-evening  | 4  | focused | 3.2   | 4.1     | 3.0     | 2.0   | 4.8   |
| sunday-morning    | 3  | focused | 3.5   | 3.0     | 4.2     | 4.0   | 3.1   |
| sunday-afternoon  | 3  | sprint  | 4.9   | 1.5     | 4.0     | 3.5   | 2.0   |
```

This tells a story: Friday evenings are model-focused. Saturday mornings are fast execution. Sunday afternoons are ship-it sprints. The human can see their own rhythms reflected back.

## Migration

### Existing manual profiles -> temporal priors (bootstrap)

The 7 existing manual profiles contain temporal signal in their names and `saved_at` timestamps. On first run, `record_temporal_observation()` can bootstrap the temporal prior store:

1. For each profile with a temporal name that maps to a segment key (e.g., `friday-evening` -> `friday-evening`, `sunday-wrapup` -> `sunday-afternoon` based on `saved_at` timestamp):
   - Record as a single observation with `observations: 1`
   - Set `first_seen` and `last_seen` to the profile's `saved_at`

2. For each session file with a `last_attuned` timestamp:
   - Determine the temporal segment from the timestamp
   - Record as an observation

3. Profiles with non-temporal names (`training`, `deep-research`, `ship-it`, `stage2-unet-setup`) are not migrated -- they are task-driven profiles, not time-driven.

**Migration is one-time and additive.** It populates the temporal prior store with initial data. After migration, the prior store is maintained by `/valuate` alone. Manual profiles continue to work as before.

### Bootstrap data available today

| Source | Segment | Dimensions available |
|---|---|---|
| `friday-evening.yaml` (profile, saved 2026-03-13 18:00) | `friday-evening` | full |
| `creative-evening.yaml` (profile, saved 2026-03-06 17:27) | `friday-evening` (based on date) | full |
| `sunday-wrapup.yaml` (profile, saved 2026-03-08 10:00) | `sunday-morning` | full |
| `swift-branching-isle` (session, attuned 2026-03-08 10:00) | `sunday-morning` | full |
| `azure-listening-tide` (session, attuned 2026-03-08 10:30) | `sunday-morning` | full |
| `mossy-spreading-leaf` (session, attuned 2026-03-08 12:00) | `sunday-afternoon` | full |

That gives 3 observations for `sunday-morning`, 1 for `friday-evening`, 1 for `sunday-afternoon`. Barely enough to start suggesting priors for Sunday mornings. The system will become useful after ~4 more weekends (16+ observations spread across segments).

### Manual profiles are NOT deprecated

Manual profiles serve a different function: named snapshots for specific work contexts. The human might want a `training-run` profile that they load regardless of what day it is. Temporal priors are the time-driven complement, not a replacement.

The priority order when `/valuate` suggests defaults:

1. Shorthand arguments (explicit, highest priority)
2. `load:name` (explicit profile load)
3. `use prior` (explicit temporal prior)
4. Temporal prior (suggested in morning inread if available)
5. Manual profile (suggested in morning inread if name matches day/time)
6. Global characteristic_states.yaml (fallback)

## Session Identity

### One terminal = one session = one identity

```
Terminal 1 (workflow: "LR schedule fix")
  │ /valuate: model_architecture=5, speed=4
  ├── Process 1: /valuate → /niche OODA → context full → plan handoff
  ├── Process 2:            /niche OODA → context full → plan handoff
  └── Process 3:            /niche OODA → ...

Terminal 2 (workflow: "probe analysis")
  │ /valuate: spatial_correctness=5, speed=3
  ├── Process 1: /valuate → /niche OODA → context full → plan handoff
  └── Process 2:            /niche OODA → ...
```

Each terminal has its OWN characteristic states because each workflow needs different things. The session identity persists across context window refreshes within a terminal. Lateral messages between terminals use a stable identity, not the ephemeral coordinator process ID.

### The problem today

Currently, session identity is fragmented:
- **Coordinator sessions** have random poetic IDs (`grey-passing-storm`, `swift-branching-isle`) that churn with every new Claude Code process
- **Supra session files** (`supra/sessions/{coordinator_id}.yaml`) are keyed by coordinator ID — so each process within a terminal gets its own file
- Lateral messages reference the coordinator ID, so a conversation between Terminal 1 Process 1 and Terminal 2 looks broken when Terminal 1 restarts as Process 2 with a new ID
- There is no concept of "these 5 coordinator processes were all part of the same Friday evening work session"

### The fix: deterministic supra session IDs

A **supra session** is identified by temporal segment + date:

```
friday-evening-2026-03-13
saturday-morning-2026-03-14
```

This is deterministic: every Claude Code process that starts on Friday evening 2026-03-13 computes the same supra session ID. All processes — across context window refreshes and across terminals — join the same supra session.

### Data model

```yaml
# supra/sessions/friday-evening-2026-03-13.yaml
supra_session_id: friday-evening-2026-03-13
temporal_segment: friday-evening
date: "2026-03-13"
created_at: "2026-03-13T18:30:00Z"
last_attuned: "2026-03-13T18:35:00Z"
coordinators:                          # All processes that have run under this identity
  - grey-passing-storm                 # Terminal 1, Process 1
  - blue-rising-tide                   # Terminal 2, Process 1
  - amber-quiet-dawn                   # Terminal 1, Process 2 (after context refresh)
mode: focused
dimensions:
  execution_speed: 4
  exploration_vs_exploitation: 2
  # ...
focus:
  - "LR schedule fix + retrain"
suppress:
  - "supra infrastructure"
```

Key design:
1. **File name is deterministic** (`{segment}-{date}.yaml`), not random
2. **`coordinators` list** is append-only — tracks ALL coordinator processes that ran under this supra
3. **`temporal_segment` field** makes the segment explicit for temporal prior updates
4. **Always created** — SessionStart hook ensures the file exists, even without `/valuate`
5. **Weights persist** — Process 2 in a terminal inherits supra weights without re-attuning

### Lifecycle

1. **SessionStart hook** (`session-start.py`):
   - Compute supra session ID from current time: `_temporal_segment_key() + "-" + date`
   - If supra session file doesn't exist, create it with defaults (temporal prior → global prior fallback)
   - Append this coordinator's ID to the `coordinators` list
   - Write `.current_supra_session_id` alongside `.current_session_id`
   - Inject supra weights into context so `/niche` can start immediately without `/valuate`

2. **`/valuate`** (runs once at terminal start, optional on subsequent processes):
   - Reads current supra session file as the starting point
   - Writes updated weights back to the supra session file (shared across all processes)
   - Records a temporal prior observation
   - On subsequent processes in the same terminal, `/valuate` is skipped — the coordinator just reads the existing supra session weights

3. **Stop hook** (`stop.py`):
   - Does NOT remove coordinator from supra session (append-only history)
   - Does NOT delete the supra session file (it persists as the observation record)
   - Fires temporal prior update if this was a valuated session (has `last_attuned`)

4. **Temporal prior update**:
   - Fires on `/valuate` (immediate) — the supra session's weights become the observation
   - One observation per supra session, not per coordinator process — keyed by supra session ID
   - If the human re-valuates (tweaks weights mid-session), the observation is updated in place

### Lateral identity (Homo Narrans)

Lateral Percept ↔ Percept coupling in the dynamic graph is **narrative exchange between terminals**. Each terminal is a narrator (homo narrans). The supra session ID is the narrator's name — it persists across percept deaths (context window refreshes) so the story stays coherent.

When coordinator processes exchange messages via `/sync`, the identity field is the **supra session ID**:

```yaml
# .claude/coordinators/messages/1710355200-friday-evening-2026-03-13.yaml
from: friday-evening-2026-03-13    # NOT grey-passing-storm
to: all
level: info
text: "Working on LR schedule fix. Claiming stage2_fusion/"
```

Terminal 2 sees a consistent sender identity even if Terminal 1 has restarted 3 times. The conversation thread stays coherent because the narrator's identity is stable.

`/loop 5m /sync` is the heartbeat of this narrative exchange — the mechanism by which the Per ↔ Per dotted edges in the dynamic graph become real. Without `/sync`, the terminals are isolated percepts with no lateral coupling, and you're back in the static graph topology.

The coordinator's poetic ID (`grey-passing-storm`) is still used for:
- Coordinator claim files (fine-grained conflict detection within a percept's lifetime)
- Scratchpad entries (which specific percept wrote what — intra-lineage memory)
- Heartbeat tracking (is this specific percept alive?)

### Backward compatibility

- Existing `supra/sessions/*.yaml` files (3 files with poetic names) are legacy
- Migration: read their timestamps, rename to `{segment}-{date}.yaml` format
- `_current_session_id()` continues to return the coordinator ID for process-level operations
- New `_current_supra_session_id()` added for supra-level operations
- `read_session_states()` updated to try supra session first, then fall back to coordinator session, then global

### Hook changes required

| Hook | Change |
|---|---|
| `session-start.py` | Compute supra session ID, create/join supra session file, register coordinator |
| `stop.py` | Fire temporal prior update if valuated session |
| `supra_reader.py` | Add `_temporal_segment_key()`, `_supra_session_id()`, `_current_supra_session_id()`, supra session read/write |
| `subagent-context.py` | Inject supra session ID alongside coordinator session ID |
| `coordinator_registry.py` | Messages use supra session ID as sender identity |

## Consequences

- **Positive**: The system learns the human's temporal rhythms without manual profile management. After ~4 weekends of use, `/valuate` can pre-fill sensible defaults that the human only needs to confirm or tweak.
- **Positive**: Thin layer -- one new YAML file, ~60 lines in supra_reader.py, small additions to valuate SKILL.md. No existing behavior changes. No existing files modified (except additive changes to supra_reader.py and SKILL.md).
- **Positive**: Satisfies the Bayesian mechanics constraint. EMA provides per-segment moving attractors with no decay toward a global mean. A future v2 can upgrade to full precision-weighted Bayesian updates.
- **Positive**: Bootstrap from existing data gives the system a head start rather than a cold start.
- **Negative**: With the weekend-warrior pattern (3-4 sessions per weekend), segments will take ~4 weekends to accumulate `min_observations=2`. Weekday segments may never accumulate data. The system is most useful for recurring patterns.
- **Negative**: `temporal_priors.yaml` is gittracked (accumulated knowledge), which means it shows up in diffs. It changes every session. Mitigation: it is a single file with small incremental changes, and the human can `.gitignore` it if the noise bothers them.
- **Negative**: EMA is a simplification. It does not track per-dimension confidence (precision). A dimension that the human always sets to 5 should be high-confidence; one that swings between 2 and 5 should be low-confidence and presented differently. This is the v2 Bayesian upgrade path.
- **Neutral**: The 4 time buckets are somewhat arbitrary. The human's "evening" might start at 16:00 or 19:00 depending on the day. But the buckets only need to be roughly right -- the EMA will average out minor timing variations.

## Graph-Driven Orchestration

The JSON graph (`deepresearch/liveability_approaches_graph.json`) is not just documentation — it should govern which communication channels are active at runtime. This section defines how the graph topology maps to `.claude/` infrastructure enforcement.

### Edge → Implementation Mapping

| Edge | Static (`/valuate`) | Dynamic (`/niche`) | Enforced by |
|---|---|---|---|
| **Ind → Per** | Read-only: git log, scratchpads, file reads | Read AND write: agent edits, commits | Pre-edit gate in `/niche` SKILL.md (already exists) |
| **Per ↔ N/D** | Bidirectional: questionnaire + human answers | N/D → Per only: weights propagate down, human holds veto at checkpoints | `/valuate` SKILL.md Steps 2-5; `/niche` gating policy |
| **N/D ↔ N/D** | Active: compound states resolve internal tradeoffs | Inactive: needs already resolved | `schema.yaml` compound_states; only invoked during `/valuate` |
| **Per ↔ Per** | Inactive: no lateral coupling | Active: `/sync` message passing | `/sync` skill + coordinator_registry messages; `/loop 5m /sync` |

### What `supra_reader.py` Should Expose

```python
def get_active_graph() -> str:
    """Return 'static' or 'dynamic' based on current mode.

    Static = during /valuate (setting weights).
    Dynamic = during /niche (executing work).
    """

def get_edge_topology(graph: str = None) -> list[dict]:
    """Load edges from liveability_approaches_graph.json for the active graph.

    Returns the edge list that defines valid information flows.
    Hooks can validate that actual flows match the topology.
    """

def is_lateral_coupling_active() -> bool:
    """True during /niche (dynamic graph), False during /valuate (static graph).

    Controls whether /sync messages are sent/received.
    Controls whether subagent-context.py injects cross-agent scratchpad content.
    """
```

### Hook Enforcement Points

| Hook | Static behavior | Dynamic behavior |
|---|---|---|
| `session-start.py` | Create supra session file, compute temporal segment | Same + register narrator identity for `/sync` |
| `subagent-context.py` | Inject supra weights only | Inject supra weights + cross-agent scratchpad signals + lateral messages |
| `stop.py` | Fire temporal prior update | Fire temporal prior update + write scratchpad (percept death trace) |
| `coordinator_registry.py` | Messages suppressed (no lateral coupling) | Messages active, using supra session_id as narrator identity |

### The `/sync` Skill as Per ↔ Per Edge

`/sync` is the runtime implementation of the dotted Per ↔ Per edges in the dynamic graph. It should:
1. Broadcast: what this terminal is working on (narrator's story)
2. Listen: what other terminals are doing (other narrators' stories)
3. Identify by supra session_id (narrator's name persists across percept deaths)
4. Only be active during `/niche` (dynamic graph topology)

`/loop 5m /sync` = continuous narrative heartbeat. Without it, concurrent terminals are isolated percepts and the system degrades to multiple independent static graphs.

### The Graph JSON as Schema Validation

Future enhancement: hooks load the JSON and validate that actual information flows match the active graph's edge list. For example:
- If in static mode and an agent attempts to write a file → warn (Ind→Per should be one-way)
- If in static mode and `/sync` fires → suppress (no Per↔Per coupling)
- If in dynamic mode and the human is asked a questionnaire → warn (N/D→Per should be one-way down)

This is enforcement-by-topology rather than enforcement-by-rules. The rules emerge from the graph structure.

## Implementation Notes

### Ordering

1. Add temporal segment + supra session functions to `supra_reader.py` (pure logic, no behavior change)
2. Update `session-start.py` to create supra session files on every session start
3. Update `stop.py` to deregister coordinator from supra session + fire temporal prior update on last coordinator exit
4. Create `supra/temporal_priors.yaml` with version header and empty segments
5. Run bootstrap migration (one-time script: convert existing 3 poetic session files + profiles to new format)
6. ~~Rename skill dirs: `skills/attune/` → `skills/valuate/`, `skills/coordinate/` → `skills/niche/`~~ DONE 2026-03-13
7. Update valuate SKILL.md: temporal prior in Steps 1, 3, 3.5, 5; graph-theoretical framing
8. Update niche SKILL.md: graph-theoretical framing (percepts, homo narrans, N/D→Per direction)
9. Add graph-driven orchestration functions to `supra_reader.py` (`get_active_graph()`, `is_lateral_coupling_active()`)
10. Gate `/sync` and `coordinator_registry.py` on active graph topology
11. Test: `/valuate` → `/niche` lifecycle, session inheritance across context refreshes

### Dependencies

- PyYAML (already a dependency)
- `datetime` for temporal segment computation (stdlib)
- No new packages required

### File inventory

| File | Change type | Scope |
|---|---|---|
| `.claude/supra/temporal_priors.yaml` | New | Temporal prior data store |
| `.claude/hooks/supra_reader.py` | Additive | ~100 lines: temporal segment, supra session ID, graph topology, temporal prior CRUD |
| `.claude/hooks/session-start.py` | Edit | Create/join supra session, register narrator identity |
| `.claude/hooks/stop.py` | Edit | Fire temporal prior update, write percept death trace |
| `.claude/hooks/subagent-context.py` | Edit | Inject supra session ID, gate cross-agent context on active graph |
| `.claude/hooks/coordinator_registry.py` | Edit | Messages use supra session_id, lateral coupling gate |
| `.claude/skills/valuate/SKILL.md` | Edit | Add temporal prior Steps, graph-theoretical framing |
| `.claude/skills/niche/SKILL.md` | Edit | Add graph-theoretical framing (percepts, homo narrans, N/D→Per) |
| `.claude/skills/sync/SKILL.md` | Edit | Gate on `is_lateral_coupling_active()`, use supra session_id |
| `deepresearch/liveability_approaches_graph.json` | Reference | Source of truth for edge topologies (loaded at runtime) |
| `scripts/one_off/bootstrap_temporal_priors.py` | New, temporary | Migrate 3 legacy session files + profiles |

### v2 upgrade path (not in this spec)

Replace EMA with a per-dimension Bayesian update that tracks both mean and precision (inverse variance). High-precision dimensions (always the same value) are presented as strong defaults. Low-precision dimensions (volatile) are flagged as "uncertain -- you should confirm this one." This would make the valuate questionnaire adaptive: skip questions for high-confidence dimensions, ask about low-confidence ones.

This matches the full Bayesian mechanics vision: precision weights relax toward moving attractor trajectories, and the precision itself encodes confidence. But it requires variance tracking from sparse data (weekend-warrior pattern gives ~12 observations per month per segment), which may not stabilize for months. EMA is the right v1.
