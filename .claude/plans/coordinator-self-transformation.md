# Plan: Coordinator Self-Transformation via Levin's Design Principles

## Context

The coordinator system has three operational layers:
1. **Vertical** (depth): Coordinator -> Sub-agents (specialist delegation via Task tool)
2. **Lateral** (breadth): Coordinator <-> Coordinator (the recently-implemented advisory chat protocol via `.claude/coordinators/`)
3. **Supra** (apex): The human user -- the innermost point of the cognitive light cone, where past and future fold, delegating across concurrent coordinator workstreams

This maps to Levin's cognitive light cone as a **tetrahedron**: the human at the vertex (where internal and external meet), the Markov blanket as the boundary line, and past/future extending outward as a V-shape -- temporal retrospection and projection, entropy and energy.

The current system works but doesn't **adapt its own protocols**. Ego has flagged the same issues for 4+ sessions (coordinator-as-implementer, missing QAQC response protocol, wave-skip without rationale). The system enforces structure but doesn't reconstruct itself based on observed failures. This plan applies Levin's 5 principles to make the coordinator self-improving.

**Paper**: [Bootstrapping Life-Inspired Machine Intelligence](https://arxiv.org/abs/2602.08079) -- Levin et al.

## Architecture: The Three-Scale Cognitive System

```
                    +-------------+
                    |   HUMAN     |  Supra-coordinator
                    |  (vertex)   |  Delegates across workstreams
                    +------+------+  Longest temporal reach
                           |
              +------------+------------+
              |            |            |
        +-----+-----+ +---+----+ +-----+-----+
        | Coord A   |<-> Coord B |<->  Coord C  |  Lateral peers
        | (session) | |(session)| | (session)  |  Mid-range temporal reach
        +-----+-----+ +---+----+ +-----+-----+  File-based chat protocol
              |            |            |
         +----+----+  +---+---+   +----+----+
         | agents  |  |agents |   | agents  |    Specialists (sub-agents)
         |(stage1, |  |(qaqc, |   |(stage3, |    Shortest temporal reach
         | stage2) |  |devops)|   | ego)    |    Scratchpad memory
         +---------+  +------+   +---------+
```

Each coordinator is a "cell" with its own OODA loop. The lateral chat protocol is the "bioelectric signaling" between cells. The human is the morphogenetic field -- setting the large-scale pattern that cells self-organize toward.

## Files to Modify

| File | Changes | Principle |
|------|---------|-----------|
| `.claude/skills/coordinate/SKILL.md` | Delegation checkpoint, QAQC response protocol, wave-skip rationale, context pressure, three-scale awareness, heartbeat integration | Continuous Reconstruction, Embodied Constraints |
| `.claude/agents/coordinator.md` | Autonomy contracts, self-modification protocol, supra-coordinator awareness | Multiscale Autonomy, Self-Assemblage |
| `.claude/hooks/subagent-context.py` | Cross-specialist signaling (stage pipeline signals) | Pervasive Signaling |
| `.claude/hooks/session-start.py` | Cognitive light cone metrics, expanded signal vocabulary | Pervasive Signaling, Embodied Constraints |
| `.claude/hooks/subagent-stop.py` | Quality trend annotation | Pervasive Signaling |
| `.claude/rules/multi-agent-protocol.md` | Autonomy scope, signal vocabulary | Multiscale Autonomy |
| `.claude/rules/coordinator-coordination.md` | Supra-coordinator concept, lateral signaling enrichment | Pervasive Signaling |

## Improvements by Principle

### 1. Continuous Reconstruction (P0)

The system rebuilds itself based on observed failure patterns, like bone remodeling under stress.

**1a. Delegation Checkpoint** -> `SKILL.md` ACT section
```markdown
### Pre-Edit Gate (self-enforcing Process Rule #3)
Before ANY Edit/Write call in coordinator mode:
1. Identify the file: does it live in stage*/, utils/, or core scripts/?
2. If yes -> MUST delegate. Name the agent in your OODA report.
3. If no (one-off scripts, .claude/ config, trivial infrastructure) -> proceed directly.
4. Log the decision in your scratchpad either way.

This gate exists because ego flagged coordinator-as-implementer in 6/8 sessions.
```

**1b. QAQC Response Protocol** -> `SKILL.md` new section after LOOP
```markdown
### QAQC Response Protocol
When QAQC reports partial-fail or fail:
1. SAME SESSION: dispatch fix agent targeting each finding, OR
2. LOG DEFERRAL: write priority + rationale in coordinator scratchpad
3. NEVER silently acknowledge. This gap caused linear_probe_viz.py:813 to persist 2+ sessions.

When ego recommends a process change for 2+ consecutive sessions:
- Address it in the NEXT session's Wave 1
- Either implement the recommendation OR log explicit disagreement with rationale
- 3+ session recommendations without action auto-escalate to P0
```

**1c. Wave-Skip Rationale** -> `SKILL.md` DECIDE section
```markdown
If deviating from a plan's wave structure:
- User-driven: note in scratchpad, proceed (healthy adaptation)
- Coordinator-driven: MUST log rationale BEFORE acting
- "Efficiency" alone is not sufficient -- explain what dependency changed
```

### 2. Pervasive Signaling (P0)

Bioelectric gradients flow between adjacent tissues. Our agents need richer signals than just BLOCKED/CRITICAL.

**2a. Cross-Specialist Signal Injection** -> `subagent-context.py`

Add a `get_sibling_signals()` function that scans today's scratchpads from pipeline-adjacent agents and injects relevant signals. The adjacency graph:

```python
PIPELINE_ADJACENCY = {
    "stage1-modality-encoder": ["stage2-fusion-architect", "srai-spatial"],
    "stage2-fusion-architect": ["stage1-modality-encoder", "stage3-analyst", "srai-spatial"],
    "stage3-analyst": ["stage2-fusion-architect"],
    "srai-spatial": ["stage1-modality-encoder", "stage2-fusion-architect", "stage3-analyst"],
    "qaqc": ["*"],  # QAQC sees all signals
    "librarian": ["*"],  # Librarian sees all signals
}

SIGNAL_KEYWORDS = [
    "BLOCKED", "URGENT", "CRITICAL", "BROKEN",
    "SHAPE_CHANGED", "INTERFACE_CHANGED", "DEPRECATED", "NEEDS_TEST",
]
```

When spawning stage3-analyst, scan stage2-fusion-architect's today scratchpad for these keywords and inject matching lines. This is the "bioelectric signaling" -- gradients propagate along the pipeline.

**2b. Structured Signal Vocabulary** -> `session-start.py`

Extend `CRITICAL_KEYWORDS` to include the full signal vocabulary. Also add to `multi-agent-protocol.md` as a reference table so agents know what signals to emit.

**2c. Coordinator Chat Heartbeat in OODA** -> `SKILL.md`

The lateral coordinator chat system already has heartbeats. Integrate message-checking into the OODA OBSERVE phase explicitly:
```markdown
#### OBSERVE (updated)
- git log/status (as before)
- /summarize-scratchpads (as before)
- **NEW**: Read coordinator messages from `.claude/coordinators/messages/`
  addressed to this session_id or "all". Surface in OODA report.
- **NEW**: Update heartbeat via coordinator_registry.update_heartbeat()
```

### 3. Multiscale Autonomy (P1)

Cells are competent at their own level. Agents shouldn't need coordinator permission for decisions within their domain.

**3a. Autonomy Contracts** -> `coordinator.md` new section

```markdown
## Autonomy Contracts

Each agent has a defined scope of autonomous decisions:

| Agent | Autonomous Decisions | Must Escalate |
|-------|---------------------|---------------|
| stage2-fusion-architect | Layer sizes, activations, normalization, optimizer params | New model classes, changed I/O contracts, new losses |
| stage3-analyst | Viz style, clustering params, probe hyperparams | New probe types, changed eval metrics |
| qaqc | Test structure, fixtures, which checks to run | Changing acceptance criteria, skipping checks |
| devops | Patch/minor package versions, branch strategy | Major version upgrades, new dependencies |
| srai-spatial | SRAI API patterns, query optimization | New spatial conventions, index changes |
| spec-writer | Doc structure, section organization | Architectural decisions, new conventions |
| ego | Assessment methodology, scoring criteria | Changing process rules, adding new rules |

Agents operating within scope do NOT need coordinator approval.
The scratchpad IS the accountability mechanism -- document the decision, don't ask permission.
```

**3b. Supra-Coordinator Awareness** -> `coordinator.md` + `coordinator-coordination.md`

Add a section documenting the human's role as supra-coordinator:
```markdown
## The Human as Supra-Coordinator

You (the coordinator) are one of potentially several concurrent coordinators.
The human user is the supra-coordinator -- the apex of the cognitive light cone.

Your relationship to the human:
- The human delegates workstreams to coordinators (lateral peers)
- Each coordinator delegates tasks to specialists (vertical depth)
- The human sees across all workstreams; you see only yours + lateral signals
- When you encounter cross-workstream dependencies, escalate to the human

The cognitive light cone tetrahedron:
- Human (vertex): longest temporal reach, widest spatial reach
- Coordinators (edges): mid-range, session-scoped, laterally connected
- Specialists (faces): shortest reach, task-scoped, vertically connected
```

### 4. Self-Assemblage Growth (P1)

The system grows new capabilities when it detects gaps, like an organism adding new organs.

**4a. Agent Gap Detection** -> ego-check skill amendment

When ego assesses process health, also check:
- Were there tasks no existing agent was well-suited for?
- Did the coordinator handle a recurring category of work itself?
- Are there patterns in scratchpads suggesting a missing specialist?

If a gap appears 2+ times, recommend new agent type with: name, domain, triggers, autonomy scope.

**4b. Convention Discovery** -> librarian-update skill amendment

When updating the codebase graph, also check:
- New coding patterns used 3+ times without a matching rule?
- Import patterns that seem convention-worthy?

Note as "Candidate Convention: [pattern]" in scratchpad.

### 5. Embodied Constraints (P2)

Context windows and session boundaries aren't bugs -- they're compression features.

**5a. Context Pressure Management** -> `SKILL.md`
```markdown
### Context Pressure Management
- Context window = organism's metabolic budget. Don't waste it.
- When a wave returns verbose results: compress to 3-5 key findings before next wave
- 80-line scratchpad limit forces signal clarity -- this is a feature, not a limitation
- Session boundaries are "cell divisions" -- scratchpads are the DNA that persists
- Write coordinator scratchpad EARLY when approaching session end; don't rush
```

**5b. Cognitive Light Cone Metrics** -> `session-start.py`

Add a brief summary to SessionStart injection:
```python
def cognitive_light_cone_summary() -> str:
    """Quantify the system's cognitive reach."""
    # Temporal depth: how many days of scratchpad history?
    all_entries = sorted(SCRATCHPAD_ROOT.rglob("????-??-??.md"))
    days = len(set(e.stem for e in all_entries))

    # Agent reach: how many agents have contributed?
    agents = len([d for d in SCRATCHPAD_ROOT.iterdir()
                  if d.is_dir() and any(d.glob("*.md"))])

    # Unresolved items (forward projection quality signal)
    recent = all_entries[-5:] if all_entries else []
    unresolved = sum(
        e.read_text(encoding="utf-8").lower().count("unresolved")
        for e in recent
    )

    # Active coordinator count (lateral reach)
    active_coords = len(list(COORDINATORS_DIR.glob("session-*.yaml")))

    return (f"Light cone: {days}d memory, {agents} agents, "
            f"~{unresolved} unresolved, {active_coords} active coordinators")
```

## Wave Structure

**Wave 1** (parallel -- SKILL.md + coordinator.md updates):
1. **spec-writer**: Update `.claude/skills/coordinate/SKILL.md` with: delegation checkpoint (1a), QAQC response protocol (1b), wave-skip rationale (1c), context pressure (5a), coordinator chat integration in OODA (2c), three-scale architecture awareness
2. **spec-writer**: Update `.claude/agents/coordinator.md` with: autonomy contracts (3a), supra-coordinator awareness (3b), self-modification protocol (1b ego escalation)

**Wave 2** (parallel -- hook + rule code changes):
3. **devops**: Update `.claude/hooks/subagent-context.py` -- add `get_sibling_signals()` with pipeline adjacency graph and SIGNAL_KEYWORDS scanning (2a)
4. **devops**: Update `.claude/hooks/session-start.py` -- add `cognitive_light_cone_summary()` and extended signal vocabulary (2b, 5b)
5. **devops**: Update `.claude/rules/multi-agent-protocol.md` -- add signal vocabulary table, autonomy scope reference (2b, 3a)
6. **devops**: Update `.claude/rules/coordinator-coordination.md` -- add supra-coordinator concept (3b)

**Wave 3** (sequential -- verify):
7. **qaqc**: Test all hooks (`echo '{"source":"startup"}' | python .claude/hooks/session-start.py`, etc.), verify SKILL.md and coordinator.md consistency, verify no contradictions between rule files

**Wave 4** (sequential -- commit):
8. **devops**: Commit all changes

**Wave 5** (parallel -- deferred improvements):
9. **spec-writer**: Amend ego-check skill with agent gap detection guidance (4a)
10. **spec-writer**: Amend librarian-update skill with convention discovery guidance (4b)

**Wave 6** (sequential -- verify + commit):
11. **qaqc**: Re-verify Wave 5 changes
12. **devops**: Commit Wave 5 changes

**Final Wave**: coordinator scratchpad + `/librarian-update` + `/ego-check`

## Verification

1. Run each hook script with test input; verify new sections appear in output
2. Read SKILL.md -- verify delegation checkpoint, QAQC protocol, and coordinator chat are present
3. Read coordinator.md -- verify autonomy contracts and supra-coordinator section
4. Verify no contradictions between SKILL.md, coordinator.md, and rule files
5. Start a fresh session to confirm SessionStart outputs cognitive light cone metrics

## Pre-Execution

Rename this plan file to `.claude/plans/coordinator-self-transformation.md` before executing.

## Execution

Invoke: `/coordinate .claude/plans/coordinator-self-transformation.md`
