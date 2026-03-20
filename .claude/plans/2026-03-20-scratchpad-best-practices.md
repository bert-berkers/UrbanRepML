# Plan: Scratchpad Protocol Upgrade — Cross-Session Memory Strands

## Executive Summary

Anthropic shipped scratchpads in their agent teams — convergent evolution with our system built a month earlier. But the systems solve fundamentally different problems. Theirs: within-session multi-agent coordination (ephemeral). Ours: **cross-session memory strands** where `/valuate` crystallizes governance (natura naturata) and `/niche` is the generative process that vanishes when context windows die (natura naturans). This plan makes that distinction operational: inject graph mode into agent context (Wave 0), fix scratchpad bloat with Anthropic's delegation best practices (Wave 1), add reconciliation discipline to protocol rules (Wave 2), and template coordinator delegation (Wave 3). Three files, ~50 lines, no new infrastructure.

## Context

Our stigmergic scratchpad system predates Anthropic's agent teams by ~1 month. But the systems solve fundamentally different problems:

- **Anthropic's scratchpads**: within-session multi-agent coordination (ephemeral, cleaned up after)
- **Our scratchpads**: **cross-session memory strands** that maintain intent coherence across context window deaths

Each context window is a hula hoop keeping a section of flexible netting alive. `/valuate` (static, natura naturata) captures the characteristic states — not the infrastructure itself but the *pattern* development took around it, like deltas or lung branching. `/niche` (dynamic, natura naturans) is the generative process — the water that carved the delta, cross-hierarchical, small-worlding, applying itself to the governing mechanism set by valuate.

The project's `.claude/` architecture mirrors its subject: UrbanRepML studies how urban development patterns (static) emerged from continual flows of people and goods (dynamic). Same structure, meta level.

This plan makes the static/dynamic distinction first-class in the `.claude/` folder behavior.

## Wave 0: Mode-Aware Session State in `.claude/` (NEW)

**Problem**: The supra session YAML lives in `.claude/supra/sessions/` but the graph mode (static/dynamic) is just a field in it. The mode should shape the *entire* `.claude/` behavior — which scratchpads get read, whether signals propagate, what hooks enforce. Currently `subagent-context.py` reads `active_graph` from supra state, but scratchpads themselves are mode-unaware.

**Files**:
- `.claude/supra/sessions/{supra_id}.yaml` — already has `active_graph: static|dynamic`
- `.claude/hooks/subagent-context.py` — inject mode awareness into agent context
- `.claude/rules/multi-agent-protocol.md` — document mode-dependent behavior

**Changes**:
1. **Inject graph mode into every agent's context** (subagent-context.py): Add `**Graph mode**: static (valuating) | dynamic (niche)` to the protocol injection. Read `active_graph` from the supra state already loaded at line 252-262.
2. **Mode-dependent scratchpad guidance** (subagent-context.py):
   - Static mode: "You are crystallizing governance — the characteristic state that will direct all future work in this strand."
   - Dynamic mode: "You are leaving traces of a process that vanishes when this context window dies. Make the invisible visible."

## Wave 1: Delegation Template (subagent-context.py)

**Problem**: Ego flagged scratchpad bloat as systemic (3 agents at 2-3x budget). Root cause: agents get no consolidation guidance in dispatch prompts. Anthropic found that vague delegation = "duplicated work, leave gaps."

**File**: `.claude/hooks/subagent-context.py` (lines 201-213)

**Change**: Expand the scratchpad protocol injection with:
1. **Consolidation directive**: "If today's entry exists, READ it first, then REWRITE as a single coherent log — not append."
2. **Structured format hint**: "Use summary tables over narrative when listing multiple items" (Anthropic's finding: narrative per-task causes bloat, tables compress)
3. **Output reference pattern**: "For large outputs (code, data, reports), write to files and reference them in your scratchpad — don't paste inline" (Anthropic's "game of telephone" fix)

## Wave 2: Protocol Rules (multi-agent-protocol.md)

**Problem**: Unresolved lists accumulate without reconciliation. Anthropic uses explicit task states.

**File**: `.claude/rules/multi-agent-protocol.md`

**Changes**:
1. **Unresolved state tags**: Each unresolved item must be tagged `[open]`, `[stale]`, or `[blocked:reason]`. Items tagged `[stale]` for 2+ sessions should be removed or escalated.
2. **Reconciliation-first rule**: Strengthen to: "Before writing ANY new Unresolved items, explicitly mark each existing item as resolved (remove) or still-open (keep with tag). Not optional."
3. **Output reference norm**: "Reference large outputs by path, don't paste inline. Scratchpads are index + reasoning, not data."
4. **Memory Strands section** (new): Document the dual nature of scratchpads. Static mode scratchpads crystallize governance — the attractor basin that channels future flow. Dynamic mode scratchpads are the fossil record of natura naturans — traces of a generative process that vanishes when the context window dies. The hula hoop model: each context window sustains the strand temporarily; scratchpads are what survives.

## Wave 3: Coordinator Delegation (coordinator.md)

**Problem**: Anthropic found delegation needs "objective, output format, tool/source guidance, clear task boundaries." Our ACT phase has bullets but no template.

**File**: `.claude/agents/coordinator.md` (ACT section, lines 80-85)

**Change**: Add structured delegation template:
```
When dispatching a specialist, include:
1. **Objective**: What to achieve (1 sentence)
2. **Acceptance criteria**: How to know it's done
3. **Scope boundary**: What NOT to touch
4. **Context pointers**: "Read scratchpad X, librarian graph shows Y at path Z"
5. **Output location**: Where results should go (file path, not scratchpad)
```

## Wave 4: Optimise the Hook (subagent-context.py)

**Problem**: The SubagentStart hook injection is a fixed template. It doesn't tell agents where they are in the strand or what came before. The injection should carry enough context that agents can orient quickly without reading multiple scratchpads.

**File**: `.claude/hooks/subagent-context.py`

**Changes**:

1. **Strand position**: Inject sub-session count within the current niche (read from agent_timer births today). "You are agent 7 of this niche" tells late agents to converge, not explore.

2. **One-line summaries of prior sub-sessions**: Read completed agent obituaries from agent_timer, inject as:
   ```
   ### Strand history (this niche):
   1. [stage2] 14:32 — ring_agg normalization
   2. [qaqc] 14:45 — 220/220 tests pass
   3. [stage3] 15:01 — R²=0.556
   ```

3. **Grep markers** in injected sections and scratchpad format:
   - `<!-- SUMMARY: one-line -->` in scratchpads for machine extraction
   - `<!-- STRAND_HISTORY -->` in injection for structured parsing
   - `<!-- SIGNALS -->`, `<!-- SUPRA_WEIGHTS -->` around existing sections

## What NOT to do

- **No file locking**: Shard model provides natural isolation via distinct `/valuate` intents.
- **No explicit task board**: Scratchpad Unresolved + OODA waves serve the same function.
- **No TeammateIdle hook**: SubagentStop already blocks on missing scratchpads.
- **No changes to signal vocabulary or ego/QAQC agents**: Working. Don't touch.
- **No new YAML files or directories**: The supra session YAML already exists — we're enriching what's injected from it, not duplicating it.

## Files Modified

| File | Wave | Change |
|------|------|--------|
| `.claude/hooks/subagent-context.py` | 0+1+4 | Graph mode + consolidation directives + strand history + grep markers |
| `.claude/rules/multi-agent-protocol.md` | 2 | State tags + output ref + Memory Strands section + scratchpad markers |
| `.claude/agents/coordinator.md` | 3 | Structured delegation template in ACT |

## Verification

1. Commit all 3 files
2. Spawn a test subagent — verify graph mode, consolidation directive, strand history, and grep markers appear in injected context
3. Check SubagentStop still blocks/allows correctly (no behavioral change)
4. Verify grep markers are parseable: `grep -o '<!-- SUMMARY:.*-->' .claude/scratchpad/*/2026-03-20.md`
5. Read updated protocol rules and coordinator.md for clarity

## Estimated Scope

~80 lines changed across 3 files. No new files. No new dependencies. Wave 0 is the conceptual anchor; Waves 1-3 are the Anthropic best practices; Wave 4 calibrates injection to strand lifecycle.

## Roadmap Position

This is **Plan 1** of 3. See `2026-03-20-coordination-roadmap.md` for the full sequence:
- **Plan 1** (this): Scratchpad protocol — static governance (ready to implement)
- **Plan 2**: Strand coordination — dynamic `/sync` overhaul (research phase)
- **Plan 3**: Agent teams — autonomous multi-instance (far future)
