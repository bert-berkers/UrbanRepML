# Plan: Niche Coordination Overhaul — Strand-Aware `/loop /sync`

## Context

The scratchpad protocol plan (2026-03-20-scratchpad-best-practices.md) handles the static side — how scratchpads crystallize governance and carry memory across context window deaths. This plan handles the dynamic side — how strands coordinate *during* their overlapping lifetimes.

### The Problem with "Within vs Cross Session"

Sessions aren't cleanly "within" or "cross." They're **strands** with:
- Varying lifetimes — some complete their telos and die quickly, others persist across days
- Overlapping development — strands coalesce when their work aligns, diverge when it doesn't
- Different birth times — a strand born later may outlive one born earlier
- No clean boundaries — a strand doesn't know at birth how long it'll live or who it'll meet

The current `.claude/coordinators/messages/` system assumes clean session boundaries (session-to-session messaging with staleness thresholds). It should be replaced with strand-aware coordination that handles overlapping lifetimes naturally.

### What Anthropic's Mailbox Gets Right

| Feature | Their Implementation | What We'd Adopt |
|---------|---------------------|-----------------|
| Auto-delivery | Messages push to recipients, no polling | Replace our message directory scanning with push-on-write |
| Task states | pending → in-progress → completed | Replace scratchpad Unresolved lists for within-strand tracking |
| Dependencies | Tasks block until dependencies complete | Our signals (BLOCKED) are informal; this makes them structural |
| File locking | Prevents race conditions on task claims | Shard model makes this less critical, but useful for shared tasks |
| Direct + broadcast | 1-to-1 and 1-to-all | Maps to our targeted messages + `/sync` broadcast |

### What Anthropic's Mailbox Gets Wrong (for our case)

- **Tied to agent teams**: Can't use as standalone library — coupled to separate Claude Code instances
- **Ephemeral**: Cleaned up after team disbands — no temporal memory
- **Star topology**: Lead ↔ teammates only — no small-world shortcuts
- **Session-scoped**: Assumes clean session lifecycle — doesn't handle overlapping strands

## Design Direction

Replace `.claude/coordinators/messages/{date}/` with a strand-aware coordination layer that:

1. **Applies to `/loop /sync`** — this is where dynamic strand-to-strand coupling happens
2. **Handles overlapping lifetimes** — strands can coalesce and diverge without assuming session boundaries
3. **Keeps small-world topology** — local coupling (scratchpad stigmergy) + long-range shortcuts (signal vocabulary, ego panoptic view)
4. **Preserves the fossil record** — dynamic scratchpads remain as traces of vanished processes (natura naturans made visible)

## Open Questions (need research before implementation)

1. **Can we enable agent teams alongside our subagent system?** Or are they mutually exclusive?
2. **What does `/loop /sync` actually do today?** Map current behavior before redesigning.
3. **Task list as replacement for Unresolved?** The scratchpad Unresolved section is strand-scoped; Anthropic's task list is team-scoped. Different lifetimes. Can we hybridize?
4. **How does strand identity work?** Currently PPID-keyed. Should strands have their own identity independent of terminal PID?
5. **What's the minimal change?** Could we just add task states to scratchpad Unresolved items and get 80% of the benefit without new infrastructure?

## Estimated Scope

Larger than the scratchpad protocol plan. Touches:
- `.claude/hooks/` — sync mechanism, potentially agent timer
- `.claude/skills/sync/` — the actual `/sync` skill
- `.claude/coordinators/` — message system replacement
- `.claude/rules/coordinator-coordination.md` — protocol rules

Not ready to implement — needs the open questions answered first. Save as reference for a future session.

## Relationship to Other Plans

- **Depends on**: `2026-03-20-scratchpad-best-practices.md` (the static side must be solid first)
- **Extends**: The Memory Strands model (scratchpads as fossil record + governance crystallization)
- **Connects to**: Isomorphism 7 in `deepresearch/structural_isomorphisms.md`
