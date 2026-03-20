# Plan: Agent Teams — PARKED (Architectural Mismatch)

> **DO NOT IMPLEMENT.** Anthropic's agent teams do not fit the shard model. This document explains why and what would need to change upstream for reconsideration.

## Why This Doesn't Work

The shard model requires: **terminal = PPID = one `/valuate` → `/niche` column**. Each shard has its own supra identity, characteristic states, and session lifecycle — all keyed by terminal PID.

Anthropic's teammates are **subprocesses of the lead agent**, not independent terminals. They:
- Share the lead's working directory and process tree
- Have no independent PPID or shell process
- Cannot run `/valuate` (no static graph column)
- Cannot hold supra identity (no PPID-isolated session files)
- Are percepts within the lead's shard, not shards themselves

This makes them long-lived subagents, not autonomous cells. They're cells without nuclei — more autonomous than current tool-call subagents but without the identity infrastructure that makes our system work.

## The Core Issue

Valuation (`/valuate`) is the human's role: negotiating characteristic states (needs/desires) per terminal. Agent teams would require either:
1. **Automated valuation** — agents setting their own characteristic states (removes human from the static graph, defeats the purpose)
2. **Shared valuation** — all teammates inheriting the lead's states (makes them subagents with extra steps, no advantage over current model)
3. **Terminal-level teammate isolation** — each teammate gets its own terminal/PPID (Anthropic doesn't ship this)

None of these options are satisfactory today.

## What Would Reopen This

- Anthropic ships teammates as independent terminal sessions (own PID, own shell)
- OR the shard model evolves to support non-terminal identity (unlikely — terminal isolation is a feature, not a limitation)

## Historical Context

Originally Plan 3 in the coordination roadmap (`2026-03-20-coordination-roadmap.md`). Parked 2026-03-20 after architectural analysis showed fundamental mismatch with PPID-keyed shard model.
