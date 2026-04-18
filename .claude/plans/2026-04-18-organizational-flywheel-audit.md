# Organizational Flywheel Audit (2026-04-18)

**Status**: ready to execute via `/niche` after `/clear`
**Mode**: audit-only — NO code changes, NO new pipelines, NO viz iteration
**Deliverable**: a single audit report at `reports/2026-04-18-organizational-flywheel-audit.md`
**Session**: golden-standing-creek (supra `golden-standing-creek-2026-04-18`)

---

## Framing — why this plan exists

Yesterday (2026-04-17, green-branching-kelp) lost four waves on a multi-level landscape viz that didn't land; user closed with "Opus 4.7 is not good in these kinds of tasks." Today the bet is the opposite: Opus 4.7 IS good at structural judgment about code & data systems. 

The deeper reason: **the organization (coordinator + subagents + human) performs OODA collectively. The OBSERVE substrate is the territory itself — code, experimental outputs, data. The ORIENT substrate is the four documentation artifacts — scratchpads, codebase_graph, specs, plans — which exist so we don't get lost in the territory over time.**

| OODA stage | What it is |
|---|---|
| **OBSERVE** | The **territory** — `stage*_*/` code, `data/study_areas/.../*` outputs, `wandb/`, `lightning_logs/`, `checkpoints/`, the actual repo state. It just IS. |
| **ORIENT** | The **map** — five artifact families that exist so the AI organization + human doesn't get lost as the territory grows: (1) **scratchpads** (intra-session memory), (2) **`codebase_graph.md`** (structural map of the territory), (3) **`specs/`** (architectural decisions), (4) **`.claude/plans/`** (queued actions), (5) **session states** — `.claude/supra/sessions/*.yaml`, `.claude/coordinators/sessions/*` claim files, `.claude/scratchpad/valuate/*.md`. Session states are the gyroscope's *current reading* — "where are we pointed right now?" |
| **DECIDE** | Human-in-the-loop chat (steered by the map) → new plan in `plans/` or revision of an existing one. |
| **ACT** | **Subagents do the work** — the coordinator's role is to DELEGATE, never to act directly. ACT can be a code change (territory modification), an audit (librarian inspecting code, ego inspecting process), or a synthesis (spec-writer compiling findings). The shape of ACT depends on the DECIDE; the constraint is that the coordinator orchestrates and the subagent executes. After acting, the map must be updated to reflect the new territory or the new understanding. |

**Markov property** (applies to the ORIENT artifacts): each OODA cycle must produce ORIENT artifacts that contain everything the next cycle needs. The next context window — fresh after `/clear`, fresh after a week, fresh from a different terminal — must be able to resume cold from the map alone, then go look at the territory. If the current ≤30-line scratchpad rule or any other format constraint breaks Markov-completeness, **change the rule.** Length is not the cost; loss-of-state is.

The flywheel spins when each ORIENT artifact's contract is honored. It rotts when the map drifts from the territory — a 3228-line codebase_graph that nobody updates, a 17-spec folder where nobody knows which are still load-bearing, an `archive/` of 53 scripts with no index. The audit's job is to:

1. Inventory how the **territory** (OBSERVE) is currently disordered (themes A, B, E) — disorder in the territory makes the map harder to keep accurate.
2. Inventory how the **map** (ORIENT) has rotted (themes C, D, F, G, H) — and **specify the contracts** that keep each of the five map artifacts Markov-sufficient. Session states are part of this — the audit must check that the gyroscope's current-reading machinery (supra files, claim files, valuate scratchpads) is itself reliable, since a wrong reference reading propagates as wrong navigation.

The audit deliverable is itself a map artifact (a report). It is meta-cartography: a map of how to keep the maps current.

### From flywheel to gyroscope

The OODA cycles spinning across context windows — coordinator + subagents + human, looping observe→orient→decide→act → again — that is a **flywheel**: motion that persists. Useful, but undirected. A flywheel alone tells you the system is alive; it does not tell you where it's going.

`/valuate` adds the **reference frame**: intent, mode, dimensions, focus, suppress. The supra session file (`.claude/supra/sessions/{supra_id}.{ppid}.yaml`) IS the gyroscope's reference axis. With that axis fixed, the same spinning motion becomes navigation:

- **Subagents have directional freedom** in HOW they execute — domain latitude within their scratchpad contract.
- **The gyroscope constrains WHAT and WHY** — through the supra reference frame, every wave is filtered against the current intent. A task that doesn't precess around the active reference axis gets deferred or rejected.
- **The human steers by re-valuating** — turning the gyroscope realigns the whole system without disturbing the flywheel's motion. (`/niche` checkpoints are for course correction, not re-valuation — the user re-runs `/valuate` to change the reference frame itself.)

This means the audit is not just "fix the maps." It's:

1. Fix the map artifacts (ORIENT) so the flywheel's persistence is real.
2. Fix the reference-frame plumbing (valuate → niche) so the flywheel's motion translates to navigation.

Theme H is really about both: niche must read the supra reference frame at every wave, not just at session start, AND the scratchpad handoff packet must carry the reference frame forward so the next context window's gyroscope starts spinning in the right orientation.

---

## Themes (eight; H is the meta-leverage point)

Each theme will be examined surface-by-surface. The audit report maps them onto the four-surface model.

- **A. Repo-root hygiene** — `nul`, `img.png`, 8+ stray logs at root (some 0-byte, one 2.5MB), `results [old 2024]/` (literal "old" + brackets), `lightning_logs/` (17+ versions), `wandb/` runs, `checkpoints/`, `keys/`, `cache/`, `files/`. → OBSERVE substrate (state observability).
- **B. Experimental result ledger missing** — `wandb/`, `lightning_logs/`, `checkpoints/` not catalogued. No mapping run → probe score → report → figure. MEMORY.md P0 #7 (checkpoint versioning, 2026-03-09) still open. → OBSERVE/ACT feedback (results not crystallized).
- **C. Specs sprawl** — 17 specs, no index, six are about Claude infra and overlap (`coordinator-hello`, `coordinator_to_coordinator`, `claude_code_multi_agent_setup`, `session-identity-architecture`, `between-wave-pause-redesign`, `hooks_architecture`). → ORIENT substrate (can't sense-make through bloat).
- **D. Archive opacity** — `scripts/archive/`: 53 files, 12 subdirs, no index. CLAUDE.md says "read-only reference" — but unsearchable is worse than deleted. → ORIENT substrate.
- **E. Stage3 results layout drift** — `data/.../stage3_analysis/` mixes `2026-02-20/` (date-stamped orphan) with `classification_probe/`, `dnn_probe/`, etc. (type-stamped). Two `cluster_results/` dirs (top-level + nested). → OBSERVE substrate (data state).
- **F. Reports↔figures decoupled** — 6 reports, ~3 figure entries. Reports cite figures that may not exist at the cited path or were overwritten. Reproducibility = "rerun + hope." → OBSERVE/ACT feedback.
- **G. `codebase_graph.md` rotted** — **3228 lines / 338KB**, header is a single paragraph summarizing 23 update rounds, `.bak-2026-03-29` sibling. Almost certainly contains drift. The librarian's job is "where does X live?" — at this size, grep wins. → THE OBSERVE artifact, currently broken.
- **H. OODA loop persistence (META)** — niche OODA is structured (Wave 0 → OBSERVE/ORIENT/DECIDE/ACT/LOOP) but the closeout obligation is thin: "write coordinator scratchpad before finishing" doesn't specify what makes a scratchpad sufficient as a **handoff packet** for a fresh context window. Current rules cover format (prior-entries index, ≤30 lines, signal vocabulary) but not **Markov-completeness**. The chain breaks where it should be strongest: end-of-window. → the meta-loop itself.

---

## Wave Plan

### Wave 0 — Clean state (mandatory)

1. Read this plan in full.
2. Read the supra session file at `.claude/supra/sessions/golden-standing-creek-2026-04-18.{ppid}.yaml` — valuation already set: `mode=focused, speed=3, explore=4, quality=4, tests=2, spatial=3, model=2, data_eng=5`. Focus: "OODA Markov-completeness, organizational flywheel, four-surface contracts." Suppress: "new code, new pipelines, viz iteration."
3. Read coordinator scratchpad at `.claude/scratchpad/coordinator/2026-04-18-golden-standing-creek.md` for prior-context-window state.
4. Narrow `claimed_paths` to: `.claude/**`, `reports/**`, `specs/**`, `scripts/archive/**` (read-only audit of others).
5. Confirm to the user: "Plan loaded. Starting Wave 1 — surface audit (six parallel specialists)."

### Wave 1 — Surface audit (six parallel specialists, all foreground)

**Hard rule: NEVER dispatch `general-purpose`.** If a task doesn't fit, handle coordinator-direct or pick the closest specialist as a stretch. Use the specialists below ONLY.

| Agent | Themes covered | Audit task (write to that agent's scratchpad) |
|---|---|---|
| `librarian` | G | Audit `codebase_graph.md` (3228 lines): identify rot (renamed files, deleted modules, stale refs), propose structural rework (split into thin TOC + per-stage child files? evict changelog?). Output: list of drift items + proposed new structure (do NOT yet implement). |
| `qaqc` | A, D | Audit `scripts/` top-level + `scripts/archive/` (53 files / 12 subdirs) + `scripts/one_off/` (4 items, shelf life). Output: bloat inventory, "what should be deleted vs archived vs indexed," shelf-life violations. |
| `spec-writer` | C | Audit `specs/` (17 files): which are still load-bearing, which are superseded, which overlap (esp. the six Claude-infra ones). Output: per-spec status table (active/superseded/redundant) + proposed consolidation. |
| `stage3-analyst` | E, F | Audit `data/.../stage3_analysis/` layout drift + `reports/` ↔ `reports/figures/` coupling. Cross-reference each report's figure refs against actual files on disk. Output: layout-drift map + dead-figure-ref list + reproducibility recommendations. |
| `devops` | A, B | Audit repo-root clutter (`nul`, `img.png`, 8+ logs, `results [old 2024]/`, `lightning_logs/`, `wandb/`, `checkpoints/`, `keys/`, `cache/`, `files/`) + `.gitignore` gaps. Catalog wandb runs / lightning_logs versions / checkpoint files — what experiment produced what. Output: hygiene action list + experiment ledger gaps. |
| `ego` | H + session-state audit | (a) Read `niche/SKILL.md`, `valuate/SKILL.md`, `multi-agent-protocol.md`, `coordinator-coordination.md`, and the last 7 days of coordinator + ego + librarian scratchpads. Assess where the OODA loop drops state across context-window boundaries. (b) Audit the session-state machinery (the gyroscope's current-reading layer): inspect `.claude/supra/sessions/`, `.claude/coordinators/sessions/`, `.claude/coordinators/supra/`, `.claude/coordinators/messages/`, `.claude/scratchpad/valuate/` — are they being written correctly? Are stale entries being archived per the documented thresholds (30 min stale, 2 hr ended, 7 day archived)? Does the niche skill actually read the supra reference frame at every wave, or just at session start? Output: failure modes + draft Markov-completeness contract for each of the five ORIENT artifacts. |

**Each Wave 1 agent must write a Markov-complete scratchpad entry** — one that a fresh context window could resume from without reading anything else. Length is not the cost; loss-of-state is. Override the ≤30-line rule if necessary, and SAY SO at the top of the entry: `<!-- OVERRIDE: Markov-completeness requires ~N lines -->`.

### Wave 2 — Synthesis (delegated to `spec-writer`)

Per the "ACT = subagent work; coordinator delegates" principle: the synthesis is delegated, not coordinator-written. The audit report is essentially a meta-spec — it specifies the contracts for the five ORIENT artifacts — which puts it squarely in `spec-writer`'s domain.

Coordinator's job in Wave 2: brief `spec-writer` with pointers to all six Wave 1 scratchpads + this plan + the supra session file. Then read the result and decide whether it satisfies the spec the plan asked for.

`spec-writer` writes the unified audit report at:

```
reports/2026-04-18-organizational-flywheel-audit.md
```

**Report structure** (mandatory sections):

1. **TL;DR** — one paragraph on the flywheel framing + the top 3 highest-leverage recommendations.
2. **Four-surface model** — the OBSERVE/ORIENT/DECIDE/ACT artifact mapping.
3. **Per-theme findings (A–H)** — for each theme: current state, whether it's a TERRITORY problem (OBSERVE) or a MAP problem (ORIENT), which artifact(s) it damages, recommended fix, priority (P0/P1/P2).
4. **Per-artifact Markov-completeness contracts** — for each of the five ORIENT artifacts (scratchpads, codebase_graph, specs, plans, session states), a written contract specifying what the artifact must guarantee for the next context window. (This is the deliverable's load-bearing core.) Each contract should answer: what must this artifact contain for a fresh context window to resume cold? What invariants must hold across writes? What enforcement mechanism keeps it from rotting?
5. **Implementation roadmap** — ordered list of remediation actions, each with: target surface, expected effort, expected payoff, who owns it (which agent type or human).
6. **Anti-patterns observed** — what the audit found we keep doing that we shouldn't.
7. **Open questions for human** — anything the audit can't decide unilaterally.

The report is the **artifact that survives this context window**. It is the DECIDE substrate's output for this session. Future sessions consume it via `specs/` (after the human approves consolidation into a spec).

### Wave 3 — Self-audit closeout

Dispatch `ego` once more with the brief: "Read `reports/2026-04-18-organizational-flywheel-audit.md`. Apply the Markov-completeness contracts the report itself proposes — does the report satisfy them? Could a fresh context window in a month resume this work from the report alone? Write findings to your scratchpad."

This is the loop closing on itself. If the report fails its own contract, we know the contract isn't quite right.

### Wave Final — Handoff

1. Coordinator writes a final scratchpad entry to `.claude/scratchpad/coordinator/2026-04-18-golden-standing-creek.md`. **Markov-complete handoff packet** — see checklist below.
2. Dispatch `devops` to commit. Suggested message: `audit: organizational flywheel — eight themes, four-surface contracts`.
3. Print to user: report path + commit hash + one-paragraph summary.

---

## Markov-completeness handoff checklist (for the final scratchpad)

The final coordinator scratchpad entry MUST contain (override ≤30-line rule):

- [ ] **What was decided** — the audit's top-3 recommendations, with reasoning.
- [ ] **What was acted on** — files created, commits made.
- [ ] **What remains open** — `[open]` items the next session must address, with enough context that the next session doesn't have to reread anything.
- [ ] **Current state of the five ORIENT artifacts** — one line each: scratchpads, codebase_graph, specs, plans, session states.
- [ ] **Pointer to the report** — `reports/2026-04-18-organizational-flywheel-audit.md`.
- [ ] **Pointer to the supra valuation** — what mode/dims/intent were active.
- [ ] **One-paragraph "if you only read this entry, here's the gist"** at the bottom.

---

## Hard rules for execution

1. **NEVER dispatch `general-purpose`.** Memory `feedback_never_general_purpose.md` is reaffirmed. Coordinator IS general-purpose. If no specialist fits, handle direct or pick closest stretch fit.
2. **NEVER write code today.** Audit-only. The output is the report + scratchpads. No edits to `stage*_*` or `scripts/` (other than reading).
3. **All Wave 1 agents foreground**, one parallel batch.
4. **Coordinator delegates Wave 2 to `spec-writer`** — synthesis is ACT, and ACT is delegated. The coordinator briefs and reads back, never writes the report directly.
5. **Override the ≤30-line scratchpad rule when Markov-completeness requires it**, and say so at the top of the entry.
6. **Do not get pulled into "fix it now" loops.** If during the audit you find a tempting fix, write it to the report's Implementation Roadmap and move on. Today's deliverable is the report.

---

## Quick reference — paths

- Plan: `.claude/plans/2026-04-18-organizational-flywheel-audit.md` (this file)
- Coordinator scratchpad: `.claude/scratchpad/coordinator/2026-04-18-golden-standing-creek.md`
- Supra session: `.claude/supra/sessions/golden-standing-creek-2026-04-18.{ppid}.yaml`
- Codebase graph (subject of audit): `.claude/scratchpad/librarian/codebase_graph.md`
- Niche skill (subject of theme H): `.claude/skills/niche/SKILL.md`
- Multi-agent protocol (subject of theme H): `.claude/rules/multi-agent-protocol.md`
- Audit report (output): `reports/2026-04-18-organizational-flywheel-audit.md`

---

## Suggested invocation after `/clear`

```
/niche follow plan .claude/plans/2026-04-18-organizational-flywheel-audit.md
```

The niche skill's Wave 0 will detect the plan reference, read it, and follow the wave structure above without re-deciding it.
