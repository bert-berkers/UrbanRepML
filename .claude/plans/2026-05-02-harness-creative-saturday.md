# Harness Creative — Saturday Evening (2026-05-02)

| Field | Value |
|---|---|
| **Status** | SHIPPED YELLOW 22:35 — tapestry [A + C + ego-Adjust1 + ego-Adjust2 + D-lite] landed across 4 waves; ego flagged self-validation drift + stigmergic-backfill as YELLOW tensions, both self-corrected. See `.claude/scratchpad/coordinator/2026-05-02-swift-waving-kelp.md` for close-out. |
| **Shard** | `swift-waving-kelp` (PID 27744) |
| **Cluster** | harness — `.claude/` infrastructure (skills, hooks, rules, supra, scratchpads, plans) |
| **Companion shard (other terminal)** | `muted-sliding-dune` — pre-baked, unclaimed; user will spawn + bind. Owns plotting code. |
| **Trigger** | User: "back for the weekend, a little tired but hungry for high-openness creative fun. lets do some work on... the harnass .claude folder" |
| **Est** | 1–3h depending on thread (deliberately open-ended) |

## Reference frame (echoed from supra session for cold-start resume)

```
mode=creative
speed=3 explore=5 quality=3 tests=2 spatial=2 model=2 urgency=3 data_eng=3
intent="Creative work on the .claude/ harness — explore improvements to skills, hooks, rules, supra system. High openness; weekend evening pace."
focus=[harness .claude/ creative improvements]
suppress=[plotting code (separate terminal claims that)]
```

If `/clear` lands you here cold: the supra yaml is `swift-waving-kelp-2026-05-02.yaml`. The valuate scratchpad is `.claude/scratchpad/valuate/2026-05-02.md`. The other terminal (`muted-sliding-dune`) handles plotting — do not touch `scripts/`, `utils/visualization.py`, or anything plot-related from this terminal.

## Frame — why a kapstok and not a fixed-target plan

The user named the shape: this session **configures the gyroscope and builds the vehicle**, where:
- **Gyroscope axis** = characteristic state (already set by `/valuate`)
- **Vehicle** = a plan kapstok that gives `/niche` a structural framework — somewhere to hang work

`high explore` + `creative` mode means the goal isn't to execute a known backlog. It's to **find the spark** within the harness domain, then crystallize it into a wave structure. So this plan does NOT prescribe one task — it lists candidate threads, gives a decision rule, and lets W0 pick.

The session is a **rate-enabler for the harness layer**: every thread below increases the harness's ability to support the *outer loop* (multi-week / multi-shard work) without becoming a maintenance burden. Distilling well: pick the thread whose payoff is most about enabling future rate, not about clearing today's queue.

## Candidate threads (the kapstok hooks)

Numbered for stable references. /niche W0 picks one; ranking below is the coordinator's read of resonance with today's mood and meta-framing.

### Thread A — `/valuate` writes a plan kapstok (★ most resonant)

**Why now**: The user just *named this in chat*. The realization that `/valuate` should produce both characteristic state AND a starter plan ("the framework, the structure, the 'kapstok'") is itself an artifact worth crystallizing into the harness. **Self-referential bootstrap**: the very file you're reading is a hand-rolled prototype of what the codified version of `/valuate` could output automatically.

**Scope**:
- Extend `.claude/skills/valuate/SKILL.md` Step 5.5 (currently writes valuate scratchpad) with an additional "Step 5.6 — write plan kapstok"
- Conditions: write `.claude/plans/{date}-{intent-slug}.md` only if (a) intent is non-empty AND (b) no plan with that intent-slug already exists today
- Format: ref-frame block + frame paragraph + candidate threads (W0 ideation seeds, mostly empty by default — `/niche` fills these on first OODA observe) + carry-items from forward-look + anti-scope
- Could be a helper module `.claude/skills/valuate/plan_kapstok.py` invoked from SKILL.md
- `/niche` Wave-0 protocol then prioritizes "is there a plan-kapstok in `.claude/plans/{today}-*.md`?" — if yes, use it as the Markov-complete handoff; if no, fall back to ad-hoc
- **Spec reuse**: this plan IS the format spec — codify what worked here

**Acceptance**:
- Next time the user runs `/valuate <free-form description>`, a kapstok plan auto-writes
- `/niche` Wave-0 reads the kapstok if present
- Old hand-rolled plans (cluster2, session-identity) still parse (don't break the schema)

**Estimated waves**: W0 audit current /valuate → W1 helper module + SKILL.md edit → W2 /niche W0 protocol update → W3 dogfood test (run it on a fake new-session). ~2h.

### Thread B — `/spawn-shard <name> <intent>` slash command

**Why now**: We just hand-rolled the cross-shard pre-bake (`muted-sliding-dune`) — generated poetic name, wrote the supra yaml, drafted the copy-paste. That's reusable. Promote to a slash command so future weekend "two-terminal" sessions don't need ad-hoc pasting.

**Scope**:
- New skill `.claude/skills/spawn-shard/SKILL.md`
- Args: `<intent-description>` (poetic name auto-generated via `coordinator_registry.generate_session_id`)
- Outputs: pre-baked supra yaml + valuate scratchpad cross-reference + the copy-paste block + a coordinator info-message to `all`
- Acceptance: running `/spawn-shard "improve plotting code"` produces a complete handoff package

**Estimated waves**: W0 spec → W1 skill file + helper → W2 dogfood. ~1.5h.

### Thread C — Hook-filename-drift fix (palate-cleanser, P0 from forward-look)

**Why now**: 5 agents across 2 sessions confirmed it. Small (~10-line edit). 5d aged — about to hit the [stale|14d] threshold next week. Coordinator-direct or `[→devops]`. Clean win in <30 min.

**Scope**: `.claude/hooks/subagent-stop.py` `markov_check` glob — extend to accept `{date}-{session_id}.md` alongside `{date}.md`. Eliminates the double-write hook-compat mirror files going forward.

**Why this thread might come AFTER A or B**: it's not creative — it's discipline. But it's a satisfying warm-up if the user wants something concrete first.

**Estimated waves**: 1 wave, ~15 min. Coordinator-direct edit.

### Thread D — Supra schema polish: nested compound states + harness-creative as a state

**Why now**: We just used `compound_states: creative-prototyping` adjacent to today's mood but it's not quite right (speed=4 not 3, no quality bias toward harness work). Adding `harness-creative` as an explicit compound state would be self-documenting for future Saturday evenings. Could also nest compound states (`harness-creative` extends `creative-prototyping` with overrides).

**Scope**: schema.yaml schema bump to support `extends:` in compound states + add `harness-creative` as the first user. Update supra_reader.detect_compound_state to handle nesting.

**Estimated waves**: ~1.5h.

### Thread E — Plan staleness + carry-forward visualization

**Why now**: Forward-looks at 8d are stale. There's no signal in the inread that says "this forward-look is 8d old — confidence reduced." The /valuate inread could grow a `[stale: 8d]` annotation per item.

**Scope**: extend `/valuate` morning-inread builder to compute age of forward-look + ego entries + saved profiles, and label staleness explicitly.

**Estimated waves**: ~1h.

### Thread F — Wild card: invent something new

**Why now**: high explore. Sometimes the best harness improvement is one I haven't seen yet. /niche W0 could spend 10 minutes brainstorming threads not on this list before picking.

## Decision rule for /niche W0

W0 ritual (~5–10 min, coordinator-direct, no specialist dispatch):

1. Surface this menu to the user. State the coordinator's read: **Thread A (★) is most resonant** because the user explicitly named the kapstok pattern in chat just before /clear. But A → B → others is also a strong sequence (A codifies, then B is the immediate dogfood test).
2. Ask the user: "Pick a thread (A–F), pick a sequence, or describe a wildcard."
3. Whichever wins, freeze its contract in W1 (spec-writer if it touches schema; coordinator if it's pure SKILL.md edit).
4. Update this plan's "Status" line to `IN PROGRESS — chose Thread X` and rewrite the wave structure below as concrete W1/W2/... waves.

## Anti-scope (do NOT do these this session)

- ❌ Plotting code — that's `muted-sliding-dune`'s domain. We suppressed it intentionally.
- ❌ Stage1/2/3 ML pipeline work — neither shard is touching ML this evening.
- ❌ Memory landings unrelated to today's threads (the `feedback_valuate_sets_state_only.md` from forward-look is already landed; check `MEMORY.md`).
- ❌ Multi-day refactor commitments — Saturday evening, user is "a little tired", scope to one-evening-shippable.
- ❌ Heavy specialist dispatch chains — creative mode means the coordinator stays in the loop. Use specialists when contract is clear, not as exploration tools.

## Carry-items (open from forward-look, deferrable)

- `[open|5d]` Hook-filename-drift fix → covered by Thread C above
- `[open|0d]` `specs/artifact_provenance.md` §Fail-mode 3 errata → spec-writer wave, ~30 min, can ride alongside any thread
- `[open|0d]` Session-identity-hardening W1+W2 → fresh-context only; not for a tired Saturday evening
- `[stale|2d]` Probe→viz `run_id` handshake → not harness-domain, defer
- `[deferred:future-plan]` ~60-site figure-provenance backfill → not harness-domain, defer

## Wave structure (W0 only — fills in after thread is chosen)

### W0 — Thread selection (coordinator-direct, ~10 min)

- Read this plan into `/niche` Wave-0 context (the SubagentStart hook injection covers this since `/niche` reads `plans/` directory).
- Surface the threads above to the user with the coordinator's read.
- User picks; rewrite this section with the chosen thread's W1/W2/...

**Acceptance for W0**: a thread is chosen AND this plan's Status line + Wave structure are updated to reflect the choice. Then proceed.

### W1+ — TBD after W0

These sections will be appended once the thread is chosen. The shape will mirror existing plans (cluster2-ledger-sidecars or session-identity-hardening) — wave purpose, owner, scope bullets, acceptance.

## Peer-terminal pointer

- `swift-waving-kelp` (this terminal, PID 27744) — harness `.claude/`
- `muted-sliding-dune` — UNCLAIMED. Pre-baked supra at `.claude/supra/sessions/muted-sliding-dune-2026-05-02.yaml`. User will spawn second PyCharm terminal + paste the bind command. Owns plotting code.

## If you only read this section

This plan is a kapstok — a structural framework /niche hangs work on, written by /valuate to crystallize today's characteristic state. State is creative + high explore + harness focus + plotting suppressed. Six candidate threads (A–F); A (codify /valuate-writes-plan-kapstok) is most resonant because the user just named this pattern in chat. Decision rule: W0 surfaces the menu, user picks, we then rewrite W1+ for the chosen thread. Companion terminal `muted-sliding-dune` handles plotting — do not cross domains. Carry-items from 8d-stale forward-look are palate-cleansers, not the main thread.
