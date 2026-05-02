# `/valuate` writes a plan kapstok

## Status: Frozen (2026-05-02, ready for W2 implementation)

## Context

A kapstok (Dutch: coatrack) is a structural framework with hooks to hang work on. Today's `/valuate` invocation by `swift-waving-kelp` produced two artifacts: the supra session yaml (characteristic state) and a hand-rolled markdown plan at `.claude/plans/2026-05-02-harness-creative-saturday.md`. The plan acted as a *teleological scaffold* for the upcoming `/niche` invocation — it crystallized the static state into a structural blueprint downstream `/niche` waves could hang work on.

This pattern works. The hand-rolled prototype is the format that worked under real conditions tonight, on a creative high-explore evening with a peer terminal claimant. Codifying it as an automatic step of `/valuate` (Step 5.6, after Step 5.5 writes the valuate scratchpad) removes the "hand-roll the kapstok every time" tax and preserves the pattern across sessions.

The pattern is self-referential: the very plan being formalized here describes its own formalization as Thread A.

## Decision

Add Step 5.6 to `/valuate` (after Step 5.5 writes the valuate scratchpad). Step 5.6 calls a helper `.claude/skills/valuate/plan_kapstok.py` that produces a markdown file at `.claude/plans/{date}-{intent-slug}.md` if trigger conditions are met. `/niche` Wave 0 then formalizes a discovery rule for kapstok files matching the active intent.

The kapstok is a *seed* not a *contract*: `/niche` may deviate per the existing Wave-deviation policy and is expected to overwrite the W1+ wave structure once a thread is chosen. The kapstok's persistent value is the frame it crystallizes, not the wave skeleton it bootstraps.

## Trigger conditions

`/valuate` writes a kapstok when **all** of:

1. **Intent is non-empty** — `intent` field in the supra session yaml has a string after Step 5 applies changes. Empty/null intent → no kapstok (a session with no strategic intent has no scaffold to write).
2. **No matching plan exists today** — `glob('.claude/plans/{today}-*-{intent-slug}.md')` returns zero hits. The intent-slug is computed from the intent field via `slugify(intent[:80])` (lowercase, alphanumerics + hyphens, max 8 hyphenated tokens). Idempotent across re-`/valuate` within the same calendar day with the same intent.
3. **No opt-out flag** — the `/valuate` invocation did not include `no-kapstok` or `skip-plan` shorthand. These are silent: they apply only to this call, not persisted.

### Distinguishing "intent set this invocation" vs "intent persisted from earlier"

Both cases trigger a kapstok write — *unless* condition (2) fires. The helper does NOT detect "intent unchanged since last write" by reading the previous yaml; it relies on the filesystem (the existing kapstok file) as the idempotency truth. This is intentional: if the user deletes the kapstok file mid-session, re-`/valuate` recreates it.

If the intent changed (slug differs), a new kapstok writes alongside the old one — same calendar day, two kapstoks, two slugs. Old kapstok is preserved as historical record (per "Never delete data results" feedback principle, plans are not data results but the same instinct applies — they are governance crystallization in static mode).

## Format schema

The hand-rolled prototype IS the format. Codify section ordering and mandatory/optional discrimination.

### Section ordering (top to bottom)

1. **Title heading** — `# {Title} — {Date Tag}` (e.g. `# Harness Creative — Saturday Evening (2026-05-02)`). Title derives from intent-slug rendered into title case with the temporal segment appended.
2. **Status table** — markdown table with `Status`, `Shard`, `Cluster`, `Trigger`, `Est` rows minimum; optional `Companion shard`, `Source`, `Parent plan`, `Depends on`, `Progress` rows where applicable.
3. **Reference frame block** — fenced code block echoing the supra yaml content. Then a one-paragraph "if `/clear` lands you here cold" pointer naming the supra yaml path, the valuate scratchpad path, and any peer-shard suppress contract.
4. **Frame paragraph** — 1-3 paragraphs answering "why this kapstok and not a fixed-target plan." States the shape (single-thread vs multi-thread), names the gyroscopic rate-enablement angle if applicable, and surfaces the meta-framing.
5. **Candidate threads** (multi-thread mode only) — numbered/lettered threads (A–F or 1–6), each with `Why now`, `Scope`, `Acceptance`, `Estimated waves`. Star (★) the most resonant per coordinator's read.
6. **Wave structure** (single-thread mode) OR **Decision rule for /niche W0** (multi-thread mode) — concrete next-step protocol.
7. **Anti-scope** — bulleted list with `❌` markers and short rationale. Hard "do NOT do these" boundary.
8. **Carry-items** (only if forward-look has them) — bulleted list of deferrable items from forward-look, tagged with age (`[open|Nd]`, `[stale|Nd]`).
9. **Peer-terminal pointer** — list of other shards (active or pre-baked), with their domain claims.
10. **"If you only read this section" gist** — single paragraph summarizing items 2–9. This is the Markov-complete summary at the bottom.

### Mandatory vs optional sections

| Section | Mandatory | Notes |
|---|---|---|
| Title heading | Yes | |
| Status table | Yes | At minimum: Status, Shard, Cluster, Trigger, Est rows |
| Reference frame block | Yes | Cold-start pointer paragraph included |
| Frame paragraph | Yes | |
| Candidate threads | Conditional | Multi-thread mode only |
| Wave structure | Conditional | Single-thread mode only |
| Decision rule for /niche W0 | Conditional | Multi-thread mode only |
| Anti-scope | Yes | At minimum 2 entries (one suppress directive + one scope-creep guard) |
| Carry-items | Optional | Only if forward-look has items the valuate scratchpad surfaced |
| Peer-terminal pointer | Yes | "Peers: none" is acceptable but must be stated |
| Gist paragraph | Yes | Heading must match `^## If you only read` |

The gist heading regex (`^## If you only read`) is load-bearing — `/niche` Wave 0 reads this section first as a context-budget optimization (~150 words to orient before deciding whether to read the rest).

## Two modes — single-thread vs multi-thread

The hand-rolled prototype is a multi-thread kapstok (six candidate threads, decision rule at W0). The cluster-2-ledger-sidecars and session-identity-hardening plans are single-thread kapstoks (one wave structure, defined acceptance per wave). Both are valid kapstoks at different points on the explore/exploit gradient.

### Decision heuristic (computed by helper, not user-facing flag)

```
multi-thread if:
  exploration_vs_exploitation >= 4
  AND mode in ['creative', 'exploratory']
single-thread otherwise
```

Rationale: high explore + creative/exploratory mode means the goal is to *find the spark*, not execute a known backlog. The kapstok lists candidates; W0 picks. Low explore or focused/sprint mode means the intent is concrete enough to prescribe one wave structure; the kapstok scaffolds the waves directly.

The heuristic is conservative — it errs toward single-thread (the simpler kapstok) when in doubt. A user who wants a multi-thread kapstok in a focused session can add candidate threads coordinator-direct after the helper writes the single-thread skeleton; the helper does NOT re-run.

### Multi-thread kapstok specifics

- 3–6 candidate threads, lettered A through F.
- Each thread: bold title, `Why now`, `Scope`, `Acceptance`, `Estimated waves` subsections.
- One thread starred (`★ most resonant`) per the helper's read of valuate-scratchpad context + carry-items recency.
- A wildcard thread ("invent something new") is included when explore=5 — surfaces brainstorm space.
- Decision rule: `/niche` W0 surfaces menu, user picks, W1+ rewritten for chosen thread.

### Single-thread kapstok specifics

- One `## Wave structure` section with W0, W1, W2... blocks.
- W0 = "audit + spec-freeze" or equivalent reading-and-orienting wave (almost always coordinator-direct or spec-writer).
- Subsequent waves: owner agent type, scope bullets, acceptance criteria.
- Final Wave block names the close-out items (scratchpad, /librarian-update, /ego-check, peer messages).
- No candidate-thread menu, no decision rule section.

### Where thread content comes from (multi-thread mode)

This is where the helper exercises judgment. The hand-rolled prototype's threads came from:
- The user's chat context just before /valuate (Thread A: "the user named the kapstok pattern in chat")
- The valuate scratchpad's carry-items index (Thread C: "Hook-filename-drift fix, P0 from forward-look")
- The intent + focus directives (Thread B: spawn-shard, derived from "weekend two-terminal" pattern)
- Adjacent supra-schema observations (Thread D: harness-creative as compound state)
- Forward-look staleness signals (Thread E: plan staleness viz)
- A wildcard slot (Thread F: high-explore default)

Codifying *where thread ideas come from* is W2's job, not W1's spec. Open question (see §Open questions). For W2 implementation: the helper may write a placeholder threads list with TODO markers if it cannot infer concrete threads from the available inputs — the human can fill in coordinator-direct before /clear.

## Helper module signature

`.claude/skills/valuate/plan_kapstok.py`:

```python
from pathlib import Path

def write_kapstok(
    supra_session_yaml: Path,
    valuate_scratchpad: Path,
    plans_dir: Path,
    date: str,
    *,
    force: bool = False,  # bypass idempotency check
) -> Path | None:
    """Write a plan kapstok if conditions are met.

    Reads:
      - supra_session_yaml: for intent, dimensions, mode, focus, suppress
      - valuate_scratchpad: for the carry-items context (already distilled
        from forward-look during the /valuate Morning Inread step)

    Does NOT read forward-look files directly. Relies on the valuate
    scratchpad having distilled the relevant carry-items.

    Returns:
      - Path to the written kapstok if written
      - None if trigger conditions not met (intent empty, file exists,
        opt-out flag, etc.)

    Idempotency:
      - Glob plans_dir for `{date}-*-{intent-slug}.md` — if any match, return
        None (don't overwrite). force=True bypasses.

    Mode selection:
      - Computes single-thread vs multi-thread from supra dimensions.

    Output filename:
      - `{date}-{intent-slug}.md` where intent-slug = slugify(intent[:80])
      - If a kapstok with the slug already exists for a different intent
        (rare, but possible if intent-slugs collide), append a 2-char hash
        of the full intent: `{date}-{intent-slug}-{hash}.md`.
    """
```

### Return contract

- **Returns `Path`**: kapstok was written. Caller (SKILL.md Step 5.6) prints the path so the user sees what was scaffolded.
- **Returns `None`**: kapstok was NOT written. Caller does not print anything (silent skip — don't add noise to the /valuate summary).

### Failure modes

- **Supra yaml unreadable** → log to stderr, return None. /valuate Step 5.6 should not fail just because plan_kapstok failed (fail-open at the helper level; the supra yaml + valuate scratchpad are the load-bearing artifacts, kapstok is bonus scaffolding).
- **plans_dir not writable** → log to stderr, return None.
- **slugify yields empty string** (e.g. intent is all punctuation) → log to stderr, return None. Don't write `{date}-.md`.

## /niche Wave-0 protocol update

Add to `.claude/skills/niche/SKILL.md` Wave 0 section, between step 4 (plan discovery) and step 5 (follow plan exactly). The current step 4 already references plan files; this update formalizes kapstok-specific glob and prioritization.

### Wording for SKILL.md (W2 will land this)

> **Step 4.5 — Kapstok discovery**: If `$ARGUMENTS` did not reference a plan file, glob `.claude/plans/{today}-*.md`:
> - **Zero matches**: fall back to ad-hoc Wave-0 with no plan. Continue to step 5.
> - **Exactly one match**: treat as primary blueprint. Continue to step 5.
> - **Multiple matches**: prefer the one whose intent-slug matches the active supra intent (computed via `slugify(supra_intent[:80])`). If still ambiguous, ask the user "I found {N} kapstoks for today: [list]. Which is primary?". Do not silently pick.
>
> Kapstok plans are *seeds*, not contracts. Wave deviation policy applies: if you deviate from a kapstok's wave structure, log rationale before acting (per the existing wave-deviation policy in step 5).

### Why this lives in /niche, not /valuate

The kapstok is written by /valuate but consumed by /niche. The discovery rule is /niche's responsibility (it owns Wave 0). Splitting the contract this way preserves the shard model: /valuate sets static state, /niche executes dynamically with that state as input.

## What kapstok does NOT do

Critical to keep the contract small:

- **Not a subagent dispatch.** The helper runs in /valuate's main context, writes a file, returns. No Task tool, no agent spawning.
- **Not a wave executor.** /niche executes waves; the kapstok scaffolds them.
- **Not authoritative for /niche.** Kapstok is a seed; /niche may deviate with rationale per Wave-deviation policy. The wave structure in a multi-thread kapstok is explicitly placeholder ("W0 only — fills in after thread is chosen").
- **Not generated retroactively.** Only at /valuate-time, not at /niche-time. If /niche starts and finds no kapstok, that is fine — fall back to ad-hoc Wave 0. Do NOT have /niche call back into the kapstok writer.
- **Not edited by the helper after first write.** Subsequent edits are coordinator-direct. The plan file is a living document during /niche; the helper writes only the seed.
- **Not provenance.** Plans are not artifacts of training/probe runs. Do NOT integrate with `SidecarWriter` or the runs ledger. (See Open questions §3.)
- **Not deletable.** Plans are preserved as historical record per the existing plan lifecycle policy. Re-/valuate with a different intent writes a new kapstok alongside the old one.

## Acceptance criteria

### Replay test

After implementation, replay today's setup against the implemented helper:
1. Read the supra yaml at `.claude/supra/sessions/swift-waving-kelp-2026-05-02.{ppid}.yaml` (or equivalent).
2. Read the valuate scratchpad at `.claude/scratchpad/valuate/2026-05-02.md`.
3. Call `write_kapstok(...)` with these inputs and `force=True` (the actual file already exists).
4. Compare the regenerated kapstok against `.claude/plans/2026-05-02-harness-creative-saturday.md`.

### Format-fidelity checklist

The replay must produce a file that:

- [x] Has all 10 mandatory section types in order (title heading → status table → ref frame → frame para → candidate threads → decision rule → anti-scope → carry-items → peer pointer → gist).
- [x] Has the gist paragraph heading matching `^## If you only read`.
- [x] Has reference frame block with all 8 supra dimensions named.
- [x] Has anti-scope with at least 2 entries (one suppress directive — "plotting code" — + one scope-creep guard).
- [x] Has at least one carry-item from forward-look (today's prototype has 5; replay must have ≥1 if the valuate scratchpad surfaced any).
- [x] Detects multi-thread mode (explore=5, mode=creative → multi-thread).
- [x] Writes 3–6 candidate threads, one starred.
- [x] Filename matches `{date}-{intent-slug}.md` pattern.

### Format-fidelity tolerances

The replay is allowed to differ from the hand-rolled prototype on:

- **Candidate-thread ranking** (which thread gets ★) — judgment call. Test passes if ★ is assigned to *some* thread.
- **Exact prose phrasing** — frame paragraph, gist paragraph, thread descriptions may differ word-for-word. Test passes if the *structure* is intact.
- **Number of candidate threads** (3 vs 6) — both are within range. Today's prototype has 6; helper may write fewer if it cannot infer 6 distinct threads.
- **Wildcard thread inclusion** (Thread F equivalent) — included when explore=5; not required at explore=4.
- **Title casing** — "Harness Creative — Saturday Evening" vs "Harness Creative Saturday" both acceptable.

### Hard failures

- Missing any mandatory section → fail.
- No gist paragraph or wrong heading → fail.
- Single-thread mode for an explore=5 + creative session → fail (heuristic violation).
- Reference frame block missing dimensions → fail.

## Open questions

1. **Should kapstok survive re-`/valuate` on the same supra session?** *Spec answer: no rewrite.* Re-`/valuate` with the same intent → idempotent skip (file exists). Re-`/valuate` with a different intent → new kapstok alongside old. Old kapstok is preserved as historical record. Rationale: re-`/valuate` mid-session signals an intent shift, and overwriting the original kapstok would erase the W0 decision context.

2. **Where does candidate-thread *content* come from in multi-thread mode?** *Open — W2's call.* The hand-rolled prototype's threads came from chat context, valuate-scratchpad carry-items, intent + focus directives, schema observations, forward-look staleness, and a wildcard slot. Codifying which inputs to read and how to rank them is W2's design space. Suggested: helper may write placeholder threads with TODO markers and rely on coordinator-direct fill-in before /clear if it cannot infer threads cleanly. The replay test only checks structure (≥3 threads, one starred), not exact thread content.

3. **Should kapstok integrate with SidecarWriter / runs ledger?** *Spec answer: no.* Plans are not provenance artifacts; they are scaffolds. The runs ledger is for training/probe artifacts with `config_hash` and `output_paths`; a plan has neither. Plans live in `.claude/plans/`, not `data/` — they are governance state, not data state.

4. **What if /valuate runs in a session where /niche has already started?** *Open — minor edge case.* The current /valuate flow allows mid-session re-tuning. If /niche is already executing and /valuate writes a new kapstok (different intent), /niche does not auto-pick it up — the user must `/clear` and re-`/niche` to consume the new kapstok. This is consistent with /valuate Step 7's handoff behavior (the human is in the loop). No special case needed.

## Implementation notes (for W2)

### Files touched

- `.claude/skills/valuate/SKILL.md` — add Step 5.6 between Step 5.5 (valuate scratchpad write) and Step 6 (print summary). The new step calls `write_kapstok(...)` and prints the path if returned.
- `.claude/skills/valuate/plan_kapstok.py` — new helper module per signature above.
- `.claude/skills/niche/SKILL.md` — add Step 4.5 to Wave 0 protocol per wording above.

### Dependencies

- Reads `.claude/supra/sessions/{supra_session_id}.{ppid}.yaml` (existing, read-only).
- Reads `.claude/scratchpad/valuate/YYYY-MM-DD.md` (existing, read-only).
- Writes `.claude/plans/{date}-{intent-slug}.md` (new file per invocation).
- No new third-party dependencies. Uses standard library (yaml, pathlib, re for slugify).

### Ordering constraints

1. Step 5.5 (valuate scratchpad) MUST run before Step 5.6 — the helper reads the scratchpad.
2. Step 5.6 MUST be additive — if the helper raises, /valuate continues to Step 6 (print summary). Wrap in try/except, log stderr on failure, return None.
3. /niche Wave 0 Step 4.5 MUST run after Step 4 (plan discovery from $ARGUMENTS) — kapstok discovery is the fallback when $ARGUMENTS does not name a plan.

### Testing

- Unit tests at `.claude/skills/valuate/test_plan_kapstok.py` (new file). Test cases:
  - Empty intent → returns None
  - File already exists → returns None (unless force=True)
  - opt-out flag in invocation context → returns None (W2 designs the flag-passing mechanism — env var, sentinel file, or function arg)
  - explore=5 + mode=creative → multi-thread mode
  - explore=2 + mode=focused → single-thread mode
  - Slugify edge cases (empty, all-punctuation, very long intent)
  - Replay test against today's prototype (golden test, structure-only)

### Out of scope for W2

- Multi-thread *thread content generation* beyond placeholder TODO markers (open question §2).
- Cross-shard kapstok coordination (e.g. peer terminal's kapstok influencing this terminal's threads). Today's prototype handled peer-shard pointers manually; codifying cross-shard kapstok coupling is a future plan.
- Kapstok templating / theming (different visual styles per mode). Single template for now.
- Kapstok-as-skill (a `/kapstok` slash command for manual invocation outside /valuate). Premature; the helper is the only writer.

## Frame summary

The hand-rolled prototype (`.claude/plans/2026-05-02-harness-creative-saturday.md`) IS the format spec — this document codifies what was load-bearing about it. Frozen sections: status table at top, gist at bottom, mandatory section order, multi-thread vs single-thread heuristic from supra dimensions. Open: how multi-thread thread content gets generated (W2 designs that). Acceptance: replay today's setup, compare structure (not prose) against the prototype. The kapstok is a seed not a contract — /niche owns Wave 0 deviation rights. Implementation lives at `.claude/skills/valuate/plan_kapstok.py` with helper signature documented; SKILL.md edits land in W2 to /valuate (Step 5.6) and /niche (Wave 0 Step 4.5).
