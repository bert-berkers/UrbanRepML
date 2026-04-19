# Terminal C — Probe-Confound Ternary (warmstart, 2026-04-19)

| Field | Value |
|---|---|
| **Status** | READY — warmstart for next `/valuate` in an unclaimed terminal |
| **Terminal ID** | C (PyCharm tab name: **C - probe**) |
| **Source** | `.claude/scratchpad/valuate/2026-04-19.md` §Terminal C + `reports/2026-04-18-organizational-flywheel-audit.md` §7 Q8 + `.claude/scratchpad/coordinator/2026-04-18-forward-look.md` §"Carried-forward escalated" |
| **Parent plan** | none (ad-hoc investigation-driven) |
| **Cluster** | empirical — probe/model investigation, not infrastructure |
| **Peer terminals** | A (paths, claimed), B (housekeeping), D (spec-governance) |
| **Est** | 1–2h for the ternary decision alone; full prioritize-path scope is 4–6h if option (a) is chosen |

## The ternary

**Q8 is escalated at 22 days** (first raised 2026-03-29, carried silently across two sessions). Human (via this terminal's work) must pick ONE:

- **(a) Prioritize this weekend** — work through the backlog now. Scope: ~4–6h. Rerun pipeline, clean report, update tables.
- **(b) Defer with explicit target date** — pick a concrete future weekend. Write the defer tag with a date, not a wave-hand. `[deferred:Q8:2026-05-XX]`.
- **(c) Abandon with rationale** — declare the investigation not worth pursuing. Record why. `[wontfix:Q8:reason]`.

**Silent-rolling is contract-forbidden** per the Markov-completeness rule (`.claude/rules/multi-agent-protocol.md`). The coordinator MUST close this out with one of the three tags; `[open|22d]` is no longer acceptable.

## Characteristic-state calibration

```
mode=exploratory speed=2 explore=4 quality=4 tests=3 spatial=3 model=5 urgency=4 data_eng=3
intent="Resolve Q8 probe-confound ternary (escalated 22d) — pick prioritize/defer/abandon and execute"
focus=[Q8-ternary, 74D-UNet, Ridge-vs-DNN, R²-tables, CE-pipeline]
suppress=[paths-consolidation, housekeeping, spec-governance, new-features]
```

Shorthand: `/valuate exploratory, quality 4, explore 4, model 5, urgency 4, focus "Q8 probe-confound ternary"`

## Claimed paths (narrow at first OODA cycle)

- `stage3_analysis/**` (probe code + analysis)
- `reports/**` EXCEPT `reports/2026-03-08-causal-emergence-phase1.md` (that file is D's Q3 scope — narrow at W1)
- Model checkpoints in the study area's `stage2_fusion/` outputs (read-only or version-new-copy)
- `data/study_areas/*/stage2_fusion/` **read-only** (don't overwrite existing checkpoints)

**Explicitly not claimed**: `scripts/**` (A), `.claude/hooks/**` (B), `specs/**` (D), `CLAUDE.md` (D), `reports/2026-03-08-causal-emergence-phase1.md` (D).

## Wave structure

### W1 — Observe (stage3-analyst)

Gather state: re-read the 2026-03-29 scratchpad entry that first raised the confound. List the five concrete open items:

1. **74D UNet file choice** — which checkpoint is canonical for probing? Are there multiple 74D checkpoints and ambiguity about which one the reports reference?
2. **Ridge vs DNN gap** — DNN probe results diverge from Ridge; which is authoritative? Is the divergence signal or noise?
3. **R² table cleaning** — which report has the stale R² table? What's the correct source-of-truth?
4. **Hardcoded `approach_dates`** — where (which probe scripts)? What's the fix path?
5. **CE pipeline rerun** — the 2026-03-29 forward-look said CE pipeline is stale (old 3-mod 781D UNet). Rerunning requires `extract_highway_exits --year 20mix` → probe → CE viz.

Post `info` message to peer terminals: "C (probe) claims stage3_analysis/** and reports/** (except 2026-03-08 file)".

### W2 — Orient & Decide the ternary (stage3-analyst + coordinator)

For each of the 5 items, estimate:
- **Fix effort** (hours): trivial <1h, moderate 1–3h, heavy 3–6h
- **Blocking status**: blocks publication? blocks next experiment? just cosmetic?
- **Confidence of current numbers**: are the cited results probably wrong, probably right, or unknown?

Output: a one-paragraph recommendation to the human:

> "Given {summary of item costs}, I recommend option {a/b/c} because {reason}. If (a), I will {scope}. If (b), target date = {YYYY-MM-DD}. If (c), rationale = {why}."

Then **ask the human** via a direct message in the terminal (not a broadcast). Wait for answer before W3.

### W3a — If option (a) Prioritize

Sub-waves to prioritize in order of cheap wins first:
- W3a.1 — Fix hardcoded `approach_dates` (likely trivial, unblocks repeatability)
- W3a.2 — Disambiguate 74D UNet checkpoint; add checkpoint-versioning per `memory/MEMORY.md` "P2 #7"
- W3a.3 — Rerun CE pipeline (`extract_highway_exits --year 20mix` → probe → CE viz). Likely slow.
- W3a.4 — Resolve Ridge vs DNN divergence; one authoritative comparison table
- W3a.5 — Update R² tables in affected reports; cite the checkpoint + date

### W3b — If option (b) Defer

- Pick explicit target date. Write a one-paragraph "what's needed to execute" block so the future session doesn't re-derive.
- Tag all five open items as `[deferred:Q8.N:YYYY-MM-DD]` with the target.
- Add a Q8 retrospective note: "why now is not the right time".

### W3c — If option (c) Abandon

- Write a rationale paragraph: why the confound no longer matters (did the model change? are the reports being rewritten? is the dataset being swapped?).
- Tag the five items `[wontfix:Q8.N:reason]`.
- Patch the affected reports: either remove the problematic R² numbers, or add a "see Q8 abandonment note" footnote pointing at this plan.

### Final Wave — close-out

- Coordinator scratchpad entry with all 7 Markov-completeness items; Q8 items tagged with their resolution form (`[done|date]`, `[deferred|date]`, or `[wontfix|reason]`).
- `done` message to peer terminals A, B, D summarizing ternary decision.
- If option (a) and any items weren't finished in the weekend, escalate those with new age-0 `[open|0d]` tags — do NOT carry forward the 22-day count (that count applied to the *ternary*, which is now resolved).

## Out of scope

- Q3 dead-figure citations in `2026-03-08-causal-emergence-phase1.md` (D owns)
- Q5/Q6/Q7 spec-governance (D owns; Q7 dissolved)
- Paths consolidation (A)
- Housekeeping + hook dev (B)
- New probe development beyond resolving the ternary

## Execution

```
/clear
/valuate exploratory, quality 4, explore 4, model 5, urgency 4, focus "Q8 probe-confound ternary"
/niche follow plan .claude/plans/2026-04-19-terminal-c-probe-confound.md
```
