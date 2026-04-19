# Terminal D — Spec Hygiene & Governance (warmstart, 2026-04-19)

| Field | Value |
|---|---|
| **Status** | READY — warmstart for next `/valuate` in an unclaimed terminal |
| **Terminal ID** | D (PyCharm tab name: **D - specs**) |
| **Source** | `.claude/scratchpad/valuate/2026-04-19.md` §Terminal D + `reports/2026-04-18-organizational-flywheel-audit.md` §7 Q3/Q5/Q6 |
| **Parent plan** | none (ad-hoc governance close-out) |
| **Cluster** | governance — docs/spec-hygiene, no infrastructure, no empirical work |
| **Peer terminals** | A (paths, claimed), B (housekeeping), C (probe) |
| **Est** | 1–2h |

## Why this plan exists

Audit §7 left four open spec-hygiene questions. Three are assigned here; Q7 (roadmap scheduling) is dissolved — the weekend's four-terminal action is itself the commitment mechanism. Ask again only if items remain unresolved after A+B+C+D close out.

## Characteristic-state calibration

```
mode=focused speed=3 explore=3 quality=5 tests=1 spatial=1 model=1 urgency=3 data_eng=1
intent="Close audit §7 spec-hygiene tail — Q3 forensics + Q5 absorption diff + Q6 canonical pointer"
focus=[Q3-forensics, Q5-absorption, Q6-CLAUDE.md-pointer]
suppress=[paths-consolidation, housekeeping, probe-confound, any-numerical-work]
```

Shorthand: `/valuate focused, quality 5, tests 1, model 1, spatial 1, focus "audit §7 spec tail"`

**Profile note**: `tests=1, spatial=1, model=1, data_eng=1` are intentionally dead dims — nothing numerical is touched. `quality=5` carries all the weight (prose and decisions).

## Claimed paths (narrow at first OODA cycle)

- `reports/2026-03-08-causal-emergence-phase1.md` (Q3 — patch or keep with footnote)
- `specs/between-wave-pause-redesign.md` (Q5 — diff vs niche/SKILL.md, then Historical-vs-Draft call)
- `CLAUDE.md` line 205 only (Q6 — spec pointer update if option (a) is chosen)
- `specs/claude_code_multi_agent_setup.md` (Q6 — preamble rewrite if option (b) is chosen)

**Explicitly not claimed**: `scripts/**` (A), `.claude/hooks/**` (B), `stage3_analysis/**` (C), other `specs/**` files, other `reports/**` files (C owns probe-adjacent ones).

## Wave structure

### W1 — Q3 forensics (spec-writer or librarian)

Dead citations in `reports/2026-03-08-causal-emergence-phase1.md`:
- `multiscale_probe_results.csv`
- `multiscale_comparison.png`
- `multiscale_delta.png`

Investigation steps:
1. `git log -p -- reports/2026-03-08-causal-emergence-phase1.md` — check if figures were ever committed, then deleted.
2. `git log --all --full-history -- '**/multiscale_probe_results.csv'` etc. — check the whole repo history (they might have lived elsewhere).
3. `rg 'multiscale_probe_results|multiscale_comparison|multiscale_delta' .` — any generator script still present?

Three possible outcomes:
- **(α) Never generated** — the report has always had dead refs. Patch the report text: remove the dead citations, or add a "planned but never produced" footnote.
- **(β) Generated and deleted** — `memory/feedback_no_delete_data.md` was violated. **Hand off to Terminal C** for regeneration (probe pipeline work is out of D's scope). Leave a `[→stage3-analyst]` tag + `info` message to C.
- **(γ) Generated but moved/renamed** — find the new names, update citations in the report.

Acceptance: outcome determined; report either patched, handed off, or citations updated.

### W2 — Q5 absorption diff (spec-writer)

`specs/between-wave-pause-redesign.md` may have been absorbed into `.claude/skills/niche/SKILL.md`. Determine:

1. Read `specs/between-wave-pause-redesign.md` — note the key design elements (Wave Results format, pause protocol, etc.).
2. Read `.claude/skills/niche/SKILL.md` — check for matching sections.
3. Produce a diff table: `{element} | in spec? | in SKILL.md? | action`.

Decision: mark the spec Historical (if fully absorbed), keep as Draft (if partially absorbed), or leave as-is + note what's missing.

Acceptance: spec status field explicitly set; any remaining design elements documented for future absorption.

### W3 — Q6 CLAUDE.md canonical spec pointer (spec-writer)

`CLAUDE.md` line 205 currently points at `specs/claude_code_multi_agent_setup.md` (Feb 13, partially stale — pre-PPID, pre-supra, pre-niche rename). Two options:

- **(a) Update the pointer** — change CLAUDE.md line 205 to point at the newer `session-identity-architecture.md` + `temporal-supra-profiles.md` (spec-writer's Wave-1 audit originally recommended this).
- **(b) Update the old spec** — add a "current state" preamble to `claude_code_multi_agent_setup.md` that links forward to the newer specs. Keeps CLAUDE.md pointer stable.

**Decision process**:
1. Read both candidate newer specs to confirm they cover what `claude_code_multi_agent_setup.md` covered.
2. If newer specs are comprehensive → option (a) — a clean pointer redirect.
3. If newer specs are topic-narrow → option (b) — the old spec remains useful as an entry point, with a forward-link preamble.

Acceptance: CLAUDE.md and/or spec updated exactly once; decision recorded in scratchpad with rationale.

### Final Wave — close-out

- Coordinator scratchpad entry with all 7 Markov-completeness items
- Q3/Q5/Q6 tagged with resolution form:
  - Q3: `[done|2026-04-19]` if patched here, or `[→stage3-analyst]` + `[blocked:upstream:terminal-c]` if handed off
  - Q5: `[done|2026-04-19]` with chosen status
  - Q6: `[done|2026-04-19]` with chosen option (a) or (b)
- `done` message to peer terminals A, B, C

## Out of scope

- Paths consolidation (A)
- Housekeeping + hook dev (B)
- Probe-confound Q8 (C)
- Q7 roadmap scheduling (dissolved — see plan header)
- Figure regeneration if Q3 outcome β (handed off to C)

## Execution

```
/clear
/valuate focused, quality 5, tests 1, model 1, spatial 1, focus "audit §7 spec tail"
/niche follow plan .claude/plans/2026-04-19-terminal-d-spec-hygiene.md
```
