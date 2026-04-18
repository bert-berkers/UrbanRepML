# Between-Wave Pause Redesign

## Status: Draft
> Absorption status into `.claude/skills/niche/SKILL.md` is unverified as of 2026-04-18. If the Wave Results block is present in niche, this spec should be re-classified as Implemented; if not, keep as Draft and schedule a wave.

## Context

The OODA report printed between work waves has 8 fields, 3 of which (Blocked, Drifting, Lateral) are "nothing" in >80% of observations. The report is a status snapshot rather than a drift detector -- it does not compare current state against intended trajectory. The human observed this is pure ceremony.

The current format costs ~150-200 tokens per report and requires running `/summarize-scratchpads` (which reads all agent scratchpads). The information-per-token ratio is poor.

### What the scratchpad record shows

Reviewing coordinator scratchpads from 2026-03-01 through 2026-03-08, the between-wave OODA reports never surfaced a real block or drift that was not already obvious from the agent's return value. The fields that carried signal were:
- **State** (what just happened)
- **Needs your call** (human decisions needed)

The fields that were dead weight:
- **Blocked**: always "nothing" between waves because the coordinator would not advance to the next wave if something were blocked
- **Drifting**: always "nothing" because the coordinator is following a plan
- **Lateral**: only relevant during Wave 0 (session start); between waves, other coordinators rarely appear mid-session
- **Mode**: does not change within a session

## Decision

Replace the between-wave OODA report with a shorter **Wave Results** block. Differentiate Wave 0 (orientation) from between-wave pauses (progress check). Drop always-empty fields. Add concrete checks that require looking at real artifacts.

### Wave 0 Report (unchanged purpose, trimmed format)

Wave 0 remains an orientation report. It establishes session context. Trimmed format:

```markdown
## Session: [name]
**Goal**: [restate user task in own words]
**Git**: [clean/dirty, unpushed count]
**Lateral**: [other coordinators, or "solo"]
**Deferred P0s**: [items flagged 2+ sessions from ego forward-look, with count]
**Needs your call**: [decisions requiring human input, or "none"]
```

Changes from current:
- Drop **Mode** (available in supra state, not actionable here)
- Drop **State** (redundant with Goal at session start)
- Drop **Blocked**/**Drifting** (nothing has happened yet)
- Add **Deferred P0s** (concrete check: scan ego forward-look for items carried 2+ sessions)

### Between-Wave Pause (replaces OODA report)

After each work wave returns, print:

```markdown
## Wave N Results
**Delivered**: [1-2 lines: what each agent returned vs what was asked]
**Surprises**: [unexpected findings from agent scratchpads, or "none"]
**Still on goal?**: [yes/no + evidence]
**Needs your call**: [human decisions, or "none -- proceeding to Wave N+1: {description}"]
```

Four fields. Each requires a concrete check:

| Field | What the coordinator actually examines | Why it earns its keep |
|---|---|---|
| **Delivered** | Compare agent output against the acceptance criteria in the delegation prompt | Forces the coordinator to evaluate delivery, not just acknowledge return |
| **Surprises** | Read each returning agent's scratchpad "Unresolved" and "Cross-agent observations" sections | Surfaces findings the agent made but the coordinator did not ask for |
| **Still on goal?** | Re-read the user's original task statement (from $ARGUMENTS or plan file) and compare against what has been accomplished so far | Prevents scope drift by making the comparison explicit |
| **Needs your call** | Aggregate any human-decision-required items from agent scratchpads, plus any from the coordinator's own observation | The only action-forcing field -- everything else is informational |

### What is NOT in the between-wave pause

- **Blocked**: If an agent is blocked, it says so in its return. The coordinator handles it in the delegation for the next wave, not in a status field.
- **Drifting**: Replaced by "Still on goal?" which is a concrete comparison, not a vague assessment.
- **Lateral**: Checked once at Wave 0. Between waves, only surface if a new message arrived (check `messages/` dir; if empty, say nothing).
- **Mode/State/Session**: Established in Wave 0, does not change.
- **Task goal**: Embedded in "Still on goal?" rather than restated every wave.
- **Deferred P0s**: Checked at Wave 0; not re-checked between waves (the ego owns this at session end).
- **`/summarize-scratchpads`**: NOT called between waves. The coordinator reads only the returning agents' scratchpads, not all of them. Full scratchpad summary belongs in Wave 0 only.

### How "Still on goal?" avoids rubber-stamping

The coordinator must perform a two-part check:

1. **Quote the goal**: Pull the user's original task statement (literal text from $ARGUMENTS or the plan file's stated objective).
2. **List what has been accomplished**: Enumerate concrete deliverables from completed waves.
3. **State the gap**: What remains between accomplished and goal. If the gap has grown (scope creep) or shifted (drift), say so.

If the coordinator writes "yes" without the gap statement, it is a protocol violation. The gap statement IS the check -- "Still on goal? Yes -- delivered X and Y, remaining: Z" is substantive. "Still on goal? Yes" is not.

### Gating behavior (unchanged)

- **Plan-driven session**: auto-proceed if "Needs your call: none". The human already approved the wave structure.
- **Ad-hoc session**: always pause for human confirmation, even if "Needs your call: none". The "Continue?" is implicit in the ad-hoc case.

### Relationship to ego's end-of-session assessment

The between-wave pause is a **progress check** (did we do what we said?). The ego assessment is a **process health check** (how well did we work?). They do not overlap:

| Concern | Between-wave pause | Ego assessment |
|---|---|---|
| Did agents deliver? | Yes (Delivered field) | No (ego checks process, not deliverables) |
| Scope drift? | Yes (Still on goal?) | Yes (Drifting section) -- but ego looks across sessions, not within |
| Deferred items? | Wave 0 only | Yes (primary owner of P0/P1/P2 tracking) |
| Agent protocol compliance? | No | Yes (scratchpad discipline, implementer violations) |
| Surprises? | Yes (Surprises field) | Indirectly (cross-agent observations) |

## Alternatives Considered

### 1. Keep current format, just make Blocked/Drifting smarter

Could add concrete checks to the existing fields (e.g., "Blocked: check if any agent returned an error"). But this preserves 8 fields when only 4 carry signal. The format's problem is structural, not just content.

### 2. Remove the pause entirely, rely on agent return values

The coordinator already reads agent outputs. A pause adds context-window cost. But the human needs visibility -- the pause is the human's window into what happened. Removing it makes the session opaque between Wave 0 and Final Wave.

### 3. Add more fields (deferred counter, agent scorecard, git diff summary)

More information is not better information. The ego already tracks deferred items. Git state is checked in Wave 0. Adding fields increases context cost and reading burden for the human.

## Consequences

- **Positive**: Shorter reports (~80 tokens vs ~150-200). Every field requires examining a real artifact. "Still on goal?" with mandatory gap statement prevents rubber-stamping. No `/summarize-scratchpads` between waves saves significant context.
- **Positive**: Wave 0 gains Deferred P0 counter, making session start more actionable.
- **Negative**: Lateral coordination is only checked at Wave 0. If another coordinator sends a message mid-session, it will not be surfaced until the next session (or Final Wave). Mitigation: add a lightweight message-dir check (ls, not read) to between-wave pause if this becomes a problem in practice.
- **Negative**: Removing `/summarize-scratchpads` from between-wave means the coordinator only sees returning agents' state, not the full multi-agent picture. This is intentional (focus on what just happened) but could miss cascading issues.
- **Neutral**: Requires updating the coordinator skill SKILL.md (sections 2 and 5 of the OODA protocol). The change is localized to the report format, not the wave structure.

## Implementation Notes

1. Edit `.claude/skills/coordinate/SKILL.md` section "2. ORIENT -- print the OODA report" to contain the Wave 0 format.
2. Add a new section "5. LOOP -- between-wave results" with the Wave N Results format.
3. Remove `/summarize-scratchpads` from the OBSERVE step for between-wave cycles (keep it in Wave 0 only).
4. Add the "Still on goal?" two-part check protocol (quote goal, list accomplished, state gap).
5. Update the gating policy text to reference "Needs your call" instead of the full OODA report.

Estimated change: ~40 lines in SKILL.md (replace ~30, add ~10).
