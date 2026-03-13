---
name: valuate
description: "Set characteristic states (valuation). Static graph: indicators → percepts ↔ needs/desires (you)."
user-invocable: true
disable-model-invocation: false
context: fork
argument-hint: "[optional: quick adjustment like 'speed 5, tests 1']"
---

## Task

Adjust the human's characteristic states (precision weights) that propagate through the multi-agent system and change how agents behave. This is the mechanism by which the supra layer (human) tunes the attention of the entire cognitive hierarchy.

$ARGUMENTS

## Protocol

### Step 1: Read Current State

Read session ID from `.claude/coordinators/.current_session_id`. Then read states:
1. Try `.claude/supra/sessions/{session_id}.yaml` (session-scoped, takes priority)
2. Fall back to `.claude/supra/characteristic_states.yaml` (global default/prior)
3. Read `.claude/supra/schema.yaml` for dimension definitions, groups, mode biases, agent relevance
4. Determine current temporal segment from local time (e.g., `friday-evening`) using `supra_reader._temporal_segment_key()`
5. Look up that segment in `supra/temporal_priors.yaml` using `supra_reader.get_temporal_prior()`
6. If it exists and has sufficient observations, it becomes the "suggested prior" — available for the morning inread and as a shorthand option

The global file is the prior — it provides defaults for sessions that haven't attuned yet. Your attunement writes session-scoped only, so parallel sessions with different goals don't overwrite each other.

### Step 2: Show Current Landscape

Print a compact summary. Use this exact layout:

```
Current attentional landscape:
Mode: {mode} | Speed: {execution_speed} | Explore: {exploration_vs_exploitation} | Quality: {code_quality} | Tests: {test_coverage} | Spatial: {spatial_correctness} | Model: {model_architecture}
Focus: {focus list or "(none)"} | Suppress: {suppress list or "(none)"}
Last attuned: {timestamp or "never"}
```

If there are dimensions beyond the six core ones (added via recommendations), append them to the second line.

### Step 3: Check for Shorthand Arguments

If the user passed arguments (via `$ARGUMENTS`), parse them as shorthand and **skip the questionnaire**. Shorthand mappings:

| Shorthand | Dimension |
|-----------|-----------|
| `speed` | `execution_speed` |
| `explore` | `exploration_vs_exploitation` |
| `quality` | `code_quality` |
| `tests` | `test_coverage` |
| `spatial` | `spatial_correctness` |
| `model` | `model_architecture` |

Additional shorthand syntax:
- Mode names recognized directly: `exploratory`, `focused`, `sprint`, `creative`
- **Compound state names** recognized directly: `creative-prototyping`, `production-hardening`, `deep-investigation`, `sprint-shipping`, `careful-exploration`, `training-run`
  - A compound state sets specific dimensions to absolute values and optionally sets the mode
  - Compound state dimensions are defined in `schema.yaml` under `compound_states:`
  - Dimensions NOT specified by the compound state are left at their current values
  - Any explicit dimension overrides in the same command take precedence over the compound state's values
  - Example: `/valuate deep-investigation, speed 2` applies deep-investigation then overrides execution_speed to 2
- `focus "..."` sets the focus directive (replaces existing focus list with single item)
- `suppress "..."` sets the suppress directive (replaces existing suppress list with single item)
- Bare dimension names without a value are treated as amplify (set to 4)
- Values are integers 1-5
- `save:name` saves current state as a named profile (after applying any other changes in the same command)
- `load:name` loads a saved profile, replacing mode, dimensions, focus, and suppress
- `profiles` or `list` lists all saved profiles
- **`use prior`**: Loads the temporal prior's rounded values as the starting state. Equivalent to setting each dimension to the temporal prior's rounded value and mode to the prior's majority-vote mode. If no temporal prior exists for the current segment, prints a warning and falls through to the questionnaire.

**Profile operations:**
- Profiles are stored in `.claude/supra/profiles/{name}.yaml`
- `save:` runs AFTER other shorthand changes, so `/valuate sprint, save:sprint-default` first applies sprint mode, then saves the result
- `load:` runs BEFORE other shorthand changes, so `/valuate load:training, speed 5` loads the training profile then overrides speed
- Use `supra_reader.save_profile()`, `supra_reader.load_profile()`, and `supra_reader.list_profiles()` from `.claude/hooks/supra_reader.py`

**Examples:**
- `/valuate speed 5, tests 1` -- sets execution_speed=5, test_coverage=1
- `/valuate sprint` -- sets mode to sprint
- `/valuate focused, model 5, focus "prove hex2vec works"` -- mode=focused, model_architecture=5, focus set
- `/valuate explore` -- ambiguous: could mean mode=exploratory or exploration_vs_exploitation=4. Resolve: if no number follows, check if it matches a mode name first. `explore` is not a mode name, so it maps to the dimension at value 4. `exploratory` IS a mode name.

**Resolution order for bare names** (no number follows):
1. Check compound state names first (e.g., `creative-prototyping` -- hyphenated names are always compound states)
2. Check mode names (`exploratory`, `focused`, `sprint`, `creative`)
3. Check dimension shorthands (`speed`, `explore`, `quality`, `tests`, `spatial`, `model`)
4. Check profile operations (`profiles`, `list`, `save:`, `load:`)
5. Check temporal prior shortcuts (`use prior`, `prior`)
- `/valuate save:creative-evening` -- saves current state as "creative-evening"
- `/valuate load:training` -- restores the "training" profile
- `/valuate sprint, speed 5, tests 1, save:ship-it` -- applies sprint + overrides, saves as "ship-it"
- `/valuate profiles` -- lists all saved profiles
- `/valuate creative-prototyping` -- applies the creative-prototyping compound state (sets explore=5, speed=4, quality=2, tests=1, mode=creative)
- `/valuate deep-investigation, focus "understand hex2vec loss landscape"` -- applies compound state + sets focus
- `/valuate training-run, speed 4` -- applies training-run then overrides execution_speed to 4
- `/valuate use prior` -- loads the learned temporal prior for the current time segment
- `/valuate use prior, speed 5` -- loads temporal prior then overrides execution_speed to 5

If shorthand is provided, apply changes and jump to Step 6 (print summary). Do not ask questions.

### Step 3.5: Morning Inread (when no shorthand AND session appears to be first of the day)

If all of these are true: (a) no shorthand was provided, (b) `last_attuned` is from a previous day or null, and (c) a saved profile exists — this is likely the human's first session of the day. Before jumping to the questionnaire, offer a **morning inread**: a curated reading list to orient the human with their cup of tea.

**Build the reading list** by scanning for the most informative files from the previous session(s):

1. **Coordinator forward-look** (highest priority): `.claude/scratchpad/coordinator/YYYY-MM-DD-forward-look.md` from the most recent date. This is written specifically to seed the next session.
2. **Ego assessment** (high priority): `.claude/scratchpad/ego/YYYY-MM-DD.md` from the most recent date. Process health, attention needed, metrics table.
3. **Recent git log**: `git log --oneline -10` — what actually shipped.
4. **Active coordinator messages**: any unread messages in `.claude/coordinators/messages/` from after last attunement.
5. **Any specialist scratchpads newer than last attunement** — only mention these by name and one-line summary, don't dump content.

**Present the inread** as a compact recommendation:

```
Good morning. Here's your inread before we valuate:

📖 Reading list:
  1. Coordinator forward-look (2026-03-08) — tomorrow's plan, LR schedule fix P0
  2. Ego assessment (2026-03-08) — 4/5 process health, checkpoint versioning P0
  3. git log: 2 commits since last session (rename + liveability graph)
  4. No unread coordinator messages.

Temporal prior for friday-evening (4 observations):
  Mode: focused | Speed: 3.7→4 | Explore: 3.2→3 | Quality: 3.1→3 |
  Tests: 2.3→2 | Spatial: 2.8→3 | Model: 4.6→5
  (Your Friday evenings tend toward focused mode with high model attention)

Manual profile also available: "friday-evening" (saved 2026-03-13)

Take your time reading. When you're ready, I'll ask 4 quick questions — or:
  "use prior" — start from the learned temporal prior
  "load friday-evening" — restore the manual profile snapshot
```

**Key principles:**
- This is a PAUSE, not a speedbump. The human should feel invited to read, not rushed.
- Keep the list to 3-5 items max. Compress aggressively — file paths + one-line summary.
- If a temporal prior exists for the current segment with >= min_observations, surface it prominently as the recommended quick-start. Show both the raw EMA value and the rounded integer.
- The one-line natural language summary highlights dimensions that differ from global default by more than 1 point.
- If a saved profile exists with "friday", "evening", "sunday", or "weekend" in the name, surface it as a secondary option alongside the temporal prior.
- After presenting the inread, WAIT for the human to respond before proceeding to the questionnaire. They might say "use prior" (load temporal prior, skip questionnaire), "load friday-evening" (load manual profile, skip questionnaire), or "ok ready" (proceed to questionnaire), or ask a question about something they read.

### Step 4: Questionnaire (if no shorthand)

Ask the user up to 4 questions. Keep it fast -- the user should be able to attune in under 30 seconds. Present all answerable questions in a single message, numbered. The user can answer with just numbers or short phrases.

**Q1 -- Session mode:**
```
1. What mode for this session?
   a) exploratory -- open-ended, creative, follow interesting threads {current indicator}
   b) focused -- clear goal, execute efficiently {current indicator}
   c) sprint -- ship it, speed over perfection {current indicator}
   d) creative -- architecture and invention mode {current indicator}
```
Mark the current mode with `[current]`.

**Q2 -- Amplify (multi-select):**
Pick the 3-4 dimensions most likely to be relevant based on recent scratchpad context. Show current values.
```
2. What to amplify? (pick any, or skip)
   a) Execution speed (currently {n})
   b) Code quality (currently {n})
   c) Spatial correctness (currently {n})
   d) Model architecture (currently {n})
```
Omit dimensions already at 4 or 5 (they are already amplified). If all are already high, say "All dimensions already amplified -- skip or name one to adjust."

**Q3 -- Ego recommendations (conditional):**
Run the recommendation logic: scan `.claude/scratchpad/ego/` for the latest entry, look for patterns suggesting new dimensions. Use the `recommend_dimensions()` logic described in `.claude/hooks/supra_reader.py`:
- Pattern: `N+ sessions deferred` suggests an `urgency` dimension
- Pattern: `BLOCKED` signals suggest an `unblocking` dimension
- Frequently mentioned terms (8+ occurrences) not covered by existing dimensions

Only show Q3 if there are recommendations. Otherwise skip it entirely.
```
3. Ego noticed patterns that suggest new dimensions. Add any?
   a) urgency -- "How aggressively to prioritize long-deferred items" (ego shows items deferred up to N sessions)
   b) {other recommendation}
   c) None
```

**Q4 -- Focus/suppress:**
```
4. Anything to focus on or suppress this session? (or skip)
   a) Focus: "{contextual suggestion based on recent scratchpads}"
   b) Focus: "{another suggestion}"
   c) Suppress: "{contextual suggestion}"
   d) Other (tell me)
```

Generate contextual suggestions by reading the coordinator's latest scratchpad for active work items and deferred items. If there is no coordinator scratchpad, offer generic options like "Focus: specific goal" / "Suppress: cleanup tasks" / "Skip".

### Step 5: Apply Changes

Based on user answers (from questionnaire or shorthand):

1. **Mode**: Update `mode` field in states
2. **Amplified dimensions**: Set selected dimensions to 4 (HIGH). If the user explicitly provides a number, use that number instead.
3. **Non-selected dimensions**: Leave at their current values. Do NOT reset them to defaults.
4. **New dimensions from recommendations**: Add to both files:
   - `schema.yaml`: add under `dimensions:` with a reasonable `group`, `default: 3`, labels, and agent_relevance
   - `characteristic_states.yaml`: add to `dimensions:` with value 3 (or 4 if the user selected it for amplification)
5. **Focus/suppress**: Replace the lists with the user's selections. If user said "skip", leave unchanged.
6. **Metadata**: Set `last_attuned` to current ISO timestamp, `last_attuned_by` to the session name (read from `.claude/coordinators/.current_session_id` if it exists, otherwise use "manual")
7. **Record temporal observation**: Call `supra_reader.record_temporal_observation(states)` to update the EMA prior for the current temporal segment. This fires regardless of how the values were set (shorthand, questionnaire, or `use prior`). Every valuation is an observation.

Write states to the **supra session file** at `.claude/supra/sessions/{supra_session_id}.yaml` using `supra_reader.write_supra_session_states()`. The supra session ID is deterministic (temporal segment + date, e.g., `friday-evening-2026-03-13`) and shared across all coordinator processes in the same time slot. If no supra session ID is available, fall back to `supra_reader.write_session_states()`. Do NOT write to the global `characteristic_states.yaml` — that file is the prior/default for sessions that haven't attuned. The file format is:
```yaml
mode: {mode}
dimensions:
  execution_speed: {n}
  exploration_vs_exploitation: {n}
  ...
focus: [{items}]
suppress: [{items}]
last_attuned: "{ISO timestamp}"
last_attuned_by: "{session id}"
```

If no session ID is available (e.g. running outside coordinator mode), fall back to writing the global file.

If adding new dimensions to `schema.yaml`, preserve the existing structure and append the new dimension under the appropriate group.

### Step 6: Print Summary

Show a before/after comparison. Keep it compact:

```
Attunement applied:
  Mode: exploratory -> focused
  execution_speed: 3 -> 5
  test_coverage: 3 -> 1
  Focus: (none) -> "prove hex2vec works"
  (4 dimensions unchanged)
```

Only show dimensions that changed. Group unchanged dimensions into a count.

### Step 7: Handoff to Coordinator

After attunement is applied and the summary is shown, offer to launch the coordinator:

```
Ready to work. What's the task?
  a) Follow the forward-look recommendations
  b) [contextual suggestion from focus directive, e.g. "hex2vec results review"]
  c) Tell me what you want to do
```

**If the user provides a task** (either by picking an option or typing freely), invoke `/niche` with that task as the argument. The attunement is already applied — the coordinator will pick it up via the session-start hook.

**If the user says "skip"**, "not yet", or similar — end the skill. The human may want to read more or work without coordination.

**If the morning inread (Step 3.5) surfaced a forward-look with specific wave recommendations**, option (a) should reference the forward-look file path so the coordinator can follow it as a plan: e.g., "Follow forward-look (.claude/scratchpad/coordinator/2026-03-06-forward-look.md)".

This makes `/valuate` the single entry point for all sessions: orient → tune → work.

## Behavioral Rules

- **Speed over thoroughness**: This skill should complete in one exchange (shorthand) or two exchanges (questionnaire + answers). Do not over-explain.
- **Idempotent**: Running `/valuate` multiple times is fine. Re-running with no arguments re-displays the landscape and re-asks questions.
- **Preserve unknowns**: If `characteristic_states.yaml` has fields you do not recognize, preserve them when writing back.
- **Clamping**: All dimension values must be integers in [1, 5]. Clamp if the user says something outside this range.
- **No scratchpad**: This skill does not write a scratchpad entry. It is a pure configuration tool.
- **Mode biases are NOT baked in**: Mode biases are applied at read-time by `supra_reader.py`. The states file stores raw values only. Do not pre-apply biases when writing.
