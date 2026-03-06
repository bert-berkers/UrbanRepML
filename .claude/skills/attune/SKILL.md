---
name: attune
description: "Adjust your precision weights -- tell the system what you care about right now."
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

Read these two files:
- `.claude/supra/characteristic_states.yaml` -- current dimension values, mode, focus/suppress
- `.claude/supra/schema.yaml` -- dimension definitions, groups, mode biases, agent relevance

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
- `focus "..."` sets the focus directive (replaces existing focus list with single item)
- `suppress "..."` sets the suppress directive (replaces existing suppress list with single item)
- Bare dimension names without a value are treated as amplify (set to 4)
- Values are integers 1-5
- `save:name` saves current state as a named profile (after applying any other changes in the same command)
- `load:name` loads a saved profile, replacing mode, dimensions, focus, and suppress
- `profiles` or `list` lists all saved profiles

**Profile operations:**
- Profiles are stored in `.claude/supra/profiles/{name}.yaml`
- `save:` runs AFTER other shorthand changes, so `/attune sprint, save:sprint-default` first applies sprint mode, then saves the result
- `load:` runs BEFORE other shorthand changes, so `/attune load:training, speed 5` loads the training profile then overrides speed
- Use `supra_reader.save_profile()`, `supra_reader.load_profile()`, and `supra_reader.list_profiles()` from `.claude/hooks/supra_reader.py`

**Examples:**
- `/attune speed 5, tests 1` -- sets execution_speed=5, test_coverage=1
- `/attune sprint` -- sets mode to sprint
- `/attune focused, model 5, focus "prove hex2vec works"` -- mode=focused, model_architecture=5, focus set
- `/attune explore` -- ambiguous: could mean mode=exploratory or exploration_vs_exploitation=4. Resolve: if no number follows, check if it matches a mode name first. `explore` is not a mode name, so it maps to the dimension at value 4. `exploratory` IS a mode name.
- `/attune save:creative-evening` -- saves current state as "creative-evening"
- `/attune load:training` -- restores the "training" profile
- `/attune sprint, speed 5, tests 1, save:ship-it` -- applies sprint + overrides, saves as "ship-it"
- `/attune profiles` -- lists all saved profiles

If shorthand is provided, apply changes and jump to Step 6 (print summary). Do not ask questions.

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

Write states back using the same YAML structure as `characteristic_states.yaml`. The file format is:
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

## Behavioral Rules

- **Speed over thoroughness**: This skill should complete in one exchange (shorthand) or two exchanges (questionnaire + answers). Do not over-explain.
- **Idempotent**: Running `/attune` multiple times is fine. Re-running with no arguments re-displays the landscape and re-asks questions.
- **Preserve unknowns**: If `characteristic_states.yaml` has fields you do not recognize, preserve them when writing back.
- **Clamping**: All dimension values must be integers in [1, 5]. Clamp if the user says something outside this range.
- **No scratchpad**: This skill does not write a scratchpad entry. It is a pure configuration tool.
- **Mode biases are NOT baked in**: Mode biases are applied at read-time by `supra_reader.py`. The states file stores raw values only. Do not pre-apply biases when writing.
