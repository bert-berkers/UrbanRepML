# Coordinator Hello Broadcast

## Status: Draft

## Context

Coordinators currently register claim files at session start (via the SessionStart hook) and check for incoming messages at each OODA OBSERVE phase. However, they never proactively SEND a message at startup. This creates a coordination gap:

- **Coordinator A** starts, does work, ends. Its claim file is deleted by the Stop hook. Its scratchpad captures what happened, but scratchpads are sequential records -- they describe completed work, not intent or in-flight risk.
- **Coordinator B** starts 10 minutes later. It reads claim files (empty -- A is gone). It reads messages (none -- A never sent any). B has no idea what A was doing, what files A touched, or what conflicts might exist with uncommitted work.

The claim file cannot solve this because it is ephemeral -- deleted on session end. The scratchpad partially solves it, but scratchpads are retrospective and unstructured. What is missing is a **prospective, structured announcement** that persists in the message log.

The hello broadcast fills this gap. It fires at Wave 0, uses the existing `write_message` infrastructure, and provides a forward-looking summary that any coordinator (current or future) can read within the 7-day message TTL.

## Decision

Add a mandatory step 7 to Wave 0 in the coordinator skill. After reading the session ID (step 6), the coordinator writes a single `info`-level message to `"all"` via `coordinator_registry.write_message()`. The message body is a structured text block containing:

| Field | Purpose |
|-------|---------|
| Session ID | Identifies this coordinator for future reference |
| Task summary | 1 sentence describing the user's request |
| Intent | What this coordinator plans to do (actions, not just topic) |
| Risk areas | Specific files or directories that may be modified |
| Claimed paths | Initial path claims (to be narrowed from `*` as usual) |

### Message Format

The hello message uses the existing message schema. No new fields are introduced.

```yaml
from: "gentle-amber-tide"
to: "all"
at: "2026-03-01T14:30:22"
level: "info"
body: |
  HELLO gentle-amber-tide
  Task: Implement POI embedder using hex2vec via SRAI
  Intent: Dispatch stage1-modality-encoder to wire up hex2vec, then QAQC to verify output shapes
  Risk: stage1_modalities/poi/*, utils/paths.py
  Claimed: stage1_modalities/**, scripts/processing_modalities/**
```

The `HELLO` prefix is a convention, not a schema change. It makes hello messages visually distinguishable when scanning the messages directory or reading them in OBSERVE.

### Firing Conditions

The hello broadcast fires ALWAYS -- even if no other coordinators are currently active. The message is not for live peers; it is for the NEXT coordinator that starts and reads messages during its OBSERVE phase. A message addressed to `"all"` with a 7-day TTL covers both cases.

### Timing

The hello fires after step 6 (read session name) and before the coordinator proceeds to the first OODA cycle. At this point, the coordinator has:
- A clean git state (steps 1-2)
- Knowledge of any active plan (step 4)
- The session ID (step 6)
- The user's task from `$ARGUMENTS`

This is sufficient to compose a meaningful hello. The hello does not require full OBSERVE output -- it is a declaration of intent, not a status report.

## Insertion Point in skill.md

The new step goes at line 43 of `.claude/skills/coordinate/skill.md`, after step 6 and before the "This is non-negotiable" closing line. The exact text to add:

```markdown
7. **Hello broadcast** -- write an `info` message to `"all"` via `coordinator_registry.write_message()`:
   ```
   HELLO {session_id}
   Task: {1-sentence summary of $ARGUMENTS}
   Intent: {what you plan to do, e.g. "dispatch stage1 encoder + QAQC verification"}
   Risk: {specific files/dirs you expect to modify}
   Claimed: {initial claimed_paths, to be narrowed in first OODA cycle}
   ```
   This fires ALWAYS, even with no other active coordinators. The message is for future coordinators, not just current ones.
```

This adds 7 lines (within the 5-10 line budget) and slots cleanly between step 6 and the existing non-negotiable closing sentence.

## Alternatives Considered

### Extending the claim file instead of using a message

The claim file already has `task_summary` and `claimed_paths`. We could add `intent` and `risk_areas` fields to the claim YAML schema.

**Rejected.** The claim file is ephemeral -- deleted on session end. The hello's primary value is for FUTURE coordinators, not current ones. Messages persist for 7 days; claim files persist only while the session is alive. Adding fields to the claim file solves the live-peer case (which already works via claim scanning) but not the sequential-coordinator case (which is the actual gap).

### Making hello optional ("send only if other coordinators are active")

**Rejected.** This misunderstands the purpose. The hello is not a handshake with a live peer -- it is a trail marker for whoever comes next. If no one is active now, the message is even more important because there is no claim file to read either.

### Adding a structured schema (new message type)

We could add a `type: "hello"` field to the message schema and teach the OBSERVE phase to parse it.

**Rejected for now.** The existing `level: "info"` + `HELLO` prefix convention is sufficient. A structured schema adds parsing logic to `coordinator_registry.py` and `subagent-context.py` for no immediate benefit. If hello messages need machine-parseable fields in the future, promote the prefix to a schema field then.

## Consequences

### Positive

- Sequential coordinators gain forward-looking context about previous sessions, not just retrospective scratchpad summaries.
- The OBSERVE phase already reads messages addressed to `"all"`, so hello messages surface automatically with zero new infrastructure.
- Risk areas declared up-front reduce the chance of silent conflicts with uncommitted work from ended sessions.
- Zero new code -- uses existing `write_message` with existing fields.

### Negative

- One additional message file per coordinator session in `.claude/coordinators/messages/`. Cleaned up by the existing 7-day TTL.
- The coordinator must compose the hello before completing OODA, which means the task/intent/risk fields are based on initial understanding. They may not perfectly reflect what the coordinator actually ends up doing. This is acceptable -- the hello is a declaration of intent, and the claim file (narrowed during OODA) serves as the authoritative path lock.

### Neutral

- The hello message is advisory, like everything in the C2C protocol. It does not enforce anything.
- Single-coordinator sessions still write the hello. The overhead is one YAML file write (~1ms).

## Implementation Notes

1. Edit `.claude/skills/coordinate/skill.md` to add step 7 in Wave 0 (after step 6, before the closing line).
2. No changes to `coordinator_registry.py`, `session-start.py`, `subagent-context.py`, or any hook.
3. No changes to the message schema or claim file schema.
4. The coordinator composes the message body inline using `$ARGUMENTS` and its session ID. No template system needed.
5. Verification: the next coordinator session should show a hello message in its OBSERVE phase output. QAQC can check this by looking for `HELLO` in the OODA report's Lateral section.
