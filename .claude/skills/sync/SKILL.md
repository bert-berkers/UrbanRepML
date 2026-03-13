---
name: sync
description: "Lateral coordinator sync pulse. Reads messages from other coordinators, broadcasts what this session is working on. Lateral-first — scratchpads are for initial setup, not pulse monitoring. Use with /loop (e.g. /loop 5m /sync monitoring CE viz + UNet review)."
user-invocable: true
allowed-tools: [Bash, Read, Glob]
---

## Sync pulse: lateral coordinator communication

Primary purpose: coordinators telling each other what they're working on.

**Graph context**: `/sync` is the runtime implementation of the **Percept ↔ Percept** dotted edges in the dynamic liveability graph. Each terminal is a narrator (homo narrans). The supra session ID is the narrator's stable identity — it persists across context window refreshes so the story stays coherent.

**Active only during niche construction**: `/sync` should check `supra_reader.is_lateral_coupling_active()` before broadcasting. During `/valuate` (static graph), there are no Per↔Per edges — lateral coupling is inactive. If called during static mode, print a brief note and skip.

### Protocol

1. **Gate check**: Call `supra_reader.is_lateral_coupling_active()`. If False (static graph / `/valuate` mode), print `"Sync skipped — static graph active (no lateral coupling)"` and return.
2. **Identify**: Read supra session ID from `.claude/coordinators/.current_supra_session_id`. This is your narrator name. Fall back to coordinator session ID if unavailable.
3. **Listen**: Read lateral messages from other coordinators (last 15m), filter out own echoes (match on supra session ID, not coordinator ID).
4. **Check**: Read active coordinator claim files for awareness.
5. **Narrate**: Broadcast this terminal's story using `coordinator_registry.write_lateral_message()`:
   - `from`: supra session ID (auto-populated by `write_lateral_message`)
   - `to`: "all"
   - `type`: "info"
   - `text`: task description from arguments + alive agent summary

### Execution

If `.claude/sync_run.py` exists, delegate:
```bash
python .claude/sync_run.py {{{ARGUMENTS}}}
```

Otherwise, execute inline:
1. Read supra session ID and check lateral coupling gate (as above)
2. Use `coordinator_registry.read_messages()` for recent messages
3. Use `coordinator_registry.write_lateral_message()` for broadcast
4. Print results directly

Pass a task description as the argument:
```
/sync monitoring CE viz + UNet review
/loop 5m /sync Stage 2 UNet investigation
```

Print the output directly. Keep it brief.
