---
name: sync
description: "Lateral coordinator sync pulse. Reads messages from other coordinators, broadcasts what this session is working on. Lateral-first — scratchpads are for initial setup, not pulse monitoring. Use with /loop (e.g. /loop 5m /sync monitoring CE viz + UNet review)."
user-invocable: true
allowed-tools: [Bash, Read, Glob]
---

## Sync pulse: lateral coordinator communication

Primary purpose: coordinators telling each other what they're working on.

1. **Read**: lateral messages from other coordinators (last 15m), filter out own echoes
2. **Check**: active coordinator claim files
3. **Broadcast**: this session's task + alive agents

Pass a task description as the argument — it gets embedded in the broadcast so other coordinators know what you're doing:

```
/sync monitoring CE viz + UNet review
/loop 5m /sync Stage 2 UNet investigation
```

```bash
python .claude/sync_run.py {{{ARGUMENTS}}}
```

Print the output directly. Keep it brief.
