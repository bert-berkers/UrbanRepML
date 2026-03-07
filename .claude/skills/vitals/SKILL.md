---
name: vitals
description: "Show agent vitals -- who's alive, remaining context budget, recent deaths. Use with /loop for continuous monitoring (e.g. /loop 5m /vitals)."
user-invocable: true
allowed-tools: [Bash]
---

Run the agent timer monitor and display results:

```bash
python .claude/hooks/agent_timer.py monitor
```

Print the output directly to the user. No commentary needed -- just the vitals.
