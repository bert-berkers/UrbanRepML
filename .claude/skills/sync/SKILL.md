---
name: sync
description: "Cross-agent sync pulse. Reads recent scratchpad entries, coordinator messages, and pipeline signals. Run offset from /vitals so agents have time to write before reading. Use with /loop (e.g. /loop 7m /sync)."
user-invocable: true
allowed-tools: [Bash, Read, Glob]
---

## Sync pulse: read the network

1. Scan `.claude/scratchpad/*/` for entries modified in the last 15 minutes
2. Check `.claude/coordinators/messages/` for unread coordinator messages
3. Surface any pipeline signals (BLOCKED, URGENT, SHAPE_CHANGED, etc.)
4. Update coordinator heartbeat

Run this command to gather everything:

```bash
python -c "
import sys, json
from pathlib import Path
from datetime import datetime, timedelta

root = Path('.claude/scratchpad')
msg_dir = Path('.claude/coordinators/messages')
now = datetime.now()
cutoff = now - timedelta(minutes=15)
signals = ['BLOCKED', 'URGENT', 'CRITICAL', 'BROKEN', 'SHAPE_CHANGED', 'INTERFACE_CHANGED']

print('## Sync Pulse', now.strftime('%H:%M:%S'))
print()

# Recent scratchpad activity
active = []
for agent_dir in sorted(root.iterdir()) if root.is_dir() else []:
    if not agent_dir.is_dir():
        continue
    for f in sorted(agent_dir.glob('*.md'), reverse=True)[:1]:
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime > cutoff:
            age = (now - mtime).total_seconds() / 60
            lines = [l.strip() for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
            last = lines[-1][:100] if lines else '(empty)'
            active.append((agent_dir.name, round(age, 1), last))

if active:
    print(f'**Recent writes** ({len(active)}):')
    for name, age, last in active:
        print(f'  {name:30s} {age:5.1f}m ago | {last}')
else:
    print('**No recent scratchpad activity** (last 15m)')

# Pipeline signals
print()
found_signals = []
for agent_dir in sorted(root.iterdir()) if root.is_dir() else []:
    if not agent_dir.is_dir():
        continue
    for f in sorted(agent_dir.glob('*.md'), reverse=True)[:1]:
        try:
            content = f.read_text(encoding='utf-8').upper()
            for sig in signals:
                if sig in content:
                    for line in f.read_text(encoding='utf-8').splitlines():
                        if sig in line.upper() and line.strip():
                            found_signals.append(f'  {agent_dir.name}: {line.strip()[:120]}')
                            break
        except: pass

if found_signals:
    print(f'**Signals** ({len(found_signals)}):')
    for s in found_signals:
        print(s)
else:
    print('**No active signals.**')

# Coordinator messages
print()
if msg_dir.is_dir():
    msgs = sorted(msg_dir.glob('*.yaml'), reverse=True)[:5]
    if msgs:
        print(f'**Coordinator messages** ({len(msgs)} recent):')
        for m in msgs:
            print(f'  {m.stem}')
    else:
        print('**No coordinator messages.**')
else:
    print('**No coordinator messages.**')
"
```

Print the output directly. Keep it brief.
