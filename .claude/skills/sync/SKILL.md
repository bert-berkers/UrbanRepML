---
name: sync
description: "Active listening sync pulse. Reads scratchpad activity, signals, and coordinator messages — then broadcasts this session's status so other coordinators see us. Two-way, not passive. Use with /loop (e.g. /loop 3m /sync)."
user-invocable: true
allowed-tools: [Bash, Read, Glob]
---

## Sync pulse: active listening

Not just reading — broadcasting. Every pulse:
1. Scan scratchpads for recent activity
2. Read coordinator messages addressed to us
3. Surface pipeline signals
4. **Broadcast**: write our status + alive agents + recent findings as a coordinator message
5. Update heartbeat

```bash
python -c "
import sys, json, os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path('.claude/hooks').resolve()))

root = Path('.claude/scratchpad')
msg_dir = Path('.claude/coordinators/messages')
coord_dir = Path('.claude/coordinators')
session_file = coord_dir / '.current_session_id'
now = datetime.now()
cutoff = now - timedelta(minutes=15)
signals_kw = ['BLOCKED', 'URGENT', 'CRITICAL', 'BROKEN', 'SHAPE_CHANGED', 'INTERFACE_CHANGED']

# Read session ID (fall back to hostname+pid if file is owned by another terminal)
session_id = ''
if session_file.exists():
    session_id = session_file.read_text(encoding='utf-8').strip()
if not session_id:
    import os as _os
    session_id = 'sync-' + str(_os.getpid())

print('## Sync Pulse', now.strftime('%H:%M:%S'))
print()

# --- LISTEN: Recent scratchpad activity ---
active = []
for agent_dir in (sorted(root.iterdir()) if root.is_dir() else []):
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

# --- LISTEN: Pipeline signals ---
print()
found_signals = []
for agent_dir in (sorted(root.iterdir()) if root.is_dir() else []):
    if not agent_dir.is_dir():
        continue
    for f in sorted(agent_dir.glob('*.md'), reverse=True)[:1]:
        try:
            content = f.read_text(encoding='utf-8').upper()
            for sig in signals_kw:
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

# --- LISTEN: Coordinator messages to us ---
print()
try:
    import coordinator_registry as cr
    since = now - timedelta(minutes=15)
    msgs = cr.read_messages(coord_dir, since=since, to_session=session_id) if session_id else []
    if msgs:
        print(f'**Messages to us** ({len(msgs)}):')
        for m in msgs[-5:]:
            sender = m.get('from', '?')
            body = m.get('body', '')[:120]
            print(f'  [{m.get(\"level\",\"info\")}] from {sender}: {body}')
    else:
        print('**No messages to us** (last 15m)')
except Exception as e:
    print(f'**Message read failed**: {e}')

# --- BROADCAST: Tell other coordinators what we're doing ---
print()
try:
    import agent_timer
    living = agent_timer.alive()
    alive_summary = ', '.join(f'{a[\"agent_type\"]}({a.get(\"estimated_remaining_pct\",\"?\")}%)' for a in living) if living else 'none'
    recent_dead = agent_timer.recent_dead(3)
    dead_summary = ', '.join(f'{d[\"agent_type\"]}({d.get(\"lived_min\",\"?\")}m)' for d in recent_dead) if recent_dead else 'none'

    # Build broadcast body
    signal_summary = f'{len(found_signals)} signals' if found_signals else 'no signals'
    activity_summary = f'{len(active)} active scratchpads' if active else 'quiet'
    body = f'Alive: [{alive_summary}] | Dead: [{dead_summary}] | {activity_summary}, {signal_summary}'

    if session_id:
        cr.write_message(coord_dir, {
            'from': session_id,
            'to': 'all',
            'level': 'info',
            'at': now.isoformat(timespec='seconds'),
            'body': body,
        })
        cr.update_heartbeat(coord_dir, session_id)
        print(f'**Broadcast sent** as {session_id}')
        print(f'  {body}')
    else:
        print('**No session ID -- broadcast skipped**')
except Exception as e:
    print(f'**Broadcast failed**: {e}')

# --- OTHER COORDINATORS ---
print()
try:
    all_claims = cr.read_all_claims(coord_dir)
    others = [c for c in all_claims if c.get('session_id') != session_id and not cr.is_stale(c)]
    if others:
        print(f'**Other coordinators** ({len(others)}):')
        for c in others:
            sid = c.get('session_id', '?')
            task = c.get('task_summary', 'no summary')[:80]
            hb = c.get('heartbeat_at', '?')
            print(f'  {sid}: {task} (heartbeat: {hb})')
    else:
        print('**Solo** -- no other active coordinators')
except Exception as e:
    print(f'**Coordinator check failed**: {e}')
"
```

Print the output directly. Keep it brief.
