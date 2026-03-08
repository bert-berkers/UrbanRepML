import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path('.claude/hooks').resolve()))

root = Path('.claude/scratchpad')
coord_dir = Path('.claude/coordinators')
session_file = coord_dir / '.current_session_id'
now = datetime.now()
cutoff = now - timedelta(minutes=15)
signals_kw = ['BLOCKED', 'URGENT', 'CRITICAL', 'BROKEN', 'SHAPE_CHANGED', 'INTERFACE_CHANGED']

session_id = ''
if session_file.exists():
    session_id = session_file.read_text(encoding='utf-8').strip()
if not session_id:
    import os as _os
    session_id = 'sync-' + str(_os.getpid())

print('## Sync Pulse', now.strftime('%H:%M:%S'))
print()

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
    print('**Recent writes** ({}):'.format(len(active)))
    for name, age, last in active:
        print('  {:30s} {:5.1f}m ago | {}'.format(name, age, last))
else:
    print('**No recent scratchpad activity** (last 15m)')

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
                            found_signals.append('  {}: {}'.format(agent_dir.name, line.strip()[:120]))
                            break
        except Exception:
            pass

if found_signals:
    print('**Signals** ({}):'.format(len(found_signals)))
    for s in found_signals:
        print(s)
else:
    print('**No active signals.**')

print()
try:
    import coordinator_registry as cr
    since = now - timedelta(minutes=15)
    msgs = cr.read_messages(coord_dir, since=since, to_session=session_id) if session_id else []
    if msgs:
        print('**Messages to us** ({}):'.format(len(msgs)))
        for m in msgs[-5:]:
            print('  [{}] from {}: {}'.format(m.get('level', 'info'), m.get('from', '?'), m.get('body', '')[:120]))
    else:
        print('**No messages to us** (last 15m)')
except Exception as e:
    print('**Message read failed**: {}'.format(e))

print()
try:
    import agent_timer
    living = agent_timer.alive()
    alive_summary = ', '.join('{}({}%)'.format(a['agent_type'], a.get('estimated_remaining_pct', '?')) for a in living) if living else 'none'
    recent_dead = agent_timer.recent_dead(3)
    dead_summary = ', '.join('{}({}m)'.format(d['agent_type'], d.get('lived_min', '?')) for d in recent_dead) if recent_dead else 'none'
    signal_summary = '{} signals'.format(len(found_signals)) if found_signals else 'no signals'
    activity_summary = '{} active scratchpads'.format(len(active)) if active else 'quiet'
    body = 'Alive: [{}] | Dead: [{}] | {}, {}'.format(alive_summary, dead_summary, activity_summary, signal_summary)
    if session_id:
        cr.write_message(coord_dir, {
            'from': session_id, 'to': 'all', 'level': 'info',
            'at': now.isoformat(timespec='seconds'), 'body': body
        })
        cr.update_heartbeat(coord_dir, session_id)
        print('**Broadcast sent** as {}'.format(session_id))
        print('  {}'.format(body))
    else:
        print('**No session ID -- broadcast skipped**')
except Exception as e:
    print('**Broadcast failed**: {}'.format(e))

print()
try:
    all_claims = cr.read_all_claims(coord_dir)
    others = [c for c in all_claims if c.get('session_id') != session_id and not cr.is_stale(c)]
    if others:
        print('**Other coordinators** ({}):'.format(len(others)))
        for c in others:
            print('  {}: {} (heartbeat: {})'.format(
                c.get('session_id', '?'),
                c.get('task_summary', 'no summary')[:80],
                c.get('heartbeat_at', '?')
            ))
    else:
        print('**Solo** -- no other active coordinators')
except Exception as e:
    print('**Coordinator check failed**: {}'.format(e))