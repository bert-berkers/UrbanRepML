"""
Sync pulse: lateral coordinator communication system.
- Stable per-terminal session ID (keyed by PID)
- Reads latest message from each other coordinator
- Sends targeted messages to known active coordinators
- Broadcasts to all as fallback for unknown newcomers

Usage: python .claude/sync_run.py [task description]
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path('.claude/hooks').resolve()))

coord_dir = Path('.claude/coordinators')
coord_dir.mkdir(exist_ok=True)
now = datetime.now()
task_arg = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else ''

# --- 1. Stable per-terminal session ID ---
pid = os.getpid()
pid_file = coord_dir / '.pid_{}'.format(pid)

if pid_file.exists():
    session_id = pid_file.read_text(encoding='utf-8').strip()
else:
    # Try current_session_id — only accept named (not sync-* ephemeral)
    session_file = coord_dir / '.current_session_id'
    candidate = ''
    if session_file.exists():
        candidate = session_file.read_text(encoding='utf-8').strip()

    if candidate and not candidate.startswith('sync-'):
        session_id = candidate
    else:
        # Try non-stale claim files
        try:
            import coordinator_registry as cr
            import yaml
            session_id = ''
            for claim_f in sorted(coord_dir.glob('session-*.yaml'), key=lambda f: f.stat().st_mtime, reverse=True):
                try:
                    data = yaml.safe_load(claim_f.read_text(encoding='utf-8'))
                    if not cr.is_stale(data):
                        session_id = data.get('session_id', '')
                        break
                except Exception:
                    pass
        except Exception:
            session_id = ''

        if not session_id:
            session_id = 'coord-{}'.format(pid)

    pid_file.write_text(session_id, encoding='utf-8')

# --- 2. Task description (arg > claim file) ---
task_description = task_arg
if not task_description:
    try:
        import yaml
        claim_file = coord_dir / 'session-{}.yaml'.format(session_id)
        if claim_file.exists():
            claim = yaml.safe_load(claim_file.read_text(encoding='utf-8'))
            task_description = claim.get('task_summary', '')
    except Exception:
        pass

print('## Sync Pulse {} | {}'.format(now.strftime('%H:%M:%S'), session_id))
if task_description:
    print('Task:', task_description)
print()

# --- 3. Read messages from other coordinators ---
try:
    import coordinator_registry as cr
    since = now - timedelta(minutes=15)
    all_msgs = cr.read_messages(coord_dir, since=since)

    # Group by sender, keep latest per coordinator (exclude self and sync-* noise)
    by_sender = {}
    for m in all_msgs:
        sender = m.get('from', '')
        if not sender or sender == session_id or sender.startswith('sync-') or sender.startswith('coord-'):
            continue
        by_sender[sender] = m  # last message wins (messages are time-ordered)

    if by_sender:
        print('**Other coordinators** ({}):'.format(len(by_sender)))
        for sender, m in sorted(by_sender.items(), key=lambda x: x[1].get('at', '')):
            body = m.get('body', '')[:200]
            level = m.get('level', 'info')
            at = m.get('at', '')
            at_fmt = at[11:16] if len(at) > 11 else at  # HH:MM
            print('  [{}] {} @ {}: {}'.format(level, sender, at_fmt, body))
    else:
        print('**No messages from other coordinators** (last 15m)')

    print()

    # --- 4. Broadcast / targeted send ---
    try:
        import agent_timer
        living = agent_timer.alive()
        alive_names = ', '.join(
            '{}({}%)'.format(a['agent_type'], a.get('estimated_remaining_pct', '?'))
            for a in living
        ) if living else 'none'
    except Exception:
        alive_names = 'unknown'

    body_parts = []
    if task_description:
        body_parts.append('Task: {}'.format(task_description))
    body_parts.append('Alive: [{}]'.format(alive_names))
    body = ' | '.join(body_parts)
    body = body.encode('ascii', errors='replace').decode('ascii')

    # Send targeted message to each known active coordinator
    sent_to = []
    for target_id in by_sender:
        try:
            cr.write_message(coord_dir, {
                'from': session_id,
                'to': target_id,
                'level': 'info',
                'at': now.isoformat(timespec='seconds'),
                'body': body,
            })
            sent_to.append(target_id)
        except Exception:
            pass

    # Also broadcast to all (for coordinators not yet seen)
    try:
        cr.write_message(coord_dir, {
            'from': session_id,
            'to': 'all',
            'level': 'info',
            'at': now.isoformat(timespec='seconds'),
            'body': body,
        })
        cr.update_heartbeat(coord_dir, session_id)
    except Exception as e:
        print('**Broadcast failed**: {}'.format(e))

    targets_str = ', '.join(sent_to) if sent_to else 'none (solo)'
    print('**Sent** to: {} + all'.format(targets_str))
    print('  {}'.format(body))

except Exception as e:
    print('**Sync failed**: {}'.format(e))
