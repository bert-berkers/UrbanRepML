"""
Sync pulse: lateral coordinator communication system.
- PPID-keyed session identity (multi-terminal safe)
- Gate check: only fires during dynamic graph (niche construction)
- Reads latest message from each other coordinator
- Broadcasts via write_lateral_message (uses supra identity)

Usage: python .claude/sync_run.py [task description]
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, str(Path('.claude/hooks').resolve()))

import coordinator_registry as cr
import supra_reader

coord_dir = Path('.claude/coordinators')
coord_dir.mkdir(exist_ok=True)
now = datetime.now()
task_arg = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else ''

# --- 0. Gate check: lateral coupling only active during /niche ---
if not supra_reader.is_lateral_coupling_active():
    print('Sync skipped -- static graph active (no lateral coupling during /valuate)')
    sys.exit(0)

# --- 1. Identity (terminal-PID-keyed, one identity per terminal) ---
identity_id = cr.read_ppid_identity(coord_dir) or ''
# Legacy locals — sync_run.py has historically named these supra_sid / session_id;
# both now resolve to the same terminal identity.
supra_sid = identity_id
session_id = identity_id
display_id = identity_id or 'unknown'

# --- 2. Task description (arg > claim file) ---
task_description = task_arg
if not task_description:
    try:
        import yaml
        claim_file = coord_dir / 'session-{}.yaml'.format(session_id)
        if claim_file.exists():
            claim = yaml.safe_load(claim_file.read_text(encoding='utf-8'))
            if claim and claim.get('status') != 'ended':
                task_description = claim.get('task_summary', '')
    except Exception:
        pass

print('## Sync Pulse {} | {}'.format(now.strftime('%H:%M:%S'), display_id))
if task_description:
    print('Task:', task_description)
print()

# --- 3. Read messages from other coordinators ---
try:
    since = now - timedelta(minutes=15)
    all_msgs = cr.read_messages(coord_dir, since=since)

    # Group by sender, keep latest per coordinator (exclude self)
    by_sender = {}
    for m in all_msgs:
        sender = m.get('from', '')
        if not sender or sender == supra_sid or sender == session_id:
            continue
        if sender.startswith('sync-') or sender.startswith('coord-'):
            continue
        by_sender[sender] = m  # last message wins (messages are time-ordered)

    if by_sender:
        print('**Other coordinators** ({}):'.format(len(by_sender)))
        for sender, m in sorted(by_sender.items(), key=lambda x: x[1].get('at', '')):
            body = m.get('body', m.get('text', ''))[:200]
            level = m.get('level', 'info')
            at = m.get('at', '')
            at_fmt = at[11:16] if len(at) > 11 else at  # HH:MM
            print('  [{}] {} @ {}: {}'.format(level, sender, at_fmt, body))
    else:
        print('**No messages from other coordinators** (last 15m)')

    print()

    # --- 4. Broadcast using supra identity ---
    body = task_description or '(no task description)'

    # Use write_lateral_message for supra-identity-aware broadcasting
    cr.write_lateral_message(coord_dir, {
        'to': 'all',
        'level': 'info',
        'at': now.isoformat(timespec='seconds'),
        'body': body,
    })

    if session_id:
        cr.update_heartbeat(coord_dir, session_id)

    print('**Broadcast sent** as {}'.format(display_id))
    print('  {}'.format(body))

except Exception as e:
    print('**Sync failed**: {}'.format(e))
