"""Sync pulse: lateral coordinator communication. Run via .claude/sync_run.py [task description]."""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path('.claude/hooks').resolve()))

coord_dir = Path('.claude/coordinators')
session_file = coord_dir / '.current_session_id'
now = datetime.now()
# Read session ID
session_id = ''
if session_file.exists():
    session_id = session_file.read_text(encoding='utf-8').strip()
if not session_id:
    import os as _os
    session_id = 'sync-' + str(_os.getpid())

# Auto-read task from our own claim file; arg overrides
task_description = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else ''
if not task_description and session_id:
    try:
        import yaml
        claim_file = coord_dir / 'session-{}.yaml'.format(session_id)
        if claim_file.exists():
            claim = yaml.safe_load(claim_file.read_text(encoding='utf-8'))
            task_description = claim.get('task_summary', '')
    except Exception:
        pass

print('## Sync Pulse', now.strftime('%H:%M:%S'))
if task_description:
    print('Task:', task_description)
print()

# --- LATERAL: Coordinator messages (primary signal) ---
try:
    import coordinator_registry as cr
    since = now - timedelta(minutes=15)
    all_msgs = cr.read_messages(coord_dir, since=since) if session_id else []

    # Split: messages from other coordinators vs our own echoes
    own_prefixes = ('sync-', session_id)
    lateral = [m for m in all_msgs if m.get('from', '') and m.get('from') != session_id
               and not m.get('from', '').startswith('sync-')]
    own_echoes = [m for m in all_msgs if m.get('from', '').startswith('sync-')]

    if lateral:
        print('**Coordinator messages** ({}):'.format(len(lateral)))
        for m in lateral[-6:]:
            sender = m.get('from', '?')
            body = m.get('body', '')[:200]
            level = m.get('level', 'info')
            at = m.get('at', '')[-8:] if m.get('at') else ''
            print('  [{}] {} @ {}: {}'.format(level, sender, at, body))
    else:
        print('**No lateral messages** from other coordinators (last 15m)')

    print()

    # Other active coordinators via claim files
    try:
        all_claims = cr.read_all_claims(coord_dir)
        others = [c for c in all_claims if c.get('session_id') != session_id and not cr.is_stale(c)]
        if others:
            print('**Active coordinators** ({}):'.format(len(others)))
            for c in others:
                sid = c.get('session_id', '?')
                task = c.get('task_summary', 'no summary')[:80]
                hb = c.get('heartbeat_at', '?')
                print('  {}: {} (hb: {})'.format(sid, task, hb))
        else:
            print('**Solo** — no other coordinators in claim files')
            if own_echoes:
                most_recent = own_echoes[-1].get('body', '')[:120]
                print('  (last own broadcast: {})'.format(most_recent))
    except Exception as e:
        print('**Claim check failed**: {}'.format(e))

except Exception as e:
    print('**Message read failed**: {}'.format(e))

# --- BROADCAST: Tell other coordinators what we're doing ---
print()
try:
    import agent_timer
    living = agent_timer.alive()
    alive_names = ', '.join(
        '{}({}%)'.format(a['agent_type'], a.get('estimated_remaining_pct', '?'))
        for a in living
    ) if living else 'none'

    body_parts = []
    if task_description:
        body_parts.append('Task: {}'.format(task_description))
    body_parts.append('Alive: [{}]'.format(alive_names))
    body = ' | '.join(body_parts)

    cr.write_message(coord_dir, {
        'from': session_id,
        'to': 'all',
        'level': 'info',
        'at': now.isoformat(timespec='seconds'),
        'body': body,
    })
    cr.update_heartbeat(coord_dir, session_id)
    print('**Broadcast sent** as {}'.format(session_id))
    print('  {}'.format(body))
except Exception as e:
    print('**Broadcast failed**: {}'.format(e))
