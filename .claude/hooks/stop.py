#!/usr/bin/env python3
"""Stop hook: verify coordinator wrote a quality scratchpad if multi-agent work occurred."""
import json
import sys
from datetime import date, datetime
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"
COORDINATORS_DIR = Path(__file__).resolve().parents[1] / "coordinators"
SESSION_ID_FILE = COORDINATORS_DIR / ".current_session_id"

MIN_LINES = 5  # Match subagent-stop quality bar


def deregister_coordinator() -> None:
    """Delete this session's claim file and clean up stale claims.

    Reads the session_id from the file written by session-start.py.
    Silently no-ops on any error (fail open).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import coordinator_registry as cr

        if not SESSION_ID_FILE.exists():
            print("stop: no session_id file found, skipping coordinator deregistration", file=sys.stderr)
            return

        session_id = SESSION_ID_FILE.read_text(encoding="utf-8").strip()
        if not session_id:
            return

        # Optionally write a farewell message (only if other claim files exist)
        other_claims = [
            c for c in cr.read_all_claims(COORDINATORS_DIR)
            if c.get("session_id") != session_id and not cr.is_stale(c)
        ]
        if other_claims:
            cr.write_message(COORDINATORS_DIR, {
                "from": session_id,
                "to": "all",
                "at": datetime.now().isoformat(timespec="seconds"),
                "type": "done",
                "text": f"Session {session_id} ended. Check git log for changes made this session.",
            })

        # Delete this session's claim file
        cr.delete_claim(COORDINATORS_DIR, session_id)

        # Remove the session ID file
        SESSION_ID_FILE.unlink(missing_ok=True)

        # Clean up crashed sessions (2h+ old)
        cr.cleanup_stale(COORDINATORS_DIR, threshold_hours=2)

        print(f"stop: coordinator deregistered (session_id={session_id})", file=sys.stderr)

    except Exception as exc:
        print(f"stop: coordinator deregistration failed: {exc}", file=sys.stderr)


def main() -> None:
    try:
        hook_input = json.loads(sys.stdin.read() or "{}")
        print(f"Stop hook input: {json.dumps(hook_input)}", file=sys.stderr)

        # Deregister coordinator claim -- do this regardless of scratchpad check outcome
        deregister_coordinator()

        today = date.today().isoformat()

        # Check if any subagent scratchpads were written today
        subagent_scratchpads = []
        for agent_dir in SCRATCHPAD_ROOT.iterdir():
            if not agent_dir.is_dir() or agent_dir.name == "coordinator":
                continue
            scratchpad = agent_dir / f"{today}.md"
            if scratchpad.exists():
                subagent_scratchpads.append(agent_dir.name)

        # If no subagent work happened today, allow
        if not subagent_scratchpads:
            print("Stop: no subagent scratchpads today, allowing", file=sys.stderr)
            json.dump({}, sys.stdout)
            return

        # Multi-agent coordination happened — check coordinator scratchpad exists
        coordinator_scratchpad = SCRATCHPAD_ROOT / "coordinator" / f"{today}.md"

        if not coordinator_scratchpad.exists():
            json.dump({
                "decision": "block",
                "reason": (
                    f"Multi-agent work occurred today (agents: {', '.join(subagent_scratchpads)}), "
                    f"but coordinator scratchpad is missing at "
                    f".claude/scratchpad/coordinator/{today}.md"
                ),
            }, sys.stdout)
            return

        # Check coordinator scratchpad has meaningful content (same bar as subagent-stop)
        lines = coordinator_scratchpad.read_text(encoding="utf-8").splitlines()
        non_empty = [l for l in lines if l.strip()]

        if len(non_empty) < MIN_LINES:
            json.dump({
                "decision": "block",
                "reason": (
                    f"Coordinator scratchpad exists but is too short "
                    f"({len(non_empty)} non-empty lines, need {MIN_LINES}+). "
                    f"Please add a meaningful summary of today's coordination."
                ),
            }, sys.stdout)
            return

        print(
            f"Stop: coordinator scratchpad OK ({len(non_empty)} lines), allowing",
            file=sys.stderr,
        )
        json.dump({}, sys.stdout)

    except Exception as e:
        print(f"Stop hook error: {e}", file=sys.stderr)
        json.dump({}, sys.stdout)  # Fail open


if __name__ == "__main__":
    main()
