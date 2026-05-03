#!/usr/bin/env python3
"""Stop hook: verify coordinator wrote a quality scratchpad if multi-agent work occurred."""
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

# Markov-completeness check — coordinator tier is fail-CLOSED
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import markov_check as _markov_check
    _MARKOV_CHECK_AVAILABLE = True
except Exception as _mc_exc:
    _MARKOV_CHECK_AVAILABLE = False
    print(f"stop: markov_check import failed: {_mc_exc}", file=sys.stderr)

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"
COORDINATORS_DIR = Path(__file__).resolve().parents[1] / "coordinators"

MIN_LINES = 5  # Match subagent-stop quality bar


def update_temporal_prior() -> None:
    """Fire temporal prior EMA update if this session was valuated.

    Reads the supra session file. If last_attuned is set, records the
    session's dimension values as a temporal observation. This fires once
    per session exit — one observation per supra session, keyed by segment.

    Silently no-ops on any error (fail open).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import coordinator_registry as cr
        import supra_reader

        # Read this terminal's identity
        identity_id = cr.read_ppid_identity(COORDINATORS_DIR)
        if not identity_id:
            return

        # Read supra session states
        states = supra_reader.read_supra_session_states(identity_id)
        if not states:
            return

        # Only update temporal prior if the session was actually valuated
        last_attuned = states.get("last_attuned")
        if not last_attuned:
            print("stop: session not valuated, skipping temporal prior update", file=sys.stderr)
            return

        # Determine temporal segment from the supra session data
        segment = states.get("temporal_segment") or supra_reader._temporal_segment_key()

        # Record the observation
        success = supra_reader.record_temporal_observation(states, segment)
        if success:
            print(f"stop: temporal prior updated for segment '{segment}'", file=sys.stderr)
        else:
            print(f"stop: temporal prior update failed for segment '{segment}'", file=sys.stderr)
    except Exception as exc:
        print(f"stop: temporal prior update failed: {exc}", file=sys.stderr)


def deregister_coordinator() -> None:
    """Mark this session's claim as ended and clean up stale claims.

    /clear is a context flush, not a lifecycle event — the terminal identity
    file is left intact so the next SessionStart re-injects the same name.
    The terminal file is only archived when its PID dies (cleanup_stale_ppid_files).

    Silently no-ops on any error (fail open).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import coordinator_registry as cr

        # Read identity from terminal file
        identity_id = cr.read_ppid_identity(COORDINATORS_DIR)
        if not identity_id:
            print("stop: no terminal identity found, skipping coordinator deregistration", file=sys.stderr)
            return

        # Optionally write a farewell message (only if other claim files exist)
        other_claims = [
            c for c in cr.read_all_claims(COORDINATORS_DIR)
            if c.get("session_id") != identity_id and not cr.is_stale(c)
        ]
        if other_claims:
            cr.write_message(COORDINATORS_DIR, {
                "from": identity_id,
                "to": "all",
                "at": datetime.now().isoformat(timespec="seconds"),
                "type": "done",
                "text": f"Session {identity_id} ended. Check git log for changes made this session.",
            })

        # Mark this session's claim as ended (does not delete the file).
        # Terminal identity file is preserved — /clear is a context flush, not
        # a lifecycle event. cleanup_stale_ppid_files archives it when PID dies.
        cr.delete_claim(COORDINATORS_DIR, identity_id)

        # Clean up crashed sessions (2h+ old)
        cr.cleanup_stale(COORDINATORS_DIR, threshold_hours=2)

        print(f"stop: coordinator deregistered (identity={identity_id})", file=sys.stderr)

    except Exception as exc:
        print(f"stop: coordinator deregistration failed: {exc}", file=sys.stderr)


def main() -> None:
    try:
        hook_input = json.loads(sys.stdin.read() or "{}")
        print(f"Stop hook input: {json.dumps(hook_input)}", file=sys.stderr)

        # Fire temporal prior update before deregistering
        update_temporal_prior()

        # Deregister coordinator claim -- do this regardless of scratchpad check outcome
        deregister_coordinator()

        today = date.today().isoformat()

        # Accept scratchpad from whichever mode the session was in:
        # valuate (static graph) or coordinator/niche (dynamic graph)
        accepted_scratchpads = ["valuate", "coordinator"]
        found_scratchpad = None
        for sp_name in accepted_scratchpads:
            sp_dir = SCRATCHPAD_ROOT / sp_name
            # Check exact date match first, then session-keyed files ({date}-{session_id}.md)
            sp_path = sp_dir / f"{today}.md"
            if sp_path.exists():
                found_scratchpad = (sp_name, sp_path)
                break
            if sp_dir.is_dir():
                session_keyed = sorted(sp_dir.glob(f"{today}-*.md"))
                if session_keyed:
                    found_scratchpad = (sp_name, session_keyed[-1])
                    break

        # Check if any subagent scratchpads were written today (for richer error message)
        subagent_scratchpads = []
        if SCRATCHPAD_ROOT.is_dir():
            for agent_dir in SCRATCHPAD_ROOT.iterdir():
                if not agent_dir.is_dir() or agent_dir.name in accepted_scratchpads:
                    continue
                scratchpad = agent_dir / f"{today}.md"
                if scratchpad.exists() or list(agent_dir.glob(f"{today}-*.md")):
                    subagent_scratchpads.append(agent_dir.name)

        if not found_scratchpad:
            agents_note = f" (agents active: {', '.join(subagent_scratchpads)})" if subagent_scratchpads else ""
            json.dump({
                "decision": "block",
                "reason": (
                    f"Session scratchpad missing{agents_note}. "
                    f"Write to .claude/scratchpad/valuate/{today}.md (if valuating) "
                    f"or .claude/scratchpad/coordinator/{today}.md (if in niche). "
                    f"Include what you did, decisions made, and unresolved items."
                ),
            }, sys.stdout)
            return

        sp_name, sp_path = found_scratchpad

        # Check scratchpad has meaningful content (same bar as subagent-stop)
        lines = sp_path.read_text(encoding="utf-8").splitlines()
        non_empty = [l for l in lines if l.strip()]

        if len(non_empty) < MIN_LINES:
            json.dump({
                "decision": "block",
                "reason": (
                    f"{sp_name} scratchpad exists but is too short "
                    f"({len(non_empty)} non-empty lines, need {MIN_LINES}+). "
                    f"Please add a meaningful summary."
                ),
            }, sys.stdout)
            return

        # Markov-completeness check — coordinator close-out is fail-CLOSED.
        # Only applies when a coordinator scratchpad EXISTS. If no scratchpad
        # was found at all, the earlier block already blocked above — so here
        # we only run when sp_name == "coordinator" (not "valuate").
        if sp_name == "coordinator" and _MARKOV_CHECK_AVAILABLE:
            missing = _markov_check.check_completeness(sp_path, agent_type="coordinator")
            if missing:
                print(
                    f"[markov_check BLOCK] coordinator close-out missing: {', '.join(missing)}",
                    file=sys.stderr,
                )
                print(f"  scratchpad: {sp_path}", file=sys.stderr)
                json.dump({
                    "decision": "block",
                    "reason": (
                        f"Coordinator scratchpad is missing Contract 1 items: "
                        f"{', '.join(missing)}. "
                        f"Complete the Markov-completeness close-out at {sp_path} "
                        f"before the session ends. See .claude/rules/multi-agent-protocol.md "
                        f"Markov-Completeness Contract section."
                    ),
                }, sys.stdout)
                return

        print(
            f"Stop: {sp_name} scratchpad OK ({len(non_empty)} lines), allowing",
            file=sys.stderr,
        )
        json.dump({}, sys.stdout)

    except Exception as e:
        print(f"Stop hook error: {e}", file=sys.stderr)
        json.dump({}, sys.stdout)  # Fail open


if __name__ == "__main__":
    main()
