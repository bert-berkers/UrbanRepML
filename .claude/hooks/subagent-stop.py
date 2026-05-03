#!/usr/bin/env python3
"""SubagentStop hook: verify subagent wrote its scratchpad (min 5 lines, warn on >80 bloat)."""
import json
import sys
from datetime import date
from pathlib import Path

# Markov-completeness check — fail-OPEN (warn only, never block)
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import markov_check as _markov_check
    _MARKOV_CHECK_AVAILABLE = True
except Exception as _mc_exc:
    _MARKOV_CHECK_AVAILABLE = False
    print(f"[markov_check WARN] import failed: {_mc_exc}", file=sys.stderr)

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"
COORDINATORS_DIR = Path(__file__).resolve().parents[1] / "coordinators"

MIN_LINES = 5
MAX_LINES = 80  # Per multi-agent-protocol.md rule

# Q4 claim-narrowing: sessions older than this (minutes) should have narrowed claimed_paths
CLAIM_NARROWING_THRESHOLD_MINUTES = 15


def check_claim_narrowing(coordinators_dir: Path) -> None:
    """Warn if any active session still has claimed_paths=['*'] past the first OODA cycle.

    Detection: session age > CLAIM_NARROWING_THRESHOLD_MINUTES AND claimed_paths == ['*'].
    Fail mode: warning to stderr only, never blocks. Silently no-ops on any error.
    Per .claude/rules/coordinator-coordination.md Anti-Patterns: claim squatting.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import coordinator_registry as cr
        from datetime import datetime, timedelta

        claims = cr.read_all_claims(coordinators_dir, include_ended=False)
        threshold = timedelta(minutes=CLAIM_NARROWING_THRESHOLD_MINUTES)
        now = datetime.now()

        for claim in claims:
            claimed_paths = claim.get("claimed_paths", [])
            # Check for wildcard squatting: ['*'] is the sentinel
            if claimed_paths != ["*"]:
                continue

            session_id = claim.get("session_id", "unknown")
            started_at_str = claim.get("started_at")
            if not started_at_str:
                continue

            try:
                started_at = datetime.fromisoformat(str(started_at_str))
            except (ValueError, TypeError):
                continue

            age = now - started_at
            if age > threshold:
                age_min = int(age.total_seconds() / 60)
                print(
                    f"\u26a0 claim-narrowing: session {session_id!r} still has "
                    f"claimed_paths=['*'] after {age_min}m. "
                    f"Per .claude/rules/coordinator-coordination.md — "
                    f"narrow to actual working paths.",
                    file=sys.stderr,
                )
    except Exception as exc:
        # Fail silently — never let the check break sessions
        print(f"SubagentStop: claim-narrowing check failed: {exc}", file=sys.stderr)


def touch_heartbeat() -> None:
    """Update the coordinator's heartbeat_at to now.

    Silently no-ops on any error (fail open).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import coordinator_registry as cr

        identity_id = cr.read_ppid_identity(COORDINATORS_DIR)
        if not identity_id:
            return

        cr.update_heartbeat(COORDINATORS_DIR, identity_id)
        print(f"SubagentStop: heartbeat updated for {identity_id}", file=sys.stderr)
    except Exception as exc:
        print(f"SubagentStop: heartbeat update failed: {exc}", file=sys.stderr)


def main() -> None:
    try:
        hook_input = json.loads(sys.stdin.read() or "{}")
        print(f"SubagentStop hook input: {json.dumps(hook_input)}", file=sys.stderr)

        # Update coordinator heartbeat on every subagent completion
        touch_heartbeat()

        # Q4 claim-narrowing check (warn only, fail-open)
        check_claim_narrowing(COORDINATORS_DIR)

        # Try multiple possible field names for agent type (runtime API may vary)
        agent_type = None
        for key in ("agent_type", "subagent_type", "type", "agentType"):
            if key in hook_input:
                agent_type = hook_input[key]
                break

        if not agent_type:
            print("SubagentStop: no agent_type found, allowing", file=sys.stderr)
            json.dump({}, sys.stdout)
            return

        today = date.today().isoformat()
        sp_dir = SCRATCHPAD_ROOT / agent_type
        date_only_path = sp_dir / f"{today}.md"
        session_keyed_paths = sorted(sp_dir.glob(f"{today}-*.md")) if sp_dir.is_dir() else []

        # Accept date-only OR session-keyed scratchpad (per multi-agent-protocol.md).
        # Prefer session-keyed when available — that is the canonical form.
        if session_keyed_paths:
            scratchpad_path = session_keyed_paths[-1]
        elif date_only_path.exists():
            scratchpad_path = date_only_path
        else:
            json.dump({
                "decision": "block",
                "reason": (
                    f"Subagent {agent_type} did not write scratchpad at "
                    f".claude/scratchpad/{agent_type}/{today}.md or {today}-*.md"
                ),
            }, sys.stdout)
            return

        # Check meaningful content (min lines)
        lines = scratchpad_path.read_text(encoding="utf-8").splitlines()
        non_empty = [l for l in lines if l.strip()]

        if len(non_empty) <= MIN_LINES:
            json.dump({
                "decision": "block",
                "reason": (
                    f"Subagent {agent_type} scratchpad at "
                    f".claude/scratchpad/{agent_type}/{today}.md "
                    f"is too short ({len(non_empty)} lines, need >{MIN_LINES})"
                ),
            }, sys.stdout)
            return

        # Check bloat (warn, don't block -- allow the stop but log)
        if len(non_empty) > MAX_LINES:
            print(
                f"SubagentStop WARNING: {agent_type} scratchpad is {len(non_empty)} lines "
                f"(>{MAX_LINES} limit). Consider consolidating.",
                file=sys.stderr,
            )

        # Record agent death in timer registry
        try:
            _hooks = str(Path(__file__).resolve().parent)
            if _hooks not in sys.path:
                sys.path.insert(0, _hooks)
            import agent_timer
            obituary = agent_timer.death(agent_type)
            if obituary:
                lived = obituary.get("lived_min", "?")
                tokens = obituary.get("estimated_tokens_used", "?")
                print(
                    f"SubagentStop: {agent_type} died after {lived}m (~{tokens} tokens)",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"SubagentStop: timer death failed: {exc}", file=sys.stderr)

        # Markov-completeness check (Contract 1, specialist tier, fail-OPEN)
        # scratchpad_path is already session-keyed when available (resolved above).
        try:
            if _MARKOV_CHECK_AVAILABLE:
                missing = _markov_check.check_completeness(scratchpad_path, agent_type="specialist")
                if missing:
                    print(
                        f"[markov_check WARN] {agent_type} scratchpad missing: {', '.join(missing)}",
                        file=sys.stderr,
                    )
        except Exception as _mc_err:
            print(f"[markov_check WARN] check failed: {_mc_err}", file=sys.stderr)
        # Never raise, never block — specialist tier is fail-OPEN.

        print(
            f"SubagentStop: {agent_type} wrote scratchpad ({len(non_empty)} lines), allowing",
            file=sys.stderr,
        )
        json.dump({}, sys.stdout)

    except Exception as e:
        print(f"SubagentStop hook error: {e}", file=sys.stderr)
        json.dump({}, sys.stdout)  # Fail open


if __name__ == "__main__":
    main()
