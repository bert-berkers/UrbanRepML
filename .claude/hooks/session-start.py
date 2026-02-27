#!/usr/bin/env python3
"""SessionStart hook: warm-start the coordinator with scratchpad context, critical signals, and recent git history."""
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"
COORDINATORS_DIR = Path(__file__).resolve().parents[1] / "coordinators"
SESSION_ID_FILE = Path(__file__).resolve().parents[1] / "coordinators" / ".current_session_id"

# Signal vocabulary: keywords that indicate specialist left critical notes.
# Extended set supports Levin's "pervasive signaling" -- richer gradients between agents.
CRITICAL_KEYWORDS = (
    "BLOCKED", "URGENT", "CRITICAL", "BROKEN",
    "SHAPE_CHANGED", "INTERFACE_CHANGED", "DEPRECATED", "NEEDS_TEST",
)


def read_file(path: Path, max_lines: int = 30) -> str:
    """Read a file, returning at most max_lines lines."""
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    return "\n".join(lines)


def latest_entry(agent_dir: Path) -> Path | None:
    """Find the most recent YYYY-MM-DD.md file in an agent's scratchpad directory."""
    if not agent_dir.is_dir():
        return None
    entries = sorted(agent_dir.glob("????-??-??.md"), reverse=True)
    return entries[0] if entries else None


def git_log(n: int = 5) -> str:
    """Get recent git log, returning empty string on failure."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{n}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def scan_critical_signals() -> list[str]:
    """Scan specialist scratchpads (today + yesterday) for BLOCKED/URGENT/CRITICAL keywords."""
    today_str = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    signals = []

    for agent_dir in sorted(SCRATCHPAD_ROOT.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name in ("coordinator", "ego"):
            continue  # Already surfaced separately

        for date_str in (today_str, yesterday_str):
            entry = agent_dir / f"{date_str}.md"
            if not entry.exists():
                continue
            try:
                content = entry.read_text(encoding="utf-8")
            except Exception:
                continue
            content_upper = content.upper()
            for keyword in CRITICAL_KEYWORDS:
                if keyword in content_upper:
                    for line in content.splitlines():
                        if keyword in line.upper() and line.strip():
                            signals.append(
                                f"- **{agent_dir.name}** ({date_str}): {line.strip()[:120]}"
                            )
                            break
                    break  # One signal per agent per date is enough
    return signals


def cognitive_light_cone_summary() -> str:
    """Quantify the system's cognitive reach (Levin's embodied constraints).

    Returns a one-line summary of temporal depth, agent reach,
    unresolved items, and active coordinator count.
    """
    # Temporal depth: how many unique days of scratchpad history?
    all_entries = sorted(SCRATCHPAD_ROOT.rglob("????-??-??.md"))
    days = len(set(e.stem for e in all_entries))

    # Agent reach: how many agents have contributed scratchpads?
    agents = len([
        d for d in SCRATCHPAD_ROOT.iterdir()
        if d.is_dir() and any(d.glob("*.md"))
    ])

    # Unresolved items (forward projection quality signal)
    recent = all_entries[-5:] if all_entries else []
    unresolved = 0
    for e in recent:
        try:
            unresolved += e.read_text(encoding="utf-8").lower().count("unresolved")
        except Exception:
            pass

    # Active coordinator count (lateral reach)
    active_coords = len(list(COORDINATORS_DIR.glob("session-*.yaml")))

    return (
        f"Light cone: {days}d memory, {agents} agents, "
        f"~{unresolved} unresolved, {active_coords} active coordinators"
    )


def staleness_note(entry: Path | None, label: str) -> str | None:
    """Return a staleness warning if entry is >3 days old."""
    if not entry:
        return None
    try:
        entry_date = date.fromisoformat(entry.stem)
        days_old = (date.today() - entry_date).days
        if days_old > 3:
            return f"  (stale: {label} is {days_old} days old -- may not reflect current state)"
    except ValueError:
        pass
    return None


def register_coordinator() -> tuple[str, list[dict]]:
    """Register this session as an active coordinator.

    Returns (session_id, list_of_existing_active_claims).
    Silently no-ops on any error (fail open).
    """
    try:
        # Import here so a missing PyYAML doesn't crash the whole hook
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import coordinator_registry as cr

        COORDINATORS_DIR.mkdir(parents=True, exist_ok=True)

        # Read existing claims BEFORE writing our own
        existing_claims = cr.read_all_claims(COORDINATORS_DIR)
        active_claims = [c for c in existing_claims if not cr.is_stale(c)]

        # Generate a unique session ID
        session_id = cr.generate_session_id(COORDINATORS_DIR)

        # Write initial claim (broad -- coordinator will narrow after user states task)
        now = datetime.now().isoformat(timespec="seconds")
        claim_data = {
            "session_id": session_id,
            "started_at": now,
            "heartbeat_at": now,
            "task_summary": "Starting up -- task not yet specified",
            "domain": {
                "primary": "unknown",
                "description": "Coordinator registering; will narrow claimed_paths after task is known",
            },
            "claimed_paths": ["*"],
            "read_only_paths": [],
            "active_agents": [],
        }
        cr.write_claim(COORDINATORS_DIR, claim_data)

        # Persist session_id for stop hook
        SESSION_ID_FILE.write_text(session_id, encoding="utf-8")

        # Cleanup obviously stale sessions (2h+ old) from prior crashes
        cr.cleanup_stale(COORDINATORS_DIR, threshold_hours=2)

        return session_id, active_claims
    except Exception as exc:
        print(f"session-start: coordinator registration failed: {exc}", file=sys.stderr)
        return "", []


def format_active_coordinators(active_claims: list[dict]) -> list[str]:
    """Format active coordinator claims as human-readable lines."""
    if not active_claims:
        return []
    lines = ["", "### Active Coordinator Sessions:"]
    for claim in active_claims:
        sid = claim.get("session_id", "unknown")
        summary = claim.get("task_summary", "no summary")
        paths = claim.get("claimed_paths", [])
        heartbeat = claim.get("heartbeat_at", "unknown")
        paths_str = ", ".join(str(p) for p in paths[:5])
        if len(paths) > 5:
            paths_str += f" (+{len(paths) - 5} more)"
        lines.append(
            f"- **{sid}**: {summary}\n"
            f"  claiming: {paths_str}\n"
            f"  last heartbeat: {heartbeat}"
        )
    lines.append(
        "\nCheck claims before modifying shared files. "
        "See `.claude/rules/coordinator-coordination.md`."
    )
    return lines


def main() -> None:
    hook_input = json.loads(sys.stdin.read())
    source = hook_input.get("source", "")

    # Register coordinator regardless of source (handles resume/compact too)
    session_id, active_claims = register_coordinator()

    # Only inject full orientation context on fresh startup, not resume/compact
    if source not in ("startup", ""):
        json.dump({}, sys.stdout)
        return

    parts = ["## Session Orientation (auto-injected by SessionStart hook)"]

    # Cognitive light cone metrics (Levin's embodied constraints)
    light_cone = cognitive_light_cone_summary()
    parts.append(f"\n**{light_cone}**")

    # Active coordinators (injected first -- most urgent info)
    if session_id:
        parts.append(f"\n**Your coordinator session ID**: `{session_id}`")
        parts.append(
            "Narrow your `claimed_paths` in `.claude/coordinators/` after the user states the task."
        )
    coord_lines = format_active_coordinators(active_claims)
    parts.extend(coord_lines)

    # Most recent coordinator scratchpad
    coord_entry = latest_entry(SCRATCHPAD_ROOT / "coordinator")
    if coord_entry:
        content = read_file(coord_entry, max_lines=25)
        if content:
            parts.extend([
                "",
                f"### Coordinator's last entry ({coord_entry.name}):",
                content,
            ])
            stale = staleness_note(coord_entry, "coordinator entry")
            if stale:
                parts.append(stale)

    # Most recent ego forward-look
    ego_entry = latest_entry(SCRATCHPAD_ROOT / "ego")
    if ego_entry:
        content = read_file(ego_entry, max_lines=20)
        if content:
            parts.extend([
                "",
                f"### Ego's latest assessment ({ego_entry.name}):",
                content,
            ])
            stale = staleness_note(ego_entry, "ego assessment")
            if stale:
                parts.append(stale)

    # Critical signals from specialist scratchpads
    signals = scan_critical_signals()
    if signals:
        parts.extend(["", "### Critical Signals from Specialists:", *signals])

    # Recent git history
    log = git_log(5)
    if log:
        parts.extend(["", "### Recent git log:", log])

    context = "\n".join(parts)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
