#!/usr/bin/env python3
"""SessionStart hook: warm-start the coordinator with scratchpad context, critical signals, and recent git history."""
import json
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"
COORDINATORS_DIR = Path(__file__).resolve().parents[1] / "coordinators"

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
    """Find the most recent dated .md file in an agent's scratchpad directory.

    Matches both daily entries (YYYY-MM-DD.md) and suffixed entries
    (YYYY-MM-DD-forward-look.md, etc.).
    """
    if not agent_dir.is_dir():
        return None
    entries = sorted(agent_dir.glob("????-??-??*.md"), reverse=True)
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
                if re.search(r'\b' + keyword + r'\b', content_upper):
                    for line in content.splitlines():
                        if re.search(r'\b' + keyword + r'\b', line.upper()) and line.strip():
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
        entry_date = date.fromisoformat(entry.stem[:10])
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

        # Cleanup stale PPID files from dead processes
        cr.cleanup_stale_ppid_files(COORDINATORS_DIR)

        # Archive old messages into per-day subdirs
        cr.archive_old_messages(COORDINATORS_DIR)

        # Archive old singleton files (migration — clean break)
        archive_dir = COORDINATORS_DIR / "archive"
        for old_singleton in (".current_session_id", ".current_supra_session_id", ".active_graph"):
            old_path = COORDINATORS_DIR / old_singleton
            if old_path.exists():
                archive_dir.mkdir(exist_ok=True)
                old_path.rename(archive_dir / old_singleton)

        # Read existing claims BEFORE writing our own
        existing_claims = cr.read_all_claims(COORDINATORS_DIR)
        active_claims = [c for c in existing_claims if not cr.is_stale(c)]

        # Check if this terminal already has a session (survives /clear)
        existing_session = cr.read_ppid_session(COORDINATORS_DIR)
        now = datetime.now().isoformat(timespec="seconds")

        if existing_session:
            # Same terminal after /clear — reuse session identity
            session_id = existing_session
            # Update heartbeat on existing claim (if claim file exists)
            cr.update_heartbeat(COORDINATORS_DIR, session_id)
        else:
            # Fresh terminal — generate new session ID
            session_id = cr.generate_session_id(COORDINATORS_DIR)

            # Write initial claim (broad -- coordinator will narrow after user states task)
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

            # Persist session_id as PPID-keyed file
            cr.write_ppid_session(COORDINATORS_DIR, session_id)

        # Cleanup obviously stale sessions (2h+ old) from prior crashes
        cr.cleanup_stale(COORDINATORS_DIR, threshold_hours=2)

    except Exception as exc:
        print(f"session-start: coordinator registration failed: {exc}", file=sys.stderr)
        return "", []

    # Compute and register supra session identity
    # Supra = this terminal. Persists across /clear cycles.
    # If a PPID-keyed supra file exists, this is a /clear cycle — join the existing supra.
    # If not, this is a fresh terminal — create a new supra named after this coordinator.
    try:
        import supra_reader
        import yaml
        import os as _os

        sessions_dir = Path(__file__).resolve().parents[1] / "supra" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        existing_supra_sid = cr.read_ppid_supra(COORDINATORS_DIR)

        if existing_supra_sid:
            # Join existing supra session (same terminal, after /clear)
            supra_session_id = existing_supra_sid
            supra_session_path = sessions_dir / f"{supra_session_id}.yaml"
            if supra_session_path.exists():
                existing = yaml.safe_load(supra_session_path.read_text(encoding="utf-8")) or {}
                coordinators_list = existing.get("coordinators", [])
                if session_id not in coordinators_list:
                    coordinators_list.append(session_id)
                    existing["coordinators"] = coordinators_list
                    tmp = supra_session_path.with_suffix(".yaml.tmp")
                    tmp.write_text(
                        yaml.dump(existing, default_flow_style=False, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                    )
                    _os.replace(str(tmp), str(supra_session_path))
        else:
            # Fresh terminal — create new supra named {poetic_name}-{date}
            supra_session_id = f"{session_id}-{date.today().isoformat()}"

            # Bootstrap from temporal prior or global states
            temporal_prior = supra_reader.get_temporal_prior()
            if temporal_prior:
                default_states = supra_reader.temporal_prior_to_states(temporal_prior)
            else:
                default_states = supra_reader.read_states()

            segment_key = supra_reader._temporal_segment_key()
            supra_data = {
                "supra_session_id": supra_session_id,
                "temporal_segment": segment_key,
                "date": date.today().isoformat(),
                "created_at": now,
                "last_attuned": None,
                "coordinators": [session_id],
                "mode": default_states.get("mode", "exploratory"),
                "dimensions": dict(default_states.get("dimensions", {})),
                "focus": list(default_states.get("focus", [])),
                "suppress": list(default_states.get("suppress", [])),
            }
            supra_session_path = sessions_dir / f"{supra_session_id}.yaml"
            tmp = supra_session_path.with_suffix(".yaml.tmp")
            tmp.write_text(
                yaml.dump(supra_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            _os.replace(str(tmp), str(supra_session_path))

        # Write PPID-keyed supra file
        cr.write_ppid_supra(COORDINATORS_DIR, supra_session_id)

    except Exception as exc:
        print(f"session-start: supra session registration failed: {exc}", file=sys.stderr)

    return session_id, active_claims


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

    # Inject full orientation context on startup AND after /clear (which sends
    # "compact" or "resume"). After /clear the coordinator loses all context, so
    # re-injection is essential — especially for the /clear → /coordinate plan workflow.
    # Only skip on sources that genuinely don't need it (none currently identified).


    parts = ["## Session Orientation (auto-injected by SessionStart hook)"]

    # Cognitive light cone metrics (Levin's embodied constraints)
    light_cone = cognitive_light_cone_summary()
    parts.append(f"\n**{light_cone}**")

    # Active coordinators (injected first -- most urgent info)
    if session_id:
        parts.append(f"\n**Your coordinator session ID**: `{session_id}`")
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            import coordinator_registry as cr
            supra_sid = cr.read_ppid_supra(COORDINATORS_DIR)
            if supra_sid:
                parts.append(f"**Supra session**: `{supra_sid}` (this terminal's supra identity, persists across /clear)")
        except Exception:
            pass
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

    # Supra attentional landscape (human's current precision weights)
    try:
        # Ensure hooks dir is on sys.path for supra_reader import
        _hooks_dir = str(Path(__file__).resolve().parent)
        if _hooks_dir not in sys.path:
            sys.path.insert(0, _hooks_dir)
        import supra_reader
        supra_states = supra_reader.read_session_states()
        schema = supra_reader.read_schema()
        if supra_states and schema:
            landscape = supra_reader.format_for_coordinator(supra_states, schema)
            if landscape:
                parts.extend(["", "### Human's Attentional Landscape:", landscape])
    except Exception as exc:
        print(f"session-start: supra state read failed: {exc}", file=sys.stderr)

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
