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

        # One terminal = one identity. Read it (or generate it on fresh terminal).
        # /clear is a context flush, not a lifecycle event — same terminal,
        # same identity, fresh context. See specs/session-identity-architecture.md.
        identity_id = cr.read_ppid_identity(COORDINATORS_DIR)
        now = datetime.now().isoformat(timespec="seconds")
        is_fresh_terminal = identity_id is None

        if is_fresh_terminal:
            identity_id = cr.generate_identity_id(COORDINATORS_DIR)

            # Write initial claim (broad -- coordinator will narrow after user states task)
            claim_data = {
                "session_id": identity_id,  # claim files key on session_id for filename compat
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

            # Persist identity in terminal file
            cr.write_ppid_identity(COORDINATORS_DIR, identity_id)
        else:
            # Existing terminal (after /clear or hook re-fire) — heartbeat the claim.
            cr.update_heartbeat(COORDINATORS_DIR, identity_id)

        # Cleanup obviously stale sessions (2h+ old) from prior crashes
        cr.cleanup_stale(COORDINATORS_DIR, threshold_hours=2)

    except Exception as exc:
        print(f"session-start: coordinator registration failed: {exc}", file=sys.stderr)
        return "", []

    # Bootstrap supra valuation file at .claude/supra/sessions/{identity_id}.yaml
    # if absent. Identity IS the supra layer now — no separate concept.
    # On /clear (existing terminal), the file already exists; do nothing.
    try:
        import supra_reader
        import yaml
        import os as _os

        sessions_dir = Path(__file__).resolve().parents[1] / "supra" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Direct path keyed by identity (no date suffix on a fresh terminal —
        # identity already includes any required uniqueness from the poetic name).
        # Note: legacy files use {identity_id}-{date}.yaml form; supra_reader's
        # path resolver handles both. New writes use {identity_id}.yaml.
        supra_session_path = sessions_dir / f"{identity_id}.yaml"

        if is_fresh_terminal and not supra_session_path.exists():
            # Fresh terminal — bootstrap valuation from temporal prior or defaults
            temporal_prior = supra_reader.get_temporal_prior()
            if temporal_prior:
                default_states = supra_reader.temporal_prior_to_states(temporal_prior)
            else:
                default_states = supra_reader.read_states()

            segment_key = supra_reader._temporal_segment_key()
            supra_data = {
                "identity_id": identity_id,
                "supra_session_id": identity_id,  # transition compat
                "temporal_segment": segment_key,
                "date": date.today().isoformat(),
                "created_at": now,
                "last_attuned": None,
                "mode": default_states.get("mode", "exploratory"),
                "dimensions": dict(default_states.get("dimensions", {})),
                "focus": list(default_states.get("focus", [])),
                "suppress": list(default_states.get("suppress", [])),
            }
            tmp = supra_session_path.with_suffix(".yaml.tmp")
            tmp.write_text(
                yaml.dump(supra_data, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            _os.replace(str(tmp), str(supra_session_path))

    except Exception as exc:
        print(f"session-start: supra session bootstrap failed: {exc}", file=sys.stderr)

    return identity_id, active_claims


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
    identity_id, active_claims = register_coordinator()

    # Daily archive sweep — gated by .last_archive_sweep, fail-open
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import archive_sweep
        archive_sweep.maybe_run_sweep()
    except Exception as _sweep_exc:
        print(f"session-start: archive sweep failed: {_sweep_exc}", file=sys.stderr)

    # Inject full orientation context on startup AND after /clear (which sends
    # "compact" or "resume"). After /clear the coordinator loses all context, so
    # re-injection is essential — especially for the /clear → /coordinate plan workflow.
    # Only skip on sources that genuinely don't need it (none currently identified).


    parts = ["## Session Orientation (auto-injected by SessionStart hook)"]

    # Cognitive light cone metrics (Levin's embodied constraints)
    light_cone = cognitive_light_cone_summary()
    parts.append(f"\n**{light_cone}**")

    # Identity (injected first -- most urgent info)
    if identity_id:
        parts.append(f"\n**Your terminal identity**: `{identity_id}` (one identity per terminal, persists across /clear)")
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
            # Inject strategic intent if set — this is the terminal's mission from /valuate
            intent = supra_states.get("intent")
            if intent:
                parts.append(f"**Intent**: {intent}")
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
