#!/usr/bin/env python3
"""SubagentStart hook: inject scratchpad protocol + coordinator/ego context + own continuity into every specialist."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"

# Lines of context to inject from each scratchpad source
CONTEXT_LINES = 5


def last_lines(path: Path, n: int = CONTEXT_LINES) -> str:
    """Return last n non-empty lines from a file, or empty string if missing."""
    if not path.exists():
        return ""
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def latest_entry(agent_dir: Path) -> Path | None:
    """Find the most recent YYYY-MM-DD.md file in an agent's scratchpad directory."""
    if not agent_dir.is_dir():
        return None
    entries = sorted(agent_dir.glob("????-??-??.md"), reverse=True)
    return entries[0] if entries else None


def staleness_warning(entry: Path | None, label: str) -> str | None:
    """Return a warning string if the entry is >3 days old, else None."""
    if not entry:
        return None
    try:
        entry_date = date.fromisoformat(entry.stem)
        days_old = (date.today() - entry_date).days
        if days_old > 3:
            return f"  (stale: {label} is {days_old} days old -- treat as low-confidence)"
    except ValueError:
        pass
    return None


def main() -> None:
    hook_input = json.loads(sys.stdin.read())
    agent_type = hook_input.get("agent_type", "unknown")
    today = date.today().isoformat()

    # Build context injection
    parts = [
        f"## Scratchpad Protocol (auto-injected by SubagentStart hook)",
        f"**Today's date**: {today}",
        f"**Your agent type**: {agent_type}",
        f"**Your scratchpad path**: `.claude/scratchpad/{agent_type}/{today}.md`",
        "",
        "Before returning, you MUST write a scratchpad entry containing:",
        "- **What I did**: actions taken, files modified, decisions made",
        "- **Cross-agent observations**: what you read from other agents, what was useful/confusing",
        "- **Unresolved**: open questions, things needing follow-up",
        "",
        "If a scratchpad entry for today already exists, UPDATE it in place (consolidate, don't append).",
        "Keep entries under 80 lines. If you need more, you're writing too much detail.",
    ]

    # Inject ego's latest assessment
    ego_entry = latest_entry(SCRATCHPAD_ROOT / "ego")
    if ego_entry:
        ego_tail = last_lines(ego_entry)
        if ego_tail:
            parts.extend(["", f"### Ego's latest assessment ({ego_entry.name}):", ego_tail])
            stale = staleness_warning(ego_entry, "ego assessment")
            if stale:
                parts.append(stale)

    # Inject coordinator's latest entry
    coord_entry = latest_entry(SCRATCHPAD_ROOT / "coordinator")
    if coord_entry:
        coord_tail = last_lines(coord_entry)
        if coord_tail:
            parts.extend(["", f"### Coordinator's latest entry ({coord_entry.name}):", coord_tail])
            stale = staleness_warning(coord_entry, "coordinator entry")
            if stale:
                parts.append(stale)

    # Inject the specialist's OWN most recent scratchpad for behavioral anchoring / continuity
    own_entry = latest_entry(SCRATCHPAD_ROOT / agent_type)
    if own_entry:
        own_tail = last_lines(own_entry)
        if own_tail:
            parts.extend(["", f"### Your last scratchpad ({own_entry.name}):", own_tail])
            stale = staleness_warning(own_entry, "your last entry")
            if stale:
                parts.append(stale)

    context = "\n".join(parts)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SubagentStart",
            "additionalContext": context,
        }
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
