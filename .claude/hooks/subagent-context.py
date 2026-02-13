#!/usr/bin/env python3
"""SubagentStart hook: inject scratchpad protocol + coordinator/ego context into every specialist."""
import json
import sys
from datetime import date
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"


def last_lines(path: Path, n: int = 3) -> str:
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
    ]

    # Inject ego's latest assessment (last 3 lines)
    ego_entry = latest_entry(SCRATCHPAD_ROOT / "ego")
    if ego_entry:
        ego_tail = last_lines(ego_entry)
        if ego_tail:
            parts.extend(["", "### Ego's latest assessment (tail):", ego_tail])

    # Inject coordinator's latest entry (last 3 lines)
    coord_entry = latest_entry(SCRATCHPAD_ROOT / "coordinator")
    if coord_entry:
        coord_tail = last_lines(coord_entry)
        if coord_tail:
            parts.extend(["", "### Coordinator's latest entry (tail):", coord_tail])

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
