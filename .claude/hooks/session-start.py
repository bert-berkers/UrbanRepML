#!/usr/bin/env python3
"""SessionStart hook: warm-start the coordinator with scratchpad context and recent git history."""
import json
import subprocess
import sys
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"


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


def main() -> None:
    hook_input = json.loads(sys.stdin.read())
    source = hook_input.get("source", "")

    # Only inject on fresh startup, not resume/compact
    if source not in ("startup", ""):
        json.dump({}, sys.stdout)
        return

    parts = ["## Session Orientation (auto-injected by SessionStart hook)"]

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
