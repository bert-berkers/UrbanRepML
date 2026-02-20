#!/usr/bin/env python3
"""SessionStart hook: warm-start the coordinator with scratchpad context, critical signals, and recent git history."""
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"

# Keywords that indicate a specialist left critical notes needing attention
CRITICAL_KEYWORDS = ("BLOCKED", "URGENT", "CRITICAL", "BROKEN")


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
