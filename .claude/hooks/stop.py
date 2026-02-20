#!/usr/bin/env python3
"""Stop hook: verify coordinator wrote a quality scratchpad if multi-agent work occurred."""
import json
import sys
from datetime import date
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"

MIN_LINES = 5  # Match subagent-stop quality bar


def main() -> None:
    try:
        hook_input = json.loads(sys.stdin.read() or "{}")
        print(f"Stop hook input: {json.dumps(hook_input)}", file=sys.stderr)

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
