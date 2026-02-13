#!/usr/bin/env python3
"""Stop hook: verify coordinator wrote scratchpad if multi-agent work occurred."""
import json
import sys
from datetime import date
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"


def main() -> None:
    try:
        # Parse hook input (may be empty)
        hook_input = json.loads(sys.stdin.read() or "{}")

        # Log raw input for debugging
        print(f"Stop hook input: {json.dumps(hook_input)}", file=sys.stderr)

        today = date.today().isoformat()

        # Check if any subagent scratchpads were written today
        # (excludes coordinator directory itself)
        subagent_scratchpads = []
        for agent_dir in SCRATCHPAD_ROOT.iterdir():
            if not agent_dir.is_dir():
                continue
            if agent_dir.name == "coordinator":
                continue

            scratchpad = agent_dir / f"{today}.md"
            if scratchpad.exists():
                subagent_scratchpads.append(agent_dir.name)

        # If no subagent work happened today, allow
        if not subagent_scratchpads:
            print("Stop: no subagent scratchpads today, allowing", file=sys.stderr)
            json.dump({}, sys.stdout)
            return

        # Multi-agent coordination happened, check coordinator scratchpad
        coordinator_scratchpad = SCRATCHPAD_ROOT / "coordinator" / f"{today}.md"

        if not coordinator_scratchpad.exists():
            json.dump({
                "decision": "block",
                "reason": f"Multi-agent work occurred today (agents: {', '.join(subagent_scratchpads)}), but coordinator scratchpad is missing at .claude/scratchpad/coordinator/{today}.md"
            }, sys.stdout)
            return

        # Coordinator scratchpad exists, allow
        print(f"Stop: coordinator scratchpad exists, allowing", file=sys.stderr)
        json.dump({}, sys.stdout)

    except Exception as e:
        # Fail open on error
        print(f"Stop hook error: {e}", file=sys.stderr)
        json.dump({}, sys.stdout)


if __name__ == "__main__":
    main()
