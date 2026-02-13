#!/usr/bin/env python3
"""SubagentStop hook: verify subagent wrote its scratchpad."""
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
        print(f"SubagentStop hook input: {json.dumps(hook_input)}", file=sys.stderr)

        # Try multiple possible field names for agent type
        agent_type = None
        for key in ["agent_type", "subagent_type", "type", "agentType"]:
            if key in hook_input:
                agent_type = hook_input[key]
                break

        # If we can't determine agent type, allow (fail open)
        if not agent_type:
            print("SubagentStop: no agent_type found, allowing", file=sys.stderr)
            json.dump({}, sys.stdout)
            return

        # Check if scratchpad file exists
        today = date.today().isoformat()
        scratchpad_path = SCRATCHPAD_ROOT / agent_type / f"{today}.md"

        if not scratchpad_path.exists():
            json.dump({
                "decision": "block",
                "reason": f"Subagent {agent_type} did not write scratchpad at .claude/scratchpad/{agent_type}/{today}.md"
            }, sys.stdout)
            return

        # Check if file has meaningful content (>5 lines)
        lines = scratchpad_path.read_text(encoding="utf-8").splitlines()
        non_empty_lines = [l for l in lines if l.strip()]

        if len(non_empty_lines) <= 5:
            json.dump({
                "decision": "block",
                "reason": f"Subagent {agent_type} scratchpad at .claude/scratchpad/{agent_type}/{today}.md is too short ({len(non_empty_lines)} lines)"
            }, sys.stdout)
            return

        # All good, allow
        print(f"SubagentStop: {agent_type} wrote scratchpad ({len(non_empty_lines)} lines), allowing", file=sys.stderr)
        json.dump({}, sys.stdout)

    except Exception as e:
        # Fail open on error
        print(f"SubagentStop hook error: {e}", file=sys.stderr)
        json.dump({}, sys.stdout)


if __name__ == "__main__":
    main()
