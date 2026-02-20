#!/usr/bin/env python3
"""SubagentStop hook: verify subagent wrote its scratchpad (min 5 lines, warn on >80 bloat)."""
import json
import sys
from datetime import date
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"

MIN_LINES = 5
MAX_LINES = 80  # Per multi-agent-protocol.md rule


def main() -> None:
    try:
        hook_input = json.loads(sys.stdin.read() or "{}")
        print(f"SubagentStop hook input: {json.dumps(hook_input)}", file=sys.stderr)

        # Try multiple possible field names for agent type (runtime API may vary)
        agent_type = None
        for key in ("agent_type", "subagent_type", "type", "agentType"):
            if key in hook_input:
                agent_type = hook_input[key]
                break

        if not agent_type:
            print("SubagentStop: no agent_type found, allowing", file=sys.stderr)
            json.dump({}, sys.stdout)
            return

        today = date.today().isoformat()
        scratchpad_path = SCRATCHPAD_ROOT / agent_type / f"{today}.md"

        # Check existence
        if not scratchpad_path.exists():
            json.dump({
                "decision": "block",
                "reason": (
                    f"Subagent {agent_type} did not write scratchpad at "
                    f".claude/scratchpad/{agent_type}/{today}.md"
                ),
            }, sys.stdout)
            return

        # Check meaningful content (min lines)
        lines = scratchpad_path.read_text(encoding="utf-8").splitlines()
        non_empty = [l for l in lines if l.strip()]

        if len(non_empty) <= MIN_LINES:
            json.dump({
                "decision": "block",
                "reason": (
                    f"Subagent {agent_type} scratchpad at "
                    f".claude/scratchpad/{agent_type}/{today}.md "
                    f"is too short ({len(non_empty)} lines, need >{MIN_LINES})"
                ),
            }, sys.stdout)
            return

        # Check bloat (warn, don't block -- allow the stop but log)
        if len(non_empty) > MAX_LINES:
            print(
                f"SubagentStop WARNING: {agent_type} scratchpad is {len(non_empty)} lines "
                f"(>{MAX_LINES} limit). Consider consolidating.",
                file=sys.stderr,
            )

        print(
            f"SubagentStop: {agent_type} wrote scratchpad ({len(non_empty)} lines), allowing",
            file=sys.stderr,
        )
        json.dump({}, sys.stdout)

    except Exception as e:
        print(f"SubagentStop hook error: {e}", file=sys.stderr)
        json.dump({}, sys.stdout)  # Fail open


if __name__ == "__main__":
    main()
