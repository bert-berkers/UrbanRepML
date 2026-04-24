#!/usr/bin/env python3
"""SubagentStop hook: auto-pulse /sync between waves when in dynamic mode.

Triggers a lightweight lateral broadcast each time a subagent returns to the
coordinator IF:
  - lateral coupling is active (supra active_graph == 'dynamic')
  - last sync pulse for this terminal was >= MIN_INTERVAL_SECONDS ago

This implements the between-wave observation step from the /niche protocol
(SKILL.md §84-87) at the hook layer, so coordinators don't have to remember
to scan messages between waves.

Fail-open: any error is swallowed. Never blocks subagent completion.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

MIN_INTERVAL_SECONDS = 300  # 5 minutes — avoid spamming on rapid wave bursts

ROOT = Path(__file__).resolve().parents[2]
HOOKS = Path(__file__).resolve().parent
TIMERS = ROOT / ".claude" / "timers"
SYNC_RUN = ROOT / ".claude" / "sync_run.py"


def main() -> int:
    try:
        sys.path.insert(0, str(HOOKS))
        import supra_reader

        if not supra_reader.is_lateral_coupling_active():
            return 0

        TIMERS.mkdir(parents=True, exist_ok=True)
        # Per-terminal throttle file (PID-keyed; falls back to PPID then "shared")
        pid = os.environ.get("CLAUDE_TERMINAL_PID") or str(os.getppid()) or "shared"
        stamp = TIMERS / f"between_waves_sync.{pid}.stamp"
        now = time.time()
        if stamp.exists():
            age = now - stamp.stat().st_mtime
            if age < MIN_INTERVAL_SECONDS:
                return 0

        if not SYNC_RUN.exists():
            return 0

        # Fire-and-forget; cap runtime so we never block the hook
        subprocess.run(
            [sys.executable, str(SYNC_RUN), "between-waves auto-pulse"],
            cwd=str(ROOT),
            timeout=8,
            capture_output=True,
        )
        stamp.touch()
    except Exception as e:
        print(f"[between_waves_sync WARN] {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
