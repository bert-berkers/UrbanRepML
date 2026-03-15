"""
Re-key session/supra files from old Claude Code PPID (32684) to terminal shell PID.

Purpose: Fix session identity files after switching from Claude Code PPID to terminal
         shell PID as the session key. One-time migration for session silver-watching-shore.
Lifetime: temporary (one-off migration, safe to delete after 2026-04-15)
Stage: infrastructure / session identity
"""

import os
import psutil

BASE = "C:/Users/Bert Berkers/PycharmProjects/UrbanRepML"
OLD_PID = "32684"

# Determine terminal shell PID by walking up the process tree to node, then taking its parent
proc = psutil.Process(os.getpid())
terminal_pid = None
while proc:
    name = proc.name().lower()
    if name.startswith("node"):
        terminal_pid = proc.parent().pid
        break
    try:
        proc = proc.parent()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        break

if terminal_pid is None:
    print("ERROR: Could not determine terminal shell PID — no 'node' process found in parent chain.")
    raise SystemExit(1)

print(f"Terminal shell PID: {terminal_pid}")
print(f"Old PPID to replace: {OLD_PID}")
print()

FILES = [
    (
        f"{BASE}/.claude/coordinators/sessions/silver-watching-shore.{OLD_PID}",
        f"{BASE}/.claude/coordinators/sessions/silver-watching-shore.{terminal_pid}",
    ),
    (
        f"{BASE}/.claude/coordinators/supra/silver-watching-shore-2026-03-15.{OLD_PID}",
        f"{BASE}/.claude/coordinators/supra/silver-watching-shore-2026-03-15.{terminal_pid}",
    ),
    (
        f"{BASE}/.claude/supra/sessions/silver-watching-shore-2026-03-15.{OLD_PID}.yaml",
        f"{BASE}/.claude/supra/sessions/silver-watching-shore-2026-03-15.{terminal_pid}.yaml",
    ),
]

for old_path, new_path in FILES:
    if not os.path.exists(old_path):
        print(f"SKIP (not found): {old_path}")
        continue
    if os.path.exists(new_path):
        print(f"SKIP (target already exists): {new_path}")
        continue
    os.rename(old_path, new_path)
    print(f"RENAMED: {old_path}")
    print(f"     ->  {new_path}")

print()
print("Done.")
