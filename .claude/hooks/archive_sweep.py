#!/usr/bin/env python3
"""Daily archive sweep: stale supra sessions and old message directories.

Gated by .claude/coordinators/.last_archive_sweep (plain ISO-8601 timestamp).
If missing, treats as epoch zero (always runs). Rewrites the gate file after
every sweep attempt (even partial) to prevent thrashing on repeated failures.

Sweep rules (preserve-don't-delete per feedback_no_delete_data.md):
  1. Supra sessions: last_attuned > 30 days → move to supra/sessions/archive/
     - Skip if session is referenced by a live terminal (coordinators/terminals/*.yaml)
     - Skip if last_attuned missing or malformed (don't guess)
  2. Message dirs: coordinators/messages/YYYY-MM-DD/ older than 7 days
     → move to coordinators/messages/archive/YYYY-MM-DD/

Fail mode: any per-item error → log to stderr, continue. Never raises from run_sweep().
"""
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SWEEP_INTERVAL_HOURS = 24
SUPRA_STALE_DAYS = 30
MESSAGES_STALE_DAYS = 7

HOOKS_DIR = Path(__file__).resolve().parent
CLAUDE_DIR = HOOKS_DIR.parent
SUPRA_SESSIONS_DIR = CLAUDE_DIR / "supra" / "sessions"
COORDINATORS_DIR = CLAUDE_DIR / "coordinators"
GATE_FILE = COORDINATORS_DIR / ".last_archive_sweep"


def _read_gate() -> datetime:
    """Return the datetime of the last sweep, or epoch zero if missing/malformed."""
    if not GATE_FILE.exists():
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        text = GATE_FILE.read_text(encoding="utf-8").strip()
        dt = datetime.fromisoformat(text)
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _write_gate() -> None:
    """Write current UTC time to gate file."""
    try:
        GATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        GATE_FILE.write_text(now_str + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"[archive-sweep] WARN: could not write gate file: {exc}", file=sys.stderr)


def _read_live_supra_ids() -> set[str]:
    """Return set of supra_session_ids currently referenced by live terminals."""
    live = set()
    terminals_dir = COORDINATORS_DIR / "terminals"
    if not terminals_dir.is_dir():
        return live
    try:
        import yaml
    except ImportError:
        return live
    for terminal_file in terminals_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(terminal_file.read_text(encoding="utf-8")) or {}
            sid = data.get("supra_session_id")
            if sid:
                live.add(str(sid))
        except Exception as exc:
            print(
                f"[archive-sweep] WARN: could not read terminal file {terminal_file.name}: {exc}",
                file=sys.stderr,
            )
    return live


def _sweep_supra_sessions() -> tuple[int, int]:
    """Move stale supra session YAMLs to archive/.

    Returns (moved_count, skipped_count).
    Stale = last_attuned > SUPRA_STALE_DAYS days ago.
    Skipped if last_attuned missing, malformed, or session is live.
    """
    if not SUPRA_SESSIONS_DIR.is_dir():
        return 0, 0

    try:
        import yaml
    except ImportError:
        print("[archive-sweep] WARN: PyYAML not available, skipping supra sweep", file=sys.stderr)
        return 0, 0

    live_ids = _read_live_supra_ids()
    archive_dir = SUPRA_SESSIONS_DIR / "archive"
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=SUPRA_STALE_DAYS)
    moved = 0
    skipped = 0

    for yaml_file in SUPRA_SESSIONS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            print(
                f"[archive-sweep] WARN: malformed supra YAML {yaml_file.name}, skipping: {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        # Skip if referenced by a live terminal
        session_id = data.get("supra_session_id", yaml_file.stem)
        if session_id in live_ids:
            skipped += 1
            continue

        # Skip if last_attuned missing or malformed
        last_attuned_raw = data.get("last_attuned")
        if not last_attuned_raw:
            skipped += 1
            continue
        try:
            last_attuned = datetime.fromisoformat(str(last_attuned_raw))
            if last_attuned.tzinfo is None:
                last_attuned = last_attuned.replace(tzinfo=timezone.utc)
        except Exception:
            skipped += 1
            continue

        if last_attuned >= cutoff:
            # Not stale yet
            continue

        # Move to archive
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            dest = archive_dir / yaml_file.name
            shutil.move(str(yaml_file), str(dest))
            moved += 1
        except Exception as exc:
            print(
                f"[archive-sweep] WARN: could not move {yaml_file.name}: {exc}",
                file=sys.stderr,
            )
            skipped += 1

    return moved, skipped


def _sweep_message_dirs() -> tuple[int, int]:
    """Move old per-day message directories to messages/archive/.

    Returns (moved_count, skipped_count).
    Old = directory date > MESSAGES_STALE_DAYS days ago.
    Skips directories that don't parse as YYYY-MM-DD.
    """
    messages_dir = COORDINATORS_DIR / "messages"
    if not messages_dir.is_dir():
        return 0, 0

    cutoff_date = (datetime.now(tz=timezone.utc) - timedelta(days=MESSAGES_STALE_DAYS)).date()
    archive_root = messages_dir / "archive"
    moved = 0
    skipped = 0

    for entry in messages_dir.iterdir():
        if not entry.is_dir() or entry.name == "archive":
            continue
        try:
            dir_date = datetime.strptime(entry.name, "%Y-%m-%d").date()
        except ValueError:
            # Not a YYYY-MM-DD dir — skip silently
            continue

        if dir_date >= cutoff_date:
            continue  # Not old enough

        try:
            dest = archive_root / entry.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(dest))
            moved += 1
        except Exception as exc:
            print(
                f"[archive-sweep] WARN: could not move messages/{entry.name}: {exc}",
                file=sys.stderr,
            )
            skipped += 1

    return moved, skipped


def maybe_run_sweep() -> None:
    """Run archive sweep if 24+ hours have passed since last sweep.

    Always rewrites the gate file after attempting a sweep (even on partial failure)
    so we don't thrash. Logs counts to stderr. Never raises.
    """
    try:
        last_sweep = _read_gate()
        now = datetime.now(tz=timezone.utc)
        if (now - last_sweep) < timedelta(hours=SWEEP_INTERVAL_HOURS):
            return  # Skip silently

        supra_moved, supra_skipped = _sweep_supra_sessions()
        msg_moved, msg_skipped = _sweep_message_dirs()

        print(
            f"[archive-sweep] sweep complete: "
            f"supra sessions moved={supra_moved} skipped={supra_skipped}, "
            f"message dirs moved={msg_moved} skipped={msg_skipped}",
            file=sys.stderr,
        )

        # Rewrite gate regardless of partial success
        _write_gate()

    except Exception as exc:
        print(f"[archive-sweep] ERROR: unexpected failure: {exc}", file=sys.stderr)
        # Still rewrite gate to avoid infinite retry on permanent errors
        try:
            _write_gate()
        except Exception:
            pass


# Allow direct invocation for smoke testing
if __name__ == "__main__":
    # Force a sweep by removing the gate file if requested
    if "--force" in sys.argv:
        GATE_FILE.unlink(missing_ok=True)
        print("[archive-sweep] gate file removed, forcing sweep", file=sys.stderr)
    maybe_run_sweep()
