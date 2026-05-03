#!/usr/bin/env python3
"""Shared library for coordinator-to-coordinator claim file I/O.

Functions:
  generate_identity_id  -- Poetic word-combination identity IDs
  read_all_claims       -- Read all session-*.yaml claim files
  write_claim           -- Atomically write a claim file
  delete_claim          -- Delete a claim file
  is_stale              -- Check heartbeat staleness
  check_conflict        -- Match a file path against other sessions' claimed_paths globs
  read_messages         -- Read messages from the messages/ subdirectory
  write_message         -- Write a message as an individual file
  cleanup_stale         -- Remove stale claim files and old messages
  update_heartbeat      -- Update heartbeat_at in claim file

Terminal-shell-keyed identity (one identity per terminal, stable across /clear):
  get_terminal_pid      -- Get the terminal shell PID (stable per tab, survives /clear)
  write_ppid_identity   -- Write terminal-PID-keyed identity file
  read_ppid_identity    -- Read identity_id for this terminal's PID
  set_active_plan       -- Set active_plan field in terminal file
  mark_wave_complete    -- Set last_wave_completed_at in terminal file
  cleanup_stale_ppid_files -- Archive terminal files for dead processes
  archive_old_messages  -- Move messages into per-day subdirs

Terminal identity schema (terminals/{pid}.yaml):
  identity_id             str       -- the one identity for this terminal (the poetic name)
  started_at              iso8601   -- when this terminal was first registered
  active_plan             str|null  -- path to plan file coordinator is following
  last_wave_completed_at  iso8601|null -- timestamp of most recent wave completion

Backward compat: `read_ppid_identity` falls back to `supra_session_id` then
`session_id` if `identity_id` is absent. This protects live terminal files
written before the 2026-05-03 collapse to one identity field.

See `specs/session-identity-architecture.md`.
"""
import fnmatch
import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# PyYAML -- already a project dependency
try:
    import yaml
except ImportError:
    print("coordinator_registry: PyYAML not available", file=sys.stderr)
    yaml = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Word lists for session ID generation
# ---------------------------------------------------------------------------

ADJECTIVES = [
    "amber", "azure", "calm", "cedar", "cool", "copper", "coral", "crimson",
    "dawn", "deep", "dusk", "fern", "gentle", "golden", "green", "grey",
    "hollow", "hushed", "indigo", "jade", "keen", "lunar", "mellow", "misty",
    "mossy", "muted", "night", "opal", "pale", "pine", "quiet", "rose",
    "russet", "sage", "salt", "silver", "slate", "still", "stone", "sunlit",
    "swift", "teal", "tender", "twilight", "verdant", "warm", "willowy",
]

PARTICIPLES = [
    "adrift", "blooming", "branching", "breaking", "burning", "cascading",
    "climbing", "coasting", "cooling", "drifting", "falling", "flowing",
    "gathering", "glowing", "hovering", "leaning", "lifting", "lingering",
    "listening", "moving", "passing", "reaching", "resting", "rising",
    "rolling", "running", "sailing", "settling", "shifting", "shining",
    "sighing", "singing", "sinking", "sliding", "slowing", "soaring",
    "spinning", "spreading", "standing", "turning", "wading", "waiting",
    "walking", "wandering", "watching", "waving", "weaving", "whispering",
]

NOUNS = [
    "ash", "bay", "birch", "brook", "canopy", "cedar", "cliff", "coast",
    "creek", "dew", "dune", "dust", "ember", "fern", "field", "fjord",
    "fog", "frost", "gale", "glade", "glen", "grain", "grove", "harbor",
    "heath", "hollow", "inlet", "isle", "kelp", "lake", "leaf", "ledge",
    "light", "loch", "maple", "marsh", "meadow", "mist", "moon", "moor",
    "moss", "peak", "pine", "pond", "rain", "reed", "ridge", "river",
    "rock", "shore", "sky", "slope", "snow", "spring", "stone", "storm",
    "stream", "tide", "timber", "vale", "wave", "willow", "wind", "wood",
]


# ---------------------------------------------------------------------------
# Session ID
# ---------------------------------------------------------------------------

def generate_identity_id(coordinators_dir: Path) -> str:
    """Generate a unique poetic identity ID (adjective-participle-noun).

    This identity is the single name carried by a terminal for its entire
    lifetime — used as both coordinator session ID and supra session ID
    (which are no longer distinct concepts after 2026-05-03; see
    `specs/session-identity-architecture.md`).

    Checks for collision against existing session-*.yaml files and regenerates
    up to 20 times before giving up with a timestamp fallback.
    """
    existing = {p.stem.removeprefix("session-") for p in coordinators_dir.glob("session-*.yaml")}

    for _ in range(20):
        candidate = (
            random.choice(ADJECTIVES)
            + "-"
            + random.choice(PARTICIPLES)
            + "-"
            + random.choice(NOUNS)
        )
        if candidate not in existing:
            return candidate

    # Fallback: timestamp-based ID (should never happen with the pool sizes)
    return "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Claim file I/O
# ---------------------------------------------------------------------------

def read_all_claims(coordinators_dir: Path, include_ended: bool = False) -> list[dict]:
    """Read all session-*.yaml claim files from coordinators_dir.

    Returns a list of dicts. Skips files that cannot be parsed.
    By default, excludes claims with status: ended.
    """
    if yaml is None or not coordinators_dir.is_dir():
        return []

    claims = []
    for claim_path in coordinators_dir.glob("session-*.yaml"):
        try:
            text = claim_path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if isinstance(data, dict):
                if not include_ended and data.get("status") == "ended":
                    continue
                claims.append(data)
        except Exception as exc:
            print(f"coordinator_registry: failed to read {claim_path}: {exc}", file=sys.stderr)
    return claims


def write_claim(coordinators_dir: Path, claim_data: dict) -> None:
    """Atomically write a claim file to coordinators_dir.

    Uses a temp file + os.replace() for atomic write. Creates the directory
    if it does not exist.
    """
    if yaml is None:
        return

    coordinators_dir.mkdir(parents=True, exist_ok=True)
    session_id = claim_data.get("session_id", "unknown")
    target_path = coordinators_dir / f"session-{session_id}.yaml"
    tmp_path = target_path.with_suffix(".yaml.tmp")

    try:
        text = yaml.dump(claim_data, default_flow_style=False, allow_unicode=True)
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(str(tmp_path), str(target_path))
    except Exception as exc:
        print(f"coordinator_registry: failed to write claim {target_path}: {exc}", file=sys.stderr)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def delete_claim(coordinators_dir: Path, session_id: str) -> None:
    """Mark a session claim as ended (status: ended) instead of deleting.

    Preserves the claim file for audit trail. Ended claims are ignored by
    read_all_claims filtering on is_stale().
    """
    if yaml is None or not coordinators_dir.is_dir():
        return
    target_path = coordinators_dir / f"session-{session_id}.yaml"
    if not target_path.exists():
        return
    try:
        data = yaml.safe_load(target_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        data["status"] = "ended"
        data["ended_at"] = datetime.now().isoformat(timespec="seconds")
        write_claim(coordinators_dir, data)
    except Exception as exc:
        print(f"coordinator_registry: failed to end claim {target_path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def is_stale(claim: dict, threshold_minutes: int = 30) -> bool:
    """Return True if claim's heartbeat_at is older than threshold_minutes.

    Treats missing or unparseable heartbeat_at as stale.
    """
    heartbeat_str = claim.get("heartbeat_at")
    if not heartbeat_str:
        return True
    try:
        heartbeat = datetime.fromisoformat(str(heartbeat_str))
        return (datetime.now() - heartbeat) > timedelta(minutes=threshold_minutes)
    except (ValueError, TypeError):
        return True


# ---------------------------------------------------------------------------
# Conflict check
# ---------------------------------------------------------------------------

def check_conflict(
    file_path: str,
    claims: list[dict],
    my_session_id: str,
) -> list[dict]:
    """Check if file_path conflicts with other sessions' claimed_paths globs.

    Returns list of conflicting claim dicts (excluding my_session_id).
    Uses fnmatch for glob matching. Normalizes separators to forward slash
    for cross-platform consistency.
    """
    # Normalize the file path to forward slashes for consistent matching
    normalized = file_path.replace("\\", "/")

    conflicts = []
    for claim in claims:
        sid = claim.get("session_id", "")
        if sid == my_session_id:
            continue
        claimed_paths = claim.get("claimed_paths", [])
        for pattern in (claimed_paths or []):
            # Normalize pattern too
            pattern_norm = str(pattern).replace("\\", "/")
            if fnmatch.fnmatch(normalized, pattern_norm):
                conflicts.append(claim)
                break  # One match per claim is enough
    return conflicts


# ---------------------------------------------------------------------------
# Message I/O (per-file approach in messages/ subdirectory)
# ---------------------------------------------------------------------------

def read_messages(
    coordinators_dir: Path,
    since: datetime | None = None,
    to_session: str | None = None,
) -> list[dict]:
    """Read messages from coordinators_dir/messages/{date}/ subdirs.

    Filters by `at` >= since and by `to` matching to_session or "all".
    Returns messages sorted by `at` ascending.
    Also reads any legacy flat .yaml files in messages/ for backward compat.
    """
    if yaml is None:
        return []

    messages_dir = coordinators_dir / "messages"
    if not messages_dir.is_dir():
        return []

    messages = []
    # Scan date subdirs + legacy flat files
    for msg_path in messages_dir.rglob("*.yaml"):
        try:
            text = msg_path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if not isinstance(data, dict):
                continue

            # Filter by recipient
            if to_session is not None:
                recipient = data.get("to", "all")
                if recipient not in ("all", to_session):
                    continue

            # Filter by time — fall back to file mtime if 'at' missing/unparseable
            # Also cap at 24h max age and reject future-dated messages
            now_dt = datetime.now()
            hard_cutoff = now_dt - timedelta(hours=24)
            if since is not None:
                at_str = data.get("at")
                at_dt = None
                if at_str:
                    try:
                        at_dt = datetime.fromisoformat(str(at_str))
                    except (ValueError, TypeError):
                        pass
                if at_dt is None:
                    at_dt = datetime.fromtimestamp(msg_path.stat().st_mtime)
                # Reject future-dated messages (clamp to now)
                if at_dt > now_dt:
                    at_dt = datetime.fromtimestamp(msg_path.stat().st_mtime)
                if at_dt < since or at_dt < hard_cutoff:
                    continue

            messages.append(data)
        except Exception as exc:
            print(f"coordinator_registry: failed to read message {msg_path}: {exc}", file=sys.stderr)

    # Sort by timestamp ascending
    def sort_key(msg: dict) -> str:
        return str(msg.get("at", ""))

    messages.sort(key=sort_key)
    return messages


def write_message(coordinators_dir: Path, message_data: dict) -> None:
    """Write a message as an individual YAML file in coordinators_dir/messages/{today}/.

    Filename: YYYYMMDD-HHMMSS-{from_session}.yaml  (atomic write)
    Creates the messages/{today}/ subdirectory if needed.
    """
    if yaml is None:
        return

    messages_dir = coordinators_dir / "messages" / date.today().isoformat()
    messages_dir.mkdir(parents=True, exist_ok=True)

    from_session = message_data.get("from", "unknown")
    # Always ensure 'at' is set in the stored data
    if "at" not in message_data:
        message_data = dict(message_data, at=datetime.now().isoformat(timespec="seconds"))
    at_str = message_data["at"]
    # Sanitize timestamp for filename: remove colons and dashes
    ts_clean = str(at_str).replace(":", "").replace("-", "").replace("T", "-")[:15]
    filename = f"{ts_clean}-{from_session}.yaml"
    target_path = messages_dir / filename
    tmp_path = target_path.with_suffix(".yaml.tmp")

    try:
        text = yaml.dump(message_data, default_flow_style=False, allow_unicode=True)
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(str(tmp_path), str(target_path))
    except Exception as exc:
        print(f"coordinator_registry: failed to write message {target_path}: {exc}", file=sys.stderr)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_stale(coordinators_dir: Path, threshold_hours: int = 2) -> None:
    """Mark stale claim files as ended. Archive very old claims (>7 days).

    Safe to call from any session.
    """
    if not coordinators_dir.is_dir() or yaml is None:
        return

    archive_dir = coordinators_dir / "archive"

    for claim_path in list(coordinators_dir.glob("session-*.yaml")):
        try:
            text = claim_path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if not isinstance(data, dict):
                continue
            heartbeat_str = data.get("heartbeat_at") or data.get("started_at")
            if not heartbeat_str:
                continue
            heartbeat = datetime.fromisoformat(str(heartbeat_str))
            age = datetime.now() - heartbeat

            # Already ended — archive if old enough
            if data.get("status") == "ended":
                if age > timedelta(days=7):
                    archive_dir.mkdir(exist_ok=True)
                    claim_path.rename(archive_dir / claim_path.name)
                    print(f"coordinator_registry: archived old claim {claim_path.name}", file=sys.stderr)
                continue

            # Active but stale — mark as ended
            if age > timedelta(hours=threshold_hours):
                data["status"] = "ended"
                data["ended_at"] = datetime.now().isoformat(timespec="seconds")
                data["ended_reason"] = "stale_heartbeat"
                write_claim(coordinators_dir, data)
                print(f"coordinator_registry: marked stale claim {claim_path.name} as ended", file=sys.stderr)
        except Exception as exc:
            print(f"coordinator_registry: cleanup error for {claim_path}: {exc}", file=sys.stderr)

    # Old messages (> 7 days) — scan date subdirs and legacy flat files
    messages_dir = coordinators_dir / "messages"
    if messages_dir.is_dir():
        cutoff = datetime.now() - timedelta(days=7)
        for msg_path in list(messages_dir.rglob("*.yaml")):
            try:
                mtime = datetime.fromtimestamp(msg_path.stat().st_mtime)
                if mtime < cutoff:
                    msg_path.unlink(missing_ok=True)
            except Exception:
                pass
        # Remove empty date subdirs
        for d in list(messages_dir.iterdir()):
            if d.is_dir():
                try:
                    if not any(d.iterdir()):
                        d.rmdir()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Lateral coupling helpers (supra session identity + graph-driven gating)
# ---------------------------------------------------------------------------

def write_lateral_message(
    coordinators_dir: Path,
    message_data: dict,
) -> None:
    """Write a lateral message using the terminal's identity as sender.

    Overrides the 'from' field with the terminal's `identity_id`. This ensures
    narrative coherence (homo narrans) across context window refreshes — the
    name is stable for the lifetime of the terminal.
    """
    try:
        identity = read_ppid_identity(coordinators_dir)
        if identity:
            message_data = dict(message_data, **{"from": identity})
    except Exception:
        pass  # Fall through with whatever 'from' was already set

    write_message(coordinators_dir, message_data)


def is_lateral_coupling_active() -> bool:
    """Check if lateral coupling is active (dynamic graph mode).

    Delegates to supra_reader.is_lateral_coupling_active().
    Returns True as default (fail-open -- don't suppress messages on import error).
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import supra_reader
        return supra_reader.is_lateral_coupling_active()
    except Exception:
        return True  # Fail open -- allow messages by default


# ---------------------------------------------------------------------------
# Heartbeat update
# ---------------------------------------------------------------------------

def update_heartbeat(coordinators_dir: Path, session_id: str) -> None:
    """Update heartbeat_at in the session's claim file to now.

    Reads the existing claim, updates the field, and atomically rewrites.
    No-ops silently if the claim file does not exist.
    """
    if yaml is None:
        return

    claim_path = coordinators_dir / f"session-{session_id}.yaml"
    if not claim_path.exists():
        return

    try:
        text = claim_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            return
        data["heartbeat_at"] = datetime.now().isoformat(timespec="seconds")
        write_claim(coordinators_dir, data)
    except Exception as exc:
        print(f"coordinator_registry: failed to update heartbeat for {session_id}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Terminal-keyed identity: coordinators/terminals/{pid}.yaml
# ---------------------------------------------------------------------------

TERMINALS_SUBDIR = "terminals"  # coordinators/terminals/


# Claude Code marker process names. The walk stops at the first match and
# returns the marker's parent PID (the terminal shell). Both names are needed:
# `node`/`node.exe` is the bundled-runtime form; `claude`/`claude.exe` is the
# packaged-binary form (current PyCharm + Windows install path:
# `AppData\Roaming\npm\node_modules\@anthropic-ai\claude-code\bin\claude.exe`).
# Add new names here when Claude Code's process name changes upstream.
_CLAUDE_PROCESS_NAMES = ("node", "node.exe", "claude", "claude.exe")


def get_terminal_pid() -> int:
    """Get the terminal shell PID — stable across /clear, unique per terminal tab.

    Walks up the process tree via psutil to find the Claude Code marker process
    (`node.exe` or `claude.exe`), then returns its parent's PID — the terminal
    shell, which is stable for the lifetime of the terminal tab.

    Process tree on Windows + PyCharm:
        pycharm64.exe -> powershell.exe (STABLE) -> claude.exe (STABLE) -> bash.exe (CHANGES per Bash tool call) -> python.exe (hook)

    Earlier docstring claimed `node.exe (CHANGES)` — that was wrong; the bundled
    Claude binary, whatever its name, is stable for the lifetime of the Claude
    Code session. The thing that changes is the per-tool-call bash subprocess.
    The marker walk anchors on the stable Claude process and returns its parent.

    Falls back to os.getppid() if psutil is unavailable or the tree walk fails —
    note that this fallback path is unreliable when called from a tool-spawned
    subprocess (it returns the ephemeral bash PID, not the terminal). Hooks that
    call this directly are fine because they run in the Claude process; ad-hoc
    `python -c "..."` invocations from Bash tool calls need the marker walk to
    succeed to get a stable PID.
    """
    try:
        import psutil
        p = psutil.Process(os.getpid())
        while True:
            p = p.parent()
            if p is None:
                break
            try:
                name = p.name().lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if name in _CLAUDE_PROCESS_NAMES:
                terminal = p.parent()
                if terminal is not None:
                    try:
                        return terminal.pid
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                break
    except Exception:
        pass
    return os.getppid()


def _is_process_alive(pid: int) -> bool:
    """Cross-platform process liveness check."""
    if sys.platform == "win32":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _read_terminal_file(coordinators_dir: Path, pid: int | None = None) -> dict | None:
    """Read terminals/{pid}.yaml, returns parsed dict or None."""
    if yaml is None:
        return None
    if pid is None:
        pid = get_terminal_pid()
    path = coordinators_dir / TERMINALS_SUBDIR / f"{pid}.yaml"
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or None
    except Exception:
        return None


def _write_terminal_file(coordinators_dir: Path, data: dict, pid: int | None = None) -> None:
    """Atomically write terminals/{pid}.yaml."""
    if yaml is None:
        return
    if pid is None:
        pid = get_terminal_pid()
    terminals_dir = coordinators_dir / TERMINALS_SUBDIR
    terminals_dir.mkdir(parents=True, exist_ok=True)
    path = terminals_dir / f"{pid}.yaml"
    tmp = path.with_suffix(".yaml.tmp")
    try:
        tmp.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
    except Exception as exc:
        print(f"coordinator_registry: failed to write {path}: {exc}", file=sys.stderr)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def write_ppid_identity(coordinators_dir: Path, identity_id: str) -> None:
    """Set identity_id in this terminal's identity file.

    One terminal = one identity for the lifetime of the terminal. Writes the
    canonical `identity_id` field; for one transition cycle also keeps
    `supra_session_id` in sync so legacy readers and existing live terminal
    files that referenced it do not break mid-session.
    """
    data = _read_terminal_file(coordinators_dir) or {}
    data["identity_id"] = identity_id
    # Transition compat: keep supra_session_id mirrored for one cycle so that
    # any external tools or in-flight reads that still query it resolve.
    data["supra_session_id"] = identity_id
    if "started_at" not in data:
        data["started_at"] = datetime.now().isoformat(timespec="seconds")
    _write_terminal_file(coordinators_dir, data)


def read_ppid_identity(coordinators_dir: Path) -> str | None:
    """Read identity_id for this terminal's PID.

    Backward-compat fallback chain: identity_id -> supra_session_id -> session_id.
    The fallback protects live terminal files written before the 2026-05-03
    collapse to one identity field — at hook-write time they still carry
    `supra_session_id` (the persistent name) without an `identity_id` field.
    """
    data = _read_terminal_file(coordinators_dir)
    if not data:
        return None
    return (
        data.get("identity_id")
        or data.get("supra_session_id")
        or data.get("session_id")
    )


def cleanup_stale_ppid_files(coordinators_dir: Path) -> None:
    """Archive terminal files for dead processes."""
    terminals_dir = coordinators_dir / TERMINALS_SUBDIR
    if not terminals_dir.is_dir():
        return
    archive = terminals_dir / "archive"
    for f in terminals_dir.iterdir():
        if f.is_dir():
            continue
        try:
            pid = int(f.stem)
            if not _is_process_alive(pid):
                archive.mkdir(exist_ok=True)
                f.rename(archive / f.name)
        except (ValueError, OSError):
            archive.mkdir(exist_ok=True)
            f.rename(archive / f.name)


def set_active_plan(coordinators_dir: Path, plan_path: str, pid: int | None = None) -> None:
    """Set active_plan in this terminal's identity file.

    Records the path to the plan file the coordinator is currently following.
    Sets active_plan to null when plan_path is empty string.
    Non-breaking — existing fields (session_id, supra_session_id, started_at) are preserved.
    """
    data = _read_terminal_file(coordinators_dir, pid=pid) or {}
    data["active_plan"] = plan_path if plan_path else None
    _write_terminal_file(coordinators_dir, data, pid=pid)


def mark_wave_complete(coordinators_dir: Path, pid: int | None = None) -> None:
    """Record ISO8601 timestamp of the most recent wave completion in this terminal's file.

    Sets last_wave_completed_at to the current time.
    Non-breaking — existing fields are preserved.
    """
    data = _read_terminal_file(coordinators_dir, pid=pid) or {}
    data["last_wave_completed_at"] = datetime.now().isoformat(timespec="seconds")
    _write_terminal_file(coordinators_dir, data, pid=pid)


def archive_old_messages(coordinators_dir: Path) -> None:
    """Move messages into messages/{date}/ subdirs based on filename date prefix."""
    messages_dir = coordinators_dir / "messages"
    if not messages_dir.is_dir():
        return
    today = date.today().isoformat()
    for f in list(messages_dir.iterdir()):
        if f.is_dir():
            continue  # Already a date dir
        # Extract date from filename (YYYY-MM-DD-* or YYYYMMDD_*)
        name = f.name
        if len(name) >= 10 and name[4] == "-" and name[7] == "-":
            file_date = name[:10]
        elif len(name) >= 8 and name[:8].isdigit():
            file_date = f"{name[:4]}-{name[4:6]}-{name[6:8]}"
        else:
            file_date = today  # Can't parse — keep in today
        target_dir = messages_dir / file_date
        target_dir.mkdir(exist_ok=True)
        f.rename(target_dir / f.name)
