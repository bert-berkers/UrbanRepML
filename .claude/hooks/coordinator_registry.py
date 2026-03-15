#!/usr/bin/env python3
"""Shared library for coordinator-to-coordinator claim file I/O.

Functions:
  generate_session_id   -- Poetic word-combination session IDs
  read_all_claims       -- Read all session-*.yaml claim files
  write_claim           -- Atomically write a claim file
  delete_claim          -- Delete a claim file
  is_stale              -- Check heartbeat staleness
  check_conflict        -- Match a file path against other sessions' claimed_paths globs
  read_messages         -- Read messages from the messages/ subdirectory
  write_message         -- Write a message as an individual file
  cleanup_stale         -- Remove stale claim files and old messages
  update_heartbeat      -- Update heartbeat_at in claim file

Terminal-shell-keyed session identity (stable across /clear):
  get_terminal_pid      -- Get the terminal shell PID (stable per tab, survives /clear)
  write_ppid_session    -- Write terminal-PID-keyed session file
  write_ppid_supra      -- Write terminal-PID-keyed supra file
  read_ppid_session     -- Read session_id for this terminal's PID
  read_ppid_supra       -- Read supra_session_id for this terminal's PID
  cleanup_stale_ppid_files -- Remove session/supra files for dead processes
  archive_old_messages  -- Move messages into per-day subdirs
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

def generate_session_id(coordinators_dir: Path) -> str:
    """Generate a unique poetic session ID (adjective-participle-noun).

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
    """Write a lateral message using the supra session ID as sender identity.

    If supra session ID is available, overrides the 'from' field with it.
    Falls back to coordinator session ID, then 'unknown'.
    This ensures narrative coherence (homo narrans) across percept deaths.
    """
    try:
        supra_sid = read_ppid_supra(coordinators_dir)
        if supra_sid:
            message_data = dict(message_data, **{"from": supra_sid})
        else:
            # Fall back to coordinator session ID
            coord_sid = read_ppid_session(coordinators_dir)
            if coord_sid:
                message_data = dict(message_data, **{"from": coord_sid})
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
# PPID-keyed session identity
# ---------------------------------------------------------------------------

SESSIONS_SUBDIR = "sessions"   # coordinators/sessions/
SUPRA_SUBDIR = "supra"         # coordinators/supra/


def get_terminal_pid() -> int:
    """Get the terminal shell PID — stable across /clear, unique per terminal tab.

    Walks up the process tree via psutil to find node.exe (Claude Code),
    then returns its parent's PID (the terminal shell, e.g. powershell.exe).
    The terminal shell PID is stable for the lifetime of the terminal tab and
    survives /clear (which only restarts node.exe).

    Process tree on this machine:
        pycharm64.exe -> powershell.exe (STABLE) -> node.exe (CHANGES) -> bash.exe -> python.exe (hook)

    Falls back to os.getppid() if psutil is unavailable or the tree walk fails
    (e.g., the hook runs in a context without a node.exe ancestor).
    """
    try:
        import psutil
        p = psutil.Process(os.getpid())
        while True:
            p = p.parent()
            if p is None:
                break
            # Find the first node / node.exe process walking upward
            try:
                name = p.name().lower()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if name in ("node", "node.exe"):
                # The terminal shell is node's parent
                terminal = p.parent()
                if terminal is not None:
                    try:
                        return terminal.pid
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                break
    except Exception:
        pass
    # Fallback: direct parent (original os.getppid() behaviour)
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


def write_ppid_session(coordinators_dir: Path, session_id: str) -> None:
    """Write terminal-PID-keyed session file: sessions/{session_id}.{terminal_pid}

    The key is the terminal shell PID (stable across /clear), not the direct
    parent process PID. This ensures the session file survives Claude Code
    restarts (/clear) within the same terminal tab.
    """
    ppid = get_terminal_pid()
    sessions_dir = coordinators_dir / SESSIONS_SUBDIR
    sessions_dir.mkdir(exist_ok=True)
    # Archive any previous session file for this PPID (from prior /clear)
    archive = sessions_dir / "archive"
    for old in sessions_dir.glob(f"*.{ppid}"):
        if old.is_dir():
            continue
        archive.mkdir(exist_ok=True)
        old.rename(archive / old.name)
    path = sessions_dir / f"{session_id}.{ppid}"
    path.write_text(session_id, encoding="utf-8")


def write_ppid_supra(coordinators_dir: Path, supra_session_id: str) -> None:
    """Write terminal-PID-keyed supra file: supra/{supra_session_id}.{terminal_pid}

    The key is the terminal shell PID (stable across /clear), not the direct
    parent process PID. Supra files persist across /clear cycles within the
    same terminal tab.
    """
    ppid = get_terminal_pid()
    supra_dir = coordinators_dir / SUPRA_SUBDIR
    supra_dir.mkdir(exist_ok=True)
    # Archive any previous supra file for this PPID
    archive = supra_dir / "archive"
    for old in supra_dir.glob(f"*.{ppid}"):
        if old.is_dir():
            continue
        archive.mkdir(exist_ok=True)
        old.rename(archive / old.name)
    path = supra_dir / f"{supra_session_id}.{ppid}"
    path.write_text(supra_session_id, encoding="utf-8")


def read_ppid_session(coordinators_dir: Path) -> str | None:
    """Read session_id for this terminal's stable PID."""
    ppid = get_terminal_pid()
    sessions_dir = coordinators_dir / SESSIONS_SUBDIR
    if not sessions_dir.is_dir():
        return None
    matches = list(sessions_dir.glob(f"*.{ppid}"))
    if matches:
        return matches[0].read_text(encoding="utf-8").strip()
    return None


def read_ppid_supra(coordinators_dir: Path) -> str | None:
    """Read supra_session_id for this terminal's stable PID."""
    ppid = get_terminal_pid()
    supra_dir = coordinators_dir / SUPRA_SUBDIR
    if not supra_dir.is_dir():
        return None
    matches = list(supra_dir.glob(f"*.{ppid}"))
    if matches:
        return matches[0].read_text(encoding="utf-8").strip()
    return None


def cleanup_stale_ppid_files(coordinators_dir: Path) -> None:
    """Archive session and supra files for dead processes into archive/ subdir."""
    for subdir in (SESSIONS_SUBDIR, SUPRA_SUBDIR):
        d = coordinators_dir / subdir
        if not d.is_dir():
            continue
        archive = d / "archive"
        for f in d.iterdir():
            if f.is_dir():
                continue
            try:
                pid = int(f.name.rsplit(".", 1)[-1])
                if not _is_process_alive(pid):
                    archive.mkdir(exist_ok=True)
                    f.rename(archive / f.name)
            except (ValueError, OSError):
                archive.mkdir(exist_ok=True)
                f.rename(archive / f.name)


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
