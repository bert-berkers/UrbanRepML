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
"""
import fnmatch
import os
import random
import sys
from datetime import datetime, timedelta
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

def read_all_claims(coordinators_dir: Path) -> list[dict]:
    """Read all session-*.yaml claim files from coordinators_dir.

    Returns a list of dicts. Skips files that cannot be parsed.
    """
    if yaml is None or not coordinators_dir.is_dir():
        return []

    claims = []
    for claim_path in coordinators_dir.glob("session-*.yaml"):
        try:
            text = claim_path.read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            if isinstance(data, dict):
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
    """Delete a session claim file. Silently ignores missing files."""
    if not coordinators_dir.is_dir():
        return
    target_path = coordinators_dir / f"session-{session_id}.yaml"
    try:
        target_path.unlink(missing_ok=True)
    except Exception as exc:
        print(f"coordinator_registry: failed to delete {target_path}: {exc}", file=sys.stderr)


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
    """Read messages from coordinators_dir/messages/.

    Filters by `at` >= since and by `to` matching to_session or "all".
    Returns messages sorted by `at` ascending.
    """
    if yaml is None:
        return []

    messages_dir = coordinators_dir / "messages"
    if not messages_dir.is_dir():
        return []

    messages = []
    for msg_path in messages_dir.glob("*.yaml"):
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

            # Filter by time
            if since is not None:
                at_str = data.get("at")
                if at_str:
                    try:
                        at_dt = datetime.fromisoformat(str(at_str))
                        if at_dt < since:
                            continue
                    except (ValueError, TypeError):
                        pass

            messages.append(data)
        except Exception as exc:
            print(f"coordinator_registry: failed to read message {msg_path}: {exc}", file=sys.stderr)

    # Sort by timestamp ascending
    def sort_key(msg: dict) -> str:
        return str(msg.get("at", ""))

    messages.sort(key=sort_key)
    return messages


def write_message(coordinators_dir: Path, message_data: dict) -> None:
    """Write a message as an individual YAML file in coordinators_dir/messages/.

    Filename: YYYYMMDD-HHMMSS-{from_session}.yaml  (atomic write)
    Creates the messages/ subdirectory if needed.
    """
    if yaml is None:
        return

    messages_dir = coordinators_dir / "messages"
    messages_dir.mkdir(parents=True, exist_ok=True)

    from_session = message_data.get("from", "unknown")
    at_str = message_data.get("at", datetime.now().isoformat(timespec="seconds"))
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
    """Remove claim files older than threshold_hours and messages older than 7 days.

    Safe to call from any session -- never deletes its own claim (caller's
    responsibility to not pass its own session_id here).
    """
    if not coordinators_dir.is_dir():
        return

    # Stale claim files
    for claim_path in list(coordinators_dir.glob("session-*.yaml")):
        try:
            text = claim_path.read_text(encoding="utf-8")
            if yaml is None:
                continue
            data = yaml.safe_load(text)
            if not isinstance(data, dict):
                claim_path.unlink(missing_ok=True)
                continue
            heartbeat_str = data.get("heartbeat_at") or data.get("started_at")
            if not heartbeat_str:
                claim_path.unlink(missing_ok=True)
                continue
            heartbeat = datetime.fromisoformat(str(heartbeat_str))
            if (datetime.now() - heartbeat) > timedelta(hours=threshold_hours):
                print(
                    f"coordinator_registry: removing stale claim {claim_path.name}",
                    file=sys.stderr,
                )
                claim_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"coordinator_registry: cleanup error for {claim_path}: {exc}", file=sys.stderr)

    # Old messages (> 7 days)
    messages_dir = coordinators_dir / "messages"
    if messages_dir.is_dir():
        cutoff = datetime.now() - timedelta(days=7)
        for msg_path in list(messages_dir.glob("*.yaml")):
            try:
                mtime = datetime.fromtimestamp(msg_path.stat().st_mtime)
                if mtime < cutoff:
                    msg_path.unlink(missing_ok=True)
            except Exception:
                pass


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
