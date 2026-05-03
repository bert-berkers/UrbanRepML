"""Single source of truth for `.claude/scratchpad/` filename conventions.

The harness has multiple readers and writers of the same scratchpad files
(ego writes the forward-look, /valuate reads it during morning inread,
coordinators reference it from /niche). When the path pattern is documented
in multiple places, those documents drift relative to each other and the
filesystem accumulates ambiguous filenames.

This module is the single function-level source of truth. Callers MUST use
these helpers rather than hand-constructing paths. If a new scratchpad type
needs a convention, add a helper here first; do not inline the pattern in
the calling code.

Conventions enforced:
- Forward-looks (ego writes, /valuate reads): `coordinator/YYYY-MM-DD-forward-look.md`.
  The bare-date form `coordinator/YYYY-MM-DD.md` is RESERVED for nothing —
  any file matching that pattern is treated as drifted and healed by
  `archive_sweep._sweep_forward_look_drift()`.
- Session-keyed entries (per-shard work logs): `{agent_type}/YYYY-MM-DD-{session_id}.md`.
- Cross-shard valuate logs (per-day, multi-shard append): `valuate/YYYY-MM-DD.md`.
  This is the ONE legitimate bare-date filename pattern, and it lives in
  `valuate/`, not `coordinator/`.

See `.claude/rules/multi-agent-protocol.md` "Scratchpad Discipline" for the
prose-level convention; this module is its enforceable form.
"""
from __future__ import annotations

from datetime import date as _date
from pathlib import Path

# Repo-rooted .claude/scratchpad/ — derived once from this file's location.
_SCRATCHPAD_ROOT = Path(__file__).resolve().parent.parent / "scratchpad"

FORWARD_LOOK_SUFFIX = "-forward-look"


def _coerce_date(d: _date | str) -> str:
    """Accept either a date object or an ISO YYYY-MM-DD string. Returns ISO string.

    Raises ValueError on malformed input — callers that don't want to handle
    that should pre-validate or pass a date object.
    """
    if isinstance(d, _date):
        return d.isoformat()
    if isinstance(d, str):
        # Validate format by round-trip through fromisoformat
        return _date.fromisoformat(d).isoformat()
    raise TypeError(f"date must be date or ISO string, got {type(d).__name__}")


def forward_look_path(d: _date | str, *, scratchpad_root: Path | None = None) -> Path:
    """Canonical path for the coordinator forward-look written for date `d`.

    Use this in EVERY place that reads or writes a forward-look:
      - ego agent (writer)
      - /valuate Step 3.5 morning inread (reader)
      - /niche Wave 0 deferred-P0 scan (reader)
      - archive_sweep._sweep_forward_look_drift (heals legacy drift)

    Args:
        d: target date (the date the forward-look is FOR — typically tomorrow
           when ego writes it, or today's date when /valuate reads it).
        scratchpad_root: override for tests; defaults to `.claude/scratchpad/`.

    Returns:
        Path to `coordinator/YYYY-MM-DD-forward-look.md`.
    """
    root = scratchpad_root if scratchpad_root is not None else _SCRATCHPAD_ROOT
    return root / "coordinator" / f"{_coerce_date(d)}{FORWARD_LOOK_SUFFIX}.md"


_FORWARD_LOOK_CONTENT_MARKERS = ("forward-look", "forward look", "recommended focus")


def is_drifted_forward_look(path: Path) -> bool:
    """True iff `path` is a drifted forward-look that should be renamed.

    Two-stage check:
    1. Filename match: `coordinator/YYYY-MM-DD.md` with no suffix.
    2. Content marker: file contains "forward-look", "forward look", or
       "recommended focus" (case-insensitive) in its first 1024 bytes.

    The content check is REQUIRED because pre-session-keyed-era coordinator
    scratchpads (Feb–Mar 2026) used `coordinator/YYYY-MM-DD.md` as a legitimate
    multi-session daily log. Filename alone cannot distinguish the two; only
    content can. Without this check, the sweep would mis-rename historical
    daily entries (a bug caught live during the W3.7 sweep-hardening 2026-05-03).

    Files that don't exist or can't be read are treated as not-drifted (fail-open
    — better to miss a rename than corrupt unreadable files).
    """
    if path.parent.name != "coordinator":
        return False
    stem = path.stem
    if not stem or len(stem) != 10:
        return False
    try:
        _date.fromisoformat(stem)
    except ValueError:
        return False
    if path.suffix != ".md":
        return False
    if not path.exists() or not path.is_file():
        return False
    try:
        # Read only enough to find a marker; forward-look files put their
        # SUMMARY comment + heading in the first few hundred bytes.
        head = path.read_text(encoding="utf-8", errors="replace")[:1024].lower()
    except OSError:
        return False
    return any(marker in head for marker in _FORWARD_LOOK_CONTENT_MARKERS)


def session_keyed_path(
    agent_type: str,
    d: _date | str,
    session_id: str,
    *,
    scratchpad_root: Path | None = None,
) -> Path:
    """Canonical path for a session-keyed scratchpad entry.

    Format: `{agent_type}/YYYY-MM-DD-{session_id}.md`.
    Used by all per-shard work logs (coordinator, ego, librarian, qaqc, etc.).
    """
    if not session_id:
        raise ValueError("session_id is required for session-keyed paths")
    root = scratchpad_root if scratchpad_root is not None else _SCRATCHPAD_ROOT
    return root / agent_type / f"{_coerce_date(d)}-{session_id}.md"


def valuate_log_path(d: _date | str, *, scratchpad_root: Path | None = None) -> Path:
    """Canonical path for the per-day cross-shard valuate log.

    Format: `valuate/YYYY-MM-DD.md`. Multi-shard append-only; each /valuate
    invocation appends one section keyed by its supra_session_id.
    """
    root = scratchpad_root if scratchpad_root is not None else _SCRATCHPAD_ROOT
    return root / "valuate" / f"{_coerce_date(d)}.md"
