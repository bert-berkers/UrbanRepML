#!/usr/bin/env python3
"""Markov-completeness checker for agent scratchpad close-out entries.

Contract 1 from the 2026-04-18 organizational flywheel audit:
  coordinator  — all 7 items required (fail-CLOSED, checked by stop.py)
  specialist   — items 1, 2, 7 only required (fail-OPEN, checked by subagent-stop.py)

Public API:
  check_completeness(scratchpad_path, agent_type) -> list[str]
    Returns list of missing item names. Empty list = PASS.
    Idempotent — does not modify the scratchpad.
"""
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex patterns for each Contract 1 item
# ---------------------------------------------------------------------------

# Item 1: SUMMARY comment at very top of the entry
#   Matches: <!-- SUMMARY: anything -->
_RE_SUMMARY = re.compile(r"<!--\s*SUMMARY:", re.IGNORECASE)

# Item 2: Prior entries index line
#   Matches: "Prior entries:" with optional markdown bold markers (**...**)
#   Real scratchpad format uses: **Prior entries**: HH:MM — summary
#   Plain format:                 Prior entries: HH:MM — summary
_RE_PRIOR_ENTRIES = re.compile(r"\*{0,2}Prior\s+entries\*{0,2}:", re.IGNORECASE)

# Item 3: Reference frame block
#   Matches: "mode=<word> speed=<digit>" (coordinator characteristic-states block)
_RE_REFERENCE_FRAME = re.compile(r"mode=\w+\s+speed=\d")

# Item 4: Aged open items
#   Accepts [open|Nd], [stale|Nd], or [escalated|Nd] — all three are valid per protocol
#   (items age through: open → stale at 14d → escalated at 21d)
#   Also accepts the edge case where ALL open items have been escalated and only
#   [escalated|Nd] tags remain — that still satisfies the "aged items present" requirement.
_RE_AGED_OPEN = re.compile(r"\[(open|stale|escalated)\|\d+d\]")

# Item 5: Active plan pointer
#   Matches: line starting with "Plan:" followed by non-whitespace
#   "Plan: none (ad-hoc session)" is explicitly valid per protocol
_RE_PLAN_POINTER = re.compile(r"^Plan:\s+\S", re.MULTILINE | re.IGNORECASE)

# Item 6: Peers pointer
#   Matches: line starting with "Peer:" or "Peers:"
#   "Peers: none" is explicitly valid per protocol
_RE_PEERS_POINTER = re.compile(r"^Peers?:", re.MULTILINE | re.IGNORECASE)

# Item 7: "If you only read" paragraph heading
#   Matches: "## If you only read" (heading form, case-insensitive)
_RE_IF_ONLY_READ = re.compile(r"^##\s+If you only read", re.MULTILINE | re.IGNORECASE)

# ---------------------------------------------------------------------------
# Item registry
# ---------------------------------------------------------------------------

# All 7 items: (name, regex, required_for_coordinator, required_for_specialist)
_ITEMS: list[tuple[str, re.Pattern, bool, bool]] = [
    ("item-1:SUMMARY-comment",           _RE_SUMMARY,         True, True),
    ("item-2:prior-entries-index",        _RE_PRIOR_ENTRIES,   True, True),
    ("item-3:reference-frame-block",      _RE_REFERENCE_FRAME, True, False),
    ("item-4:aged-open-items",            _RE_AGED_OPEN,       True, False),
    ("item-5:plan-pointer",               _RE_PLAN_POINTER,    True, False),
    ("item-6:peers-pointer",              _RE_PEERS_POINTER,   True, False),
    ("item-7:if-you-only-read-paragraph", _RE_IF_ONLY_READ,    True, True),
]


# ---------------------------------------------------------------------------
# Entry extraction
# ---------------------------------------------------------------------------

def _extract_last_entry(text: str) -> str:
    """Extract the last appended entry from a scratchpad file.

    An entry starts with a line matching:
      ## HH:MM    (standard timestamped section)
    OR an OVERRIDE block:
      <!-- OVERRIDE: ... -->

    If no entry boundary is found, returns the entire text (small or
    non-session-keyed file — we check it as-is rather than blocking).

    Returns empty string for empty or whitespace-only input.
    """
    if not text.strip():
        return ""

    # Find all positions where a new entry begins
    # Pattern: line starting with "## " followed by digits:digits (HH:MM)
    # OR a line that is an <!-- OVERRIDE: ... --> comment
    entry_boundary = re.compile(
        r"^(?:##\s+\d{1,2}:\d{2}|<!--\s*OVERRIDE:)",
        re.MULTILINE | re.IGNORECASE,
    )

    positions = [m.start() for m in entry_boundary.finditer(text)]

    if not positions:
        # No section headers found — treat the whole file as the entry
        return text

    # The last entry starts at the last boundary position
    last_entry_text = text[positions[-1]:]
    return last_entry_text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_completeness(scratchpad_path: Path, agent_type: str) -> list[str]:
    """Return list of missing Contract 1 items. Empty list means PASS.

    agent_type == 'coordinator' → all 7 items required (fail-CLOSED)
    agent_type == 'specialist'  → items 1, 2, 7 only required (fail-OPEN)

    Any agent_type that is not 'coordinator' is treated as 'specialist'.

    Search scope per item:
    - Item 1 (SUMMARY): whole file — SUMMARY is a file-level marker placed at line 1
      of session-keyed files, which may appear before the first ## HH:MM entry header.
    - Item 2 (prior entries): whole file — same rationale; often in file preamble.
    - Items 3-7: last entry only — these are entry-specific and should appear in the
      most-recent close-out section, not in stale earlier entries.

    Idempotent — does not modify the scratchpad.
    Returns empty list (PASS) when:
      - scratchpad_path does not exist (caller decides policy)
      - scratchpad_path is empty or whitespace-only
      - file cannot be read (fail-open on I/O errors)
    """
    if not scratchpad_path.exists():
        return []

    try:
        full_text = scratchpad_path.read_text(encoding="utf-8")
    except Exception as exc:
        print(
            f"[markov_check] Could not read {scratchpad_path}: {exc}",
            file=sys.stderr,
        )
        return []

    if not full_text.strip():
        return []

    entry_text = _extract_last_entry(full_text)

    is_coordinator = agent_type == "coordinator"
    missing = []

    # Items searched in whole file (file-level markers)
    _FILE_LEVEL_ITEMS = {"item-1:SUMMARY-comment", "item-2:prior-entries-index"}

    for name, pattern, coord_required, spec_required in _ITEMS:
        required = coord_required if is_coordinator else spec_required
        if not required:
            continue
        search_text = full_text if name in _FILE_LEVEL_ITEMS else entry_text
        if not pattern.search(search_text):
            missing.append(name)

    return missing
