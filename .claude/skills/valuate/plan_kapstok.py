"""Plan kapstok writer — invoked from /valuate Step 5.6.

Writes a structural markdown plan to `.claude/plans/{date}-{intent-slug}.md`
that scaffolds the next `/niche` invocation. The kapstok is a SEED, not a
contract — `/niche` may deviate per the Wave-deviation policy.

See `specs/valuate_plan_kapstok.md` for the full format and behavior contract.

Public API:
    write_kapstok(supra_session_yaml, valuate_scratchpad, plans_dir, date,
                  *, force=False) -> Path | None

    slugify(text, max_tokens=8) -> str  (also exposed for /niche Wave-0 use)
"""
from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import yaml


_MULTITHREAD_MODES = frozenset({"creative", "exploratory"})
_TODO = "TODO — coordinator-direct fill-in"


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def slugify(text: str, max_tokens: int = 8) -> str:
    """Lowercase ASCII slug. Alphanumerics + hyphens. Max max_tokens hyphenated tokens.

    Returns empty string for input that yields no slug content.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    if not s:
        return ""
    parts = [p for p in s.split("-") if p]
    return "-".join(parts[:max_tokens])


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

def _is_multithread(supra: dict) -> bool:
    """True iff exploration_vs_exploitation >= 4 AND mode in {creative, exploratory}."""
    dims = supra.get("dimensions") or {}
    try:
        explore = int(dims.get("exploration_vs_exploitation", 3))
    except (TypeError, ValueError):
        return False
    mode = (supra.get("mode") or "").strip().lower()
    return explore >= 4 and mode in _MULTITHREAD_MODES


# ---------------------------------------------------------------------------
# Carry-items extraction (from valuate scratchpad entry)
# ---------------------------------------------------------------------------

_CARRY_KEYWORDS = ("carry item", "carry-item", "forward-look", "[open|", "[stale|", "p0", "p1")


def _carry_items_from_scratchpad(scratchpad: Path, supra_session_id: str) -> list[str]:
    """Pull short carry-item clauses from this terminal's valuate entry.

    The valuate scratchpad uses one section per terminal: `## {supra_session_id} — HH:MM`.
    Carry-items are referenced in prose; we extract sentence-level clauses containing
    any of _CARRY_KEYWORDS.
    """
    if not scratchpad.exists() or not supra_session_id:
        return []
    try:
        text = scratchpad.read_text(encoding="utf-8")
    except OSError:
        return []

    pattern = re.compile(
        rf"^## {re.escape(supra_session_id)} — \d{{1,2}}:\d{{2}}\s*\n(.*?)(?=\n## |\Z)",
        flags=re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(text)
    if not m:
        return []
    block = m.group(1)

    items: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", block):
        sent_clean = sent.strip().lstrip("-* ").strip()
        if not sent_clean or sent_clean.startswith("#"):
            continue
        low = sent_clean.lower()
        if any(kw in low for kw in _CARRY_KEYWORDS):
            short = sent_clean.replace("\n", " ").strip()
            if len(short) > 160:
                short = short[:157].rstrip() + "..."
            items.append(short)
        if len(items) >= 4:
            break
    return items


# ---------------------------------------------------------------------------
# Title rendering
# ---------------------------------------------------------------------------

_TITLE_LOWERCASE = frozenset({"a", "an", "the", "of", "in", "for", "on", "to", "and", "or"})


def _intent_to_title(intent: str, date: str) -> str:
    """Render intent as a Title Case heading + date tag."""
    if not intent:
        return f"Untitled — {date}"
    words = re.split(r"\s+", intent.strip())[:6]
    rendered = []
    for i, w in enumerate(words):
        wl = w.lower()
        if i > 0 and wl in _TITLE_LOWERCASE:
            rendered.append(wl)
        else:
            rendered.append(w[:1].upper() + w[1:].lower() if w else w)
    return f"{' '.join(rendered)} — {date}"


# ---------------------------------------------------------------------------
# Peer-terminal pointer
# ---------------------------------------------------------------------------

def _peer_pointer(supra_session_yaml: Path, supra_session_id: str) -> str:
    """Read peer terminals (live + recently-baked supra YAMLs) and list them."""
    sessions_dir = supra_session_yaml.parent
    coordinators_dir = supra_session_yaml.parent.parent.parent / "coordinators" / "terminals"

    peers_supra: list[str] = []
    try:
        date_suffix = supra_session_yaml.stem.split("-", 3)[-1] if "-" in supra_session_yaml.stem else ""
        if sessions_dir.is_dir():
            for path in sorted(sessions_dir.glob("*.yaml")):
                if path == supra_session_yaml or path.name.startswith("archive"):
                    continue
                if date_suffix and date_suffix not in path.stem:
                    continue
                peers_supra.append(path.stem)
    except OSError:
        pass

    live_terminals: list[str] = []
    try:
        if coordinators_dir.is_dir():
            for path in sorted(coordinators_dir.glob("*.yaml")):
                if path.is_dir() or path.name.startswith("archive"):
                    continue
                try:
                    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                except (yaml.YAMLError, OSError):
                    continue
                ssid = data.get("supra_session_id", "")
                if ssid and ssid != supra_session_id:
                    live_terminals.append(f"{ssid} (PID {path.stem})")
    except OSError:
        pass

    lines = ["## Peer-terminal pointer\n"]
    if live_terminals:
        for entry in live_terminals:
            lines.append(f"- {entry}\n")
    elif peers_supra:
        lines.append("Live terminal bindings not detected; supra YAMLs present today:\n")
        for ssid in peers_supra:
            lines.append(f"- {ssid}\n")
    else:
        lines.append("- Peers: none\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _ref_frame_block(supra: dict) -> str:
    dims = supra.get("dimensions") or {}
    intent = (supra.get("intent") or "").strip()
    focus = list(supra.get("focus") or [])
    suppress = list(supra.get("suppress") or [])
    mode = (supra.get("mode") or "neutral").strip()

    def _d(key: str, default: int = 3) -> int:
        v = dims.get(key, default)
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    return (
        "```\n"
        f"mode={mode} "
        f"speed={_d('execution_speed')} "
        f"explore={_d('exploration_vs_exploitation')} "
        f"quality={_d('code_quality')} "
        f"tests={_d('test_coverage')} "
        f"spatial={_d('spatial_correctness')} "
        f"model={_d('model_architecture')} "
        f"urgency={_d('urgency')} "
        f"data_eng={_d('data_engineering_diligence')}\n"
        f'intent="{intent}"\n'
        f"focus={focus!r}\n"
        f"suppress={suppress!r}\n"
        "```\n"
    )


def _status_table(supra_session_id: str, intent: str, multithread: bool) -> str:
    shard = supra_session_id.rsplit("-", 3)[0] if supra_session_id else "unknown"
    mode_label = "multi-thread (W0 picks)" if multithread else "single-thread"
    return (
        "| Field | Value |\n"
        "|---|---|\n"
        f"| **Status** | DRAFT — kapstok written by `/valuate` to seed `/niche` Wave-0 |\n"
        f"| **Shard** | `{shard}` |\n"
        f"| **Mode** | {mode_label} |\n"
        f"| **Intent** | {intent or '(none set)'} |\n"
        "| **Est** | TBD by /niche W0 |\n"
    )


def _frame_paragraph(multithread: bool) -> str:
    if multithread:
        return (
            "## Frame — why a kapstok and not a fixed-target plan\n\n"
            "High explore + creative/exploratory mode means the goal isn't to execute a known "
            "backlog. It's to **find the spark** within the focus domain, then crystallize it "
            "into a wave structure. So this plan does NOT prescribe one task — it lists candidate "
            "threads, gives a decision rule, and lets W0 pick.\n"
        )
    return (
        "## Frame — single-thread kapstok\n\n"
        "Focused/exploit mode + concrete intent: this plan prescribes one wave structure. "
        "`/niche` W0 reads it as the primary blueprint and proceeds. Wave deviation is allowed "
        "but must be logged with rationale per the Wave-deviation policy.\n"
    )


def _multithread_threads(carry_items: list[str], intent: str, focus: list[str], wildcard: bool) -> str:
    """Generate placeholder threads. Content is TODO; structure is correct.

    Letters: A = intent-derived primary, B-D = carry-items (1-3), E = focus-derived (if focus
    list has items), F = wildcard (if explore=5).
    """
    out: list[str] = ["## Candidate threads (the kapstok hooks)\n\n"]
    out.append(
        "Numbered for stable references. /niche W0 picks one; ranking below is the "
        "coordinator's read of resonance with today's mood and meta-framing. The thread "
        "content here is scaffolded by the kapstok writer — coordinator-direct fill-in "
        "is expected before W0 surfaces the menu to the user.\n\n"
    )

    letter = ord("A")

    out.append(
        f"### Thread {chr(letter)} — Intent-derived primary (★ default star)\n\n"
        f"**Why now**: directly serves the intent set during `/valuate`: \"{intent[:120]}\".\n\n"
        f"**Scope**: {_TODO}\n\n"
        f"**Acceptance**: {_TODO}\n\n"
        f"**Estimated waves**: {_TODO}\n\n"
    )
    letter += 1

    for item in carry_items[:3]:
        out.append(
            f"### Thread {chr(letter)} — Carry-item: {item[:100]}\n\n"
            "**Why now**: surfaced from valuate scratchpad as a deferred forward-look item.\n\n"
            f"**Scope**: {_TODO}\n\n"
            f"**Acceptance**: {_TODO}\n\n"
            f"**Estimated waves**: {_TODO}\n\n"
        )
        letter += 1

    if focus:
        focus_str = ", ".join(focus[:3])
        out.append(
            f"### Thread {chr(letter)} — Focus-derived: {focus_str[:80]}\n\n"
            "**Why now**: amplifies the focus directive for this terminal.\n\n"
            f"**Scope**: {_TODO}\n\n"
            f"**Acceptance**: {_TODO}\n\n"
            f"**Estimated waves**: {_TODO}\n\n"
        )
        letter += 1

    if wildcard:
        out.append(
            "### Thread F — Wild card: invent something new\n\n"
            "**Why now**: high explore. Sometimes the best contribution is one we haven't seen yet. "
            "/niche W0 may spend 5–10 min brainstorming threads off this menu before picking.\n\n"
        )

    out.append(
        "## Decision rule for /niche W0\n\n"
        "1. Surface this menu to the user.\n"
        "2. State coordinator's read on resonance (which thread is most timely).\n"
        "3. Ask the user to pick a thread (A–F), pick a sequence, or describe a wildcard.\n"
        "4. Update Status line to `IN PROGRESS — chose Thread X` and rewrite W1+ as concrete waves.\n"
    )
    return "".join(out)


def _single_thread_waves(intent: str) -> str:
    return (
        "## Wave structure\n\n"
        "### W0 — Audit + spec-freeze (coordinator-direct or spec-writer)\n\n"
        f"- Read intent context: \"{intent[:120]}\"\n"
        "- Identify load-bearing files / decisions\n"
        f"- **Acceptance**: {_TODO}\n\n"
        "### W1+ — Implementation waves (coordinator-direct fill-in)\n\n"
        f"- {_TODO}\n\n"
        "### Final Wave — close-out\n\n"
        "- Write coordinator scratchpad with Markov 7/7\n"
        "- `/librarian-update` + `/ego-check` (parallel)\n"
        "- Commit (devops or coordinator-direct if `.claude/`-only)\n"
    )


def _anti_scope(suppress: list[str]) -> str:
    out = ["## Anti-scope (do NOT do these this session)\n\n"]
    if suppress:
        for item in suppress:
            out.append(f"- ❌ {item}\n")
    else:
        out.append(f"- ❌ {_TODO} (no suppress directive set; coordinator-direct review)\n")
    out.append(
        "- ❌ Multi-day refactor commitments — scope to one-session-shippable unless intent explicitly "
        "says otherwise.\n"
    )
    return "".join(out)


def _carry_items_section(carry_items: list[str]) -> str:
    if not carry_items:
        return ""
    out = ["## Carry-items (open from forward-look, deferrable)\n\n"]
    for item in carry_items:
        out.append(f"- {item}\n")
    return "".join(out)


def _gist(intent: str, multithread: bool, n_threads: int = 0) -> str:
    if multithread:
        body = (
            f"This plan is a kapstok — a structural framework `/niche` hangs work on, written by "
            f"`/valuate` to crystallize today's characteristic state. Multi-thread mode "
            f"({n_threads} candidate threads). Intent: \"{intent or '(none)'}\". Decision rule: "
            "W0 surfaces the menu, user picks, W1+ rewritten for the chosen thread. "
            "Threads are scaffolded with TODO markers — coordinator-direct fill-in expected before "
            "W0 surfaces. Carry-items (if any) are palate-cleansers, not the main thread."
        )
    else:
        body = (
            f"This plan is a kapstok — a structural framework `/niche` hangs work on, written by "
            f"`/valuate` to crystallize today's characteristic state. Single-thread mode. "
            f"Intent: \"{intent or '(none)'}\". W0 reads the wave structure as the primary blueprint "
            "and proceeds. Wave deviation allowed with rationale."
        )
    return f"## If you only read this section\n\n{body}\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_kapstok(
    supra_session_yaml: Path,
    valuate_scratchpad: Path,
    plans_dir: Path,
    date: str,
    *,
    force: bool = False,
) -> Optional[Path]:
    """Write a plan kapstok if conditions are met. See module docstring + spec.

    Returns Path on success, None on skip / failure (fail-open).
    """
    try:
        if not supra_session_yaml.exists():
            print(f"plan_kapstok: supra yaml not found: {supra_session_yaml}", file=sys.stderr)
            return None

        try:
            supra = yaml.safe_load(supra_session_yaml.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError) as e:
            print(f"plan_kapstok: supra yaml parse error: {e}", file=sys.stderr)
            return None
        if not isinstance(supra, dict):
            return None

        intent = (supra.get("intent") or "").strip()
        if not intent:
            return None

        intent_slug = slugify(intent[:80])
        if not intent_slug:
            print(f"plan_kapstok: slugify yielded empty for intent: {intent!r}", file=sys.stderr)
            return None

        try:
            plans_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"plan_kapstok: cannot create plans dir {plans_dir}: {e}", file=sys.stderr)
            return None

        if not force:
            existing = list(plans_dir.glob(f"{date}-*{intent_slug}*.md"))
            if existing:
                return None

        supra_session_id = (supra.get("supra_session_id") or "").strip()
        multithread = _is_multithread(supra)
        carry_items = _carry_items_from_scratchpad(valuate_scratchpad, supra_session_id)

        try:
            explore_int = int((supra.get("dimensions") or {}).get("exploration_vs_exploitation", 3))
        except (TypeError, ValueError):
            explore_int = 3
        wildcard = multithread and explore_int >= 5

        focus = list(supra.get("focus") or [])
        title = _intent_to_title(intent, date)

        sections: list[str] = [
            f"# {title}\n\n",
            _status_table(supra_session_id, intent, multithread),
            "\n## Reference frame (echoed from supra session for cold-start resume)\n\n",
            _ref_frame_block(supra),
            f"\nIf `/clear` lands you here cold: the supra yaml is `{supra_session_yaml}`. "
            f"The valuate scratchpad is `{valuate_scratchpad}`. Read both for full state.\n\n",
            _frame_paragraph(multithread),
            "\n",
        ]

        if multithread:
            threads_md = _multithread_threads(carry_items, intent, focus, wildcard)
            sections.append(threads_md)
            n_threads = threads_md.count("### Thread ")
        else:
            sections.append(_single_thread_waves(intent))
            n_threads = 0

        sections.extend([
            "\n",
            _anti_scope(list(supra.get("suppress") or [])),
            "\n",
            _carry_items_section(carry_items),
            "\n" if carry_items else "",
            _peer_pointer(supra_session_yaml, supra_session_id),
            "\n",
            _gist(intent, multithread, n_threads=n_threads),
        ])
        body = "".join(sections)

        out_path = plans_dir / f"{date}-{intent_slug}.md"
        if out_path.exists() and not force:
            short_hash = hashlib.sha1(intent.encode("utf-8")).hexdigest()[:2]
            out_path = plans_dir / f"{date}-{intent_slug}-{short_hash}.md"

        try:
            out_path.write_text(body, encoding="utf-8")
        except OSError as e:
            print(f"plan_kapstok: write failed for {out_path}: {e}", file=sys.stderr)
            return None
        return out_path

    except Exception as exc:
        print(f"plan_kapstok: unexpected error: {exc}", file=sys.stderr)
        return None
