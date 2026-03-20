#!/usr/bin/env python3
"""SubagentStart hook: inject scratchpad protocol + coordinator/ego context + own continuity into every specialist."""
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

SCRATCHPAD_ROOT = Path(__file__).resolve().parents[1] / "scratchpad"
COORDINATORS_DIR = Path(__file__).resolve().parents[1] / "coordinators"

# Lines of context to inject from each scratchpad source
CONTEXT_LINES = 5

# Pipeline adjacency graph: which agents are "neighbors" in the processing pipeline.
# When spawning an agent, we scan its neighbors' scratchpads for critical signals.
# "*" means the agent sees signals from ALL other agents.
PIPELINE_ADJACENCY = {
    "stage1-modality-encoder": ["stage2-fusion-architect", "srai-spatial"],
    "stage2-fusion-architect": ["stage1-modality-encoder", "stage3-analyst", "srai-spatial"],
    "stage3-analyst": ["stage2-fusion-architect"],
    "srai-spatial": ["stage1-modality-encoder", "stage2-fusion-architect", "stage3-analyst"],
    "qaqc": ["*"],
    "librarian": ["*"],
    "ego": ["*"],
}

# Signal keywords that propagate between pipeline-adjacent agents.
# Agents should emit these in their scratchpads when relevant.
SIGNAL_KEYWORDS = [
    "BLOCKED", "URGENT", "CRITICAL", "BROKEN",
    "SHAPE_CHANGED", "INTERFACE_CHANGED", "DEPRECATED", "NEEDS_TEST",
]


def last_lines(path: Path, n: int = CONTEXT_LINES) -> str:
    """Return last n non-empty lines from a file, or empty string if missing."""
    if not path.exists():
        return ""
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return "\n".join(lines[-n:])


def latest_entry(agent_dir: Path) -> Path | None:
    """Find the most recent dated .md file in an agent's scratchpad directory.

    Matches both daily entries (YYYY-MM-DD.md) and suffixed entries
    (YYYY-MM-DD-forward-look.md, etc.).
    """
    if not agent_dir.is_dir():
        return None
    entries = sorted(agent_dir.glob("????-??-??*.md"), reverse=True)
    return entries[0] if entries else None


def staleness_warning(entry: Path | None, label: str) -> str | None:
    """Return a warning string if the entry is >3 days old, else None."""
    if not entry:
        return None
    try:
        entry_date = date.fromisoformat(entry.stem[:10])
        days_old = (date.today() - entry_date).days
        if days_old > 3:
            return f"  (stale: {label} is {days_old} days old -- treat as low-confidence)"
    except ValueError:
        pass
    return None


def get_sibling_signals(agent_type: str) -> list[str]:
    """Scan pipeline-adjacent agents' today scratchpads for signal keywords.

    This implements Levin's "pervasive signaling" -- bioelectric gradients
    propagate along the pipeline adjacency graph, so stage3-analyst sees
    SHAPE_CHANGED from stage2-fusion-architect, etc.
    """
    neighbors = PIPELINE_ADJACENCY.get(agent_type)
    if not neighbors:
        return []

    today_str = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    signals = []

    # Determine which agent directories to scan
    if "*" in neighbors:
        scan_dirs = [
            d for d in sorted(SCRATCHPAD_ROOT.iterdir())
            if d.is_dir() and d.name != agent_type
        ]
    else:
        scan_dirs = [
            SCRATCHPAD_ROOT / n for n in neighbors
            if (SCRATCHPAD_ROOT / n).is_dir()
        ]

    for agent_dir in scan_dirs:
        for date_str in (today_str, yesterday_str):
            entry = agent_dir / f"{date_str}.md"
            if not entry.exists():
                continue
            try:
                content = entry.read_text(encoding="utf-8")
            except Exception:
                continue
            content_upper = content.upper()
            for keyword in SIGNAL_KEYWORDS:
                if re.search(r'\b' + keyword + r'\b', content_upper):
                    for line in content.splitlines():
                        if re.search(r'\b' + keyword + r'\b', line.upper()) and line.strip():
                            signals.append(
                                f"- **{agent_dir.name}** ({date_str}): {line.strip()[:120]}"
                            )
                            break  # One match per keyword per agent per date

    return signals


def get_coordinator_messages() -> list[str]:
    """Return recent coordinator messages addressed to this session or 'all'.

    Fires every wave (via SubagentStart), giving continuous lateral awareness.
    Silently returns empty list on any error.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import coordinator_registry as cr

        my_session_id = cr.read_ppid_session(COORDINATORS_DIR) or ""
        if not my_session_id:
            return []

        # Read messages from last 2 hours addressed to us or "all"
        from datetime import datetime
        since = datetime.now() - timedelta(hours=2)
        messages = cr.read_messages(
            COORDINATORS_DIR, since=since, to_session=my_session_id
        )

        if not messages:
            return []

        lines = ["", "### Coordinator Messages (lateral signals):"]
        for msg in messages[-5:]:  # Cap at 5 most recent
            sender = msg.get("from", "unknown")
            level = msg.get("level", "info")
            body = msg.get("body", "")[:150]
            lines.append(f"- **[{level}]** from `{sender}`: {body}")
        return lines
    except Exception as exc:
        print(f"subagent-context: coordinator message read failed: {exc}", file=sys.stderr)
        return []


def get_other_coordinator_note() -> list[str]:
    """Return lines noting any active coordinator sessions other than this one.

    Silently returns empty list on any error.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import coordinator_registry as cr

        my_session_id = cr.read_ppid_session(COORDINATORS_DIR) or ""

        all_claims = cr.read_all_claims(COORDINATORS_DIR)
        other_active = [
            c for c in all_claims
            if c.get("session_id") != my_session_id and not cr.is_stale(c)
        ]

        if not other_active:
            return []

        lines = ["", "### Other Active Coordinator Sessions:"]
        for claim in other_active:
            sid = claim.get("session_id", "unknown")
            summary = claim.get("task_summary", "no summary")
            paths = claim.get("claimed_paths", [])
            paths_str = ", ".join(str(p) for p in paths[:4])
            if len(paths) > 4:
                paths_str += f" (+{len(paths) - 4} more)"
            lines.append(f"- **{sid}**: {summary} | claiming: {paths_str}")
        lines.append(
            "Avoid modifying files claimed by other coordinators without coordinator approval."
        )
        return lines
    except Exception as exc:
        print(f"subagent-context: coordinator claim check failed: {exc}", file=sys.stderr)
        return []


def main() -> None:
    hook_input = json.loads(sys.stdin.read())
    agent_type = hook_input.get("agent_type", "unknown")
    today = date.today().isoformat()

    # Build context injection
    parts = [
        f"## Scratchpad Protocol (auto-injected by SubagentStart hook)",
        f"**Today's date**: {today}",
        f"**Your agent type**: {agent_type}",
        f"**Your scratchpad path**: `.claude/scratchpad/{agent_type}/{today}.md`",
        "",
        "Before returning, you MUST write a scratchpad entry containing:",
        "- `<!-- SUMMARY: one-line summary of what you did -->` (first line, machine-extractable)",
        "- **What I did**: actions taken, files modified, decisions made",
        "- **Cross-agent observations**: what you read from other agents, what was useful/confusing",
        "- **Unresolved**: each item tagged `[open]`, `[stale]`, or `[blocked:reason]`",
        "",
        "**Consolidation**: If today's entry exists, READ it first, then REWRITE as a single coherent",
        "log — not append. Reconcile existing Unresolved items against reality before adding new ones.",
        "Use summary tables over narrative when listing multiple items.",
        "For large outputs (code, data, reports), write to files and reference by path — don't paste inline.",
        "Keep entries under 80 lines. If you need more, you're writing too much detail.",
    ]

    # Inject ego's latest assessment
    ego_entry = latest_entry(SCRATCHPAD_ROOT / "ego")
    if ego_entry:
        ego_tail = last_lines(ego_entry)
        if ego_tail:
            parts.extend(["", f"### Ego's latest assessment ({ego_entry.name}):", ego_tail])
            stale = staleness_warning(ego_entry, "ego assessment")
            if stale:
                parts.append(stale)

    # Inject coordinator's latest entry
    coord_entry = latest_entry(SCRATCHPAD_ROOT / "coordinator")
    if coord_entry:
        coord_tail = last_lines(coord_entry)
        if coord_tail:
            parts.extend(["", f"### Coordinator's latest entry ({coord_entry.name}):", coord_tail])
            stale = staleness_warning(coord_entry, "coordinator entry")
            if stale:
                parts.append(stale)

    # Inject the specialist's OWN most recent scratchpad for behavioral anchoring / continuity
    own_entry = latest_entry(SCRATCHPAD_ROOT / agent_type)
    if own_entry:
        own_tail = last_lines(own_entry)
        if own_tail:
            parts.extend(["", f"### Your last scratchpad ({own_entry.name}):", own_tail])
            stale = staleness_warning(own_entry, "your last entry")
            if stale:
                parts.append(stale)

    # Inject other active coordinator sessions (claim awareness)
    coord_note = get_other_coordinator_note()
    parts.extend(coord_note)

    # Inject supra precision weights + graph mode filtered for this agent type
    # Also: load supra_reader once here so we can call is_lateral_coupling_active() below
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import supra_reader
        supra_states = supra_reader.read_session_states()
        schema = supra_reader.read_schema()
        if supra_states and schema:
            agent_weights = supra_reader.format_for_agent(supra_states, schema, agent_type)
            if agent_weights:
                parts.extend(["", "<!-- SUPRA_WEIGHTS -->", "### Human's Precision Weights:", agent_weights])
        # Wave 0: Inject graph mode into every agent's context
        graph_mode = supra_reader.get_active_graph()
        if graph_mode == "static":
            parts.extend([
                "", f"**Graph mode**: static (valuating)",
                "You are crystallizing governance — the characteristic state that will direct all future work in this strand.",
            ])
        else:
            parts.extend([
                "", f"**Graph mode**: dynamic (niche)",
                "You are leaving traces of a process that vanishes when this context window dies. Make the invisible visible.",
            ])
    except Exception as exc:
        print(f"subagent-context: supra state read failed: {exc}", file=sys.stderr)

    # Inject supra session identity (deterministic narrator ID across percept deaths)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import coordinator_registry as cr
        supra_sid = cr.read_ppid_supra(COORDINATORS_DIR)
        if supra_sid:
            parts.append(f"**Supra session**: `{supra_sid}` (narrator identity for /sync)")
    except Exception:
        pass

    # Determine lateral coupling state ONCE — used to gate both sibling signals and
    # coordinator messages. Defaults to True (fail-open) on any error.
    lateral_active = True
    try:
        lateral_active = supra_reader.is_lateral_coupling_active()
    except Exception:
        pass

    # Inject pipeline-adjacent signals (Levin's pervasive signaling)
    # Only active during dynamic graph (niche construction)
    if lateral_active:
        sibling_signals = get_sibling_signals(agent_type)
        if sibling_signals:
            parts.extend([
                "",
                "<!-- SIGNALS -->",
                "### Pipeline Signals (from adjacent agents):",
                *sibling_signals,
            ])

    # Inject unread coordinator messages (lateral awareness every wave)
    # Only active during dynamic graph (niche construction)
    if lateral_active:
        coord_messages = get_coordinator_messages()
        parts.extend(coord_messages)

    # Inject mortality awareness (context window = lifespan)
    try:
        _hooks = str(Path(__file__).resolve().parent)
        if _hooks not in sys.path:
            sys.path.insert(0, _hooks)
        import agent_timer
        agent_id = datetime.now().strftime("%H%M%S")
        timer = agent_timer.birth(agent_type, agent_id)
        parts.append(agent_timer.format_mortality_context(timer))
    except Exception as exc:
        print(f"subagent-context: timer registration failed: {exc}", file=sys.stderr)

    # Wave 4: Inject strand history — position in niche + prior agent summaries
    try:
        import agent_timer as _at
        alive_agents = _at.alive()
        dead_agents = _at.recent_dead(n=10)
        # Count today's agents (alive + dead) for strand position
        today_dead = [d for d in dead_agents if d.get("born_at", "").startswith(today)]
        strand_position = len(alive_agents) + len(today_dead)
        parts.append(f"\n**Strand position**: You are agent {strand_position} of this niche.")
        if strand_position > 5:
            parts.append("Late in the strand — converge, don't explore.")
        # One-line summaries of prior sub-sessions today
        if today_dead:
            parts.append("\n<!-- STRAND_HISTORY -->")
            parts.append("### Strand history (this niche):")
            for i, d in enumerate(today_dead, 1):
                atype = d.get("agent_type", "?")
                born = d.get("born_at", "??:??")
                born_time = born[11:16] if len(born) > 16 else born
                # Extract summary from scratchpad if available
                summary = d.get("summary", atype)
                parts.append(f"{i}. [{atype}] {born_time} — {summary}")
    except Exception as exc:
        print(f"subagent-context: strand history failed: {exc}", file=sys.stderr)

    context = "\n".join(parts)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SubagentStart",
            "additionalContext": context,
        }
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
