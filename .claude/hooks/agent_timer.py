#!/usr/bin/env python3
"""Agent timer registry: tracks agent lifespans so they know how long they have to live.

An agent's lifespan IS its context window. When spawned, agents receive their birth time
and token budget. When they die (SubagentStop), their death is recorded. The monitor
script (called via /loop) reports on who's alive, who's old, and who died.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

TIMERS_DIR = Path(__file__).resolve().parents[1] / "timers"

# Context window estimates per model tier (tokens).
# These are approximate usable budgets after system prompt overhead.
CONTEXT_BUDGETS = {
    "opus": 180_000,
    "sonnet": 180_000,
    "haiku": 180_000,
    "default": 180_000,
}

# Rough token-per-minute burn rates by agent behavior pattern.
# Exploration-heavy agents burn faster (lots of file reads).
# Implementation agents burn moderate (read + write).
# Lightweight agents burn slow (focused tasks).
BURN_PROFILES = {
    "Explore": "fast",       # ~8k tokens/min (lots of reads)
    "librarian": "fast",
    "qaqc": "moderate",      # ~5k tokens/min
    "stage2-fusion-architect": "moderate",
    "stage1-modality-encoder": "moderate",
    "stage3-analyst": "moderate",
    "srai-spatial": "moderate",
    "spec-writer": "slow",   # ~3k tokens/min (thinking-heavy)
    "ego": "slow",
    "devops": "moderate",
    "execution": "fast",
    "general-purpose": "moderate",
}

BURN_RATES = {
    "fast": 8000,
    "moderate": 5000,
    "slow": 3000,
}


def ensure_dir():
    TIMERS_DIR.mkdir(parents=True, exist_ok=True)


def timer_path(agent_type: str, agent_id: str) -> Path:
    """Path for a specific agent timer file."""
    return TIMERS_DIR / f"{agent_type}--{agent_id}.json"


def _read_supra_weights() -> dict:
    """Read effective supra weights. Returns empty dict on failure (fail-open)."""
    try:
        hooks_dir = str(Path(__file__).resolve().parent)
        if hooks_dir not in sys.path:
            sys.path.insert(0, hooks_dir)
        import supra_reader
        states = supra_reader.read_states()
        schema = supra_reader.read_schema()
        if states and schema:
            enriched = supra_reader.apply_mode_biases(states, schema)
            return enriched.get("effective_dimensions", {})
    except Exception:
        pass
    return {}


def _modulate_from_weights(
    budget: int, weights: dict
) -> tuple[int, tuple[int, int, int]]:
    """Adjust budget and lifecycle phases based on supra precision weights.

    Returns (adjusted_budget, (explore_pct, work_pct, wrap_pct)).

    execution_speed: 5 = tighter budget (0.7x), 1 = generous (1.3x)
    exploration_vs_exploitation: shifts explore/work balance
    """
    speed = weights.get("execution_speed", 3)
    explore = weights.get("exploration_vs_exploitation", 3)

    # Budget scaling: speed 1->1.3x, 3->1.0x, 5->0.7x
    budget_scale = 1.3 - 0.15 * (speed - 1)
    adjusted_budget = round(budget * budget_scale)

    # Lifecycle phases: exploration weight shifts explore vs work
    # explore=1 -> 15/65/20, explore=3 -> 30/50/20, explore=5 -> 45/35/20
    explore_pct = 15 + round(7.5 * (explore - 1))
    wrap_pct = 20
    work_pct = 100 - explore_pct - wrap_pct

    return adjusted_budget, (explore_pct, work_pct, wrap_pct)


def birth(agent_type: str, agent_id: str = "", model_tier: str = "default") -> dict:
    """Record an agent's birth. Returns the timer data (for injection into context)."""
    ensure_dir()
    now = datetime.now()
    base_budget = CONTEXT_BUDGETS.get(model_tier, CONTEXT_BUDGETS["default"])
    profile = BURN_PROFILES.get(agent_type, "moderate")
    burn_rate = BURN_RATES[profile]

    # Read supra weights and modulate
    weights = _read_supra_weights()
    budget, phases = _modulate_from_weights(base_budget, weights)
    estimated_lifespan_min = budget / burn_rate

    timer = {
        "agent_type": agent_type,
        "agent_id": agent_id,
        "born_at": now.isoformat(timespec="seconds"),
        "died_at": None,
        "status": "alive",
        "context_budget_tokens": budget,
        "base_budget_tokens": base_budget,
        "burn_profile": profile,
        "burn_rate_tokens_per_min": burn_rate,
        "estimated_lifespan_min": round(estimated_lifespan_min, 1),
        "lifecycle_phases": {
            "explore_pct": phases[0],
            "work_pct": phases[1],
            "wrap_pct": phases[2],
        },
        "supra_modulation": {
            "execution_speed": weights.get("execution_speed", 3),
            "exploration_vs_exploitation": weights.get("exploration_vs_exploitation", 3),
        },
    }

    if agent_id:
        path = timer_path(agent_type, agent_id)
        path.write_text(json.dumps(timer, indent=2), encoding="utf-8")

    return timer


def death(agent_type: str, agent_id: str = "") -> dict | None:
    """Record an agent's death. Returns updated timer or None if not found."""
    if not agent_id:
        # Find most recent alive timer for this agent_type
        candidates = sorted(TIMERS_DIR.glob(f"{agent_type}--*.json"), reverse=True)
        for c in candidates:
            try:
                data = json.loads(c.read_text(encoding="utf-8"))
                if data.get("status") == "alive":
                    agent_id = data.get("agent_id", "")
                    break
            except Exception:
                continue
        if not agent_id:
            return None

    path = timer_path(agent_type, agent_id)
    if not path.exists():
        return None

    try:
        timer = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    now = datetime.now()
    born = datetime.fromisoformat(timer["born_at"])
    lived_min = (now - born).total_seconds() / 60

    timer["died_at"] = now.isoformat(timespec="seconds")
    timer["status"] = "dead"
    timer["lived_min"] = round(lived_min, 1)
    timer["estimated_tokens_used"] = round(lived_min * timer["burn_rate_tokens_per_min"])

    path.write_text(json.dumps(timer, indent=2), encoding="utf-8")
    return timer


def alive() -> list[dict]:
    """Return all currently alive agents."""
    ensure_dir()
    result = []
    for f in sorted(TIMERS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("status") == "alive":
                born = datetime.fromisoformat(data["born_at"])
                age_min = (datetime.now() - born).total_seconds() / 60
                data["age_min"] = round(age_min, 1)
                budget = data.get("context_budget_tokens", 180_000)
                burn = data.get("burn_rate_tokens_per_min", 5000)
                estimated_used = age_min * burn
                data["estimated_remaining_tokens"] = max(0, round(budget - estimated_used))
                data["estimated_remaining_pct"] = max(0, round(100 * (1 - estimated_used / budget)))
                result.append(data)
        except Exception:
            continue
    return result


def recent_dead(n: int = 5) -> list[dict]:
    """Return the N most recently deceased agents."""
    ensure_dir()
    dead = []
    for f in sorted(TIMERS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("status") == "dead":
                dead.append(data)
                if len(dead) >= n:
                    break
        except Exception:
            continue
    return dead


def cleanup(max_age_hours: int = 24):
    """Remove timer files older than max_age_hours."""
    ensure_dir()
    cutoff = datetime.now()
    for f in TIMERS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            ts = data.get("died_at") or data.get("born_at")
            if ts:
                age_hours = (cutoff - datetime.fromisoformat(ts)).total_seconds() / 3600
                if age_hours > max_age_hours:
                    f.unlink()
        except Exception:
            continue


def format_mortality_context(timer: dict) -> str:
    """Format timer data as context injection for the agent."""
    budget = timer["context_budget_tokens"]
    base = timer.get("base_budget_tokens", budget)
    burn = timer["burn_profile"]
    lifespan = timer["estimated_lifespan_min"]
    born = timer["born_at"]
    phases = timer.get("lifecycle_phases", {"explore_pct": 30, "work_pct": 50, "wrap_pct": 20})
    modulation = timer.get("supra_modulation", {})

    # Build modulation note if weights deviate from neutral
    speed = modulation.get("execution_speed", 3)
    explore = modulation.get("exploration_vs_exploitation", 3)
    mod_notes = []
    if speed != 3:
        direction = "tighter" if speed > 3 else "generous"
        mod_notes.append(f"execution_speed={speed} -> {direction} budget ({budget:,} vs {base:,} base)")
    if explore != 3:
        direction = "more exploration" if explore > 3 else "more exploitation"
        mod_notes.append(f"exploration={explore} -> {direction} ({phases['explore_pct']}% explore)")
    mod_line = f"**Supra modulation**: {'; '.join(mod_notes)}\n" if mod_notes else ""

    return (
        f"\n### Mortality Awareness (auto-injected)\n"
        f"**Born**: {born}\n"
        f"**Context budget**: ~{budget:,} tokens (your entire lifespan)\n"
        f"**Burn profile**: {burn} (~{timer['burn_rate_tokens_per_min']:,} tokens/min)\n"
        f"**Estimated lifespan**: ~{lifespan:.0f} minutes\n"
        f"{mod_line}"
        f"\n"
        f"Budget your work:\n"
        f"- **First {phases['explore_pct']}%**: Explore, read, orient\n"
        f"- **Middle {phases['work_pct']}%**: Do the work\n"
        f"- **Final {phases['wrap_pct']}%**: Write scratchpad, clean up, return results\n"
        f"- If you sense you're deep into a task, write your scratchpad NOW -- don't wait\n"
        f"- Prefer targeted reads over broad exploration\n"
    )


def format_monitor_report() -> str:
    """Format a status report for /loop monitoring."""
    lines = ["## Agent Vitals"]

    living = alive()
    if living:
        lines.append(f"\n**Alive** ({len(living)}):")
        for a in living:
            pct = a.get("estimated_remaining_pct", "?")
            age = a.get("age_min", "?")
            remaining = a.get("estimated_remaining_tokens", 0)
            bar = _life_bar(pct if isinstance(pct, (int, float)) else 50)
            lines.append(
                f"  {a['agent_type']:30s} {bar} {pct}% remaining  "
                f"(age: {age}m, ~{remaining:,} tokens left)"
            )
    else:
        lines.append("\n**No agents currently alive.**")

    dead = recent_dead(5)
    if dead:
        lines.append(f"\n**Recently deceased** ({len(dead)}):")
        for d in dead:
            lived = d.get("lived_min", "?")
            used = d.get("estimated_tokens_used", "?")
            lines.append(
                f"  {d['agent_type']:30s} lived {lived}m, "
                f"~{used:,} tokens consumed"
            )

    return "\n".join(lines)


def _life_bar(pct: float, width: int = 20) -> str:
    """Render a life bar: [========............]"""
    filled = max(0, min(width, round(width * pct / 100)))
    empty = width - filled
    if pct > 60:
        return f"[{'#' * filled}{'.' * empty}]"
    elif pct > 25:
        return f"[{'=' * filled}{'.' * empty}]"
    else:
        return f"[{'-' * filled}{'.' * empty}]"


if __name__ == "__main__":
    # CLI: python agent_timer.py monitor
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        print(format_monitor_report())
    elif len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup()
        print("Timer cleanup complete.")
    else:
        print("Usage: python agent_timer.py [monitor|cleanup]")