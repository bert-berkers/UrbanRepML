#!/usr/bin/env python3
"""Shared library for reading/writing human characteristic states (supra layer)."""
import os, re, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("supra_reader: PyYAML not available", file=sys.stderr)
    yaml = None  # type: ignore[assignment]

SUPRA_DIR = Path(__file__).resolve().parents[1] / "supra"
STATES_PATH = SUPRA_DIR / "characteristic_states.yaml"
SCHEMA_PATH = SUPRA_DIR / "schema.yaml"
LEVEL_LABELS = {1: "SUPPRESS", 2: "LOW", 3: "NEUTRAL", 4: "HIGH", 5: "AMPLIFY"}

# -- Core I/O ---------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    """Read a YAML file, returning empty dict on any error."""
    if yaml is None or not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"supra_reader: failed to read {path}: {exc}", file=sys.stderr)
        return {}

def read_states() -> dict:
    """Parse characteristic_states.yaml."""
    return _read_yaml(STATES_PATH)

def read_schema() -> dict:
    """Parse schema.yaml."""
    return _read_yaml(SCHEMA_PATH)

def write_states(states: dict) -> bool:
    """Write updated states to characteristic_states.yaml. Returns True on success."""
    if yaml is None:
        return False
    try:
        SUPRA_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = STATES_PATH.with_suffix(".yaml.tmp")
        tmp_path.write_text(
            yaml.dump(states, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(STATES_PATH))
        return True
    except Exception as exc:
        print(f"supra_reader: failed to write {STATES_PATH}: {exc}", file=sys.stderr)
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return False

# -- Mode bias application ---------------------------------------------------

def apply_mode_biases(states: dict, schema: dict) -> dict:
    """Apply mode biases to raw dimensions (clamped [1,5]). Adds 'effective_dimensions' key."""
    dims = states.get("dimensions", {})
    biases = schema.get("modes", {}).get(states.get("mode", ""), {}).get("biases", {})
    effective = {}
    for name, raw in dims.items():
        try:
            r = int(raw)
        except (TypeError, ValueError):
            r = 3
        effective[name] = max(1, min(5, r + biases.get(name, 0)))
    return {**states, "effective_dimensions": effective}

# -- Formatting --------------------------------------------------------------

def _staleness(last_attuned) -> str:
    if not last_attuned:
        return "never (stale -- consider running /attune)"
    try:
        parsed = datetime.fromisoformat(str(last_attuned))
        # Normalise to UTC-aware so naive/aware subtraction never raises TypeError
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = datetime.now(tz=timezone.utc) - parsed
        if age < timedelta(hours=24):
            return f"{last_attuned} (fresh)"
        return f"{last_attuned} (stale -- {age.days}d ago, consider /attune)"
    except (ValueError, TypeError):
        return f"{last_attuned} (unparseable)"

def _arrow_label(eff: int, low: str, high: str) -> str:
    if eff <= 2:
        return f"<- {low}"
    if eff >= 4:
        return f"{high} ->"
    return "(neutral)"

def format_for_coordinator(states: dict, schema: dict) -> str:
    """Full attentional landscape as markdown for the coordinator."""
    enriched = apply_mode_biases(states, schema)
    eff_dims = enriched.get("effective_dimensions", {})
    raw_dims = states.get("dimensions", {})
    mode = states.get("mode", "unknown")
    mode_desc = schema.get("modes", {}).get(mode, {}).get("description", "no description")
    dim_defs = schema.get("dimensions", {})
    lines = [
        f"**Mode**: {mode} -- {mode_desc}",
        "**Dimensions** (effective, after mode biases):",
        "| Dimension | Raw | Effective | Label |",
        "|-----------|-----|-----------|-------|",
    ]
    for name in raw_dims:
        raw = raw_dims.get(name, 3)
        eff = eff_dims.get(name, raw)
        d = dim_defs.get(name, {})
        lines.append(f"| {name} | {raw} | {eff} | {_arrow_label(eff, d.get('low_label', 'low'), d.get('high_label', 'high'))} |")
    focus, suppress = states.get("focus", []), states.get("suppress", [])
    focus_str = ', '.join(f'"{item}"' for item in focus) if focus else "(none)"
    lines.append(f"**Focus**: {focus_str}")
    lines.append(f"**Suppress**: {', '.join(f'\"{s}\"' for s in suppress) if suppress else '(none)'}")
    lines.append(f"**Last attuned**: {_staleness(states.get('last_attuned'))}")
    return "\n".join(lines)

def format_for_agent(states: dict, schema: dict, agent_type: str) -> str:
    """Filtered view for a specific agent. Dimensions with relevance >= 0.5 only."""
    enriched = apply_mode_biases(states, schema)
    eff_dims = enriched.get("effective_dimensions", {})
    dim_defs = schema.get("dimensions", {})
    mode = states.get("mode", "unknown")
    mode_desc = schema.get("modes", {}).get(mode, {}).get("description", "")
    lines = ["The human's current precision weights for your work:"]
    for name, eff in eff_dims.items():
        d = dim_defs.get(name, {})
        if d.get("agent_relevance", {}).get(agent_type, 0) < 0.5:
            continue
        level = LEVEL_LABELS.get(eff, "NEUTRAL")
        guidance = d.get("low_label", "") if eff <= 2 else (d.get("high_label", "") if eff >= 4 else "")
        suffix = f" -- {guidance}" if guidance else ""
        lines.append(f"- {name}: {eff}/5 ({level}){suffix}")
    lines.append(f"Mode: {mode} -- {mode_desc}")
    focus, suppress = states.get("focus", []), states.get("suppress", [])
    if focus:
        lines.append(f"Focus: {', '.join(f'\"{f}\"' for f in focus)}")
    if suppress:
        lines.append(f"Suppress: {', '.join(f'\"{s}\"' for s in suppress)}")
    return "\n".join(lines)

# -- Named profiles ----------------------------------------------------------

PROFILES_DIR = SUPRA_DIR / "profiles"


def list_profiles() -> list[str]:
    """Return sorted list of saved profile names."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def save_profile(name: str, states: dict) -> bool:
    """Save current states as a named profile. Returns True on success."""
    if yaml is None:
        return False
    try:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "mode": states.get("mode", "exploratory"),
            "dimensions": dict(states.get("dimensions", {})),
            "focus": list(states.get("focus", [])),
            "suppress": list(states.get("suppress", [])),
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        path = PROFILES_DIR / f"{name}.yaml"
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.dump(snapshot, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
        return True
    except Exception as exc:
        print(f"supra_reader: failed to save profile {name}: {exc}", file=sys.stderr)
        return False


def load_profile(name: str) -> dict | None:
    """Load a named profile. Returns states dict or None if not found."""
    path = PROFILES_DIR / f"{name}.yaml"
    data = _read_yaml(path)
    return data if data else None


# -- Dimension recommendation ------------------------------------------------

_STOP_WORDS = frozenset(
    "session sessions scratchpad coordinator agent this that with from have been "
    "were will which their about would should could does done more than into also "
    "next still items needs must".split()
)

def recommend_dimensions(schema: dict, scratchpad_root: Path) -> list[dict]:
    """Scan ego's latest scratchpad for patterns suggesting new dimensions."""
    ego_dir = scratchpad_root / "ego"
    if not ego_dir.is_dir():
        return []
    ego_files = sorted(ego_dir.glob("*.md"), reverse=True)
    if not ego_files:
        return []
    try:
        text = ego_files[0].read_text(encoding="utf-8")
    except Exception:
        return []

    existing = set((schema.get("dimensions") or {}).keys())
    suggestions: list[dict] = []

    # "N sessions deferred/carried" -> urgency dimension
    deferred = re.findall(r"(\d+)\+?\s+sessions?\s+(?:deferred|unexecuted|flagged|carried)", text, re.I)
    if deferred and "urgency" not in existing:
        n = max(int(x) for x in deferred)
        suggestions.append({"name": "urgency",
            "description": "How aggressively to prioritize long-deferred items",
            "reason": f"Ego shows items deferred up to {n} sessions"})
    # BLOCKED signal -> unblocking dimension
    if re.search(r"\bBLOCKED\b", text) and "unblocking" not in existing:
        suggestions.append({"name": "unblocking",
            "description": "Priority of resolving blocked dependencies vs working around them",
            "reason": "Ego scratchpad contains BLOCKED signals"})
    # Frequent topics not covered by existing dimensions
    counts: dict[str, int] = {}
    for w in re.findall(r"\b([a-z_]{4,})\b", text.lower()):
        if w not in existing and w not in _STOP_WORDS:
            counts[w] = counts.get(w, 0) + 1
    seen = {s["name"] for s in suggestions}
    for topic, ct in sorted(counts.items(), key=lambda x: -x[1])[:3]:
        if ct >= 8 and topic not in seen:
            suggestions.append({"name": topic,
                "description": f"Frequently mentioned in ego assessment ({ct} occurrences)",
                "reason": f"'{topic}' appears {ct} times but has no matching dimension"})
    return suggestions
