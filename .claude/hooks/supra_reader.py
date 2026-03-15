#!/usr/bin/env python3
"""Shared library for reading/writing human characteristic states (supra layer)."""
import json, os, re, sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("supra_reader: PyYAML not available", file=sys.stderr)
    yaml = None  # type: ignore[assignment]

SUPRA_DIR = Path(__file__).resolve().parents[1] / "supra"
STATES_PATH = SUPRA_DIR / "characteristic_states.yaml"
SCHEMA_PATH = SUPRA_DIR / "schema.yaml"
SESSIONS_DIR = SUPRA_DIR / "sessions"
COORDINATORS_DIR = Path(__file__).resolve().parents[1] / "coordinators"
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

# -- Session-scoped states ---------------------------------------------------

def _current_session_id() -> str | None:
    """Read the current session ID via PPID-keyed file."""
    try:
        _hooks = str(Path(__file__).resolve().parent)
        if _hooks not in sys.path:
            sys.path.insert(0, _hooks)
        import coordinator_registry as cr
        return cr.read_ppid_session(COORDINATORS_DIR)
    except Exception:
        return None


def _session_states_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.yaml"


def read_session_states(session_id: str | None = None) -> dict:
    """Read session-scoped states, falling back to global characteristic_states.yaml."""
    if session_id is None:
        session_id = _current_session_id()
    if session_id:
        session_data = _read_yaml(_session_states_path(session_id))
        if session_data:
            return session_data
    return read_states()


def write_session_states(states: dict, session_id: str | None = None) -> bool:
    """Write states to session-scoped file. Returns True on success."""
    if yaml is None:
        return False
    if session_id is None:
        session_id = _current_session_id()
    if not session_id:
        return write_states(states)
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = _session_states_path(session_id)
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.dump(states, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
        return True
    except Exception as exc:
        print(f"supra_reader: failed to write session states for {session_id}: {exc}", file=sys.stderr)
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
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

# -- Compound state detection ------------------------------------------------

def detect_compound_state(effective_dims: dict, schema: dict) -> dict | None:
    """Check if effective dimensions match a compound state within +/-1 tolerance.

    Returns the matching compound state dict (with name, description, tension_warning)
    or None. If multiple match, returns the one with the most dimensions specified.
    """
    compounds = schema.get("compound_states", {})
    if not compounds:
        return None

    best_match: dict | None = None
    best_specificity = 0

    for name, spec in compounds.items():
        target_dims = spec.get("dimensions", {})
        if not target_dims:
            continue

        all_match = True
        for dim_name, target_val in target_dims.items():
            eff_val = effective_dims.get(dim_name)
            if eff_val is None:
                all_match = False
                break
            try:
                if abs(int(eff_val) - int(target_val)) > 1:
                    all_match = False
                    break
            except (TypeError, ValueError):
                all_match = False
                break

        if all_match and len(target_dims) > best_specificity:
            best_specificity = len(target_dims)
            best_match = {
                "name": name,
                "description": spec.get("description", ""),
                "tension_warning": spec.get("tension_warning"),
                "mode": spec.get("mode"),
                "dimensions": dict(target_dims),
            }

    return best_match


# -- Formatting --------------------------------------------------------------

def _staleness(last_attuned) -> str:
    if not last_attuned:
        return "never (stale -- consider running /valuate)"
    try:
        parsed = datetime.fromisoformat(str(last_attuned))
        # Normalise to UTC-aware so naive/aware subtraction never raises TypeError
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = datetime.now(tz=timezone.utc) - parsed
        if age < timedelta(hours=24):
            return f"{last_attuned} (fresh)"
        return f"{last_attuned} (stale -- {age.days}d ago, consider /valuate)"
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
    # Compound state detection
    compound = detect_compound_state(eff_dims, schema)
    if compound:
        cs_line = f"**Compound state**: {compound['name']} -- {compound['description']}"
        lines.append(cs_line)
        if compound.get("tension_warning"):
            lines.append(f"**Warning**: {compound['tension_warning']}")
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


# -- Temporal segment & supra session identity --------------------------------

TEMPORAL_PRIORS_PATH = SUPRA_DIR / "temporal_priors.yaml"
# Legacy singleton removed — now PPID-keyed via coordinator_registry
GRAPH_JSON_PATH = Path(__file__).resolve().parents[2] / "deepresearch" / "liveability_approaches_graph.json"

TIME_BUCKETS = [
    ("morning", 6, 12),    # 06:00-11:59
    ("afternoon", 12, 17), # 12:00-16:59
    ("evening", 17, 22),   # 17:00-21:59
    ("night", 22, 6),      # 22:00-05:59 (wraps around midnight)
]

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _temporal_segment_key() -> str:
    """Return current temporal segment key like 'friday-evening' from local time."""
    now = datetime.now()
    day = DAYS[now.weekday()]
    hour = now.hour
    bucket = "night"  # default fallback
    for name, start, end in TIME_BUCKETS:
        if name == "night":
            # Night wraps around midnight: 22-23 or 0-5
            if hour >= start or hour < end:
                bucket = name
                break
        else:
            if start <= hour < end:
                bucket = name
                break
    return f"{day}-{bucket}"


def _supra_session_id() -> str:
    """Return supra session ID like 'hushed-spinning-glen-2026-03-14'.

    The supra session = this terminal. The first coordinator to start names it.
    Subsequent /clear cycles within the same terminal join it via
    _current_supra_session_id(). Format: {coordinator_poetic_name}-{date}.
    """
    coordinator_id = _current_session_id()
    if not coordinator_id:
        coordinator_id = "unnamed"
    return f"{coordinator_id}-{date.today().isoformat()}"


def _current_supra_session_id() -> str | None:
    """Read the current supra session ID via PPID-keyed file."""
    try:
        _hooks = str(Path(__file__).resolve().parent)
        if _hooks not in sys.path:
            sys.path.insert(0, _hooks)
        import coordinator_registry as cr
        return cr.read_ppid_supra(COORDINATORS_DIR)
    except Exception:
        return None


def _supra_session_path(sid: str) -> Path:
    """Return terminal-PID-keyed path for a supra session file.

    Format: .claude/supra/sessions/{sid}.{terminal_pid}.yaml
    The key is the terminal shell PID (stable across /clear), not the direct
    parent PID. Falls back to non-PPID path if the keyed file doesn't exist (migration).
    """
    _hooks = str(Path(__file__).resolve().parent)
    if _hooks not in sys.path:
        sys.path.insert(0, _hooks)
    try:
        import coordinator_registry as cr
        ppid = cr.get_terminal_pid()
    except Exception:
        ppid = os.getppid()
    ppid_path = SESSIONS_DIR / f"{sid}.{ppid}.yaml"
    if ppid_path.is_file():
        return ppid_path
    # Fallback: try legacy non-PPID path
    legacy_path = SESSIONS_DIR / f"{sid}.yaml"
    if legacy_path.is_file():
        return legacy_path
    # New file: use PPID-keyed path
    return ppid_path


def read_supra_session_states(supra_session_id: str | None = None) -> dict:
    """Read supra session file, falling back to coordinator session then global.

    Priority: supra_session_id param -> _current_supra_session_id() ->
    _current_session_id() -> global characteristic_states.yaml.
    """
    # Try the provided or detected supra session ID first
    sid = supra_session_id or _current_supra_session_id()
    if sid:
        data = _read_yaml(_supra_session_path(sid))
        if data:
            return data
    # Fall back to coordinator session ID
    coordinator_id = _current_session_id()
    if coordinator_id:
        data = _read_yaml(_session_states_path(coordinator_id))
        if data:
            return data
    # Final fallback: global states
    return read_states()


def write_supra_session_states(states: dict, supra_session_id: str | None = None) -> bool:
    """Write states to the supra session file. Returns True on success.

    If no supra_session_id is provided, computes it from _supra_session_id().
    Creates the sessions directory if needed.
    """
    if yaml is None:
        return False
    sid = supra_session_id or _current_supra_session_id() or _supra_session_id()
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = _supra_session_path(sid)
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.dump(states, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
        return True
    except Exception as exc:
        print(f"supra_reader: failed to write supra session states for {sid}: {exc}", file=sys.stderr)
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return False


# -- Graph-driven orchestration -----------------------------------------------


def get_active_graph() -> str:
    """Return 'static' or 'dynamic' from the supra session file.

    Static = during /valuate (setting weights).
    Dynamic = during /niche (executing work).
    Stored per-supra (PPID-isolated) so multi-terminal doesn't clash.
    Defaults to 'dynamic' if no supra session exists.
    """
    try:
        supra_sid = _current_supra_session_id()
        if supra_sid:
            data = _read_yaml(SESSIONS_DIR / f"{supra_sid}.yaml")
            val = data.get("active_graph", "dynamic")
            if val in ("static", "dynamic"):
                return val
    except Exception:
        pass
    return "dynamic"


def set_active_graph(graph: str) -> None:
    """Write 'static' or 'dynamic' into the supra session file (PPID-isolated)."""
    if graph not in ("static", "dynamic"):
        raise ValueError(f"graph must be 'static' or 'dynamic', got {graph!r}")
    try:
        supra_sid = _current_supra_session_id()
        if not supra_sid:
            print("supra_reader: no supra session, cannot set active graph", file=sys.stderr)
            return
        path = SESSIONS_DIR / f"{supra_sid}.yaml"
        data = _read_yaml(path)
        if not data:
            data = {"supra_session_id": supra_sid}
        data["active_graph"] = graph
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(path))
    except Exception as exc:
        print(f"supra_reader: failed to write active graph: {exc}", file=sys.stderr)


def get_edge_topology(graph: str | None = None) -> list[dict]:
    """Load edges from the liveability_approaches_graph.json for the active (or specified) graph.

    Returns the 'edges' list from the JSON's 'static' or 'dynamic' section.
    Returns empty list on any error.
    """
    target = graph or get_active_graph()
    try:
        raw = GRAPH_JSON_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data.get(target, {}).get("edges", [])
    except Exception as exc:
        print(f"supra_reader: failed to load edge topology for {target}: {exc}", file=sys.stderr)
        return []


def is_lateral_coupling_active() -> bool:
    """Return True during /niche (dynamic graph), False during /valuate (static graph).

    Controls whether /sync messages are sent/received.
    """
    return get_active_graph() == "dynamic"


# -- Temporal prior store -----------------------------------------------------

_TEMPORAL_PRIOR_DEFAULT: dict = {
    "version": 1,
    "learning_rate": 0.3,
    "min_observations": 2,
    "segments": {},
}


def read_temporal_priors() -> dict:
    """Read temporal_priors.yaml, returning defaults if the file is missing or unreadable."""
    data = _read_yaml(TEMPORAL_PRIORS_PATH)
    if not data:
        return dict(_TEMPORAL_PRIOR_DEFAULT)
    # Ensure required keys exist
    result = dict(_TEMPORAL_PRIOR_DEFAULT)
    result.update(data)
    if not isinstance(result.get("segments"), dict):
        result["segments"] = {}
    return result


def get_temporal_prior(segment: str | None = None) -> dict | None:
    """Get the prior dict for a temporal segment.

    If segment is None, uses the current temporal segment key.
    Returns the segment's 'prior' dict only if observations >= min_observations.
    Returns None if the segment is unknown or has insufficient observations.
    """
    seg = segment or _temporal_segment_key()
    try:
        priors = read_temporal_priors()
        min_obs = priors.get("min_observations", 2)
        seg_data = priors.get("segments", {}).get(seg)
        if not seg_data:
            return None
        if seg_data.get("observations", 0) >= min_obs:
            return seg_data.get("prior")
        return None
    except Exception as exc:
        print(f"supra_reader: failed to get temporal prior for {seg}: {exc}", file=sys.stderr)
        return None


def record_temporal_observation(states: dict, segment: str | None = None) -> bool:
    """Record a valuation observation and update the EMA prior for the segment.

    Creates the segment entry if it does not exist. EMA update:
        prior[dim] = (1 - alpha) * prior[dim] + alpha * v_new

    Mode: sliding window (last 5), majority vote; ties broken by most recent.
    Focus/suppress are NOT stored in temporal priors.

    Returns True on success.
    """
    if yaml is None:
        return False
    seg = segment or _temporal_segment_key()
    try:
        priors = read_temporal_priors()
        alpha = float(priors.get("learning_rate", 0.3))
        segments = priors.get("segments", {})
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        new_dims = states.get("dimensions", {})
        new_mode = states.get("mode", "focused")

        if seg not in segments:
            # Cold start: first observation
            segments[seg] = {
                "observations": 1,
                "first_seen": now_iso,
                "last_seen": now_iso,
                "prior": {
                    "mode": new_mode,
                    "dimensions": {k: float(v) for k, v in new_dims.items()},
                },
                "mode_history": [new_mode],
            }
        else:
            entry = segments[seg]
            entry["observations"] = entry.get("observations", 0) + 1
            entry["last_seen"] = now_iso

            # EMA update on each dimension
            prior_dims = entry.get("prior", {}).get("dimensions", {})
            for dim, v_new in new_dims.items():
                try:
                    v_new_f = float(v_new)
                    old = float(prior_dims.get(dim, v_new_f))
                    prior_dims[dim] = (1 - alpha) * old + alpha * v_new_f
                except (TypeError, ValueError):
                    pass
            # Ensure any dims present in prior but not in new_dims are preserved
            if "prior" not in entry:
                entry["prior"] = {}
            entry["prior"]["dimensions"] = prior_dims

            # Mode: sliding window majority vote, tiebreak = most recent
            history = list(entry.get("mode_history", []))
            history.append(new_mode)
            history = history[-5:]  # keep last 5
            entry["mode_history"] = history

            # Majority vote: count occurrences, ties broken by last occurrence
            counts: dict[str, int] = {}
            for m in history:
                counts[m] = counts.get(m, 0) + 1
            max_count = max(counts.values())
            # Among tied modes, pick the one that appeared most recently
            winner = new_mode  # default to most recent
            for m in reversed(history):
                if counts[m] == max_count:
                    winner = m
                    break
            entry["prior"]["mode"] = winner

        priors["segments"] = segments

        # Atomic write
        TEMPORAL_PRIORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = TEMPORAL_PRIORS_PATH.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.dump(priors, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(TEMPORAL_PRIORS_PATH))
        return True
    except Exception as exc:
        print(f"supra_reader: failed to record temporal observation for {seg}: {exc}", file=sys.stderr)
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return False


def temporal_prior_to_states(prior: dict) -> dict:
    """Convert a temporal prior dict to a states dict for use as session defaults.

    Rounds float dimensions to integers (round half up).
    Returns a dict in the same shape as characteristic_states.yaml:
    {mode, dimensions, focus=[], suppress=[]}.
    """
    import math

    dims_raw = prior.get("dimensions", {})
    dims_int = {}
    for k, v in dims_raw.items():
        try:
            # Round half up: math.floor(x + 0.5)
            dims_int[k] = int(math.floor(float(v) + 0.5))
        except (TypeError, ValueError):
            dims_int[k] = 3  # fallback to neutral

    return {
        "mode": prior.get("mode", "focused"),
        "dimensions": dims_int,
        "focus": [],
        "suppress": [],
    }
