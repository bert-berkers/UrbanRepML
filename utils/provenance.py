"""
Artifact provenance: per-artifact sidecars + JSONL ledger.

Every stage3 artifact written by a probe or figure call gets a sibling
``*.run.yaml`` sidecar and a row in ``data/ledger/runs.jsonl``.  Figure write
sites use the narrower ``*.provenance.yaml`` specialisation instead.

See ``specs/artifact_provenance.md`` for the frozen schema, fail-mode rationales,
``config_hash`` algorithm, and ``run_id`` format.  This module implements W1 of
plan ``.claude/plans/2026-04-18-cluster2-ledger-sidecars.md``.

File-lock implementation: ``filelock`` (already a transitive dep via torch /
huggingface-hub).  Three-attempt, 100 ms backoff.  Raises ``IOError`` after
exhausting retries — per spec §Fail-mode 2, ledger integrity beats run
completion.
"""

import contextvars
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml
from filelock import FileLock, Timeout

# ---------------------------------------------------------------------------
# Active SidecarWriter registry (read by save_voronoi_figure for parent_run_id)
# ---------------------------------------------------------------------------
#
# When a ``SidecarWriter`` is active (``__enter__`` has fired but ``__exit__``
# has not), figure-write helpers can read this contextvar to auto-populate
# ``parent_run_id`` on emitted ``*.provenance.yaml`` siblings.  This makes the
# parent->figure edge in ``data/ledger/runs.jsonl`` walkable without manually
# threading ``run_id`` through every plot call.
#
# ContextVar (not threading.local) so async/await contexts behave correctly;
# nested ``SidecarWriter`` blocks restore the outer writer on inner exit via
# ``ContextVar.reset(token)``.
_ACTIVE_SIDECAR: "contextvars.ContextVar[Optional['SidecarWriter']]" = (
    contextvars.ContextVar("_ACTIVE_SIDECAR", default=None)
)


def get_active_sidecar() -> "Optional[SidecarWriter]":
    """Return the innermost currently-active :class:`SidecarWriter`, or None.

    Read by :func:`utils.visualization.save_voronoi_figure` to auto-populate
    ``parent_run_id`` on figure-provenance yaml siblings.  Callers outside the
    figure-provenance helper SHOULD NOT rely on this — pass ``run_id`` through
    explicit kwargs instead.  This is a leak-prevention fence: the contextvar
    is ``None`` whenever no writer is active.
    """
    return _ACTIVE_SIDECAR.get()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Optional[Path] = None


def _project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                _PROJECT_ROOT = current
                return _PROJECT_ROOT
            current = current.parent
        raise RuntimeError("Cannot locate project root — no pyproject.toml found")
    return _PROJECT_ROOT


def _relativise(p: Union[Path, str]) -> str:
    """Return path relative to project root as a forward-slash string."""
    try:
        return Path(p).resolve().relative_to(_project_root()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _stringify_with_warn(value: object) -> str:
    """Coerce non-JSON-serialisable leaf to repr string and emit a UserWarning.

    Kept as stringify-with-warn (not raise) so probes remain functional when
    a config contains Path or numpy scalars.  The warning surfaces to callers
    at stacklevel=3 so the frame shown is the call to compute_config_hash, not
    this helper.  Raise-on-non-serialisable is a planned stage-3 hardening step
    (see spec §config_hash, "strict" mode).
    """
    warnings.warn(
        f"config_hash: non-JSON-serialisable value {type(value).__name__!r} "
        f"coerced via repr(); hash may differ across Python versions.",
        UserWarning,
        stacklevel=3,
    )
    return repr(value)


def _git_info() -> tuple[Optional[str], Optional[bool]]:
    """Return (full_commit_sha, is_dirty).  Both None if not in a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(_project_root()),
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=str(_project_root()),
        ).decode().strip()
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _json_safe(v: object) -> object:
    """Make a value safe for json.dumps without a custom default."""
    if isinstance(v, Path):
        return v.as_posix()
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _make_json_safe(d: dict) -> dict:
    return {k: _json_safe(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Public API 1 — compute_config_hash
# ---------------------------------------------------------------------------

def compute_config_hash(cfg: dict) -> str:
    """Return the first 16 hex chars of SHA-256 over canonical-JSON of *cfg*.

    Canonical form: keys sorted recursively, no extra whitespace, ASCII-only.
    Non-JSON-serialisable leaves are coerced via :func:`_stringify_with_warn`.
    See ``specs/artifact_provenance.md`` §config_hash algorithm.
    """
    canonical = json.dumps(
        cfg,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_stringify_with_warn,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API 2 — SidecarWriter
# ---------------------------------------------------------------------------

_LEDGER_COLUMNS = [
    "run_id", "git_commit", "git_dirty", "config_hash", "config_path",
    "seed", "wall_time_seconds", "started_at", "ended_at", "producer_script",
    "study_area", "stage", "schema_version", "sidecar_path",
]


class SidecarWriter:
    """Context manager that writes a ``*.run.yaml`` sidecar for one artifact.

    Usage::

        with SidecarWriter(
            artifact_path=metrics_csv,
            config=cfg_dict,
            input_paths=[emb_path, target_path],
            producer_script="stage3_analysis/linear_probe.py",
            study_area="netherlands",
            stage="stage3",
            seed=42,
        ) as sw:
            sw.output_paths.append(metrics_csv)
            sw.extra["overall_r2"] = 0.53
            # ... do the actual work ...

    On clean exit: writes ``{artifact_path}.run.yaml`` then calls
    :func:`ledger_append`.  On exception: writes a failed sidecar with
    ``extra.status="failed"`` then re-raises — does NOT call ledger_append
    (per spec §Fail-mode 3 — failed sidecar is detectable by W4 audit as a
    distinct bucket from missing-sidecar-entirely).
    """

    def __init__(
        self,
        artifact_path: Union[Path, str],
        config: dict,
        input_paths: list,
        *,
        producer_script: Optional[str] = None,
        study_area: Optional[str] = None,
        stage: str = "stage3",
        seed: Optional[int] = None,
        config_path: Union[Path, str, None] = None,
        extra: Optional[dict] = None,
    ) -> None:
        self._artifact_path = Path(artifact_path)
        self._config = config
        self._input_paths = [_relativise(p) for p in input_paths]
        self._stage = stage
        self._seed = seed
        self._config_path = _relativise(config_path) if config_path else None
        self._study_area = study_area
        self._producer_script = producer_script
        self._extra_init = dict(extra) if extra else {}

        self.extra: dict = {}
        self.output_paths: list[Path] = []

        self._state: Optional[dict] = None
        self._started_at: Optional[datetime] = None
        # Token returned by ContextVar.set(); used to restore prior value on exit.
        self._active_token: Optional[contextvars.Token] = None

    # read-only so callers can pass sw.run_id to figure provenance
    @property
    def run_id(self) -> str:
        if self._state is None:
            raise RuntimeError("SidecarWriter.run_id accessed before __enter__")
        return self._state["run_id"]

    def __enter__(self) -> "SidecarWriter":
        self.extra = dict(self._extra_init)

        self._started_at = _now_utc()
        git_commit, git_dirty = _git_info()
        config_hash = compute_config_hash(self._config)

        # producer_script: use explicit arg, else infer from sys.argv[0]
        if self._producer_script is not None:
            producer_script = self._producer_script
        else:
            try:
                producer_script = _relativise(sys.argv[0])
            except Exception:
                producer_script = sys.argv[0] if sys.argv else "<unknown>"

        # producer name = stem of producer_script (for run_id)
        producer_name = Path(producer_script).stem.replace("-", "_")
        started_compact = self._started_at.strftime("%Y%m%dT%H%M%S")
        config_hash_short = config_hash[:8]
        run_id = f"{self._stage}-{producer_name}-{started_compact}-{config_hash_short}"

        self._state = {
            "run_id": run_id,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "config_hash": config_hash,
            "config_path": self._config_path,
            "seed": self._seed,
            "producer_script": producer_script,
            "study_area": self._study_area,
            "stage": self._stage,
        }
        # Register self as active writer; nested writers stack via ContextVar
        # tokens (inner exit restores the outer).  Read by helpers like
        # :func:`utils.visualization.save_voronoi_figure`.
        self._active_token = _ACTIVE_SIDECAR.set(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        tb: object,
    ) -> bool:
        # Always restore the active-writer contextvar — even on failure.  Nested
        # writers stack: this reset() pops *self* and exposes whatever writer
        # (if any) was active when *self* entered.  Guard against double-exit
        # or pre-enter exit (token is None).
        if self._active_token is not None:
            try:
                _ACTIVE_SIDECAR.reset(self._active_token)
            except (ValueError, LookupError):
                # Token from a different context; fall back to clearing.
                _ACTIVE_SIDECAR.set(None)
            self._active_token = None

        ended_at = _now_utc()
        wall_time = (ended_at - self._started_at).total_seconds()

        extra = dict(self.extra)

        if exc_type is not None:
            # fail-open per spec §Fail-mode 3 — original exception takes precedence
            extra["status"] = "failed"
            extra["exception_class"] = exc_type.__name__
            extra["exception_message"] = str(exc_val)[:500]
            try:
                self._write_sidecar(ended_at, wall_time, extra)
            except Exception as sidecar_err:
                print(
                    f"[SidecarWriter WARN] sidecar write failed during exception "
                    f"handling: {sidecar_err}",
                    file=sys.stderr,
                )
            # do NOT call ledger_append on failed runs — spec §Fail-mode 3
            return False  # propagate original exception

        # success path
        extra.setdefault("status", "success")
        sidecar_path = self._write_sidecar(ended_at, wall_time, extra)
        ledger_append(sidecar_path)
        return False

    def _write_sidecar(
        self,
        ended_at: datetime,
        wall_time: float,
        extra: dict,
    ) -> Path:
        sidecar_path = Path(str(self._artifact_path) + ".run.yaml")

        data = {
            "run_id": self._state["run_id"],
            "git_commit": self._state["git_commit"],
            "git_dirty": self._state["git_dirty"],
            "config_hash": self._state["config_hash"],
            "config_path": self._state["config_path"],
            "input_paths": self._input_paths,
            "output_paths": [_relativise(p) for p in self.output_paths],
            "seed": self._seed,
            "wall_time_seconds": round(wall_time, 3),
            "started_at": _iso(self._started_at),
            "ended_at": _iso(ended_at),
            "producer_script": self._state["producer_script"],
            "study_area": self._state["study_area"],
            "stage": self._state["stage"],
            "schema_version": "1.0",
            "extra": extra,
        }

        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)

        return sidecar_path


# ---------------------------------------------------------------------------
# Public API 3 — ledger_append
# ---------------------------------------------------------------------------

_LOCK_TIMEOUT_S = 0.1  # 100 ms per attempt
_LOCK_RETRIES = 3


def ledger_append(sidecar_path: Union[Path, str]) -> None:
    """Append one row to ``data/ledger/runs.jsonl`` from the given sidecar.

    Projects the 15 sidecar minimum fields minus ``input_paths`` and
    ``output_paths`` (too large for the ledger), plus ``sidecar_path``
    (relative to project root).

    Uses :class:`filelock.FileLock` with 3 × 100 ms retries before raising
    ``IOError`` — per spec §Fail-mode 2.

    Idempotence: if the ``run_id`` already appears in the last ~100 JSONL
    lines, the append is skipped and a stderr warning is emitted.
    """
    sidecar_path = Path(sidecar_path)

    with open(sidecar_path, encoding="utf-8") as fh:
        sidecar = yaml.safe_load(fh)

    row = {
        "run_id": sidecar.get("run_id"),
        "git_commit": sidecar.get("git_commit"),
        "git_dirty": sidecar.get("git_dirty"),
        "config_hash": sidecar.get("config_hash"),
        "config_path": sidecar.get("config_path"),
        "seed": sidecar.get("seed"),
        "wall_time_seconds": sidecar.get("wall_time_seconds"),
        "started_at": sidecar.get("started_at"),
        "ended_at": sidecar.get("ended_at"),
        "producer_script": sidecar.get("producer_script"),
        "study_area": sidecar.get("study_area"),
        "stage": sidecar.get("stage"),
        "schema_version": sidecar.get("schema_version"),
        "sidecar_path": _relativise(sidecar_path),
    }
    run_id = row["run_id"]

    ledger_dir = _project_root() / "data" / "ledger"
    ledger_path = ledger_dir / "runs.jsonl"
    lock_path = str(ledger_path) + ".lock"

    ledger_dir.mkdir(parents=True, exist_ok=True)

    # idempotence check — scan last ~100 lines before acquiring the lock
    if ledger_path.exists() and run_id:
        try:
            lines = ledger_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-100:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                    if existing.get("run_id") == run_id:
                        print(
                            f"[ledger_append WARN] run_id={run_id!r} already in "
                            f"ledger — skipping duplicate append.",
                            file=sys.stderr,
                        )
                        return
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass  # fail-open on idempotence check; proceed to write

    row_str = json.dumps(row, default=_json_safe, ensure_ascii=True) + "\n"

    lock = FileLock(lock_path, timeout=_LOCK_TIMEOUT_S)
    for attempt in range(1, _LOCK_RETRIES + 1):
        try:
            with lock.acquire(timeout=_LOCK_TIMEOUT_S):
                with open(ledger_path, "a", encoding="utf-8") as fh:
                    fh.write(row_str)
            return
        except Timeout:
            if attempt < _LOCK_RETRIES:
                time.sleep(_LOCK_TIMEOUT_S)
            else:
                raise IOError(
                    f"ledger_append: could not acquire lock after "
                    f"{_LOCK_RETRIES}× {int(_LOCK_TIMEOUT_S * 1000)} ms retry"
                )


# ---------------------------------------------------------------------------
# Public API 4 — read_ledger
# ---------------------------------------------------------------------------

def read_ledger(path: Union[Path, str, None] = None) -> pd.DataFrame:
    """Read ``data/ledger/runs.jsonl`` into a DataFrame, newest-first.

    Malformed rows are skipped with a stderr warning (fail-open per spec
    §Fail-mode 1).  Missing file returns an empty DataFrame with the 14 ledger
    column names.
    """
    if path is None:
        path = _project_root() / "data" / "ledger" / "runs.jsonl"
    path = Path(path)

    if not path.exists():
        return pd.DataFrame(columns=_LEDGER_COLUMNS)

    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                print(
                    f"[read_ledger WARN] line {n}: {err}",
                    file=sys.stderr,
                )
                continue
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_LEDGER_COLUMNS)

    df = pd.DataFrame(rows)
    if "started_at" in df.columns:
        df = df.sort_values("started_at", ascending=False).reset_index(drop=True)
    return df
