"""
Figure-provenance wrapper for stage3 viz.

Writes a sibling ``{figure_path}.provenance.yaml`` next to every saved figure,
pointing back at the source probe ``run_id``s and capturing the plot config.

See ``specs/artifact_provenance.md`` §Figure-provenance specialisation — this
module implements the ``*.provenance.yaml`` sidecar layer for stage3 figures
under the W2b sub-wave of plan
``.claude/plans/2026-04-18-cluster2-ledger-sidecars.md``.

Fields written:
    source_runs       — list of upstream probe run_ids
    source_artifacts  — list of files read to produce the plot
    plot_config       — free-form dict of plot settings (dpi, top_n, etc.)
    git_commit        — full 40-char SHA from ``git rev-parse HEAD``
    git_dirty         — bool from ``git status --porcelain``
    started_at        — ISO 8601 UTC (figure-write is effectively instantaneous)
    ended_at          — same (started == ended)
    producer_script   — relative path of the calling script/module
    schema_version    — "1.0"

Non-JSON-serialisable values in ``plot_config`` are coerced via ``repr()`` with
a warning — consistent with ``utils.provenance.compute_config_hash``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import yaml

SCHEMA_VERSION = "1.0"

_PROJECT_ROOT: Optional[Path] = None


def _project_root() -> Path:
    """Locate project root by walking up until pyproject.toml is found."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                _PROJECT_ROOT = current
                return _PROJECT_ROOT
            current = current.parent
        # fail-open: return the stage3_analysis parent
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    return _PROJECT_ROOT


def _relativise(p: Union[Path, str]) -> str:
    """Return *p* relative to project root as a forward-slash string."""
    try:
        return Path(p).resolve().relative_to(_project_root()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _git_info() -> tuple[Optional[str], Optional[bool]]:
    """Return (full_commit_sha, is_dirty). Both None if not in a git repo."""
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


def _coerce_plot_config(d: dict) -> dict:
    """Stringify non-JSON-serialisable values in plot_config with a warning."""
    out: dict = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            warnings.warn(
                f"save_figure: plot_config[{k!r}] value of type "
                f"{type(v).__name__!r} is not JSON-serialisable; coercing via repr().",
                UserWarning,
                stacklevel=3,
            )
            out[k] = repr(v)
    return out


def save_figure(
    fig,
    path: Union[Path, str],
    sources: Optional[list[str]] = None,
    *,
    source_artifacts: Optional[list[Union[Path, str]]] = None,
    plot_config: Optional[dict] = None,
    producer_script: Optional[str] = None,
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: Optional[str] = None,
) -> Path:
    """Save *fig* and a sibling ``{path}.provenance.yaml`` provenance sidecar.

    Args:
        fig: matplotlib Figure.
        path: target path for the image file (PNG/PDF/SVG).
        sources: list of upstream probe ``run_id`` strings whose data backs this
            figure. Empty list (or None) emits a warning — a figure with no
            declared sources has weak provenance.
        source_artifacts: optional explicit list of files read to produce the
            plot (e.g. ``predictions_{target}.parquet``). Redundant with
            ``sources`` but useful for content-hash checks.
        plot_config: free-form dict of plot settings (colormap, top_n, etc.).
            Non-JSON-serialisable values are coerced via ``repr()`` with a
            warning.
        producer_script: override for the script path (default: caller's
            ``sys.argv[0]`` relativized to project root).
        dpi: passed through to ``fig.savefig``.
        bbox_inches: passed through to ``fig.savefig``.
        facecolor: optional, passed through to ``fig.savefig`` if not None.

    Returns:
        The image path (same as *path*).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    savefig_kwargs = {"dpi": dpi, "bbox_inches": bbox_inches}
    if facecolor is not None:
        savefig_kwargs["facecolor"] = facecolor
    fig.savefig(path, **savefig_kwargs)

    if not sources:
        warnings.warn(
            f"save_figure: no source run_ids declared for {path.name!r}; "
            f"figure provenance is weak.",
            UserWarning,
            stacklevel=2,
        )
        sources = list(sources) if sources else []
    else:
        sources = list(sources)

    source_artifacts_rel: list[str] = []
    if source_artifacts:
        source_artifacts_rel = [_relativise(p) for p in source_artifacts]

    plot_config_safe = _coerce_plot_config(plot_config or {})

    git_commit, git_dirty = _git_info()

    if producer_script is not None:
        producer_script_rel = producer_script
    else:
        try:
            producer_script_rel = _relativise(sys.argv[0]) if sys.argv else None
        except Exception:
            producer_script_rel = sys.argv[0] if sys.argv else None

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    provenance = {
        "source_runs": sources,
        "source_artifacts": source_artifacts_rel,
        "plot_config": plot_config_safe,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "started_at": now,  # figure-write is effectively instantaneous
        "ended_at": now,
        "producer_script": producer_script_rel,
        "schema_version": SCHEMA_VERSION,
    }

    sidecar = path.with_suffix(path.suffix + ".provenance.yaml")
    with sidecar.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            provenance,
            fh,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )

    return path
