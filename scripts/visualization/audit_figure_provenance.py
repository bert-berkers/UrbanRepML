"""Audit figure-provenance coverage under data/study_areas/{area}/stage3_analysis/.

Lifetime: durable.
Stage: visualization toolkit / W4 audit (rasterize-voronoi-toolkit plan).

Lists every PNG/SVG figure under stage3_analysis/ subtrees and reports which
have a sibling ``*.provenance.yaml`` (covered) vs which don't (uncovered).

The provenance yaml schema is defined in:
    - specs/artifact_provenance.md §"Figure-provenance specialisation"
    - specs/rasterize_voronoi.md §"Provenance integration hook"

Most pre-W4 figures will report as uncovered — that is the expected baseline.
W4 does NOT backfill old figures; new figures generated post-W4 via
``utils.visualization.save_voronoi_figure`` (or the older
``stage3_analysis.save_figure.save_figure``) will emit the sibling yaml on
write.

Usage::

    python scripts/visualization/audit_figure_provenance.py
    python scripts/visualization/audit_figure_provenance.py --study-area netherlands
    python scripts/visualization/audit_figure_provenance.py --show-covered
    python scripts/visualization/audit_figure_provenance.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Make project root importable when run directly.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.paths import StudyAreaPaths

# Image extensions that constitute a "figure" for the audit.
_FIGURE_EXTS = (".png", ".svg", ".pdf")
# Companion suffix written by save_voronoi_figure / save_figure.
_PROV_SUFFIX = ".provenance.yaml"


@dataclass
class AuditResult:
    """Single-study-area audit result."""

    study_area: str
    total: int = 0
    covered: list[Path] = field(default_factory=list)
    uncovered: list[Path] = field(default_factory=list)

    @property
    def covered_count(self) -> int:
        return len(self.covered)

    @property
    def uncovered_count(self) -> int:
        return len(self.uncovered)

    @property
    def coverage_pct(self) -> float:
        return 100.0 * self.covered_count / self.total if self.total else 0.0


def _study_areas_root() -> Path:
    """Return the absolute path to ``data/study_areas/`` at project root."""
    # StudyAreaPaths exposes project_root via its __init__ logic; build a probe
    # path and walk back.  Using a known throwaway name avoids hardcoding.
    probe = StudyAreaPaths("__probe__")
    return probe.project_root / "data" / "study_areas"


def _list_study_areas() -> list[str]:
    """Discover existing study areas on disk."""
    root = _study_areas_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _scan_one(study_area: str) -> AuditResult:
    """Scan one study area's stage3_analysis/ subtree for figure coverage."""
    paths = StudyAreaPaths(study_area)
    stage3_root = paths.root / "stage3_analysis"
    result = AuditResult(study_area=study_area)
    if not stage3_root.exists():
        return result

    for ext in _FIGURE_EXTS:
        # rglob is recursive; scan all subdirs (probe runs, viz, plots, ...).
        for figure_path in stage3_root.rglob(f"*{ext}"):
            if not figure_path.is_file():
                continue
            sidecar = figure_path.with_suffix(figure_path.suffix + _PROV_SUFFIX)
            result.total += 1
            if sidecar.exists():
                result.covered.append(figure_path)
            else:
                result.uncovered.append(figure_path)

    result.covered.sort()
    result.uncovered.sort()
    return result


def _format_human(
    results: Iterable[AuditResult],
    *,
    show_covered: bool,
) -> str:
    """Render results as a human-readable text report."""
    lines: list[str] = []
    grand_total = 0
    grand_covered = 0
    grand_uncovered = 0

    for r in results:
        grand_total += r.total
        grand_covered += r.covered_count
        grand_uncovered += r.uncovered_count

        lines.append("")
        lines.append(f"=== Study area: {r.study_area} ===")
        if r.total == 0:
            lines.append("  (no figures found under stage3_analysis/)")
            continue

        lines.append(
            f"  total: {r.total}   "
            f"covered: {r.covered_count}   "
            f"uncovered: {r.uncovered_count}   "
            f"coverage: {r.coverage_pct:.1f}%"
        )

        if r.uncovered:
            lines.append("")
            lines.append(f"  Uncovered ({r.uncovered_count}):")
            for p in r.uncovered:
                lines.append(f"    - {p}")

        if show_covered and r.covered:
            lines.append("")
            lines.append(f"  Covered ({r.covered_count}):")
            for p in r.covered:
                lines.append(f"    + {p}")

    lines.append("")
    lines.append("=== TOTAL ===")
    pct = 100.0 * grand_covered / grand_total if grand_total else 0.0
    lines.append(
        f"  total: {grand_total}   "
        f"covered: {grand_covered}   "
        f"uncovered: {grand_uncovered}   "
        f"coverage: {pct:.1f}%"
    )
    return "\n".join(lines)


def _format_json(results: Iterable[AuditResult]) -> str:
    """Render results as a JSON document for machine consumption."""
    out = []
    for r in results:
        out.append({
            "study_area": r.study_area,
            "total": r.total,
            "covered_count": r.covered_count,
            "uncovered_count": r.uncovered_count,
            "coverage_pct": round(r.coverage_pct, 2),
            "uncovered": [str(p) for p in r.uncovered],
            "covered": [str(p) for p in r.covered],
        })
    return json.dumps({"results": out}, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit figure-provenance coverage under "
            "data/study_areas/{area}/stage3_analysis/."
        ),
    )
    parser.add_argument(
        "--study-area",
        default=None,
        help=(
            "Limit audit to a single study area (e.g. 'netherlands').  "
            "Default: all areas under data/study_areas/."
        ),
    )
    parser.add_argument(
        "--show-covered",
        action="store_true",
        help="Print the list of covered figures (default: only uncovered).",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit a JSON document instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    if args.study_area:
        areas = [args.study_area]
    else:
        areas = _list_study_areas()

    if not areas:
        print("No study areas found under data/study_areas/.", file=sys.stderr)
        return 1

    results = [_scan_one(a) for a in areas]

    if args.as_json:
        print(_format_json(results))
    else:
        print(_format_human(results, show_covered=args.show_covered))

    return 0


if __name__ == "__main__":
    sys.exit(main())
