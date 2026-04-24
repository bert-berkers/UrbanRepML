"""
Audit sidecar coverage across stage3 probe + viz outputs for a study area.

Purpose:
    Walk stage3_analysis output directories, classify every artifact into one
    of four buckets — covered, missing, ghost-sidecar, sidecar-without-ledger-row
    — and print a structured coverage report to stdout.  Non-zero exit code only
    on ghost-sidecars or unexplained invariant mismatches.

Lifetime: temporary (30-day shelf life from 2026-04-24; expire ~2026-05-24).
Stage: stage3 provenance audit.

Part of cluster-2 ledger-sidecars plan, Wave 4 (W4 qaqc gate).
See specs/artifact_provenance.md §qaqc invariants for the three invariants
this script verifies.

Usage:
    uv run python scripts/one_off/audit_sidecar_coverage.py [--study-area {area}]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from utils.paths import StudyAreaPaths  # noqa: E402
from utils.provenance import read_ledger  # noqa: E402


# ---------------------------------------------------------------------------
# Artifact classification helpers
# ---------------------------------------------------------------------------

# Extensions that identify a substantive artifact (not sidecars, not dirs)
_ARTIFACT_EXTENSIONS = {
    ".csv", ".parquet", ".json", ".png", ".jpg", ".jpeg", ".html",
    ".pkl", ".pickle", ".npy", ".npz", ".geojson",
}

# Sidecar suffixes we recognise (in priority order)
_SIDECAR_SUFFIXES = (".run.yaml", ".provenance.yaml")


def _is_sidecar(p: Path) -> bool:
    """Return True if this path is a sidecar file (not an artifact)."""
    name = p.name
    return any(name.endswith(sfx) for sfx in _SIDECAR_SUFFIXES)


def _is_artifact(p: Path) -> bool:
    """Return True if this path is a substantive artifact to audit."""
    if _is_sidecar(p):
        return False
    # Accept any file with a known extension; also accept extensionless files
    # inside probe run dirs that are not sidecars (e.g. lock files — excluded).
    suffix = p.suffix.lower()
    return suffix in _ARTIFACT_EXTENSIONS


def _sidecar_for(artifact: Path) -> tuple[Path | None, str | None]:
    """Return (sidecar_path, sidecar_type) for an artifact, or (None, None)."""
    for sfx in _SIDECAR_SUFFIXES:
        candidate = Path(str(artifact) + sfx)
        if candidate.exists():
            return candidate, sfx
    return None, None


def _artifact_for_sidecar(sidecar: Path) -> Path | None:
    """Return the artifact that *sidecar* is a sibling of, or None if not found."""
    for sfx in _SIDECAR_SUFFIXES:
        if sidecar.name.endswith(sfx):
            artifact = Path(str(sidecar)[: -len(sfx)])
            return artifact if artifact.exists() else None
    return None


def _read_run_id_from_sidecar(sidecar: Path) -> str | None:
    """Parse the run_id field from a sidecar YAML, or None on error."""
    try:
        with open(sidecar, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("run_id") if isinstance(data, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Walk + classify
# ---------------------------------------------------------------------------

def _walk_stage3(stage3_root: Path) -> tuple[
    list[Path],  # artifacts
    list[Path],  # sidecars
]:
    """Recursively collect artifacts and sidecars under stage3_root."""
    artifacts: list[Path] = []
    sidecars: list[Path] = []

    if not stage3_root.is_dir():
        return artifacts, sidecars

    for p in stage3_root.rglob("*"):
        if not p.is_file():
            continue
        if _is_sidecar(p):
            sidecars.append(p)
        elif _is_artifact(p):
            artifacts.append(p)

    return artifacts, sidecars


def run_audit(study_area: str) -> int:
    """
    Walk stage3 output dirs, classify artifacts, print report.

    Returns exit code:
      0 — OK (ghost=0, invariant holds or is explained)
      1 — ghost-sidecars found, or unexplained invariant mismatch
    """
    paths = StudyAreaPaths(study_area, project_root=PROJECT_ROOT)
    stage3_root = paths.stage3("")  # data/study_areas/{area}/stage3_analysis/

    print(f"Sidecar coverage audit — study area: {study_area}")
    print("=" * 52)

    # ------------------------------------------------------------------
    # 1. Collect artifacts and sidecars
    # ------------------------------------------------------------------
    artifacts, sidecars = _walk_stage3(stage3_root)

    total_artifacts = len(artifacts)
    print(f"Scanned {total_artifacts} artifacts across stage3_analysis/\n")

    # ------------------------------------------------------------------
    # 2. Load ledger
    # ------------------------------------------------------------------
    ledger_df = read_ledger()
    ledger_run_ids: set[str] = set()
    if not ledger_df.empty and "run_id" in ledger_df.columns:
        ledger_run_ids = set(ledger_df["run_id"].dropna().tolist())

    # ------------------------------------------------------------------
    # 3. Classify artifacts
    # ------------------------------------------------------------------
    covered_artifacts: list[Path] = []
    covered_probes: list[Path] = []
    covered_figures: list[Path] = []
    missing_artifacts: list[Path] = []

    for artifact in artifacts:
        sidecar, sfx = _sidecar_for(artifact)
        if sidecar is not None:
            covered_artifacts.append(artifact)
            if sfx == ".run.yaml":
                covered_probes.append(artifact)
            else:
                covered_figures.append(artifact)
        else:
            missing_artifacts.append(artifact)

    # ------------------------------------------------------------------
    # 4. Classify sidecars
    # ------------------------------------------------------------------
    ghost_sidecars: list[Path] = []
    sidecars_without_ledger_row: list[Path] = []

    for sidecar in sidecars:
        # Ghost sidecar: sidecar exists but pointed-at artifact doesn't
        artifact_back = _artifact_for_sidecar(sidecar)
        if artifact_back is None:
            ghost_sidecars.append(sidecar)
            continue

        # Sidecar-without-ledger-row: sidecar has a run_id not in ledger
        # (only applies to .run.yaml — figure .provenance.yaml have no run_id)
        if sidecar.name.endswith(".run.yaml"):
            run_id = _read_run_id_from_sidecar(sidecar)
            if run_id is not None and run_id not in ledger_run_ids:
                sidecars_without_ledger_row.append(sidecar)

    # ------------------------------------------------------------------
    # 5. Print results
    # ------------------------------------------------------------------
    n_covered = len(covered_artifacts)
    n_missing = len(missing_artifacts)
    n_ghost = len(ghost_sidecars)
    n_no_ledger = len(sidecars_without_ledger_row)
    n_sidecars_on_disk = len(sidecars)

    print(f"  covered:                   {n_covered:4d}  (probes: {len(covered_probes)}, figures: {len(covered_figures)})")
    print(f"  missing (legacy, expected):{n_missing:4d}  (pre-W2 artifacts)")
    print(f"  ghost-sidecar:             {n_ghost:4d}")
    print(f"  sidecar-without-ledger-row:{n_no_ledger:4d}", end="")
    if n_no_ledger > 0:
        print("  <- WRITE-SIDE FAILURE SIGNATURE")
    else:
        print()

    # ------------------------------------------------------------------
    # 6. Sample outputs
    # ------------------------------------------------------------------
    if missing_artifacts:
        print(f"\nSample of missing ({min(5, n_missing)} of {n_missing}):")
        for p in missing_artifacts[:5]:
            try:
                rel = p.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = p
            print(f"  {rel}")

    if ghost_sidecars:
        print(f"\nSample of ghost-sidecars ({min(5, n_ghost)} of {n_ghost}):")
        for sidecar in ghost_sidecars[:5]:
            try:
                rel = sidecar.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = sidecar
            print(f"  {rel}")

    if sidecars_without_ledger_row:
        print(f"\nSample of sidecar-without-ledger-row ({min(5, n_no_ledger)} of {n_no_ledger}):")
        for sidecar in sidecars_without_ledger_row[:5]:
            try:
                rel = sidecar.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = sidecar
            run_id = _read_run_id_from_sidecar(sidecar)
            print(f"  {rel}")
            print(f"    run_id={run_id}")

    # ------------------------------------------------------------------
    # 7. Ledger summary
    # ------------------------------------------------------------------
    n_ledger = len(ledger_df)
    print()
    if n_ledger == 0:
        ledger_newest = "n/a"
        ledger_oldest = "n/a"
    else:
        if "started_at" in ledger_df.columns:
            ledger_newest = str(ledger_df.iloc[0]["started_at"])[:16]
            ledger_oldest = str(ledger_df.iloc[-1]["started_at"])[:16]
        else:
            ledger_newest = ledger_oldest = "n/a"

    print(f"Ledger: {n_ledger} rows, newest={ledger_newest}, oldest={ledger_oldest}")
    print(f"Stage3 sidecars on disk: {n_sidecars_on_disk}")

    # ------------------------------------------------------------------
    # 8. Invariant check
    # ------------------------------------------------------------------
    print(f"\nInvariant check (len(read_ledger()) == stage3 sidecar count):")

    # The invariant compares ledger rows against on-disk stage3 .run.yaml sidecars.
    # Ledger rows from pytest/temp dirs are NOT stage3 sidecars on disk, so we
    # compare only against sidecars found in stage3_root.
    stage3_run_yaml_count = sum(1 for s in sidecars if s.name.endswith(".run.yaml"))

    # Ledger rows that point into the stage3_root (filter out temp/pytest rows)
    if not ledger_df.empty and "sidecar_path" in ledger_df.columns:
        stage3_root_str = str(stage3_root).replace("\\", "/")
        stage3_ledger_rows = ledger_df[
            ledger_df["sidecar_path"].fillna("").str.replace("\\", "/").str.contains(
                stage3_root_str.replace("C:", "").lstrip("/"), regex=False
            )
        ]
    else:
        stage3_ledger_rows = ledger_df.iloc[0:0]  # empty

    n_stage3_ledger = len(stage3_ledger_rows)

    if n_stage3_ledger == stage3_run_yaml_count:
        print(f"  PASS — {n_stage3_ledger} ledger rows == {stage3_run_yaml_count} stage3 .run.yaml sidecars")
    else:
        diff = abs(n_stage3_ledger - stage3_run_yaml_count)
        if diff == n_no_ledger and n_ghost == 0:
            print(
                f"  FAIL (explained) — {n_stage3_ledger} stage3 ledger rows vs "
                f"{stage3_run_yaml_count} sidecars. "
                f"Difference ({diff}) attributable to {n_no_ledger} sidecar-without-ledger-row "
                f"(write-side failure signature)."
            )
        else:
            print(
                f"  FAIL (unexplained) — {n_stage3_ledger} stage3 ledger rows vs "
                f"{stage3_run_yaml_count} sidecars. "
                f"diff={diff}, ghost={n_ghost}, without_row={n_no_ledger}"
            )

    # ------------------------------------------------------------------
    # 9. Exit code
    # ------------------------------------------------------------------
    # Exit 1: ghost-sidecars found, or unexplained invariant mismatch
    unexplained_mismatch = (
        n_stage3_ledger != stage3_run_yaml_count
        and not (abs(n_stage3_ledger - stage3_run_yaml_count) == n_no_ledger and n_ghost == 0)
    )
    if n_ghost > 0 or unexplained_mismatch:
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit sidecar coverage across stage3 probe + viz outputs."
    )
    parser.add_argument(
        "--study-area",
        default="netherlands",
        help="Study area name (default: netherlands).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = run_audit(args.study_area)
    sys.exit(exit_code)
