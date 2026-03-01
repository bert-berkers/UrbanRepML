"""
Centralized path management for the three-stage pipeline.

Every data path that stage1, stage2, or stage3 code needs should be constructed
through StudyAreaPaths. This eliminates hardcoded path strings scattered across
14+ source files.

See specs/experiment_paths.md for the full directory convention and refactor table.
"""

import json
import re
import subprocess
from datetime import date, datetime
from pathlib import Path


class StudyAreaPaths:
    """Single source of truth for all data paths in the three-stage pipeline.

    Usage::

        from utils import StudyAreaPaths

        paths = StudyAreaPaths("netherlands")
        emb = pd.read_parquet(paths.embedding_file("alphaearth", 10))
        target = pd.read_parquet(paths.target_file("leefbaarometer", 10, 2022))

    The class does NOT create directories. Callers use
    ``path.mkdir(parents=True, exist_ok=True)`` as needed.
    """

    def __init__(self, study_area: str, project_root: "Path | str | None" = None):
        self.study_area = study_area
        self.project_root = Path(project_root) if project_root else _find_project_root()
        self.root = self.project_root / "data" / "study_areas" / study_area

    # -----------------------------------------------------------------
    # Stage 1: Unimodal embeddings
    # -----------------------------------------------------------------

    def stage1(self, modality: str) -> Path:
        """Base directory for a modality's embeddings."""
        return self.root / "stage1_unimodal" / modality

    def embedding_file(
        self,
        modality: str,
        resolution: int,
        year: int = 2022,
        sub_embedder: "str | None" = None,
    ) -> Path:
        """Parquet file for full-dimensional unimodal embeddings.

        Args:
            modality: Modality name (e.g. ``"poi"``, ``"alphaearth"``).
            resolution: H3 resolution integer.
            year: Data year (default ``2022``).
            sub_embedder: Optional sub-embedder name.  When provided, the file
                is placed in a subdirectory of the modality's stage1 dir.
                Example: ``sub_embedder="hex2vec"`` yields
                ``stage1_unimodal/poi/hex2vec/netherlands_res10_2022.parquet``.
        """
        base = self.stage1(modality)
        if sub_embedder:
            base = base / sub_embedder
        return base / f"{self.study_area}_res{resolution}_{year}.parquet"

    def pca_embedding_file(
        self, modality: str, resolution: int, n_components: int, year: int = 2022
    ) -> Path:
        """Parquet file for PCA-reduced unimodal embeddings."""
        return (
            self.stage1(modality)
            / f"{self.study_area}_res{resolution}_pca{n_components}_{year}.parquet"
        )

    def intermediate(self, modality: str) -> Path:
        """Directory for per-tile / per-chunk processing artifacts."""
        return self.stage1(modality) / "intermediate"

    # -----------------------------------------------------------------
    # Stage 2: Multimodal fusion
    # -----------------------------------------------------------------

    def stage2(self, model_name: str) -> Path:
        """Base directory for a fusion model's outputs."""
        return self.root / "stage2_multimodal" / model_name

    def checkpoints(self, model_name: str) -> Path:
        """Checkpoint directory for a fusion model."""
        return self.stage2(model_name) / "checkpoints"

    def model_embeddings(self, model_name: str) -> Path:
        """Fused embedding output directory for a model."""
        return self.stage2(model_name) / "embeddings"

    def fused_embedding_file(
        self, model_name: str, resolution: int, year: int = 2022
    ) -> Path:
        """Parquet file for fused multimodal embeddings."""
        return (
            self.model_embeddings(model_name)
            / f"{self.study_area}_res{resolution}_{year}.parquet"
        )

    def plots(self, model_name: str) -> Path:
        """Plot directory for a fusion model."""
        return self.stage2(model_name) / "plots"

    def training_logs(self, model_name: str) -> Path:
        """Training log directory for a fusion model."""
        return self.stage2(model_name) / "training_logs"

    # -----------------------------------------------------------------
    # Stage 3: Analysis
    # -----------------------------------------------------------------

    def stage3(self, analysis_type: str) -> Path:
        """Base directory for an analysis type (linear_probe, dnn_probe, etc.)."""
        return self.root / "stage3_analysis" / analysis_type

    # -----------------------------------------------------------------
    # Shared / stage-independent
    # -----------------------------------------------------------------

    def regions(self) -> Path:
        """Directory containing H3 tessellation parquet files."""
        return self.root / "regions_gdf"

    def region_file(self, resolution: int) -> Path:
        """Parquet file for H3 regions at a given resolution."""
        return self.regions() / f"{self.study_area}_res{resolution}.parquet"

    def area_gdf(self) -> Path:
        """Directory containing study area boundary files."""
        return self.root / "area_gdf"

    def area_gdf_file(self, fmt: str = "geojson") -> Path:
        """Boundary file for the study area."""
        return self.area_gdf() / f"{self.study_area}_boundary.{fmt}"

    def target(self, target_name: str) -> Path:
        """Base directory for a target dataset (e.g. leefbaarometer)."""
        return self.root / "target" / target_name

    def target_file(
        self, target_name: str, resolution: int, year: int
    ) -> Path:
        """Parquet file for an H3-indexed target variable."""
        return (
            self.target(target_name)
            / f"{target_name}_h3res{resolution}_{year}.parquet"
        )

    def cones(self) -> Path:
        """Base directory for hierarchical cone data."""
        return self.root / "cones"

    def cone_cache(self, parent_res: int, target_res: int) -> Path:
        """Directory containing individual cone pickle files."""
        return self.cones() / f"cone_cache_res{parent_res}_to_{target_res}"

    def cone_lookup(self, parent_res: int, target_res: int) -> Path:
        """Pickle file mapping parent hexagons to their cone children."""
        return (
            self.cones()
            / f"parent_to_children_res{parent_res}_to_{target_res}.pkl"
        )

    def osm_dir(self) -> Path:
        """Directory for historical OSM PBF files.

        Layout::

            data/study_areas/{area}/osm/
            ├── {area}-internal.osh.pbf      # Full OSM history extract
            ├── {area}-latest.osm.pbf        # Most recent snapshot
            └── {area}-2022-01-01.osm.pbf    # Date-specific snapshot

        Both POI and roads processors use files from this directory
        via SRAI's ``OSMPbfLoader``.
        """
        return self.root / "osm"

    def osm_history_pbf(self) -> Path:
        """Full OSM history PBF file (``*.osh.pbf``).

        This is the complete history extract from which date-specific
        snapshots are derived using ``osmium time-filter``.
        """
        return self.osm_dir() / f"{self.study_area}-internal.osh.pbf"

    def osm_snapshot_pbf(self, date: str = "latest") -> Path:
        """OSM PBF snapshot for a specific date.

        Args:
            date: Date string (e.g. ``"2022-01-01"``) or ``"latest"``
                for the most recent snapshot.

        Returns:
            Path to the PBF file.  The file may not exist on disk yet;
            the caller is responsible for downloading or extracting it.
        """
        if date == "latest":
            return self.osm_dir() / f"{self.study_area}-latest.osm.pbf"
        return self.osm_dir() / f"{self.study_area}-{date}.osm.pbf"

    def accessibility(self) -> Path:
        """Directory for accessibility graph artifacts."""
        return self.root / "accessibility"

    # -----------------------------------------------------------------
    # Run-level provenance
    # -----------------------------------------------------------------

    def stage1_run(self, modality: str, run_id: str) -> Path:
        """Run directory for a stage1 modality processing run."""
        return self.stage1(modality) / run_id

    def stage2_run(self, model_name: str, run_id: str) -> Path:
        """Run directory for a stage2 fusion model run."""
        return self.stage2(model_name) / run_id

    def stage3_run(self, analysis_type: str, run_id: str) -> Path:
        """Run directory for a stage3 analysis run."""
        return self.stage3(analysis_type) / run_id

    def latest_run(self, stage_dir: Path) -> "Path | None":
        """Find the most recent run directory by YYYY-MM-DD prefix sort.

        Args:
            stage_dir: Parent directory to search
                (e.g. ``paths.stage3("linear_probe")``).

        Returns:
            Path to most recent run directory, or None if no runs found.
        """
        if not stage_dir.is_dir():
            return None
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
        run_dirs = sorted(
            d for d in stage_dir.iterdir()
            if d.is_dir() and pattern.match(d.name)
        )
        return run_dirs[-1] if run_dirs else None

    def create_run_id(self, descriptor: str = "") -> str:
        """Generate a run ID string: ``YYYY-MM-DD_descriptor``.

        Does NOT create any directories.  Just returns the string.
        If *descriptor* is empty, returns just ``YYYY-MM-DD``.
        """
        today = date.today().isoformat()
        if descriptor:
            return f"{today}_{descriptor}"
        return today


def write_run_info(
    run_dir: Path,
    *,
    stage: str,
    study_area: str,
    config: dict,
    upstream_runs: "dict[str, str] | None" = None,
) -> Path:
    """Write ``run_info.json`` to a run directory.

    Captures the current git short hash automatically (``None`` if unavailable).
    Creates *run_dir* and any missing parents.

    Returns:
        Path to the written ``run_info.json``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Capture git hash, gracefully handle failure
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_hash = result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.SubprocessError, FileNotFoundError):
        git_hash = None

    info = {
        "run_id": run_dir.name,
        "stage": stage,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
        "study_area": study_area,
        "config": config,
        "upstream_runs": upstream_runs or {},
    }

    out_path = run_dir / "run_info.json"
    out_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return out_path


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")