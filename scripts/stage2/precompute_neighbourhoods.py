"""Precompute H3Neighbourhood pickles for the shared neighbourhood cache.

For each specified resolution, loads the regions_gdf and constructs an
H3Neighbourhood with _available_indices populated (region-filtered).  Saves
to StudyAreaPaths.neighbourhood_dir() so MultiResolutionLoader and other
consumers can load without recomputing on every training run.

Lifetime: durable
Stage: 2 (fusion, preprocessing)

Usage::

    python scripts/stage2/precompute_neighbourhoods.py --study-area netherlands
    python scripts/stage2/precompute_neighbourhoods.py --study-area netherlands --resolutions 7 8 9 10
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import geopandas as gpd
from srai.neighbourhoods import H3Neighbourhood

# Ensure project root is on path when run directly
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils.paths import StudyAreaPaths  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def precompute_neighbourhood(
    study_area: str,
    resolution: int,
    force: bool = False,
) -> Path:
    """Build and persist a region-filtered H3Neighbourhood for one resolution.

    Parameters
    ----------
    study_area:
        Study area name, e.g. "netherlands".
    resolution:
        H3 resolution (e.g. 7, 8, 9).
    force:
        Overwrite existing pickle if True.

    Returns
    -------
    Path to the saved pickle file.
    """
    paths = StudyAreaPaths(study_area)
    out_dir = paths.neighbourhood_dir()
    out_path = out_dir / f"{study_area}_res{resolution}_neighbourhood.pkl"

    if out_path.exists() and not force:
        logger.info(f"Pickle already exists (use --force to overwrite): {out_path}")
        return out_path

    # Load regions_gdf
    region_file = paths.region_file(resolution)
    if not region_file.exists():
        raise FileNotFoundError(
            f"regions_gdf not found for res{resolution}: {region_file}\n"
            f"Run H3Regionalizer for this resolution first."
        )

    logger.info(f"Loading regions_gdf from {region_file}")
    regions_gdf = gpd.read_parquet(region_file)
    n_regions = len(regions_gdf)
    logger.info(f"Loaded {n_regions:,} regions at res{resolution}")

    # Build neighbourhood with region filter populated
    # H3Neighbourhood(regions_gdf) sets _available_indices = set(regions_gdf.index)
    # This makes get_neighbours() return only in-study-area neighbours, matching
    # the same format used by the POI processor's neighbourhood cache.
    logger.info(f"Building H3Neighbourhood for res{resolution} ({n_regions:,} regions)...")
    neighbourhood = H3Neighbourhood(regions_gdf=regions_gdf)

    n_available = len(neighbourhood._available_indices) if neighbourhood._available_indices else 0
    logger.info(f"  _available_indices: {n_available:,} regions")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(neighbourhood, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved neighbourhood pickle ({size_mb:.1f} MB) -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute H3Neighbourhood pickles.")
    parser.add_argument(
        "--study-area",
        default="netherlands",
        help="Study area name (default: netherlands)",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[7, 8, 9],
        help="H3 resolutions to precompute (default: 7 8 9)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing pickles",
    )
    args = parser.parse_args()

    for res in args.resolutions:
        try:
            out_path = precompute_neighbourhood(args.study_area, res, force=args.force)
            logger.info(f"res{res}: OK -> {out_path}")
        except FileNotFoundError as exc:
            logger.error(f"res{res}: {exc}")
            sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
