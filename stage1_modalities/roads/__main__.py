"""
CLI entry point for the Roads modality processor.

Usage:
    # Full pipeline (load OSM data, regionalize, train Highway2Vec, embed)
    python -m stage1_modalities.roads --study-area netherlands --resolution 9

    # With specific OSM date (auto-resolves PBF from osm/ directory)
    python -m stage1_modalities.roads --study-area netherlands --resolution 9 \
        --data-source pbf --osm-date 2022-01-01 --year 2022

    # With explicit PBF file
    python -m stage1_modalities.roads --study-area netherlands --resolution 9 \
        --pbf-path data/study_areas/netherlands/osm/netherlands-2022-01-01.osm.pbf
"""

import argparse
import logging
import sys

print("Loading Roads processor (importing dependencies)...", flush=True)

import geopandas as gpd

from stage1_modalities.roads.processor import RoadsProcessor
from utils import StudyAreaPaths

logger = logging.getLogger(__name__)


def run_full_pipeline(processor: RoadsProcessor, study_area_name: str,
                      resolution: int, year: int):
    """Run the full roads pipeline (load data, regionalize, embed, save)."""
    paths = StudyAreaPaths(study_area_name)

    # Load study area boundary
    boundary_path = paths.area_gdf_file("geojson")
    if not boundary_path.exists():
        boundary_path = paths.area_gdf_file("parquet")
    if not boundary_path.exists():
        logger.error(
            f"Study area boundary not found. Looked for "
            f"{paths.area_gdf_file('geojson')} and {paths.area_gdf_file('parquet')}"
        )
        sys.exit(1)

    output_path = processor.run_pipeline(
        study_area=str(boundary_path),
        h3_resolution=resolution,
        study_area_name=study_area_name,
        year=year,
    )

    if output_path:
        logger.info(f"Full pipeline complete: {output_path}")
    else:
        logger.warning("Full pipeline produced no output (empty road data?)")

    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Roads modality processor for UrbanRepML Stage 1",
    )
    parser.add_argument(
        "--study-area", required=True,
        help="Study area name (e.g. 'netherlands')",
    )
    parser.add_argument(
        "--resolution", type=int, default=9,
        help="H3 resolution (default: 9)",
    )
    parser.add_argument(
        "--year", type=int, default=2022,
        help="Data year for output filename (default: 2022)",
    )
    parser.add_argument(
        "--save-intermediate", action="store_true",
        help="Save intermediate SRAI data (regions, features, joint) during pipeline",
    )
    parser.add_argument(
        "--pbf-path", default=None,
        help="Path to OSM PBF file. If omitted with --data-source pbf, "
             "auto-resolves from data/study_areas/{area}/osm/",
    )
    parser.add_argument(
        "--data-source", choices=("osm_online", "pbf"), default=None,
        help="Data source: 'osm_online' (Overpass API) or 'pbf' (local PBF file). "
             "Default: 'osm_online', or 'pbf' if --pbf-path is provided.",
    )
    parser.add_argument(
        "--osm-date", default="latest",
        help="OSM snapshot date (e.g. '2022-01-01') when auto-resolving PBF path. "
             "Default: 'latest'. Only used when --data-source pbf without --pbf-path.",
    )
    parser.add_argument(
        "--embedding-size", type=int, default=None,
        help="Highway2Vec embedding dimensions (overrides config.yaml default)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Highway2Vec training epochs (overrides config.yaml default)",
    )

    args = parser.parse_args()

    # Build config for RoadsProcessor
    config = {
        "study_area": args.study_area,
        "year": args.year,
        "save_intermediate": args.save_intermediate,
    }

    # Determine data source: explicit flag > inferred from --pbf-path > default
    if args.data_source:
        config["data_source"] = args.data_source
    elif args.pbf_path:
        config["data_source"] = "pbf"

    if args.pbf_path:
        config["pbf_path"] = args.pbf_path

    # Pass osm_date so auto-resolve picks the right snapshot
    config["osm_date"] = args.osm_date

    # Optional Highway2Vec overrides
    highway2vec_overrides = {}
    if args.embedding_size is not None:
        highway2vec_overrides["embedding_size"] = args.embedding_size
    if args.epochs is not None:
        highway2vec_overrides["epochs"] = args.epochs
    if highway2vec_overrides:
        config["highway2vec"] = highway2vec_overrides

    processor = RoadsProcessor(config)

    # Always save intermediates in full-pipeline mode
    processor.save_intermediate = True

    run_full_pipeline(processor, args.study_area, args.resolution, args.year)


if __name__ == "__main__":
    main()
