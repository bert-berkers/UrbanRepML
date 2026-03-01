"""
CLI entry point for the POI modality processor.

Usage:
    # Full pipeline (load OSM data, regionalize, embed)
    python -m stage1_modalities.poi --study-area netherlands --resolution 10

    # Single embedder from pre-saved intermediates
    python -m stage1_modalities.poi --study-area netherlands --resolution 10 --embedder hex2vec
    python -m stage1_modalities.poi --study-area netherlands --resolution 10 --embedder geovex
    python -m stage1_modalities.poi --study-area netherlands --resolution 10 --embedder count
"""

import argparse
import logging
import sys

print("Loading POI processor (importing dependencies)...", flush=True)

import geopandas as gpd

from stage1_modalities.poi.processor import POIProcessor
from utils import StudyAreaPaths

logger = logging.getLogger(__name__)

VALID_EMBEDDERS = ("count", "hex2vec", "geovex")


def _build_output_path(paths: StudyAreaPaths, embedder_name: str,
                       resolution: int, year: int):
    """Build the output parquet path for a single-embedder run.

    Uses ``StudyAreaPaths.embedding_file`` with the ``sub_embedder``
    parameter if available, otherwise falls back to manual path
    construction.
    """
    import inspect
    sig = inspect.signature(paths.embedding_file)
    if "sub_embedder" in sig.parameters:
        return paths.embedding_file("poi", resolution, year, sub_embedder=embedder_name)
    else:
        # Fallback: another agent has not yet added sub_embedder param
        base = paths.stage1("poi") / embedder_name
        return base / f"{paths.study_area}_res{resolution}_{year}.parquet"


def run_single_embedder(processor: POIProcessor, embedder_name: str,
                        resolution: int, year: int):
    """Load intermediates and run a single embedder, saving the result."""
    # Always load with neighbourhood -- it's None if no cache exists
    regions_gdf, features_gdf, joint_gdf, neighbourhood = processor.load_intermediates(
        resolution, include_neighbourhood=True,
    )

    if embedder_name == "count":
        embeddings_df = processor.run_count_embeddings(regions_gdf, features_gdf, joint_gdf)
    elif embedder_name == "hex2vec":
        embeddings_df = processor.run_hex2vec(
            regions_gdf, features_gdf, joint_gdf, neighbourhood=neighbourhood,
        )
    elif embedder_name == "geovex":
        embeddings_df = processor.run_geovex(
            regions_gdf, features_gdf, joint_gdf, neighbourhood=neighbourhood,
        )
    else:
        raise ValueError(f"Unknown embedder: {embedder_name}. Valid: {VALID_EMBEDDERS}")

    output_path = _build_output_path(processor._paths, embedder_name, resolution, year)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_parquet(output_path)

    logger.info(f"Saved {embedder_name} embeddings to {output_path}")
    logger.info(f"Shape: {embeddings_df.shape[0]} hexagons x {embeddings_df.shape[1]} features")
    return str(output_path)


def run_full_pipeline(processor: POIProcessor, study_area_name: str,
                      resolution: int):
    """Run the full POI pipeline (load data, regionalize, embed, save)."""
    paths = StudyAreaPaths(study_area_name)

    # Load study area boundary
    boundary_path = paths.area_gdf_file("geojson")
    if not boundary_path.exists():
        # Try parquet fallback
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
    )

    if output_path:
        logger.info(f"Full pipeline complete: {output_path}")
    else:
        logger.warning("Full pipeline produced no output (empty POI data?)")

    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="POI modality processor for UrbanRepML Stage 1",
    )
    parser.add_argument(
        "--study-area", required=True,
        help="Study area name (e.g. 'netherlands')",
    )
    parser.add_argument(
        "--resolution", type=int, default=10,
        help="H3 resolution (default: 10)",
    )
    parser.add_argument(
        "--embedder", choices=VALID_EMBEDDERS, default=None,
        help=(
            "Run a single embedder from pre-saved intermediates. "
            "If omitted, runs the full pipeline."
        ),
    )
    parser.add_argument(
        "--year", type=int, default=2022,
        help="Data year for output filename (default: 2022)",
    )
    parser.add_argument(
        "--save-intermediate", action="store_true",
        help="Save intermediate SRAI data (regions, features, joint) during full pipeline",
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
        "--device", choices=("auto", "cpu", "gpu"), default="auto",
        help="Accelerator for hex2vec/geovex training: auto (default), cpu, or gpu",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="Target batch size for hex2vec/geovex training (default: 4096). "
             "Training starts at --initial-batch-size and ramps up to this value.",
    )
    parser.add_argument(
        "--initial-batch-size", type=int, default=512,
        help="Initial batch size at epoch 0; ramps up to --batch-size (default: 512)",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=3,
        help="EarlyStopping patience in epochs (0 to disable; default: 3)",
    )

    args = parser.parse_args()

    # Build config for POIProcessor
    #
    # When running in single-embedder mode, hex2vec/geovex flags on the
    # processor are irrelevant -- we call run_hex2vec/run_geovex directly.
    # When running the full pipeline, defaults apply (both disabled).
    # save_intermediate is forced on in full-pipeline mode (see below)
    # so that single-embedder runs can find intermediates afterward.
    config = {
        "study_area": args.study_area,
        "year": args.year,
        "save_intermediate": args.save_intermediate,
        "batch_size": args.batch_size,
        "initial_batch_size": args.initial_batch_size,
        "early_stopping_patience": args.early_stopping_patience,
        "accelerator": args.device,
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

    processor = POIProcessor(config)

    if args.embedder:
        # Single-embedder mode: load intermediates, run one embedder, save
        logger.info(f"Single-embedder mode: {args.embedder}")
        run_single_embedder(processor, args.embedder, args.resolution, args.year)
    else:
        # Full pipeline mode
        # Ensure intermediates are saved so single-embedder runs work later
        processor.save_intermediate = True
        run_full_pipeline(processor, args.study_area, args.resolution)


if __name__ == "__main__":
    main()
