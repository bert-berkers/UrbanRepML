"""
CLI entry point for the GTFS modality processor.

Usage:
    python -m stage1_modalities.gtfs --study-area netherlands --resolution 9
    python -m stage1_modalities.gtfs --study-area netherlands --resolution 9 --skip-validation
    python -m stage1_modalities.gtfs --study-area netherlands --resolution 9 --local-gtfs path/to/feed.zip
"""

import argparse
import logging
import sys

from stage1_modalities.gtfs.processor import GTFSProcessor
from utils import StudyAreaPaths

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="GTFS modality processor for UrbanRepML Stage 1",
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
        "--skip-validation", action="store_true",
        help="Skip GTFS feed validation (faster, recommended for known-good feeds)",
    )
    parser.add_argument(
        "--local-gtfs", default=None,
        help="Path to local GTFS .zip file (skips download)",
    )
    parser.add_argument(
        "--embedding-size", type=int, default=64,
        help="GTFS2Vec embedding dimensionality (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="GTFS2Vec training epochs (default: 10)",
    )
    parser.add_argument(
        "--skip-autoencoder", action="store_true",
        help="Skip autoencoder, use raw aggregated features instead",
    )
    parser.add_argument(
        "--save-intermediate", action="store_true",
        help="Save intermediate SRAI data (features, regions, joint)",
    )

    args = parser.parse_args()

    # Build config dict
    config = {
        "study_area": args.study_area,
        "skip_validation": args.skip_validation,
        "save_intermediate": args.save_intermediate,
        "gtfs2vec": {
            "embedding_size": args.embedding_size,
            "epochs": args.epochs,
            "skip_autoencoder": args.skip_autoencoder,
        },
    }

    if args.local_gtfs:
        config["data_source"] = "local"
        config["gtfs_path"] = args.local_gtfs
    else:
        config["data_source"] = "download"

    # Resolve study area boundary
    paths = StudyAreaPaths(args.study_area)
    boundary_path = paths.area_gdf_file("geojson")
    if not boundary_path.exists():
        boundary_path = paths.area_gdf_file("parquet")
    if not boundary_path.exists():
        logger.error(
            f"Study area boundary not found. Looked for "
            f"{paths.area_gdf_file('geojson')} and {paths.area_gdf_file('parquet')}"
        )
        sys.exit(1)

    logger.info(f"Starting GTFS pipeline for {args.study_area} at resolution {args.resolution}")

    processor = GTFSProcessor(config)
    output_path = processor.run_pipeline(
        study_area=str(boundary_path),
        h3_resolution=args.resolution,
        study_area_name=args.study_area,
    )

    if output_path:
        logger.info(f"Pipeline complete. Output: {output_path}")
    else:
        logger.warning("Pipeline produced no output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
