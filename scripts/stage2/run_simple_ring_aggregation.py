"""
Run simple ring aggregation on concatenated Stage 2 embeddings.

Applies geometric spatial smoothing via weighted k-ring neighbourhood means.
Takes the concat baseline parquet as input and produces a spatially smoothed
version as output.

Lifetime: durable
Stage: Stage 2 (fusion)

Usage:
    python scripts/stage2/run_simple_ring_aggregation.py \
        --study-area netherlands --resolution 9 --K 3 --weighting exponential
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from stage2_fusion.models.ring_aggregation import SimpleRingAggregator
from utils.paths import StudyAreaPaths


def main():
    parser = argparse.ArgumentParser(
        description="Simple ring aggregation on concat embeddings."
    )
    parser.add_argument(
        "--study-area", required=True, help="Study area name (e.g. netherlands)"
    )
    parser.add_argument(
        "--resolution", type=int, default=9, help="H3 resolution (default: 9)"
    )
    parser.add_argument(
        "--year", type=int, default=2022, help="Data year (default: 2022)"
    )
    parser.add_argument(
        "--K", type=int, default=3, help="Max ring distance (default: 3)"
    )
    parser.add_argument(
        "--weighting",
        default="exponential",
        choices=["exponential", "logarithmic", "linear", "flat"],
        help="Ring weighting scheme (default: exponential)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    paths = StudyAreaPaths(args.study_area)

    # Load concat embeddings
    concat_path = paths.fused_embedding_file("concat", args.resolution, args.year)
    logger.info("Loading concat embeddings from %s", concat_path)
    embeddings_df = pd.read_parquet(concat_path)
    logger.info("Loaded embeddings: %s", embeddings_df.shape)

    # Load neighbourhood
    nb_path = paths.neighbourhood_dir() / (
        f"{args.study_area}_res{args.resolution}_neighbourhood.pkl"
    )
    logger.info("Loading neighbourhood from %s", nb_path)
    with open(nb_path, "rb") as f:
        neighbourhood = pickle.load(f)

    # Run ring aggregation
    aggregator = SimpleRingAggregator(
        neighbourhood=neighbourhood,
        K=args.K,
        weighting=args.weighting,
    )
    result_df = aggregator.aggregate(embeddings_df)

    # Save output
    out_path = paths.fused_embedding_file("ring_agg", args.resolution, args.year)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out_path)
    logger.info("Saved ring-aggregated embeddings to %s", out_path)
    logger.info("Output shape: %s", result_df.shape)


if __name__ == "__main__":
    main()
