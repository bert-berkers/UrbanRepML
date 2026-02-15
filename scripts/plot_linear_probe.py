#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate all linear probe visualizations for a completed run.

Loads results (metrics, predictions, coefficients) from a linear probe
output directory and generates all plots via LinearProbeVisualizer.

Usage:
    python scripts/plot_linear_probe.py
    python scripts/plot_linear_probe.py --run-dir path/to/run
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stage3_analysis.dnn_probe import DNNProbeRegressor
from stage3_analysis.linear_probe_viz import LinearProbeVisualizer
from utils import StudyAreaPaths

logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate all linear probe visualizations"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to linear probe run directory "
             "(default: latest run in stage3/linear_probe)",
    )
    parser.add_argument(
        "--study-area",
        type=str,
        default="netherlands",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    paths = StudyAreaPaths(args.study_area)

    # Resolve run directory
    if args.run_dir:
        linear_dir = Path(args.run_dir)
    else:
        linear_dir = paths.latest_run(paths.stage3("linear_probe"))
        if linear_dir is None:
            logger.error("No linear probe runs found")
            sys.exit(1)

    logger.info(f"Loading linear probe results from {linear_dir}")

    plot_dir = linear_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load results with full coefficients and fold metrics
    results = DNNProbeRegressor.load_linear_results(linear_dir)
    logger.info(f"Loaded {len(results)} target results")

    # Get embeddings path for RGB top-3 maps
    embeddings_path = paths.embedding_file("alphaearth", 10, 2022)
    if not embeddings_path.exists():
        logger.warning(f"Embeddings not found at {embeddings_path}, "
                       f"RGB top-3 maps will be skipped")
        embeddings_path = None

    # Get boundary GDF for map backgrounds
    boundary_gdf = None
    boundary_path = paths.area_gdf_file()
    if boundary_path.exists():
        import geopandas as gpd
        boundary_gdf = gpd.read_file(boundary_path)
        logger.info(f"Loaded boundary from {boundary_path}")
    else:
        logger.warning(f"Boundary file not found at {boundary_path}")

    # Create visualizer and generate all plots
    viz = LinearProbeVisualizer(
        results=results, output_dir=plot_dir, study_area=args.study_area,
    )
    generated = viz.plot_all(
        embeddings_path=embeddings_path,
        boundary_gdf=boundary_gdf,
    )

    print(f"\nGenerated {len(generated)} plots to {plot_dir}")
    for p in generated:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
