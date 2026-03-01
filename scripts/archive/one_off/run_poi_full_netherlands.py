#!/usr/bin/env python3
"""
Run POI processor on full Netherlands with Hex2Vec and GeoVex embeddings.
Resolution 10 (approx 1.9km hexagons).
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys
import logging
import multiprocessing

# Setup multiprocessing for Windows
if sys.platform == 'win32':
    multiprocessing.set_start_method('spawn', force=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)

from stage1_modalities.poi.processor import POIProcessor

def run_poi_processor():
    log.info("=== POI Processor: Full Netherlands H3 Res 10 ===")

    # Load study area
    area_path = Path("data/study_areas/netherlands/area_gdf/netherlands_boundary.parquet")
    area_gdf = gpd.read_parquet(area_path)
    log.info(f"Study area loaded: {area_gdf.shape[0]} regions")

    # Configure processor
    config = {
        'data_source': 'pbf',
        'pbf_path': 'data/raw/osm/netherlands-latest.osm.pbf',
        'study_area': 'netherlands',
        'use_hex2vec': True,
        'use_geovex': True,
        'hex2vec_epochs': 10,
        'geovex_epochs': 10,
        'batch_size': 512,
        'device': 'cuda',  # Use GPU
    }

    log.info(f"Config: {config}")

    processor = POIProcessor(config)
    log.info("POIProcessor initialized")

    output = processor.run_pipeline(
        study_area=area_gdf,
        h3_resolution=10,
        study_area_name="netherlands"
    )
    log.info(f"Pipeline completed: {output}")

    # Verify output
    from utils import StudyAreaPaths
    paths = StudyAreaPaths("netherlands")
    poi_file = paths.embedding_file("poi", 10, 2022)

    if Path(poi_file).exists():
        poi = pd.read_parquet(poi_file)
        log.info(f"Output verified: shape={poi.shape}, index={poi.index.name}")
        log.info(f"Sample columns: {list(poi.columns[:5])}")
        assert poi.index.name == "region_id", f"Bad index name: {poi.index.name}"
        assert all(c.startswith("P") for c in poi.columns), "Not all columns start with 'P'"
        log.info("SUCCESS: POI embeddings generated and verified")
        return 0
    else:
        log.error(f"Output file not found: {poi_file}")
        return 1


def main():
    try:
        return run_poi_processor()
    except Exception as e:
        log.error(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
