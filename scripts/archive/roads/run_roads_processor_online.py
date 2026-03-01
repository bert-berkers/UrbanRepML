#!/usr/bin/env python
"""
Run Roads processor on full Netherlands study area using OSMOnlineLoader (instead of PBF).
This avoids the memory issues with QuackOSM's PBF parsing on large files.
"""

import sys
import geopandas as gpd
from pathlib import Path
from stage1_modalities.roads.processor import RoadsProcessor

def main():
    print("=" * 80)
    print("ROADS PROCESSOR: Full Netherlands with Highway2Vec (Online Loader)")
    print("=" * 80)

    # Load area boundary
    area_path = Path("data/study_areas/netherlands/area_gdf/netherlands_boundary.parquet")
    area_gdf = gpd.read_parquet(area_path)
    print(f"\nArea GDF loaded: {area_gdf.shape}")
    print(f"CRS: EPSG:4326")

    # Configure processor to use online loader (avoids PBF memory issues)
    config = {
        'data_source': 'online',  # Use OSMOnlineLoader instead
        'study_area': 'netherlands',
        'highway2vec': {
            'embedding_size': 64,
            'epochs': 50,
            'batch_size': 512,
            'learning_rate': 0.001,
        }
    }

    print(f"\nProcessor config:")
    print(f"  Data source: {config['data_source']} (OSMOnlineLoader)")
    print(f"  H3 Resolution: 10")
    print(f"  Highway2Vec embedding: {config['highway2vec']['embedding_size']}d")
    print(f"  Note: Avoiding PBF memory issues by using online OSM API")

    # Initialize processor
    processor = RoadsProcessor(config)
    print(f"\nProcessor initialized")

    # Run pipeline
    print(f"\nRunning pipeline...")
    output_path = processor.run_pipeline(
        study_area=area_gdf,
        h3_resolution=10,
        study_area_name="netherlands"
    )

    print(f"\n" + "=" * 80)
    print(f"COMPLETE: Output saved to {output_path}")
    print("=" * 80)

    return 0

if __name__ == '__main__':
    sys.exit(main())
