# Roads Modality

**Road Network Embeddings using SRAI and NetworkX**

## ✅ Status: Complete

This modality processes road network data from OpenStreetMap into H3 hexagon-based embeddings. It calculates road lengths, densities, and connectivity metrics, and can perform graph-based centrality analysis.

## Features
- Fetches road network data from OpenStreetMap (online or from a PBF file).
- Calculates statistics per H3 hexagon:
  - **Length and count** for various road types (e.g., `motorway`, `primary`, `residential`).
  - **Total road length** and **road density**.
  - A **hierarchy score** weighted by road importance.
  - **Intersection count** and a **connectivity index**.
- Optionally computes graph-based **centrality metrics** (degree, betweenness) for the entire network and aggregates them per hexagon.

## Generated Features (Examples)
- `road_motorway_length`, `road_residential_count`, ...
- `total_road_length`
- `road_density`
- `road_hierarchy_score`
- `intersection_count`
- `avg_degree_centrality`
- `max_betweenness_centrality`

## Example Usage
```python
from modalities.roads import RoadsProcessor
import geopandas as gpd

# Define a study area (e.g., load from a file)
study_area_gdf = gpd.read_file("path/to/your/study_area.geojson")

# Configuration for the processor
config = {
    'output_dir': 'data/processed/embeddings/roads',
    'data_source': 'osm_online',  # or 'pbf' if you have a file
    'compute_network_metrics': True,
}

# Initialize and run the processor
processor = RoadsProcessor(config)
embeddings_path = processor.run_pipeline(
    study_area=study_area_gdf,
    h3_resolution=10
)

print(f"Roads embeddings saved to: {embeddings_path}")
```

## Data Sources
- OpenStreetMap (via `srai` and `osmnx` libraries)
