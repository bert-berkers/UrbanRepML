# Library Cheatsheet for UrbanRepML

## SRAI (Spatial Representations for Artificial Intelligence)

### Installation
```bash
pip install srai[all]  # Full installation with all dependencies
pip install srai       # Basic installation
```

### Core Components

#### 1. Regionalizers
Convert areas into spatial regions (H3, S2, Voronoi, Admin boundaries)

```python
from srai.regionalizers import H3Regionalizer, S2Regionalizer, VoronoiRegionalizer

# H3 Hexagons
h3_regionalizer = H3Regionalizer(resolution=8)  # Resolution 0-15
regions_gdf = h3_regionalizer.transform(area_gdf)

# S2 Squares
s2_regionalizer = S2Regionalizer(level=10)  # Level 0-30
regions_gdf = s2_regionalizer.transform(area_gdf)

# Voronoi Polygons
voronoi_regionalizer = VoronoiRegionalizer(seeds=points_gdf)
regions_gdf = voronoi_regionalizer.transform(area_gdf)
```

#### 2. Loaders
Load geospatial data from various sources

```python
from srai.loaders import OSMPbfLoader, OSMOnlineLoader, OSMWayLoader

# Load from PBF file
pbf_loader = OSMPbfLoader()
gdf = pbf_loader.load(pbf_file_path, tags={'building': True})

# Load from OSM API
online_loader = OSMOnlineLoader()
gdf = online_loader.load(area_gdf, tags={'amenity': ['restaurant', 'cafe']})

# Load OSM ways
way_loader = OSMWayLoader(way_type='highway')
roads_gdf = way_loader.load(area_gdf)
```

#### 3. Embedders
Create embeddings from geospatial features

```python
from srai.embedders import CountEmbedder, GTFS2VecEmbedder, Hex2VecEmbedder

# Count-based embeddings
count_embedder = CountEmbedder()
embeddings = count_embedder.transform(regions_gdf, features_gdf)

# GTFS transit embeddings
gtfs_embedder = GTFS2VecEmbedder()
embeddings = gtfs_embedder.transform(regions_gdf, gtfs_data)

# Hex2Vec embeddings (contextual)
hex2vec_embedder = Hex2VecEmbedder()
embeddings = hex2vec_embedder.fit_transform(regions_gdf, features_gdf)
```

#### 4. Joiners
Spatial joins between regions and features

```python
from srai.joiners import IntersectionJoiner

joiner = IntersectionJoiner()
joint_gdf = joiner.transform(regions_gdf, features_gdf)
```

#### 5. Plotting
Visualization utilities

```python
from srai.plotting import plot_regions, plot_numeric_data, plot_categorical_data

# Basic region plot
import folium
m = folium.Map()
plot_regions(regions_gdf, map=m)

# Numeric data visualization
plot_numeric_data(
    regions_gdf, 
    column='population',
    map=m,
    colormap='viridis'
)

# Categorical data
plot_categorical_data(
    regions_gdf,
    column='land_use',
    map=m
)
```

#### 6. Neighborhoods
Get neighboring regions

```python
from srai.neighbourhoods import H3Neighbourhood, AdjacencyNeighbourhood

# H3 neighbors
h3_neighbourhood = H3Neighbourhood()
neighbors_df = h3_neighbourhood.get_neighbours(regions_gdf)

# Adjacency-based neighbors
adj_neighbourhood = AdjacencyNeighbourhood()
neighbors_df = adj_neighbourhood.get_neighbours(regions_gdf)
```

### Common Workflows

#### Create H3 hexagons for a city
```python
import geopandas as gpd
from srai.regionalizers import H3Regionalizer
from srai.plotting import plot_regions
import folium

# Load city boundary
city = gpd.read_file('city_boundary.geojson')

# Create hexagons
regionalizer = H3Regionalizer(resolution=9)
hexagons = regionalizer.transform(city)

# Visualize
m = folium.Map(location=[lat, lon], zoom_start=11)
plot_regions(hexagons, map=m)
m.save('hexagons.html')
```

#### Generate embeddings from OSM data
```python
from srai.loaders import OSMOnlineLoader
from srai.embedders import CountEmbedder
from srai.joiners import IntersectionJoiner

# Load POIs
loader = OSMOnlineLoader()
pois = loader.load(
    area_gdf,
    tags={'amenity': True, 'shop': True}
)

# Join with regions
joiner = IntersectionJoiner()
joint_gdf = joiner.transform(hexagons, pois)

# Create embeddings
embedder = CountEmbedder()
embeddings = embedder.transform(hexagons, pois, joint_gdf)
```

#### Contextual embeddings with Hex2Vec
```python
from srai.embedders import Hex2VecEmbedder
from srai.neighbourhoods import H3Neighbourhood

# Get neighbors
neighbourhood = H3Neighbourhood()
neighbors = neighbourhood.get_neighbours(hexagons)

# Train Hex2Vec
embedder = Hex2VecEmbedder(
    dimensions=64,
    walk_length=30,
    num_walks=200
)
contextual_embeddings = embedder.fit_transform(
    hexagons, 
    features, 
    neighbors
)
```

### Performance Tips

1. **Use appropriate resolution**: Higher H3 resolutions = more hexagons = slower processing
2. **Filter data early**: Load only necessary tags from OSM
3. **Batch processing**: Process large areas in chunks
4. **Use Parquet**: Save GeoDataFrames as Parquet for faster I/O
5. **Parallel processing**: Many SRAI functions support n_jobs parameter

### Useful H3 Resolution Reference

| Resolution | Avg Hexagon Area | Avg Edge Length | Use Case |
|------------|------------------|-----------------|----------|
| 5 | 252.9 km² | 9.2 km | Country/State level |
| 6 | 36.1 km² | 3.5 km | Large city overview |
| 7 | 5.2 km² | 1.3 km | City districts |
| 8 | 0.74 km² | 461 m | Neighborhoods |
| 9 | 0.11 km² | 174 m | City blocks |
| 10 | 0.015 km² | 66 m | Individual buildings |

### Common Issues & Solutions

**Issue**: Memory error with large regions
```python
# Solution: Process in chunks
from srai.regionalizers import H3Regionalizer

regionalizer = H3Regionalizer(resolution=10)
chunks = []
for chunk in area_gdf.iterrows():
    chunk_regions = regionalizer.transform(chunk)
    chunks.append(chunk_regions)
result = pd.concat(chunks)
```

**Issue**: Slow OSM loading
```python
# Solution: Use PBF files instead of API
from srai.loaders import OSMPbfLoader

loader = OSMPbfLoader()
# Download PBF from Geofabrik first
gdf = loader.load('netherlands.osm.pbf', tags={'building': True})
```

**Issue**: Large visualization files
```python
# Solution: Simplify geometry or sample data
hexagons_simplified = hexagons.copy()
hexagons_simplified['geometry'] = hexagons_simplified.geometry.simplify(0.001)

# Or sample for preview
sample = hexagons.sample(n=1000)
plot_regions(sample, map=m)
```

---

## H3 Library

### Basic H3 Operations
```python
import h3

# Convert lat/lng to H3
h3_index = h3.latlng_to_cell(lat, lng, resolution)

# Get H3 center
lat, lng = h3.cell_to_latlng(h3_index)

# Get H3 boundary
boundary = h3.cell_to_boundary(h3_index)

# Get neighbors
neighbors = h3.grid_disk(h3_index, k=1)  # k=ring size

# Get parent/children
parent = h3.cell_to_parent(h3_index, parent_resolution)
children = h3.cell_to_children(h3_index, child_resolution)

# Calculate distance
distance = h3.grid_distance(h3_index1, h3_index2)

# Check validity
is_valid = h3.is_valid_cell(h3_index)

# Get resolution
resolution = h3.get_resolution(h3_index)

# Area and edge length
area = h3.cell_area(h3_index, unit='km^2')
edge_length = h3.edge_length(resolution, unit='m')
```

---

## GeoPandas Essentials

### Quick Operations
```python
import geopandas as gpd

# Read/Write
gdf = gpd.read_file('file.geojson')
gdf.to_file('output.geojson', driver='GeoJSON')
gdf.to_parquet('output.parquet')  # Faster I/O

# CRS transformations
gdf = gdf.to_crs('EPSG:4326')  # WGS84
gdf = gdf.to_crs('EPSG:28992')  # Dutch RD

# Spatial operations
buffer = gdf.buffer(100)  # Buffer in CRS units
centroids = gdf.centroid
envelope = gdf.envelope  # Bounding box
simplified = gdf.simplify(tolerance=0.001)

# Spatial joins
joined = gpd.sjoin(gdf1, gdf2, how='inner', predicate='intersects')

# Dissolve/Aggregate
dissolved = gdf.dissolve(by='category', aggfunc='sum')

# Quick plot
gdf.plot(column='value', cmap='viridis', legend=True)
```

---

## Roads Processing with SRAI

### Basic Road Network Loading
```python
from srai.loaders import OSMWayLoader, OSMPbfLoader
from modalities.roads import RoadsProcessor

# Load from OSM online
way_loader = OSMWayLoader(
    way_type='highway',
    osm_way_filter={'highway': ['primary', 'secondary', 'tertiary']}
)
roads_gdf = way_loader.load(area_gdf)

# Load from PBF file
pbf_loader = OSMPbfLoader()
roads_gdf = pbf_loader.load(
    'netherlands.osm.pbf',
    tags={'highway': True},
    area=area_gdf
)

# Use UrbanRepML Roads Processor
processor = RoadsProcessor({
    'data_source': 'osm_online',
    'output_dir': 'data/processed/roads',
    'compute_network_metrics': True
})
embeddings_path = processor.run_pipeline(area_gdf, h3_resolution=9)
```

### Road Network Analysis
```python
import networkx as nx
import osmnx as ox
from modalities.roads.processor import RoadsProcessor

# Build graph from road data
G = nx.Graph()

# Add road segments as edges
for idx, road in roads_gdf.iterrows():
    if hasattr(road.geometry, 'coords'):
        coords = list(road.geometry.coords)
        start, end = coords[0], coords[-1]
        G.add_edge(start, end, weight=road.geometry.length)

# Calculate centrality metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, k=100)  # Sample for speed

# Road hierarchy scoring
road_hierarchy = {
    'motorway': 1.0, 'trunk': 0.85, 'primary': 0.75,
    'secondary': 0.65, 'tertiary': 0.55, 'residential': 0.35
}
```

### Road Embeddings Features
```python
# Generated features per H3 hexagon:
# - road_{type}_length: Total length by road type
# - road_{type}_count: Number of segments by road type  
# - total_road_length: Total length of all roads
# - road_density: Roads per km²
# - road_hierarchy_score: Weighted importance score
# - intersection_count: Number of road intersections
# - connectivity_index: Connectivity ratio
# - avg_degree_centrality: Average node centrality
# - avg_betweenness_centrality: Average betweenness
```

---

## POI Processing with SRAI

### Basic POI Loading
```python
from srai.loaders import OSMOnlineLoader, OSMPbfLoader
from srai.embedders import CountEmbedder, Hex2VecEmbedder
from modalities.poi import POIProcessor

# Load POIs by category
online_loader = OSMOnlineLoader()
pois_gdf = online_loader.load(
    area_gdf,
    tags={'amenity': ['restaurant', 'cafe', 'hospital', 'school']}
)

# Load from PBF
pbf_loader = OSMPbfLoader()
pois_gdf = pbf_loader.load(
    'netherlands.osm.pbf',
    tags={'amenity': True, 'shop': True, 'leisure': True}
)

# Use UrbanRepML POI Processor
processor = POIProcessor({
    'data_source': 'osm_online',
    'output_dir': 'data/processed/poi',
    'use_hex2vec': True,
    'compute_diversity_metrics': True
})
embeddings_path = processor.run_pipeline(area_gdf, h3_resolution=9)
```

### Count-Based POI Embeddings
```python
from srai.embedders import CountEmbedder
from srai.joiners import IntersectionJoiner

# Spatial join POIs with H3 hexagons
joiner = IntersectionJoiner()
joint_gdf = joiner.transform(hexagons_gdf, pois_gdf)

# Create count embeddings
count_embedder = CountEmbedder()
embeddings = count_embedder.transform(
    regions_gdf=hexagons_gdf,
    features_gdf=pois_gdf,
    joint_gdf=joint_gdf,
    aggregation_column='amenity'  # Group by POI type
)

# Result: Matrix with hexagons × POI categories
```

### Contextual POI Embeddings with Hex2Vec
```python
from srai.embedders import Hex2VecEmbedder
from srai.neighbourhoods import H3Neighbourhood

# Get hexagon neighborhoods
neighbourhood = H3Neighbourhood()
neighbors_df = neighbourhood.get_neighbours(hexagons_gdf)

# Train Hex2Vec with POI context
hex2vec_embedder = Hex2VecEmbedder(
    dimensions=64,           # Embedding dimensions
    walk_length=30,         # Random walk length
    num_walks=100,          # Number of walks per node
    window_size=5,          # Skip-gram window
    p=1,                    # Return parameter
    q=1                     # In-out parameter
)

# Fit and transform
contextual_embeddings = hex2vec_embedder.fit_transform(
    regions_gdf=hexagons_gdf,
    features_gdf=pois_gdf,
    neighbourhood=neighbors_df,
    base_embeddings=count_embeddings  # Use count features
)

# Result: 64-dimensional contextual embeddings per hexagon
```

### POI Diversity Metrics
```python
import numpy as np
import pandas as pd

def calculate_poi_diversity(poi_categories):
    """Calculate diversity indices for POI categories."""
    category_counts = poi_categories.value_counts()
    total = category_counts.sum()
    proportions = category_counts / total
    
    # Shannon entropy (diversity)
    shannon = -sum(p * np.log(p) if p > 0 else 0 for p in proportions)
    
    # Simpson diversity
    simpson = 1 - sum(p**2 for p in proportions)
    
    # Richness (unique categories)
    richness = len(category_counts)
    
    # Evenness (normalized Shannon)
    max_entropy = np.log(richness) if richness > 1 else 1
    evenness = shannon / max_entropy if max_entropy > 0 else 0
    
    return {
        'shannon_entropy': shannon,
        'simpson_diversity': simpson,
        'richness': richness,
        'evenness': evenness
    }

# Apply to hexagon POIs
diversity_metrics = poi_by_hex.apply(calculate_poi_diversity)
```

### POI Categories and Filtering
```python
# Common POI category hierarchies
poi_categories = {
    'food_drink': ['restaurant', 'cafe', 'bar', 'pub', 'fast_food'],
    'healthcare': ['hospital', 'clinic', 'pharmacy', 'doctors'],
    'education': ['school', 'university', 'library', 'kindergarten'],
    'retail': ['supermarket', 'shop', 'convenience', 'department_store'],
    'recreation': ['park', 'playground', 'sports_centre', 'cinema'],
    'transport': ['parking', 'fuel', 'station', 'stop_position']
}

# Load specific categories
for category, values in poi_categories.items():
    category_pois = online_loader.load(
        area_gdf,
        tags={'amenity': values}
    )
    category_pois['category_group'] = category
```

### POI Embeddings Features
```python
# Generated features per H3 hexagon:
# Count features:
# - {category}_count: Count per POI category
# - total_poi_count: Total POIs in hexagon
# - poi_density: POIs per km²

# Diversity features:
# - poi_shannon_entropy: Category diversity
# - poi_simpson_diversity: Simpson index
# - poi_richness: Number of unique categories
# - poi_evenness: Distribution evenness

# Contextual features (if Hex2Vec enabled):
# - hex2vec_0 to hex2vec_63: 64D contextual embeddings
```

### Complete Roads + POI Workflow
```python
from modalities.roads import RoadsProcessor
from modalities.poi import POIProcessor
import pandas as pd

# Process both modalities
roads_processor = RoadsProcessor({'data_source': 'osm_online'})
poi_processor = POIProcessor({'data_source': 'osm_online', 'use_hex2vec': True})

# Generate embeddings
roads_path = roads_processor.run_pipeline(area_gdf, h3_resolution=9)
poi_path = poi_processor.run_pipeline(area_gdf, h3_resolution=9)

# Load and combine
roads_df = pd.read_parquet(roads_path)
poi_df = pd.read_parquet(poi_path)

# Merge on H3 index
combined_df = roads_df.merge(poi_df, on='h3_index', how='outer')

# Fill missing values
combined_df = combined_df.fillna(0)

print(f"Combined embeddings: {combined_df.shape}")
print(f"Features: {list(combined_df.columns)}")
```

### Performance Tips for Roads & POI

1. **Large Areas**: Use PBF files instead of OSM API
```python
# Download PBF first
import requests
pbf_url = "https://download.geofabrik.de/europe/netherlands-latest.osm.pbf"
response = requests.get(pbf_url)
with open("netherlands.osm.pbf", "wb") as f:
    f.write(response.content)
```

2. **Memory Management**: Process in chunks
```python
# Chunk large study areas
import geopandas as gpd
from shapely.geometry import box

def chunk_area(bounds, chunk_size=0.1):
    """Split large area into smaller chunks."""
    minx, miny, maxx, maxy = bounds
    chunks = []
    
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            chunk = box(x, y, min(x + chunk_size, maxx), min(y + chunk_size, maxy))
            chunks.append(chunk)
            y += chunk_size
        x += chunk_size
    
    return gpd.GeoDataFrame(geometry=chunks, crs='EPSG:4326')
```

3. **Filter Early**: Reduce data before processing
```python
# Filter POIs by importance
important_amenities = ['hospital', 'school', 'restaurant', 'supermarket']
pois_filtered = pois_gdf[pois_gdf['amenity'].isin(important_amenities)]
```

---

*Last updated: January 2025*