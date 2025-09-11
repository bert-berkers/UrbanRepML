# Library Cheatsheet for UrbanRepML

## SRAI (Spatial Representations for Artificial Intelligence)

### Installation
```bash
pip install srai[all]  # Full installation with all dependencies
pip install srai       # Basic installation

# For advanced embedders (required for UrbanRepML processors)
pip install gensim     # Required for Hex2Vec
pip install torch pytorch-lightning  # Required for GeoVex and Highway2Vec
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
from srai.embedders import CountEmbedder, GTFS2VecEmbedder
from srai.embedders.hex2vec.embedder import Hex2VecEmbedder
from srai.embedders.geovex.embedder import GeoVexEmbedder
from srai.embedders.highway2vec.embedder import Highway2VecEmbedder

# Count-based embeddings
count_embedder = CountEmbedder()
embeddings = count_embedder.transform(regions_gdf, features_gdf, joint_gdf)

# GTFS transit embeddings
gtfs_embedder = GTFS2VecEmbedder()
embeddings = gtfs_embedder.transform(regions_gdf, gtfs_data)

# Hex2Vec embeddings (contextual, uses Gensim Word2Vec)
hex2vec_embedder = Hex2VecEmbedder(
    dimensions=32,    # Embedding size
    walk_length=30,   # Random walk length
    num_walks=15,     # Walks per node
    window_size=5,    # Skip-gram window
    workers=4         # Gensim workers
)
embeddings = hex2vec_embedder.fit_transform(
    regions_gdf, features_gdf, joint_gdf, 
    neighbourhood, base_embeddings
)

# GeoVex embeddings (Graph Neural Network-based)
geovex_embedder = GeoVexEmbedder(
    embedding_size=32,  # Output embedding dimensions
    hidden_size=64,     # Hidden layer size
    num_layers=2        # Number of GNN layers
)
embeddings = geovex_embedder.fit_transform(
    regions_gdf, features_gdf, joint_gdf,
    neighbourhood,
    trainer_kwargs={'max_epochs': 15, 'accelerator': 'auto'}
)

# Highway2Vec embeddings (for road networks)
highway2vec_embedder = Highway2VecEmbedder(
    hidden_size=64,     # Hidden layer size
    embedding_size=30   # Output embedding dimensions
)
embeddings = highway2vec_embedder.fit_transform(
    regions_gdf, roads_gdf, joint_gdf
)
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

#### Advanced Embeddings

##### Hex2Vec (Skip-gram embeddings)
```python
from srai.embedders import Hex2VecEmbedder
from srai.neighbourhoods import H3Neighbourhood

# Prepare neighbourhood graph
neighbourhood = H3Neighbourhood(regions_gdf)

# Initialize Hex2Vec
embedder = Hex2VecEmbedder(
    encoder_sizes=[32],  # Hidden layer sizes
    expected_output_features=['amenity', 'shop'],  # Feature columns to use
    count_subcategories=True  # Count subcategories
)

# Generate embeddings
contextual_embeddings = embedder.fit_transform(
    regions_gdf=hexagons,
    features_gdf=features,
    joint_gdf=joint_gdf,
    neighbourhood=neighbourhood,
    negative_sample_k_distance=2,  # Distance for negative sampling
    batch_size=32,
    learning_rate=0.001
)
```

##### GeoVex (Convolutional embeddings)
```python
from srai.embedders import GeoVexEmbedder

# Initialize convolutional embedder
geovex_embedder = GeoVexEmbedder(
    target_features=['amenity', 'shop', 'leisure'],  # Features to embed
    count_subcategories=True,
    batch_size=32,
    neighbourhood_radius=4,         # Neighborhood hops
    convolutional_layers=2,         # Number of conv layers
    embedding_size=32,              # Output dimensions
    convolutional_layer_size=256    # Conv layer hidden size
)

# Generate embeddings (no neighbourhood needed)
geovex_embeddings = geovex_embedder.fit_transform(
    regions_gdf=hexagons,
    features_gdf=features,
    joint_gdf=joint_gdf
)
```

##### Highway2Vec (Road network embeddings)
```python
from srai.embedders import Highway2VecEmbedder

# Specialized embedder for road networks
highway2vec_embedder = Highway2VecEmbedder(
    hidden_size=64,     # Hidden layer size
    embedding_size=30   # Output dimensions (default: 30)
)

# Process road network into embeddings
road_embeddings = highway2vec_embedder.fit_transform(
    regions_gdf=hexagons,
    features_gdf=roads_gdf,  # Road segments with 'highway' tag
    joint_gdf=joint_gdf
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

## Roads Processing with Highway2Vec

### Road Network Processing with Highway2Vec
```python
from modalities.roads import RoadsProcessor

# Configure Roads processor
processor = RoadsProcessor({
    'data_source': 'osm_online',  # or 'pbf' with pbf_path
    'output_dir': 'data/processed/roads',
    
    # Road types to include (OSM highway tags)
    'road_types': [
        'motorway', 'trunk', 'primary', 'secondary',
        'tertiary', 'unclassified', 'residential', 'service'
    ],
    
    # Highway2Vec parameters (only 2!)
    'embedding_size': 30,  # Output dimensions (default: 30)
    'hidden_size': 64      # Hidden layer size (default: 64)
})

# Run pipeline
embeddings_path = processor.run_pipeline(area_gdf, h3_resolution=9)

# Note: Requires srai[torch] for Highway2Vec
```

### Highway2Vec Processing Workflow
```python
# The Roads processor with Highway2Vec executes:

# 1. Data Loading (OSM Online or PBF)
loader = OSMOnlineLoader() if data_source == 'osm_online' else OSMPbfLoader()
roads_gdf = loader.load(
    area_gdf,
    tags={'highway': road_types}  # Load specified road types
)

# 2. Filter to line geometries only
roads_gdf = roads_gdf[roads_gdf.geometry.type.isin(['LineString', 'MultiLineString'])]

# 3. H3 Regionalization
regionalizer = H3Regionalizer(resolution=h3_resolution)
regions_gdf = regionalizer.transform(area_gdf)

# 4. Spatial Joining (roads to hexagons)
joiner = IntersectionJoiner()
joint_gdf = joiner.transform(regions_gdf, roads_gdf)

# 5. Highway2Vec Embedding
# Highway2Vec analyzes the distribution of road types in each hexagon
# Uses an autoencoder to learn compressed representations
highway2vec_embedder = Highway2VecEmbedder(
    embedding_size=32,
    hidden_size=64
)

# 6. Training and transformation
embeddings_gdf = highway2vec_embedder.fit_transform(
    regions_gdf=regions_gdf,
    features_gdf=roads_gdf,
    joint_gdf=joint_gdf,
    trainer_kwargs={'max_epochs': 15, 'accelerator': 'auto'}
)

# 7. Output: 32-dimensional road network embeddings per hexagon
# Captures road hierarchy, connectivity, and spatial patterns
```

### Road Network Embeddings Features
```python
# Highway2Vec generates learned embeddings per H3 hexagon:
# - highway2vec_0 to highway2vec_31: 32D learned representations
#   These embeddings capture:
#   * Road type distribution (motorway vs residential)
#   * Network connectivity patterns
#   * Spatial road hierarchy
#   * Infrastructure density
#   * Learned features from autoencoder compression

# The autoencoder learns to:
# 1. Encode the road network characteristics into latent space
# 2. Reconstruct the original road distribution from embeddings
# 3. Capture non-linear relationships between road types
# 4. Identify spatial patterns in infrastructure

# Benefits over manual features:
# - Automatically learns relevant patterns
# - Captures complex interactions between road types
# - Produces dense, information-rich representations
# - Consistent dimensionality regardless of road complexity
```

---

## POI Processing with SRAI

### POI Processing with SRAI
```python
from modalities.poi import POIProcessor

# Configure POI processor
processor = POIProcessor({
    'data_source': 'osm_online',  # or 'pbf' with pbf_path
    'output_dir': 'data/processed/poi',
    
    # POI categories to load from OSM
    'poi_categories': {
        'amenity': True,      # All amenity types
        'shop': True,         # All shop types
        'leisure': True,      # All leisure types
        'tourism': True,      # Tourist attractions
        'office': True        # Office locations
    },
    
    # Feature configuration
    'compute_diversity_metrics': True,  # Shannon, Simpson, richness, evenness
    'use_hex2vec': False,               # Skip-gram embeddings (requires srai[torch])
    'use_geovex': False                 # Convolutional embeddings (requires srai[torch])
})

# Run pipeline
embeddings_path = processor.run_pipeline(area_gdf, h3_resolution=9)

# Note: Hex2Vec and GeoVex are optional and require srai[torch]
# They add learned embeddings on top of count and diversity features
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

### POI Processing Workflow Internals
```python
# The POI processor executes these steps:

# 1. Data Loading (OSM Online or PBF)
loader = OSMOnlineLoader() if data_source == 'osm_online' else OSMPbfLoader()
pois_gdf = loader.load(area_gdf, tags=poi_categories)

# 2. H3 Regionalization
regionalizer = H3Regionalizer(resolution=h3_resolution)
regions_gdf = regionalizer.transform(area_gdf)

# 3. Spatial Joining
joiner = IntersectionJoiner()
joint_gdf = joiner.transform(regions_gdf, pois_gdf)

# 4. Base Embeddings (Count-based)
count_embedder = CountEmbedder()
count_embeddings = count_embedder.transform(regions_gdf, pois_gdf, joint_gdf)

# 5. Diversity Metrics (Vectorized calculation)
# Shannon entropy: -sum(p * log(p))
# Simpson diversity: 1 - sum(p^2)
# Richness: count of categories > 0
# Evenness: Shannon / log(richness)

# 6. Advanced Embeddings (Concurrent execution)
# Hex2Vec: Biased random walks + Word2Vec
# GeoVex: Graph Neural Network autoencoder

# 7. Output: Combined DataFrame with all embeddings
# Saved as Parquet for efficient storage and loading
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

# Count features (from CountEmbedder):
# - amenity: Count of amenity POIs
# - shop: Count of shop POIs
# - leisure: Count of leisure POIs
# - tourism: Count of tourism POIs
# - office: Count of office POIs
# - craft: Count of craft POIs
# - emergency: Count of emergency POIs
# - total_poi_count: Total POIs in hexagon

# Diversity features (if compute_diversity_metrics=True):
# - poi_shannon_entropy: Category diversity measure
# - poi_simpson_diversity: Simpson diversity index
# - poi_richness: Number of unique POI categories
# - poi_evenness: Pielou's evenness (distribution uniformity)

# Contextual features (if use_hex2vec=True):
# - hex2vec_0 to hex2vec_31: 32D contextual embeddings
#   Generated using biased random walks on H3 neighborhood graph

# GNN features (if use_geovex=True):
# - geovex_0 to geovex_31: 32D graph neural network embeddings
#   Learned representations considering spatial relationships
```

### Complete Multi-Modal Workflow with SRAI
```python
from modalities.roads import RoadsProcessor
from modalities.poi import POIProcessor
import pandas as pd
import geopandas as gpd

# Load study area
study_area = gpd.read_file('netherlands_boundary.geojson')

# Configure processors with SRAI embedders
roads_config = {
    'data_source': 'osm_online',
    'h2v_model_params': {'embedding_size': 32, 'hidden_size': 64},
    'h2v_trainer_kwargs': {'max_epochs': 15, 'accelerator': 'auto'}
}

poi_config = {
    'data_source': 'osm_online',
    'compute_diversity_metrics': True,
    'use_hex2vec': True,
    'use_geovex': True,
    'hex2vec_params': {'dimensions': 32, 'num_walks': 15},
    'geovex_model_params': {'embedding_size': 32, 'hidden_size': 64}
}

# Initialize processors
roads_processor = RoadsProcessor(roads_config)
poi_processor = POIProcessor(poi_config)

# Generate embeddings (both use SRAI's advanced embedders)
h3_resolution = 9
roads_path = roads_processor.run_pipeline(study_area, h3_resolution)
poi_path = poi_processor.run_pipeline(study_area, h3_resolution)

# Load and combine embeddings
roads_df = pd.read_parquet(roads_path)
poi_df = pd.read_parquet(poi_path)

# Merge on H3 index (outer join to keep all hexagons)
combined_df = roads_df.merge(poi_df, on='h3_index', how='outer', suffixes=('_roads', '_poi'))

# Handle missing values
combined_df = combined_df.fillna(0)

# Summary
print(f"Combined shape: {combined_df.shape}")
print(f"Total features: {combined_df.shape[1]}")
print(f"Highway2Vec dims: 32")
print(f"Hex2Vec dims: 32")
print(f"GeoVex dims: 32")
print(f"Diversity metrics: 4")
print(f"Count features: {len(poi_config['poi_categories'])}")

# Feature groups in combined DataFrame:
# - highway2vec_*: Road network embeddings (32D)
# - hex2vec_*: POI contextual embeddings (32D)
# - geovex_*: POI GNN embeddings (32D)
# - poi_*_entropy/diversity/richness/evenness: Diversity metrics
# - amenity/shop/leisure/etc: POI counts by category
# - total_poi_count: Total POIs per hexagon
# - h3_resolution: Resolution level (9 in this case)
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