# CLAUDE.md - Developer Instructions & Principles

Instructions for Claude Code and developers working with the UrbanRepML project.

## ⚠️ CRITICAL: WE USE SRAI, NOT H3 DIRECTLY

**This project uses SRAI (Spatial Representations for AI) for ALL H3 hexagon operations.**
- NEVER use h3-py directly
- Always use SRAI's H3Regionalizer
- SRAI provides H3 functionality plus additional spatial analysis tools
- This applies EVERYWHERE in the codebase

## 🎯 Core Development Principles

1. **HONEST COMPLEXITY**: Development is hard. Wrangling parallel datasets during training is challenging. Late-fusion makes it manageable.

2. **ONE THING AT A TIME**: Focus on individual modalities before fusion. It's easier to develop when you tackle one component at a time.

3. **DENSE WEB OVER OFFSHOOTS**: Every component must connect meaningfully to the core pipeline. No isolated features.

4. **STUDY-AREA BASED**: All work is organized by study areas. Each area is self-contained with its own data.

5. **DATA-CODE SEPARATION**: Absolute boundary between data/ and code directories. Never mix.

6. **SRAI EVERYWHERE**: Use SRAI for all spatial operations, H3 tessellation, and neighborhood analysis.

7. **THINK BEFORE ADDING**: Deeply consider how new code integrates with existing architecture before adding it.

## 🌍 Study Area Organization

All processing is study-area based. Each study area contains:

```
data/study_areas/{area_name}/
├── area_gdf/           # Study area boundary
├── regions_gdf/        # H3 tessellation (via SRAI!)
├── embeddings/         # Per-modality embeddings
│   ├── alphaearth/
│   ├── poi/
│   ├── roads/
│   └── gtfs/
├── urban_embedding/    # Fused results
└── plots/             # Visualizations
```

Primary study areas:
- **netherlands**: Complete coverage for training volume (primary)
- **cascadia**: Coastal urban-forest interface
- **south_holland**: Dense urban subset
- Others as configured in `study_areas/configs/`

## 🏗️ Two-Stage Architecture

### Why Late-Fusion?

**Honest answer**: Because development is hard and handling multiple parallel datasets during training is challenging. Late-fusion lets us:
1. Develop one modality at a time
2. Debug issues in isolation
3. Prototype without breaking everything

### Stage 1: Individual Modality Encoders

Process each modality independently into H3-indexed embeddings:

```python
from srai.regionalizers import H3Regionalizer  # NOT import h3!
from modalities.alphaearth import AlphaEarthProcessor

# Always use SRAI for H3 operations
regionalizer = H3Regionalizer(resolution=9)
regions_gdf = regionalizer.transform(area_gdf)

# Process modality
processor = AlphaEarthProcessor(config)
embeddings = processor.process_to_h3(data, regions_gdf)
```

**Core Modalities:**
- **AlphaEarth**: Pre-computed Google Earth Engine embeddings (PRIMARY)
- **POI**: OpenStreetMap points → categorical density features
- **Roads**: OSM network topology → connectivity metrics  
- **GTFS**: Transit stops → accessibility potential
- **Aerial Imagery**: PDOK Netherlands → DINOv3 (if needed)

### Stage 2: Urban Embedding Fusion (U-Net)

Graph Convolutional U-Net with accessibility constraints:

```python
from urban_embedding.pipeline import UrbanEmbeddingPipeline
from urban_embedding.graph_construction import AccessibilityGraphConstructor

# Accessibility-based graph pruning
graph_constructor = AccessibilityGraphConstructor(
    use_floodfill=True,
    gravity_weighting=True,
    percentile_pruning=0.95  # Keep top 5% of edges
)

# Late fusion pipeline
pipeline = UrbanEmbeddingPipeline(config)
urban_embeddings = pipeline.run(
    modality_embeddings,
    spatial_graph=graph_constructor.build()
)
```

## 📊 Accessibility Graph Pipeline

Accessibility graphs guide the fusion network through spatial constraints:

1. **Floodfill Travel Time**: Calculate travel times with local cutoff (few minutes)
2. **Gravity Weighting**: Weight by building density (attraction)
3. **Percentile Pruning**: Keep only top percentile of edge strengths
4. **Multi-Resolution**: Different pruning thresholds per H3 level (5-11)

```python
# ALWAYS use SRAI for neighborhoods
from srai.neighbourhoods import H3Neighbourhood

neighbourhood = H3Neighbourhood()
neighbors = neighbourhood.get_neighbours(regions_gdf)
```

## 🔧 Development Workflow

### Setup

```bash
# Clone repository
git clone https://github.com/bertberkers/UrbanRepML.git
cd UrbanRepML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with SRAI
pip install -e .
pip install srai[all]  # Critical: SRAI with all components
```

### Processing New Study Area

```python
from srai.regionalizers import H3Regionalizer  # ALWAYS SRAI!
from study_areas.tools import StudyAreaManager

# Define study area
manager = StudyAreaManager()
area_gdf = manager.load_area('netherlands')

# Create H3 regions with SRAI
regionalizer = H3Regionalizer(resolution=9)
regions_gdf = regionalizer.transform(area_gdf)

# Save for consistent use
regions_gdf.to_parquet('data/study_areas/netherlands/regions_gdf/h3_res9.parquet')
```

## 📝 Code Style Guidelines

### SRAI Usage Examples

```python
# ✅ CORRECT: Using SRAI
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
from srai.embedders import Hex2VecEmbedder

regionalizer = H3Regionalizer(resolution=9)
regions = regionalizer.transform(area_gdf)

# ❌ WRONG: Direct H3
import h3  # NO!
h3.geo_to_h3(lat, lon, 9)  # NO!
```

### Function Documentation

```python
def process_study_area(
    area_name: str,
    h3_resolution: int = 9,
    use_srai: bool = True  # Always True!
) -> gpd.GeoDataFrame:
    """Process study area into H3 embeddings using SRAI.
    
    Args:
        area_name: Name of study area (e.g., 'netherlands')
        h3_resolution: H3 resolution level (5-11)
        use_srai: Must always be True - we use SRAI!
        
    Returns:
        GeoDataFrame with H3 regions from SRAI
        
    Note:
        Always uses SRAI's H3Regionalizer, never h3-py directly
    """
```

## 🚀 Key Commands

```bash
# Process modalities for study area
python -m modalities.alphaearth --study-area netherlands --use-srai

# Run fusion pipeline
python -m urban_embedding.pipeline --study-area netherlands --modalities alphaearth,poi,roads

# Generate accessibility graphs
python scripts/accessibility/generate_graphs.py --study-area netherlands --use-srai

# Analyze results
python -m urban_embedding.analytics --study-area netherlands
```

## ⚠️ Common Pitfalls

1. **Using h3-py directly** → Always use SRAI's H3Regionalizer
2. **Mixing data and code** → Keep strict separation
3. **Processing without study area** → Everything is study-area based
4. **Ignoring SRAI capabilities** → SRAI has built-in embedders and analysis tools
5. **Over-engineering** → Keep it simple, late-fusion is about manageability

## 📊 Performance Considerations

- **Memory**: SRAI operations can be memory-intensive for large areas
- **Chunking**: Process large study areas in tiles
- **Caching**: Save intermediate SRAI results (regions_gdf)
- **Parallel**: Use SRAI's built-in parallel processing where available

## 🔗 Essential Resources

- **SRAI Documentation**: https://srai.readthedocs.io/
- **SRAI GitHub**: https://github.com/kraina-ai/srai
- **H3 via SRAI**: https://srai.readthedocs.io/en/stable/user_guide/regionalizers/h3.html
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

## 📌 Remember

- **SRAI for ALL spatial operations** - This cannot be emphasized enough
- **Study areas organize everything** - Data, embeddings, results
- **Late-fusion for manageability** - One thing at a time
- **Honest about complexity** - Development is hard, that's why we chose this approach

---

*Last updated: January 2025*