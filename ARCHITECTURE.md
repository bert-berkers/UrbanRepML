# UrbanRepML Architecture Documentation

## 🎯 Project Vision

UrbanRepML creates **high-quality urban embeddings** through a **manageable late-fusion approach** that processes modalities independently before combining them with spatial awareness. Built entirely on **SRAI** (not h3-py) for all spatial operations.

### Core Innovation

Two-stage architecture that acknowledges the difficulty of multi-modal development:
1. **Individual modality encoders** create H3-indexed embeddings independently
2. **Spatial fusion network** combines embeddings using accessibility-based graph constraints

**Key Insight**: Late-fusion enables compartmentalized development while maintaining spatial coherence through accessibility graphs.

## 🌍 Study Area Organization

All work is organized by study areas, each containing a complete data ecosystem:

```
data/study_areas/{area_name}/
├── area_gdf/              # Study area boundary polygon
├── regions_gdf/           # H3 tessellation (via SRAI!)
│   ├── h3_res5.parquet   # Regional patterns
│   ├── h3_res8.parquet   # District analysis  
│   ├── h3_res9.parquet   # Neighborhood patterns
│   └── h3_res10.parquet  # Block-level detail
├── embeddings/            # Per-modality embeddings
│   ├── alphaearth/       # Primary visual features
│   ├── poi/              # Urban function indicators
│   ├── roads/            # Connectivity structure
│   └── gtfs/             # Transit accessibility
├── urban_embedding/       # Fused results
│   ├── embeddings/       # Final urban representations
│   ├── graphs/           # Accessibility graphs
│   └── models/           # Trained fusion networks
└── plots/                 # Visualizations and analysis
```

### Primary Study Areas

- **netherlands**: Complete coverage for maximum training data volume
- **cascadia**: Urban-forest interface research 
- **south_holland**: Dense urban subset for detailed analysis
- Additional areas configured in `study_areas/configs/`

## 🏗️ Two-Stage Architecture

### Stage 1: Individual Modality Encoders

**Philosophy**: Process one thing at a time because development is hard.

```python
# Always use SRAI for H3 operations
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood

# Create H3 regions
regionalizer = H3Regionalizer(resolution=9)
regions_gdf = regionalizer.transform(area_gdf)

# Process each modality independently
for modality in ['alphaearth', 'poi', 'roads', 'gtfs']:
    processor = get_processor(modality)
    embeddings = processor.process_to_h3(data, regions_gdf)
    save_embeddings(embeddings, f"data/study_areas/{area}/embeddings/{modality}/")
```

#### Core Modalities

1. **AlphaEarth Processor** (`modalities/alphaearth/`)
   - Pre-computed Google Earth Engine embeddings
   - Primary visual features for urban environments
   - Direct H3 indexing of satellite-derived features

2. **POI Processor** (`modalities/poi/`)
   - OpenStreetMap points of interest
   - Uses SRAI for spatial aggregation and density calculation
   - Categorical embeddings for urban function

3. **Roads Processor** (`modalities/roads/`)
   - OSM street network topology
   - SRAI-based connectivity analysis
   - Graph embeddings for accessibility potential

4. **GTFS Processor** (`modalities/gtfs/`)
   - Public transit stop locations and schedules
   - Accessibility zone calculation via SRAI
   - Transit-oriented development indicators

### Stage 2: Urban Embedding Fusion

**Philosophy**: Wrangling multiple parallel datasets during training is challenging. Late-fusion with spatial constraints makes it manageable.

```
┌─────────────────────────────────────────────────────────┐
│                INPUT: MODALITY EMBEDDINGS               │
│  AlphaEarth | POI | Roads | GTFS (per H3 cell)         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           ACCESSIBILITY GRAPH CONSTRUCTION               │
│  • Floodfill travel time calculation                   │
│  • Gravity weighting (building density)                │
│  • Percentile-based edge pruning                       │
│  • SRAI neighborhood analysis                          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│            GRAPH CONVOLUTIONAL U-NET                    │
│  • Multi-modal input concatenation                     │
│  • GCN layers on pruned accessibility graphs           │
│  • Multi-resolution processing (H3 5-11)               │
│  • Skip connections for hierarchical consistency       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              HIGH-QUALITY URBAN EMBEDDINGS              │
│  • Dense representations suitable for generation       │
│  • Multi-scale spatial awareness                       │
│  • Ready for downstream tasks                          │
└─────────────────────────────────────────────────────────┘
```

## 📊 Accessibility Graph Construction

**Core Innovation**: Spatial constraints guide the fusion network through accessibility relationships.

### Process (via SRAI)

1. **Floodfill Travel Time**
   ```python
   from srai.neighbourhoods import H3Neighbourhood
   
   # Calculate travel times with local cutoff
   neighbourhood = H3Neighbourhood()
   travel_times = calculate_floodfill_times(
       regions_gdf, 
       cutoff_minutes=5,
       use_srai_neighbors=True
   )
   ```

2. **Gravity Weighting**
   ```python
   # Weight edges by building density attraction
   weights = gravity_model(
       origin_density=building_counts,
       destination_density=building_counts,
       travel_times=travel_times
   )
   ```

3. **Percentile Pruning**
   ```python
   # Keep only strongest connections per resolution
   pruning_thresholds = {
       5: 0.99,  # Regional: very sparse
       8: 0.95,  # District: sparse
       9: 0.90,  # Neighborhood: moderate
       10: 0.85  # Block: denser
   }
   ```

### Multi-Resolution Hierarchy (via SRAI)

```
H3 Resolution Stack:
├── Res 5  (8.5km)    → Regional patterns, bioregional analysis
├── Res 8  (460m)     → District-level urban structure  
├── Res 9  (170m)     → Neighborhood accessibility zones
└── Res 10 (66m)      → Block-level daily patterns
```

## 🔧 Implementation Components

### Spatial Operations (SRAI Only)

```python
# ✅ CORRECT: All spatial operations via SRAI
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
from srai.embedders import Hex2VecEmbedder

# ❌ WRONG: Never use h3-py directly
# import h3  # FORBIDDEN!
```

### Data Flow

1. **Study Area Definition**
   ```python
   area_gdf = load_study_area('netherlands')
   regions_gdf = H3Regionalizer(resolution=9).transform(area_gdf)
   ```

2. **Modality Processing** (Independent)
   ```python
   for modality in ['alphaearth', 'poi', 'roads', 'gtfs']:
       embeddings = process_modality(modality, regions_gdf)
       save_study_area_embeddings(embeddings, 'netherlands', modality)
   ```

3. **Graph Construction** (SRAI Neighborhoods)
   ```python
   accessibility_graph = build_accessibility_graph(
       regions_gdf, 
       use_srai_neighborhoods=True
   )
   ```

4. **Fusion Training**
   ```python
   fusion_model = UrbanUNet(
       input_dim=sum(modality_dims),
       graph=accessibility_graph,
       resolutions=[5, 8, 9, 10]
   )
   ```

## 🎯 Design Principles

### Why This Architecture?

1. **Honest Complexity**: We acknowledge that development is hard
2. **Manageable Development**: One modality at a time
3. **Spatial Awareness**: Accessibility graphs provide meaningful constraints
4. **Study Area Organization**: Self-contained data ecosystems
5. **SRAI Integration**: Leverage existing spatial analysis tools

### Performance Considerations

- **Memory**: SRAI operations can be memory-intensive
- **Caching**: Save intermediate results (regions_gdf, embeddings)
- **Chunking**: Process large study areas in tiles
- **Graph Sparsity**: Percentile pruning reduces computational load

## 🔄 Data Pipeline

```
Raw Data Sources
├── Google Earth Engine (AlphaEarth)
├── OpenStreetMap (POI, Roads)
├── GTFS Transit Data
└── PDOK Netherlands (Aerial)
         │
         ▼
SRAI-based H3 Processing
├── Regionalizer creates H3 tessellation
├── Individual modality processors
└── Embeddings saved per study area
         │
         ▼
Accessibility Graph Construction
├── SRAI neighborhood analysis
├── Travel time calculation
└── Gravity-weighted pruning
         │
         ▼
Fusion Network Training
├── Graph Convolutional U-Net
├── Multi-resolution processing
└── High-quality urban embeddings
```

## 🚀 Future Extensions

### Planned Enhancements

1. **Aerial Image Generation**: Use embeddings to generate PDOK-style imagery
2. **Temporal Dynamics**: Multi-year embedding evolution
3. **Interactive Analysis**: Real-time exploration tools
4. **Transfer Learning**: Cross-study-area knowledge transfer

### Research Directions

- **Optimal Graph Pruning**: Adaptive percentile thresholds
- **Modality Weighting**: Learned importance scores
- **Hierarchical Consistency**: Cross-resolution constraints
- **Generative Applications**: Urban scenario modeling

---

## 📚 References

- **SRAI Framework**: https://srai.readthedocs.io/
- **H3 Specification**: https://h3geo.org/ (accessed via SRAI)
- **Graph Neural Networks**: PyTorch Geometric
- **Urban Computing**: Spatial analysis for cities

---

*This architecture reflects our honest approach to complex multi-modal urban modeling through manageable late-fusion development.*