# CLAUDE.md - Developer Instructions & Principles

Instructions for Claude Code and developers working with the UrbanRepML project.

## CRITICAL: WE USE SRAI, NOT H3 DIRECTLY

**This project uses SRAI (Spatial Representations for AI) for ALL H3 hexagon operations.**
- NEVER use h3-py directly
- Always use SRAI's H3Regionalizer
- SRAI provides H3 functionality plus additional spatial analysis tools
- This applies EVERYWHERE in the codebase

## Core Development Principles

1. **HONEST COMPLEXITY**: Development is hard. Wrangling parallel datasets during training is challenging. Late-fusion makes it manageable.
2. **ONE THING AT A TIME**: Focus on individual modalities before fusion.
3. **DENSE WEB OVER OFFSHOOTS**: Every component must connect meaningfully to the core pipeline.
4. **STUDY-AREA BASED**: All work is organized by study areas. Each area is self-contained.
5. **DATA-CODE SEPARATION**: Absolute boundary between data/ and code directories. Never mix.
6. **SRAI EVERYWHERE**: Use SRAI for all spatial operations, H3 tessellation, and neighborhood analysis.
7. **THINK BEFORE ADDING**: Deeply consider how new code integrates with existing architecture.
8. **ANTI-CLUTTER**: Keep documentation minimal and focused.

## Study Area Organization

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

## Two-Stage Architecture

### Stage 1: Individual Modality Encoders

Process each modality independently into H3-indexed embeddings:

```python
from srai.regionalizers import H3Regionalizer  # NOT import h3!
from modalities.alphaearth import AlphaEarthProcessor

regionalizer = H3Regionalizer(resolution=9)
regions_gdf = regionalizer.transform(area_gdf)

processor = AlphaEarthProcessor(config)
embeddings = processor.process_to_h3(data, regions_gdf)
```

**Active Modalities:**
- **AlphaEarth**: Pre-computed Google Earth Engine embeddings (PRIMARY, working)
- **POI**: OpenStreetMap points → categorical density features (partial)
- **Roads**: OSM network topology → connectivity metrics (partial)
- **GTFS**: Transit stops → accessibility potential (planned)
- **Aerial Imagery**: PDOK Netherlands → DINOv3 (optional)

### Stage 2: Urban Embedding Fusion

Three model architectures (all in `urban_embedding/models/`):

1. **UrbanUNet** (`urban_unet.py`): The OG that worked. Full study area processing with lateral accessibility graph. Multi-resolution U-Net (res 8-10) with ModalityFusion, SharedSparseMapping, symmetric 3-level encoder-decoder with skip connections, and per-resolution output heads.

2. **ConeLatticeUNet** (`cone_unet.py`): Most promising future direction. Cone-based hierarchical U-Net processing independent computational "cones" spanning res5→res10. Memory efficient (each cone ~1,500 hexagons vs ~6M for full graph), parallelizable, multi-scale.

3. **AccessibilityUNet** (`accessibility_unet.py`): Planned — accessibility-weighted variant using Hanssen's gravity model.

## Accessibility Graph Pipeline

1. **Floodfill Travel Time**: Calculate travel times with local cutoff
2. **Gravity Weighting**: Weight by building density (attraction)
3. **Percentile Pruning**: Keep only top percentile of edge strengths
4. **Multi-Resolution**: Different pruning thresholds per H3 level (5-11)

```python
from srai.neighbourhoods import H3Neighbourhood
neighbourhood = H3Neighbourhood()
neighbors = neighbourhood.get_neighbours(regions_gdf)
```

## Setup

```bash
# Clone and install with uv
git clone https://github.com/bertberkers/UrbanRepML.git
cd UrbanRepML
uv sync              # Install all dependencies
uv sync --extra dev  # Include dev tools
```

## SRAI region_id Index Standard

- **ALWAYS** work with SRAI's `region_id` index format
- **DO NOT** rename to `h3_index` — adapt scripts to use `region_id`
- H3Regionalizer creates: `GeoDataFrame` with `region_id` as index containing H3 hex strings

### TIFF Processing Architecture
```
1. Pre-regionalize study area (SRAI H3Regionalizer)
   ↓
2. For each TIFF: spatial intersection → find relevant hexagons
   ↓
3. Process pixels → spatial join with hexagons → aggregate by region_id
   ↓
4. Merge tiles → weighted average overlapping hexagons
   ↓
5. Output: embeddings indexed by region_id
```

**Key Insight**: Hexagons are defined by the study area, not by individual tiles. Each tile contributes data to pre-existing hexagons.

## Cone-Based Training Memory Optimization

**TRUE Lazy Loading** with individual cone files:
- Each cone saved as separate `cone_{hex}.pkl` file (~144 MB each)
- `LazyConeBatcher` loads only 32 files at a time (~4.5 GB vs ~60 GB for all cones)
- 92% memory reduction with on-demand loading

```python
from urban_embedding.data.hierarchical_cone_masking import (
    HierarchicalConeMaskingSystem,
    LazyConeBatcher
)

batcher = LazyConeBatcher(
    parent_hexagons=sorted(parent_hexagons),
    cache_dir="data/study_areas/netherlands/cones/cone_cache_res5_to_10",
    batch_size=32
)
```

### Hierarchical Consistency

When working with multiple H3 resolutions (e.g., res5-10 for cone-based models):
- `ConeDataset` automatically filters hexagons to ensure all are descendants of available parent hexagons
- Filters ~25% of res10 hexagons that have res5 parents outside the study area
- Ensures clean parent-child relationships throughout the hierarchy

## Archived Techniques Reference

### KDTree Pixel-to-Hexagon Mapping
Uses `scipy.spatial.cKDTree` for efficient nearest-neighbor pixel-to-hexagon assignment:
pre-compute hexagon centroids, build KDTree, query per-pixel, filter by max distance (~0.01 degrees).
Includes adaptive sampling: dense near tile edges, sparse in center.

### Gap Elimination for Tile Stitching
Eliminates tile boundary discontinuities by consistently averaging embeddings for ALL hexagons (including single-tile ones). Targeted gap elimination specifically handles boundary hexagons. Track quality via boundary_hexagons and filled_gaps metrics.

## Key Commands

```bash
# Process modalities for study area
python -m modalities.alphaearth --study-area netherlands --use-srai

# Run fusion pipeline
python -m urban_embedding.pipeline --study-area netherlands --modalities alphaearth,poi,roads

# Generate accessibility graphs
python scripts/accessibility/generate_graphs.py --study-area netherlands --use-srai

# Train cone-based model
python scripts/netherlands/train_lattice_unet_res10_cones.py
```

## Common Pitfalls

1. **Using h3-py directly** → Always use SRAI's H3Regionalizer
2. **Mixing data and code** → Keep strict separation
3. **Processing without study area** → Everything is study-area based
4. **Ignoring SRAI capabilities** → SRAI has built-in embedders and analysis tools
5. **Over-engineering** → Keep it simple, late-fusion is about manageability

## Essential Resources

- **SRAI Documentation**: https://srai.readthedocs.io/
- **SRAI GitHub**: https://github.com/kraina-ai/srai
- **H3 via SRAI**: https://srai.readthedocs.io/en/stable/user_guide/regionalizers/h3.html
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

---

*Last updated: February 2025*
