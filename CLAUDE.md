# CLAUDE.md - Developer Instructions & Principles

Instructions for Claude Code and developers working with the UrbanRepML project.

## CRITICAL: SRAI-First, h3-py for Hierarchy Only

**This project uses SRAI (Spatial Representations for AI) as the primary spatial interface.**
- Use SRAI for: tessellation, neighborhoods, regionalization, spatial joins
- **Exception**: `h3` is acceptable for parent-child hierarchy operations that SRAI does not wrap:
  - `h3.cell_to_parent()`, `h3.cell_to_children()`, `h3.cell_to_center_child()`
  - `h3.get_resolution()`, `h3.cell_to_local_ij()` and similar introspection
- NEVER use h3 for tessellation or neighborhood queries — SRAI handles those
- Always index by `region_id` (SRAI convention), never `h3_index`

## Core Development Principles

1. **HONEST COMPLEXITY**: Development is hard. Wrangling parallel datasets during training is challenging. Late-fusion makes it manageable.
2. **ONE THING AT A TIME**: Focus on individual modalities before fusion.
3. **DENSE WEB OVER OFFSHOOTS**: Every component must connect meaningfully to the core pipeline.
4. **STUDY-AREA BASED**: All work is organized by study areas. Each area is self-contained.
5. **DATA-CODE SEPARATION**: Absolute boundary between data/ and code directories. Never mix.
6. **SRAI-FIRST**: Use SRAI for tessellation, neighborhoods, and regionalization. h3-py is OK for parent-child hierarchy traversal only.
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

## Three-Stage Architecture

### Stage 1: Individual Modality Encoders

Process each modality independently into H3-indexed embeddings:

```python
from srai.regionalizers import H3Regionalizer  # NOT import h3!
from stage1_modalities.alphaearth import AlphaEarthProcessor

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

Three model architectures (all in `stage2_fusion/models/`):

1. **FullAreaUNet** (`full_area_unet.py`): The OG that worked. Full study area processing with lateral accessibility graph. Multi-resolution U-Net (res 8-10) with ModalityFusion, SharedSparseMapping, symmetric 3-level encoder-decoder with skip connections, and per-resolution output heads.

2. **ConeBatchingUNet** (`cone_batching_unet.py`): Most promising future direction. Cone-based hierarchical U-Net processing independent computational "cones" spanning res5→res10. Memory efficient (each cone ~1,500 hexagons vs ~6M for full graph), parallelizable, multi-scale.

3. **AccessibilityUNet** (`accessibility_unet.py`): Planned — accessibility-weighted variant using Hanssen's gravity model.

### Stage 3: Analysis & Visualization

Post-training analysis and clustering (all in `stage3_analysis/`):
- **UrbanEmbeddingAnalyzer** (`analytics.py`): Cluster visualization and statistics
- **HierarchicalClusterAnalyzer** (`hierarchical_cluster_analysis.py`): Multi-scale clustering across H3 resolutions
- **HierarchicalLandscapeVisualizer** (`hierarchical_visualization.py`): Beautiful multi-resolution plots

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

### Stage Boundary Convention
- **Stage 1 parquet outputs** use `h3_index` as column name for backwards compatibility with existing saved data
- **Stage 2+** uses `region_id` internally (SRAI convention)
- The `MultiModalLoader` bridges this by normalizing column names at the stage1→stage2 boundary

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
- Each cone saved as separate `cone_{hex}.pkl` file (~12-23 MB each)
- `LazyConeBatcher` loads only 32 files at a time (~0.4-0.7 GB vs ~6-9 GB for all cones)
- Significant memory reduction with on-demand loading

```python
from stage2_fusion.data.hierarchical_cone_masking import (
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
# Stage 1: Process modalities for study area
python -m stage1_modalities.alphaearth --study-area netherlands --use-srai

# Stage 2: Run fusion pipeline
python -m stage2_fusion.pipeline --study-area netherlands --modalities alphaearth,poi,roads

# Generate accessibility graphs
python scripts/accessibility/generate_graphs.py --study-area netherlands --use-srai

# Train cone-based model
python scripts/netherlands/train_lattice_unet_res10_cones.py

# Stage 3: Analyze and visualize embeddings
python -m stage3_analysis.analytics --study-area netherlands
```

## Multi-Agent Workflow

**The main conversation agent (Claude Code) MUST delegate specialist work, not do it inline.**

### Delegation Rules
1. **Single-domain, small tasks** (one file edit, quick grep): direct action is fine
2. **Multi-step or multi-domain work**: spawn the **coordinator** agent, which orchestrates specialists
3. **Never do specialist work yourself** — if it touches spatial ops, model code, training, analysis, or testing, delegate to the matching specialist agent via the Task tool
4. **Each specialist writes its own scratchpad** — this is mandatory, not optional. The scratchpad is the agent's proof of work and its message to future sessions

### Stigmergic Logging (MANDATORY)
Every agent that does work MUST write a dated scratchpad entry (`.claude/scratchpad/{agent}/YYYY-MM-DD.md`) containing:
- **What I did**: actions taken, files modified, decisions made
- **Cross-agent observations**: what I read from other agents' scratchpads, what was useful, what confused me, what I disagree with or would do differently
- **Unresolved**: open questions, things that need follow-up

This is not optional documentation — it is the coordination mechanism. Without scratchpad entries, the next session starts blind.

### Why This Matters
- Scratchpads are how agents communicate across sessions (stigmergy)
- The coordinator reads all scratchpads to prioritize work
- The ego agent monitors scratchpads for process health
- The librarian's codebase graph is the shared map everyone orients from
- Cross-agent observations catch disagreements, confusion, and integration issues early

## Common Pitfalls

1. **Using h3-py for tessellation/neighborhoods** → Use SRAI for those. h3-py is OK only for hierarchy traversal (cell_to_parent, cell_to_children, etc.)
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

*Last updated: February 2026*
