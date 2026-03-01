# CLAUDE.md - Developer Instructions & Principles

Instructions for Claude Code and developers working with the UrbanRepML project.

## Core Development Principles

1. **HONEST COMPLEXITY**: Development is hard. Wrangling parallel datasets during training is challenging. Late-fusion makes it manageable.
2. **ONE THING AT A TIME**: Focus on individual modalities before fusion.
3. **DENSE WEB OVER OFFSHOOTS**: Every component must connect meaningfully to the core pipeline.
4. **STUDY-AREA BASED**: All work is organized by study areas. Each area is self-contained.
5. **DATA-CODE SEPARATION**: Absolute boundary between data/ and code directories. Never mix. (See `.claude/rules/data-code-separation.md`)
6. **SRAI-FIRST**: Use SRAI for tessellation, neighborhoods, and regionalization. h3-py is OK for parent-child hierarchy traversal only. (See `.claude/rules/srai-spatial.md`)
7. **THINK BEFORE ADDING**: Deeply consider how new code integrates with existing architecture.
8. **ANTI-CLUTTER**: Keep documentation minimal and focused.

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
- **POI**: OpenStreetMap points -> categorical density features (partial)
- **Roads**: OSM network topology -> connectivity metrics (partial)
- **GTFS**: Transit stops -> accessibility potential (planned)
- **Aerial Imagery**: PDOK Netherlands -> DINOv3 (optional)

### Stage 2: Urban Embedding Fusion

Two model architectures (all in `stage2_fusion/models/`):

1. **FullAreaUNet** (`full_area_unet.py`): The OG that worked. Full study area processing with lateral accessibility graph. Multi-resolution U-Net (res 8-10) with ModalityFusion, SharedSparseMapping, symmetric 3-level encoder-decoder with skip connections, and per-resolution output heads.

2. **ConeBatchingUNet** (`cone_batching_unet.py`): Most promising future direction. Cone-based hierarchical U-Net processing independent computational "cones" spanning res5->res10. Memory efficient (each cone ~1,500 hexagons vs ~6M for full graph), parallelizable, multi-scale.

### Stage 3: Analysis & Visualization

Post-training analysis and clustering (all in `stage3_analysis/`):
- **UrbanEmbeddingAnalyzer** (`analytics.py`): Cluster visualization and statistics
- **HierarchicalClusterAnalyzer** (`hierarchical_cluster_analysis.py`): Multi-scale clustering across H3 resolutions
- **HierarchicalLandscapeVisualizer** (`hierarchical_visualization.py`): Beautiful multi-resolution plots

**Probe Infrastructure:**
- **LinearProbeRegressor** (`linear_probe.py`): Ridge/Lasso regression probes for leefbaarometer
- **DNNProbeRegressor** (`dnn_probe.py`): MLP-based DNN probes with cross-validation
- **ClassificationProber** (`classification_probe.py`): Logistic regression for urban taxonomy classification
- **DNNClassificationProber** (`dnn_classification_probe.py`): DNN classification with MLP
- **Target builders**: `leefbaarometer_target.py`, `urban_taxonomy_target.py`

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

## Utility Infrastructure

- **`utils/paths.py`** — `StudyAreaPaths`: single source of truth for all data path construction
- **`utils/spatial_db.py`** — `SpatialDB`: SedonaDB spatial engine with GeoPandas fallback for bulk H3 geometry queries (centroids, geometry, extent). Preferred over raw `h3_to_geoseries` for all visualization and analysis code.

### Study Area Data Layout

```
data/study_areas/{area_name}/
├── area_gdf/              # Study area boundary
├── regions_gdf/           # H3 tessellation
├── osm/                   # Historical OSM PBF files (shared by POI + roads)
│   ├── {area}-internal.osh.pbf    # Full history extract
│   ├── {area}-latest.osm.pbf     # Most recent snapshot
│   └── {area}-2022-01-01.osm.pbf # Date-specific snapshot
└── stage1_unimodal/       # Per-modality embeddings
    ├── alphaearth/
    ├── poi/
    └── roads/
```

Access via `StudyAreaPaths.osm_dir()`, `.osm_snapshot_pbf(date)`, `.osm_history_pbf()`. POI and roads processors auto-resolve PBF paths from `osm/` when `data_source='pbf'` and no explicit path is given.

## Script Organization

Three tiers in `scripts/`:

- **`scripts/{domain}/`** — Durable scripts that are part of the active workflow. Must have a module docstring and use `StudyAreaPaths` for all paths.
- **`scripts/one_off/`** — Temporary scripts (debug, migration, one-time plots). 30-day shelf life; coordinator flags stale ones for archive or deletion.
- **`scripts/archive/{category}/`** — Historical scripts kept for reference. Read-only.

**Every new script** requires a module docstring stating its purpose, lifetime (`durable`/`temporary`), and which stage it exercises. No hardcoded `data/study_areas/...` paths.

**No tests in scripts/**. Tests go in `tests/`. Debug scripts go in `one_off/`.

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

## Multi-Agent Workflow

The coordinator, SRAI, data-code, and index conventions are now enforced via:
- **Rules**: `.claude/rules/` (auto-loaded based on file context)
- **Hooks**: `.claude/settings.json` (SessionStart, SubagentStart/Stop, Stop)
- **Skills**: `/coordinate`, `/summarize-scratchpads`, `/ego-check`, `/librarian-update`

See `specs/claude_code_multi_agent_setup.md` for the full architecture description.

## Common Pitfalls

1. **Using h3-py for tessellation/neighborhoods** -> Use SRAI for those. h3-py is OK only for hierarchy traversal (cell_to_parent, cell_to_children, etc.)
2. **Mixing data and code** -> Keep strict separation
3. **Processing without study area** -> Everything is study-area based
4. **Ignoring SRAI capabilities** -> SRAI has built-in embedders and analysis tools
5. **Over-engineering** -> Keep it simple, late-fusion is about manageability

## Essential Resources

- **SRAI Documentation**: https://srai.readthedocs.io/
- **SRAI GitHub**: https://github.com/kraina-ai/srai
- **H3 via SRAI**: https://srai.readthedocs.io/en/stable/user_guide/regionalizers/h3.html
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

---

*Last updated: February 2026*
