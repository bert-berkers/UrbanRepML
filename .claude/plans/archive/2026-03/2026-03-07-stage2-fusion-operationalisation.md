# Stage 2: Fusion Operationalisation Plan

**Created**: 2026-03-07 by coordinator slate-glowing-fog
**Supra state**: focused, model_architecture=5, exploration=4, code_quality=4
**Suppress**: Stage 1 modalities, supra infrastructure, highway2vec debugging
**Handoff message**: `.claude/coordinators/messages/20260307_122900_slate-glowing-fog_handoff.json`

## Context

Stage 1 res9 embeddings exist (commit f4b6510):
- AlphaEarth: 398K hexagons, 67 features (A00-A66), index=`h3_index` (NOT `region_id`)
- POI: 868K hexagons, 687 features (P000-P686), index=`region_id`
- Roads: 252K hexagons, 30 features (R00-R29), index=`region_id`
- Inner join coverage: ~121K hexagons where all three overlap

A parallel agent handles Stage 1 modality fixes and Stage 3 plotting. This plan is Stage 2 only.

**Convention (existing)**: Neighbourhood objects (SRAI `H3Neighbourhood` pickles) are already computed and stored. Currently at `stage1_unimodal/poi/intermediate/neighbourhood/` but should be promoted to study area level since they're shared across all modalities and fusion levels. Proposed location: `data/study_areas/{area}/neighbourhood/` (they're serialized SRAI objects, not GeoDataFrames). `StudyAreaPaths` should gain a `neighbourhood_dir()` method. All fusion levels (ring agg, GCN, UNet) load from here — compute once, reuse everywhere.

## Audit Findings (Wave 1 of slate-glowing-fog)

### What works
- `concat.py`: 95% ready, fix AE `h3_index` → `region_id` index bug
- `models/full_area_unet.py`: Architecturally complete, needs data pipeline
- `models/cone_batching_unet.py`: Architecturally complete, has latent `nn.Sequential` bug
- `graphs/hexagonal_graph_constructor.py`: Working, minor `.reserve` bug on list
- `geometry/h3_geometry.py`: Clean utility
- `losses/cone_losses.py`: Clean
- `data/hierarchical_cone_masking.py`: Working (LazyConeBatcher)
- `data/cone_dataset.py`: Working but hardcoded to AlphaEarth only

### What's broken
- `pipeline.py`: Imports `stage2_fusion.analysis.analytics` which doesn't exist (moved to stage3). DELETE.
- `models/accessibility_unet.py`: Empty placeholder, raises NotImplementedError. DELETE.
- `models/__init__.py`: Imports broken AccessibilityUNet. FIX.
- `data/spatial_batching.py`: Uses old h3 v3 API (`h3_to_parent`). STALE.
- `training/unified_trainer.py`: `_prepare_full_data` returns dummy random data. Never used. REWRITE needed.
- `data/feature_processing.py`: Path mismatches with current data layout.
- `data/study_area_loader.py`: Path mismatches with current data layout.
- `data/study_area_filter.py`: Over-engineered with forestry/bioregional concepts. ARCHIVE candidate.
- `graphs/graph_construction.py`: Deprecated but `EdgeFeatures` dataclass used by other modules.
- `utils/validation.py`: Empty file.

## Progressive Architecture

Each level adds ONE new concept. Later levels reuse infrastructure from earlier ones.

| Level | Name | New concept | Learns? | Spatial? | Reuses from |
|-------|------|-------------|---------|----------|-------------|
| 0 | Cleanup | — | — | — | — |
| 1 | Concat + PCA | Baseline fusion | No | No | — |
| 2 | Simple Ring Agg | Spatial context via k-rings | No | Yes (SRAI k-rings) | Level 1 concat embeddings |
| 3 | GNN on H3 lattice | Graph convolutions (GCNConv) | Yes | Yes (PyG edge_index) | Level 2 ring structure, `HexagonalLatticeConstructor` |
| 4 | UNet / UNet++ | Multi-resolution encoder-decoder | Yes | Yes (hierarchical) | Level 3 GNN layers, graph construction |

### Why this order
- **1→2**: Does spatial context help at all? Simple ring agg answers this with zero learnable params.
- **2→3**: What do learnable graph convolutions add over fixed ring averaging? Introduces PyG + `HexagonalLatticeConstructor`, needed for UNet anyway.
- **3→4**: UNet is just a multi-resolution GNN with skip connections. All building blocks already exist.

### Why no learnt ring aggregation
Thesis results showed learnt RA (MLPs + circle loss + triplet sampling) added marginal value over simple RA for significant implementation cost. Skip straight to GCN which introduces the same learnability but with explicit graph structure — and that infrastructure is needed for UNet regardless.

## Ring Aggregation Reference (thesis Ch 3.2.1 / Appendix 7.3)

Ring Aggregation is a spatial convolution that eliminates graph construction by using H3's k-ring indexing directly. SRAI's `H3Neighbourhood.get_neighbours_at_distance(k)` provides exact k-rings.

### Mathematical Formulation

**Step 1 — Within-ring** (shared weights φ): `M_k = (1/I) * Σ_i f_φ(R_i)`
**Step 2 — Across-ring** (shared weights θ): `S = Σ_k W_k * f_θ(M_k)`
**Combined**: `S = Σ_{k=0}^{K} W_k * f_θ( (1/I) * Σ_{i=1}^{I} f_φ(R_i) )`

Where `R_i` = concatenated embedding of hex i in ring k, `I` = hex count in ring k, `W_k` = ring weight.

### Weighting Schemes
- Exponential: `W_k = e^{-k}`
- Logarithmic: `W_k = 1 / log2(k + 2)`
- Linear: `W_k = 1 - k/K`
- Flat: `W_k = 1/K`

### Variant used: Simple only
No `f_φ`, `f_θ` — just mean per ring + weighted sum across rings. No learnable params. Learnt variant (MLPs + circle loss) dropped — marginal gains for significant complexity.

## Wave Structure

### Wave 1: Cleanup + Concat baseline

**Agent: stage2-fusion-architect** — Delete dead code, fix imports, fix concat
- DELETE `pipeline.py`, `models/accessibility_unet.py`, `utils/validation.py`
- FIX `models/__init__.py` — remove AccessibilityUNet import
- FIX `__init__.py` — clean commented-out imports
- FIX `concat.py` — handle AE `h3_index` → `region_id` index
- RUN `python -m stage2_fusion.concat --modalities alphaearth,poi,roads --study-area netherlands --resolution 9 --pca 64`
- Acceptance: `import stage2_fusion` succeeds; concat output parquet exists

### Wave 2: Simple Ring Aggregation

**Agent: stage2-fusion-architect** — Implement simple ring aggregation

New file: `stage2_fusion/models/ring_aggregation.py`
- `SimpleRingAggregator` — no learnable params, just mean per ring + weighted sum
- Load pre-computed `H3Neighbourhood` from `data/study_areas/{area}/neighbourhood/` (move from current poi/intermediate location as part of this wave)
- Support all 4 weighting schemes (exponential, logarithmic, linear, flat)
- Handle boundary hexagons (incomplete k-rings)
- Add `StudyAreaPaths.neighbourhood_dir()` method
- Script: `scripts/stage2/run_simple_ring_aggregation.py` (durable, docstring, StudyAreaPaths)
  - Load concat embeddings, apply simple RA with configurable K and weighting
  - Save output to `stage2_multimodal/ring_agg/`

Acceptance: produces fused embedding parquet indexed by `region_id`; neighbourhood graph saved to intermediary

### Wave 3: GNN on H3 lattice

**Agent: stage2-fusion-architect** — Simple GCN using existing graph infrastructure

New file: `stage2_fusion/models/lattice_gcn.py`

- Load pre-computed `H3Neighbourhood` from `neighbourhood/`, convert to PyG edge_index via `HexagonalLatticeConstructor`
- Simple 2-3 layer GCNConv (from `torch_geometric.nn`) on the hex lattice at res9
- Input: concatenated embeddings (same as ring agg input)
- Training script: `scripts/stage2/train_lattice_gcn.py` (durable, docstring, StudyAreaPaths)

This level answers: do explicit message-passing graph convolutions improve over ring aggregation's k-ring averaging? Also builds PyG familiarity needed for UNet.

Acceptance: GCN trains, produces fused embedding parquet

### Wave 4: Verification + probes (parallel)

**Agent: qaqc** — Verify all outputs
- Clean `import stage2_fusion`
- All output parquets exist with expected shapes (concat, simple RA, GCN)
- No stale imports (old h3 v3 API, broken module refs)
- All scripts have docstrings and use StudyAreaPaths
- Commit-readiness verdict

**Agent: stage3-analyst** — Comparative probes
- Linear probes on ALL fusion outputs vs leefbaarometer targets
- R² comparison table: concat vs simple RA vs GCN
- Does spatial context help? Do graph convolutions add over ring averaging?

### Wave 5: Commit (after Wave 4 passes)

**Agent: devops** — Stage and commit
- Commit message: `feat: stage2 fusion progression — concat, ring aggregation, lattice GCN`
- Only commit if QAQC passes

### Final Wave: Close-out (mandatory)

- Write coordinator scratchpad
- `/librarian-update`
- `/ego-check`

## Future Sessions (out of scope here)

- **Level 5: UNet / UNet++** — Multi-resolution encoder-decoder using GNN layers from level 4. Evaluate UNet++ (arxiv 2602.17250). Wire up `FullAreaUNet` with real data pipeline.
- Accessibility graph as edge weights (travel time, not just adjacency)
- Location-based accessibility as sampling heuristic
- ConeBatchingUNet for memory-efficient training
- Multi-year temporal fusion

## Execution

Invoke: `/coordinate .claude/plans/2026-03-07-stage2-fusion-operationalisation.md`
