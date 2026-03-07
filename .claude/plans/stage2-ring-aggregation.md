# Stage 2: Ring Aggregation Implementation Plan

**Created**: 2026-03-07 by coordinator slate-glowing-fog
**Supra state**: focused, model_architecture=5, exploration=4, code_quality=4
**Suppress**: Stage 1 modalities, supra infrastructure, highway2vec debugging

## Context

Stage 1 res9 embeddings exist (commit f4b6510):
- AlphaEarth: 398K hexagons, 67 features (A00-A66), index=`h3_index` (NOT `region_id`)
- POI: 868K hexagons, 687 features (P000-P686), index=`region_id`
- Roads: 252K hexagons, 30 features (R00-R29), index=`region_id`
- Inner join coverage: ~121K hexagons where all three overlap

A parallel agent handles Stage 1 modality fixes and Stage 3 plotting. This plan is Stage 2 only.

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

## Architecture: Ring Aggregation (from thesis Ch 3.2.1 / Appendix 7.3)

Ring Aggregation is a spatial convolution that eliminates graph construction by using H3's k-ring indexing directly. SRAI's `H3Neighbourhood.get_neighbours_at_distance(k)` provides exact k-rings.

### Mathematical Formulation

For each central hexagon, given K concentric k-rings:

**Step 1 — Within-ring aggregation** (shared weights φ):
```
M_k = (1/I) * Σ_i f_φ(R_i)
```
Where `R_i` is the concatenated embedding of hexagon i in ring k, `I` is the number of hexagons in ring k.

**Step 2 — Across-ring aggregation** (shared weights θ):
```
S = Σ_k W_k * f_θ(M_k)
```
Where `W_k` is the ring weight, `S` is the final fused embedding for the central hexagon.

**Combined**:
```
S = Σ_{k=0}^{K} W_k * f_θ( (1/I) * Σ_{i=1}^{I} f_φ(R_i) )
```

### Weighting Schemes
- Exponential: `W_k = e^{-k}`
- Logarithmic: `W_k = 1 / log2(k + 2)`
- Linear: `W_k = 1 - k/K`
- Flat: `W_k = 1/K`

### Two Variants
1. **Simple**: Remove `f_φ`, `f_θ` — just mean per ring + weighted sum across rings. No learnable parameters in the aggregation itself.
2. **Learnt**: Include `f_φ`, `f_θ` as MLPs with shared weights. BatchNorm on first layer, GELU activation (not ReLU — causes excessive sparsity).

### Training (learnt variant)
- Circle loss (gamma=250, m=0.15)
- Triplet sampling based on proximity measure (Euclidean distance or location-based accessibility)
- Top 2% as positive pairs
- Batch size 256, lr 0.0001, Adam, no scheduler
- 1 + 6K hexagons per sample (e.g., K=5 → 31 hexagons per anchor)

### Key insight
Ring aggregation is a graph convolution without the graph. H3's isotropic hexagonal grid means k-rings are uniform in distance and count — no adjacency matrix needed. SRAI provides the k-ring lookups natively.

## Implementation Progression

| Level | What | Status | Depends on |
|-------|------|--------|------------|
| 0 | Cleanup dead code | Not started | Nothing |
| 1 | Concat + PCA baseline | 95% ready | Fix AE index bug |
| 2 | Simple Ring Aggregation | Not started | Concat working, SRAI k-rings |
| 3 | Learnt Ring Aggregation | Not started | Simple RA, circle loss, triplet sampling |
| 4 | UNet (FullAreaUNet) | 60% exists | Learnt RA concepts, graph + data pipeline |

## Wave Structure

### Wave 1: Cleanup (parallel)

**Agent: stage2-fusion-architect** — Delete dead code, fix broken imports
- DELETE `stage2_fusion/pipeline.py` (broken imports, hardcoded to old layout)
- DELETE `stage2_fusion/models/accessibility_unet.py` (empty placeholder)
- DELETE `stage2_fusion/utils/validation.py` (empty file)
- FIX `stage2_fusion/models/__init__.py` — remove AccessibilityUNet import
- FIX `stage2_fusion/__init__.py` — clean up commented-out imports if any remain valid
- Acceptance: `python -c "import stage2_fusion"` succeeds without errors

**Agent: stage2-fusion-architect** (or same agent) — Fix concat.py
- FIX the AE `h3_index` column → `region_id` index handling in `_load_modality`
- Verify: `python -m stage2_fusion.concat --modalities alphaearth,poi,roads --study-area netherlands --resolution 9 --pca 64` produces output parquet
- Acceptance: output parquet exists at `StudyAreaPaths.fused_embedding_file("concat", 9, 2022)`

### Wave 2: Ring Aggregation model (sequential after Wave 1)

**Agent: stage2-fusion-architect** — Implement `stage2_fusion/models/ring_aggregation.py`

Core classes:
```python
class RingAggregator(nn.Module):
    """Base ring aggregation — both simple and learnt variants.

    Uses SRAI H3Neighbourhood for k-ring lookups (no graph construction).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_rings=5,
                 weighting='exponential', learnt=True):
        ...

    def forward(self, embeddings: dict[str, torch.Tensor],
                hex_ids: list[str]) -> torch.Tensor:
        """
        Args:
            embeddings: {hex_id: concatenated_embedding_vector}
            hex_ids: list of central hexagons to compute fused embeddings for
        Returns:
            Tensor of shape (len(hex_ids), output_dim)
        """
        ...
```

Key implementation notes:
- Use `srai.neighbourhoods.H3Neighbourhood` for k-ring lookups
- `f_φ` (within-ring MLP): input_dim → hidden_dim, BatchNorm + GELU
- `f_θ` (across-ring MLP): hidden_dim → output_dim, GELU
- For simple variant: skip MLPs, just do mean + weighted sum, output_dim = input_dim
- Ring weights are NOT learnable — fixed by weighting scheme choice
- Weights φ and θ are shared across all rings (same MLP applied to every ring)
- Handle boundary hexagons where k-ring is incomplete (some hexagons missing from embeddings)

Also implement `stage2_fusion/data/ring_dataset.py`:
```python
class RingDataset(torch.utils.data.Dataset):
    """Dataset that pre-computes k-ring neighborhoods for all hexagons.

    Uses SRAI to build ring lookups once, then serves (anchor, positive, negative)
    triplets for circle loss training.
    """
```

Acceptance criteria:
- Simple ring aggregation can produce fused embeddings from concat parquets (no training needed)
- Learnt ring aggregation forward pass works on a small batch
- Both variants produce output shape (batch_size, output_dim)

### Wave 3: Training infrastructure (sequential after Wave 2)

**Agent: stage2-fusion-architect** — Training loop for learnt ring aggregation

- Implement circle loss in `stage2_fusion/losses/circle_loss.py` (or use an existing implementation from PyTorch Metric Learning)
- Implement triplet sampling in `stage2_fusion/data/ring_dataset.py` — Euclidean distance-based for now (simpler than accessibility)
- Training script: `scripts/stage2/train_ring_aggregation.py`
  - Load concat embeddings
  - Build k-ring lookup table via SRAI
  - Train learnt ring aggregation with circle loss
  - Save fused embeddings to `stage2_multimodal/ring_agg/`
  - Module docstring, `StudyAreaPaths` for all paths, lifetime: durable

Acceptance criteria:
- Training loop runs for at least 1 epoch without crashing
- Produces fused embedding parquet indexed by `region_id`

### Wave 4: Verification (parallel, after Wave 3)

**Agent: qaqc** — Verify all outputs
- `python -c "import stage2_fusion"` clean import
- Concat output parquet exists and has expected shape
- Ring aggregation model instantiates (both simple and learnt)
- No stale imports in stage2_fusion/ (grep for old h3 v3 API, broken module refs)
- Training script has module docstring and uses StudyAreaPaths
- Commit-readiness verdict

**Agent: stage3-analyst** — Probe ring aggregation output
- Run linear probes on simple ring aggregation output vs concat baseline
- Compare R² across leefbaarometer targets
- Does spatial context (ring aggregation) improve over naive concat?

### Wave 5: Commit (after Wave 4 passes)

**Agent: devops** — Stage and commit
- Commit message: `feat: ring aggregation fusion — simple and learnt variants with SRAI k-rings`
- Only commit if QAQC passes

### Final Wave: Close-out (mandatory)

- Write coordinator scratchpad to `.claude/scratchpad/coordinator/2026-03-07.md`
- `/librarian-update` — sync codebase_graph.md with new files
- `/ego-check` — process health assessment

## Out of Scope (for future sessions)

- UNet++ (arxiv 2602.17250) — evaluate after ring aggregation baseline exists
- FullAreaUNet integration with ring aggregation concepts
- Location-based accessibility as sampling heuristic (needs travel time computation)
- Street-view image modality
- Multi-year temporal analysis

## Execution

Invoke: `/coordinate .claude/plans/stage2-ring-aggregation.md`
