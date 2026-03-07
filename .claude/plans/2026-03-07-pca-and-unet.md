# Stage 2: PCA Investigation + UNet with Real Data

**Created**: 2026-03-07 by stage2-fusion-coord (second OODA)
**Supra state**: focused, model_architecture=5, code_quality=4
**Depends on**: commit c16570f (stage2 fusion progression)
**Suppress**: Stage 1 modalities, supra infrastructure

## Context

Stage 2 levels 1-3 complete (concat, ring agg, GCN). DNN probe results:
- Concat PCA-64: mean R²=0.448
- Ring Agg: 0.501
- GCN: 0.504

PCA bottleneck identified: 64-dim PCA loses ~10pp vs raw 144-dim (prev session concat=0.544).
FullAreaUNet exists but needs real data pipeline. It's designed for res10→res9→res8, but we only have res9 embeddings.

### Architecture investigation findings

**FullAreaUNet** (`full_area_unet.py`):
- Input: `features_dict` (per-modality tensors at finest resolution), `edge_indices` + `edge_weights` (per-res graphs), `mappings` (cross-res sparse matrices)
- 3-level symmetric U-Net: fine→coarse encoder, coarse→fine decoder with skip connections
- `ModalityFusion`: learned softmax-weighted sum across modalities
- Loss: reconstruction (MSE) + cross-scale consistency
- `FullAreaModelTrainer` is the real trainer (NOT `UnifiedTrainer` which has dummy data)

**What's missing for real data**:
1. Multi-resolution graphs (edge_index per resolution) — need fast construction
2. Cross-resolution sparse mappings (parent-child aggregation matrices)
3. Data loader that orchestrates parquets → graphs → mappings → GPU
4. Resolution adaptation: model is parametric, can work with any 3 consecutive resolutions

**Decision**: Adapt to res9→res8→res7 (our embeddings are res9). This avoids generating res10 stage1 embeddings.

**Quick wins also in scope**:
- Fix `spatial_batching.py:187` h3 v3 API (add `import h3`, `h3_to_parent` → `cell_to_parent`)
- Push 5 unpushed commits (ego flag x5)

## Wave Structure

### Wave 1: Quick wins + PCA experiment (parallel)

**Agent: devops** — Push commits + fix h3 v3 bug
- `git push` the 5 unpushed commits
- Fix `stage2_fusion/data/spatial_batching.py:187`: add `import h3`, change `h3.h3_to_parent` → `h3.cell_to_parent`
- Acceptance: pushed, bug fixed

**Agent: stage2-fusion-architect** — PCA dimensionality experiment
- Re-run `python -m stage2_fusion.concat` with `--pca 128` (save to different filename or temp location — don't overwrite the 64-dim output)
  - IMPORTANT: The current concat saves to a fixed path. Either modify to accept `--output` arg, or save the 64-dim first, run 128, then re-run 64 to restore. Or just accept the overwrite and re-run 64 after.
- Re-run `python -m stage2_fusion.concat` WITHOUT `--pca` (raw concatenated dims, ~781 features)
- Train LatticeGCN on 128-dim input, same config as before (500 epochs)
- Train LatticeGCN on raw dims (may need smaller hidden_dim if 781 input is too large)
- Report: R² comparison (DNN probes) for GCN-64 vs GCN-128 vs GCN-raw
- The DNN probes can be linear probes here for speed — the question is relative, not absolute
- Acceptance: clear answer on whether PCA-64 is the bottleneck

### Wave 2: UNet data pipeline

**Agent: stage2-fusion-architect** — Build multi-resolution data loader

Based on Wave 1 results, choose the input dimensionality for UNet.

New file: `stage2_fusion/data/multi_resolution_loader.py`

```python
class MultiResolutionLoader:
    """Load embeddings and construct multi-resolution graph structure for UNet.

    Builds the four inputs FullAreaUNet needs:
    - features_dict: per-modality tensors at finest resolution (res9)
    - edge_indices: per-resolution adjacency graphs (res9, res8, res7)
    - edge_weights: per-resolution edge weights (uniform for now, accessibility later)
    - mappings: cross-resolution sparse parent-child matrices
    """
```

Key components:
1. **Graph construction**: Use the fast SRAI H3Neighbourhood approach (not HexagonalLatticeConstructor) for res9. For res8 (~35K nodes) and res7 (~5K nodes), either approach works.
2. **Cross-resolution mappings**: For each res9 hex, `h3.cell_to_parent(hex, 8)` gives res8 parent. Build sparse `[N8, N9]` matrix. Same for res8→res7. This is h3 hierarchy traversal (allowed per CLAUDE.md).
3. **Feature loading**: Load concat parquet(s), split back into per-modality if FullAreaUNet's ModalityFusion needs it, OR feed as single "fused" modality.
4. **Device management**: Move everything to GPU.

Adapt FullAreaUNet config to use resolutions (9, 8, 7) instead of (10, 9, 8).

### Wave 3: UNet training

**Agent: stage2-fusion-architect** — Train FullAreaUNet with real data

New file: `scripts/stage2/train_full_area_unet.py` (durable)

- Uses `MultiResolutionLoader` from Wave 2
- Uses `FullAreaModelTrainer` from `full_area_unet.py` (the real trainer, NOT UnifiedTrainer)
- Self-supervised: reconstruction + consistency loss
- Extract res9 embeddings after training, save to `stage2_multimodal/unet/embeddings/`
- CLI args: epochs, hidden_dim, learning_rate, study_area
- Report training time, final loss, embedding shape

Acceptance: UNet trains to convergence, produces embedding parquet indexed by `region_id`

### Wave 4: Verification + probes (parallel)

**Agent: qaqc** — Verify all new outputs
- Clean imports
- Output parquets exist with expected shapes
- No new stale imports
- Scripts have docstrings and use StudyAreaPaths
- Commit-readiness verdict

**Agent: stage3-analyst** — DNN probes on UNet output
- Compare: concat vs ring agg vs GCN vs UNet
- Same methodology as the 2026-03-07 DNN probes (hidden_dim=256, patience=20, max_epochs=200, 5-fold spatial block CV)
- Save to `stage3_analysis/dnn_probe/YYYY-MM-DD_res9_stage2_unet_comparison/`
- Key question: does multi-resolution UNet beat single-resolution GCN?

### Wave 5: Commit

**Agent: devops** — Commit all changes
- Message: `feat: stage2 UNet with real data pipeline — multi-resolution loader, PCA investigation`

### Final Wave: Close-out (mandatory)

- Write coordinator scratchpad
- `/librarian-update`
- `/ego-check`

## Risk Notes

- **Memory**: FullAreaUNet processes all 247K res9 nodes + ~35K res8 + ~5K res7 in one forward pass. With hidden_dim=128 this should fit in RTX 3090 (24GB). If OOM, reduce hidden_dim or batch.
- **ModalityFusion vs pre-fused**: The model expects per-modality feature dicts. If we feed pre-fused concat, we bypass ModalityFusion — simpler but loses learned fusion weights. Decision for the architect agent.
- **PCA experiment may change plan**: If raw dims significantly outperform PCA-64, Wave 2-3 should use raw dims. The architect agent should adapt based on Wave 1 findings.

## Execution

Invoke: `/coordinate .claude/plans/2026-03-07-pca-and-unet.md`
