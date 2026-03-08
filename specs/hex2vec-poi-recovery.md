# Hex2Vec POI Recovery: Raw Counts to Spatial Embeddings

## Status: Draft

## Context

All Stage 2 training runs to date have consumed raw POI category counts (687 dimensions, columns P000-P686) instead of hex2vec spatial embeddings (50 dimensions, columns hex2vec_0..49). This was never the intended design. The hex2vec infrastructure was built (commit ab6e58f added `SUB_EMBEDDER_MAP` to `stage1_modalities/__init__.py` and sub-embedder path resolution to `concat.py`), but no training script ever invoked it. Every published result in `reports/` is trained on 781D input (64 AlphaEarth + 687 raw POI + 30 roads) instead of the intended 144D input (64 + 50 + 30).

### Why this matters

Raw POI counts are a 687-dimensional sparse categorical vector (one column per OSM tag combination). This is a terrible input for a GCN-based UNet:

1. **Dimensionality mismatch**: 687 POI dims vs 64 AlphaEarth dims means the reconstruction loss is dominated by POI. The model spends most capacity memorizing category counts rather than learning spatial structure.
2. **No spatial learning**: Raw counts carry zero information about spatial context. Hex2vec embeddings encode neighborhood co-occurrence patterns -- exactly what a place-representation model needs.
3. **Sparsity**: Most hexagons have zeros in most of the 687 columns. The model reconstructs sparse vectors, which is easy but uninformative.
4. **Curse of dimensionality**: 687D from a single modality in a 781D total input means POI accounts for 88% of the input space. AlphaEarth (64D, 8%) and roads (30D, 4%) are drowned out.

### What exists on disk

**Raw POI (currently used):**
- `stage1_unimodal/poi/netherlands_res9_latest.parquet` -- 691 cols (687 P-cols + 4 diversity metrics)
- `stage1_unimodal/poi/netherlands_res10_latest.parquet` -- same schema

**Hex2vec embeddings (available, unused):**
- `stage1_unimodal/poi/hex2vec/netherlands_res9_latest.parquet` -- 50D (hex2vec_0..49)
- `stage1_unimodal/poi/hex2vec_27feat/netherlands_res10_latest.parquet` -- 50D, trained on 27 input features

**Note:** The `hex2vec/` and `hex2vec_27feat/` directories represent two different hex2vec training runs. The `hex2vec/` directory is the canonical sub-embedder path that `SUB_EMBEDDER_MAP` resolves. The `hex2vec_27feat/` directory is a variant. Only `hex2vec/` has res9 data, which is what the current training pipeline (res9/8/7) needs.

## Scope of contamination

### Results that used raw POI (need annotation)

| File | What it contains | Status |
|------|-----------------|--------|
| `reports/2026-03-08-causal-emergence-phase1.md` | Multi-scale UNet R^2 = 0.531-0.542 | Annotate: trained on 781D with raw POI |
| `reports/2026-03-08-causal-emergence-visualizations.md` | CE visualization analysis | Annotate: based on raw-POI model |
| `data/.../stage2_multimodal/concat/embeddings/netherlands_res9_2022_raw.parquet` | 781D concat (64 AE + 687 POI + 30 roads) | Mark stale, regenerate |
| `data/.../stage2_multimodal/unet/` (checkpoints + embeddings) | Trained on 781D raw-POI input | Mark stale, retrain |

### Results that remain valid

- **AlphaEarth-only baselines**: Any result using only AlphaEarth embeddings is clean.
- **Stage 1 modality processors**: POI processor, hex2vec training, roads processor -- all correct.
- **Stage 3 analysis code**: Visualization and probe infrastructure is input-agnostic. No changes needed to analysis code itself.
- **Causal emergence finding (directionally valid)**: The multi-scale > single-scale conclusion holds regardless of input quality. The absolute R^2 values will change but the relative ordering (concat > avg > res9-only) is architectural, not input-dependent.

## Decision: Recovery plan

Four waves, each independently deployable. Waves 1-3 are the minimum viable fix. Wave 4 is an enhancement.

### Wave 1: Fix concat to load hex2vec

**File**: `stage2_fusion/concat.py` (no code change needed -- it already supports sub-embedders)

**Action**: The concat script already handles hex2vec via `SUB_EMBEDDER_MAP`. The fix is in how it is invoked. Currently, training scripts and manual runs use `--modalities alphaearth,poi,roads`. The fix is to use `--modalities alphaearth,hex2vec,roads`.

Regenerate the concat parquet:
```bash
python -m stage2_fusion.concat --modalities alphaearth,hex2vec,roads --study-area netherlands --resolution 9 --year 2022
```

This produces a 144D parquet (64 + 50 + 30) instead of the current 781D.

**Validation**: Confirm output has 144 columns. Confirm column prefixes are A (64), hex2vec_ (50), R (30).

### Wave 2: Update MultiResolutionLoader and training script

**File**: `stage2_fusion/data/multi_resolution_loader.py`

Changes:
1. Update `get_model_config()` hardcoded `"fused": 781` to `"fused": 144` (or better: remove the hardcoded value and read from loaded data).
2. Update docstring/comment on line 139 (`features_dict : dict with single key "fused" -> Tensor [N, 781]`) to reflect new dimensionality.

**File**: `scripts/stage2/train_full_area_unet.py`

Changes:
1. Update module docstring (line 7): `781D = 64 AE + 687 POI + 30 roads` --> `144D = 64 AE + 50 hex2vec + 30 roads`.
2. No functional code changes needed -- the script already reads feature_dim dynamically from the loaded data (line 100).

### Wave 3: Retrain and compare

Retrain FullAreaUNet on the new 144D input with identical hyperparameters to the raw-POI run:
- Resolutions: 9, 8, 7
- Hidden dim: 128
- LR: 1e-3
- Epochs: 500
- Patience: 100

**Comparison protocol**:
1. Run the same leefbaarometer linear probes (ridge + DNN) on new embeddings.
2. Compare R^2 per target dimension against the raw-POI baseline table from `reports/2026-03-08-causal-emergence-phase1.md`.
3. Write results to a new report: `reports/YYYY-MM-DD-hex2vec-recovery.md`.

**Expected outcome**: R^2 may go up or down. The model is now working with a 5.4x smaller input (144 vs 781), which means:
- Less reconstruction capacity wasted on sparse counts
- More balanced modality contribution (AE: 44%, hex2vec: 35%, roads: 21% vs AE: 8%, POI: 88%, roads: 4%)
- But also: less raw information available (687 category counts encode fine-grained POI type, hex2vec compresses to 50D)

If R^2 drops significantly, that tells us something important: the model was genuinely using POI category detail, and we need either higher-dimensional hex2vec or a different compression strategy. This is useful information either way.

### Wave 4: Fix ModalityFusion (enhancement, not recovery)

**Problem**: `MultiResolutionLoader._load_features()` returns a single `{"fused": tensor}` key. This means `ModalityFusion.modality_weights` is a 1-element parameter with zero gradient (softmax of a single value is always 1.0). The learnable per-modality weighting that ModalityFusion was designed for is dead code.

**Fix**: Update `_load_features()` to return per-modality tensors:
```python
features_dict = {
    "alphaearth": tensor[:, 0:64],
    "hex2vec": tensor[:, 64:114],
    "roads": tensor[:, 114:144],
}
```

This requires:
1. Storing column-to-modality mapping in the concat parquet metadata or a sidecar file.
2. Updating `FullAreaUNet.__init__` to receive the actual per-modality feature_dims dict.
3. Updating training scripts to pass per-modality dims.

**Consequence**: ModalityFusion's learnable weights become active. The model can learn to weight AlphaEarth vs hex2vec vs roads. Reconstruction loss becomes per-modality, preventing any single modality from dominating.

**Note**: Wave 4 breaks checkpoint compatibility with Wave 3 models. The state_dict key `fusion.projections.fused.*` changes to `fusion.projections.alphaearth.*`, `fusion.projections.hex2vec.*`, `fusion.projections.roads.*`. This is a clean break -- old checkpoints cannot be loaded.

## Alternatives considered

### A. Keep raw POI, reduce with PCA

The concat script already supports `--pca N`. We could do `--modalities alphaearth,poi,roads --pca 144`.

**Rejected**: PCA on the raw 781D is dominated by POI variance (687/781 = 88% of dims). The first 50 PCA components would mostly encode POI category structure, not useful cross-modal patterns. This defeats the purpose of balanced fusion.

### B. Use hex2vec_27feat instead of hex2vec

The `hex2vec_27feat/` directory contains a res10 variant trained on 27 input features. The canonical `hex2vec/` has res9.

**Rejected for now**: The training pipeline runs at res9/8/7. Only `hex2vec/` has res9 data. The `hex2vec_27feat/` variant could be useful for cone-based training (which operates at res10), but that is a separate concern.

### C. Concatenate both raw POI and hex2vec

Use `--modalities alphaearth,poi,hex2vec,roads` for a 831D input.

**Rejected**: This makes the dimensionality imbalance even worse and does not solve the fundamental problem. Raw counts add noise, not signal, for a spatial embedding model.

## Consequences

**Positive:**
- Input dimensionality drops from 781 to 144 (5.4x reduction)
- Balanced modality representation (44/35/21% instead of 8/88/4%)
- Hex2vec embeddings encode spatial co-occurrence, which is what place representation needs
- Existing infrastructure already supports this -- no new code required for Waves 1-2
- Faster training (smaller input projection layer, less memory)

**Negative:**
- All existing Stage 2 results become baseline comparisons rather than final results
- Wave 3 retrain costs ~15-30 min GPU time
- R^2 might drop if raw category detail was genuinely useful
- Wave 4 breaks checkpoint compatibility

**Neutral:**
- The causal emergence finding (multi-scale > single-scale) remains valid regardless
- Hex2vec quality is bounded by the POI processor's output -- garbage in, garbage out

## Documentation requirements

After recovery, every file in the data pipeline that references POI must state which format it uses:

1. **`stage2_fusion/concat.py`**: Already correct (uses `SUB_EMBEDDER_MAP` to resolve).
2. **`stage2_fusion/data/multi_resolution_loader.py`**: Add comment in `_load_features()` noting expected input format (hex2vec, not raw POI counts).
3. **`scripts/stage2/train_full_area_unet.py`**: Update module docstring to state 144D = 64 AE + 50 hex2vec + 30 roads.
4. **`reports/` affected files**: Add a header note: `> Note: This report uses raw POI counts (687D), not hex2vec spatial embeddings (50D). See specs/hex2vec-poi-recovery.md.`

## Acceptance criteria

1. **Wave 1 done**: `stage2_multimodal/concat/embeddings/netherlands_res9_2022_raw.parquet` contains 144 columns (A00-A63, hex2vec_0-49, R00-R29).
2. **Wave 2 done**: `MultiResolutionLoader.get_model_config()` no longer hardcodes 781. Training script docstring updated.
3. **Wave 3 done**: New UNet trained on 144D input. Leefbaarometer probe R^2 table compared to raw-POI baseline. Results written to `reports/`.
4. **Wave 4 done**: `_load_features()` returns per-modality tensors. `ModalityFusion.modality_weights` gradient is non-zero during training.
5. **Documentation done**: All four files listed above have correct POI format annotations.
6. **Reports annotated**: Both existing reports in `reports/` have header notes about raw POI.

## Implementation notes

- **Ordering**: Wave 1 -> Wave 2 -> Wave 3 are sequential. Wave 4 is independent of Wave 3 (but Wave 4 invalidates Wave 3 checkpoints, so Wave 3 should complete and be evaluated first).
- **No data deletion**: Raw POI parquets stay on disk. Old concat parquets stay (renamed with `_rawpoi` suffix if needed). Old checkpoints stay (for comparison).
- **Hex2vec res9 coverage**: Verify that `hex2vec/netherlands_res9_latest.parquet` covers the same hexagons as `alphaearth` and `roads` at res9. If coverage differs, the inner join in concat will drop hexagons. Check before Wave 1.
- **The `hex2vec_27feat/` variant**: Not part of this recovery. If cone-based training needs res10 hex2vec, that is a separate spec.
