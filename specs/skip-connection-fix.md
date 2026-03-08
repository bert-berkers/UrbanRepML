# Skip Connection Collapse Fix -- Implementation Spec

**Date**: 2026-03-08
**Author**: stage2-fusion-architect
**Status**: SPEC (no code changes)
**Affected file**: `stage2_fusion/models/full_area_unet.py`
**Training script**: `scripts/stage2/train_full_area_unet.py`

## Problem Statement

The FullAreaUNet produces embeddings at res7/8/9 with cosine similarity 0.9994. The multi-scale hierarchy has collapsed -- all three resolution levels encode nearly identical information despite having separate encoder blocks, decoder blocks, and output heads.

Despite this near-total collapse, causal emergence probing shows that the tiny remaining directional differences on the hypersphere DO carry useful scale-specific information (vrz R2 jumps from 0.739 at res9-only to 0.851 at multiscale-concat). This means the architecture has the right intuition but is being crushed by its own training dynamics.

## Diagnosed Causes (5)

### Cause 1: Consistency Loss Trains for Collapse (PRIMARY)

**Location**: `LossComputer.compute_losses()` lines 302-312

**Current behavior**: MSE between `sparse_mm(mapping.T, embeddings[fine])` and `embeddings[coarse]` with weight 1.0. This literally says: "the mean-aggregated fine embeddings should EQUAL the coarse embeddings." The model obeys.

**Why it is the primary cause**: This is the only loss term that directly couples resolutions. Without it, the encoder/decoder architecture would naturally develop some scale differentiation from the reconstruction objective alone. With it, the gradient actively pushes all resolutions toward the same point on the hypersphere.

**Before** (`full_area_unet.py` lines 300-312):
```python
# 2. Consistency loss between scales (with equal weighting)
consistency_losses = {}
for (res_fine, res_coarse), mapping in mappings.items():
    # Map fine embeddings to coarse resolution
    fine_mapped = torch.sparse.mm(mapping.t(), embeddings[res_fine])
    fine_mapped = F.normalize(fine_mapped, p=2, dim=1, eps=1e-8)
    coarse_emb = F.normalize(embeddings[res_coarse], p=2, dim=1, eps=1e-8)

    # Calculate MSE for this scale pair
    consistency_losses[(res_fine, res_coarse)] = F.mse_loss(fine_mapped, coarse_emb)

# Average the consistency losses (equal weight per scale transition)
total_consistency_loss = (sum(consistency_losses.values()) / len(consistency_losses)) * loss_weights['consistency']
```

**After** -- Replace MSE equality constraint with soft mutual information proxy (cosine similarity with a target margin):
```python
# 2. Hierarchical coherence loss (soft, NOT equality)
# Goal: aggregated fine and coarse should be RELATED but NOT IDENTICAL.
# Use cosine similarity with a target range [0.3, 0.8] rather than MSE toward 1.0.
coherence_losses = {}
target_sim = 0.6  # Coarse should be related to aggregated fine, not identical
for (res_fine, res_coarse), mapping in mappings.items():
    fine_mapped = torch.sparse.mm(mapping.t(), embeddings[res_fine])
    fine_mapped = F.normalize(fine_mapped, p=2, dim=1, eps=1e-8)
    coarse_emb = F.normalize(embeddings[res_coarse], p=2, dim=1, eps=1e-8)

    # Cosine similarity per node, penalize deviation from target
    cos_sim = (fine_mapped * coarse_emb).sum(dim=-1)
    coherence_losses[(res_fine, res_coarse)] = F.mse_loss(cos_sim, torch.full_like(cos_sim, target_sim))

total_coherence_loss = (sum(coherence_losses.values()) / len(coherence_losses)) * loss_weights['consistency']
```

**Key design choice**: `target_sim = 0.6` is a hyperparameter. It should be tuned, but the critical insight is that ANY value less than 1.0 breaks the collapse attractor. Start at 0.6 (moderate correlation) and sweep [0.3, 0.5, 0.7] after initial validation.

**Risk**: If set too low, resolutions become unrelated and the multi-scale hierarchy loses its semantic grounding.

---

### Cause 2: Six L2 Normalizations Crush Variance

**Locations** (6 total):
1. `ModalityFusion.forward()` line 50 -- per-modality projection output
2. `ModalityFusion.forward()` line 55 -- fused output
3. `SharedSparseMapping.forward()` line 80 -- cross-res mapping output
4. `EncoderBlock.forward()` line 119 -- encoder block output
5. `DecoderBlock.forward()` line 160 -- decoder block output
6. `FullAreaUNet.forward()` lines 264-266 -- all 3 output heads

Plus 2 more in `LossComputer.compute_losses()` lines 295-296 (pred AND target normalization in reconstruction loss).

**Why it matters**: L2 normalization projects all vectors onto the unit hypersphere. When applied at every processing stage, it eliminates magnitude as a differentiating signal. The only way resolutions can differ is in direction, and with consistency loss pushing directions together, the system converges to a single point.

**Before** (example from `EncoderBlock.forward()`):
```python
out = out + identity
return F.normalize(out, p=2, dim=-1, eps=1e-8)
```

**After** -- Keep L2 normalization ONLY at the final output heads (lines 264-266). Remove from all intermediate stages:
```python
# EncoderBlock.forward() -- REMOVE normalization
out = out + identity
return out  # Let magnitude carry information through the network

# DecoderBlock.forward() -- REMOVE normalization
out = out + identity
return out

# SharedSparseMapping.forward() -- REMOVE normalization
transformed = self.transform(mapped)
return transformed

# ModalityFusion.forward() -- REMOVE per-projection normalization (line 50)
# KEEP the final fused output normalization (line 55) as input conditioning
proj = self.projections[name](features)
projected[name] = proj  # Was: F.normalize(proj, p=2, dim=-1, eps=1e-8)
# ...
fused = sum(proj * w for (_, proj), w in zip(projected.items(), weights))
return F.normalize(fused, p=2, dim=-1, eps=1e-8)  # Keep this one -- input conditioning

# FullAreaUNet.forward() -- REMOVE output head normalization
# Let downstream consumers (probes, loss) normalize if needed
embeddings = {
    rf: self.output[str(rf)](d1),  # Raw output, no normalization
    rm: self.output[str(rm)](d2),
    rc: self.output[str(rc)](d3),
}

# LossComputer -- REMOVE double normalization in reconstruction loss
# The output embeddings are already normalized; the target features are raw.
# Normalizing both makes the loss purely angular, which is fine, but
# normalizing the prediction AGAIN (already normalized from output head) is redundant.
recon_losses[name] = F.mse_loss(pred, F.normalize(target, p=2, dim=1, eps=1e-8))
# Was: F.mse_loss(F.normalize(pred, ...), F.normalize(target, ...))
```

**Summary of normalization points** (human decision: only keep at input):
| Location | Line(s) | Action | Rationale |
|----------|---------|--------|-----------|
| ModalityFusion per-projection | 50 | REMOVE | Crushes per-modality variance before fusion |
| ModalityFusion fused output | 55 | KEEP | Only input conditioning point — human approved |
| SharedSparseMapping output | 80 | REMOVE | Prevents magnitude from encoding scale info |
| EncoderBlock output | 119 | REMOVE | Crushes encoder representations |
| DecoderBlock output | 160 | REMOVE | Crushes decoder representations |
| Output heads | 264-266 | REMOVE | Let downstream consumers normalize if needed |
| LossComputer pred | 295 | REMOVE | Already handled upstream |
| LossComputer target | 296 | KEEP | Target features need normalization for scale-invariant loss |

**Risk**: Removing intermediate normalizations may cause training instability (exploding activations). Mitigate with gradient clipping (already at 1.0) and consider reducing learning rate from 1e-3 to 5e-4. The LayerNorm inside each block (lines 91-93, 131-133) provides sufficient stabilization.

---

### Cause 3: Unweighted Additive Skip Connections

**Location**: `DecoderBlock.forward()` line 150

**Before**:
```python
combined = x + skip
```

**After** -- Gated skip connection with learned interpolation:
```python
# In DecoderBlock.__init__():
self.skip_gate = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.Sigmoid()
)

# In DecoderBlock.forward():
gate = self.skip_gate(torch.cat([x, skip], dim=-1))
combined = gate * x + (1 - gate) * skip
```

**Why gating works**: The sigmoid gate learns per-dimension which signal (decoder path vs encoder skip) is more informative. Early in training, the gate will be ~0.5 (uniform mixing). As training progresses, some dimensions will favor the decoder (new coarse-scale info) while others favor the encoder (fine-scale detail). This naturally produces scale-differentiated representations.

**Alternative considered**: Attention-based skip (as in Attention U-Net). Rejected for now -- attention is more expressive but adds significant parameters and the gating approach addresses the core issue. Can revisit if gating proves insufficient.

**Risk**: Low. Gating is a strict generalization of additive skip (the model can learn gate=0.5 to recover current behavior). The additional parameters (2 * hidden_dim * hidden_dim per decoder block = 3 * 2 * 128 * 128 = 98K params) are modest relative to the model size.

---

### Cause 4: Shared Mapping Transform

**Location**: `FullAreaUNet.__init__()` line 190, used at lines 246, 249, 255, 258

**Before**:
```python
# Single shared transform for ALL cross-resolution mappings
self.mapping_transform = SharedSparseMapping(hidden_dim)
```

**After** -- Separate transforms for encoder (fine-to-coarse) and decoder (coarse-to-fine) directions:
```python
# Separate transforms for encode vs decode direction
self.encoder_mapping = SharedSparseMapping(hidden_dim)  # fine -> coarse (aggregation)
self.decoder_mapping = SharedSparseMapping(hidden_dim)  # coarse -> fine (broadcasting)
```

And in `forward()`:
```python
# Encoder path -- fine to coarse (aggregation)
e1_mapped = self.encoder_mapping(e1, mappings[(rf, rm)].t())
# ...
e2_mapped = self.encoder_mapping(e2, mappings[(rm, rc)].t())

# Decoder path -- coarse to fine (broadcasting)
d3_mapped = self.decoder_mapping(d3, mappings[(rm, rc)])
# ...
d2_mapped = self.decoder_mapping(d2, mappings[(rf, rm)])
```

**Why**: Fine-to-coarse aggregation and coarse-to-fine broadcasting are fundamentally different operations. Aggregation should learn to summarize (mean + context). Broadcasting should learn to distribute (replicate + local refinement). Sharing a single transform biases both toward identity-like behavior, which contributes to collapse.

**Risk**: Minimal. Adds one `SharedSparseMapping` worth of parameters (~16K for hidden_dim=128). The two transforms will naturally specialize during training.

---

### Cause 5: ~~No Diversity Pressure in Loss~~ DROPPED

**Human decision**: Differentiation loss and consistency loss cancel each other out — one pushes resolutions apart, the other pushes them together. Pick one. We're keeping regularized consistency (Cause 1 fix) because it forces information through the hierarchy, which is the architectural intent. No differentiation loss needed.

**Updated loss_weights dict** (in `FullAreaModelTrainer.__init__`):
```python
self.loss_weights = loss_weights or {
    'reconstruction': 1.0,
    'consistency': 0.3,       # Was 1.0 -- reduced, soft target (cos_sim=0.6)
}
```

---

## Implementation Order

**Guiding principle**: Make the highest-impact, lowest-risk change first. Validate with cos_sim measurement before adding complexity.

### Phase 1: Fix the Loss (Cause 1 only) -- HIGHEST PRIORITY

**Estimated effort**: 30 minutes
**Expected impact**: cos_sim drops from 0.9994 to < 0.90

1. Replace MSE consistency loss with soft cosine coherence loss (Cause 1)
2. Reduce consistency weight from 1.0 to 0.3
3. Add `target_sim` as a hyperparameter (default 0.6)

**Why first**: The loss function is the root cause. No architectural change matters if the loss actively trains the model to collapse. This change is also the most contained -- it only touches `LossComputer` and `FullAreaModelTrainer.__init__`, leaving the model architecture intact. The regularized consistency loss preserves the architectural intent of forcing information through the hierarchy while allowing scale differentiation.

**Validation**: Train for 200 epochs with default hyperparameters. Measure cos_sim between resolutions at checkpoints [50, 100, 150, 200].

### Phase 2: Remove Intermediate Normalizations (Cause 2)

**Estimated effort**: 20 minutes
**Expected impact**: cos_sim drops further to < 0.80, embedding variance increases

1. Remove 4 intermediate `F.normalize` calls (see table above)
2. Remove redundant pred normalization in LossComputer
3. Reduce learning rate from 1e-3 to 5e-4 (stability precaution)
4. Monitor gradient norms during training (already computed, line 427)

**Why second**: Once the loss stops forcing collapse, removing normalizations lets the network exploit magnitude as a differentiating signal. But without fixing the loss first, removing normalizations alone won't help -- the loss will still push toward identity.

**Validation**: Compare gradient norm trajectories and loss curves with Phase 1-only baseline.

### Phase 3: Gated Skip Connections (Cause 3)

**Estimated effort**: 30 minutes
**Expected impact**: skip gate values diverge from 0.5, decoder develops scale-specific processing

1. Add `skip_gate` to `DecoderBlock.__init__`
2. Replace `combined = x + skip` with gated version
3. Log mean gate values per decoder block during training (diagnostic)

**Why third**: Gating is an architectural improvement that gives the decoder more expressiveness. It benefits most when the loss is already fixed (Phase 1) and the network can exploit magnitude (Phase 2). Without those, gating would learn gate ~= 0.5 because the loss doesn't reward differentiation.

**Validation**: Track gate statistics -- if mean gate diverges from 0.5 per block, the gating is actively contributing.

### Phase 4: Separate Mapping Transforms (Cause 4)

**Estimated effort**: 10 minutes
**Expected impact**: Minor further improvement

1. Split `self.mapping_transform` into `self.encoder_mapping` and `self.decoder_mapping`
2. Update forward pass to use appropriate transform per direction

**Why last**: This is the lowest-impact change. The shared transform contributes to collapse but is not the driver. It becomes more important if Phases 1-3 leave residual collapse.

**Validation**: Compare encoder-mapping and decoder-mapping weight matrices -- if they diverge significantly, the separation is justified.

---

## Success Criteria

### Quantitative Targets

| Metric | Current | Target (Phase 1) | Target (Phase 1+2+3) | Measurement Method |
|--------|---------|-------------------|----------------------|-------------------|
| cos_sim(res9, res8_upsampled) | 0.9994 | < 0.90 | < 0.70 | Mean cosine similarity between res9 embeddings and their parent's res8 embedding |
| cos_sim(res8, res7_upsampled) | ~0.999 | < 0.90 | < 0.70 | Same, one level up |
| Embedding variance (per res) | ~0.0 (on sphere) | > 0.01 | > 0.05 | Variance of L2 norms before final normalization |
| DNN probe mean R2 (multiscale-concat) | 0.542 | >= 0.542 | >= 0.56 | 6-target leefbaarometer DNN probe, 5-fold spatial block CV |
| DNN probe mean R2 (res9-only) | 0.500 | >= 0.500 | >= 0.50 | Must not regress |
| vrz R2 (multiscale-concat) | 0.851 | >= 0.851 | >= 0.87 | Most scale-sensitive target |

### Qualitative Targets

1. **Gate divergence**: Mean skip gate values should differ meaningfully across the 3 decoder blocks (d3, d2, d1), indicating scale-specific processing.
2. **Embedding visualization**: t-SNE/UMAP of res9 vs res8 vs res7 embeddings should show OVERLAPPING but DISTINGUISHABLE clusters (not identical point clouds).
3. **Reconstruction quality**: Reconstruction loss should not increase by more than 2x from baseline (currently 1.52e-4 total). Some reconstruction degradation is acceptable in exchange for scale differentiation.

### Failure Modes to Watch For

1. **Over-differentiation**: cos_sim drops below 0.3. Resolutions become semantically unrelated. Fix: increase `target_sim` or reduce `differentiation` weight.
2. **Training instability**: Loss diverges or oscillates wildly after removing normalizations. Fix: reduce learning rate, add gradient clipping, or restore ModalityFusion output normalization.
3. **Reconstruction collapse**: Model achieves low coherence loss but reconstruction loss explodes. Fix: increase reconstruction weight or restore some intermediate normalizations.

---

## Monitoring Script

After implementing changes, the training script should log these additional metrics per epoch:

```python
# Add to training loop after forward pass:
with torch.no_grad():
    for (rf_key, rc_key), mapping in mappings.items():
        fine_mapped = torch.sparse.mm(mapping.t(), embeddings[rf_key])
        fine_norm = F.normalize(fine_mapped, p=2, dim=1)
        coarse_norm = F.normalize(embeddings[rc_key], p=2, dim=1)
        cos_sim = (fine_norm * coarse_norm).sum(dim=-1).mean()
        logger.info(f"  cos_sim({rf_key},{rc_key}): {cos_sim:.4f}")

    # Gate statistics (if gating is enabled)
    for name, block in [('dec3', model.dec3), ('dec2', model.dec2), ('dec1', model.dec1)]:
        if hasattr(block, 'skip_gate'):
            # Would need to cache gate values during forward -- add as diagnostic
            pass

    # Embedding L2 norms before final normalization (requires model modification to expose)
    for res in resolutions:
        emb = embeddings[res]
        logger.info(f"  emb_norm_var(res{res}): {emb.norm(dim=-1).var():.6f}")
```

---

## Checkpoint Compatibility

All changes break backward compatibility with the existing `best_model.pt` checkpoint (epoch 499, loss 1.52e-4). This is acceptable because:

1. The existing checkpoint produces collapsed embeddings -- there is no value in preserving it for continued training.
2. The architecture changes (gated skip, separate mappings) alter `state_dict` keys.
3. The existing trained embeddings on disk (parquet files) remain valid for analysis.

**Action**: The existing checkpoint should be archived (renamed to `best_model_v1_collapsed.pt`) before retraining.

---

## Relationship to UNet++

UNet++ (nested dense skip connections) was evaluated and **rejected** for this problem. UNet++ adds more skip pathways between encoder and decoder at every level, which would:

1. Create more channels for fine-scale information to leak into coarse levels
2. Amplify the collapse problem by giving the consistency loss more surface area to enforce similarity
3. Add significant parameter overhead without addressing the root causes

The gated skip connection approach (Cause 3) is a targeted version of the same intuition -- give the model more control over information flow -- without the dense connectivity that would worsen collapse.

After Phases 1-4 are validated and cos_sim is in the target range (0.5-0.7), UNet++ could be revisited as a way to further enrich multi-scale representations. But it is not the right tool for fixing collapse.
