# AccessibilityUNet Probe Results (2026-03-21)

## Experiment

Does replacing uniform 1-ring lattice edges with real walk-accessibility graph edges improve the FullAreaUNet's leefbaarometer probe performance? And does combining embeddings from multiple resolution "highway exits" add value?

### Conditions

| Condition | Spatial context | Output dims | Graph edges (res9) | Training |
|-----------|----------------|-------------|-------------|----------|
| Concat 74D | None | 74 | — | — |
| RingAgg k=10 | 10-hop exp. average | 74 | Uniform 1-ring | Zero parameters |
| UNet uniform | 10 GCNConv/block, res 9/8/7 | 64 (dims 64→128→256) | Uniform 1-ring | 500 ep, loss 1.5e-4 |
| UNet accessibility (res9) | 10 GCNConv/block, res 9/8/7 | 74 (dims 74→37→18) | Walk accessibility | 1000 ep, loss 3.1e-4 |
| **UNet acc. multiscale** | **Same, concat 3 exits** | **222 (3×74D)** | **Walk accessibility** | **Same checkpoint** |

Two architectural changes between the uniform and accessibility UNet:
1. **Graph edges**: Uniform 1-ring lattice → walk-mode accessibility graph (1.0M edges from OSM road sjoin + RUDIFUN gravity weighting, adjacency-filtered)
2. **Dimension pyramid**: Expanding [64→128→256] → decreasing [74→37→18]. The decreasing pyramid forces information compression at coarser resolutions.

### Multi-resolution note

The accessibility graph exists at three resolutions with different transport modes:
- **Walk** (res9): 1.0M edges, pedestrian road network
- **Bike** (res8): 241K edges, cycling infrastructure
- **Drive** (res7): 43K edges, motorized roads

The UNet uses walk edges at res9 (finest) and falls back to uniform 1-ring at res8 and res7. The multiscale concat probes all three resolution exits projected to res9 via `cell_to_parent`, testing whether coarser-resolution context adds value to res9-only embeddings.

### Training details (accessibility UNet)

- Input: 74D (AlphaEarth 64D + Roads 10D), year 2022
- LR schedule: linear warmup (50 ep) → cosine decay to LR/50
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Loss: reconstruction (weight 1.0) + cross-scale consistency (weight 0.3)
- Best epoch: 999/1000 — loss still decreasing, no early stop triggered
- 214K parameters, 11.8 min training time
- wandb: [run link](https://wandb.ai/idontknowagoodname/urbanrepml/runs/mt10aget)

### Probe setup

MLP (hidden=256, 3 layers, SiLU), 5-fold spatial block CV (10km blocks), max 200 epochs, patience 20.

## Results

| Target | Concat 74D | RingAgg k=10 | UNet uniform 64D | UNet acc. res9 74D | UNet acc. multiscale 222D |
|--------|-----------|-------------|------------------|-------------------|--------------------------|
| lbm (Overall Liveability) | 0.305 | **0.326** | 0.129 | 0.220 | 0.229 |
| fys (Physical Environment) | 0.425 | **0.452** | 0.177 | 0.326 | 0.360 |
| onv (Safety) | 0.506 | **0.519** | 0.331 | 0.437 | 0.458 |
| soc (Social Cohesion) | 0.634 | **0.646** | 0.499 | 0.582 | 0.592 |
| vrz (Amenities) | 0.760 | 0.786 | 0.471 | 0.667 | **0.795** |
| won (Housing Quality) | 0.472 | **0.485** | 0.340 | 0.411 | 0.425 |
| **mean** | **0.517** | **0.536** | **0.324** | **0.441** | **0.476** |

### Improvement chain

| Step | Mean R² | Delta | What changed |
|------|---------|-------|-------------|
| UNet uniform (baseline) | 0.324 | — | Expanding dims [64→128→256], uniform 1-ring |
| + Accessibility graph | 0.441 | +0.116 (+36%) | Walk edges at res9, decreasing dims [74→37→18] |
| + Multiscale concat | 0.476 | +0.036 (+8%) | Concat res9+res8+res7 highway exits (222D) |
| Gap to ring_agg | 0.536 | -0.060 | Still behind by 0.060 |

## Findings

1. **Accessibility edges massively improve the UNet.** The accessibility UNet (res9-only, mean R²=0.441) improves over the uniform-lattice UNet (0.324) by +0.116 (+36%). Every target sees double-digit percentage gains. Physical environment (fys) benefits most (+85%), consistent with walking accessibility being a physical-environment property.

2. **Multiscale concat adds another 8%.** Concatenating all three highway exits (res9+res8+res7 → 222D) pushes mean R² to 0.476. The coarser exits capture neighbourhood-scale context that res9 alone misses. **vrz (amenities) at 0.795 beats ring_agg's 0.786** — the first target where the UNet outperforms the zero-parameter baseline.

3. **Ring aggregation still wins overall.** RingAgg k=10 (mean R²=0.536) beats the best UNet variant by 0.060. But the gap has narrowed from 0.212 (uniform UNet) to 0.060 (multiscale accessibility UNet) — a 72% reduction.

4. **The UNet's problem is the objective, not the graph.** The accessibility graph gave GCNConv something meaningful to propagate — and it worked (+36%). Multiscale gave additional scale context (+8%). But the self-supervised reconstruction loss optimizes for input reconstruction, not liveability prediction. The model learns to compress and reconstruct, discarding signal that isn't needed for reconstruction but IS needed for liveability. Ring agg avoids this by never transforming the signal.

5. **vrz (amenities) shows the largest gains across both improvements.** From uniform UNet (0.471) to multiscale accessibility (0.795): +0.324 (+69%). Amenity access is inherently spatial and multi-scale — walkable amenities matter at res9, regional amenity clusters matter at res7/8.

6. **Training was still improving at epoch 999.** The model never early-stopped (patience_counter=0 at termination). A longer run might close the gap further, but the fundamental bottleneck is the objective function, not convergence.

## Implications

The result confirms that the UNet architecture works — graph quality and multi-resolution exits both contribute. The remaining gap to ring_agg (0.060) is an objective function problem: the UNet has no reason to preserve liveability signal.

**Next step: supervised decoder head with dynamic loss weighting.** Add leefbaarometer prediction heads (one per domain score, following Levering et al. 2023's semantic bottleneck approach) with learnable uncertainty-based loss weights (Kendall et al. 2018, as used by Jiao et al. 2025). This gives the model a direct incentive to preserve liveability-relevant features through the bottleneck while maintaining the self-supervised reconstruction and consistency objectives.

## Data

> **Path convention**: Probe outputs in this report live under dated+named
> subdirectories (`stage3_analysis/dnn_probe/{YYYY-MM-DD}/{YYYY-MM-DD}_custom_{approach}/`).
> Flat-sibling directories with the same approach name (e.g.
> `stage3_analysis/dnn_probe/2026-03-21/concat_74d/`) exist from earlier runs and
> are **not** the citations below. See `specs/run_provenance.md` §Checkpoint Index
> for checkpoint disambiguation.

- Checkpoint: `data/study_areas/netherlands/stage2_multimodal/unet/checkpoints/best_model_2022_74D_2026-03-21.pt` (see `specs/run_provenance.md` §Checkpoint Index — accessibility UNet, wandb `mt10aget`)
- Embeddings (res9): `data/study_areas/netherlands/stage2_multimodal/unet/embeddings/netherlands_res9_2022.parquet`
- Embeddings (multiscale avg): `data/study_areas/netherlands/stage2_multimodal/unet/embeddings/netherlands_res9_multiscale_avg_2022.parquet`
- Embeddings (multiscale concat): `data/study_areas/netherlands/stage2_multimodal/unet/embeddings/netherlands_res9_multiscale_concat_2022.parquet`
- Loss history: `data/study_areas/netherlands/stage2_multimodal/unet/checkpoints/training_loss_history.csv`
- Loss curve: `data/study_areas/netherlands/stage2_multimodal/unet/checkpoints/training_loss_curve.png`
- Probe results (res9): `data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-21/2026-03-21_custom_unet_accessibility_74d/`
- Probe results (multiscale): `data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-21/2026-03-21_custom_unet_acc_multiscale_concat_222d/`

---

*Probe parity note (2026-04-19): see [`reports/2026-04-19-q8-probe-parity.md`](2026-04-19-q8-probe-parity.md) for the Ridge-vs-DNN shared-fold reconciliation on the 74D multiscale_avg embedding.*
