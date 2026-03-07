# Causal Emergence UNet: Highway Exits + Multi-Scale Averaging

**Created**: 2026-03-07 by pca-and-unet coordinator
**Supra state**: focused, model_architecture=5, code_quality=4
**Depends on**: commit 472be10 (UNet with real data pipeline)
**Suppress**: Stage 1 modalities, supra infrastructure, POI pipeline

## The Greater Picture

### Why the fusion progression plateaus — and why it's not what it looks like

The Stage 2 fusion progression tells a misleading story on the surface:

```
concat (0.445) → ring_agg (0.499) → GCN (0.501) → UNet (0.501)
```

The naive reading: "architecture refinement is exhausted, we need more data." But this misses what's actually happening. Ring aggregation's +5.4pp gain is the largest jump — simple spatial smoothing does more than learned graph convolution or multi-resolution encoding. Why?

Because **livability is not a micro-scale phenomenon.** A single hexagon's POI count, satellite features, and road connectivity don't determine livability — the *neighborhood's functional character* does. When ring aggregation averages features over k=3 rings, it approximates the macro-scale description where the causal structure actually lives. GCN and UNet don't improve further because they also ultimately output micro-scale (res9) embeddings — they just get there through fancier computation.

### Causal emergence as the theoretical frame

Rosas et al. (2025, arxiv 2510.02649) formalize this: macro-scale descriptions of complex systems can have **more causal power** (higher effective information, EI) than micro-scale descriptions. The coarse-grained model isn't a lossy compression — it's a better causal model.

For urban systems:
- **res9 (micro, ~100m)**: individual hexagon. High entropy, noisy, many confounders. Knowing "this hex has 3 cafes" tells you little about livability.
- **res8 (meso, ~450m)**: neighborhood. Noise averages out, functional character emerges. "This is a mixed-use residential area with good amenities" — this is predictive.
- **res7 (macro, ~1.2km)**: district. Urban typology crystallizes. "This is a gentrifying inner-city district" — this is where livability is causally determined.

The UNet's encoder-decoder architecture naturally computes these coarse-grained descriptions through its bottleneck. The res7 decoder exit isn't just "averaged res9 features" — it's a **nonlinear bottleneck-compressed representation** of district-level structure. That's precisely what causal emergence predicts is more informative.

### What we've been doing wrong

We extract embeddings only from the res9 (finest) decoder exit. This is like having a microscope, a regular camera, and a satellite, but only looking through the microscope to decide if a neighborhood is livable. The multi-scale information exists in the model — we just throw it away.

### The highway exit metaphor

Each decoder level is an exit ramp. Currently everyone is forced to drive to the last exit (res9). But some destinations — livability prediction, urban taxonomy, regional planning — are better served by earlier exits (res8, res7) or by blending information from all exits.

Averaging the decoder exits (after upsampling to shared geometry) creates a **learned hierarchical smoothing**: each res9 hexagon's embedding becomes a blend of micro (what it looks like), meso (what its neighborhood is like), and macro (what district type it's in). The downstream probe can then extract the signal from whichever scale carries it — without us having to know a priori.

## Context

From the pca-and-unet session (commit 472be10):
- FullAreaUNet trained: 500 epochs, 3.6 min, 789K params, loss 0.000152
- Model already outputs embeddings at all 3 resolutions: `{9: [247K, 128], 8: [52K, 128], 7: [8.4K, 128]}`
- Training script only saves res9. The other two are computed but discarded.
- `MultiResolutionLoader` provides parent-child sparse mappings for upsampling
- Deep research files in `deepresearch/`: UNet++ skeleton, HOPE memory blocks, Circle Loss spec

## Phase 1: Quick Test — Highway Exit Extraction

### Wave 1: Extract multi-scale embeddings (single agent)

**Agent: stage2-fusion-architect** — Extract what the model already computes

The model's `forward()` already returns `embeddings = {9: tensor, 8: tensor, 7: tensor}`. The training script just only saves `embeddings[9]`. This is likely a script change, not a model change.

1. **Load the existing checkpoint** (`data/study_areas/netherlands/stage2_multimodal/unet/checkpoints/best_model.pt`). Do a forward pass with `MultiResolutionLoader` data. Extract all 3 embedding dicts.

2. **Save per-resolution embeddings**:
   - `unet/embeddings/netherlands_res9_2022.parquet` (247K × 128D) — already exists, verify it matches
   - `unet/embeddings/netherlands_res8_2022.parquet` (52K × 128D) — NEW
   - `unet/embeddings/netherlands_res7_2022.parquet` (8.4K × 128D) — NEW

3. **Create upsampled + blended variants** (new script or extend existing):
   - Upsample res8 → res9: for each res9 hex, assign its res8 parent's embedding (via `h3.cell_to_parent`)
   - Upsample res7 → res9: for each res9 hex, assign its res7 grandparent's embedding
   - **Average**: `(emb_res9 + emb_res8_up + emb_res7_up) / 3` → `embeddings_res9_multiscale_avg.parquet` (247K × 128D)
   - **Concat**: `[emb_res9; emb_res8_up; emb_res7_up]` → `embeddings_res9_multiscale_concat.parquet` (247K × 384D)

   The parent-child mapping from `MultiResolutionLoader._hex_lists` provides the hex-to-parent lookup.

4. **No retraining.** Just checkpoint load → forward pass → save.

Acceptance: 5 parquet files (3 per-res + avg + concat), all with `region_id` index.

### Wave 2: Probe all variants (parallel)

**Agent: stage3-analyst** — DNN probes on multi-scale embeddings

Same methodology as previous probes (hidden=256, patience=20, max_epochs=200, 5-fold spatial block CV). Targets: all 6 leefbaarometer dimensions.

| Variant | Shape | Tests |
|---|---|---|
| res9 only (baseline) | 247K × 128D | Micro-only. Should reproduce 0.501. |
| res8 upsampled | 247K × 128D | Meso-only. Is neighborhood-level more predictive? |
| res7 upsampled | 247K × 128D | Macro-only. Is district-level more predictive? |
| Average (res9+8+7)/3 | 247K × 128D | **Core causal emergence test.** Should beat res9-only if EI peaks at coarser scale. |
| Concat [res9;res8;res7] | 247K × 384D | Lets probe choose per-target scale. If concat >> avg, different lbm targets have different causal scales. |

Key questions the results answer:
1. **Does averaged > res9-only?** → Causal emergence confirmed for UNet embeddings
2. **Which single-scale is best per target?** → Identifies the scale of causal determination for each lbm dimension
3. **Does concat > average?** → If yes, different targets operate at different scales (probe learns which)
4. **Is res8-only > res9-only for any target?** → Direct evidence that coarser = more causal for that target

**Agent: qaqc** — Verify embedding files

Quick check: shapes, index names, NaN count, duplicate check on all 5 parquets.

### Wave 3: Interpret + decide (coordinator OODA)

Coordinator reads probe results, compares against fusion progression table, writes OODA report.

**Decision tree:**

```
averaged > res9-only by ≥2pp?
├── YES → Causal emergence confirmed with current self-supervised objective
│         Proceed to Phase 2a (UNet++ dense skips) for better multi-scale extraction
│         The model already learns scale-differentiated representations
│
└── NO  → Self-supervised reconstruction doesn't incentivize scale-specific learning
          The decoder exits are redundant copies of roughly the same representation
          Phase 2b (Circle Loss with deep supervision) becomes the priority
          The OBJECTIVE must change before the ARCHITECTURE matters

Either way → Phase 2 is warranted, but the results determine the priority ordering
```

Per-target analysis may reveal mixed results (e.g., `soc` benefits from macro scale but `fys` doesn't). Report this — it's informative for Phase 2 loss design.

### Wave 4: Commit

**Agent: devops** — Commit all changes
- Message: `feat: multi-scale UNet embeddings — highway exits for causal emergence test`
- Include: extraction script/modifications, new parquets (if small enough, or just the code)

### Final Wave: Close-out (mandatory)
- Write coordinator scratchpad
- `/librarian-update`
- `/ego-check`

---

## Phase 2: Spatio-Temporal Nested Hex-UNet++ (future sessions)

This phase implements the architecture from the deep research files (`deepresearch/`). Execute after Phase 1 analysis. The ordering of 2a/2b depends on Phase 1 results.

### Phase 2a: UNet++ Dense Skip Connections

Replace additive skip connections with dense intermediate nodes per UNet++ (Zhou et al. 2018):

```
Current (UNet):   Enc1 ─────────────────── Dec1    (additive skip: x + skip)
                  Enc2 ──────── Dec2
                  Enc3 ── Dec3

UNet++ target:    Enc1 ── X01 ── X02 ── Dec1       (dense: each X sees all left + below)
                  Enc2 ── X11 ── Dec2
                  Enc3 ── Dec3
```

Each intermediate node `Xij`:
- Receives dense connections from all preceding nodes at the same resolution
- Receives upsampled output from the node below
- Processes through GCNConv blocks (same EncoderBlock/DecoderBlock pattern)
- Has its own deep supervision exit (output head + loss)

New file: `stage2_fusion/models/hex_unet_plus_plus.py`

The deep supervision exits are now first-class outputs, not afterthoughts — every node in the UNet++ grid produces a probed embedding. Per `deepresearch/path-scopedrule.md`: `forward()` MUST return `Dict[str, Tensor]` keyed by resolution, NOT a single tensor.

### Phase 2b: Circle Loss with Gravity Sampling

Replace MSE reconstruction with contrastive Circle Loss at every deep supervision exit:

- **Anchor**: a hexagon's embedding at resolution r
- **Positive**: embedding of a hexagon with high gravity-model accessibility to the anchor
- **Negative**: embedding of a random hexagon outside the accessibility cone

The gravity model: `A_ij = mass_j * f(distance_ij)` where mass is building density (from AlphaEarth) and distance is travel time (from floodfill). This encodes **functional similarity** — hexagons that are accessible to each other should have similar embeddings because they share an urban functional region.

Per `deepresearch/path-scopedrule.md`: Apply Circle Loss at EVERY resolution exit. This forces scale-specific learning — the res7 exit must predict macro-scale accessibility patterns, not just copy res9 features upward.

Dependencies:
- Accessibility graph data (floodfill travel times + gravity weights)
- Circle Loss sampler integrated with `MultiResolutionLoader`

### Phase 2c: HOPE Memory Blocks (temporal, deferred)

Per `deepresearch/pytorchskeleton.md`: multi-timescale associative memory. Coarser resolutions update slowly (low `update_frequency`), acting as "continuum memory" of regional structure. Finer resolutions update quickly, tracking local changes.

Only relevant when multi-year embeddings exist (2020, 2022, 2024). Defer until Stage 1 temporal pipeline delivers.

### Phase 2 task tracker

See `deepresearch/stigmergicscratchpad.md` for the full 4-phase breakdown with agent assignments.

## Risk Notes

- **Phase 1 is zero-risk**: no architecture change, no retraining, just extract + probe
- **Phase 2a** (UNet++): moderate — new model file, same data pipeline, existing loss
- **Phase 2b** (Circle Loss): high — requires accessibility graph, new loss, new sampler. Multiple sessions.
- **Phase 2c** (HOPE): low risk but blocked on temporal data
- **Memory**: concat 384D at 247K = ~380MB. Fits easily on RTX 3090.
- **The null result is informative**: if Phase 1 shows no multi-scale benefit, it means reconstruction loss doesn't create scale-differentiated representations — that's a FINDING (reconstruction is a weak objective for urban representation learning), not a failure

## Execution

Invoke: `/coordinate .claude/plans/2026-03-07-causal-emergence-unet.md`
