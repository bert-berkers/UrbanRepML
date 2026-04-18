# Ring Agg + UNet Combined Embedding Probe Comparison

## Status: Ready

## Objective

Test whether concatenating ring_agg (local k=10 neighbourhood smoothing, 208D) with UNet
multiscale (hierarchical multi-resolution, 192D) into a 400D combined embedding recovers
the fys regression that UNet caused while preserving the lbm/onv/vrz gains.

## Motivation

The tension in current results (DNN probes, 5-fold spatial block CV):

| Target | concat 74D | ring_agg k10 74D | UNet supervised 192D | Ring+UNet 400D |
|--------|------------|------------------|----------------------|----------------|
| lbm    | 0.305      | 0.321            | **0.559**            | ?              |
| fys    | 0.425      | **0.452**        | 0.354 (REGRESSED)    | ?              |
| onv    | 0.506      | 0.522            | **0.597**            | ?              |
| soc    | **0.634**  | **0.644**        | 0.627                | ?              |
| vrz    | 0.760      | 0.786            | **0.826**            | ?              |
| won    | 0.472      | 0.484            | 0.483                | ?              |
| MEAN   | 0.517      | 0.535            | **0.574**            | ?              |

**Sources:**
- concat 74D: `dnn_probe/2026-03-21/2026-03-21_custom_concat_74d/`
- ring_agg k10 74D: `dnn_probe/2026-03-21/2026-03-21_custom_ring_agg_k10/`
- UNet supervised 192D: `dnn_probe/2026-03-22/unet_supervised_multiscale/` (ProbeResultsWriter format)

UNet dramatically improves lbm (+0.238), onv (+0.075), vrz (+0.040) but regresses fys
(-0.098) and soc (-0.017). Ring_agg uniformly improves every target over raw concat. The
hypothesis: a DNN probe given both signals should learn to use UNet features for lbm/onv/vrz
and ring_agg features for fys/soc/won, yielding a Pareto improvement.

## Existing Infrastructure — Zero New Scripts

Everything lives in `scripts/stage3/run_probe_comparison.py`:
- `EmbeddingSource` dataclass for specifying embedding sources
- `pca_reduce()` helper with z-scoring + variance logging
- `_concat_path()`, `_unet_multiscale_path()` path resolvers
- `NAMED_CONFIGS` registry with builder functions
- `--write-standardized` flag for ProbeResultsWriter output
- DNN probe: hidden_dim=256, 3 layers, SiLU, patience=20, 200 epochs, 5-fold spatial block CV

**One file modified, zero new scripts.** The builder function handles combine + z-score + save inline.

---

## Wave 1: Add builder + run probes

**Agent**: `stage3-analyst`
**File**: `scripts/stage3/run_probe_comparison.py` (MODIFY only)

Add a `build_ring_agg_plus_unet` builder function that:
1. Loads ring_agg 208D from `paths.fused_embedding_file("ring_agg", H3_RESOLUTION, year)`
2. Loads UNet multiscale 192D from `_unet_multiscale_path(paths, year)`
3. Inner-joins on `region_id`, asserts <1% row loss
4. Z-scores the UNet 192D block (ring_agg already z-scored) — follows `feedback_normalize_before_fusion.md`
5. Saves combined 400D parquet to `output_dir / f"ring_unet_combined_{year}.parquet"`
6. Returns 4 `EmbeddingSource` entries: concat 208D, ring_agg 208D, UNet multiscale 192D, combined 400D

Register as `"ring_agg_plus_unet"` in `NAMED_CONFIGS`.

**Acceptance criteria:**
- New builder function follows existing pattern (same signature, same return type)
- Combined parquet created on first run, reused on subsequent runs
- 4 embedding sources returned with correct paths

## Wave 2: Execute probe comparison

**Agent**: `execution`
**Command**: `python scripts/stage3/run_probe_comparison.py --config ring_agg_plus_unet --write-standardized`

**Runtime**: ~60 min (4 sources x 6 targets x 5 folds = 120 probe runs)

**Acceptance criteria:**
- Comparison CSV + bar chart PNG saved to `stage3_analysis/dnn_probe/YYYY-MM-DD/`
- ProbeResultsWriter output saved
- Per-target R² table printed

## Wave 3: Read results together

**Agent**: coordinator + human
- Per-target R² comparison table (printed automatically by probe runner)
- Key questions:
  - Does Ring+UNet 400D recover fys to >= 0.45 (ring_agg level)?
  - Does Ring+UNet 400D maintain lbm >= 0.55 (UNet level)?
  - Mean R² above 0.574 (current best)?
- Write brief report to `reports/` if results are significant

## Final Wave: Close-out

**Parallel agents:**
- **librarian**: Update codebase graph + compact it (currently 3152 lines — should be a navigational index, not a full mirror)
- **ego**: Process health assessment + forward-look

---

## Risks

1. **Curse of dimensionality**: 400D with ~130K supervised targets might overfit.
   Mitigation: probe has weight_decay=1e-4 + early stopping (patience=20).

2. **Collinearity**: Ring_agg is smoothed concat, UNet is learned from concat. Shared upstream.
   Probe hidden layers should handle this. Monitor per-fold variance.

3. **Coverage mismatch**: Ring_agg and UNet multiscale may differ at edges.
   Assert <1% row loss on inner join.

---

## File Summary

| File | Action | Wave |
|------|--------|------|
| `scripts/stage3/run_probe_comparison.py` | MODIFY (add builder + config) | 1 |
| `data/.../stage3_analysis/dnn_probe/YYYY-MM-DD/...` | OUTPUT | 2 |

---

## Execution

Invoke: `/niche .claude/plans/ring-agg-plus-unet-probe-comparison.md`
