# Causal Emergence Phase 1: Multi-Scale UNet Highway Exits

**Date**: 2026-03-08
**Session**: causal-emergence-coord
**Commits**: `637a30a`, `09df3aa`
**Plan**: [.claude/plans/2026-03-07-causal-emergence-unet.md](../.claude/plans/2026-03-07-causal-emergence-unet.md)

## Hypothesis

Livability is a macro-causal phenomenon. The UNet's decoder outputs at res8 (neighborhood, ~450m) and res7 (district, ~1.2km) carry more effective information than res9 (hexagon, ~100m) alone. Averaging across decoder exits should beat micro-scale-only probing.

Theoretical frame: causal emergence (Rosas et al. 2025, arxiv 2510.02649).

## Result

**Confirmed.** Multi-scale averaging beats res9-only by +3.1pp mean R². Concatenation beats by +4.1pp.

```
Variant           | lbm   | fys   | onv   | soc   | vrz   | won   | mean
------------------|-------|-------|-------|-------|-------|-------|------
res9-only         | 0.263 | 0.344 | 0.506 | 0.663 | 0.739 | 0.485 | 0.500
multiscale-avg    | 0.292 | 0.379 | 0.526 | 0.667 | 0.830 | 0.491 | 0.531
multiscale-concat | 0.311 | 0.400 | 0.535 | 0.663 | 0.851 | 0.490 | 0.542
```

Full fusion progression:
```
concat (0.445) -> ring_agg (0.499) -> GCN (0.501) -> UNet-res9 (0.500) -> UNet-avg (0.531) -> UNet-concat (0.542)
```

## Per-Target Analysis

| Target | Delta (avg) | Delta (concat) | Interpretation |
|--------|-------------|----------------|----------------|
| vrz (Amenities) | +0.091 | +0.112 | Largest gain. Access to facilities is inherently macro-scale. |
| fys (Physical Env) | +0.035 | +0.056 | Neighborhood context matters for physical environment. |
| lbm (Overall) | +0.029 | +0.048 | Composite benefits from multi-scale across sub-dimensions. |
| onv (Safety) | +0.019 | +0.028 | Moderate gain. Safety has both local and area-level drivers. |
| won (Housing) | +0.006 | +0.005 | Minimal. Housing quality is determined at micro-scale. |
| soc (Social) | +0.004 | +0.000 | Flat. Social cohesion already well-captured at res9. |

Key insight: **different leefbaarometer dimensions operate at different causal scales**. Concat lets the probe choose per-target, which is why it beats uniform averaging.

## Method

- No retraining. Loaded existing checkpoint (`best_model.pt`, 500 epochs, loss 0.000152).
- Forward pass extracts decoder outputs at res9 (247K), res8 (52K), res7 (8.4K).
- Upsampling: `h3.cell_to_parent` assigns each res9 hex its parent's embedding at res8/res7.
- Avg: `(res9 + res8_up + res7_up) / 3`. Concat: `[res9; res8_up; res7_up]`.
- DNN probes: MLP hidden=256, 3 layers, 5-fold spatial block CV (10km blocks), patience=20, max 200 epochs.
- 121,368 hexagons with both embeddings and leefbaarometer targets.

## Files

### Scripts

- [scripts/stage2/extract_highway_exits.py](../scripts/stage2/extract_highway_exits.py) -- Extract multi-scale embeddings (~12s)
- [scripts/stage3/probe_multiscale_embeddings.py](../scripts/stage3/probe_multiscale_embeddings.py) -- DNN probes on all variants (~4min CUDA)

### Embeddings

All under `data/study_areas/netherlands/stage2_multimodal/unet/embeddings/`:

| File | Shape | Description |
|------|-------|-------------|
| `netherlands_res9_2022.parquet` | 247,281 x 128 | Micro-scale decoder exit |
| `netherlands_res8_2022.parquet` | 52,000 x 128 | Meso-scale at native res8 |
| `netherlands_res7_2022.parquet` | 8,445 x 128 | Macro-scale at native res7 |
| `netherlands_res9_multiscale_avg_2022.parquet` | 247,281 x 128 | Averaged highway exits |
| `netherlands_res9_multiscale_concat_2022.parquet` | 247,281 x 384 | Concatenated highway exits |

### Probe Results

> **Note (2026-04-19, audit §7 Q3 patch)**: The aggregate citations originally listed here (`multiscale_probe_results.csv`, `multiscale_comparison.png`, `multiscale_delta.png` under `stage3_analysis/`) were aspirational — those flat aggregate artifacts were never produced. The numeric R² table above is backed by the per-variant `metrics_summary.csv` files in the dated run directories listed below. To regenerate the aggregate CSV/plots, re-run `scripts/stage3/probe_multiscale_embeddings.py` (output goes to a new date-keyed run dir under `dnn_probe/YYYY-MM-DD/`).

Per-variant run directories (training curves, predictions, configs, `metrics_summary.csv`):
- `data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-07_multiscale_res9_only/`
- `data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-07_multiscale_multiscale_avg/`
- `data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-07_multiscale_multiscale_concat/`

## Decision

Per the plan's decision tree: avg > res9-only by >= 2pp --> **YES** --> self-supervised reconstruction creates scale-differentiated representations. Proceed to Phase 2a (UNet++ dense skip connections). Phase 2b (Circle Loss) becomes enhancement, not prerequisite.

## Next Steps

1. **Phase 2a**: UNet++ dense skip connections — better multi-scale feature extraction
2. **Phase 2b**: Circle Loss with gravity sampling — contrastive objective at every decoder exit
3. **Phase 2c**: HOPE temporal memory blocks (blocked on multi-year data)

See [.claude/plans/2026-03-07-causal-emergence-unet.md](../.claude/plans/2026-03-07-causal-emergence-unet.md) Phase 2 for full spec.
