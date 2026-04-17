# Ring Agg + UNet Combined Embedding Probe Comparison (2026-03-29)

## Setup

- **Target**: Leefbaarometer (6 dimensions), res9, year 20mix, ~130K hexagons with target coverage
- **Probe**: Concat, RingAgg, Ring+UNet use DNN (MLP hidden=256, 3 layers, SiLU), 5-fold spatial block CV (10km blocks), max 200 epochs, patience 20. UNet-MS uses LinearProbeRegressor (Ridge regression) via `probe_supervised_unet.py` -- making its wins *more* impressive (weaker probe, stronger results).
- **Question**: Can concatenating ring_agg (good at fys) and UNet (good at lbm/onv/vrz) into a single 400D input give a DNN probe the best of both worlds?

Four conditions tested, spanning raw concat through learned multi-resolution embeddings to combined:

| Condition | Spatial context | Dims | Source |
|-----------|----------------|------|--------|
| Concat 74D | None | 74 | AE 64 + Roads 10, year 2022 |
| RingAgg k=10 74D | 10-hop exp. average, res9 | 74 | Zero parameters over concat |
| UNet-MS 192D | AccessibilityUNet, 3 resolution exits | 192 | Supervised multiscale, walk/bike/drive edges |
| Ring+UNet 400D | Both: 10-hop average + learned multiscale | 400 | 208D ring_agg + 192D UNet, inner join, UNet block z-scored |

Note: Concat and RingAgg use 74D (AE 64 + Roads 10) rather than 208D because POI (hex2vec 50D) and GTFS (64D) were excluded from the UNet training input. The 74D conditions match the UNet's input modalities for a fair comparison.

**Dimensionality confound**: Ring+UNet 400D uses 208D ring_agg (all 4 modalities) not 74D. The extra POI and GTFS features give Ring+UNet access to information absent from the standalone 74D RingAgg column. Prior results (03-15) show 208D ring_agg scores ~+0.02 over 74D, so some of Ring+UNet's fys gain likely comes from having POI/GTFS features rather than from combining paradigms.

## Target Correlation Structure

![Leefbaarometer Correlation Matrix](../data/study_areas/netherlands/target/leefbaarometer/plots/correlation_matrix.png)

The six leefbaarometer targets split into two groups:

- **Correlated cluster** (positive and negative): lbm, onv, soc, vrz, won are all mutually correlated. Some positively (onv-soc: 0.75, soc-won: 0.72, onv-won: 0.64, lbm-onv: 0.64), some negatively along the urban/rural axis (vrz-soc: -0.66, vrz-onv: -0.62, vrz-won: -0.53). Crucially, negative correlation is still correlation -- a model that learns "high vrz = low soc" exploits the same spatial signal with a flipped sign. UNet proves this by scoring well on both vrz (0.826) and soc (0.627).
- **Orthogonal outlier**: fys has *zero* correlations with the cluster (fys-onv: -0.00, fys-vrz: 0.00, fys-soc: -0.12). Not anti-correlated -- genuinely uncorrelated. Physical environment quality (noise, air quality, infrastructure condition) is driven by spatial patterns that are independent of the neighbourhood quality / urban-rural structure. This is why it requires different features entirely.

## Results

| Target | Concat 74D | RingAgg k10 74D | UNet-MS 192D | Ring+UNet 400D |
|--------|-----------|-----------------|-------------|----------------|
| lbm (Overall Liveability) | 0.305 | 0.321 | **0.559** | 0.334 |
| fys (Physical Environment) | 0.425 | 0.452 | 0.354 | **0.468** |
| onv (Safety) | 0.506 | 0.522 | **0.597** | 0.542 |
| soc (Social Cohesion) | 0.634 | 0.644 | 0.627 | **0.674** |
| vrz (Amenities) | 0.760 | 0.786 | **0.826** | 0.811 |
| won (Housing Quality) | 0.472 | 0.484 | 0.483 | **0.504** |
| **mean** | 0.517 | 0.535 | **0.574** | 0.555 |

## Findings

1. **UNet-MS wins overall (mean R²=0.574) and dominates the correlated cluster.** The supervised multiscale UNet outperforms all other conditions on lbm (+0.238 over ring_agg), onv (+0.075), and vrz (+0.040). These three targets share the neighbourhood quality pattern that the UNet's multi-resolution hierarchy captures well.

2. **fys remains ring_agg's territory.** Physical environment quality is orthogonal to the correlated cluster (r=0.00 with onv, -0.12 with soc). Ring_agg's k=10 neighbourhood smoothing captures this local property better than UNet's hierarchical abstraction. The combined 400D embedding improves fys further to 0.468 (+0.016 over ring_agg alone) -- the only target where concatenation genuinely helps.

3. **The combined 400D embedding finds a compromise, not a best-of-both.** The hypothesis was that a DNN probe given both ring_agg and UNet features could learn to use the right source per target. Instead, Ring+UNet 400D lands between the two individual sources on most targets. It improves fys and soc but loses lbm badly (0.334 vs 0.559 UNet).

4. **soc and won benefit from combination.** Social Cohesion (0.674) and Housing Quality (0.504) are the two targets where Ring+UNet 400D beats both individual sources. These targets may benefit from both local smoothing (ring_agg) and hierarchical context (UNet).

5. **vrz confirms the urban/rural axis is multi-scale.** Amenities/Services at 0.826 (UNet-MS) is the highest R² in the table. The anti-correlation with onv/soc/won (-0.53 to -0.66) reflects a spatial scale pattern that the UNet's res7/8/9 hierarchy captures naturally.

## Why Concatenation Underperforms

The DNN probe cannot selectively route features to targets because:

- Each target is probed independently -- a separate MLP is trained per target, with no cross-target learning.
- A shared 3-layer MLP processes all 400 features identically. There is no attention or gating mechanism to weight ring_agg features for fys and UNet features for lbm.
- At 400D input with ~130K training samples, the probe learns a compromise representation rather than selectively attending to the relevant feature block.
- The lbm result is telling: Ring+UNet (0.334) performs far worse than UNet alone (0.559) on lbm, falling close to ring_agg's 0.321. The additional ring_agg features actively interfere with the UNet signal for this target.

## Implications

The target correlation structure suggests the right architecture is **per-target routing**, not feature concatenation:

1. **Model selection**: Use UNet-MS embeddings for the correlated cluster (lbm, onv, vrz) and ring_agg for fys. For soc and won, the combined embedding offers a marginal gain.
2. **Supervised decoder head**: The already-implemented supervised UNet head with per-target output layers addresses this directly -- separate heads can specialize per target while sharing a multi-resolution encoder.
3. **Gating mechanism**: A learned gate that weights embedding sources per target would be the principled solution, but may be over-engineering given the model selection approach works.

The practical takeaway: **do not concatenate embeddings from different spatial learning paradigms and expect a DNN probe to sort it out.** The probe lacks the architectural structure to perform source selection.

## Data

- Concat 74D probe: [`dnn_probe/2026-03-21/2026-03-21_custom_concat_74d/`](../data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-21/2026-03-21_custom_concat_74d/)
- RingAgg k10 74D probe: [`dnn_probe/2026-03-21/2026-03-21_custom_ring_agg_k10/`](../data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-21/2026-03-21_custom_ring_agg_k10/)
- UNet-MS 192D probe: [`dnn_probe/2026-03-22/unet_supervised_multiscale/`](../data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-22/unet_supervised_multiscale/)
- Ring+UNet 400D probe: [`dnn_probe/2026-03-29/2026-03-29_custom_ring+unet_400d/`](../data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-29/2026-03-29_custom_ring+unet_400d/)
- Combined 400D embeddings: [`ring_unet_combined_20mix.parquet`](../data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-29/2026-03-29_ring_agg_plus_unet_comparison/ring_unet_combined_20mix.parquet)
- Probe comparison script: [`scripts/stage3/run_probe_comparison.py`](../scripts/stage3/run_probe_comparison.py) (config: `ring_agg_plus_unet`)
