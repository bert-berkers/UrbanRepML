# Probe Alpha: Embedding Quality Audit (2026-03-15)

## Setup

- **Target**: Leefbaarometer (6 dimensions), res9, year 20mix, ~130K hexagons
- **Probes**: Ridge regression (208D full) and DNN MLP (h=256, 3 layers, SiLU, 5-fold spatial block CV)
- **Input**: 208D late-fusion concat (AlphaEarth 64 + hex2vec 50 + Roads 30 + GTFS 64)
- **Scope**: 40 DNN + 4 linear probe runs catalogued; new ablation and normalization experiments

## Current R-squared Ceiling

Best overall: **normalized ring_agg k=10, DNN probe, 208D -- mean R2 = 0.556**.

| Target | Best R2 | Description |
|--------|---------|-------------|
| vrz (Amenities) | 0.801 | Easiest -- satellite captures amenity-rich areas well |
| soc (Social Cohesion) | 0.673 | POI signals contribute strongly here |
| onv (Safety) | 0.539 | Moderate difficulty |
| won (Housing Quality) | 0.506 | Moderate difficulty |
| fys (Physical Environment) | 0.474 | Hard |
| lbm (Overall Liveability) | 0.341 | Hardest -- composite of all 5 sub-indicators |

Previous best was res9_concat_all 144D at 0.544 (unnormalized, DNN). The 0.556 result uses per-modality z-scoring before ring aggregation.

## Normalization Unlocks Cross-Modality Interactions

Per-modality z-scoring before ring_agg k=10 boosted DNN R2 from 0.532 to 0.556 (+0.024), while Ridge R2 stayed flat at 0.5125 both times. The DNN-over-linear gap doubled from 0.019 to 0.043.

| Condition | Ridge R2 | DNN R2 | Gap |
|-----------|----------|--------|-----|
| Un-normalized ring_agg | 0.5125 | 0.5318 | 0.019 |
| Normalized ring_agg | 0.5125 | 0.5558 | 0.043 |

**Interpretation**: normalization puts modalities on the same scale, enabling the MLP to learn cross-modality feature interactions that linear models cannot capture. Without normalization, Roads' larger magnitude dominated the ring averaging, suppressing weaker but more informative signals.

## Modality Importance (Ablation)

Ridge probes on normalized concat-208D, single-modality and drop-one configurations:

| Config | lbm | fys | onv | soc | vrz | won | Mean | Drop-delta |
|--------|-----|-----|-----|-----|-----|-----|------|------------|
| Full (208D) | .268 | .386 | .478 | .628 | .729 | .441 | .488 | -- |
| AE-only (64D) | .237 | .334 | .433 | .583 | .682 | .406 | .446 | -0.089 |
| POI-only (50D) | .171 | .214 | .415 | .579 | .536 | .391 | .384 | -0.030 |
| Roads-only (30D) | .050 | .061 | .155 | .325 | .223 | .232 | .174 | -0.004 |
| GTFS-only (64D) | .011 | .000 | .090 | .090 | .117 | .057 | .061 | -0.002 |

Key takeaways:

1. **AlphaEarth carries ~90% of signal.** Best single modality for all 6 targets. Drop-delta 22x larger than Roads, 3x larger than POI.
2. **POI is the clear second.** Consistent ~0.03 contribution. Strongest on safety (onv) and social cohesion (soc) -- nearly matching AE on those targets. Ring_agg amplifies POI more than AE (discrete signals benefit more from spatial smoothing).
3. **Roads are redundant.** Solo R2 = 0.174, but all signal is already captured by AE+POI. Drop-delta = -0.004.
4. **GTFS is pure noise.** 97% background vectors confirmed zero-information. Dropping it actually improves lbm and fys by a hair. Solo R2 = 0.061.
5. **AE+POI only (114D) loses just 0.006 R2** vs full 208D. The embedding could be halved with negligible cost.

## 40-Run Census Highlights

Across all catalogued probe runs:

- **SAGEConv and 192D UNet underperform raw concat** -- learned representations lose information vs the input they were trained on
- **Ring_agg k=10 (zero parameters) beats all learned UNet variants** -- spatial smoothing preserves signal that reconstruction objectives destroy
- **Multiscale concat (384D) excels at vrz** -- the only method to break R2 = 0.85 on amenities, suggesting multi-resolution structure matters for facility access prediction

## Implications

The current embedding quality is the bottleneck, not the probe architecture (DNN-over-linear gap is only 0.02-0.04). Improving R2 requires better embeddings, not better probes.

Concrete next steps:
- Drop GTFS and Roads from the fusion pipeline (saves 94D, loses 0.006 R2)
- Gate or mask GTFS background vectors if the modality is retained at all
- Explore contrastive objectives that preserve discriminative signal (reconstruction destroys it)
- Ring aggregation as default preprocessing for all downstream tasks

## Data

- [Normalized ring_agg Ridge](../data/study_areas/netherlands/stage3_analysis/linear_probe/2026-03-15_normalized_ring_agg/)
- [Feature ablation (concat)](../data/study_areas/netherlands/stage3_analysis/linear_probe/2026-03-15_feature_ablation/)
- [Feature ablation (ring_agg)](../data/study_areas/netherlands/stage3_analysis/linear_probe/2026-03-15_feature_ablation_ring_agg/)
- [Ablation script](../scripts/stage3/run_modality_ablation.py)
