# Causal Emergence Visualizations: Scale-Dependent Livability

**Date**: 2026-03-08
**Companion to**: [Causal Emergence Phase 1](2026-03-08-causal-emergence-phase1.md)

Different livability dimensions operate at different causal scales. vrz (safety/amenity access) is the clearest case of true causal emergence -- it achieves BOTH higher R² (0.784) and higher Gini concentration (0.84) at res8 compared to res9 (R²=0.739, Gini=0.65). soc (social cohesion) shows the opposite: R² drops sharply at coarser scales, indicating micro-scale causation. The UNet's embedding cosine similarity of 0.9994 between res8 and res9 suggests skip connections may be collapsing scale differences, motivating UNet++ exploration.

## Diamond Plot

The diamond plot encodes two metrics per target per resolution: R² as horizontal width and Gini concentration as dot color/size. True causal emergence requires both a wider diamond (higher R²) AND a darker/larger dot (higher concentration) at coarser scale. Only vrz satisfies both criteria at res8.

![Causal emergence diamond plot](figures/causal-emergence/causal_emergence_diamonds.png)

## Lollipop Chart

Per-target R² across resolutions 7-8-9, showing the direction and magnitude of scale sensitivity. vrz trends upward from res9 to res8; soc and won trend sharply downward.

![Scale lollipop chart](figures/causal-emergence/scale_lollipop.png)

## Scale Fingerprint Radar

Radar chart comparing the per-target R² profiles at each resolution. The asymmetry between targets reveals which dimensions are scale-sensitive.

![Scale fingerprint radar](figures/causal-emergence/scale_fingerprint_radar.png)

## Causal Scale Matrix

Heat map of R² values across targets (rows) and resolutions (columns), with delta annotations. Provides the full numeric picture underlying the other visualizations.

![Causal scale matrix](figures/causal-emergence/causal_scale_matrix.png)

## Fusion Progression

End-to-end R² progression from naive concat through GCN, UNet-res9, and multi-scale variants. Multi-scale concat achieves R²=0.542 overall, +4.2pp over res9-only (0.500).

![Fusion progression](figures/causal-emergence/fusion_progression.png)

## Spatial Improvement Map (vrz)

Geographic distribution of vrz prediction improvement when moving from res9-only to multi-scale embeddings. Shows where macro-scale context helps most.

![Spatial improvement for vrz](figures/causal-emergence/spatial_improvement_vrz.png)

## Embedding Divergence

Cosine similarity distribution between res8 and res9 decoder outputs. Median similarity = 0.9994 -- the UNet's skip connections may be preventing meaningful scale differentiation. This motivates the UNet++ dense connection architecture (Phase 2a).

![Embedding divergence res8 vs res9](figures/causal-emergence/embedding_divergence_res8_res9.png)

## Key Findings

- **vrz shows true causal emergence**: R² increases from 0.739 (res9) to 0.784 (res8), AND Gini concentration increases from 0.65 to 0.84. Both metrics align -- the macro-scale representation is strictly more informative.
- **soc shows no emergence**: R² drops from 0.663 (res9) to 0.574 (res8) to 0.519 (res7). Social cohesion is a micro-scale phenomenon.
- **won (housing) also micro-scale**: R² drops from 0.485 to 0.421 at res8.
- **Embedding near-collapse**: cos_sim=0.9994 between res8 and res9 suggests skip connections are propagating micro-scale information into coarser levels, suppressing scale differentiation.
- **Concat > average**: multi-scale concat (R²=0.542) beats averaging (0.531) because the probe can learn per-target scale preferences rather than treating all scales equally.

## Data

Raw metrics: [causal_emergence_metrics.csv](figures/causal-emergence/causal_emergence_metrics.csv)
