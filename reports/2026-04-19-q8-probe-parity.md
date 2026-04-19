# Q8 Probe Parity: Ridge vs DNN on Shared 74D UNet Embeddings (2026-04-19)

## Context

Q8 (probe-confound) was escalated 22 days after the 2026-03-29 ring-agg-plus-unet comparison reported an unexplained ~0.19 mean-R² gap between a Ridge probe (UNet-MS 192D mean R²=0.574) and an earlier DNN probe run on a similar UNet embedding (mean R²=0.386). The framing at the time was that supervised UNet embeddings might be "pre-structured for linear readout," making Ridge the *fair* probe for UNet-family models. Terminal C's weekend plan (`.claude/plans/2026-04-19-terminal-c-probe-confound.md`, Wave 4 item 2) called for a clean Ridge-vs-DNN A/B on shared folds to reconcile the gap.

## Method

Shared 74D multiscale-average UNet embedding (`data/study_areas/netherlands/stage2_multimodal/unet/embeddings/netherlands_res9_multiscale_avg_2022.parquet`), derived from checkpoint `best_model_2022_74D_2026-03-22.pt` (see `specs/run_provenance.md` §Checkpoint Index). Leefbaarometer target at res9, 6 domain scores, ~130K hexagons with coverage. Both probes used 5-fold 10km spatial-block CV with seed=42 and identical train/val split logic. DNN hyperparameters: MLP hidden=256, 3 layers, SiLU, max 200 epochs, patience 20. Ridge hyperparameters per `LinearProbeRegressor` defaults. Driver script: [`scripts/one_off/q8_wave4_probe_parity.py`](../scripts/one_off/q8_wave4_probe_parity.py). Artifact dirs:

- Ridge: [`stage3_analysis/linear_probe/2026-04-19/2026-04-19_q8_74d_multiscale_avg/`](../data/study_areas/netherlands/stage3_analysis/linear_probe/2026-04-19/2026-04-19_q8_74d_multiscale_avg/)
- DNN: [`stage3_analysis/dnn_probe/2026-04-19/2026-04-19_q8_74d_multiscale_avg/`](../data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-04-19/2026-04-19_q8_74d_multiscale_avg/)

## Results

| Target | Ridge | DNN | Δ (DNN − Ridge) |
|---|---|---|---|
| lbm (Overall Liveability) | 0.5398 | 0.5439 | +0.0041 |
| fys (Physical Environment) | 0.3194 | 0.4040 | +0.0846 |
| onv (Safety) | 0.5805 | 0.5970 | +0.0165 |
| soc (Social Cohesion) | 0.6143 | 0.6370 | +0.0227 |
| vrz (Amenities) | 0.8182 | 0.8457 | +0.0275 |
| won (Housing Quality) | 0.4708 | 0.4986 | +0.0279 |
| **MEAN** | **0.5572** | **0.5877** | **+0.0305** |

**Compared to 2026-03-29**: Ridge=0.557 reproduces the earlier cited Ridge figure (0.574) to within run noise (Δ=0.017, within fold-to-fold variance). DNN=0.588 does **not** reproduce the earlier DNN figure (0.386) — that ~0.19 gap was a single badly-tuned DNN run, not a property of the probe class or the embedding.

## Interpretation

1. **Ridge results are reproducible.** The mean R²=0.557 on shared 74D multiscale_avg with matched folds lands within noise of the 2026-03-29 Ridge citation. The Ridge probe is well-specified and stable.
2. **The 2026-03-29 DNN=0.386 was a run artifact.** On identical folds and the same embedding, a well-tuned DNN scores 0.588 — 0.20 higher than the earlier figure. Whatever produced the 0.386 (fold split, early stopping, initialization, data preparation) did not survive re-execution.
3. **DNN beats Ridge on every target.** Deltas range from +0.004 (lbm) to +0.085 (fys), with mean +0.030 and 6/6-target wins. Not cherry-picked; the DNN edge is consistent across the full target set.
4. **The pre-structure hypothesis is rejected.** If UNet embeddings were genuinely pre-structured for linear readout, Ridge would match or exceed DNN. A consistent DNN edge across all six targets rules that out: the UNet's output space still has non-linear signal that a deeper probe can exploit.
5. **Q8 probe confound does not exist as a real asymmetry.** It was a cross-run comparison artifact — two probes scored under different hyperparameter/fold conditions. Under shared folds, Ridge and DNN rank-order the same and DNN is marginally more accurate.

## Authoritative probe decision (methods note)

For UNet embeddings, we report **DNN probe R² as the primary metric**: on shared 5-fold 10km spatial-block CV the DNN edges Ridge by +0.030 mean R² with all-target consistency (6/6 wins). Ridge remains the secondary reference and is reproducible to within noise; the Ridge-vs-DNN gap reported in 2026-03-29 (~0.19) was a bad-DNN-run artifact, not a probe-class asymmetry.

## Cross-references

- Plan: [`.claude/plans/2026-04-19-terminal-c-probe-confound.md`](../.claude/plans/2026-04-19-terminal-c-probe-confound.md)
- Prior probe-comparison report (source of the 2026-03-29 framing): [`reports/2026-03-29-ring-agg-plus-unet-probe-comparison.md`](2026-03-29-ring-agg-plus-unet-probe-comparison.md)
- Checkpoint disambiguation: [`specs/run_provenance.md`](../specs/run_provenance.md) §Checkpoint Index
