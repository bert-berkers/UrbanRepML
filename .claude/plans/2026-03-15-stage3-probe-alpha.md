# Stage 3: Squeeze More Alpha from Probes

**Intent**: Investigate whether current DNN probes leave R² on the table and find improvements. The leefbaarometer prediction task is our primary evaluation — if we can't probe it well, the embeddings aren't capturing liveability signal.

## Valuate guidance

This is an explorative Stage 3 analysis session. Suggested weights:
- model_architecture=4 (trying new probe architectures)
- exploration_vs_exploitation=4 (investigating, not just executing)
- code_quality=3 (experimental code, can be rough)
- test_coverage=3 (standard)
- spatial_correctness=3 (probes use existing spatial block CV)

Intent: "Audit existing probe results, identify R² ceiling, test alternative architectures and per-target feature selection"

## Context

Current probe results (from `dnn_probe/2026-03-14/`):
- MLP probes with 256 hidden, 3 layers, SiLU, 5-fold spatial block CV
- Ring_agg k=10 beats all UNet variants on leefbaarometer
- But R² values suggest room for improvement — especially on `onv` (safety) and `soc` (social cohesion) which are harder targets

Open questions:
- Are we bottlenecked by embedding quality or probe architecture?
- Would per-target feature selection help? Different targets may respond to different modality mixes.
- Are there non-linear probe architectures beyond MLP that would do better?
- The normalized concat (from Terminal 1) needs probing as soon as it's available.

Key files:
- `stage3_analysis/dnn_probe.py` — DNNProbeRegressor, MLPProbeModel
- `stage3_analysis/linear_probe.py` — LinearProbeRegressor (Ridge/Lasso)
- `scripts/stage3/run_probe_comparison.py` — 6 named configs for comparison runs
- `dnn_probe/2026-03-14/` — latest probe results (ring_agg, sageconv, fair_pca comparisons)

## Wave 0: Clean State
- `git status`, commit any dirty state
- Start `/loop 5m /sync` for lateral coordination
- Check `/sync` messages — if Terminal 1 has produced normalized concat embeddings, pick those up

## Wave 1: Baseline audit (parallel)

1. **stage3-analyst**: Compile a table of ALL existing probe results across `dnn_probe/` dated dirs. For each: embedding type, dimensionality, R² per target, overall mean R². Identify: which targets are easiest/hardest, which embeddings score best per target, what the current ceiling is.

2. **stage3-analyst**: Run linear probes (Ridge + Lasso) on the current best embedding (ring_agg k=10 if available, else concat 208D). Compare linear vs DNN R² per target. The gap tells us how much non-linearity the DNN is capturing. If linear ≈ DNN, the bottleneck is embedding quality, not probe architecture.

## Wave 2: Architecture experiments (parallel, depends on Wave 1 findings)

3. **stage2-fusion-architect**: If linear << DNN (probe architecture matters): try gradient boosted trees (XGBoost/LightGBM) as probes — they handle feature interactions differently than MLPs. Write a `GBTProbeRegressor` that fits the existing probe interface.

4. **stage3-analyst**: Per-target feature ablation. For each leefbaarometer target, run probes on each modality block independently (AE-only, POI-only, Roads-only, GTFS-only). Then try all 2-modality and 3-modality combos. This reveals which modalities matter for which targets and whether some modalities are noise for certain predictions.

## Wave 3: Normalized concat probes (depends on Terminal 1)

5. **execution**: Check if Terminal 1 has produced normalized concat embeddings via `/sync`. If yes, run the full probe suite on them. If not, skip — this wave can be deferred.

6. **stage3-analyst**: Compare normalized vs unnormalized probe results. Generate delta-R² table and spatial improvement maps.

## Wave 4: Synthesize + Commit

7. **stage3-analyst**: Write a summary report to `reports/` with findings: what worked, what didn't, recommended probe configuration going forward.
8. **qaqc**: Verify all new probe results are in dated subdirs. Run tests.
9. **devops**: Commit.

## Final Wave: Close-out
- Coordinator scratchpad
- `/ego-check`

## Execution

```
/valuate
```
Then:
```
/niche .claude/plans/2026-03-15-stage3-probe-alpha.md
```
