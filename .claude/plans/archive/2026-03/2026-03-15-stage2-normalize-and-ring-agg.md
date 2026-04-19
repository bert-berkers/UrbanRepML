# Stage 2: Normalize Concat + Generate Ring_Agg Embeddings

**Intent**: Fix the variance domination problem in concat embeddings by adding per-modality normalization, then generate ring_agg embeddings that have been missing.

## Valuate guidance

This is a focused Stage 2 fusion session. Suggested weights:
- model_architecture=5 (core architectural change to concat pipeline)
- code_quality=4 (this touches shared infrastructure)
- spatial_correctness=4 (ring_agg needs correct neighbourhood computation)
- test_coverage=3 (standard)
- exploration_vs_exploitation=2 (we know what to do, just execute)

Intent: "Normalize concat embeddings per-modality, regenerate ring_agg k=10 exponential, re-probe to measure impact"

## Context

The 2026-03-15 concat variance plot showed Roads dominating ~97% of total variance in the raw 208D concat (AE:64 + hex2vec:50 + Roads:30 + GTFS:64). This is because highway2vec embeddings have much larger absolute values than other modalities. Per-modality z-score normalization before concatenation is the standard fix for late-fusion. See memory `feedback_normalize_before_fusion.md`.

Ring_agg was the best-performing model (k=10 exponential beat all UNet variants on leefbaarometer probes) but the `ring_agg/embeddings/` directory is currently empty. Need to regenerate.

Key files:
- `stage2_fusion/concat.py` — the concat pipeline, currently does NO normalization
- `stage2_fusion/models/ring_aggregation.py` — SimpleRingAggregator with 4 weighting schemes
- `scripts/stage3/run_probe_comparison.py` — consolidated probe runner with named configs
- Scratchpad: `.claude/scratchpad/stage2-fusion-architect/2026-03-15.md` — documents the variance problem

## Wave 0: Clean State
- `git status`, commit any dirty state
- Start `/loop 5m /sync` for lateral coordination with other terminals

## Wave 1: Add normalization to concat pipeline

1. **stage2-fusion-architect**: Modify `stage2_fusion/concat.py` to add per-modality z-score normalization (StandardScaler per modality block) BEFORE horizontal concatenation. The scaler should be fit on the non-zero rows of each block (so background/empty hexagons don't skew the mean/std). Save both `_raw.parquet` (unnormalized, for reference) and `.parquet` (normalized, as the new default). Log per-block mean/std so we can verify the normalization is working.

## Wave 2: Regenerate concat embeddings

2. **execution**: Run the updated concat pipeline: `python -m stage2_fusion.concat --study-area netherlands --modalities alphaearth,poi,roads --year 20mix`
   - Verify output: 4 modality blocks, each with mean≈0, std≈1
   - Check variance contribution is now roughly balanced across modalities

## Wave 3: Generate ring_agg embeddings

3. **stage2-fusion-architect**: Write or update a script to run SimpleRingAggregator on the normalized concat embeddings. Parameters: K=10, weighting=exponential (the configuration that beat UNet). Needs an SRAI H3Neighbourhood — check if one is cached or needs building.

4. **execution**: Run the ring_agg generation. Output to `data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet`.

## Wave 4: Probe comparison — measure normalization impact

5. **execution**: Run DNN probes on: (a) normalized concat 208D, (b) ring_agg k=10 embeddings. Compare R² against the old unnormalized results from `dnn_probe/2026-03-14/`.

6. **stage3-analyst**: Generate comparison plots (bar chart, spatial improvement map) between old-unnormalized and new-normalized.

## Wave 5: Verify + Commit

7. **qaqc**: Run `pytest tests/ -x`. Review probe results — did normalization improve R²?
8. **devops**: Commit in logical chunks.

## Final Wave: Close-out
- Coordinator scratchpad
- `/ego-check`

## Execution

```
/valuate
```
Then:
```
/niche .claude/plans/2026-03-15-stage2-normalize-and-ring-agg.md
```
