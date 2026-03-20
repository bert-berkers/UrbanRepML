# Clean House Part 2 — Deferred Items, Bug Fixes, Plot Rerun

**Intent**: Finish what the first clean-house session deferred. Fix chronic bugs, organize data folder, rerun plots to verify all fixes produce good output.

**Session**: silver-watching-shore (same valuation as part 1)

## Wave 0: Clean State
- `git status`, commit any dirty state
- Read coordinator scratchpad `2026-03-15.md` for context from part 1

## Wave 1: Quick Fixes (parallel)

1. **devops**: Fix `group_columns_by_modality()` bug in `scripts/stage3/plot_concat_embeddings.py`. The single-letter prefix matching misses `gtfs2vec_` columns. Use regex or prefix-list matching instead. This has been deferred 6+ sessions — trivial fix. [carry: 6]

2. **devops**: Deduplicate GTFS dominant-vector filter. It's currently in both `scripts/plot_embeddings.py` and `scripts/stage3/plot_cluster_maps.py`. Move the logic into `utils/visualization.py` (where the other shared helpers live) and have both scripts import it.

3. **devops**: Fix `probe_20mix_multiscale.py` — replace 8 hardcoded `data/study_areas/...` paths with `StudyAreaPaths`. If the duplicated training loop can use `DNNProbeRegressor`, refactor it. If not, archive to `scripts/archive/probes/` with a note explaining why it's special.

4. **qaqc**: Fix two chronic QAQC carry items. User explicitly said "FIX, not WONTFIX":
   - `spatial_db` CRS test — srai-spatial confirmed SedonaDB is live, so the test should be straightforward
   - AE geometry leakage — check what this refers to in prior qaqc scratchpads, then fix

## Wave 2: Data Folder Organization (needs human check-in)

5. **devops**: Organize loose output files in `data/study_areas/netherlands/` into per-day subdirs where they aren't already. Rules:
   - **NO deletes, NO archiving** — data results are sacred
   - Sort loose CSVs/JSONs into `YYYY-MM-DD/` subdirs based on file modification date
   - Stage 3 `dnn_probe/` already has dated run dirs — leave those alone
   - Stage 1/2 flat plot dirs — add date subdirs if there are multiple generations of plots
   - Check in with coordinator (who checks with human) before moving anything ambiguous
   - Update `StudyAreaPaths` if any canonical paths change

## Wave 3: Rerun Plots (parallel — verify all fixes produce good output)

6. **execution**: Rerun GTFS cluster plot with `--filter-empty` to verify dominant-vector detection works: `python scripts/plot_embeddings.py --study-area netherlands --modality gtfs --filter-empty`

7. **execution**: Rerun concat embeddings plot to verify `group_columns_by_modality()` fix: `python scripts/stage3/plot_concat_embeddings.py --study-area netherlands`

8. **execution**: Rerun cluster maps for ring_agg (the current best performer): `python scripts/stage3/plot_cluster_maps.py --study-area netherlands --embedding-path data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet --filter-empty`

9. **qaqc**: Review the generated plots. Check: correct extents, no 0-padded hexagons, clean colormaps, proper labels. Compare against known-bad plots from earlier sessions if available.

## Wave 4: Commit + Verify

10. **devops**: Commit all changes in logical chunks.

11. **qaqc**: Final test run — `python -m pytest tests/ -x` to confirm nothing broke.

## Final Wave: Close-out
- Write coordinator scratchpad
- `/ego-check`

## Execution
Invoke: `/niche .claude/plans/2026-03-15-clean-house-pt2.md`
