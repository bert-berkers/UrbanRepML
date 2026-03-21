# Plan: Cleanup — probe/cluster paths, colorbar bug, scratchpad migration

## Context

Post-ship cleanup from the probe/cluster pipeline OODA loop. Coral flagged two issues:
probe_results/ is in the wrong location (should be under stage3_analysis/) and the
comparison plotter has a colorbar overlap bug. Additionally, today's scratchpad files
use the old naming convention (pre session-keyed protocol fix).

## Issues

| # | Issue | Severity | Owner |
|---|-------|----------|-------|
| 1 | `probe_results/` path should be `stage3_analysis/probe_results/` | Breaking — wrong convention | jade |
| 2 | `cluster_results/` path should be `stage3_analysis/cluster_results/` | Same fix | jade |
| 3 | Colorbar overlaps figure in residual maps (UNet panel) | Visual bug | jade |
| 4 | Today's scratchpad files are old-style `2026-03-21.md` not `2026-03-21-{session}.md` | Protocol debt | jade |
| 5 | `run_probe_comparison.py` doesn't call ProbeResultsWriter | Integration gap | defer |
| 6 | Coral's data at wrong path needs moving | Data migration | coral (after our fix) |

## Wave Structure

### Wave 1 — Fix paths + colorbar (parallel, 2 agents)

| # | Agent | Task | Files |
|---|-------|------|-------|
| 1a | stage3-analyst | Fix `paths.py`: change `probe_results_root()` and `cluster_results_root()` to return `self.root / "stage3_analysis" / "probe_results"` and `self.root / "stage3_analysis" / "cluster_results"` respectively. Also update any path references in `probe_results_writer.py`, `cluster_results_writer.py`, `comparison_plotter.py`, `cluster_comparison_plotter.py` if they construct paths independently of StudyAreaPaths. | `utils/paths.py`, all 4 writer/plotter files |
| 1b | stage3-analyst | Fix comparison_plotter.py: (1) Fix colorbar overlap in `plot_residual_maps()` — use `constrained_layout=True` or explicit colorbar axes. Reference `linear_probe_viz.py:454-635`. (2) Expand `plot_scatter()` to full grid: rows=6 targets (lbm,fys,onv,soc,vrz,won), cols=approaches. (3) Remove area_gdf boundary outline from residual maps — just painted hex blobs on black/white background, no boundary polygon overlay. Pass `boundary_gdf=None` or skip `plot_spatial_map` boundary rendering. | `stage3_analysis/comparison_plotter.py` |

**__init__.py ownership**: Agent 1a (if any __init__ changes needed).

| 1c | stage3-analyst | Wire writers into experiment scripts: (1) Add `--write-standardized` flag to `scripts/stage3/run_probe_comparison.py` — after each probe run, call `ProbeResultsWriter.write_from_regressor(regressor, approach_slug, study_area)`. Capture the regressor object instead of discarding it. (2) Wire ClusterResultsWriter into `scripts/stage3/plot_cluster_maps.py` or wherever MiniBatchKMeans assignments are produced — after clustering, call `ClusterResultsWriter.write_from_clustering()`. | `scripts/stage3/run_probe_comparison.py`, `scripts/stage3/plot_cluster_maps.py` |

### SYNC 1

> Go/no-go: paths resolve correctly? Colorbar renders cleanly?

### Wave 2 — QAQC (1 agent)

| # | Agent | Task |
|---|-------|------|
| 2 | qaqc | Verify path changes: import check, confirm `probe_results_root()` returns `stage3_analysis/probe_results`, no hardcoded paths. Visual check on colorbar if possible. |

### SYNC 2

> Committable?

### Wave 3 — Commit + notify (coordinator direct)

- Commit fixes
- Notify coral: paths updated, move your data from `probe_results/` to `stage3_analysis/probe_results/`

### Wave 4 — Migrate scratchpad files (coordinator direct)

Rename today's scratchpad files from old to new convention. **Coral's warning**: some files
(especially coordinator/) have entries from multiple terminals — those must be SPLIT, not renamed.

**Safe to rename** (jade-only agents):
```
stage3-analyst/2026-03-21.md → stage3-analyst/2026-03-21-jade-falling-wind.md
qaqc/2026-03-21.md → qaqc/2026-03-21-jade-falling-wind.md
Plan/2026-03-21.md → Plan/2026-03-21-jade-falling-wind.md
```

**Must check content first** (may have multi-terminal entries):
```
coordinator/2026-03-21.md — READ first. Split jade entries → jade file, coral entries → coral file.
coordinator/2026-03-21-forward-look.md — check authorship before renaming.
ego/2026-03-21.md — check if multiple sessions wrote to it.
librarian/2026-03-21.md — check if multiple sessions wrote to it.
Explore/2026-03-21.md — check if multiple sessions wrote to it.
```

### Final Wave — Close-out (mandatory)

1. Coordinator scratchpad (session-keyed this time!)
2. `/librarian-update`
3. `/ego-check`

## Deferred

- Coral's data migration from old `probe_results/` to new `stage3_analysis/probe_results/` (after we commit)

## Execution
Invoke: `/niche .claude/plans/2026-03-21-cleanup-probe-cluster-paths.md`
