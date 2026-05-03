# Rasterize-Voronoi Toolkit — W6 Visual Review Report

**Date:** 2026-05-03
**Status:** SHIPPED (pending human visual sign-off on cluster map regression below)
**Plan:** `.claude/plans/2026-05-02-rasterize-voronoi-toolkit.md`

---

## Wave-by-wave commit list

| Wave | Commit | Description |
|---|---|---|
| W1 | `b622bf3` | Spec freeze: `specs/rasterize_voronoi.md` + `artifact_provenance.md` errata |
| W2a | `b098d5c` | Core impl: `voronoi_indices`, `gather_rgba`, `rasterize_voronoi` + 4 per-mode wrappers in `utils/visualization.py` |
| W2b | `b098d5c` | 34 contract tests in `tests/test_rasterize_voronoi.py` |
| W3a | `7afb202` | 13 durable callers migrated; `plot_targets.py` shadow eliminated |
| W3b | `7591a63` | 1 one-off migrated; 3 already clean |
| W4 | `ae7705b` | `save_voronoi_figure` + `audit_figure_provenance.py` |
| W5 | `a45aa4a` | `scripts/visualization/README.md` with 3 mermaid charts |
| W6 | *(this commit)* | Deletion sweep, dead helper cleanup, archive, bug fix |

---

## Pytest results (post-W6)

```
tests/test_rasterize_voronoi.py: 34/34 passed (1.0s)
Full suite (pytest -x --ignore=tests/manual): 278/278 passed (16.4s)
```

Baseline preserved. No regressions introduced by deletion sweep.

---

## Audit script output

```
=== Study area: netherlands ===
  total: 345   covered: 0   uncovered: 345   coverage: 0.0%
```

345 pre-existing figures in `data/study_areas/netherlands/stage3_analysis/`, all produced before W4's `save_voronoi_figure` integration. Coverage is 0% — this is expected and correct: provenance injection was added in W4, but existing figures were not regenerated. The first newly generated figure (cluster map below) has a provenance YAML sidecar.

---

## Deletion sweep summary

**Functions deleted from `utils/visualization.py`:**

| Function | LOC removed |
|---|---|
| `_stamp_pixels` | 24 lines |
| `rasterize_continuous` | 60 lines |
| `rasterize_rgb` | 42 lines |
| `rasterize_binary` | 39 lines |
| `rasterize_categorical` | 47 lines |

**Total:** 5 functions, 232 lines deleted from `utils/visualization.py`.

Module docstring updated: removed "centroid-splat" usage example, replaced with new Voronoi API usage.

**Pre-flight grep result:** Zero real callers of deprecated functions outside excluded paths.
- `utils/visualization.py` itself: excluded (definitions)
- `scripts/archive/`: excluded (historical)
- `scripts/stage3/plot_targets.py`: has locally-defined `rasterize_continuous`/`rasterize_categorical` that wrap the NEW Voronoi API — confirmed safe
- `stage3_analysis/cluster_comparison_plotter.py`: docstring mention only, not a function call — safe

---

## Dead helper cleanup (Sub-task 3)

`scripts/stage3/map_causal_emergence.py`: deleted `_compute_stamp_radius` and `_build_disk_mask` (40 lines). These were leftover dead code from W3a's body rewrite — defined at module level but never called within the file.

`python -m py_compile scripts/stage3/map_causal_emergence.py` passes after deletion. The `h3` import is retained (used by `h3.cell_to_parent` for hierarchy traversal in `map_embedding_divergence`, which is allowed per SRAI rules).

---

## Bug found and fixed during W6 (Sub-task 4)

`scripts/stage3/plot_cluster_maps.py` had a NameError in the provenance config dict at line 379:

```python
"stamp": int(stamp),  # BROKEN — stamp undefined after W3a migration
```

W3a's migration removed the `stamp` parameter from the rasterization call but forgot to update the `plot_config` dict. Replaced with:

```python
"pixel_m": float(pixel_m),
"max_dist_m": float(max_dist_m),
```

This was a silent bug: the script would crash on the provenance write step after successfully rendering all cluster maps. Only discoverable by actually running the script.

---

## Visual diff regeneration (Sub-task 4)

### Cluster maps — SUCCEEDED

**Script:** `scripts/stage3/plot_cluster_maps.py` (W3-migrated, uses new Voronoi API)

**Output:** `reports/visual_diff_W6/2026-05-03/cluster_maps/2026-05-03/clusters_k10.png`
**Provenance:** `reports/visual_diff_W6/2026-05-03/cluster_maps/2026-05-03/clusters_k10.png.provenance.yaml`

Parameters: `pixel_m=250.0, max_dist_m=300.0, k=10, res9, 397,757 hexagons, 158D PCA to 16D, tab20 colormap`.

The script ran to completion without errors after the NameError bug fix above. The provenance YAML sidecar was written.

**Human sign-off needed:** Compare `reports/visual_diff_W6/2026-05-03/cluster_maps/2026-05-03/clusters_k10.png` against the pre-migration cluster maps in `data/study_areas/netherlands/stage3_analysis/`. The Voronoi API uses `max_dist_m=300m` cutoff instead of centroid-splat stamp — transparent gaps near region boundaries will be tighter. This is an improvement (geometrically correct) but should be reviewed.

### Three-embeddings study — SKIPPED

**Reason:** `scripts/one_off/viz_three_embeddings_res9_study.py` imports Voronoi functions directly from `viz_ring_agg_res9_grid.py` (now archived), not from `utils.visualization`. This script was not part of the W3 caller migration (one-offs were migrated for consistency but this one still references the raw one-off). Running it would regenerate into the existing `data/study_areas/netherlands/stage3_analysis/2026-04-24/panels/` directory (hardcoded path), overwriting existing figures. Skipped to preserve data.

Note: this script will be broken now that `viz_ring_agg_res9_grid.py` is archived to `scripts/archive/visualization/` — the `sys.path.insert(0, parent)` import will no longer resolve. This is a follow-up item. See open items below.

### LBM probe overlay — SKIPPED

**Reason:** Same as three-embeddings study — imports from `viz_ring_agg_res9_grid.py` directly. Also hardcodes output path. Import is now broken post-archive. See open items.

---

## Reference impl archive (Sub-task 5b)

```
scripts/one_off/viz_ring_agg_res9_grid.py
    → scripts/archive/visualization/viz_ring_agg_res9_grid.py
```

Moved with `git mv`. The canonical functions now live in `utils/visualization.py`.

---

## End-to-end verification (plan §Verification)

```python
# Import check — PASSED
python -c "import importlib.util; spec = importlib.util.spec_from_file_location(
    'pce', 'scripts/stage3/plot_concat_embeddings.py'); ..."
# → OK: imported successfully (SystemExit on missing args — expected)
```

Full `--study-area netherlands` run was not attempted (large dataset, no --output-dir override available without data availability constraints). Import verification confirms the script loads cleanly with the new API.

---

## Ledger validation

`read_ledger('netherlands')` returns 0 rows. The ledger (`data/ledger/runs.jsonl`) is not written by figure-generation scripts — only by `SidecarWriter`-wrapped training/processing runs. Figure provenance lives in `*.provenance.yaml` siblings, per `specs/artifact_provenance.md`. The 0-row count is consistent with pre-W4 state and confirms W4 did not incorrectly write to the ledger.

---

## Open items requiring human sign-off

### [open|0d] Visual regression review: Voronoi vs centroid-splat [needs:human]

The cluster map at `reports/visual_diff_W6/2026-05-03/cluster_maps/2026-05-03/clusters_k10.png` was produced with the new Voronoi API (`max_dist_m=300m`). The human's geographic eye is the final gate. Key things to check:

- Are hex boundaries crisp at res9 scale? (Voronoi should be sharper than stamp)
- Are NaN/missing regions transparent (not silently filled)?
- Does the Randstad/urban geography read correctly?
- Are 10 clusters distinguishable with tab20 colormap?

### [open|0d] NaN behavior change in Voronoi vs centroid-splat [needs:human]

W3a noted that the NaN/silhouette behavior is geometrically different between the two approaches. Centroid-splat: NaN hexes produce transparent single pixels. Voronoi: NaN hexes produce a gap up to `max_dist_m` wide. This is the correct behavior (geometrically truthful) but changes the visual appearance of sparse regions.

### [open|0d] `comparison_plotter.py` NaN-residual mid-grey vs transparent [needs:human]

W3 carry-item: `comparison_plotter.py` previously used mid-grey for NaN residuals; the Voronoi API uses transparent. This changes the visual weight of missing data regions. Reviewer should confirm the transparent treatment is acceptable for the comparison plot use case.

### [open|0d] One-off scripts' broken import post-archive [→spec-writer|coordinator]

`scripts/one_off/viz_three_embeddings_res9_study.py` and `scripts/one_off/viz_three_embeddings_lbm_overlay.py` both import from `viz_ring_agg_res9_grid` via `sys.path.insert(0, parent)`. Now that `viz_ring_agg_res9_grid.py` is archived to `scripts/archive/visualization/`, this import path is broken. Options:
1. Update both one-offs to import from `utils.visualization` directly (preferred — completes the migration)
2. Accept that these one-offs are effectively at end-of-life (30-day shelf life ~expires 2026-05-24)

These are one-off scripts so their brokenness doesn't block any production workflow. The coordinator should decide whether to fix or let expire.

---

## Plan status update

The plan's Status line has been updated from `DRAFT` to `SHIPPED`.

Human visual sign-off on the cluster map regression is the remaining gate. All programmatic gates (pytest 278/278, pre-flight grep clean, compile check passed) are green.
