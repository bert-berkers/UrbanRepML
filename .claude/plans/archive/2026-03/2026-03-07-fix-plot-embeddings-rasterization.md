# Plan: Fix plot_embeddings.py Rasterization

## Context

`scripts/plot_embeddings.py` has been edited by 3+ parties (coordinator, stage3-analyst agent x2, user/linter) in a single session. The file is now 1245 lines with several features added (stamp, RD grid, CLI args, filter) but the rendering is broken — plots are either washed out or have water hex flooding. The file needs ONE coherent pass to fix the rendering pipeline end-to-end.

## Root Cause Analysis

The proven reference (`linear_probe_viz.py:_rasterize_centroids`, line 752) uses:
- `np.ones` (white opaque background, alpha=1)
- boundary drawn FIRST, then imshow at zorder=2
- The white image covers the boundary → but this is fine because probed data doesn't include water hexes

`plot_embeddings.py` faces a different challenge: POI has 868K hexes covering the FULL tessellation including water. So:
- `np.ones` → white rect covers boundary, AND water hexes get colored = background bleed
- `np.zeros` → transparent background, boundary shows through, water hexes still get colored but without the white flood

**Neither init alone fixes the water hex problem.** The filter is the real fix. The init just determines what non-data areas look like.

## Decision: np.zeros + robust filter

- **np.zeros**: transparent background so the grey boundary (#f0f0f0) shows through in non-data areas. This gives the Netherlands shape naturally.
- **Robust filter**: remove water/empty hexes BEFORE plotting so they don't get colored at all.
- This matches what worked visually in the session: the POI cluster k16 plot looked good with np.zeros + filter (the user confirmed "massive improvement" before noticing remaining bleed).

## Implementation

### Step 1: Verify np.zeros in all 4 rasterize functions
Lines ~194, ~236, ~273, ~316. All should be `np.zeros((height, width, 4), dtype=np.float32)`.
Update docstrings to say "transparent background" not "white background".

### Step 2: Fix the filter threshold
Current filter uses row-wise std with `p10/p50 < 0.8` bimodal detection. For POI hex2vec:
- p10=0.153, p50=0.240, ratio=0.64 → triggers, cutoff=(0.153+0.240)/2=0.196
- This filters 380K of 868K (43.8%)
- Remaining 488K includes legit rural hexes

But the user still saw bleed with 488K. The issue: some rural hexes with very few POIs produce embeddings that cluster with water. This is a DATA issue, not a rendering issue. The filter is doing its job — the remaining hexes have genuinely different embeddings.

**Approach**: Don't over-filter. The 488K set IS the real data. If plots still look noisy, that's honest representation of hex2vec quality at res9. Instead, add a `--filter-threshold` CLI arg (default 0.10) so the user can tune it.

### Step 3: Verify plot_spatial_map renders correctly with np.zeros
The rendering order is:
1. boundary_gdf.plot() → grey shape at zorder=1
2. ax.imshow(image) → RGBA overlay at zorder=2 (transparent pixels let boundary show)
3. _add_rd_grid() → grid lines at zorder=10

This should work. If alpha=0 pixels aren't transparent in imshow, we may need to explicitly handle this (e.g., set image dtype, or use masked arrays). Test and verify.

### Step 4: Run full suite and verify visually
```bash
# AlphaEarth (clean data, no filter needed — baseline test)
python scripts/plot_embeddings.py --study-area netherlands --resolution 9 --modality alphaearth

# POI hex2vec (the problematic modality)
python scripts/plot_embeddings.py --study-area netherlands --resolution 9 --modality poi --sub-embedder hex2vec

# Roads (sparse, 29% coverage)
python scripts/plot_embeddings.py --study-area netherlands --resolution 9 --modality roads
```

### Step 5: Commit
Stage only `scripts/plot_embeddings.py`. Other dirty files (deepresearch/, hooks/) are from other coordinators.

## Files to modify
- `scripts/plot_embeddings.py` — the only file. All changes within this file.

## Key functions to verify/fix
| Function | Line | Check |
|----------|------|-------|
| `rasterize_continuous` | ~147 | np.zeros, stamp param, docstring |
| `rasterize_rgb` | ~198 | np.zeros, stamp param, docstring |
| `rasterize_binary` | ~240 | np.zeros, stamp param, docstring |
| `rasterize_categorical` | ~277 | np.zeros, stamp param, docstring |
| `_stamp_pixels` | ~110 | Correct loop bounds, alpha setting |
| `_filter_empty_hexagons` | ~994 | Two-pass filter, threshold 0.8 ratio |
| `plot_spatial_map` | ~370 | Boundary at base, imshow zorder=2 |
| `_add_rd_grid` | ~325 | 50km step, raw RD values (not km) |
| `process_modality` | ~1068 | Uses emb_df_full for coverage, filtered for rest |

## Reference implementations
- `stage3_analysis/linear_probe_viz.py:_rasterize_centroids` (line 701) — proven rasterize with np.ones
- `stage3_analysis/linear_probe_viz.py:plot_spatial_residuals` (line 454) — proven boundary+imshow rendering

## Verification
1. AlphaEarth dim_grid: should show visible colored Netherlands shape on grey background
2. AlphaEarth clusters k16: should show distinct colored regions within Netherlands
3. POI hex2vec clusters k16: should show colored land areas, NO water flooding
4. POI hex2vec coverage: should show full 868K hex rectangle (unfiltered)
5. Roads coverage: should show sparse 29% coverage pattern
6. All plots: RD grid lines at 50km intervals with coordinate labels

## Execution
Invoke: `/coordinate .claude/plans/2026-03-07-fix-plot-embeddings-rasterization.md`
