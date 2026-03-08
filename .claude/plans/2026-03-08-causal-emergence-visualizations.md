# Causal Emergence Visualizations: Maps + Publication-Quality Plots

**Created**: 2026-03-08 by causal-emergence-coord
**Depends on**: commit 09df3aa (Phase 1 probe results)
**Suppress**: Stage 1 modalities, supra infrastructure, hex2vec

## Goal

Create publication-quality visualizations of the causal emergence results. Two categories:
1. **Comparison plots** — show R2 gains across variants and leefbaarometer dimensions
2. **Spatial maps** — show WHERE multi-scale embeddings differ geographically

## Design Language

From the user's thesis (TIL-thesis-2024), adopt the established leefbaarometer visual identity:

```python
COLORS = {
    'lbm': '#808080',   # Dark Grey — Overall Liveability (was 'afw' in thesis)
    'vrz': '#FF4500',   # Orange Red — Amenities
    'fys': '#32CD32',   # Lime Green — Physical Environment
    'soc': '#8A2BE2',   # Blue Violet — Social Cohesion
    'onv': '#1E90FF',   # Dodger Blue — Safety
    'won': '#FFA500',   # Orange — Housing Stock
}

TARGET_ORDER = ['lbm', 'vrz', 'fys', 'soc', 'onv', 'won']  # Overall first, then by magnitude of multi-scale gain

TARGET_NAMES = {
    'lbm': 'Overall Liveability',
    'vrz': 'Amenities',
    'fys': 'Physical Environment',
    'soc': 'Social Cohesion',
    'onv': 'Safety',
    'won': 'Housing Stock',
}
```

Style: `plt.style.use('default')`, white background, `facecolor='white'`, font 12/14, grid `axis='y', linestyle=':', alpha=0.3`, `dpi=300`.

## Reuse Infrastructure

**Critical**: Do NOT reinvent spatial rendering. The codebase has mature plotting infrastructure:

- **`stage3_analysis/dnn_probe_viz.py`** — `DNNProbeVisualizer`:
  - `_rasterize_centroids()` — fast hex-to-raster via SpatialDB centroids (the rendering engine)
  - `plot_spatial_improvement()` — RdBu diverging hex map with boundary overlay. Adapt for Map 1.
  - `plot_comparison_bars()` — grouped bar chart with delta labels
  - `TARGET_NAMES` dict already defined
  - Uses `StudyAreaPaths` for boundary GDF, handles Caribbean filtering, EPSG:28992 projection

- **`scripts/plot_embeddings.py`** — `plot_spatial_map()`, `plot_pca_rgb()`, stamp-based hex rasterizer

- **`scripts/plot_targets.py`** — 30+ leefbaarometer spatial plots, boundary handling patterns

- **`utils/spatial_db.py`** — `SpatialDB.for_study_area()` for bulk hex geometry/centroid queries

All spatial maps should use `_rasterize_centroids` from `dnn_probe_viz.py` (or extract it as a shared utility if needed). Do NOT write a new rasterizer.

## Wave 1: Comparison Plots (stage3-analyst)

### Plot 1: Radar/Spider Chart — "Scale Fingerprint"

One radar chart with 6 axes (one per lbm dimension). Three overlaid polygons: res9-only, avg, concat. Shows the SHAPE of improvement — vrz expands dramatically while soc barely moves. More visually striking than a bar chart for showing "which dimensions benefit from macro-scale."

### Plot 2: Heatmap — "Causal Scale Matrix"

Rows = embedding variants (res9, avg, concat), columns = lbm dimensions. Cell color = R2. Annotation with actual values. Orders columns by delta magnitude (vrz first, soc last). Clean seaborn heatmap with the thesis color scheme for column headers.

### Plot 3: Bump Chart / Slope Chart — "Fusion Progression"

X-axis: the 6 fusion methods in order (concat → ring_agg → GCN → UNet-res9 → UNet-avg → UNet-concat). Y-axis: R2. One colored line per lbm dimension. Shows where each dimension's gains come from — vrz climbs steadily with spatial methods then JUMPS at multi-scale, while soc flatlines after GCN. This is the "story of the research" in one plot.

### Plot 4: Causal Emergence Lollipop — "Scale Profile" (DONE)

**Already created**: `scripts/stage3/plot_causal_emergence_lollipop.py`
**Output**: `reports/figures/causal-emergence/scale_lollipop.png` + `.pdf`

Vertical lollipop inspired by Rosas et al. causal emergence paper (see `deepresearch/causalemergence-lollipop.png` for reference). Y-axis = H3 resolution (res7/res8/res9), X-axis = R² from native-resolution DNN probes. Colored dots per lbm dimension, sized by R², grey envelope polygon, thesis colors. Reveals that **vrz is the only dimension with true causal emergence** (res8=0.784 > res9=0.739).

Native-resolution probe data (new, cached at `data/study_areas/netherlands/stage3_analysis/native_resolution_probe_results.csv`):

| Target | res9 | res8 | res7 |
|--------|------|------|------|
| lbm | 0.263 | 0.244 | 0.235 |
| fys | 0.344 | 0.324 | 0.319 |
| onv | 0.506 | 0.467 | 0.468 |
| soc | 0.663 | 0.574 | 0.519 |
| vrz | 0.739 | **0.784** | 0.764 |
| won | 0.485 | 0.421 | 0.430 |

### Output

All plots saved to `reports/figures/causal-emergence/`.
Scripts: `scripts/stage3/plot_causal_emergence.py` (Plots 1-3), `scripts/stage3/plot_causal_emergence_lollipop.py` (Plot 4, done)

### Acceptance

- 4 plots generated, all using thesis color scheme
- All have white backgrounds, 300 DPI, tight layout
- radar + heatmap + slope + lollipop (lollipop already done)

## Wave 2: Spatial Maps (stage3-analyst)

### Map 1: "Where Multi-Scale Matters" — Prediction Improvement Map

**Reuse**: Adapt `dnn_probe_viz.py:plot_spatial_improvement()` (line 547). It already does exactly this pattern: computes |resid_A| - |resid_B|, maps to RdBu colormap, rasterizes via `_rasterize_centroids`, overlays boundary. Just swap "linear vs DNN" for "res9-only vs concat".

For each res9 hexagon with lbm targets:
- Compute |prediction_error_res9| - |prediction_error_concat| (positive = concat is better)
- Plot on hex map of Netherlands. Blue = concat much better, white = same, red = res9 better.
- This shows GEOGRAPHICALLY where macro-scale information helps.

OOF predictions already saved in `data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-07_multiscale_res9_only/` and `2026-03-07_multiscale_multiscale_concat/`.

### Map 2: "Scale Dominance" — Which Resolution Wins Where

For the concat variant, the DNN probe implicitly learns which of the 3 scale blocks (res9 dims 0-127, res8 dims 128-255, res7 dims 256-383) matters for each hex. We can approximate this by:
- Train 3 separate probes on each 128D block → get per-hex predictions
- Color each hex by which single-scale probe predicts it best
- Red = res9 wins (micro-dominated), green = res8 (meso), blue = res7 (macro)

This is more expensive (3 extra probe runs) but creates a stunning "scale landscape" map.

### Map 3: Embedding Similarity Divergence

For each res9 hex, compute cosine similarity between its res9 embedding and its upsampled res8 parent embedding. Low similarity = the micro and meso representations disagree about this location. These are the "interesting" hexagons where scale matters.

Plot as hex map. **Reuse**: `_rasterize_centroids` from `dnn_probe_viz.py` for rendering, `SpatialDB` for centroid lookups.

### Output

All maps saved to `reports/figures/causal-emergence/`.
Script: `scripts/stage3/map_causal_emergence.py`

### Acceptance

- At minimum: Map 1 (prediction improvement) and Map 3 (embedding divergence)
- Map 2 is bonus if time permits (requires 3 extra probe runs)
- Maps MUST use `_rasterize_centroids` from `dnn_probe_viz.py` (or extracted shared version) — no new rasterizer
- Netherlands extent, boundary overlay, EPSG:28992, white background, 300 DPI

## Wave 3: QAQC + Report Update

**Agent: qaqc** — Visual quality review of all plots. Check: readable labels, no overlapping text, colorblind-safe, consistent style.

**Coordinator** — Update `reports/2026-03-08-causal-emergence-phase1.md` to embed the new figures with relative links.

## Wave 4: Commit

**Agent: devops** — Commit scripts + report update.
- Message: `feat: causal emergence visualizations — radar, heatmap, slope, spatial maps`

## Final Wave: Close-out (mandatory)
- Update coordinator scratchpad
- `/librarian-update`
- `/ego-check`

## Execution

Invoke: `/coordinate .claude/plans/2026-03-08-causal-emergence-visualizations.md`
