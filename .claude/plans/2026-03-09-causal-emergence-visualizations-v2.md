# Causal Emergence Visualizations v2: Proper CE Metrics + Publication Plots

**Created**: 2026-03-08 by coordinator (user-directed redesign)
**Depends on**: commit 09df3aa (Phase 1 probe results), native resolution probe results
**Supersedes**: `.claude/plans/2026-03-08-causal-emergence-visualizations.md` (Plot 4 redesigned, rest preserved)
**Suppress**: Stage 1 modalities, supra infrastructure, hex2vec, UNet++

## Goal

Create publication-quality visualizations of causal emergence results using **actual causal emergence metrics** (effective information, causal power), not R2 as proxy. The centerpiece is 6 vertical diamond profiles in the style of Rosas et al. / Hoel's causal emergence figures.

## Design Language

From the user's thesis (TIL-thesis-2024), the established leefbaarometer visual identity:

```python
COLORS = {
    'lbm': '#808080',   # Dark Grey -- Overall Liveability
    'vrz': '#FF4500',   # Orange Red -- Amenities
    'fys': '#32CD32',   # Lime Green -- Physical Environment
    'soc': '#8A2BE2',   # Blue Violet -- Social Cohesion
    'onv': '#1E90FF',   # Dodger Blue -- Safety
    'won': '#FFA500',   # Orange -- Housing Stock
}

TARGET_ORDER = ['lbm', 'vrz', 'fys', 'soc', 'onv', 'won']
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

- **`stage3_analysis/dnn_probe_viz.py`** -- `DNNProbeVisualizer`:
  - `_rasterize_centroids()` -- fast hex-to-raster via SpatialDB centroids (the rendering engine)
  - `plot_spatial_improvement()` -- RdBu diverging hex map with boundary overlay
  - Uses `StudyAreaPaths` for boundary GDF, handles Caribbean filtering, EPSG:28992 projection

- **`utils/spatial_db.py`** -- `SpatialDB.for_study_area()` for bulk hex geometry/centroid queries

All spatial maps MUST use `_rasterize_centroids` from `dnn_probe_viz.py`. Do NOT write a new rasterizer.

## Data Available

**Embeddings** (all in `data/study_areas/netherlands/stage2_multimodal/unet/embeddings/`):
- `netherlands_res9_2022.parquet` -- 128D embeddings at res9
- `netherlands_res8_2022.parquet` -- 128D embeddings at res8
- `netherlands_res7_2022.parquet` -- 128D embeddings at res7
- `netherlands_avg_2022.parquet` -- average of res7/8/9
- `netherlands_concat_2022.parquet` -- 384D concatenation of res7/8/9

**Targets** (in `data/study_areas/netherlands/target/leefbaarometer/`):
- `leefbaarometer_h3res9_2022.parquet` -- res9 target values (6 LBM dimensions)

**Probe results** (in `data/study_areas/netherlands/stage3_analysis/`):
- `native_resolution_probe_results.csv` -- R2 per resolution per target (from lollipop v1)
- `dnn_probe/2026-03-07_multiscale_res9_only/` -- OOF predictions
- `dnn_probe/2026-03-07_multiscale_multiscale_concat/` -- OOF predictions

## Wave 0: Clean State + Push

**Coordinator direct actions:**
- `git push` (4 unpushed commits -- 7th ego flag)
- Commit dirty tree items: `CLAUDE.md`, `scripts/plot_embeddings.py`, plan files, `reports/`, lollipop script
- Working tree triage for `.claude/scheduled_tasks.lock`, `.claude/skills/sync/SKILL.md`

## Wave 1: Causal Emergence Computation + Diamond Lollipop Plots (stage3-analyst)

### The Causal Emergence Framework

For each LBM target dimension, at each H3 resolution (res7, res8, res9):

1. **Build Transition Probability Matrix (TPM)**:
   - For each resolution's embeddings, find the K strongest correlations with the target
   - These correlation strengths are the "weights" in the transition matrix
   - Discretize embedding states (e.g., quantile binning of top-K correlated dimensions)
   - TPM: `P(target_bin | embedding_state)` -- how deterministically does the embedding predict the target?

2. **Compute Effective Information (EI)**:
   - `EI = log2(N) - mean_over_states(H(output | do(state)))` where N = number of states
   - This measures the causal power of the representation at this scale
   - Higher EI = more deterministic mapping from embedding to target = stronger causal power

3. **Compute Causal Emergence (CE)**:
   - `CE = EI(macro) - EI(micro)` for each scale pair
   - Positive CE at res8 means the neighbourhood scale has MORE causal power than the hexagon scale
   - This is the actual quantity from Hoel/Rosas, not an R2 proxy

### Plot: 6 Vertical Diamond Profiles ("Causal Emergence Lollipops")

**Reference**: `deepresearch/causalemergence-lollipop.png` (Rosas et al. figure)

Six subplots (2x3 or 3x2 grid), one per LBM dimension, each:
- **Vertical Y-axis** = H3 resolution (res7 at top, res9 at bottom) -- like dimensionality in the reference
- **Horizontal X-axis** = Effective Information (or delta-EI / causal power metric)
- **Dot at each resolution** -- size proportional to EI magnitude
- **Grey envelope polygon** connecting dots -- creates the diamond/kite shape
- **Colored in thesis color** for that LBM dimension
- **Clean, minimal** -- match the aesthetic of the reference figure

The diamond shape emerges naturally: if res8 has the highest EI (like vrz/amenities), the envelope bulges at the middle level, creating the characteristic kite. If EI decreases monotonically (like soc), the shape is a simple triangle.

This is the KEY insight visualization: vrz should show clear causal emergence (bulge at res8), while soc should show none (monotone decrease = no macro-scale advantage).

### Script

`scripts/stage3/plot_causal_emergence_lollipop.py` -- **replace** existing script entirely.

### Output

`reports/figures/causal-emergence/causal_emergence_diamonds.png` + `.pdf`

### Acceptance

- 6 diamond subplots, each with correct thesis color
- Uses actual EI/CE metrics, NOT R2
- Visual resemblance to the Rosas et al. reference figure
- Diamond/kite shape visible for vrz (causal emergence confirmed)
- White background, 300 DPI, tight layout
- Also save the raw CE quantities to CSV for the report

## Wave 2: Comparison Plots (stage3-analyst)

### Plot 1: Radar/Spider Chart -- "Scale Fingerprint"

One radar chart with 6 axes (one per LBM dimension). Three overlaid polygons: res9-only, avg, concat. Shows the SHAPE of improvement -- vrz expands dramatically while soc barely moves.

### Plot 2: Heatmap -- "Causal Scale Matrix"

Rows = embedding variants (res9, avg, concat), columns = LBM dimensions. Cell color = R2. Annotation with actual values. Orders columns by delta magnitude (vrz first, soc last).

### Plot 3: Bump Chart / Slope Chart -- "Fusion Progression"

X-axis: the 6 fusion methods in order (concat -> ring_agg -> GCN -> UNet-res9 -> UNet-avg -> UNet-concat). Y-axis: R2. One colored line per LBM dimension. Shows where each dimension's gains come from.

### Script + Output

`scripts/stage3/plot_causal_emergence.py` for Plots 1-3.
All saved to `reports/figures/causal-emergence/`.

### Acceptance

- 3 plots generated, all using thesis color scheme
- White backgrounds, 300 DPI, tight layout
- R2 data sourced from existing probe results (no new probe runs needed)

## Wave 3: Spatial Maps (stage3-analyst)

### Map 1: "Where Multi-Scale Matters" -- Prediction Improvement Map

**Reuse**: Adapt `dnn_probe_viz.py:plot_spatial_improvement()`. Compute |resid_res9| - |resid_concat| per hexagon. Blue = concat better, red = res9 better.

OOF predictions in `data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-07_multiscale_res9_only/` and `2026-03-07_multiscale_multiscale_concat/`.

### Map 2: Embedding Similarity Divergence

For each res9 hex, compute cosine similarity between its res9 embedding and its upsampled res8 parent embedding. Low similarity = scale disagreement = "interesting" hexagons.

**Reuse**: `_rasterize_centroids` from `dnn_probe_viz.py`, `SpatialDB` for centroids.

### Script + Output

`scripts/stage3/map_causal_emergence.py`
All saved to `reports/figures/causal-emergence/`.

### Acceptance

- Map 1 (prediction improvement) and Map 2 (embedding divergence) generated
- Maps use `_rasterize_centroids` -- no new rasterizer
- Netherlands extent, boundary overlay, EPSG:28992, white background, 300 DPI

## Wave 4: QAQC + Report

**Agent: qaqc** -- Visual quality review of all plots. Check: readable labels, no overlapping text, colorblind considerations, consistent style, diamond plots match reference aesthetic.

**Coordinator** -- Update or create `reports/2026-03-09-causal-emergence-visualizations.md` to embed figures with relative links.

## Wave 5: Commit

**Agent: devops** -- Commit scripts + figures + report.
- Message: `feat: causal emergence visualizations -- diamond profiles, radar, heatmap, slope, spatial maps`

## Final Wave: Close-out (mandatory)

- Update coordinator scratchpad
- `/librarian-update`
- `/ego-check`
- `git push`

## Execution

Invoke: `/coordinate .claude/plans/2026-03-09-causal-emergence-visualizations-v2.md`
