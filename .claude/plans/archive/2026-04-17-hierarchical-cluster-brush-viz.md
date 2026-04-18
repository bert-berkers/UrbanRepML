# Hierarchical Cluster-Brush Visualization

## Status: Ready

## Objective

Build a local, self-contained HTML app that lets the user *brush* MiniBatchKMeans
clusters with custom colors to discover semantic patterns in hierarchical H3
embeddings. Three H3 resolutions stacked isometric; click/drag a hex → global
recolor of that cluster label across all resolutions. Switchable colormap presets
and manual "focus depth" / opacity sliders.

## Motivation

The user's geographic knowledge is the real QA (per `memory/feedback_viz_investment.md`).
Brushing makes the user's interpretive eye the primary analysis channel: "I
brush this area red — oh, the whole urban spine lit up red, so cluster 7 is
'urban core'". Discovery by interaction, not by pre-assigned labels. Same idea
as published interactive cluster refinement (BMC Bioinformatics 2017, Clustergrammer,
RATH) but adapted to geospatial H3 hierarchical data.

## Cynefin Triage (per user request)

| Zone | Problem | Response |
|------|---------|----------|
| Simple | MBKMeans k=10, deck.gl CDN load, tab10 default, HTML `<input type=color>` | Apply heuristic — just execute |
| Complicated | Isometric stack geometry (pitch × elevation × layer spacing), brush-repaints-cluster-globally, hierarchical majority-vote aggregation, JSON payload budget | Domain care — needs analysis |
| Complex | Does brushing actually aid discovery? Attention-precision → opacity mapping naturalness? | Ship MVP, experiment via user testing, defer |
| Chaotic | (none) | — |

## Inputs & Infrastructure (Simple zone — just use)

- **Embeddings**: `data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet` (208D, z-scored at creation time)
- **Clustering**: `stage3_analysis/visualization/clustering_utils.py` — `apply_pca_reduction(n_components=32)` + `perform_minibatch_clustering([10])`
- **Spatial**: `h3.cell_to_parent(hex, parent_res)` for hierarchy traversal (allowed per `srai-spatial.md` — hierarchy-only, no polyfill/grid_disk)
- **Target resolutions**: res5 (~5k hexes), res6 (~36k), res7 (~250k) — res7 is finest visible, res9 is only for label generation
- **Rendering**: deck.gl 9.x via CDN (`https://unpkg.com/deck.gl@^9.0.0/dist.min.js`), H3HexagonLayer × 3 with `highPrecision: false`

## Output Structure

```
scripts/one_off/cluster_brush_viz/
├── build.py       # Reads parquet, clusters, aggregates, writes viz.html
├── viz.html       # Self-contained artifact — deck.gl via CDN + embedded JSON
└── README.md      # How to run + what to click
```

## Wave 1: Data pipeline (Simple — just execute)

**Agent**: `stage3-analyst`
**File**: `scripts/one_off/cluster_brush_viz/build.py` (NEW)

Steps:
1. Load `ring_agg/embeddings/netherlands_res9_20mix.parquet`
2. Drop non-feature columns; keep 208D feature block
3. `apply_pca_reduction(embeddings, n_components=32)` — ~95%+ variance retained
4. `perform_minibatch_clustering(reduced, [10], standardize=False)` → labels at res9
5. For each res9 hex, compute parents at res5, res6, res7 via `h3.cell_to_parent`
6. For each parent hex at each target resolution: majority-vote of children labels
   (use `scipy.stats.mode` or `collections.Counter.most_common(1)`)
7. Also record per-parent: count of children, label entropy (Shannon, nats) — used later for opacity in v2, just store for now
8. Emit a single `data.json` dict:
   ```json
   {
     "k": 10,
     "layers": {
       "5": [{"hex": "85...", "cluster": 3, "n_children": 49, "entropy": 0.82}, ...],
       "6": [...],
       "7": [...]
     }
   }
   ```
   Inline into `viz.html` as `const DATA = {...}`

**Acceptance criteria:**
- `viz.html` exists and is < 15 MB (JSON payload sanity check)
- All three layers present, each hex has a cluster label 0-9
- Opening `viz.html` in a browser shows something (even if ugly)

**Autonomous decisions (stage3-analyst):**
- PCA `n_components`: default 32, bump to 64 if variance < 0.9
- Dropping hexes with < 4 children at higher resolutions (low-confidence aggregates)
- Column drop logic for non-feature columns (inspect parquet schema)

## Wave 2: Rendering + brush (Complicated — needs care)

**Agent**: `coordinator` (general UI work, no specific domain agent fits)
**File**: `scripts/one_off/cluster_brush_viz/viz.html` (NEW)

Embedded JS implements:

1. **Scene setup**
   - `deck.gl` `Deck` instance attached to fullscreen `<div>`
   - `MapView` with `pitch: 55, bearing: -20` for isometric feel
   - Base coords centered on Netherlands (~lat 52.2, lon 5.3, zoom 7)
   - No basemap tile layer — pure hex rendering on dark background (embeddings speak for themselves)

2. **Three H3HexagonLayers, stacked**
   - One per resolution (5, 6, 7)
   - `extruded: true`, `getElevation: () => ELEVATION_OFFSETS[res]`
   - `getElevation` values: res5 → 8000m, res6 → 4000m, res7 → 0m (coarsest floats highest, overlapping stack effect)
   - `getFillColor: d => COLORMAP[d.cluster]` (reactive to state)
   - `highPrecision: false` for performance
   - `stroked: false` (borders clutter the brush feel)
   - `opacity: LAYER_OPACITY[res]` (reactive to attention depth slider)

3. **State**
   - `colorMap`: `{0: [r,g,b], 1: [r,g,b], ...}` initialized to `tab10` preset
   - `activeBrush`: `[r,g,b]` of currently selected brush color
   - `attentionDepth`: `0..2` float (which layer is in focus, 0=res7, 1=res6, 2=res5)
   - `focusSigma`: `0.5..3.0` float (Gaussian falloff width)
   - Layer opacity: `exp(-(layerIdx - attentionDepth)² / focusSigma²) * MAX_OPACITY`

4. **Brush interaction**
   - On hex click: `colorMap[d.cluster] = activeBrush` → trigger layer update (all three layers recolor globally)
   - On hex drag (mousedown + mousemove while button held): same as click, continuously updates
   - On hover: status bar shows `Cluster N — M hexes at res X`

5. **Controls panel (fixed right sidebar)**
   - **Brush palette**: 20 preset color swatches + `<input type="color">` for custom
   - **Colormap dropdown**: `tab10`, `pastel10`, `set3`, `paired`, `viridis(10)`, `plasma(10)`, `turbo(10)` — applying preset overwrites current `colorMap`
   - **Attention depth slider**: 0 to 2, float, live-updates layer opacity
   - **Focus σ slider**: 0.5 to 3.0
   - **Reset palette**: restore current preset
   - **Status bar**: hovering info, active brush color preview, current preset name

**Acceptance criteria:**
- Rendering is smooth at initial load (pan/zoom > 30 FPS)
- Clicking any hex recolors *all* hexes with the same cluster label across *all three* resolutions simultaneously (this is the discovery mechanic)
- Colormap preset swap completes in < 100ms
- Attention depth slider gives visible opacity gradient across layers

**Autonomous decisions (coordinator):**
- Specific pitch/bearing values — tune by eye
- Exact elevation offsets — whatever reads as "isometric stack"
- Colormap preset list — expand or shrink as looks good
- Drag-paint sensitivity (throttle interval)

## Wave 3: User exploration (Complex — experiment)

**Agent**: human user
**Artifact**: open `viz.html` in browser, play for 10-30 min

Open questions (user reports back):
- Does clicking-to-recolor *feel* like discovery, or does it feel like a toy?
- Is k=10 right? Too granular, too coarse?
- Is 3 resolutions enough, or should we add res8?
- Should brushing work per-resolution (brush the res6 layer, only res6 changes), or is global recolor the right default?
- What would make the attention-depth slider feel natural enough to auto-map to supra precision weights?

**Deliverable**: brief notes file at `reports/2026-04-17-cluster-brush-viz-notes.md` capturing the felt experience. This informs whether v2 work (attention-precision mapping, per-layer brush, etc.) is worth pursuing.

## Close-out wave

**Parallel agents:**
- **ego**: Process health assessment — did the Cynefin triage actually help? Were "complicated" decisions smoother than usual because we named them in advance?
- **coordinator**: Commit decision — `scripts/one_off/cluster_brush_viz/` as a `feat` commit; `viz.html` itself may be gitignored if JSON payload crosses 10 MB.

## Risks

1. **JSON payload too heavy**: res7 at ~250k hexes × ~40 bytes/record = ~10MB inline. Mitigation: if > 12MB, write `data.json` as sidecar file loaded via `fetch('./data.json')`.
2. **Deck.gl API changes between 9.x minor versions**: pin to a specific CDN version (`@9.1.x`).
3. **Brush feels janky on 250k hexes**: `onClick`/`onDrag` handlers need throttling. Falls into "Complicated — tune by experiment."
4. **Majority-vote at coarse resolutions loses nuance**: expected — that's why opacity + multi-layer view exist. Consider storing entropy per parent so v2 can show "aggregation confidence".

## File Summary

| File | Action | Wave |
|------|--------|------|
| `scripts/one_off/cluster_brush_viz/build.py` | CREATE | 1 |
| `scripts/one_off/cluster_brush_viz/viz.html` | CREATE | 2 |
| `scripts/one_off/cluster_brush_viz/README.md` | CREATE | 2 |
| `reports/2026-04-17-cluster-brush-viz-notes.md` | CREATE (post-play) | 3 |

## Execution

Invoke: `/niche .claude/plans/hierarchical-cluster-brush-viz.md`
