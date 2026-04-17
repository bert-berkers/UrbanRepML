# Hierarchical Cluster-Brush Visualization — v2 (Raster Rewrite)

## Status: Ready

## Objective

Rewrite the hierarchical cluster-brush viz as a **rasterised PNG stack** with
browser-side canvas colorization, matching the aesthetic of
`scripts/stage3/plot_cluster_maps.py` and `scripts/plot_embeddings.py`.
Three resolutions (res5 / res6 / res7) render as label-encoded PNGs in Python,
stack in the browser via CSS 3D transforms, and recolor instantly when the user
clicks a hex — a palette-swap in canvas, not a re-render.

## Motivation — what's different from v1, what carries over

v1 (commit `a2f051f`) shipped a deck.gl `H3HexagonLayer` stack with global
cluster repaint on click. It works as a proof of concept but fails on two axes:

1. **Broken extrusion**. deck.gl 9.1.9's `H3HexagonLayer.extruded=true` path
   drops into a `ColumnLayer` sub-layer that tries to resolve h3-js
   `cellToLatLng` from `window` globals. The globals aren't wired when deck.gl
   loads via CDN, so the stack renders flat. Multiple workarounds attempted,
   none clean. We could fix it upstream; we shouldn't.
2. **Wrong aesthetic**. The rest of UrbanRepML uses rasterised hexagon maps
   (`utils/visualization.rasterize_categorical` → matplotlib PNG). The user
   explicitly wants the same look here. Vector hex outlines are also unwanted;
   the user wants fill-only slabs.

**Carries over from v1** (reuse verbatim or near-verbatim):
- Data pipeline: `ring_agg` res9 parquet → `apply_pca_reduction(n_components=32)` →
  `perform_minibatch_clustering([10])` → per-hex label
- Hierarchical aggregation: majority-vote via `h3.cell_to_parent` (hierarchy-only
  h3-py use is allowed per `.claude/rules/srai-spatial.md`)
- Entropy per parent hex, colormap presets, 20-swatch brush palette, color picker
- v1 artifacts remain on disk — `build.py`, `viz.html`, v1 `README.md` are NOT
  deleted. v2 lives alongside as `build_raster.py`, `viz_raster.html`, and the
  `README.md` is updated in place to cover both (with v2 marked recommended).

**Changes from v1**:
- Rendering: matplotlib rasters, not deck.gl vectors
- Stacking: CSS 3D transforms on three `<canvas>` elements, not deck.gl elevation
- Vertical separation: "cathedral" — 3+ layer-heights of gap between slabs
- Rotation: button cycling through 4 isometric angles (NW/NE/SE/SW)
- No outlines: `rasterize_categorical` already does fill-only, so this is free
- Brush mechanism: **label-encoded PNG + canvas palette swap**, not layer reactive
  prop updates

## Cynefin Triage

| Zone | Problem | Response |
|------|---------|----------|
| Simple | Reuse v1 data pipeline; matplotlib raster via `utils.visualization.rasterize_categorical` with `cmap` swapped for identity-label palette; CSS 3D `perspective` + `translateZ` for stacking | Apply heuristic — just execute |
| Complicated | Label-encoded PNG scheme (label in RGB channels + alpha=0 outside extent); browser canvas palette-map loop performance on 3 × ~2000×2400 images; click-to-cluster pixel lookup → global repaint across all 3 canvases; 4-angle rotation coordinated across canvases without re-layout jitter | Domain care — needs analysis |
| Complex | Does rasterised + canvas-colorize feel as responsive as vector deck.gl did? Only user testing answers. Best UX for rotate button: snap vs animate, arrow-key support? k=10 still right at cathedral spacing? | Ship MVP, test, defer |
| Chaotic | (none) | — |

## Inputs & Infrastructure (Simple zone)

- **Embeddings**: `data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet`
  (208D, z-scored at creation; 2026-03-09 hex2vec-integrated late-fusion)
- **Clustering**: `stage3_analysis.visualization.clustering_utils.apply_pca_reduction` +
  `perform_minibatch_clustering` — identical invocation to v1
- **Hierarchy**: `h3.cell_to_parent(hex, parent_res)` for hierarchy traversal
- **Centroids**: `utils.spatial_db.SpatialDB.for_study_area("netherlands").centroids(region_ids, resolution=R, crs=28992)`
  — vectorized bulk query, matches `plot_cluster_maps.py` style
- **Extent**: from `utils.visualization.load_boundary(paths)` (Netherlands boundary)
  or `SpatialDB.extent(...)` as fallback
- **Rasterizer**: `utils.visualization.rasterize_categorical(cx, cy, labels, extent, n_clusters=10, cmap=..., stamp=...)`
  — produces a `uint8` RGBA image. See note below for the identity-palette trick.
- **Target resolutions**: res5 (~5k parents), res6 (~36k), res7 (~250k) — res9 only
  for label generation. Per `memory/feedback_raster_stamp_size.md`, use full
  2000×2400 per panel; stamp scales with resolution (`stamp = max(1, 11 - res)`)

### Label-encoded PNG trick (Complicated zone, decided)

`rasterize_categorical` uses a matplotlib colormap and `stamp` splats to paint
category IDs into an RGBA image. For v2, **we do NOT render finished colors to
disk**. Instead:

- Build a custom `ListedColormap` with 10 entries where entry `i` maps to
  `(i, i, i)` — i.e., a grayscale identity palette: cluster 0 → `(0,0,0)`,
  cluster 1 → `(1,1,1)`, …, cluster 9 → `(9,9,9)`. The PNG becomes a label
  map: R channel encodes the cluster ID (0..9).
- Alpha channel: `rasterize_categorical` already writes `alpha=0` for pixels
  with no hex underneath. Keep that — the browser uses alpha to skip
  no-data pixels during colorize.
- Final PNG is an indexed label map at the same 2000×2400 matplotlib shape,
  written to `scripts/one_off/cluster_brush_viz/labels_res{5,6,7}.png`.

Browser side: `drawImage` → `getImageData` → iterate pixels → if alpha>0, map
`R` through current palette → `putImageData`. One forward pass per layer per
repaint. For 2000×2400 × 3 layers = ~14.4M pixels, this is ~50–100 ms budget
on modern hardware — fast enough for interactive brush.

## Output Structure

```
scripts/one_off/cluster_brush_viz/
├── build.py              # v1 (deck.gl pipeline) — KEEP, do not modify
├── viz.html              # v1 (deck.gl renderer) — KEEP, do not modify
├── build_raster.py       # v2 (NEW) — reuses v1 clustering, emits PNGs + labels.json
├── viz_raster.html       # v2 (NEW) — canvas stack + brush
├── labels_res5.png       # v2 output (label-encoded PNG, ~small)
├── labels_res6.png       # v2 output
├── labels_res7.png       # v2 output
├── labels.json           # v2 sidecar: extent, counts per cluster per resolution
└── README.md             # UPDATED in place — documents v1 and v2, v2 marked recommended
```

## Wave 1: Data pipeline + PNG rendering (Simple + one Complicated bit)

**Agent**: `stage3-analyst` (continuity with v1 — same agent wrote both v1 waves,
preserves builder↔viz data contract knowledge. Scratchpad lineage already
exists. No reason to switch.)

**Files**: `scripts/one_off/cluster_brush_viz/build_raster.py` (NEW)

Steps:
1. Copy the clustering block from v1 `build.py` verbatim (load parquet → drop
   non-numeric → PCA(32, fallback 64 if variance < 0.9) → MBKMeans(k=10)). This
   is proven; don't re-derive.
2. Hierarchical aggregation: same `_aggregate_to_parent_res` as v1 — majority
   vote + entropy + n_children per parent hex. Keep entropy output for future v3
   opacity use (not consumed by v2, but cheap to compute).
3. For each target resolution r in [5, 6, 7]:
   - Build `labels_df` with `region_id` index and `cluster` column
   - `SpatialDB.for_study_area("netherlands").centroids(labels_df.index, resolution=r, crs=28992)`
     → `cx, cy` numpy arrays
   - Load boundary via `utils.visualization.load_boundary(paths)` once; reuse its
     `.total_bounds` for consistent `extent` across all 3 resolutions (same
     extent is required for CSS stacking to align)
   - `stamp = max(1, 11 - r)` — res5 → 6, res6 → 5, res7 → 4 (consistent with
     `plot_cluster_maps.py`)
   - Build identity `ListedColormap` for label encoding (see trick above)
   - Call `rasterize_categorical(cx, cy, labels, extent, n_clusters=10, cmap=identity, stamp=stamp)`
   - Save the returned RGBA array directly as PNG via PIL or matplotlib
     `imsave` — **do NOT wrap in a full `plot_spatial_map` figure**. We want the
     raw pixel grid with no axes/margins/boundary overlay. The boundary can be
     drawn as a separate transparent PNG in v3 if the user requests it.
4. Write sidecar `labels.json`:
   ```json
   {
     "k": 10,
     "extent": [minx, miny, maxx, maxy],
     "raster_shape": [height, width],
     "resolutions": [5, 6, 7],
     "pixel_cluster_counts": {
       "5": [n_pixels_cluster_0, ..., n_pixels_cluster_9],
       "6": [...],
       "7": [...]
     },
     "source": {"study_area": "netherlands", "model": "ring_agg", "resolution": 9, "year": "20mix",
                "pca_components": 32, "pca_variance_retained": 0.xxx}
   }
   ```
   Tiny (< 2 KB). The HTML fetches it on load.
5. Use `StudyAreaPaths` for the input parquet — no hardcoded `data/study_areas/...`
   strings (per `.claude/rules/script-discipline.md` and `data-code-separation.md`).
6. Module docstring: `Lifetime: temporary`, `Stage: 3`, expiry ~30 days after
   creation (per one-off script rules).

**Acceptance criteria:**
- Three PNGs exist (`labels_res5.png`, `labels_res6.png`, `labels_res7.png`),
  each 2000×2400, RGBA
- Opening one in an image viewer shows near-black (labels 0–9 are nearly
  indistinguishable grayscale) with alpha=0 outside the NL boundary — this is
  correct, it's a label map, not a visualization
- Sanity check: `np.unique(img[:, :, 0][img[:, :, 3] > 0])` at each resolution
  should return a subset of `[0..9]`
- `labels.json` ≤ 2 KB, contains all expected keys
- All three raster extents are identical (required for CSS alignment)

**Autonomous decisions (stage3-analyst):**
- Exact PNG writer (PIL `Image.fromarray(arr).save(...)` vs
  `matplotlib.image.imsave`) — both fine, pick one, stay consistent
- Where to put identity colormap construction — inline in `build_raster.py`
- Whether to also write a preview finished-color PNG for sanity (helpful but not
  required; if added, put in `debug_preview_res{r}.png` alongside)
- PCA fallback logic — same as v1, keep the variance floor check

## Wave 2: HTML + canvas + 3D stack + brush + rotate (Complicated)

**Agent**: `stage3-analyst` (continuity — per user constraint, HTML/JS + data
work stays with the agent that owns the cluster pipeline; not
`general-purpose`, never `general-purpose`)

**Files**:
- `scripts/one_off/cluster_brush_viz/viz_raster.html` (NEW)
- `scripts/one_off/cluster_brush_viz/README.md` (UPDATE in place)

Embedded JS implements:

### 1. Scene setup

- Three `<canvas>` elements (`#layer-res5`, `#layer-res6`, `#layer-res7`),
  absolutely positioned, same pixel dimensions (2000×2400 but displayed
  downscaled via CSS to fit viewport while maintaining aspect ratio)
- Wrapping `<div id="stage">` with CSS `perspective: 2500px` (or larger if
  cathedral effect needs more depth)
- Each canvas wrapped in its own `<div class="slab">` with `transform: translateZ(...)`
  — **cathedral spacing**: gaps equal to 3+ canvas-heights of Z distance between
  slabs. Concretely: res7 at `translateZ(0)`, res6 at `translateZ(1200px)`,
  res5 at `translateZ(2400px)` (rough guidance; tune to taste). The coarsest
  resolution floats highest, as in v1.
- Background: dark (`#111`) to match stage3 rasters and make isometric depth
  legible

### 2. Load + render

```
async function boot() {
  const meta = await fetch('./labels.json').then(r => r.json());
  const imgs = await Promise.all([5, 6, 7].map(r => loadImage(`./labels_res${r}.png`)));
  // For each image: drawImage to its canvas, getImageData, store as Uint8ClampedArray
  // on window.LAYER_DATA[r] alongside {width, height, imageData}
  render();
}
```

Store per-layer `ImageData` once; never re-fetch the PNG.

### 3. Colorize pass (the hot loop)

```
function repaint(r) {
  const L = LAYER_DATA[r];
  const out = new Uint8ClampedArray(L.imageData.data.length);
  const palette = state.colorMap; // {0: [r,g,b], 1: [r,g,b], ...}
  const src = L.imageData.data;
  for (let i = 0; i < src.length; i += 4) {
    if (src[i + 3] === 0) continue; // alpha=0 → no-data, skip (out is already 0)
    const label = src[i];
    const c = palette[label];
    out[i] = c[0]; out[i + 1] = c[1]; out[i + 2] = c[2]; out[i + 3] = 255;
  }
  L.ctx.putImageData(new ImageData(out, L.width, L.height), 0, 0);
}
```

Tight tight loop — no function calls in the inner path. ~50–100 ms per layer
on ~4.8M pixels. Call `repaint(5), repaint(6), repaint(7)` after any brush
click or palette swap.

### 4. Brush interaction (pixel-lookup → cluster → global repaint)

- On `click` / `mousedown+mousemove` on any canvas:
  - Convert mouse (clientX, clientY) to canvas pixel (accounting for CSS
    display size scaling and the 3D transform — use
    `canvas.getBoundingClientRect()` and divide)
  - `const px = ctx.getImageData(x, y, 1, 1).data` — BUT this reads the
    *currently rendered* colorized pixel, which doesn't tell us the label.
    Instead, read from the **stored source `ImageData`**:
    `const label = LAYER_DATA[r].imageData.data[(y * width + x) * 4]`
  - If alpha=0 at that pixel, ignore the click (user clicked outside the extent)
  - Else: `state.colorMap[label] = state.activeBrush; repaint(5); repaint(6); repaint(7);`
- Hover: cheap — on `mousemove`, read the label the same way and update a status
  bar `"cluster N at res R"`. Throttle to requestAnimationFrame to avoid spam.

### 5. 4-angle rotate button

- State: `state.angleIdx` in `{0, 1, 2, 3}` for NW/NE/SE/SW
- On button click: `angleIdx = (angleIdx + 1) % 4; applyRotation()`
- `applyRotation` sets `#stage` CSS transform to
  `rotateX(${PITCH}deg) rotateZ(${BEARINGS[angleIdx]}deg)` with
  `PITCH = 55`, `BEARINGS = [-45, -135, 135, 45]` (NW, NE, SE, SW — confirm
  sign convention when implementing)
- CSS `transition: transform 400ms ease-in-out` for a snap-feel animation.
  No arrow-key support in v2 — button is enough for the user test.

### 6. Controls panel (fixed right sidebar)

Keep v1's controls, minus the attention-depth / focus-sigma sliders
(those were opacity-per-layer gradients that don't port cleanly to the cathedral
layout — v2 shows all three slabs always). Retain:
- **Brush palette**: 20 preset color swatches (reuse v1's list) + `<input type="color">` for custom
- **Colormap preset dropdown**: `tab10` (default), `pastel10`, `set3`, `paired`,
  `viridis10`, `plasma10`, `turbo10` — applying preset overwrites current
  `state.colorMap`, triggers all three repaints
- **Rotate button**: cycles angles, shows current (`NW`, `NE`, `SE`, `SW`)
- **Reset palette**: restore current preset
- **Status bar**: hovered cluster label + resolution + active brush color swatch

### 7. README.md update

Top of file: one-paragraph "How to run" targeting v2 (preferred):
```
python scripts/one_off/cluster_brush_viz/build_raster.py
# then open viz_raster.html in a browser
```
Lower section documents v1 (`build.py` + `viz.html`) as "original deck.gl
prototype, preserved for reference". Explain the rewrite rationale (deck.gl
extrusion broken; raster aesthetic matches project). Do NOT delete v1 text —
append and reorganize.

**Acceptance criteria:**
- Opening `viz_raster.html` (via `file://` — no server needed) shows three
  stacked colorized slabs on a dark background
- Clicking a hex-pixel on any slab recolors that cluster's pixels across all
  three slabs simultaneously (this is the discovery mechanic)
- Applying a colormap preset repaints all three slabs in < 300 ms total
  (perceived instant)
- Rotate button cycles through 4 isometric angles smoothly
- No hex outlines visible — solid fill only (automatic from
  `rasterize_categorical`)
- Cathedral spacing is obvious — the slabs look like floating floors, not a
  flat pancake

**Autonomous decisions (stage3-analyst):**
- Exact `perspective` and `translateZ` distances — tune by eye, aim for
  "cathedral"
- Rotation duration / easing curve
- Whether to display the canvas at native pixel size (scroll to pan) or
  fit-to-viewport with CSS scaling — probably fit-to-viewport with
  `object-fit: contain`-equivalent
- Status-bar layout
- The 20-swatch brush palette — reuse v1's swatches verbatim; if v1 doesn't
  have exactly 20, pick sensibly

## Wave 3: User exploration (Complex — experiment)

**Agent**: human user

**Artifact**: open `viz_raster.html` in browser, play for 10–30 min

**Open questions** (user reports back):
- Does rasterised + canvas-colorize feel as responsive as v1 deck.gl did? If
  not, where's the lag?
- Is the cathedral spacing right, or does it feel too spread out / too cramped?
- Is the 4-angle rotation cycle sufficient, or is continuous drag-to-rotate
  missed?
- Does brushing a raster still feel like "discovery" vs the vector v1?
- Pixel-lookup click vs hex-lookup click — any precision issues at res7 (stamp=4)?

**Deliverable**: brief notes file at `reports/2026-04-17-cluster-brush-viz-v2-notes.md`
capturing the felt experience. The v1 and v2 notes together form the record of
"what rendering approach felt best" for future viz infrastructure decisions.

## Final Wave: Close-out

**Parallel agents** (dispatched by coordinator in one Task batch after Wave 3):
- **devops**: Commit `feat` for `build_raster.py`, `viz_raster.html`, updated
  `README.md`. The PNG outputs and `labels.json` are reproducible from the
  builder — gitignore via existing `*.png` in `scripts/one_off/**` if that rule
  exists; otherwise add to gitignore. Do **not** touch v1 files in the commit.
- **librarian**: Update `codebase_graph.md` with the new viz lineage. Note
  v1 vs v2 explicitly — both coexist, v2 is primary.
- **ego**: Process health assessment — did Cynefin triage help again? Did
  "complicated" decisions actually get more care than "simple" ones? Was
  stage3-analyst continuity across v1+v2 worthwhile, or would a fresh context
  have been cleaner (per `memory/feedback_fresh_contexts.md`)?

## Risks

1. **Label-encoded PNG misreads on old browsers**. `getImageData` is universal
   since ~2015, but some browsers apply color management that shifts R-channel
   values by ±1. Mitigation: write PNGs with `sRGB` color profile explicitly
   stripped, or tolerate ±1 label drift by binning in the lookup. Low risk on
   modern Chrome/Firefox/Safari.
2. **Canvas colorize loop too slow on 4.8M pixels × 3 layers**. If repaint
   budget blows 200 ms and brush feels sluggish, options: (a) WebGL shader
   instead of `getImageData`/`putImageData` — big rewrite; (b) cache canvases
   as `OffscreenCanvas` with `transferControlToOffscreen` → worker thread;
   (c) downscale PNGs to 1000×1200 (loses resolution detail). Defer choice to
   user test results.
3. **CSS 3D transform + large canvases trigger GPU memory pressure on low-end
   hardware**. Three 2000×2400 canvases = ~45 MB RGBA each on the GPU. If
   user reports stutter during rotate, offer a "flat mode" toggle that sets
   `translateZ(0)` on all slabs.
4. **Click precision at res7 stamp=4**. Tiny splat per centroid may leave
   stippled no-data pixels between hexes; user might miss the target. Mitigation:
   stamp=5 or stamp=6 at res7 if stage3-analyst observes patchy labels during
   build. Full pixel coverage > exact stamp discipline for this viz.
5. **v1 vs v2 confusion**. Two pipelines with overlapping artifacts. Mitigation:
   README makes v2 the recommended entrypoint; v1 kept "for reference". Explicit
   naming (`build_raster.py` vs `build.py`) prevents accidental double-execution.
6. **Non-identical raster extents across resolutions**. If `load_boundary` and
   `SpatialDB.extent` disagree between res5/6/7 calls, slabs will misalign.
   Mitigation: compute extent once in `build_raster.py` from the boundary and
   pass it explicitly to all three `rasterize_categorical` calls.

## File Summary

| File | Action | Wave |
|------|--------|------|
| `scripts/one_off/cluster_brush_viz/build.py` | KEEP (v1) — do not modify | — |
| `scripts/one_off/cluster_brush_viz/viz.html` | KEEP (v1) — do not modify | — |
| `scripts/one_off/cluster_brush_viz/build_raster.py` | CREATE (v2) | 1 |
| `scripts/one_off/cluster_brush_viz/labels_res5.png` | CREATE (builder output) | 1 |
| `scripts/one_off/cluster_brush_viz/labels_res6.png` | CREATE (builder output) | 1 |
| `scripts/one_off/cluster_brush_viz/labels_res7.png` | CREATE (builder output) | 1 |
| `scripts/one_off/cluster_brush_viz/labels.json` | CREATE (builder output) | 1 |
| `scripts/one_off/cluster_brush_viz/viz_raster.html` | CREATE (v2) | 2 |
| `scripts/one_off/cluster_brush_viz/README.md` | UPDATE in place | 2 |
| `reports/2026-04-17-cluster-brush-viz-v2-notes.md` | CREATE (post-play, by user) | 3 |

## Execution

Invoke: `/coordinate .claude/plans/hierarchical-cluster-brush-viz-v2.md`
