# Cluster Brush Viz

Self-contained HTML artifact for brush-discovering MiniBatchKMeans clusters
on Netherlands H3 embeddings. Three resolutions (res5 / res6 / res7)
aggregated upward from res9 `ring_agg` labels.

Two pipelines coexist:

- **v2 (recommended)**: rasterised label PNGs + canvas palette swap.
  `build_raster.py` + `viz_raster.html`. Matches the project aesthetic
  (rasterised hexagon maps everywhere else in stage3), no extrusion bugs,
  solid-fill cathedral slabs.
- **v1 (reference)**: deck.gl `H3HexagonLayer` stack. `build.py` + `viz.html`.
  Preserved for reference only — see rewrite rationale below.

**Plan**: `.claude/plans/hierarchical-cluster-brush-viz-v2.md`
(v1 plan: `.claude/plans/hierarchical-cluster-brush-viz.md`)

---

## How to run (v2, recommended)

```bash
uv run python scripts/one_off/cluster_brush_viz/build_raster.py
```

Then open `scripts/one_off/cluster_brush_viz/viz_raster.html` in a browser —
no server needed, the HTML loads the sibling `labels.json` and three
`labels_res{5,6,7}.png` files via `fetch('./...')` from `file://`.

**Interactions** (v2):

- Click any hex-pixel on any slab to recolor that cluster across **all three
  slabs** simultaneously. Drag to paint.
- Pick an active brush from the 20 preset swatches, or click the big swatch
  to open a custom color picker.
- Switch colormap preset (`tab10`, `pastel10`, `set3`, `paired`, `viridis10`,
  `plasma10`, `turbo10`) — repaints all three slabs in well under 300 ms.
- Rotate button cycles through 4 isometric angles (NW → NE → SE → SW).
- Hover for `cluster N @ resR` in the status bar; the swatch-mini shows the
  cluster's current color.

---

## How to run (v1, reference only)

```bash
uv run python scripts/one_off/cluster_brush_viz/build.py
```

Then open `viz.html`. Self-contained when the inline data is under 12 MB
(the current run is ~0.73 MB so it inlines).

---

## v1 vs v2: what changed and why

v1 was a deck.gl prototype that shipped working brush + colormap presets but
failed on two axes the user flagged:

1. **Broken extrusion**. deck.gl 9.1.9's `H3HexagonLayer.extruded=true` path
   drops into a `ColumnLayer` sub-layer that tries to resolve `h3-js`
   `cellToLatLng` from `window` globals. The globals aren't wired when
   deck.gl loads via CDN, so the stack renders flat. Multiple workarounds
   attempted, none clean.
2. **Wrong aesthetic**. The rest of UrbanRepML uses rasterised hexagon maps
   (`utils/visualization.rasterize_categorical` → matplotlib PNG). v2 matches
   that look; vector hex outlines (unwanted) are eliminated automatically.

v2 also drops the attention-depth / focus-sigma opacity sliders (v1-only:
they were per-layer opacity gradients; under v2's always-visible cathedral
layout they don't port cleanly). If per-layer opacity becomes interesting
again, it's one `slab.style.opacity` line.

---

## Inputs (both versions)

`data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet`
(208D, ~398k hexes). Resolved via `StudyAreaPaths.fused_embedding_file('ring_agg', 9, '20mix')`.

---

## Outputs (alongside this README)

### v2

- `labels_res5.png`, `labels_res6.png`, `labels_res7.png` — 2400×2000 RGBA
  label-encoded PNGs. R channel = cluster id 0..9, alpha=0 outside the
  Netherlands extent. Opening one in an image viewer looks near-black —
  that's expected, it's a label map, not a rendered visualization.
- `debug_preview_res{5,6,7}.png` — finished-color tab10 previews for sanity
  checking the builder output. Not loaded by the viz.
- `labels.json` (~500 B) — sidecar with `extent`, `raster_shape`, `k`,
  per-resolution pixel counts, and source provenance.

### v1

- `viz.html` — fully self-contained when inline data is under 12 MB.
- `data.json` — sidecar, only written if inline payload exceeds 12 MB.

---

## v1 data-shape contract (for reference)

```jsonc
{
  "k": 10,
  "source": {
    "study_area": "netherlands",
    "model": "ring_agg",
    "resolution": 9,
    "year": "20mix",
    "pca_components": 32,
    "pca_variance_retained": 0.9xxx
  },
  "bounds": {"minLon": ..., "maxLon": ..., "minLat": ..., "maxLat": ...},
  "layers": {
    "5": [{"hex": "85...", "cluster": 3, "n_children": 49, "entropy": 0.82}, ...],
    "6": [...],
    "7": [...]
  }
}
```

Per-row fields:

| field         | type      | meaning                                                   |
| ------------- | --------- | --------------------------------------------------------- |
| `hex`         | string    | H3 cell id at the layer's resolution                      |
| `cluster`     | int 0..9  | majority-vote label from res9 children                    |
| `n_children`  | int       | how many res9 cells aggregated to this parent             |
| `entropy`     | float     | Shannon entropy of child-label distribution in **nats**   |

v1 uses `entropy` for aggregation-confidence opacity. v2 carries entropy
through `build_raster.py`'s aggregation but does not currently consume it
(reserved for a possible v3 per-hex opacity mode).

---

## Known limits

- No basemap tile layer; dark background only (embeddings speak for
  themselves).
- Clustering is deterministic (`random_state=42` inside `clustering_utils`).
- v2 cathedral Z-spacing (res7=0 / res6=1200 / res5=2400 px) is tuned by eye —
  easy to adjust in `viz_raster.html` under `Z_BY_RES`.
- v2 rotation is 4-snap (NW/NE/SE/SW); no continuous drag-to-rotate.
