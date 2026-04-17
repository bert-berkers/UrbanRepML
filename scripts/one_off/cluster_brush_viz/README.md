# Cluster Brush Viz

Self-contained HTML artifact for brush-discovering MiniBatchKMeans clusters
on Netherlands H3 embeddings. Three resolutions (res5/res6/res7) aggregated
upward from res9 ring_agg labels.

**Plan**: `.claude/plans/hierarchical-cluster-brush-viz.md`

## Status

- **Wave 1 (this commit)**: data pipeline + minimal viz shell (single
  H3HexagonLayer at res7, flat, tab10 colors, no interactivity)
- **Wave 2**: stacked isometric layers, brush recolor, sliders, colormap presets
- **Wave 3**: user exploration — open `viz.html`, play, write notes

## Run

```bash
uv run python scripts/one_off/cluster_brush_viz/build.py
```

Then open `viz.html` in a browser (no server needed).

## Inputs

- `data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet`
  (208D, ~398k hexes). Built via `StudyAreaPaths.fused_embedding_file('ring_agg', 9, '20mix')`.

## Outputs (beside this README)

- `viz.html` — fully self-contained when data is small; `fetch('./data.json')` otherwise.
- `data.json` — sidecar, only present if inline payload exceeds 12 MB.

Both artifacts live under `scripts/one_off/` (temporary, not checked in unless small).

## Data Shape Contract (for Wave 2)

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

Wave 2 uses `entropy` for aggregation-confidence opacity.

## Known limits (Wave 1)

- No basemap tile layer; dark background only (embeddings speak for themselves).
- Single-res rendering — stacking + isometric camera is Wave 2.
- No interactivity — Wave 2 adds brush, sliders, colormap dropdown.
- Clustering is deterministic (`random_state=42` inside `clustering_utils`).
