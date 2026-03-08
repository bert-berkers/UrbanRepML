# CE Map Visualization: Stamp Size Fix + Improved Maps

**Created**: 2026-03-08 afternoon
**Context**: Recurring map visualization problem — single-pixel centroids leave gaps. User has seen this before. Fix: stamp a radius around each centroid proportional to H3 cell size.

## The Problem

`rasterize_centroids()` (used by `map_causal_emergence.py` and `linear_probe_viz.py`) writes exactly 1 pixel per hexagon:
```python
image[py, px, :3] = rgb_masked  # single pixel!
```

At res9, 247K hexagons on a 2000×2400 canvas = sparse single-pixel dots with visible gaps. The hexagons should fill their area on the map.

## The Fix

Replace single-pixel stamping with a disk/square stamp whose radius is derived from the H3 cell size at the target resolution. At res9 in EPSG:28992, cells are ~175m edge-to-edge. On a 2000px-wide canvas covering ~300km, that's ~1.2 pixels per hex — so a stamp radius of 1-2 pixels fills the gaps without excessive overlap.

The stamp radius should be calculated dynamically:
```python
hex_edge_m = h3.average_hexagon_edge_length(resolution, unit='m')
pixel_per_meter = width / (maxx - minx)
stamp_radius = max(1, int(hex_edge_m * pixel_per_meter * 0.8))
```

Then stamp a filled disk of that radius around each centroid pixel.

## Wave 0: Fix the rasterization stamp

**Agent: stage3-analyst** — Modify `rasterize_centroids()` in `scripts/stage3/map_causal_emergence.py`:
1. Calculate dynamic stamp radius from H3 resolution + canvas extent
2. Replace single-pixel write with disk stamp (use `skimage.draw.disk` or manual numpy indexing)
3. Handle edge cases: clamp stamp to image bounds
4. Also check if `stage3_analysis/linear_probe_viz.py` has the same 1-pixel pattern — if so, fix there too (it's the original source of this helper)

**Acceptance**: Re-run `map_causal_emergence.py` and visually confirm hexagons fill their area without excessive overlap.

## Wave 1: Re-generate CE maps

**Agent: execution** — Run `python scripts/stage3/map_causal_emergence.py` to regenerate both maps with the stamp fix.

**Agent: qaqc** — Visual review of output PNGs. Check:
- No visible gaps between hexagons
- No blob overlap that obscures spatial patterns
- Colorbar still readable
- Boundary overlay visible under/around the raster

## Wave 2: Commit

**Agent: devops** — Commit the stamp fix + regenerated figures. Push.

## Final Wave: Close-out

- Coordinator scratchpad
- `/librarian-update`
- `/ego-check`

## Execution

Invoke: `/coordinate .claude/plans/2026-03-08-ce-maps-stamp-fix.md`
