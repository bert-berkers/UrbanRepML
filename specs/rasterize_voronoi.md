# Rasterize-Voronoi Plotting Toolkit — Frozen Contract

## Status: Approved (frozen for W2 implementation, 2026-05-02)

**Frame**: Plan `.claude/plans/2026-05-02-rasterize-voronoi-toolkit.md` (W1 spec freeze).
**Replaces**: The four centroid-splat helpers in `utils/visualization.py:130-365` (`rasterize_continuous`, `rasterize_rgb`, `rasterize_binary`, `rasterize_categorical`) — these are deprecated by W2 and deleted in W6.
**Lifted from**: Reference impl `scripts/one_off/viz_ring_agg_res9_grid.py:74-167` (commits `703aa41`, `511d808`; visually validated by the human across the three-embeddings study and the LBM probe overlay in `reports/`).
**Complements**: `specs/artifact_provenance.md` (W4 wires `save_voronoi_figure` into `*.provenance.yaml` siblings).

## Purpose

Centroid-splat with a `stamp` radius (degrees, fudge factor) was a serviceable approximation for H3-indexed maps, but it carries three structural bugs at NL scale: directional south-east bleed from asymmetric splat offsets, density-dependent speckle holes from hexagonal-vs-rectangular packing, and lat-lon aspect distortion (1° lon ≠ 1° lat at 52° N). The KDTree-Voronoi approach replaces all three: query the nearest hex centroid per pixel in a metric CRS, paint the cell's value if within `max_dist_m` of the centroid, leave transparent otherwise. The output is a geometrically-truthful Voronoi tessellation of the input centroids, clipped to a sane geographic neighbourhood.

The decisive operational win is **gallery reuse**: the KDTree query is the dominant cost (~5 s for NL res9 at 250 m/px), so a single `voronoi_indices` call amortises across N panels via N cheap `gather_rgba` fancy-index gathers. The 8-panel ring-agg gallery renders in ~5 s + 8×(~0.05 s); the same gallery via centroid-splat would have been 8×(~3 s).

## Public API surface

Five functions in `utils/visualization.py`. Three core, two adapter/wrapper.

### Core: `voronoi_indices`

```python
def voronoi_indices(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    ...
```

Build the per-pixel nearest-hex index array + alpha mask once. This is the gallery primitive.

**Inputs**:
- `cx_m`, `cy_m`: hex centroid coordinates in a metric CRS (default contract: EPSG:28992 RD New for NL). 1-D float arrays of identical length `N`.
- `extent_m`: `(minx, miny, maxx, maxy)` bounding box in the same metric CRS. Note the input order is `(minx, miny, maxx, maxy)` — matches `geopandas.total_bounds` and shapely conventions.
- `pixel_m`: output pixel size in metres. Keyword-only.
- `max_dist_m`: Voronoi cutoff in metres. Keyword-only.

**Outputs** (3-tuple):
- `nearest_idx`: `(H, W)` `int64`, index into the input centroid array.
- `inside`: `(H, W)` `bool`, `True` where pixel is within `max_dist_m` of its nearest centroid.
- `extent_xy`: `(minx, maxx, miny, maxy)` — note the **swapped Y order vs input** because matplotlib `imshow(..., extent=...)` expects `(left, right, bottom, top)`.

**Pixel-centre convention**: pixel `(0, 0)` covers the rectangle `[minx, minx+pixel_m] × [miny, miny+pixel_m]` and is queried at its centre `(minx + 0.5*pixel_m, miny + 0.5*pixel_m)`. This matches `origin='lower'` in `imshow`.

**Implementation note**: width = `max(1, ceil((maxx-minx)/pixel_m))`, height = `max(1, ceil((maxy-miny)/pixel_m))`. The KDTree is `scipy.spatial.cKDTree` queried with `k=1`. Output dtypes are load-bearing — downstream `gather_rgba` assumes `int64` indexing into `(N, 3)` colour arrays.

### Core: `gather_rgba`

```python
def gather_rgba(
    nearest_idx: np.ndarray,
    inside: np.ndarray,
    rgb_per_hex: np.ndarray,
) -> np.ndarray:
    ...
```

Project a per-hex colour table onto the precomputed index grid. Cheap fancy-index gather.

**Inputs**:
- `nearest_idx`: `(H, W)` int from `voronoi_indices`.
- `inside`: `(H, W)` bool from `voronoi_indices`.
- `rgb_per_hex`: `(N, 3)` float in `[0, 1]`. Length must match the centroid array length passed to `voronoi_indices`.

**Output**: `(H, W, 4)` `float32` RGBA. RGB is `rgb_per_hex[nearest_idx]`; alpha is `inside.astype(float32)` (1.0 inside, 0.0 outside). No anti-aliasing — alpha is hard-edged by design (the matplotlib `interpolation='nearest'` default preserves the Voronoi silhouette).

### Core: `rasterize_voronoi`

```python
def rasterize_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    rgb_per_hex: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ...
```

One-shot wrapper for the single-panel case. Equivalent to:

```python
nearest_idx, inside, extent_xy = voronoi_indices(cx_m, cy_m, extent_m,
                                                  pixel_m=pixel_m,
                                                  max_dist_m=max_dist_m)
return gather_rgba(nearest_idx, inside, rgb_per_hex), extent_xy
```

**Output**: `(image, extent_xy)`. `image` is `(H, W, 4)` `float32` RGBA. `extent_xy` is `(minx, maxx, miny, maxy)` — directly usable as `imshow(image, extent=extent_xy, origin='lower', interpolation='nearest', aspect='equal')`.

### Per-mode wrappers (replace the four current `rasterize_*` functions)

Each wrapper accepts the input data + a CRS-mode dispatch + the KDTree parameters, builds an `rgb_per_hex` from its mode-specific input, and forwards to `rasterize_voronoi`.

```python
def rasterize_continuous_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    values: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ...
```

Continuous scalar values → colormap → RGBA Voronoi raster. `vmin`/`vmax` default to the 2nd/98th percentiles of `values` (preserves existing behaviour from `rasterize_continuous`).

```python
def rasterize_categorical_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    labels: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    n_clusters: int,
    cmap: str = "tab20",
    color_map: dict[int, tuple[float, float, float]] | None = None,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ...
```

Integer labels → categorical RGBA. `color_map` (optional `dict[label → (r, g, b)]`) overrides `cmap` when provided — this absorbs the `plot_targets.py` shadow's distinguishing parameter (W3 case 1).

```python
def rasterize_binary_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    color: tuple[float, float, float] = (0.2, 0.5, 0.8),
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ...
```

Presence-only Voronoi raster. All hexes painted in `color`; alpha mask cuts at `max_dist_m`.

```python
def rasterize_rgb_voronoi(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    rgb_array: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ...
```

Pre-computed `(N, 3)` RGB → RGBA Voronoi raster. Identity wrapper around `rasterize_voronoi` — kept for naming parity with the deprecated `rasterize_rgb`.

### Adapter: `latlon_to_metric`

```python
def latlon_to_metric(
    lats: np.ndarray,
    lons: np.ndarray,
    target_crs: int = 28992,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

Reproject lat/lon (EPSG:4326) arrays to the metric `target_crs`. Returns `(cx_m, cy_m)` arrays of the same shape as the inputs. Uses `pyproj.Transformer` with `always_xy=True`; `target_crs` defaults to `28992` (RD New) for NL. Required for callers whose centroid arrays are in EPSG:4326 (e.g. SRAI defaults).

## CRS contract

**Hard rule**: all `voronoi_indices` / `gather_rgba` / `rasterize_*_voronoi` inputs are in a metric CRS. Default for NL work is **EPSG:28992 (RD New)**. Callers passing EPSG:4326 arrays MUST first call `latlon_to_metric`. There is no automatic detection or aliasing — passing degrees as if they were metres is a caller bug, not an API responsibility.

**Replacement, not aliasing**: the deprecated `stamp` parameter (degrees, fudge factor) is GONE. There is no `stamp=` keyword on the new API. There is no compatibility shim that translates `stamp=4` to `max_dist_m=400` or similar. Per `memory/feedback_no_fallbacks.md`: clean breaks over compatibility chains. W3's caller migration is mechanical; W6 deletes the old functions.

## Index contract

The toolkit accepts inputs in two equivalent forms:

1. **Coordinate arrays** (low-level): `(cx_m, cy_m)` numpy 1-D float arrays. Caller is responsible for CRS — no metadata is carried with naked arrays.
2. **GeoDataFrame indexed by `region_id`** (preferred): SRAI convention. The toolkit MUST extract `(cx_m, cy_m)` from `gdf.geometry.centroid.x` / `.y` after a `.to_crs(target_crs)` call. The GeoDataFrame's index name MUST be `region_id` (not `h3_index`) per `.claude/rules/index-contracts.md`.

Both forms are first-class and produce byte-identical output. The GeoDataFrame form is preferred at API boundaries because it carries CRS metadata explicitly. W2 implementations MAY add a thin GeoDataFrame-overload wrapper above each per-mode function (e.g. `rasterize_continuous_voronoi_gdf(gdf, value_col, ...)`); this is an implementation choice, not a contract requirement — the contract requires only that callers can plumb a SRAI-indexed GeoDataFrame through to the rasterizer without manually extracting coords.

**Stage-boundary note**: stage-1 parquet artifacts on disk use `h3_index` as the column name (legacy). `MultiModalLoader` and stage-2+ code rename to `region_id`. The toolkit lives at the visualization edge — stage-3 — so it sees `region_id`. Callers reading raw stage-1 parquet must rename before passing to the toolkit.

## Output contract

Every public function returns `(image, extent_xy)` where:
- `image`: `(H, W, 4)` `float32` RGBA. Alpha is hard-edged (0.0 or 1.0, no AA).
- `extent_xy`: `(minx, maxx, miny, maxy)` 4-tuple in the input CRS. **Y-order is swapped from input** so the tuple plugs directly into matplotlib `imshow(..., extent=extent_xy, origin='lower', interpolation='nearest', aspect='equal')`.

Pixel `(0, 0)` (top-left in array order, bottom-left geographically with `origin='lower'`) is centred at `(minx + 0.5*pixel_m, miny + 0.5*pixel_m)`. The image is `H` rows × `W` columns where `H = max(1, ceil((maxy-miny)/pixel_m))` and `W = max(1, ceil((maxx-minx)/pixel_m))`. The actual rendered geographic extent slightly exceeds `extent_m` when `(maxx-minx)/pixel_m` is non-integer (the last pixel is full-width; the rendered extent rounds up to the nearest pixel boundary).

## Determinism contract

**Strong determinism**: same `(cx_m, cy_m)`, same `extent_m`, same `pixel_m`, same `max_dist_m` ⇒ byte-identical `nearest_idx`, byte-identical `inside`. `cKDTree.query(k=1)` with the default `eps=0` is deterministic; ties are broken by the order in which centroids were inserted (i.e. by input array order).

**Compositional determinism**: same `nearest_idx`/`inside` + same `rgb_per_hex` ⇒ byte-identical `gather_rgba` output. Numpy fancy indexing is deterministic.

**Caveats** (not contract violations, but caller-relevant):
- Floating-point construction of pixel-centre coordinates (`minx + (np.arange(width) + 0.5) * pixel_m`) is platform-dependent at the last bit. In practice this never affects KDTree results because `max_dist_m` is far above ULP noise. If a caller needs *cross-platform* byte-identical raster output, they MUST round inputs to a fixed precision before calling.
- The colormap LUTs in matplotlib have versioned across releases. Per-mode wrappers that take a `cmap` string inherit any matplotlib non-determinism — this is matplotlib's contract, not ours.

## Defaults

- `pixel_m = 250.0` — NL res9 baseline. ~1200 × 1400 pixel raster covering the European Netherlands.
- `max_dist_m = 300.0` — covers H3 res9 corner-to-centroid distance (~174 m) with a generous margin while keeping the NL silhouette crisp. Anything beyond 300 m from a res9 centroid is outside the tessellation.

These defaults are CONTRACT defaults: the function signatures bake them in. Callers working at a different resolution MUST override:

| H3 resolution | Suggested `pixel_m` | Suggested `max_dist_m` |
|---:|---:|---:|
| 7 | 1500 | 2000 |
| 8 | 600 | 800 |
| 9 | 250 | 300 |
| 10 | 100 | 120 |
| 11 | 40 | 50 |

(These are guidance, not contract — they're documented in the W5 README, not enforced in code.)

## Plot-side helpers (preserved from current API)

The following functions in `utils/visualization.py` are **unchanged** by this spec:
- `load_boundary(paths, crs=28992)` — boundary polygon loader, filters to largest part for NL.
- `filter_empty_hexagons(emb_df, display_name, constant_threshold=0.10)` — three-pass background filter.
- `detect_embedding_columns(df)` — column name inference (`A00`, `emb_0`, `gtfs2vec_0`, ...).
- `plot_spatial_map(ax, image, extent, boundary_gdf, ...)` — matplotlib wrapper.

`plot_spatial_map` gains two **optional** kwargs (W3 absorbs the `plot_targets.py` shadow per case 3):
- `disable_rd_grid: bool = False` — suppress the 50-km RD grid overlay even when `show_rd_grid=True` (the shadow's reason for existing).
- `title_fontsize: int = 11` — already exists in current API; preserved.

`_add_colorbar` (private but used by `plot_targets.py`'s shadow) gains:
- `label_fontsize: int = 10` — already exists.
- `tick_fontsize: int = 8` — already exists.

These two are documented here for completeness; their existence in current code means W3 case 3 is "delete shadows, callers use existing kwargs."

## Provenance integration hook (W4 forward-reference)

W4 will add one wrapper:

```python
def save_voronoi_figure(
    fig: matplotlib.figure.Figure,
    path: pathlib.Path | str,
    source_runs: list[str],
    source_artifacts: list[pathlib.Path | str],
    plot_config: dict,
    *,
    provenance: bool = True,
) -> None:
    ...
```

**Reserved name** — W2 must not collide. Semantics (full contract is in W4 of the plan):
- Calls `fig.savefig(path, dpi=..., bbox_inches="tight")`.
- When `provenance=True` (default), emits `{path}.provenance.yaml` per `specs/artifact_provenance.md` §"Figure-provenance specialisation". `source_runs` and `source_artifacts` are recorded verbatim; `plot_config` is recorded under the figure-provenance `plot_config` field with non-JSON-serialisable values coerced via `_stringify_with_warn` (see provenance spec errata 2026-05-02).
- When `provenance=False`, behaves as a thin `fig.savefig` wrapper (escape hatch for ad-hoc exploratory plotting, per plan §W4).
- When invoked under an outer `SidecarWriter`, the figure-provenance yaml's `source_runs` SHOULD include the active sidecar's `run_id` automatically. W4 specifies the threading mechanism; the spec here only reserves the name.

## Peer function: `rasterize_labels`

W3 case 2 moves `rasterize_labels_to_grid` from `scripts/stage3/plot_targets.py:66-204` into `utils/visualization.py` as `rasterize_labels`. **Reserved name, peer to the Voronoi API but NOT folded into it** — different output type:

```python
def rasterize_labels(
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    labels: np.ndarray,
    extent_m: tuple[float, float, float, float],
    *,
    pixel_m: float = 250.0,
    max_dist_m: float = 300.0,
    fill_value: int = -1,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ...
```

Returns `(label_grid, extent_xy)` where `label_grid` is `(H, W)` `int64` — the *integer label* of the nearest hex (or `fill_value` outside `max_dist_m`). Used downstream by edge detection and label-boundary visualisation; an RGBA gather would erase the integer-label structure that those algorithms need. Internally a thin wrapper over `voronoi_indices` + `np.where(inside, labels[nearest_idx], fill_value)`.

## Worked example — gallery reuse

The canonical pattern from the reference impl. Eight panels share one Voronoi query:

```python
import numpy as np
from utils.visualization import voronoi_indices, gather_rgba
from utils.spatial_db import SpatialDB

# 1. Load embeddings + centroids (metric CRS) -----------------------------
db = SpatialDB.for_study_area("netherlands")
hex_ids = emb_df.index.to_numpy()
cx_m, cy_m = db.centroids(list(hex_ids), resolution=9, crs=28992)

extent_m = (cx_m.min() - 2_000, cy_m.min() - 2_000,   # (minx, miny,
            cx_m.max() + 2_000, cy_m.max() + 2_000)   #  maxx, maxy)

# 2. Build the Voronoi index ONCE -----------------------------------------
nearest_idx, inside, extent_xy = voronoi_indices(
    cx_m, cy_m, extent_m, pixel_m=250.0, max_dist_m=300.0,
)

# 3. Gather N panels via fancy-index ---------------------------------------
panels = []
for rgb_per_hex in (cluster_rgb_tab10, cluster_rgb_set1, cluster_rgb_dark2,
                    pc_rgb, turbo_rgb, cividis_rgb, hsv_rgb, ...):
    panels.append(gather_rgba(nearest_idx, inside, rgb_per_hex))

# 4. imshow each panel with the shared extent ------------------------------
for ax, img in zip(axes.flat, panels):
    ax.imshow(img, extent=extent_xy, origin="lower",
              interpolation="nearest", aspect="equal")
```

For the single-panel case, use `rasterize_voronoi` instead — same output, no manual `voronoi_indices` plumbing. For the per-mode case (continuous scalars, integer labels, presence-only), use the `rasterize_*_voronoi` wrappers.

## What's NOT in this spec (intentional exclusions)

The following are explicitly out of scope. Implementations MUST NOT add them:

1. **No `stamp=` shim or `stamp_to_max_dist_m` translator**. The old `stamp` (degrees) and new `max_dist_m` (metres) are not interconvertible — `stamp` was a fudge that mixed pixel-radius semantics with degree units. Mapping it to metres would smuggle in lat-lon aspect distortion that the new API exists to eliminate. Per `memory/feedback_no_fallbacks.md`: clean breaks. Callers update their call sites in W3.
2. **No Voronoi-vs-centroid-splat regression test**. The two methods are intentionally not equivalent — Voronoi is a replacement, not an extension. A regression test would lock in the bugs the new API exists to fix. W2b's test contract (in the plan) explicitly forbids this case.
3. **No automatic CRS detection from naked arrays**. If a caller passes `(cx_m, cy_m)` arrays whose values look like degrees (`-180 ≤ x ≤ 180`), the toolkit does NOT warn or auto-reproject. CRS is the caller's responsibility on the array path; the GeoDataFrame path enforces CRS via `gdf.crs`. Auto-detection would be a fallback and falls under the no-fallbacks rule.
4. **No `width=` / `height=` overrides**. The output dimensions are derived from `extent_m` and `pixel_m`. Callers wanting a specific output size MUST adjust `pixel_m` (geometrically meaningful) or pad/crop the result themselves. The deprecated `RASTER_W=2000, RASTER_H=2400` constants from `utils/visualization.py:47-48` are removed in W6 — they were a centroid-splat-specific hack and don't survive the geometric-CRS rewrite.
5. **No interpolation modes other than nearest-hex**. Bilinear / IDW / kriging are alternative tessellation models, not `voronoi_indices` modes. If a future plan needs them, they get their own peer function (e.g. `rasterize_idw`) — not a `mode=` kwarg on this API.
6. **No multi-resolution dispatch**. The toolkit operates on one centroid array at a time. Multi-resolution figures (e.g. res5+res8+res10 stacked) are a caller orchestration concern: call `voronoi_indices` once per resolution and composite the resulting RGBAs on the matplotlib axis.
7. **No alpha blending or sub-pixel anti-aliasing**. The Voronoi silhouette is hard-edged by design. Anti-aliasing the cell boundaries would erase the geometric truthfulness that motivated the rewrite. Callers wanting softer edges MUST post-process the RGBA output (e.g. `scipy.ndimage.gaussian_filter`) themselves.

## Migration table (for W3 and W6)

| Deprecated (W2 docstring tag, W6 deleted) | Replacement (W2 added) | Notes |
|---|---|---|
| `rasterize_continuous(cx, cy, values, extent, *, stamp=...)` | `rasterize_continuous_voronoi(cx_m, cy_m, values, extent_m, *, max_dist_m=...)` | `extent` order unchanged (`(minx, miny, maxx, maxy)`); `stamp` removed. |
| `rasterize_rgb(cx, cy, rgb_array, extent, *, stamp=...)` | `rasterize_rgb_voronoi(cx_m, cy_m, rgb_array, extent_m, *, max_dist_m=...)` | Identity wrapper around `rasterize_voronoi`. |
| `rasterize_binary(cx, cy, extent, *, color=..., stamp=...)` | `rasterize_binary_voronoi(cx_m, cy_m, extent_m, *, color=..., max_dist_m=...)` | |
| `rasterize_categorical(cx, cy, labels, extent, n_clusters, *, cmap=..., stamp=...)` | `rasterize_categorical_voronoi(cx_m, cy_m, labels, extent_m, *, n_clusters=..., cmap=..., color_map=...)` | `n_clusters` becomes keyword-only; `color_map` (dict override) is new. |
| `RASTER_W = 2000, RASTER_H = 2400` (module constants) | (deleted; derive from `extent_m` and `pixel_m`) | No replacement; centroid-splat-specific. |
| `_stamp_pixels` (private) | (deleted) | Centroid-splat helper, not needed. |
| `scripts/stage3/plot_targets.py:66-204` shadows | Use `utils/visualization.py` versions with new optional kwargs | W3 case 1+3. |
| `scripts/stage3/plot_targets.py:rasterize_labels_to_grid` | `utils/visualization.py:rasterize_labels` | W3 case 2 — peer function, not Voronoi-API. |

## qaqc invariants (for W2b and W6)

W2b test contract MUST verify:

1. **Determinism**: `voronoi_indices(...)` called twice with identical inputs produces identical `nearest_idx` (byte-equal) and identical `inside` (byte-equal).
2. **Cutoff correctness**: for a synthetic input where one centroid is placed `>max_dist_m + 1` from any pixel centre, the corresponding pixel has `inside == False`. For a centroid placed `<max_dist_m - 1`, the corresponding pixel has `inside == True`.
3. **GeoDataFrame ≡ coord-array**: passing a SRAI-indexed `GeoDataFrame[region_id]` produces byte-identical RGBA to passing the equivalent `(cx_m, cy_m)` arrays.
4. **`latlon_to_metric` round-trip**: a known NL point (e.g. Amsterdam centre, 52.3676° N 4.9041° E) round-trips through `latlon_to_metric` → EPSG:28992 → back via `pyproj` to within 1 cm.
5. **Gallery equivalence**: `voronoi_indices(...)` followed by `gather_rgba(...)` produces byte-identical RGBA to the one-shot `rasterize_voronoi(...)` for the same `rgb_per_hex`.

W6 visual review (human gate) MUST verify:
- The three canonical figures (`reports/` three-embeddings study, LBM probe overlay, cluster maps) regenerated with the new API match the pre-migration versions visually. They will NOT match byte-identically — Voronoi ≠ stamp by design.
- No regressions in cell silhouette crispness, NL boundary alignment, or colour fidelity.

## Open questions punted to W2 (impl detail, not contract)

These are NOT contract items — they're implementation choices the spec deliberately leaves to W2:

1. Should `latlon_to_metric` accept GeoDataFrame input (in addition to `(lats, lons)` arrays)? The contract requires only the array form; the GDF overload is a convenience.
2. Should `rasterize_categorical_voronoi` validate that `labels.max() < n_clusters`? Defensive programming — the spec is silent.
3. Where in `utils/visualization.py` should the new functions live relative to the existing helpers? Section ordering is style, not contract.
4. Should the GDF-overload wrappers be separate functions (`rasterize_continuous_voronoi_gdf`) or `@singledispatch`? Style choice.

W2 records its choice in its scratchpad; W3 picks up whatever choice was made.
