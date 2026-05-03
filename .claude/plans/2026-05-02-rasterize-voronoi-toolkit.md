# Rasterize-Voronoi Plotting Toolkit (DRAFT, 2026-05-02)

| Field | Value |
|---|---|
| **Status** | SHIPPED — W6 complete 2026-05-03; human visual sign-off pending on cluster map regression |
| **Source** | Reference impl in `scripts/one_off/viz_ring_agg_res9_grid.py` (commits 703aa41, 511d808); forward-look 2026-04-24 P0; user request 2026-05-02 |
| **Cluster** | Visualization infrastructure (touches `utils/visualization.py` + 17 callers + new `scripts/visualization/` README) |
| **Depends on** | Cluster-2 ledger-sidecars SHIPPED (provenance API in `utils/provenance.py` is stable) |
| **Est** | ~6–8h across 6 waves; plan is shippable wave-by-wave |
| **Supra** | `muted-sliding-dune-2026-05-02` — see `Characteristic-states reference frame` below for per-wave dispositions |

## Telos (the why — read this first)

**The KDTree-Voronoi rasterizer is the standard now.** Centroid-splat with stamp radius (the current `utils/visualization.py` API) was a serviceable approximation; the Voronoi nearest-hex approach is geometrically truthful, faster on gallery panels (8 panels for the price of one Voronoi via index reuse), produces crisper silhouettes, and respects tessellation boundaries via `max_dist_m` rather than fudging coverage with a `stamp` radius. Recent visual studies validated it: the human's geographic eye signed off on the three-embeddings study and the LBM probe overlay, both built on this technique.

**This plan moves the technique from a one-off script into the project's standard plotting infrastructure** — a `scripts/visualization/` toolkit with its own README + mermaid charts, full provenance integration (sidecars + ledger + figure-provenance), data-pipeline-grade input contracts, and a clean migration of the 17 existing callers. The one-off becomes the canonical pattern; the centroid-splat helpers are deprecated and removed.

**Teleology is wave-encoded.** Each wave's *characteristic-state disposition* is annotated below. `/niche` should read the disposition before dispatching the wave's agent — it tells the agent *how* to work (slow vs fast, rigorous vs exploratory, test-heavy vs ship-it). The disposition is the bridge from `/valuate`'s static state to `/niche`'s dynamic execution.

## Characteristic-states reference frame (for `/niche` post-`/clear`)

Inherited from supra `muted-sliding-dune-2026-05-02`:

```
mode=focused
speed=3 explore=3 quality=4 tests=2 spatial=4 model=2 urgency=3 data_eng=3
intent="Improve plotting; toolkit imbues characteristic-states framework into wave structure"
focus=[high-resolution rasterized plotting code, characteristic-states-aware plan handoff]
suppress=[harness .claude/ work (swift-waving-kelp claims that)]
```

**Reading the frame:** focused mode means clear-goal execution; quality=4 + spatial=4 are the load-bearing dimensions (geometry must be right; output must look good); tests=2 is intentionally low (visual QA via the human's eye is the primary gate, not pytest coverage); model=2 means no architecture invention (this is plumbing, not a new approach); speed=3 is neutral (don't rush; don't dawdle).

Each wave below overrides specific dimensions when the wave's character differs from this baseline.

## Waves

### W1 — Spec freeze (spec-writer)

**Disposition:** `quality=5 spatial=5 speed=2 tests=3` — contract-first, no implementation drift.

Lift `voronoi_indices()` + `gather_rgba()` + `rasterize_voronoi()` from `scripts/one_off/viz_ring_agg_res9_grid.py:74-167` into a frozen contract at `specs/rasterize_voronoi.md`. The contract names:

- **Public API surface:** `voronoi_indices(cx_m, cy_m, extent_m, *, pixel_m, max_dist_m) -> (nearest_idx, inside, extent_xy)`; `gather_rgba(nearest_idx, inside, rgb_per_hex) -> (H,W,4) RGBA`; `rasterize_voronoi(...)` one-shot wrapper; plus per-mode wrappers `rasterize_continuous_voronoi`, `rasterize_categorical_voronoi`, `rasterize_binary_voronoi` to replace the four current `rasterize_*` functions in `utils/visualization.py`.
- **CRS contract:** input coords in metric CRS (default EPSG:28992 for NL); a `latlon_to_metric(lats, lons, target_crs)` adapter for callers that pass EPSG:4326. The plan deliberately replaces `stamp` (degrees, fudge factor) with `max_dist_m` (meters, geometric meaning).
- **Index contract:** inputs accept either `(cx, cy)` arrays or a GeoDataFrame indexed by `region_id` (SRAI convention; not `h3_index`). Both forms are first-class; the GeoDataFrame form is preferred because it carries CRS metadata.
- **Output contract:** `(image, extent)` where `extent = (minx, maxx, miny, maxy)` in the input CRS, suitable for `imshow(..., origin='lower', extent=extent)`.
- **Determinism contract:** byte-identical output for byte-identical input + `pixel_m` + `max_dist_m`. KDTree query is deterministic; document this.
- **Default values:** `pixel_m=250.0`, `max_dist_m=300.0` (NL res9 baseline; these match the reference impl).
- **Plot-side helpers:** `plot_spatial_map(ax, image, extent, boundary_gdf, ...)` is preserved (the matplotlib wrapper); `load_boundary`, `filter_empty_hexagons`, `detect_embedding_columns` are unchanged.
- **Errata for `specs/artifact_provenance.md`:** Fail-mode 3 one-line errata ("ledger-append does NOT fire on failed runs; sidecar-without-row is the detectable W4 signal"). Document `repr()`-over-`str()` in `_stringify_with_warn`. Spec-writer task ~10 min, rides alongside W1.

**W1 deliverables:** `specs/rasterize_voronoi.md` frozen; `specs/artifact_provenance.md` errata patched; spec scratchpad written.

### W2a ‖ W2b — Core impl + tests (parallel; geometric-or-developer ‖ qaqc)

**Disposition (W2a impl):** `quality=4 spatial=5 speed=3 tests=2 model=2` — geometric correctness over speed; tests get written alongside but are not the primary gate.
**Disposition (W2b tests):** `quality=4 tests=4 spatial=4 speed=2` — the test-writer pairs against the *spec*, not the impl, and runs higher-than-baseline test rigor for this wave only.

W2a (geometric-or-developer): port `voronoi_indices`, `gather_rgba`, `rasterize_voronoi` from the one-off into `utils/visualization.py`. Add the four per-mode wrappers (`rasterize_continuous_voronoi`, etc.) with signatures matching the existing centroid-splat ones except `stamp` → `max_dist_m`. Add `latlon_to_metric` adapter. Keep the old `rasterize_continuous`/`rasterize_rgb`/`rasterize_binary`/`rasterize_categorical` for one wave only (W3 migrates callers; W6 deletes them). Tag the old functions with a deprecation docstring pointing at the new ones.

W2b (qaqc): write contract tests in `tests/test_rasterize_voronoi.py` against the frozen spec. Required cases: (1) deterministic output; (2) `max_dist_m` cutoff produces correct alpha mask; (3) GeoDataFrame input form equivalent to coord-array form; (4) `latlon_to_metric` round-trips for known NL points; (5) gallery reuse via `voronoi_indices` + multiple `gather_rgba` calls produces same RGBA as the one-shot wrapper. **No Voronoi-vs-centroid-splat regression test** — they are intentionally not equivalent; the new approach replaces, not extends.

**W2 deliverables:** `utils/visualization.py` extended with new API; `tests/test_rasterize_voronoi.py` ≥ 5 test cases passing; W2a + W2b commits.

### W3 — Caller migration (geometric-or-developer + librarian)

**Disposition:** `quality=4 spatial=4 speed=4 tests=2` — speed bumped because the migration is mechanical; quality stays high to catch silent visual regressions.

Migrate the 17 callers in two sub-waves:

- **W3a** — durable stage3 callers: `scripts/stage3/plot_targets.py`, `plot_cluster_maps.py`, `plot_cluster_comparison.py`, `plot_concat_embeddings.py`, `plot_gtfs_embeddings.py`, `plot_probe_comparison.py`, `plot_supervised_probe_viz.py`, `plot_accessibility_maps.py`, `plot_accessibility_overview.py`, `map_causal_emergence.py`, plus `scripts/plot_embeddings.py`, and stage3_analysis library code (`cluster_comparison_plotter.py`, `comparison_plotter.py`).
- **W3b** — one-off callers: `accessibility_viz_all.py`, `cluster_brush_viz/build_raster.py`, `visualize_walk_accessibility.py`, `viz_three_embeddings_res9_study.py`. These get migrated for consistency but flagged as one-offs (30-day shelf life).

**Special case — `scripts/stage3/plot_targets.py:66-204` shadow override:** lines 66-204 contain a local `rasterize_continuous`, `rasterize_categorical`, `rasterize_labels_to_grid`, plus wrappers for `plot_spatial_map` and `_add_colorbar` that shadow the utils versions. Audit each shadow:
1. **Local `rasterize_continuous` / `rasterize_categorical`**: merge their distinguishing params (white background, custom categorical color_map dict) into the utils API as optional kwargs. Then delete the shadows.
2. **`rasterize_labels_to_grid`**: this one is genuinely distinct (label-grid for edge detection, not RGBA output). Move it to utils as a peer function `rasterize_labels`. Don't fold it into the Voronoi API; it has a different output type.
3. **`plot_spatial_map` wrapper / `_add_colorbar` wrapper**: the only differences are `disable_rd_grid=True` and `fontsize` overrides — promote both to kwargs on the utils versions and delete the shadows.

**W3 librarian deliverable:** update `.claude/scratchpad/librarian/codebase_graph/infrastructure.md` with the new API surface and the resolved shadow-override.

**W3 deliverables:** all 17 callers using the new API; `plot_targets.py` shadow eliminated; one commit per sub-wave (W3a + W3b).

### W4 — Provenance integration (geometric-or-developer)

**Disposition:** `quality=4 data_eng=4 tests=3 spatial=3` — data-engineering-diligence bumped because this wave is about audit-trail integrity, not geometry.

Wire the toolkit into `utils/provenance.py`. Two integration points:

1. **Per-figure provenance.** Every figure produced via the toolkit writes a `*.provenance.yaml` sibling — same pattern as cluster-2 used for the 22 viz sites. The new wrapper `save_voronoi_figure(fig, path, source_runs, source_artifacts, plot_config)` wraps `fig.savefig()` and emits the provenance yaml. `source_runs` is the list of `run_id` strings the figure depends on (read from sidecars of input embeddings); `source_artifacts` is the list of input parquet/csv paths; `plot_config` is the dict of `(pixel_m, max_dist_m, mode, color_kwargs, ...)`.
2. **Toolkit run-context (optional but recommended).** When the toolkit is invoked from a script that is itself running under a `SidecarWriter`, the provenance yaml's `parent_run_id` is auto-populated from the active sidecar's `run_id`. This gives `data/ledger/runs.jsonl` a clean parent→figure trail without manual threading.

The toolkit ALSO accepts a `provenance=False` escape hatch for ad-hoc plotting that intentionally skips the audit trail (one-off exploratory work). Default is `provenance=True`.

**Audit script:** `scripts/visualization/audit_figure_provenance.py` — lists every PNG/SVG under `data/study_areas/*/stage3_analysis/` and reports which have a `*.provenance.yaml` sibling vs. which don't (the W4 audit pattern from cluster-2). Same 30-day shelf-life flag as other one-off audit scripts; this one earns durable status if it gets reused.

**W4 deliverables:** `save_voronoi_figure()` in `utils/visualization.py`; audit script; one commit.

### W5 — Documentation + mermaid charts (spec-writer + geometric-or-developer)

**Disposition:** `quality=5 explore=2 speed=2 tests=1 spatial=3` — quality maxed because the README is the load-bearing artifact for future contributors. Tests dropped to 1 (docs aren't tested). Speed dropped (don't ship a sloppy README).

Create `scripts/visualization/README.md` with:

1. **Telos paragraph** — why Voronoi is the standard.
2. **Quick start** — three code snippets (one-shot, gallery via index reuse, with provenance).
3. **API reference** — per-function signature + one-line description, linking to the frozen spec.
4. **Mermaid chart 1 — data flow:**
   ```
   GeoDataFrame[region_id] --> voronoi_indices --> (nearest_idx, inside, extent)
                              \\
                               +--> gather_rgba <-- rgb_per_hex --> RGBA image
                                                                     |
                                                                     v
                                                                save_voronoi_figure
                                                                     |
                                                                     v
                                                              *.provenance.yaml
                                                                     |
                                                                     v
                                                          data/ledger/runs.jsonl (parent_run_id)
   ```
5. **Mermaid chart 2 — wave structure (this plan):** spec → impl ‖ tests → migration → provenance → docs → QA, with arrows showing rate-enabling dependencies.
6. **Mermaid chart 3 — characteristic-states gradient across waves:** a flowchart annotating each wave's disposition shift (`quality=5 speed=2` → `quality=4 speed=4` → ...). This is the *teleological* chart — it shows future contributors how the plan's character changes phase by phase.
7. **Migration notes** — deprecation table mapping old `rasterize_*(stamp=...)` calls to new `rasterize_*_voronoi(max_dist_m=...)` calls.
8. **Style reference** — pointer to `scripts/one_off/cluster_brush_viz/README.md` for the canonical one-off README style; this README is its durable counterpart.

**Module docstrings:** every script in `scripts/visualization/` gets a docstring with `Lifetime: durable` and a one-line stage reference.

**W5 deliverables:** `scripts/visualization/README.md` with three mermaid charts; module docstrings on existing `visualize_clusters.py`; one commit.

### W6 — QA + visual review + cleanup (qaqc + human)

**Disposition:** `quality=5 spatial=5 tests=3 speed=2` — final visual gate is the human's eye on actual rendered maps; pytest is necessary but not sufficient.

1. **Visual diff:** regenerate three canonical figures from `reports/` (three-embeddings study, LBM probe overlay, cluster maps) using the new API. Side-by-side compare with the pre-migration versions. Human signs off on visual quality. Any regression → triage: (a) parameter mismatch (fix call site), (b) genuine improvement (accept), (c) new approach loses something (open `[escalated]` ticket).
2. **Audit script:** run `scripts/visualization/audit_figure_provenance.py`; expect 100% coverage on figures generated post-W4.
3. **Pytest:** `tests/test_rasterize_voronoi.py` must pass; coverage report.
4. **Deletion sweep:** with all 17 callers migrated, delete the deprecated `rasterize_continuous`/`rasterize_rgb`/`rasterize_binary`/`rasterize_categorical` functions from `utils/visualization.py`. Update the docstring header. (No backward-compatibility shims per `feedback_no_fallbacks.md`.)
5. **Ledger validation:** `read_ledger()` returns the same row count as before W4 (figure-provenance lives in `*.provenance.yaml` siblings, not in `runs.jsonl`).

**W6 deliverables:** visual-review report in `reports/2026-05-XX-rasterize-voronoi-toolkit.md`; old centroid-splat functions deleted; final commit; plan marked SHIPPED.

## Cross-shard coordination

- Peer terminal `swift-waving-kelp` (supra `swift-waving-kelp-2026-05-02`) claims `.claude/` harness work and explicitly suppresses plotting. Mutual non-overlap is clean.
- This plan touches: `utils/visualization.py`, `tests/test_rasterize_voronoi.py`, 17 caller files, `scripts/visualization/README.md`, `scripts/visualization/audit_figure_provenance.py`, `specs/rasterize_voronoi.md`, `specs/artifact_provenance.md` (errata only). Coordinator should narrow `claimed_paths` to these in the first OODA cycle.

## Critical files (for `/niche` cold-start)

- **Reference impl:** `scripts/one_off/viz_ring_agg_res9_grid.py:74-167` (lift functions; the file itself stays as a one-off until W6's deletion sweep where it gets archived to `scripts/archive/visualization/` since the reference is now in `utils/`).
- **Current API:** `utils/visualization.py:130-365` (the four `rasterize_*` functions + `plot_spatial_map`).
- **Shadow override:** `scripts/stage3/plot_targets.py:66-204`.
- **Provenance API:** `utils/provenance.py:122-413` (`compute_config_hash`, `SidecarWriter`, `ledger_append`, `read_ledger`).
- **Sidecar spec:** `specs/artifact_provenance.md` (15-field schema + figure-provenance variant).
- **Style reference:** `scripts/one_off/cluster_brush_viz/README.md`.
- **Plan style reference:** `.claude/plans/2026-04-18-cluster2-ledger-sidecars.md`.

## Verification

End-to-end test: run `scripts/stage3/plot_concat_embeddings.py --study-area netherlands` post-migration. Expect:
1. PNG output at `data/study_areas/netherlands/stage3_analysis/embeddings/{date}/concat_voronoi.png`
2. `concat_voronoi.png.provenance.yaml` sibling with `source_runs`, `source_artifacts`, `plot_config`
3. Visual match (rough; voronoi != stamp so not byte-identical) with the pre-migration version
4. Audit script reports figure as covered

## Coordinator notes (for `/niche`)

- **Wave commits:** one commit per wave on green, per cluster-2's pattern. Don't batch.
- **Pre-edit gate:** any direct edit by the coordinator (not via dispatched agent) is restricted to (a) this plan file, (b) `.gitignore`. All other edits go through specialist agents.
- **Markov-completeness:** coordinator close-out at end-of-session must satisfy Contract 1 (7 items). Specialists' scratchpads need 1+2+7.
- **Deferred items from forward-look that this plan resolves:**
  - rasterize-Voronoi refactor plan ✅ (this plan)
  - `plot_targets.py:66` shadow override ✅ (W3 special case)
  - `specs/artifact_provenance.md` Fail-mode 3 errata ✅ (W1 ride-along)
- **Deferred items from forward-look NOT in this plan:**
  - Hook-filename-drift fix in `subagent-stop.py` markov_check — that's harness work, claimed by `swift-waving-kelp`
  - Memory landing for `feedback_valuate_sets_state_only.md` — already exists per MEMORY.md, no action needed

## If you only read this section, here's the gist

This plan replaces the centroid-splat rasterization API in `utils/visualization.py` with a KDTree-Voronoi rasterizer lifted from a working one-off (`scripts/one_off/viz_ring_agg_res9_grid.py`), wires it into the existing provenance system (sidecars + ledger + figure-provenance), migrates 17 callers, eliminates a shadow override in `plot_targets.py`, and ships a `scripts/visualization/README.md` with three mermaid charts (data flow, wave dependencies, characteristic-states gradient). Six waves, each with an explicit characteristic-states disposition that `/niche` reads before dispatching the wave's agent — the disposition is how `/valuate`'s static state shapes `/niche`'s dynamic execution. Ship wave-by-wave; commit on green; visual review by human is the final gate.
