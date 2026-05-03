# Make Nice Plots Using Yesterday's Voronoi — 2026-05-03

| Field | Value |
|---|---|
| **Status** | SHIPPED 2026-05-03 — single commit `2de38b6`, 41 PNGs, 2232-word THE_BOOK.md |
| **Shard** | `russet-rolling-brook` |
| **Mode** | book-builder: parallel chapter-makers → collator → review |
| **Intent** | Make a fuckton of plots using yesterday's Voronoi rasterizer; collate into a cool book you can read on the sofa. So beautiful. |
| **Est** | ~3-4h across 5 waves |
| **Output** | `reports/2026-05-03-book/` containing chapter PNGs + `THE_BOOK.md` |

## Reference frame (echoed from supra session for cold-start resume)

```
mode=creative speed=3 explore=4 quality=3 tests=1 spatial=4 model=2 urgency=2 data_eng=3
intent="Make nice plots using yesterday's Voronoi rasterizer toolkit; lighter Sunday pace, leveraging Saturday's kapstok harness so /niche W0 surfaces a candidate plot menu."
focus=['nice plots via Voronoi rasterizer toolkit', "leverage Saturday's kapstok harness for multi-thread W0"]
suppress=['voronoi-toolkit carry sweep (defer to next session)', 'heavy infra/harness work']
```

If `/clear` lands you here cold: supra yaml is `.claude/supra/sessions/russet-rolling-brook-2026-05-03.yaml`. Valuate scratchpad is `.claude/scratchpad/valuate/2026-05-03.md`. Read both for full state.

## Frame — why a kapstok and not a fixed-target plan

High explore + creative mode means the goal isn't to execute a known backlog. It's to **find the spark** within the focus domain (nice plots), then crystallize it into a wave structure. So this plan does NOT prescribe one task — it lists candidate threads, gives a decision rule, and lets W0 pick. The kapstok itself is the artifact Saturday's harness work produced — eating our own dog food.

## Candidate threads

Numbered for stable references. Coordinator's read of resonance is in the ★ rankings (more stars = more timely). User picks one, picks a sequence, or describes a wildcard.

### Thread A — Voronoi modality gallery ★★★

**Why now**: The Voronoi rasterizer just shipped 4 mode wrappers (continuous, categorical, binary, RGB). No single artifact yet showcases the full surface. A 1-page beauty gallery (one panel per mode, Netherlands at res9) is a natural showcase + sanity check.

**Scope**:
- 1 figure, 4 panels (continuous, categorical, binary, RGB) at res9 over Netherlands.
- Each panel uses the cleanest data we have for that mode: AlphaEarth PC1 for continuous, kmeans cluster labels for categorical, an LBM-quartile threshold for binary, RGB-stitched (PC1, PC2, PC3) for RGB.
- Voronoi `max_dist_m=300m` consistent across panels for visual coherence.
- Save with figure-provenance sidecar via `SidecarWriter`.

**Acceptance**:
- 1 PNG + 1 `.provenance.yaml` at `reports/2026-05-03-voronoi-gallery/gallery_res9.png`.
- All four panels render without NaN holes inside the study-area boundary.
- Visual coherence: same colormap family for continuous + RGB, distinct categorical, monochrome for binary.

**Estimated waves**: W1 stage3-analyst (single dispatch, ~1h). Final wave.

### Thread B — Three-embeddings update via Voronoi backend ★★★★ (sneaky carry-closure)

**Why now**: Two scripts (`viz_three_embeddings_res9_study.py`, `viz_three_embeddings_lbm_overlay.py`) had their imports patched in W6 of voronoi-toolkit but were never smoke-called ("would overwrite data figures"). The forward-look at `coordinator/2026-05-04.md` flagged them for smoke-call before the 30-day shelf expires (2026-05-24). This thread reframes the smoke-call as making nice plots — the figures DO get overwritten, but they're overwritten with the new rasterizer's improved versions. Closes the carry by amplifying it.

**Scope**:
- Run both scripts end-to-end (`python scripts/one_off/viz_three_embeddings_res9_study.py` and `python scripts/one_off/viz_three_embeddings_lbm_overlay.py`).
- Capture stdout/stderr; verify no NameError, no kwarg-mismatch.
- Compare new figures vs git-tracked old versions side-by-side (the visual-diff machinery from W4-W6 already exists).
- Write provenance sidecars if not already.

**Acceptance**:
- Both scripts return exit 0.
- New figures visibly improved (boundary tightness from `max_dist_m=300m` vs old centroid-stamp fudge).
- A 2026-05-03 visual-diff folder showing old-vs-new for both figures.
- Forward-look carry `[→geometric-or-developer]` smoke-call → marked `[done|2026-05-03]`.

**Estimated waves**: W1 execution (run scripts, parallel) → W2 qaqc (visual-diff, brief). Final wave.

### Thread C — Hierarchical landscape: res8/9/10 in one figure ★★

**Why now**: We have data at multiple H3 resolutions but no single artifact stitches them into a hierarchical-landscape view. The new Voronoi rasterizer with `voronoi_params_for_resolution(res)` helper (added in W3a) makes this trivial — three panels with auto-tuned `max_dist_m` per resolution. Showcases the resolution-aware API.

**Scope**:
- 1 figure, 3 panels stacked or side-by-side: res8 (coarse), res9 (medium), res10 (fine) over the same Netherlands sub-region (e.g., Amsterdam metropolitan core for visual interest).
- Same modality (AlphaEarth PC1, since it has all three resolutions).
- `voronoi_params_for_resolution(res)` for max_dist auto-tuning.
- Annotate hex count per panel.

**Acceptance**:
- 1 PNG + sidecar at `reports/2026-05-03-hierarchical-landscape/amsterdam_res8910.png`.
- Visible spatial detail progression coarse→fine.
- Provenance sidecar records resolution + max_dist_m per panel.

**Estimated waves**: W1 stage3-analyst (~1h). Final wave.

### Thread D — Wildcard / your call

User-described thread. Examples that fit the focus + don't trip the suppress:
- A specific city's modality side-by-side (AE / hex2vec / roads / GTFS) at res9.
- A "before/after Voronoi" pedagogical figure for the report at `reports/2026-05-03-rasterize-voronoi-toolkit.md` (the file you have open in IDE — possibly the natural complement).
- A leefbaarometer probe-quality map using the RGB mode to encode (R²-residual, prediction, target).
- Beauty-shot rerender of an existing figure at higher resolution + better colormap.

**Scope/Acceptance/Waves**: TBD on user description.

## Decision rule for /niche W0

1. ✅ Surface menu — done.
2. ✅ User picked: A + B + C, all of Netherlands.
3. ✅ User upgraded to: "make a fuckton of plots, collate into a cool book to read on the sofa, so beautiful."
4. → Threads A/B/C absorbed into a chapter-based book structure; W1 dispatches chapter-makers in parallel; W2 collates; W3 qaqc reviews; W4 devops commits.

## The Book of Netherlands — chapter structure

Output: `reports/2026-05-03-book/` with subdirs per chapter + a unified `THE_BOOK.md` collating all figures with prose framing. Coffee-table aesthetic: large figures, minimal chrome, captions short and evocative. PNGs at full raster resolution (2000x2400 per panel — never downsize for grid layouts; raster_stamp_size memory feedback).

**Chapter 1 — Frontispiece** (3-4 plots)
- Cover: Netherlands at res9, AlphaEarth RGB-stitched (PC1, PC2, PC3) — the country as embedding.
- Hex-grid teaser: small Netherlands subregion showing the H3 res9 tessellation density.
- Foreword figure: H3 hex tessellation density at res5/7/8/9/10 (5 panels, just the grid).

**Chapter 2 — The Modalities** (5-6 plots)
- AlphaEarth res9: PC1 continuous + RGB stitched.
- POI / hex2vec res9: kmeans clusters categorical + continuous PC1.
- Roads res9: connectivity / network density continuous.
- GTFS res9: accessibility continuous (or stop density binary).
- 2x2 collage: all four modalities side-by-side at the same res9 view.

**Chapter 3 — The Voronoi Showcase** (4 plots = Thread A absorbed)
- Continuous mode (AE PC1).
- Categorical (kmeans cluster labels).
- Binary (LBM quartile threshold or POI presence).
- RGB (stitched PCA from concat embeddings).

**Chapter 4 — The Resolution Hierarchy** (4-5 plots = Thread C absorbed, using REAL hierarchical UNet outputs)
- UNet res7 PC1 over Netherlands.
- UNet res8 PC1 over Netherlands.
- UNet res9 PC1 over Netherlands.
- UNet res9 multiscale_avg vs multiscale_concat side-by-side.
- (Bonus) Same point at all three resolutions in a 3-panel zoomed view.

**Chapter 5 — The Three Embeddings** (3-4 plots = Thread B absorbed)
- Run `scripts/one_off/viz_three_embeddings_res9_study.py` — 3 embeddings (concat raw, ring_agg, UNet) at res9. Smoke-call closes carry.
- Run `scripts/one_off/viz_three_embeddings_lbm_overlay.py` — same 3 with LBM overlay.
- Visual-diff old vs new (W6 patched imports — verify figures look better, not regressed).

**Chapter 6 — Clusters of the Land** (4-5 plots)
- W6 cluster_maps_k10.png (the [needs:human] visual sign-off artifact — embedded as a chapter figure resolves the carry naturally).
- Hierarchical clusters: `kmeans_clustering_hierarchical/` if it has multi-k outputs.
- Single-layer clusters at k=5, k=10, k=20 if available.

**Chapter 7 — Liveability** (3-4 plots)
- Leefbaarometer target map (raw target).
- LBM probe predictions (best probe model, e.g. DNN on ring_agg per recent comparison).
- LBM residuals (prediction - target).
- Cross-method comparison if `stage3_analysis/comparison/` has a R²-by-method bar chart we can re-render.

**Chapter 8 — Closing / Colophon** (2-3 plots)
- Best-of: highest-resolution beauty shot of one panel from earlier chapters at full quality.
- Method colophon: a sketch/diagram of the pipeline (stage1 → stage2 → stage3) with file pointers.
- Provenance footer: link list to `reports/2026-05-03-rasterize-voronoi-toolkit.md` (the user has it open) + the supra session yaml.

Total: ~30 plots. Fuckton-class. Everything via `rasterize_voronoi` API + `SidecarWriter` provenance.

## Wave structure (locked)

**Wave 1** (parallel, 2 dispatches):

1. **stage3-analyst** — book-builder: writes `scripts/one_off/build_the_book_2026_05_03.py` with chapter-by-chapter functions for Ch 1, 2, 3, 4, 6, 7, 8 (skip Ch5 — execution handles that in dispatch 2). Runs the script, producing ~25 PNGs + per-figure `.provenance.yaml` sidecars in `reports/2026-05-03-book/ch{1,2,3,4,6,7,8}_*/`. Sequential per chapter to avoid OOM. Coffee-table aesthetic: minimal chrome, full 2000x2400 raster per panel, evocative captions baked into figure titles.
2. **execution** — Ch 5: run `scripts/one_off/viz_three_embeddings_res9_study.py` and `scripts/one_off/viz_three_embeddings_lbm_overlay.py` end-to-end with `--no-overwrite` if supported (else just run; the new figures via Voronoi rasterizer ARE the improved versions). Capture stdout/stderr, confirm exit 0. Identify output paths (read script source); list new figures.

**Wave 2** (sequential, 1 dispatch):

5. **spec-writer** — Collator: assembles `reports/2026-05-03-book/THE_BOOK.md`. Writes the cover page, chapter intros, captions, ToC, colophon. Pulls all PNGs from chapter subdirs into the markdown with relative-link images. Tone: cozy, evocative, sit-on-sofa. Not technical paper; not bullet-point summary. Prose.

**Wave 3** (parallel, 1-2 dispatches):

6. **qaqc** — Visual review: open every chapter folder, verify (a) every PNG renders without NaN holes inside study area, (b) every PNG has a `.provenance.yaml` sidecar, (c) THE_BOOK.md renders cleanly in PyCharm preview, (d) all image links resolve. Returns pass/partial-fail/fail with per-chapter notes.

**Wave 4** (sequential):

7. **devops** — Commit: stage `reports/2026-05-03-book/` + the W1 chapter-builder scripts + this kapstok update. Single commit message: `feat(reports): The Book of Netherlands — coffee-table report (~30 plots, 8 chapters, Voronoi rasterizer showcase)`.

**Final Wave**:
- Coordinator scratchpad to `.claude/scratchpad/coordinator/2026-05-03.md` (russet-rolling-brook entry)
- `/librarian-update`
- `/ego-check`

## Anti-scope (do NOT do this session)

- ❌ Voronoi-toolkit carry sweep (visual sign-off + spec-tail + cosmetic + smoke-call as a bundled sweep) — explicitly suppressed by user. Note: Thread B touches the smoke-call carry but as plot-making, not as carry-sweep.
- ❌ Heavy infra/harness work — Saturday's harness session owns that domain (currently dormant; revive only if needed).
- ❌ Multi-day refactor commitments — scope to one-session-shippable. Lighter Sunday.
- ❌ Spec edits, model architecture work, training runs.

## Carry-items (open from forward-look, mostly deferred)

- `[open|0d] [needs:human] [contract:4]` cluster-maps visual sign-off at `reports/visual_diff_W6/2026-05-03/cluster_maps/2026-05-03/clusters_k10.png` — surface natural moment if user is already looking at plots; otherwise defer.
- `[open|0d] [→spec-writer]` spec-tail audit + W3a/W4 errata bundle — DEFERRED.
- `[open|0d] [→qaqc]` cosmetic stale `:func:` refs at `utils/visualization.py:715,884` — DEFERRED.
- `[open|0d] [→geometric-or-developer]` smoke-call for two patched one-off scripts — Thread B closes this if chosen.

## Peer-terminal pointer

- `muted-sliding-dune` (PID 31720, supra `muted-sliding-dune-2026-05-02`) — ENDED. Last activity 2026-05-03 09:50, plan SHIPPED.
- `swift-waving-kelp` (supra `swift-waving-kelp-2026-05-02`) — ENDED.
- `verdant-wading-storm` (PID 32920) — ENDED.
- **No live peers.** Solo session.

## If you only read this section

This plan is a kapstok — Saturday's harness work productized as a wave-scaffolding tool. Multi-thread (creative + explore=4). Intent: nice plots via the new Voronoi rasterizer; lighter Sunday. Three concrete threads (A: 4-mode gallery; B: three-embeddings smoke-call as plot-making; C: hierarchical res8/9/10 landscape) plus D wildcard. Coordinator's pick if user wants a default: **B → A** (B closes a forward-look carry while making a nice plot; A is the cleanest pure-showcase). Anti-scope is firm: no carry-sweep, no infra, no spec edits. Solo session, no peers. Sidecar provenance via `SidecarWriter` is the figure-output contract.

## Execution

After user picks, /niche W0 rewrites this Status line + the wave structure below. Then:
Invoke: `/coordinate .claude/plans/2026-05-03-make-nice-plots-using-yesterday-s-voronoi-rasterizer.md`
