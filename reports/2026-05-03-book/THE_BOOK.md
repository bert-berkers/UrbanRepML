# The Book of Netherlands

*An afternoon atlas of urban embeddings, rendered through the Voronoi lens.*

*2026-05-03 · Sunday morning*

![Cover: Netherlands as embedding](ch1_frontispiece/cover_alphaearth_rgb_res9.png)

---

## Table of Contents

- [Chapter 1 — Frontispiece](#chapter-1--frontispiece)
- [Chapter 2 — The Four Modalities](#chapter-2--the-four-modalities)
- [Chapter 3 — The Voronoi Showcase](#chapter-3--the-voronoi-showcase)
- [Chapter 4 — Climbing the Hierarchy](#chapter-4--climbing-the-hierarchy)
- [Chapter 5 — Three Embeddings, Side by Side](#chapter-5--three-embeddings-side-by-side)
- [Chapter 6 — Partitioning the Country](#chapter-6--partitioning-the-country)
- [Chapter 7 — A Liveable Land](#chapter-7--a-liveable-land)
- [Chapter 8 — Closing Plate](#chapter-8--closing-plate)
- [Colophon](#colophon)

---

## Chapter 1 — Frontispiece

A country, seen sideways. The Netherlands here is not coastline-and-rivers but something the embedding model dreamed up — three principal components of AlphaEarth's satellite footprint, painted onto the screen as red, green, blue. The result reads like a stained-glass window of land use. The pink hot zones are the Randstad belt, magenta ringing Amsterdam and Rotterdam and stretching down into Brabant. The blue is mostly farmland — the long quiet rectangles of the polders. The yellow-green smudges along the eastern flank are the Veluwe and the Utrechtse Heuvelrug, our forested highlands such as they are. Notice the Wadden Islands across the top, beaded along the horizon like punctuation marks.

![Cover — AlphaEarth as RGB](ch1_frontispiece/cover_alphaearth_rgb_res9.png)
*Three components, one country. Where the magenta clusters tightest, that's where the people are.*

![Hex grid teaser, Amsterdam](ch1_frontispiece/hex_grid_teaser_res9_amsterdam.png)
*A close-up of the underlying grid. Every figure in this book is built from these — H3 hexagons at resolution 9, each about the size of a city block. Around Amsterdam they fit the canals and ringways like a soft mesh.*

![Multi-resolution density](ch1_frontispiece/tessellation_density_multires.png)
*Resolution five through nine, left to right. The country resolves the way a photograph develops in a tray — first a blob, then provinces, then cities, then streets. We'll mostly live at the rightmost panel.*

In the next chapter, we'll meet the four modalities one by one.

---

## Chapter 2 — The Four Modalities

Before fusion, there are ingredients. UrbanRepML eats four of them — satellite, points-of-interest, road network, transit feed — and each one sees the country differently. Walk through them slowly. The same Netherlands looks like a different animal each time.

![AlphaEarth PC1](ch2_modalities/alphaearth_pc1_res9.png)
*AlphaEarth's first principal component. A continuous land-cover gradient — built-up bright, agricultural mid-range, water and forest darker. The Randstad reads as a glowing lattice.*

![POI hex2vec, k-means](ch2_modalities/poi_kmeans_res9.png)
*OpenStreetMap points-of-interest, encoded by hex2vec, then partitioned into ten categorical neighbourhoods. Each colour is a flavour of place — restaurant strips, residential blocks, industrial zones, park edges. The Randstad fragments into a confetti of categories; the rural east settles into longer single-colour fields.*

![Roads density PC1](ch2_modalities/roads_density_res9.png)
*Road network embeddings reduced to one dimension. The brighter corridors trace the A-network — A2 down the spine, A12 east-west across the middle, A4 hugging the coast — connecting arteries to the heart. With a little squinting you can almost see the morning commute.*

![GTFS accessibility PC1](ch2_modalities/gtfs_accessibility_res9.png)
*Transit. Most of the map fades to grey because most hexagons aren't near a stop — public transport in the Netherlands is excellent in the cities and very, very sparse outside them. The handful of bright hexes are exactly where you'd expect: Amsterdam Centraal, Utrecht, Rotterdam, Eindhoven, the Friesland trunk.*

![Four modalities, 2x2](ch2_modalities/four_modalities_2x2.png)
*Four windows on the same country. Each modality emphasises a different geography. The whole point of fusion is to find where they agree and where they argue.*

In the next chapter we'll meet the rasterizer that drew all of this.

---

## Chapter 3 — The Voronoi Showcase

A small detour, just for the craft. Yesterday a new rasterizer shipped — instead of stamping each hex as a fuzzy circle around its centroid, it tessellates the plane into Voronoi cells around centroids and clips against the hex boundary. The result is tighter edges, cleaner interiors, and four distinct rendering modes the toolkit handles natively. This chapter is the rasterizer showing its receipts.

![Continuous mode](ch3_voronoi_showcase/mode_continuous.png)
*Continuous: a scalar (here, UNet PC1) painted across a smooth colormap. Notice the edges — no centroid bleed, no halos.*

![Categorical mode](ch3_voronoi_showcase/mode_categorical.png)
*Categorical: ten cluster labels, one colour each, no interpolation. The boundaries between clusters are crisp where they should be crisp.*

![Binary mode](ch3_voronoi_showcase/mode_binary.png)
*Binary: above-median liveability against below. The country reads as a quiet inversion of the Randstad — most of the densely-built west is, surprisingly, on the under-the-median side. (Liveability scores are not the same thing as economic density.)*

![RGB mode](ch3_voronoi_showcase/mode_rgb.png)
*RGB: three principal components mapped to red, green, blue, no colormap involved. Same rendering mode as the cover, but stitched from a different embedding — here it's the late-fusion concat (208 dimensions of AlphaEarth + hex2vec + roads + GTFS, projected down to three components). The colours land differently because the underlying signal is different. Most information-dense of the four modes, and the most painterly.*

In the next chapter we'll climb the H3 ladder and watch resolution emerge.

---

## Chapter 4 — Climbing the Hierarchy

H3 nests. Each parent hexagon contains seven children, and each of those contains seven more, and so on. A model trained at one resolution can be pushed up or down the ladder. Here we climb from coarse to fine and see what the country chooses to reveal at each step.

![UNet PC1 at res7](ch4_hierarchy/unet_pc1_res7.png)
*Resolution 7. Hexagons span several kilometres. The country reads as a few dozen tiles — provinces, basically. You can see structure but not detail.*

![UNet PC1 at res8](ch4_hierarchy/unet_pc1_res8.png)
*Resolution 8. Cities start to differentiate from countryside. The Randstad's ring shape becomes legible.*

![UNet PC1 at res9](ch4_hierarchy/unet_pc1_res9.png)
*Resolution 9. The neighbourhood scale. Suddenly individual towns have texture — Groningen separates from its hinterland, Maastricht from the Heuvelland, the Wadden Islands from the mainland coast.*

![Multiscale comparison](ch4_hierarchy/multiscale_avg_vs_concat_res9.png)
*Two ways to combine information across resolutions: averaging the embeddings (left) versus concatenating them (right). The avg panel keeps the texture of every level visible at once, leaving a busier, more granular surface — every scale's grain shows through. The concat panel integrates them more aggressively, settling into larger coherent regions where similar embeddings cluster together. Two honest answers to the same question.*

In the next chapter we'll put three different fusion strategies head-to-head.

---

## Chapter 5 — Three Embeddings, Side by Side

This is the meat of the matter. Three ways of fusing the four modalities into a single 64-dimensional view of the country, each with the same four panels: PC1 in the turbo colormap, top-three components as RGB, k-means with the dark2 palette, k-means with the tab10 palette. Read each one as a personality study.

### Concat — the raw mash-up

Just stack the modalities side by side and call it a vector. No learning, no smoothing. It's the strongest baseline you'd grab in an afternoon — and it has a tendency to get dominated by whichever modality has the loudest variance. Watch how the cluster maps fragment in cities and smooth out in farmland.

![Concat PC1](ch5_three_embeddings/concat/pc1_turbo.png)
*Concat PC1. The raw axis of greatest variance.*

![Concat PC RGB](ch5_three_embeddings/concat/pcrgb_rank.png)
*Concat top-3 components as RGB. Sharp, but possibly over-influenced by whichever modality dominates.*

![Concat clusters dark2](ch5_three_embeddings/concat/clusters_dark2.png)
*Concat k-means in the dark2 palette — eight rich, earthy colours.*

![Concat clusters tab10](ch5_three_embeddings/concat/clusters_tab10.png)
*Same partition, the more familiar tab10 palette. Cities are confetti; farmland sweeps long.*

### Ring aggregation — the gentle smoother

A zero-parameter approach that just averages each hexagon with its k-ring neighbours, weighted by distance. No training. It quietly outperforms the learned UNet on the liveability probe — sometimes the best thing you can do is blur a little, generously and locally.

![Ring agg PC1](ch5_three_embeddings/ring_agg/pc1_turbo.png)
*Ring-aggregated PC1. Notice the gradient: nothing fights, everything flows.*

![Ring agg PC RGB](ch5_three_embeddings/ring_agg/pcrgb_rank.png)
*Top-three components as RGB. The country reads almost watercolour.*

![Ring agg clusters dark2](ch5_three_embeddings/ring_agg/clusters_dark2.png)
*Ring-agg clusters in dark2. Smooth boundaries, large coherent regions.*

![Ring agg clusters tab10](ch5_three_embeddings/ring_agg/clusters_tab10.png)
*Ring-agg clusters in tab10.*

![Ring agg clusters annotated](ch5_three_embeddings/ring_agg/clusters_tab10_annotated.png)
*Same partition with text labels overlaid — a legend baked into the geography. Worth pausing on. The cluster names are pinned to the centroid of each region; you can read the country like a guidebook.*

### UNet — the learned representation

A multi-resolution U-Net trained on the four modalities with an accessibility graph for message passing. It's the most ambitious of the three, and the cluster maps show it: distinct urban-rural boundaries, sharp transitions, a country that knows where its cities end.

![UNet PC1](ch5_three_embeddings/unet/pc1_turbo.png)
*UNet PC1. Sharp city-countryside contrast — the model has clearly learned an urban prior.*

![UNet PC RGB](ch5_three_embeddings/unet/pcrgb_rank.png)
*UNet top-3 RGB. The Randstad pulls together as a single visual block.*

![UNet clusters dark2](ch5_three_embeddings/unet/clusters_dark2.png)
*UNet clusters in dark2. Coherent regional shapes — provinces are visible as cluster patches even though provincial boundaries were never an input.*

![UNet clusters tab10](ch5_three_embeddings/unet/clusters_tab10.png)
*UNet clusters in tab10.*

### The three side by side — the punch line

This is the comparison you came for. Same metric, three embeddings, three columns. Read across each row to see what each method emphasises. Pay attention to the Randstad and the eastern frontier — those are the regions where the methods disagree most, and disagreement is where the question lives.

![Three-way PC1](ch5_three_embeddings/comparison/pc1_turbo_3way.png)
*PC1 across all three embeddings. Concat is busy, ring-agg is smooth, UNet is structured.*

![Three-way PC RGB](ch5_three_embeddings/comparison/pcrgb_rank_3way.png)
*Top-three components as RGB across all three. Three different paintings of the same country.*

![Three-way clusters](ch5_three_embeddings/comparison/clusters_tab10_3way.png)
*Cluster partitions across all three. Concat fragments the cities; ring-agg blurs them; UNet draws sharp regional rings around them.*

### A sanity check

![Province boundaries](ch5_three_embeddings/control/province_boundaries.png)
*Provinces, drawn from the cadastre. None of the cluster maps were given this information — but you can squint at the UNet clusters and find Friesland, find Limburg. The model rediscovered geography from scratch.*

In the next chapter we'll keep clustering but climb the cluster count.

---

## Chapter 6 — Partitioning the Country

Five clusters. Ten clusters. Twenty clusters. Same UNet embedding underneath; the only thing changing is how many cuts we make. Each k tells a different story about granularity.

![Clusters k=5](ch6_clusters/clusters_k5_voronoi.png)
*k=5. The barest sketch — coast, urban Randstad, agricultural belt, eastern hinterland, southern hills. Five flavours of Netherlands.*

![Clusters k=10](ch6_clusters/clusters_k10_voronoi.png)
*k=10. The middle setting. Cities separate from suburbs, the Wadden separate from the coastal strip, the Veluwe gets its own colour.*

![Clusters k=20](ch6_clusters/clusters_k20_voronoi.png)
*k=20. Now we're seeing neighbourhoods rather than regions. The Randstad fractures into half a dozen flavours — historical centre, post-war ring, industrial periphery, suburb, edge village. The cost of this granularity is that some clusters become small and noisy.*

In the next chapter we leave abstract embeddings and ask: do they predict something humans care about?

---

## Chapter 7 — A Liveable Land

The Leefbaarometer is a Dutch government index of neighbourhood liveability — built from dozens of indicators across safety, amenities, housing, demographics. We have the 2022 version on the H3 grid. This chapter compares what the embeddings predict (via a simple ridge regression on ring-aggregated features) against what the official index says, and looks closely at where they disagree.

![Liveability target](ch7_liveability/lbm_target_res9.png)
*The target. Green is more liveable, red less. The Randstad is a complicated patchwork — some of the most liveable hexagons sit right next to some of the least. The countryside reads steadily green. The far north and far southeast peripheries cool slightly.*

![Liveability prediction](ch7_liveability/lbm_prediction_res9.png)
*The prediction. A ridge regression on UNet 20mix embeddings, no fine-tuning, just a linear probe. The broad strokes are right — green countryside, mottled cities — but the prediction is smoother than reality. The model has captured the gist, not the texture.*

![Liveability residuals](ch7_liveability/lbm_residuals_res9.png)
*Where the model is wrong. Read this as a "what surprised the predictor" map. Red hexes are places where reality is worse than the model expected — pockets of urban distress the embeddings didn't see coming, concentrated in Rotterdam-Zuid, parts of Amsterdam-West, the southern industrial belt around Heerlen. Blue hexes are places where reality is better than the model thought — quietly thriving small towns, peripheral neighbourhoods that punch above their weight. The pattern is geography, not noise: the residuals carry information the embeddings missed. That's the next thing to work on.*

In the closing chapter, one last picture.

---

## Chapter 8 — Closing Plate

![Best-of, high resolution](ch8_closing/best_of_high_res.png)
*The cover, re-rendered at poster scale. Same data, more pixels — every hex sharp, every coastline clean. We chose this image to close the book because it's the one that needs no caption. The Netherlands as embedding, the embedding as portrait. Pin it above the desk.*

---

## Colophon

This book was assembled on Sunday morning, 2026-05-03, in a single session keyed to supra `russet-rolling-brook-2026-05-03`. Forty-one figures, eight chapters, two hours of work, one cup of tea.

**Plans (kapstok)**
- Today's: `.claude/plans/2026-05-03-make-nice-plots-using-yesterday-s-voronoi-rasterizer.md`
- Yesterday's (the rasterizer that made it possible): `.claude/plans/2026-05-02-rasterize-voronoi-toolkit.md`

**Generation scripts**
- Chapters 1–4, 6–8: `scripts/one_off/build_the_book_2026_05_03.py` (23 PNGs in 2m 13s)
- Chapter 5: `scripts/one_off/viz_three_embeddings_res9_study.py` and `scripts/one_off/viz_three_embeddings_lbm_overlay.py`

**Rasterizer toolkit**
- `utils/visualization.py` — `rasterize_continuous_voronoi`, `rasterize_categorical_voronoi`, `rasterize_binary_voronoi`, `rasterize_rgb_voronoi`, `voronoi_params_for_resolution`

**Datasets**
- AlphaEarth: `data/study_areas/netherlands/stage1_unimodal/alphaearth/netherlands_res9_2022.parquet`
- POI hex2vec: `data/study_areas/netherlands/stage1_unimodal/poi/hex2vec/netherlands_res9_latest.parquet`
- Roads: `data/study_areas/netherlands/stage1_unimodal/roads/netherlands_res9_latest.parquet`
- GTFS: `data/study_areas/netherlands/stage1_unimodal/gtfs/netherlands_res9_latest.parquet`
- UNet (multimodal fusion): `data/study_areas/netherlands/stage2_multimodal/unet/embeddings/netherlands_res9_20mix.parquet` (and res7, res8, multiscale variants)
- Concat: `data/study_areas/netherlands/stage2_multimodal/concat/embeddings/netherlands_res9_20mix.parquet`
- Ring aggregation: `data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet`
- Leefbaarometer 2022: `data/study_areas/netherlands/target/leefbaarometer/leefbaarometer_h3res9_2022.parquet`
- Tessellation: `data/study_areas/netherlands/regions_gdf/netherlands_res{5,6,7,8,9}.parquet`

**Per-figure provenance**
- Aggregate: `ch8_closing/book_provenance.yaml` (23 figures from build_the_book script)
- Per-PNG sidecars: `ch{N}/{figure}.png.provenance.yaml` (Ch5 figures predate W4 sidecar integration — provenance lives in the script and stats JSONs)

**Run metadata**
- `build.book.run.yaml` (run-level summary at the chapter root)

Made possible by the Voronoi rasterizer that shipped one day earlier — a small tool, an afternoon's atlas. See you next weekend.
