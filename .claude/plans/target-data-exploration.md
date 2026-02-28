# Plan: Comprehensive Target Data Exploration

## Context

Two target datasets, zero visualizations. The data reveals rich structure worth exploring:

**Leefbaarometer** (546K hex, 6 dimensions): Overall liveability (lbm, range 3.4-5.0) with sub-dimensions (fys=physical, onv=safety, soc=social, vrz=services, won=housing). The vrz dimension is *anti-correlated* with most others (r=-0.62 with onv, -0.66 with soc). Mean lbm=4.19.

**Urban Taxonomy** (935K hex, 7 hierarchy levels): Morphotope classification from urbantaxonomy.org. Level 1 splits 75%/25% Incoherent vs Coherent fabric. L3 has 8 named types. L7 reaches 106 classes. Entropy grows from 0.81 -> 5.71 bits across levels.

**Cross-target insight**: 87% overlap (477K hexagons). Incoherent Small-Scale Sparse Fabric has the *highest* liveability (4.24), while Coherent Dense Adjacent Fabric has the *lowest* (4.12). The bottom 5% of lbm is dominated by Coherent Dense Adjacent (39%), while the top 5% is dominated by Incoherent Small-Scale Sparse (40%).

Output directories:
- `data/study_areas/netherlands/target/leefbaarometer/plots/`
- `data/study_areas/netherlands/target/urban_taxonomy/plots/`

## Approach

Create `scripts/plot_targets.py` -- a comprehensive EDA script. Reuse centroid-rasterization pattern from `linear_probe_viz.py:701-760` (SpatialDB -> EPSG:28992 -> pixel array -> imshow).

## Data Facts (discovered during planning)

- Leefbaarometer file: `leefbaarometer_h3res10_2022.parquet` (546,927 rows, 8 cols: weight_sum, n_grid_cells, lbm, fys, onv, soc, vrz, won)
- Urban Taxonomy file: `urban_taxonomy_h3res10_2025.parquet` (935,329 rows, 11 cols: type_level1-7, n_morphotopes, name_level1-3)
  - NOTE: year is 2025, not 2022
- Boundary: `netherlands_boundary.geojson` -- Polygon (not MultiPolygon), EPSG:4326
- `matplotlib_venn` is NOT available -- use a simple bar chart for coverage_venn instead

## Plots to Generate (~30 PNGs)

### A. Leefbaarometer Spatial Maps (`leefbaarometer/plots/`)

| # | File | What & Why |
|---|------|-----------|
| 1 | `spatial_all_dimensions.png` | 2x3 grid: all 6 dims on Netherlands map (viridis, percentile-clipped). The core spatial overview. |
| 2 | `spatial_lbm_detail.png` | Single large map of lbm with custom diverging colormap centered at median. Shows where NL lives well/poorly. |
| 3 | `spatial_vrz_vs_soc.png` | Side-by-side vrz and soc maps -- they're r=-0.66 correlated. Visually confirm the anti-correlation is spatial. |
| 4 | `distributions.png` | 2x3 faceted histograms with KDE. Shows the distribution shape of each dimension. |
| 5 | `correlation_matrix.png` | Annotated heatmap of pairwise correlations. Highlights the vrz anti-correlation cluster. |
| 6 | `pairplot_subsample.png` | Seaborn pairplot of 5K random hexagons across all 6 dims. Reveals non-linear relationships. |
| 7 | `dimension_profiles_boxplot.png` | Boxplots of all 6 dims on shared y-axis. Quick visual of spread and outliers per dimension. |
| 8 | `spatial_extremes.png` | Map highlighting bottom 5% (red) and top 5% (green) of lbm. Where are the best/worst neighborhoods? |

### B. Urban Taxonomy -- Every Level (`urban_taxonomy/plots/`)

Individual spatial class maps for every hierarchy level:

| # | File | What & Why |
|---|------|-----------|
| 9 | `spatial_level1.png` | 2 classes: Coherent vs Incoherent. The fundamental binary split. |
| 10 | `spatial_level2.png` | 4 classes. First subdivision -- scale + density. |
| 11 | `spatial_level3.png` | 8 classes (all named). The richest interpretable level. |
| 12 | `spatial_level4.png` | 16 classes. Where subtypes start appearing. |
| 13 | `spatial_level5.png` | 25 classes. Mid-hierarchy -- still visually distinct on the map. |
| 14 | `spatial_level6.png` | 56 classes. Fine-grained -- use grouped colormap (hue by L3 parent, lightness by L6 subclass). |
| 15 | `spatial_level7.png` | 106 classes. The full resolution. Use L3-parent-grouped colormap for visual coherence. |

Overview + statistical plots:

| # | File | What & Why |
|---|------|-----------|
| 16 | `spatial_named_levels.png` | 1x3 grid: L1, L2, L3 side by side. Shows hierarchy refinement spatially. |
| 17 | `class_distributions_all_levels.png` | 7-panel faceted bar chart for L1-L7. Shows class imbalance at each level. |
| 18 | `hierarchy_entropy.png` | Dual-axis line plot: entropy (bits) + n_classes vs level. Quantifies hierarchy complexity growth. |
| 19 | `class_imbalance_ratio.png` | Bar chart: largest/smallest class ratio per level. L6 has 104,774x imbalance. |
| 20 | `hierarchy_tree.png` | Dendrogram/sunburst: L1->L3 hierarchy with class counts. Shows how the tree splits. |
| 21 | `boundary_confidence.png` | Histogram of n_morphotopes. 64% of hexagons have n_morphotopes=1 (clean interior). |
| 22 | `spatial_boundary_hexagons.png` | Map coloring hexagons by n_morphotopes (1=solid, 2+=boundary). Shows where morphotope edges cluster. |

### C. Cross-Target Analysis (`urban_taxonomy/plots/`)

| # | File | What & Why |
|---|------|-----------|
| 23 | `lbm_by_taxonomy_l1.png` | Violin: lbm per L1 class. Quick binary comparison. |
| 24 | `lbm_by_taxonomy_l3.png` | Violin: lbm per L3 morphotype (8 classes). The headline finding. |
| 25 | `all_dims_by_taxonomy_l3.png` | Heatmap: mean of each lbm dim (rows) per L3 type (cols). Multidimensional signature per morphotype. |
| 26 | `all_dims_by_taxonomy_l1.png` | Same heatmap but L1. Simpler signal to read. |
| 27 | `extreme_composition.png` | Stacked bar: L3 composition of bottom 5% vs top 5% lbm hexagons. |
| 28 | `weight_sum_by_taxonomy.png` | Boxplot: weight_sum (population proxy) per L3 class. Shows density-morphotype link. |
| 29 | `spatial_lbm_with_taxonomy_overlay.png` | lbm spatial map with L1 contour boundaries overlaid. Does morphology align with liveability? |
| 30 | `coverage_venn.png` | Bar chart: leefbaarometer coverage (547K) vs taxonomy coverage (935K) vs overlap (477K). |

## Key Files

| File | Action |
|------|--------|
| `scripts/plot_targets.py` | CREATE |
| `stage3_analysis/linear_probe_viz.py:701-760` | READ -- rasterization pattern |
| `utils/spatial_db.py` | READ -- `SpatialDB.for_study_area().centroids()` |
| `utils/paths.py` | READ -- `StudyAreaPaths.target()`, `target_file()`, `area_gdf_file()` |

## Implementation Notes

- Rasterize spatial plots via centroid projection to EPSG:28992 pixel array (same as linear_probe_viz)
- Use Netherlands boundary from `paths.area_gdf_file()` for extent + background fill
- Filter to European NL (largest polygon when MultiPolygon) to exclude Caribbean islands
- Subsample for pairplot (5K points) to keep it readable
- L1-L5: use `tab10`/`tab20` categorical colormaps (<=25 classes)
- L6-L7: use L3-parent-grouped colormap -- assign hue by L3 ancestor, vary lightness within. This keeps 56/106 classes visually interpretable by preserving the hierarchy.
- Continuous maps: `viridis` for values, `RdBu_r` for diverging
- DPI 150, tight bbox, white facecolor

## Waves

### Wave 1: Implementation (2 parallel stage3-analyst agents)
- **Agent A**: Write `scripts/plot_targets.py` -- Section A (leefbaarometer plots 1-8) + shared infrastructure (rasterization helper, boundary loading, CLI)
- **Agent B**: Write `scripts/plot_targets.py` -- Sections B+C (urban taxonomy plots 9-22, cross-target plots 23-30)

Problem: two agents can't write the same file in parallel.

**Revised**: Single stage3-analyst agent writes the entire `scripts/plot_targets.py`. The script is ~800-1000 lines of visualization code -- one agent can handle this.

### Wave 2: Execution (execution agent)
- Run `python scripts/plot_targets.py --study-area netherlands`
- Capture output, report success/failure and PNG counts

### Wave 3: QAQC (qaqc agent)
- Verify all ~30 PNGs exist in expected directories
- Spot-check a few images for visual quality (read them)
- Check for any runtime warnings or errors

### Wave 4: Final Wave
- Coordinator scratchpad
- `/librarian-update`
- `/ego-check`

## Verification

```bash
python scripts/plot_targets.py --study-area netherlands
# Expected: ~8 PNGs in leefbaarometer/plots/, ~22 PNGs in urban_taxonomy/plots/
```
