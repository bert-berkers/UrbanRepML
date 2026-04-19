# Plan: 4-Modality UNet Training + Interpretability Plots

**Session**: deep-settling-moon (2026-03-09)
**Status**: ready for execution

## Context

- GTFS modality activated this session (868K hexagons × 64D at res9)
- 4-modality concat completed: 247K hexagons × 845 cols (AE 64 + POI 687 + Roads 30 + GTFS 64)
- 38% coverage loss from roads inner join (decision: keep inner join, conservative)
- Existing FullAreaUNet trained on 3-modality 781D (commit 09df3aa) — baseline R² = 0.501
- Temporal mismatch: AE=2022, POI/Roads=latest(2026), GTFS=2026. Fine for prototyping.
- **Year label**: Use `20mix` (not `2022`) in all file names to make the mixed provenance explicit.

## Wave 1 — Interpretability Plots (parallel)

### 1a. stage3-analyst: GTFS embedding maps
- **What**: PCA/UMAP of GTFS 64D embeddings, spatial maps colored by top PCA components
- **Input**: `data/study_areas/netherlands/stage1_unimodal/gtfs/gtfs_embeddings_res9.parquet` (or the renamed `netherlands_res9_latest.parquet`)
- **Output**: `data/study_areas/netherlands/stage1_unimodal/gtfs/plots/res9/`
- **Plots**: (1) PCA variance explained, (2) top-3 PCA components as spatial maps, (3) coverage map (which hexagons have transit data vs zeros), (4) UMAP 2D colored by region
- **Use**: `scripts/plot_embeddings.py` infrastructure (rasterize_continuous, SpatialDB for centroids)
- **Accept**: at least 4 plots saved as PNG, interpretable transit patterns visible

### 1b. stage3-analyst: Concatenated embeddings overview
- **What**: Summary stats and spatial maps of the 4-modality concat
- **Input**: `data/study_areas/netherlands/stage2_multimodal/concat/embeddings/netherlands_res9_20mix.parquet`
- **Output**: `data/study_areas/netherlands/stage2_multimodal/concat/plots/res9/`
- **Plots**: (1) per-modality contribution heatmap (variance by modality block), (2) PCA of full 845D — top 3 components as maps, (3) modality correlation matrix (how do AE/POI/Roads/GTFS blocks correlate?), (4) coverage/density map
- **Accept**: at least 4 plots saved as PNG

## Wave 2 — Prepare and Train UNet (sequential)

### 2a. stage2-fusion-architect: Re-concat as 20mix and train
- **What**:
  1. Re-run concat with `--year 20mix`: `python -m stage2_fusion.concat --modalities alphaearth,poi,roads,gtfs --study-area netherlands --resolution 9 --year 20mix` (this produces `netherlands_res9_20mix.parquet` and `_raw.parquet`)
  2. Train: `python scripts/stage2/train_full_area_unet.py --study-area netherlands --year 20mix --resolutions 9,8,7 --epochs 500 --patience 100`
  3. `--year` already accepts strings (patched this session in loader, train script, and concat)
  4. Report: best epoch, best loss, training time, parameter count
  5. Compare to baseline (3-modality: loss 1.52e-4, 789K params)
- **Input**: `data/study_areas/netherlands/stage2_multimodal/concat/embeddings/netherlands_res9_20mix_raw.parquet` (the new 4-modality 845D)
- **Output**: 128D embeddings at `stage2_multimodal/unet/embeddings/`, checkpoint at `stage2_multimodal/unet/checkpoints/`
- **Accept**: training completes, embeddings saved, comparison numbers reported
- **Note**: This will take ~10-30 minutes on GPU. The 845D input (vs 781D before) changes the ModalityFusion input projection but the rest of the architecture adapts automatically via `feature_dim` detection.

## Wave 3 — Post-Training Interpretability (after Wave 2)

### 3a. stage3-analyst: Compare old vs new UNet embeddings
- **What**: Side-by-side comparison of 3-modality vs 4-modality UNet outputs
- **Input**: old embeddings (from commit 09df3aa) + new embeddings (from Wave 2)
- **Output**: `data/study_areas/netherlands/stage2_multimodal/unet/plots/`
- **Plots**: (1) PCA alignment between old/new, (2) spatial difference map (where do embeddings change most?), (3) probe R² comparison if time permits
- **Accept**: at least 2 comparison plots

### 3b. stage3-analyst: DNN probe on 4-modality concat (raw, no UNet)
- **What**: Probe the raw 845D concat directly against leefbaarometer to measure the R² ceiling without UNet processing
- **Input**: `data/study_areas/netherlands/stage2_multimodal/concat/embeddings/netherlands_res9_20mix_raw.parquet`
- **Output**: R² score comparison: raw concat vs UNet 128D
- **Accept**: R² number reported, comparison to UNet baseline (0.501)
- **Note**: Ego flagged this as high-value, low-cost (~15 min). If raw concat beats UNet, it redirects Phase 2 entirely.

### Note on UNet++
UNet++ was originally planned for Wave 3 but is now OFF THE TABLE. Skip connection collapse investigation (cos_sim=0.9994 between res8/res9, diagnosed by stage2-fusion-architect on 2026-03-08) showed denser skip connections would worsen the problem. The revised Phase 2a direction is gated skip connections + consistency loss removal — separate plan needed.

## Wave 4 — Commit + Close-out

### 4a. devops: Commit all changes
- Stage and commit in logical chunks:
  - Commit A: GTFS activation (processor patch, __main__.py, embeddings convention)
  - Commit B: concat.py coverage logging + ModalityFusion docs
  - Commit C: 4-modality concat + UNet training results
  - Commit D: interpretability plots and scripts
- Push all commits

### 4b. Final Wave (mandatory)
- Write coordinator scratchpad
- `/librarian-update`
- `/ego-check`

## Execution
Invoke: `/coordinate .claude/plans/2026-03-09-4modality-unet-and-interpretability.md`
