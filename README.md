# UrbanRepML

**High-Quality Urban Embeddings through Manageable Late-Fusion**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SRAI](https://img.shields.io/badge/Spatial-SRAI-green)](https://github.com/kraina-ai/srai)

Learn dense urban representations by processing modalities independently, then fusing them with spatial awareness through multi-resolution U-Net architectures on H3 hexagonal grids. Built on [SRAI](https://github.com/kraina-ai/srai) for all spatial operations.

For philosophical background and preliminary results see the [Active Inference Institute talk](https://www.youtube.com/watch?v=UYD8CR_Xorg&ab_channel=ActiveInferenceInstitute).

## Three-Stage Pipeline

### Stage 1: Modality Encoders (`stage1_modalities/`)

Each modality is processed independently into H3-indexed embeddings (resolution 9, ~545K hexagons for Netherlands):

| Modality | Source | Status |
|----------|--------|--------|
| **AlphaEarth** | Google Earth Engine pre-computed embeddings | Working (64-dim) |
| **Aerial Imagery** | PDOK Netherlands orthophotos → DINOv3 | Partial |
| **POI** | OpenStreetMap points → categorical density | Partial |
| **Roads** | OSM network topology → connectivity metrics | Partial |
| **GTFS** | Transit stops → accessibility potential | Planned |

### Stage 2: Fusion (`stage2_fusion/`)

Two multi-resolution U-Net architectures fuse modality embeddings using H3 hierarchy and spatial graphs:

- **FullAreaUNet** — processes entire study area with lateral accessibility graph. Multi-resolution encoder-decoder (res 8-10) with skip connections.
- **ConeBatchingUNet** — hierarchical cones (res 5→10), each ~1,500 hexagons. Memory-efficient and parallelizable. Most promising direction.

### Stage 3: Analysis (`stage3_analysis/`)

Post-training analysis, probing, and visualization:

- **Linear Probe** — fits OLS linear regression with spatial block cross-validation from embeddings to external targets. Currently probing against 6 [Leefbaarometer](https://www.leefbaarometer.nl/) livability indicators across the Netherlands.
- **Linear Probe Viz** — coefficient bar charts, heatmaps, faceted cross-target comparisons, spatial residual maps (quantile-binned dissolve), RGB top-3 coefficient maps.
- **Clustering** — hierarchical multi-scale clustering across H3 resolutions with landscape visualization.

## Current State

The Netherlands study area has 64-dimensional AlphaEarth embeddings covering ~545K H3 res-9 hexagons. Linear probes against Leefbaarometer livability scores show the embeddings contain meaningful urban signal. Fusion training (Stage 2) and multi-modality integration are the active frontier.

## Setup

```bash
git clone https://github.com/bertberkers/UrbanRepML.git
cd UrbanRepML
uv sync              # Install all dependencies
uv sync --extra dev  # Include dev tools
```

## Study Areas

All processing is study-area based. Each area is self-contained:

```
data/study_areas/{area_name}/
├── area_gdf/           # Study area boundary
├── regions_gdf/        # H3 tessellation (via SRAI)
├── embeddings/         # Per-modality embeddings (Stage 1)
├── cones/              # Cone cache for ConeBatchingUNet
├── urban_embedding/    # Fused results (Stage 2)
├── analysis/           # Probe results & cluster assignments (Stage 3)
└── plots/              # Visualizations
```

- **Netherlands** — primary area, complete AlphaEarth coverage (~545K hexagons)
- **Cascadia** — coastal urban-forest interface, Pacific Northwest
- **South Holland** — dense urban subset

## Project Structure

```
UrbanRepML/
├── stage1_modalities/   # Modality encoders (AlphaEarth, POI, Roads, Aerial)
├── stage2_fusion/       # Fusion models, data loading, graph construction, training
├── stage3_analysis/     # Linear probes, clustering, visualization
├── scripts/             # Processing & training scripts
├── specs/               # Architecture decision documents
├── tests/               # Import smoke tests & H3 compliance
├── data/                # Study-area organized data (not in repo)
└── CLAUDE.md            # Developer principles & architecture details
```

## Why Late-Fusion?

Development is hard. Wrangling parallel datasets during training is challenging. Late-fusion makes it manageable — process modalities one at a time, iterate independently, fuse when ready.

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- [SRAI](https://github.com/kraina-ai/srai) — Spatial Representations for AI
- [H3](https://h3geo.org/) — hexagonal hierarchical geospatial indexing (via SRAI)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — graph neural networks
- [Leefbaarometer](https://www.leefbaarometer.nl/) — Dutch livability scoring (target data)
