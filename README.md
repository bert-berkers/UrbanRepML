# UrbanRepML

For philosophical background and preliminary results of fused spatial U-NET embeddings see https://www.youtube.com/watch?v=UYD8CR_Xorg&ab_channel=ActiveInferenceInstitute

**High-Quality Urban Embeddings through Manageable Late-Fusion**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SRAI](https://img.shields.io/badge/Spatial-SRAI-green)](https://github.com/kraina-ai/srai)

Learn dense urban representations by processing modalities one at a time, then fusing with spatial awareness. Built on **SRAI** (not raw H3!) for all spatial operations.

## Important: SRAI for H3 Operations

This project uses SRAI (Spatial Representations for AI) for ALL H3 hexagon operations. We do NOT use h3-py directly.

## Core Goal

Create high-quality urban embeddings capable of reconstructing urban environments. The ultimate aim is to enable aerial image generation from learned representations — paint new developments with embeddings as our paint and hexagonally indexed regions as our canvas.

## Architecture

**Three-stage pipeline:**

### Stage 1: Individual Modality Encoders
- **AlphaEarth**: Google Earth Engine embeddings (primary, working)
- **POI**: OpenStreetMap points via SRAI (partial)
- **Roads**: OSM networks via SRAI (partial)
- **GTFS**: Transit data (planned)

### Stage 2: Urban Embedding Fusion
- **FullAreaUNet**: Full study area U-Net with lateral accessibility graph
- **ConeBatchingUNet**: Cone-based hierarchical U-Net (res5→res10, most promising)
- **AccessibilityUNet**: Planned — Hanssen's gravity model variant

### Stage 3: Analysis & Visualization
- **UrbanEmbeddingAnalyzer**: Cluster visualization and statistics
- **HierarchicalClusterAnalyzer**: Multi-scale clustering across H3 resolutions
- **HierarchicalLandscapeVisualizer**: Beautiful multi-resolution plots

## Quick Start

```bash
git clone https://github.com/bertberkers/UrbanRepML.git
cd UrbanRepML
uv sync              # Install all dependencies
uv sync --extra dev  # Include dev tools
```

```python
from srai.regionalizers import H3Regionalizer  # NOT import h3!
from stage1_modalities.alphaearth import AlphaEarthProcessor

regionalizer = H3Regionalizer(resolution=9)
regions_gdf = regionalizer.transform(area_gdf)
```

## Study Areas

All work is organized by study areas:
- **Netherlands**: Primary area with complete coverage
- **Cascadia**: Urban-forest interface in Pacific Northwest
- **South Holland**: Dense urban subset

```
data/study_areas/{area_name}/
├── area_gdf/           # Study area boundary
├── regions_gdf/        # H3 regions via SRAI
├── embeddings/         # Per-modality embeddings (Stage 1)
├── urban_embedding/    # Fused results (Stage 2)
├── analysis/           # Cluster assignments (Stage 3)
└── plots/              # Visualizations (Stage 3)
```

## Project Structure

```
UrbanRepML/
├── stage1_modalities/   # Stage 1: Individual encoders
├── stage2_fusion/       # Stage 2: Fusion pipeline & models
├── stage3_analysis/     # Stage 3: Analysis & visualization
├── study_areas/         # Area configurations
├── scripts/             # Processing & training scripts
├── data/                # Study-area organized data
├── CLAUDE.md            # Developer principles & architecture details
└── README.md            # This file
```

## Why Late-Fusion?

Development is hard. Processing modalities separately is manageable. Late-fusion lets us tackle one thing at a time while maintaining compartmentalized prototyping.

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- [SRAI](https://github.com/kraina-ai/srai) - Spatial Representations for AI
- [H3](https://h3geo.org/) - Hexagonal hierarchical geospatial indexing (via SRAI)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural networks
