# UrbanRepML [under heavy construction]

For philosophical background see https://www.youtube.com/watch?v=UYD8CR_Xorg&ab_channel=ActiveInferenceInstitute

**High-Quality Urban Embeddings through Manageable Late-Fusion**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SRAI](https://img.shields.io/badge/Spatial-SRAI-green)](https://github.com/kraina-ai/srai)

Learn dense urban representations by processing modalities one at a time, then fusing with spatial awareness. Built on **SRAI** (not raw H3!) for all spatial operations.

## ⚠️ Important: SRAI for H3 Operations

**This project uses SRAI (Spatial Representations for AI) for ALL H3 hexagon operations.**
- We do NOT use h3-py directly
- SRAI provides H3 functionality with additional spatial analysis tools
- All regionalizers and neighborhoods use SRAI's implementations

## 🎯 Core Goal

Create high-quality urban embeddings capable of reconstructing urban environments. The ultimate aim is to enable aerial image generation from learned representations, that way we can paint new developments with embeddings as our paint and hexagonally indiced regions as our canvas. We focus on manageable development through late-fusion because handling multiple parallel datasets is hard.

## 🌍 Study Areas

All work is organized by study areas. We process multiple regions:

- **Netherlands**: Primary area with complete coverage for training volume
- **Cascadia**: Urban-forest interface in Pacific Northwest
- **South Holland**: Dense urban subset for detailed analysis
- Additional areas configurable in `study_areas/configs/`

Each study area maintains a consistent structure:
```
data/study_areas/{area_name}/
├── area_gdf/           # Study area boundary
├── regions_gdf/        # H3 regions via SRAI
├── embeddings/         # Per-modality embeddings
├── urban_embedding/    # Fused results
└── plots/              # Visualizations
```

## 🏗️ Architecture

**Two-stage late-fusion pipeline:**

### Stage 1: Individual Modality Encoders
Process one modality at a time (manageable development!):
- **AlphaEarth**: Google Earth Engine embeddings (primary visual features)
- **POI**: OpenStreetMap points via SRAI → urban function indicators
- **Roads**: OSM networks via SRAI → connectivity structure
- **GTFS**: Transit data → accessibility potential
- **Aerial Imagery**: PDOK Netherlands → DINOv3 (optional)

### Stage 2: Urban Embedding Fusion
Graph Convolutional U-Net with spatial constraints:
- Concatenated modality embeddings per H3 cell
- Accessibility-based graph pruning (SRAI neighborhoods)
- Multi-resolution H3 hierarchy (levels 5-11 via SRAI)
- Floodfill travel time with gravity weighting

## 💡 Why Late-Fusion?

**Honest answer**: Development is hard. Processing modalities separately is manageable. Wrangling multiple parallel datasets during training is challenging - late fusion lets us tackle one thing at a time while maintaining compartmentalized prototyping.

## 📊 Modality Status

| Modality | Status | Purpose | Uses SRAI |
|----------|--------|---------|-----------|
| AlphaEarth | ✅ Working | Primary visual features | For H3 indexing |
| POI | 🚧 Partial | Urban function indicators | Yes - regions & aggregation |
| Roads | 🚧 Partial | Connectivity structure | Yes - network analysis |
| GTFS | 📋 Planned | Transit accessibility | Yes - accessibility zones |
| Aerial | 🔧 Optional | PDOK imagery → DINOv3 | For H3 indexing |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/bertberkers/UrbanRepML.git
cd UrbanRepML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e .

# CRITICAL: Install SRAI with all components
pip install srai[all]
```

### Basic Usage

```python
from srai.regionalizers import H3Regionalizer  # NOT import h3!
from modalities.alphaearth import AlphaEarthProcessor
from urban_embedding.pipeline import UrbanEmbeddingPipeline

# Define study area with SRAI
regionalizer = H3Regionalizer(resolution=9)
regions_gdf = regionalizer.transform(area_gdf)

# Process individual modality
processor = AlphaEarthProcessor(config)
alphaearth_embeddings = processor.process_to_h3(data, regions_gdf)

# Run fusion pipeline
pipeline = UrbanEmbeddingPipeline(config)
urban_embeddings = pipeline.run(
    study_area='netherlands',
    modalities=['alphaearth', 'poi', 'roads']
)
```

### Command Line

```bash
# Process modalities for study area (uses SRAI)
python -m modalities.alphaearth --study-area netherlands

# Run fusion pipeline
python -m urban_embedding.pipeline \
    --study-area netherlands \
    --modalities alphaearth,poi,roads

# Generate accessibility graphs (via SRAI)
python scripts/accessibility/generate_graphs.py --study-area netherlands

# Analyze results
python -m urban_embedding.analytics --study-area netherlands
```

## 🔑 Key Innovation

**Accessibility-Pruned Graphs**: We use floodfill travel time calculation with gravity weighting (building density) to create sparse, meaningful spatial constraints for the GCN layers. This is implemented using SRAI's neighborhood analysis tools, not raw H3.

## 📁 Project Structure

```
UrbanRepML/
├── modalities/          # Stage 1: Individual encoders
├── urban_embedding/     # Stage 2: Fusion pipeline
├── study_areas/         # Area configurations
├── scripts/             # Utilities & preprocessing
├── data/                # Study-area organized data
└── docs/                # Documentation
```

## 📚 Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture
- [CLAUDE.md](CLAUDE.md) - Developer principles & SRAI usage
- [docs/](docs/) - Additional documentation

## 🤝 Contributing

We welcome contributions! Please ensure:
1. All H3 operations use SRAI, not h3-py
2. Code is study-area based
3. Late-fusion architecture is maintained
4. Documentation is updated

## 📜 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [SRAI](https://github.com/kraina-ai/srai) - Spatial Representations for AI
- [H3](https://h3geo.org/) - Hexagonal hierarchical geospatial indexing (via SRAI)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural networks

---

**Remember**: We use SRAI for all H3 operations. Development is hard - that's why we chose late-fusion for manageable, compartmentalized development.
