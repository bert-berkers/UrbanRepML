# UrbanRepML

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Multi-Modal Urban Representation Learning for Geospatial Intelligence**

UrbanRepML is a scalable framework for processing and fusing multiple urban data modalities (satellite imagery, POIs, transit networks, road topology, building footprints, street-level imagery) into unified H3 hexagon-based representations for advanced urban analysis and machine learning.

## 🎯 Key Features

- **Multi-Modal Data Fusion**: Seamlessly integrate diverse urban data sources
- **H3 Hexagon Standard**: Uniform spatial representation across all modalities
- **Modular Architecture**: Plug-and-play data processors following common interfaces
- **Graph Neural Networks**: State-of-the-art UrbanUNet architecture for spatial learning
- **Scalable Processing**: Efficient handling of city to regional-scale datasets
- **Bioregional Analysis**: Built-in support for ecological and urban system studies

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bertberkers/UrbanRepML.git
cd UrbanRepML

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install optional dependencies for visualization
pip install -e ".[viz]"
```

### Basic Usage

```bash
# Process satellite imagery for a study area
python -m modalities.alphaearth --study-area cascadia --resolution 8

# Run multi-modal urban embedding pipeline
python -m urban_embedding --study-area south_holland --modalities alphaearth,poi,roads

# Analyze results with clustering
python -m urban_embedding.analytics --study-area netherlands --method clustering
```

## 🏗️ Architecture

```
UrbanRepML/
├── modalities/           # Data modality processors
│   ├── alphaearth/      # Satellite imagery → embeddings
│   ├── poi/             # Points of interest → categorical features
│   ├── gtfs/            # Transit data → accessibility metrics
│   ├── roads/           # OSM networks → graph embeddings
│   ├── buildings/       # Footprints → density features
│   └── streetview/      # Street imagery → visual embeddings
│
├── urban_embedding/     # ML pipeline
│   ├── pipeline.py      # Multi-modal fusion orchestrator
│   ├── model.py         # UrbanUNet GNN architecture
│   └── analytics.py     # Clustering & visualization tools
│
├── study_areas/         # Geographic research areas
│   ├── configs/         # YAML boundary definitions
│   ├── cascadia/        # Pacific Northwest coastal forests
│   └── netherlands/     # Dense urban regions analysis
│
└── data/                # Unified data storage
    ├── raw/             # Original source data
    └── processed/       # H3 embeddings & networks
```

## 📊 Supported Data Modalities

| Modality | Status | Description | Data Source | Output Format |
|----------|--------|-------------|-------------|---------------|
| **AlphaEarth** | ✅ **Complete** | Deep learning satellite embeddings | Google Earth Engine | 64-dim feature vectors |
| **Semantic Segmentation** | ✅ **Complete** | AlphaEarth + DINOv3 fusion | Satellite + aerial imagery | Categorical land cover classes |
| **Aerial Imagery** | ✅ **Complete** | High-res DINOv3 encoding | PDOK (Netherlands) | 768-dim visual features |
| **Buildings** | ✅ **Complete** | Building density (FSI) analysis | OpenStreetMap, cadastral | Density and morphology metrics |
| **POI** | ✅ **Complete** | POI counts, diversity, and Hex2Vec embeddings | OpenStreetMap | Count & contextual embeddings |
| **Roads** | ✅ **Complete** | Road network topology and centrality metrics | OpenStreetMap | Connectivity & graph metrics |
| **GTFS** | 🚧 *Planned* | Public transit accessibility | Transit agencies | Travel time matrices |
| **StreetView** | 🚧 *Planned* | Ground-level visual features | Mapillary, KartaView | Scene embeddings |

## 🗺️ Study Areas

The framework includes pre-configured study areas with research-specific parameters:

### Cascadia Bioregion
- **Focus**: Multi-year agricultural and forest analysis, GEO-INFER integration
- **Extent**: 52 counties (Northern CA + Oregon), ~421,000 km²
- **Resolution**: H3 levels 10-5 (6-scale hierarchy)
- **Temporal**: Multi-year AlphaEarth data (2017-2024)
- **Applications**: Agricultural patterns, forest-urban interface, synthetic data generation

### Netherlands Urban Systems
- **Focus**: High-density urbanism with multiple analysis variants
- **Regions**: South Holland with FSI filtering (0.1, 95%, 99% thresholds)
- **Resolution**: H3 levels 10-5 (full hierarchy)
- **Data**: Building density (FSI), accessibility networks, PDOK aerial imagery
- **Applications**: Urban density analysis, cycling infrastructure, compact development

### Custom Study Areas
```python
from study_areas.tools import create_study_area

create_study_area(
    name='your_region',
    bounds=(-122.8, 45.2, -122.4, 45.7),  # xmin, ymin, xmax, ymax
    description='Your study area description'
)
```

## 💻 Advanced Examples

### AlphaEarth + Semantic Segmentation Processing
```python
from modalities.semantic_segmentation import SemanticSegmentationProcessor

config = {
    'study_area': 'cascadia',
    'model_config': {
        'alphaearth_dim': 64,
        'dinov3_dim': 768,
        'conditioning_dim': 256
    }
}

processor = SemanticSegmentationProcessor(config)
results = processor.run_pipeline(
    study_area='cascadia',
    h3_resolution=10,
    output_dir='data/processed/embeddings/semantic_segmentation'
)
```

### Renormalizing Multi-Scale Architecture
```python
from urban_embedding import RenormalizingUrbanPipeline, create_renormalizing_config_preset

# Create configuration for 7-level hierarchy
config = create_renormalizing_config_preset("default")
config['city_name'] = 'south_holland'

# Initialize renormalizing pipeline
pipeline = RenormalizingUrbanPipeline(config)
embeddings_by_resolution = pipeline.run()

# Access multi-resolution embeddings
for resolution in [10, 9, 8, 7, 6, 5]:
    print(f"Resolution {resolution}: {embeddings_by_resolution[resolution].shape}")
```

### Standard Multi-Modal Pipeline
```python
from urban_embedding import UrbanEmbeddingPipeline

config = {
    'city_name': 'south_holland',
    'model': {
        'hidden_dim': 128,
        'output_dim': 32,
        'num_convs': 4
    },
    'training': {
        'num_epochs': 500,
        'learning_rate': 1e-4
    }
}

pipeline = UrbanEmbeddingPipeline(config)
results = pipeline.run()

# Results include embeddings at multiple resolutions
embeddings = results['embeddings']
clusters = results['clusters']
```

### Visualizing Results
```python
from urban_embedding.analytics import UrbanAnalytics

analytics = UrbanAnalytics(study_area='netherlands')
analytics.plot_clusters(
    embeddings_path='results/embeddings.parquet',
    output_path='visualizations/clusters.html',
    interactive=True
)
```

## 📋 Requirements

- **Python**: 3.8 or higher
- **Core Dependencies**:
  - PyTorch ≥ 2.0.0 & PyTorch Geometric
  - GeoPandas, H3, SRAI (spatial analysis)
  - rasterio, rioxarray (raster processing)
  - pandas, numpy, scikit-learn
- **Optional**:
  - wandb (experiment tracking)
  - folium, plotly (interactive visualizations)

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository and create a feature branch
2. Follow the `ModalityProcessor` interface for new data sources
3. Add tests for new functionality
4. Update documentation and examples
5. Submit a pull request with a clear description

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black urban_embedding/ modalities/

# Lint code
flake8 urban_embedding/ modalities/
```

## 📖 Documentation

- **[Architecture Overview](ARCHITECTURE.md)**: System design and components
- **[Configuration Guide](CONFIG_GUIDE.md)**: Detailed configuration options
- **[Developer Instructions](CLAUDE.md)**: Development practices and guidelines
- **[API Reference](docs/api/)**: Complete API documentation

## 📝 Citation

If you use UrbanRepML in your research, please cite:

```bibtex
@software{urbanrepml2025,
  title = {UrbanRepML: Multi-Modal Urban Representation Learning},
  author = {Berkers, Bert},
  year = {2025},
  url = {https://github.com/bertberkers/UrbanRepML}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AlphaEarth satellite embeddings from Earth Engine by Google
- H3 hexagonal indexing system by Uber
- SRAI spatial analysis framework
- OpenStreetMap contributors worldwide
---
