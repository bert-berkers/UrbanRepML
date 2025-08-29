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
python -m urban_embedding --study-area south_holland --modalities alphaearth,poi,gtfs

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

| Modality | Description | Data Source | Output Format |
|----------|-------------|-------------|---------------|
| **AlphaEarth** | Deep learning satellite embeddings | 10m resolution imagery | 64-dim feature vectors |
| **POI** | Points of interest patterns | OpenStreetMap, Overture | Categorical embeddings |
| **GTFS** | Public transit accessibility | Transit agencies | Travel time matrices |
| **Roads** | Street network topology | OpenStreetMap | Graph embeddings |
| **Buildings** | Built environment density | OpenStreetMap, cadastral | Morphology metrics |
| **StreetView** | Ground-level visual features | Mapillary, KartaView | Scene embeddings |

## 🗺️ Study Areas

The framework includes pre-configured study areas with research-specific parameters:

### Cascadia Bioregion
- **Focus**: Coastal temperate rainforest ecosystems
- **Extent**: Pacific Northwest (west of -121° longitude)
- **Resolution**: H3 level 8-10
- **Applications**: Forest fragmentation, urban-wildland interface

### Netherlands Urban Regions
- **Focus**: High-density sustainable urbanism
- **Regions**: South Holland, Utrecht, Amsterdam metropolitan
- **Resolution**: H3 level 8-11
- **Applications**: Cycling infrastructure, compact development patterns

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

### Processing AlphaEarth Satellite Embeddings
```python
from modalities import load_modality_processor

processor = load_modality_processor('alphaearth', {
    'source_dir': 'path/to/alphaearth/tiles',
    'max_workers': 10  # Parallel processing threads
})

embeddings = processor.run_pipeline(
    study_area='cascadia',
    h3_resolution=8,
    output_dir='data/processed/embeddings/alphaearth'
)
```

### Multi-Modal Urban Learning Pipeline
```python
from urban_embedding import UrbanEmbeddingPipeline

config = {
    'study_area': 'south_holland',
    'modalities': ['alphaearth', 'poi', 'gtfs', 'roads'],
    'h3_resolution': 9,
    'model': {
        'architecture': 'UrbanUNet',
        'hidden_dim': 128,
        'num_layers': 3
    },
    'training': {
        'epochs': 200,
        'batch_size': 32,
        'learning_rate': 0.001
    }
}

pipeline = UrbanEmbeddingPipeline(config)
results = pipeline.run()

# Access learned representations
embeddings = results['embeddings']  # H3 → feature vectors
clusters = results['clusters']      # Spatial clustering results
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

- AlphaEarth satellite embeddings from Scale AI
- H3 hexagonal indexing system by Uber
- SRAI spatial analysis framework
- OpenStreetMap contributors worldwide

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/bertberkers/UrbanRepML/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bertberkers/UrbanRepML/discussions)
- **Email**: bert.berkers@example.com

---

*UrbanRepML is actively developed for advancing urban sustainability and geospatial intelligence research.*