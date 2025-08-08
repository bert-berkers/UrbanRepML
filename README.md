# UrbanRepML 🏙️

**Multi-scale urban representation learning using Graph Neural Networks**

UrbanRepML learns meaningful representations of urban areas by integrating multiple data sources (transit, roads, buildings, points of interest) across different spatial scales. Using a novel U-Net inspired GNN architecture, it captures how cities function at walking, cycling, and driving scales simultaneously.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/UrbanRepML.git
cd UrbanRepML

# Install the package
pip install -e .
```

### Run Your First Experiment

```bash
# Run South Holland with 95th percentile density filtering
python run_south_holland_fsi95.py
```

This will:
1. Setup H3 hexagonal regions for South Holland
2. Calculate building density from OpenStreetMap data
3. Filter to the densest 5% of urban areas
4. Generate multi-modal accessibility graphs
5. Train the UrbanUNet model
6. Output learned embeddings and cluster visualizations

## 📊 What Does UrbanRepML Do?

### Input
- **Multi-modal urban embeddings**: Pre-computed representations from GTFS (transit), aerial imagery, POIs, and road networks
- **Building density data**: FSI (Floor Space Index) from OpenStreetMap or government sources
- **Street networks**: From OpenStreetMap for accessibility calculations

### Process
1. **Hierarchical regionalization**: Creates multi-scale H3 hexagonal grids
2. **Urban filtering**: Focuses on dense urban areas using FSI thresholds
3. **Accessibility graphs**: Models how areas connect via walking, cycling, and driving
4. **Representation learning**: Trains a GNN to learn urban embeddings
5. **Analysis**: Clusters and visualizes learned representations

### Output
- **Urban embeddings**: Dense vector representations of urban areas
- **Cluster maps**: Spatial visualization of urban typologies
- **Accessibility networks**: Travel-time based urban connectivity graphs

## 🎯 Use Cases

- **Urban Planning**: Identify similar neighborhoods across cities
- **Transport Analysis**: Understand multi-modal accessibility patterns
- **Real Estate**: Characterize location types for valuation models
- **Policy Making**: Compare urban development patterns
- **Research**: Study urban morphology and function

## 📁 Project Structure

```
UrbanRepML/
├── scripts/                 # Executable scripts
│   ├── preprocessing/       # Data preparation pipeline
│   └── experiments/         # Experiment orchestration
├── urban_embedding/         # Core package
├── experiments/            # Experiment outputs
├── data/                   # Input data
└── docs/                   # Documentation
    ├── ARCHITECTURE.md     # Technical details
    └── CONFIG_GUIDE.md     # Configuration reference
```

## 🔧 Custom Experiments

### Basic Usage

```python
from urban_embedding import UrbanEmbeddingPipeline

# Configure your experiment
config = UrbanEmbeddingPipeline.create_default_config(
    city_name="amsterdam",
    threshold=90  # Top 10% densest areas
)

# Run the pipeline
pipeline = UrbanEmbeddingPipeline(config)
embeddings = pipeline.run()
```

### Advanced: Run Complete Experiment Pipeline

```bash
python scripts/experiments/run_experiment.py \
  --experiment_name amsterdam_dense \
  --city amsterdam \
  --fsi_percentile 90 \
  --run_training \
  --epochs 200
```

### Step-by-Step Data Preparation

```bash
# 1. Create H3 regions
python scripts/preprocessing/setup_regions.py \
  --city_name amsterdam \
  --resolutions 8,9,10

# 2. Calculate building density
python scripts/preprocessing/setup_density.py \
  --city_name amsterdam \
  --building_data path/to/buildings.shp

# 3. Filter by density
python scripts/preprocessing/setup_fsi_filter.py \
  --city_name amsterdam \
  --fsi_percentile 90 \
  --output_dir experiments/amsterdam_dense/data

# 4. Generate accessibility graphs
python scripts/preprocessing/setup_hierarchical_graphs.py \
  --data_dir experiments/amsterdam_dense/data \
  --output_dir experiments/amsterdam_dense/graphs
```

## 📊 Key Concepts

### H3 Resolution Strategy
- **Resolution 8**: City-scale, driving accessibility (~450m hexagons)
- **Resolution 9**: Neighborhood-scale, cycling accessibility (~170m hexagons)
- **Resolution 10**: Block-scale, walking accessibility (~65m hexagons)

### FSI (Floor Space Index) Filtering
- Focuses analysis on urban areas with significant built density
- Can use percentile (e.g., top 5%) or absolute threshold (e.g., FSI ≥ 0.1)
- Hierarchical: selecting a parent hexagon includes all its children

### Multi-Modal Integration
- **GTFS**: Public transit accessibility
- **Aerial**: Land use and building patterns
- **POI**: Urban amenities and services
- **Roads**: Street network structure

## 🛠️ Configuration

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for detailed parameter documentation.

### Quick Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `fsi_percentile` | Urban density threshold | 95 |
| `resolutions` | H3 levels to analyze | [8,9,10] |
| `epochs` | Training iterations | 100 |
| `hidden_dim` | Model capacity | 128 |

## 📚 Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical system design
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Configuration parameters
- [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) - Development history
- [CLAUDE.md](CLAUDE.md) - AI assistant context

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines (coming soon).

## 📖 Citation

If you use UrbanRepML in your research, please cite:

```bibtex
@software{urbanrepml2025,
  title = {UrbanRepML: Multi-scale Urban Representation Learning},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/UrbanRepML}
}
```

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- H3 spatial indexing by Uber
- OpenStreetMap contributors
- PyTorch Geometric team

---

**Need help?** Check the [documentation](ARCHITECTURE.md) or open an issue!