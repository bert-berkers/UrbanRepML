# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation
```bash
pip install -e .
```

### Running Tests
```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python tests/test_pipeline.py
```

### Running the Pipeline
```python
from urban_embedding import UrbanEmbeddingPipeline

# Create configuration
config = UrbanEmbeddingPipeline.create_default_config(
    city_name="south_holland",
    threshold=50  # Optional: create threshold-filtered variant
)

# Initialize and run pipeline
pipeline = UrbanEmbeddingPipeline(config)
embeddings = pipeline.run()
```

## Architecture Overview

### Multi-Level Urban Analysis System
This project implements a Graph Neural Network-based approach for learning urban representations at multiple spatial resolutions using H3 hexagonal grids.

### Core Components

1. **Pipeline System** (`urban_embedding/pipeline.py`):
   - Orchestrates the entire workflow
   - Handles threshold-based filtering of urban areas
   - Manages data loading, processing, training, and visualization

2. **Feature Processing** (`urban_embedding/feature_processing.py`):
   - Processes multimodal urban data (GTFS, road networks, aerial imagery, POIs)
   - Applies PCA for dimensionality reduction
   - Handles cross-scale feature mapping between H3 resolutions

3. **Graph Construction** (`urban_embedding/graph_construction.py`):
   - Builds spatial graphs based on travel time accessibility
   - Supports multiple travel modes (walk, bike, drive)
   - Uses exponential decay for edge weights based on travel time

4. **Model Architecture** (`urban_embedding/model.py`):
   - Implements UrbanUNet - a U-Net style GNN architecture
   - Processes features at resolutions 8, 9, and 10 simultaneously
   - Uses both reconstruction and consistency losses

5. **Analytics** (`urban_embedding/analytics.py`):
   - Handles embedding visualization
   - Performs clustering analysis
   - Saves results in multiple formats

### Key Design Patterns

- **Multi-Resolution Processing**: The system works with H3 resolutions 8, 9, and 10, where each resolution corresponds to different travel modes (drive, bike, walk)
- **Caching Strategy**: Extensive caching of processed graphs, PCA models, and network data to speed up repeated runs
- **Modular Architecture**: Each component is self-contained with clear interfaces
- **Device-Agnostic**: Automatically uses GPU if available, falls back to CPU

### Data Flow

1. **Preprocessing**: Raw urban data → H3 hexagonal regions with building density
2. **Feature Extraction**: Multimodal embeddings → PCA-reduced features
3. **Graph Building**: Spatial accessibility calculations → Weighted graphs
4. **Model Training**: Multi-resolution GNN training → Urban embeddings
5. **Analysis**: Embeddings → Clustering and visualization

### Important Configurations

- **Thresholds**: Building density thresholds (50%, 70%, 80%, 90%) filter urban areas
- **Travel Parameters**: Speeds, max travel times, and search radii for each mode
- **Model Parameters**: Hidden dimensions, number of convolutions, learning rates
- **Loss Weights**: Balance between reconstruction and cross-scale consistency

### Dependencies

Core libraries: PyTorch, PyTorch Geometric, GeoPandas, H3, OSMnx, scikit-learn, WandB