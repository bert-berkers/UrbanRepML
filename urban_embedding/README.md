# `urban_embedding` Core Library

This directory contains the core Python package for the `UrbanRepML` project. It is a modular and extensible library for learning multi-scale urban representations from geospatial data.

## Core Functionality

The library is built around a central `UrbanEmbeddingPipeline` that orchestrates the entire workflow. The key modules are:

- **`pipeline.py`**: The main entry point and orchestrator. It manages the configuration and runs the end-to-end process of data loading, preprocessing, model training, and analysis.

- **`feature_processing.py`**: Handles the loading, validation, and processing of multi-modal embeddings (e.g., from aerial imagery, points of interest, transit networks). It uses PCA for dimensionality reduction and prepares features for the model.

- **`graph_construction.py`**: Constructs the hierarchical spatial graphs that form the backbone of the model. It calculates accessibility networks based on different travel modes (walking, biking, driving) using OpenStreetMap data.

- **`model.py`**: Defines the `UrbanUNet`, a U-Net-inspired Graph Neural Network. This model simultaneously processes data at multiple H3 resolutions, using skip connections to share information across scales.

- **`analytics.py`**: Contains tools for analyzing and visualizing the learned embeddings. This includes clustering algorithms (K-means), dimensionality reduction for visualization (UMAP), and plotting utilities.

- **`study_area_filter.py`**: Provides functionality for filtering the study region based on data-driven metrics like Floor Space Index (FSI), allowing experiments to focus on the most relevant or dense urban areas.

## Key Concepts

- **Multi-Scale H3 Grids**: The library uses Uber's H3 hexagonal grid system to represent space at multiple resolutions. This allows the model to learn patterns at different scales, from individual blocks (res 10) to neighborhoods (res 9) and districts (res 8).

- **Hierarchical GNN**: The `UrbanUNet` model is designed to explicitly model the hierarchical relationship between different H3 resolutions, allowing it to learn how local patterns aggregate into larger-scale urban structures.

## Basic Usage

While the main way to use the library is through the experiment scripts, you can also interact with the pipeline directly:

```python
from urban_embedding.pipeline import UrbanEmbeddingPipeline

# Configure and run a new pipeline
config = UrbanEmbeddingPipeline.create_default_config(
    city_name="amsterdam",
    threshold=90  # Filter to top 10% densest areas
)
pipeline = UrbanEmbeddingPipeline(config)
embeddings = pipeline.run()

print("Learned embeddings shape:", embeddings.shape)
```

For more technical details on the architecture, please refer to the main `ARCHITECTURE.md` file in the root directory.
