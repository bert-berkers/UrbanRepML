# UrbanRepML

A Python package for urban representation learning with multi-level analysis.

## Installation

```bash
pip install -e .
```

## Features

- Multi-scale urban graph construction
- Feature processing and dimensionality reduction
- Urban embedding generation using Graph Neural Networks
- Analysis and visualization tools

## Usage

```python
from urban_embedding import UrbanEmbeddingPipeline

# Create default configuration
config = UrbanEmbeddingPipeline.create_default_config()

# Initialize pipeline
pipeline = UrbanEmbeddingPipeline(config)

# Run the pipeline
embeddings = pipeline.run(config)
```

## License

MIT License