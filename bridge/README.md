# Bridge Infrastructure

## Overview

The `bridge/` directory contains components designed to facilitate the integration of the `UrbanRepML` framework with external systems, primarily the `GEO-INFER` project. It provides the necessary tools for data conversion, model wrapping, and advanced, experimental processing pipelines.

This infrastructure allows `UrbanRepML` to act not just as a standalone representation learning system, but also as a powerful feature engineering and processing frontend for other analytical and inference frameworks.

## Components

### `data_exchange.py`
- **Purpose**: Data format conversion between `UrbanRepML` and `GEO-INFER`.
- **Key Functions**:
  - `urbanreml_to_geoinfer()`: Converts `UrbanRepML`'s H3-indexed DataFrames into a dictionary format compatible with `GEO-INFER`.
  - `geoinfer_to_urbanreml()`: Converts `GEO-INFER` data back into `UrbanRepML`'s DataFrame or GeoDataFrame format.
  - `h3_data_bridge()`: Prepares `UrbanRepML` data for specific `GEO-INFER` modules like agricultural or climate analysis.

### `model_integration.py`
- **Purpose**: Provides wrappers and adapters to combine models and data from both frameworks.
- **Key Functions**:
  - `combine_embeddings()`: Merges embeddings from `UrbanRepML` with features from `GEO-INFER` using methods like concatenation or averaging.
  - `active_inference_wrapper()`: Wraps an `UrbanRepML` model to operate within an active inference loop, calculating prediction errors and updating beliefs.
  - `multi_resolution_adapter()`: Adapts `UrbanRepML`'s multi-resolution outputs to a single target resolution required by `GEO-INFER`.

### `padic_mdp_scheduler.py`
- **Purpose**: An advanced, experimental task scheduler for distributed processing.
- **Framework**: This is a highly theoretical component that implements a Markov Decision Process (MDP) operating in p-adic time. It uses concepts from number theory and reinforcement learning to schedule tasks (like TIFF file processing) across multiple temporal scales simultaneously. This is intended for optimizing resource allocation in large-scale, parallel processing environments.

### `rxinfer_server.py`
- **Purpose**: A `FastAPI` server for providing real-time, reactive, GPU-accelerated H3 processing.
- **Features**:
  - Implements reactive programming patterns (`RxInfer`) for streaming data processing.
  - Uses active inference principles for spatial predictions.
  - Can use different backend processors (`SRAI` or `PyTorch`-based).
  - Provides results via WebSockets and Server-Sent Events for real-time applications.

## Usage

These components are generally not used in the standard `UrbanRepML` data processing pipeline. They are intended for advanced use cases involving integration with `GEO-INFER` or for experimental, high-performance processing setups.

```python
# Example of using the data exchange
from bridge.data_exchange import urbanreml_to_geoinfer
import pandas as pd

# Assume you have a DataFrame of UrbanRepML embeddings
urbanreml_df = pd.read_parquet("path/to/embeddings.parquet")

# Convert to GEO-INFER format
geoinfer_payload = urbanreml_to_geoinfer(
    "path/to/embeddings.parquet",
    data_type="h3_embeddings",
    resolution=8
)

# Now `geoinfer_payload` can be sent to a GEO-INFER service
```
