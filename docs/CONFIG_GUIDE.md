# UrbanRepML Configuration Guide

This guide documents all configuration parameters for UrbanRepML experiments. Parameters are organized by their role in the pipeline.

## 📋 Table of Contents
1. [Experiment Configuration](#experiment-configuration)
2. [Data Selection Parameters](#data-selection-parameters)
3. [Graph Construction Parameters](#graph-construction-parameters)
4. [Feature Processing Parameters](#feature-processing-parameters)
5. [Model Architecture Parameters](#model-architecture-parameters)
6. [Training Parameters](#training-parameters)
7. [Analysis Parameters](#analysis-parameters)
8. [Example Configurations](#example-configurations)

---

## Experiment Configuration

### Basic Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | required | Unique identifier for the experiment |
| `city` | str | "south_holland" | City or region to analyze |
| `resolutions` | list | [10,9,8,7,6,5] | H3 resolutions to process (full hierarchy) |

### File Paths
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `building_data` | str | "data/preprocessed/density/PV28__00_Basis_Bouwblok.shp" | Path to building shapefile |
| `project_dir` | str | Current directory | Root directory of the project |

---

## Data Selection Parameters

### FSI (Floor Space Index) Filtering

Choose **one** of these filtering methods:

| Parameter | Type | Range | Description | Example |
|-----------|------|-------|-------------|---------|
| `fsi_percentile` | float | 0-100 | Keep top X percentile of dense areas | 95 = top 5% densest |
| `fsi_threshold` | float | 0-10+ | Absolute FSI minimum value | 0.1 = FSI ≥ 0.1 |

### Hierarchical Filtering
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_resolution` | int | 8 | Resolution for primary filtering |
| `hierarchical` | bool | True | Include all children of selected parents |

**How it works:**
1. Filter hexagons at `base_resolution` by FSI
2. Automatically include ALL children at higher resolutions
3. Example: If res-8 hex selected → all its res-9 and res-10 children included

---

## Graph Construction Parameters

### Travel Speeds
| Mode | Parameter | Default (m/s) | Real-world equivalent |
|------|-----------|---------------|---------------------|
| Walk | `speeds.walk` | 1.4 | 5 km/h |
| Bike | `speeds.bike` | 4.17 | 15 km/h |
| Drive | `speeds.drive` | 11.11 | 40 km/h |

### Maximum Travel Time
| Mode | Parameter | Default (seconds) | Description |
|------|-----------|-------------------|-------------|
| Walk | `max_travel_time.walk` | 900 | 15 minutes |
| Bike | `max_travel_time.bike` | 900 | 15 minutes |
| Drive | `max_travel_time.drive` | 900 | 15 minutes |

### Search Radius
| Mode | Parameter | Default (meters) | Description |
|------|-----------|------------------|-------------|
| Walk | `search_radius.walk` | 1200 | Max Euclidean distance |
| Bike | `search_radius.bike` | 3000 | Max Euclidean distance |
| Drive | `search_radius.drive` | 10000 | Max Euclidean distance |

### Distance Decay (Beta)
| Mode | Parameter | Default | Effect |
|------|-----------|---------|--------|
| Walk | `beta.walk` | 0.002 | Higher = faster decay |
| Bike | `beta.bike` | 0.0012 | Higher = shorter range influence |
| Drive | `beta.drive` | 0.0008 | Higher = more local connections |

**Formula:** `accessibility = building_volume * exp(-beta * travel_time)`

### Graph Filtering
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cutoff_time` | int | 300 | Max travel time to consider (seconds) |
| `percentile_threshold` | float | 90 | Keep top X% of edges by weight |
| `graph_fsi_threshold` | float | 0.1 | Min FSI for "active" hexagons |
| `batch_size` | int | 500 | Hexagons per batch in graph computation |

---

## Feature Processing Parameters

### PCA Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variance_threshold` | float | 0.95 | Variance to retain (0-1) |
| `max_components` | int | 100 | Maximum PCA dimensions |

### Modality-Specific Minimums
| Modality | Parameter | Default | Purpose |
|----------|-----------|---------|---------|
| Aerial | `min_components.aerial` | 50 | High-dim for visual complexity |
| POI | `min_components.poi` | 20 | Medium-dim for amenity diversity |
| GTFS | `min_components.gtfs` | 10 | Low-dim for transit patterns |
| Road | `min_components.road` | 10 | Low-dim for network structure |

---

## Model Architecture Parameters

### Core Architecture
| Parameter | Type | Default | Description | Impact |
|-----------|------|---------|-------------|--------|
| `hidden_dim` | int | 128 | GNN hidden layer size | Higher = more capacity |
| `output_dim` | int | 64 | Final embedding dimension | Higher = richer representations |
| `num_convs` | int | 4 | GCN layers per block | Deeper = larger receptive field |
| `dropout` | float | 0.1 | Dropout probability | Higher = more regularization |

### Architecture Variants
| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `architecture` | str | "unet", "simple" | Model architecture type |
| `skip_connections` | bool | True | Use U-Net skip connections |
| `batch_norm` | bool | False | Use batch normalization |
| `activation` | str | "gelu" | Activation function |

---

## Training Parameters

### Optimization
| Parameter | Type | Default | Description | Guidelines |
|-----------|------|---------|-------------|------------|
| `epochs` | int | 100 | Training iterations | Start: 100, Production: 500+ |
| `learning_rate` | float | 0.001 | Initial learning rate | Lower if unstable |
| `weight_decay` | float | 0.0001 | L2 regularization | Higher for simpler model |
| `batch_size` | int | 1 | Graphs per batch | Usually 1 (full graph) |

### Learning Rate Schedule
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scheduler` | str | "cosine" | LR schedule type |
| `warmup_epochs` | int | 10 | Linear warmup period |
| `min_lr` | float | 1e-6 | Minimum learning rate |

### Loss Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_weights.reconstruction` | float | 1.0 | Weight for reconstruction loss |
| `loss_weights.consistency` | float | 3.0 | Weight for cross-scale consistency |

**Tuning guide:**
- Increase consistency weight for better multi-scale alignment
- Increase reconstruction weight for better feature preservation

### Early Stopping
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patience` | int | 20 | Epochs without improvement |
| `min_delta` | float | 0.0001 | Minimum change to qualify as improvement |

---

## Analysis Parameters

### Clustering
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters.8` | int | 8 | Clusters for resolution 8 |
| `n_clusters.9` | int | 8 | Clusters for resolution 9 |
| `n_clusters.10` | int | 8 | Clusters for resolution 10 |
| `clustering_method` | str | "kmeans" | Algorithm to use |

### Visualization
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cmap` | str | "Accent" | Matplotlib colormap |
| `dpi` | int | 600 | Image resolution |
| `figsize` | tuple | (12, 12) | Figure size in inches |
| `save_format` | str | "png" | Output image format |

### Experiment Tracking
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wandb_project` | str | "urban-embedding" | WandB project name |
| `wandb_entity` | str | None | WandB team/user |
| `debug` | bool | False | Enable debug logging |
| `save_checkpoints` | bool | True | Save model checkpoints |

---

## Example Configurations

### Dense Urban Core Analysis
```python
config = {
    "experiment_name": "city_center_study",
    "fsi_percentile": 99,  # Top 1% densest areas
    "epochs": 200,
    "hidden_dim": 256,  # Larger model for complex areas
    "loss_weights": {
        "reconstruction": 1.0,
        "consistency": 5.0  # Strong multi-scale alignment
    }
}
```

### Suburban Expansion Study
```python
config = {
    "experiment_name": "suburban_growth",
    "fsi_percentile": 50,  # Include suburban areas
    "max_travel_time": {
        "walk": 600,   # 10 minutes
        "bike": 1200,  # 20 minutes
        "drive": 1800  # 30 minutes - longer for suburban
    },
    "n_clusters": {8: 12, 9: 12, 10: 12}  # More clusters for diversity
}
```

### Quick Test Run
```python
config = {
    "experiment_name": "test_run",
    "fsi_percentile": 95,
    "epochs": 10,  # Very short
    "hidden_dim": 64,  # Small model
    "resolutions": [8, 9],  # Skip resolution 10
    "debug": True
}
```

### High-Performance Training
```python
config = {
    "experiment_name": "production_model",
    "fsi_percentile": 90,
    "epochs": 500,
    "learning_rate": 0.0005,
    "weight_decay": 0.00001,
    "patience": 50,
    "hidden_dim": 256,
    "num_convs": 6,
    "dropout": 0.2,
    "wandb_project": "urbanrepml-production"
}
```

---

## Command Line Usage

### Using run_experiment.py
```bash
python scripts/experiments/run_experiment.py \
  --experiment_name my_experiment \
  --city amsterdam \
  --fsi_percentile 95 \
  --epochs 100 \
  --hidden_dim 128 \
  --learning_rate 0.001 \
  --run_training
```

### Environment Variables
You can also set parameters via environment variables:
```bash
export URBANREPML_CITY=amsterdam
export URBANREPML_FSI_PERCENTILE=95
export URBANREPML_EPOCHS=100
python scripts/experiments/run_experiment.py --experiment_name my_experiment
```

---

## Performance Tuning Guide

### Memory Issues
- Reduce `hidden_dim` (try 64 or 32)
- Increase `percentile_threshold` (keep fewer edges)
- Reduce `cutoff_time` (smaller graphs)
- Process fewer resolutions

### Training Too Slow
- Reduce `epochs`
- Increase `learning_rate` (carefully)
- Reduce `num_convs`
- Use smaller `hidden_dim`

### Poor Results
- Increase `epochs`
- Adjust `loss_weights` balance
- Try different `fsi_percentile`
- Increase model capacity (`hidden_dim`, `num_convs`)

### Overfitting
- Increase `dropout`
- Increase `weight_decay`
- Reduce model size
- Use fewer `epochs`

---

## Validation Metrics

The system automatically tracks:
- **Loss curves**: Reconstruction and consistency losses
- **Clustering metrics**: Silhouette score, Davies-Bouldin index
- **Graph statistics**: Number of edges, average degree
- **Training metrics**: Learning rate, gradient norms

Access metrics via:
- WandB dashboard (if configured)
- `experiments/{name}/metrics.json`
- Training logs in `urban_embedding.log`