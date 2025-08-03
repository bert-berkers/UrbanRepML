# UrbanRepML Project Rebuild Guide

This document explains how to rebuild the complete UrbanRepML pipeline from scratch after cleanup.

## 🏗️ Core Architecture

### Pipeline Overview
1. **Data Ingestion**: Multi-modal urban embeddings (GTFS, aerial, POI, road networks)
2. **Regionalization**: H3 hexagonal grids with building density filtering 
3. **Graph Construction**: Travel-time based accessibility graphs (walk/bike/drive)
4. **Model Training**: UrbanUNet - Multi-resolution GNN with U-Net architecture
5. **Analysis**: Clustering and visualization of learned urban representations

### H3 Resolution Strategy
- **Resolution 8**: Drive accessibility (larger regions)
- **Resolution 9**: Bike accessibility (medium regions)  
- **Resolution 10**: Walk accessibility (fine-grained regions)

## 📁 Required Data Sources

### Input Embeddings
```
data/embeddings/
├── gtfs_v2/                     # Public transit embeddings
├── aerial_finetune/             # Fine-tuned aerial imagery embeddings  
├── poi_hex2vec/                 # Point of Interest embeddings
├── roadnetwork/                 # Road network embeddings
└── alphaearth_2023/            # Satellite-derived embeddings (validation)
```

### Administrative Data
```
data/skeleton/
├── boundaries/                  # Study area boundaries (e.g., South Holland)
└── density_filters/            # Building density thresholds (50%, 70%, 80%, 90%)
```

## 🔄 Preprocessing Pipeline

### 1. Region Generation
```python
from urban_embedding import UrbanEmbeddingPipeline

# Create H3 regions with building density filtering
config = {
    'study_area': 'south_holland',
    'h3_resolutions': [8, 9, 10],
    'density_threshold': 80,  # Filter to keep only urban areas
    'building_data_source': 'osm'  # or 'bag' for Netherlands
}
```

### 2. Network Processing  
```python
# Generate travel-time accessibility networks
networks = {
    'walk': {'max_time': 15, 'speed': 4.5},   # 15min walking at 4.5 km/h
    'bike': {'max_time': 20, 'speed': 12},    # 20min cycling at 12 km/h  
    'drive': {'max_time': 30, 'speed': 35}    # 30min driving at 35 km/h
}
```

### 3. Feature Processing
```python
# Apply PCA dimensionality reduction to embeddings
pca_config = {
    'n_components': 50,  # Reduce to 50 dimensions
    'datasets': ['gtfs', 'aerial', 'poi', 'roadnetwork']
}
```

## 🚀 Running the Pipeline

### Complete Pipeline
```python
from urban_embedding import UrbanEmbeddingPipeline

# Create configuration
config = UrbanEmbeddingPipeline.create_default_config(
    city_name="south_holland",
    threshold=80
)

# Run full pipeline
pipeline = UrbanEmbeddingPipeline(config)
embeddings = pipeline.run()
```

### Individual Components
```python
# Just preprocessing
pipeline.preprocess_data()

# Just graph construction  
pipeline.build_graphs()

# Just model training
pipeline.train_model()

# Just analysis
pipeline.analyze_embeddings()
```

## 📊 Output Structure

### Model Outputs
- **Graphs**: Travel-time accessibility graphs for each resolution/mode
- **Embeddings**: Learned urban representations (64-dim per resolution)
- **Models**: Trained UrbanUNet checkpoints
- **Clusters**: K-means clustering results for different k values

### Cache Strategy
All expensive computations are cached:
- OSM network downloads and processing
- Travel-time calculations
- PCA model fitting
- Graph construction

## 🗄️ Data Requirements

### Minimum Required Files
1. **Study area boundary** (GeoDataFrame)
2. **Input embeddings** (parquet files with H3 index)
3. **Building density data** (for urban filtering)

### Optional Enhancements
- Custom travel speeds per region
- Additional embedding modalities
- Different clustering algorithms
- Custom visualization themes

## 🔧 Key Parameters

### Model Architecture
```yaml
model:
  hidden_dim: 128
  num_conv_layers: 3
  dropout: 0.1
  
training:
  learning_rate: 0.001
  epochs: 100
  batch_size: 32
  
loss:
  reconstruction_weight: 1.0
  consistency_weight: 0.5  # Cross-scale consistency
```

### Performance Tuning
- **GPU recommended** for model training
- **16GB+ RAM** for large study areas  
- **SSD storage** for faster data loading

## 🧪 Validation Workflow

### Internal Validation
- Cross-scale embedding consistency
- Reconstruction loss convergence
- Cluster stability across runs

### External Validation  
- Compare with known urban typologies
- Correlation with census/survey data
- Expert knowledge validation

## 📋 Dependencies

See `requirements.txt` for complete list. Key dependencies:
- PyTorch + PyTorch Geometric (GNN framework)
- GeoPandas + OSMnx (spatial analysis)
- scikit-learn (clustering, PCA)
- H3 (hexagonal indexing)
- WandB (experiment tracking)

## 🚨 Common Issues

1. **Memory**: Large study areas may require chunked processing
2. **OSM Networks**: May need manual topology cleaning for some regions
3. **Embedding Alignment**: Ensure all embeddings use same H3 indexing
4. **Travel Times**: GTFS data quality varies by region