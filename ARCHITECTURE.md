# UrbanRepML Architecture Documentation

## 🎯 Project Vision

UrbanRepML is an advanced **multi-modal, multi-scale urban representation learning system** that combines sophisticated neural architectures with geospatial intelligence to understand urban dynamics from regional patterns to fine-grained local features.

### Core Innovation
The system implements **hierarchical spatial-temporal modeling** using H3 hexagonal grids (resolutions 5-11) with multiple neural architectures, including renormalizing generative models and active inference frameworks, to capture urban complexity across seven spatial scales.

## 🏗️ System Architecture Overview

### Multi-Resolution Spatial Framework
```
H3 Resolution Hierarchy (6 Levels):
├── Res 10 (66m edge)     → Liveability: Block-level analysis, daily patterns
├── Res 9  (170m edge)    → Neighborhood patterns, accessibility zones
├── Res 8  (460m edge)    → District analysis, GEO-INFER standard
├── Res 7  (1.2km edge)   → Municipal boundaries, urban structure
├── Res 6  (3.2km edge)   → County-level planning, watershed dynamics
└── Res 5  (9.2km edge)   → Sustainability: Regional patterns, bioregional analysis
```

### Neural Architecture Stack
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT MODALITIES                         │
│  AlphaEarth | Semantic Segmentation | POI | GTFS | Roads    │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                RENORMALIZING U-NET                          │
│  • Upward Flow: Accumulated updates (res 10→5)             │
│  • Downward Flow: Direct pass-through (res 5→10)           │
│  • Normalization-style batching for hierarchical learning   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              HIERARCHICAL SPATIAL U-NET                     │
│  • Cross-scale consistency via skip connections             │
│  • SRAI-powered hexagonal convolutions                      │
│  • Ring aggregation for neighborhood awareness              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               ACTIVE INFERENCE ENGINE                       │
│  • Hierarchical belief updating across scales              │
│  • Policy optimization for intervention modeling           │
│  • Bridge integration with external inference systems      │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Neural Architecture Components

### 1. Renormalizing Urban U-Net
**Purpose**: Implements renormalizing generative models inspired by Friston et al. for hierarchical urban pattern learning.

**Key Features**:
- **Upward Flow** (res 10→5): Momentum-based accumulation from liveability to sustainability
- **Downward Flow** (res 5→10): Direct pass-through from sustainability to liveability  
- **Simple MSE Losses**: Reconstruction at res 10 (liveability) + consistency between all adjacent levels
- **No Active Inference**: Pure transmission architecture focused on representation learning

**Use Cases**: 
- Multi-scale urban pattern discovery
- Hierarchical feature extraction from fine to coarse scales
- Sustainability (slow dynamics, res 5-7) to liveability (fast dynamics, res 9-10) modeling

### 2. Hierarchical Spatial U-Net  
**Purpose**: SRAI-integrated spatial processing with hexagonal awareness.

**Key Features**:
- Ring aggregation for hexagonal neighborhoods
- Cross-scale skip connections maintaining spatial hierarchy
- Hexagonal convolutions respecting H3 topology
- Multi-modal feature fusion at each resolution

### 3. Active Inference Module
**Purpose**: Hierarchical belief updating and policy optimization.

**Key Features**:
- Bayesian belief updating across spatial scales
- Integration with external inference systems (RxInfer)
- Policy optimization for urban intervention modeling
- Hierarchical Markov Decision Process support

### 4. Bridge Infrastructure
**Purpose**: Integration with external systems and advanced inference.

**Components**:
- **Model Integration**: Seamless switching between neural architectures
- **P-adic MDP Scheduler**: Advanced mathematical scheduling for hierarchical decisions
- **RxInfer Server**: Bayesian inference backend integration
- **Data Exchange**: Standardized interfaces for multi-modal data flow

## 📊 Data Pipeline Architecture

### Multi-Modal Processing Pipeline

#### 1. AlphaEarth + Semantic Segmentation
**Primary Modality**: Satellite imagery with deep learning embeddings + semantic understanding.

```
Google Earth Engine (2017-2024) → AlphaEarth 64-dim → DINOv3 Encoder → 
Semantic Segmentation → Categorical Urban Classes → H3 Aggregation
```

**Features**:
- Multi-year temporal coverage (8 years)
- 64-dimensional AlphaEarth embeddings at 10m resolution
- DINOv3-powered semantic segmentation with AlphaEarth conditioning
- Categorical land cover classes with urban/natural classification

#### 2. Aerial Imagery (Netherlands)
**Regional Specialization**: High-resolution PDOK imagery with DINOv3 encoding.

```
PDOK WMS Service → RGB Images → DINOv3 (Remote Sensing) → 
Hierarchical Aggregation → H3 Embeddings
```

#### 3. Additional Modalities
- **POI**: Points of interest with Hex2Vec embeddings
- **GTFS**: Public transit accessibility matrices
- **Roads**: OSM network topology with graph metrics
- **Buildings**: Footprint density and morphology (FSI calculations)

### Study Area Implementations

#### Cascadia Bioregion
**Scale**: 52 counties (CA + OR), ~421,000 km²
**Purpose**: Agricultural analysis, forest-urban interface, GEO-INFER integration
**Data**: Multi-year AlphaEarth (2017-2024), all H3 resolutions 5-10
**Innovation**: Actualization framework for synthetic data generation in gaps

#### Netherlands Urban Systems  
**Scale**: Dense urban regions, multiple variants (FSI thresholds 0.1, 95%, 99%)
**Purpose**: Urban density analysis, cycling infrastructure, compact development
**Data**: Building density (FSI), accessibility networks, aerial imagery
**Focus**: High-resolution urban pattern analysis (res 8-10)

## 🔄 Hierarchical Processing Workflow

### 1. Data Preparation Pipeline

```bash
# Multi-resolution region setup
python scripts/preprocessing/setup_regions.py --city cascadia --resolutions 10,9,8,7,6,5

# Building density calculation (where applicable)  
python scripts/preprocessing/setup_density.py --city netherlands

# Hierarchical filtering with parent-child preservation
python scripts/preprocessing/setup_fsi_filter.py --city netherlands --fsi-percentile 95

# Multi-modal accessibility graphs
python scripts/preprocessing/setup_hierarchical_graphs.py --city netherlands
```

### 2. Neural Architecture Selection

```python
from urban_embedding import (
    RenormalizingUrbanPipeline,        # For hierarchical pattern learning
    UrbanEmbeddingPipeline,            # For standard multi-modal fusion  
    SemanticSegmentationProcessor      # For satellite+aerial fusion
)

# Renormalizing architecture for multi-scale learning
config = create_renormalizing_config_preset("default")
pipeline = RenormalizingUrbanPipeline(config)
embeddings = pipeline.run()

# Semantic segmentation for satellite imagery
processor = SemanticSegmentationProcessor(config)
segmentation = processor.run_pipeline(study_area, h3_resolution=10, output_dir)
```

### 3. Advanced Analysis Capabilities

**Hierarchical Clustering**: Cross-scale pattern discovery
**Active Inference**: Policy optimization and intervention modeling  
**Actualization**: Synthetic data generation for missing regions/timepoints
**Bridge Integration**: External system connectivity

## 🎛️ Configuration & Tuning

### Architecture Selection Guidelines

| Use Case | Architecture | Resolutions | Notes |
|----------|--------------|-------------|-------|
| **Regional Analysis** | RenormalizingUrbanUNet | 5-8 | Sustainability focus, slow dynamics |
| **Urban Planning** | HierarchicalSpatialUNet | 7-10 | Cross-scale planning integration |
| **Neighborhood Study** | Standard UrbanUNet | 8-10 | Proven for local-scale analysis |
| **Satellite Analysis** | SemanticSegmentation | 10 | AlphaEarth + DINOv3 fusion |
| **Policy Modeling** | Active Inference | 5-11 | Full hierarchy with intervention modeling |

### Key Parameters

**Renormalizing Flow**:
- `upward_momentum`: 0.9 (accumulation strength)
- `normalization_type`: "layer" (LayerNorm, GroupNorm, BatchNorm)
- `accumulation_mode`: "grouped" (batching strategy)

**Multi-Resolution**:
- `resolutions`: [10,9,8,7,6,5] (full hierarchy)
- `primary_resolution`: 8 (GEO-INFER compatibility)
- `fine_resolution`: 10 (liveability focus)

**Loss Functions**:
- `reconstruction_weight`: 1.0 (only at res 10)
- `consistency_weight`: 2.0-3.0 (between adjacent levels)

## 🚀 Performance & Scalability

### Computational Requirements

| Study Area | Hexagons | Memory | Processing Time | Storage |
|------------|----------|---------|-----------------|---------|
| Netherlands (res 10) | 261K | 16GB | 2-4 hours | 5GB |
| Cascadia (res 8) | 915K | 32GB | 12-24 hours | 50GB |
| Cascadia (res 10) | 67M | 64GB+ | 3-5 days | 500GB |

### Optimization Strategies

**Memory Management**:
- Adaptive sampling for high resolutions (5-7)
- Batch processing with configurable sizes
- Gradient checkpointing for deep hierarchies

**Processing Efficiency**:
- GPU acceleration for neural architectures
- Parallel processing for data preparation  
- Smart caching and intermediate result storage

## 🔗 Integration Points

### GEO-INFER Compatibility
- Primary interface at H3 resolution 8
- County-level aggregation support  
- Agricultural pattern analysis alignment
- Cross-border regional analysis (CA-OR)

### External Systems
- **Google Earth Engine**: Satellite data export
- **RxInfer**: Bayesian inference backend
- **SRAI**: Spatial analysis and regionalization
- **Weights & Biases**: Experiment tracking

## 🧪 Experimental Framework

### Experiment Orchestration
```bash
# Complete experiment workflow
python scripts/experiments/run_experiment.py \
  --experiment_name cascadia_multi_resolution \
  --study_area cascadia \
  --resolutions 10,9,8,7,6,5 \
  --architecture renormalizing \
  --run_training
```

### Results & Analysis
- **Embeddings**: Multi-resolution learned representations
- **Clusters**: Hierarchical spatial typologies  
- **Visualizations**: Interactive and static maps
- **Metrics**: Cross-scale validation and performance tracking

## 📈 Current Capabilities Summary

✅ **Multi-Resolution Processing**: 6 H3 levels (10-5)  
✅ **Advanced Neural Architectures**: 4+ specialized architectures  
✅ **Semantic Segmentation**: Complete satellite+aerial fusion  
✅ **Multi-Year Analysis**: 8-year temporal coverage (Cascadia)  
✅ **Professional Tooling**: Experiment orchestration, visualization  
✅ **Bridge Infrastructure**: External system integration  
✅ **Active Inference**: Hierarchical belief updating and policy optimization

---

*Last Updated: August 29, 2025 - Reflects current implementation with renormalizing architectures, semantic segmentation, and multi-year Cascadia experiments*