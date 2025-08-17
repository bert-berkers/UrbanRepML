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

### Running Experiments

#### Quick Start - South Holland FSI 95%
```bash
# Run complete experiment with one command
python run_south_holland_fsi95.py
```

#### Custom Experiments
```bash
# Run any experiment with the orchestrator
python scripts/experiments/run_experiment.py \
  --experiment_name my_experiment \
  --city amsterdam \
  --fsi_percentile 90 \
  --run_training \
  --epochs 200
```

#### Step-by-Step Data Preparation
```bash
# 1. Create H3 regions
python scripts/preprocessing/setup_regions.py \
  --city_name amsterdam \
  --resolutions 8,9,10

# 2. Calculate building density
python scripts/preprocessing/setup_density.py \
  --city_name amsterdam \
  --input_dir data/preprocessed/amsterdam_base \
  --building_data data/preprocessed/density/PV28__00_Basis_Bouwblok.shp

# 3. Filter by FSI threshold
python scripts/preprocessing/setup_fsi_filter.py \
  --city_name amsterdam \
  --input_dir data/preprocessed/amsterdam_base \
  --output_dir experiments/amsterdam_dense/data \
  --fsi_percentile 90

# 4. Generate accessibility graphs
python scripts/preprocessing/setup_hierarchical_graphs.py \
  --data_dir experiments/amsterdam_dense/data \
  --output_dir experiments/amsterdam_dense/graphs
```

### Running the Pipeline Directly
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

---

# CASCADIA ALPHAEARTH EXPERIMENT

**Status:** ACTIVE (January 2025)  
**Purpose:** Multi-resolution spatial representation learning using AlphaEarth satellite embeddings for GEO-INFER agricultural analysis  
**Location:** `experiments/cascadia_geoinfer_alphaearth/`

## Quick Start Guide

### Prerequisites
```bash
# 1. Install dependencies
pip install -e .

# 2. Set up Google Earth Engine authentication
earthengine authenticate --project=boreal-union-296021

# 3. Navigate to experiment directory
cd experiments/cascadia_geoinfer_alphaearth
```

### Current Status Check
```bash
# Check AlphaEarth export progress
python scripts/gee/check_export_status.py

# View comprehensive tile tracking log
cat TILE_EXPORT_LOG.md

# Monitor tasks at: https://code.earthengine.google.com/tasks?project=boreal-union-296021
```

## Project Overview

### Core Concept: "Actualization"
This experiment implements **actualization** - a philosophical approach to machine learning that:
1. **Identifies gaps** in satellite data coverage (spatial, temporal, quality)
2. **Learns relational structures** between existing data points
3. **Generates synthetic data** for missing regions through understanding of underlying patterns
4. **"Carves nature at its joints"** - discovers natural boundaries and relationships in geographic data

### Geographic Scope
- **Region:** Cascadia Bioregion (Northern California + Oregon)
- **Counties:** 52 total (16 CA + 36 OR)  
- **Bounds:** [-124.6, 39.0] to [-116.5, 46.3]
- **Area:** ~421,000 km²

### Data Pipeline
```
AlphaEarth (GEE) → H3 Multi-Resolution → Gap Detection → Synthetic Generation → GEO-INFER Format
```

## Data Sources & Configuration

### AlphaEarth Satellite Embeddings
- **Collection:** `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`  
- **Dimensions:** 64 embedding features (bands A00-A63)
- **Resolution:** 10 meters native
- **Years:** 2017-2024 (8 years)
- **Coverage:** 288 images over Cascadia region
- **Export Format:** 3072×3072 pixel tiles per year → Google Drive

### H3 Hexagonal Processing
- **Resolutions:** 5-11 (adaptive based on analysis needs)
- **Primary Resolution:** 8 (GEO-INFER standard)  
- **Architecture:** Hierarchical parent-child mappings
- **Memory Management:** Chunked processing for high resolutions

## Current Implementation Status

### ✅ COMPLETED (January 2025)
1. **Google Earth Engine Integration**
   - Authentication with project `boreal-union-296021`
   - AlphaEarth collection access confirmed
   - Export architecture validated

2. **Export Infrastructure** 
   - **1,093+ tiles queued** across multiple years
   - Systematic tile tracking and logging
   - Automated progress monitoring
   - Real-time status checking

3. **Data Organization**
   - Comprehensive configuration system
   - Modular script architecture
   - Error handling and retry mechanisms
   - Structured logging throughout

### 🔄 IN PROGRESS
- **AlphaEarth Exports:** 21.2% coverage (1,093/5,152 tiles)
  - 2017: 70+ tiles queued
  - 2018: 569+ tiles queued  
  - 2019: 81+ tiles queued
  - 2020: 294+ tiles queued
  - 2021: 75+ tiles submitting
  - 2022: Ready to start
  - 2023: 69 tiles (16 completed, 27 failed)
  - 2024: Ready to start

### ⏳ PENDING
1. **H3 Multi-Resolution Processing**
2. **Gap Detection & Analysis** 
3. **Synthetic Data Generation**
4. **GEO-INFER Dataset Preparation**

## Key Scripts & Usage

### Google Earth Engine Operations
```bash
# Check AlphaEarth data availability
python scripts/gee/check_years_availability.py --save_report

# Start exports for specific years
python scripts/gee/export_cascadia_alphaearth.py --year 2022
python scripts/gee/export_cascadia_alphaearth.py --years 2022 2024
python scripts/gee/export_cascadia_alphaearth.py --all_years

# Monitor export progress  
python scripts/gee/check_export_status.py
```

### H3 Processing Pipeline
```bash
# Process downloaded AlphaEarth to H3 format
python scripts/h3/process_alphaearth_to_h3.py --year 2023 --resolution 8

# Generate multi-resolution hierarchy
python scripts/h3/create_hierarchical_mapping.py --resolutions 5,6,7,8,9,10,11

# Validate H3 data quality
python scripts/h3/validate_h3_data.py --year 2023 --check_coverage
```

### Gap Detection & Actualization
```bash  
# Detect spatial/temporal gaps
python scripts/actualization/gap_detector.py --all_years --all_resolutions --save_report

# Generate synthetic embeddings for gaps
python scripts/actualization/synthetic_generator.py --year 2023 --method vae --validate

# Quality assessment of synthetic data
python scripts/actualization/validate_synthetic.py --comparison_year 2022
```

### GEO-INFER Integration
```bash
# Prepare data for GEO-INFER agricultural analysis
python scripts/geoinfer/prepare_for_geoinfer.py --year 2023 --include_synthetic

# Validate GEO-INFER compatibility
python scripts/geoinfer/prepare_for_geoinfer.py --validate_only --all_years

# Generate final datasets
python scripts/geoinfer/create_final_datasets.py --output_format parquet
```

## Medium Term Goals (Next 30 Days)

### Phase 1: Complete Data Acquisition
1. **Finish AlphaEarth Exports**
   ```bash
   # Start remaining years
   python scripts/gee/export_cascadia_alphaearth.py --years 2022 2024
   
   # Monitor completion
   python scripts/gee/check_export_status.py
   ```

2. **Local Data Sync**
   - Monitor Google Drive "AlphaEarth_Cascadia" folder
   - Validate downloaded tile integrity
   - Organize by year/tile structure

### Phase 2: H3 Multi-Resolution Processing  
1. **Convert to H3 Format**
   ```bash
   # Process each year as data becomes available
   for year in {2017..2024}; do
     python scripts/h3/process_alphaearth_to_h3.py --year $year --all_resolutions
   done
   ```

2. **Build Hierarchical Mappings**
   ```bash
   python scripts/h3/create_hierarchical_mapping.py --validate_consistency
   ```

### Phase 3: Gap Detection & Analysis
1. **Comprehensive Gap Analysis**
   ```bash
   python scripts/actualization/gap_detector.py --all_years --all_resolutions --save_report
   ```

2. **Spatial Coverage Assessment**
   - Identify regions with missing data
   - Quantify temporal inconsistencies  
   - Flag quality issues

### Phase 4: Actualization - Synthetic Data Generation
1. **Train Generative Models**
   ```bash
   python scripts/actualization/synthetic_generator.py --all_years --method vae
   ```

2. **Validate Synthetic Quality** 
   ```bash
   python scripts/actualization/validate_synthetic.py --comprehensive
   ```

## Long Term Vision (6+ Months)

### Advanced Actualization Research
1. **Multi-Modal Learning**
   - Integrate additional satellite data (Sentinel-2, Landsat)
   - Cross-validate synthetic quality
   - Temporal consistency learning

2. **Relational Pattern Discovery**
   - Agricultural boundary detection
   - Seasonal pattern learning
   - Cross-county relationship modeling

### GEO-INFER Agricultural Integration
1. **Policy Analysis Ready Datasets**
   ```bash
   python scripts/geoinfer/create_policy_datasets.py --county_level --time_series
   ```

2. **Agricultural Trend Analysis**
   - Multi-year change detection
   - Climate impact assessment
   - Sustainable farming indicators

### Scaling & Methodology Development
1. **Expand Geographic Coverage**
   - Pacific Northwest (Washington, British Columbia)
   - Other bioregions (Colorado Plateau, Great Basin)

2. **Framework Generalization**
   - Create reusable actualization pipeline
   - Multi-domain synthetic data generation
   - Cross-regional validation

## Configuration & Customization

### Key Configuration Files
- **Primary Config:** `experiments/cascadia_geoinfer_alphaearth/config.yaml`
- **GEE Settings:** `scripts/gee/` (authentication, export parameters)
- **H3 Processing:** Configure resolutions, memory management
- **Actualization:** Gap detection thresholds, synthetic generation methods

### Customization Options
```yaml
# Example config modifications in config.yaml:

# Change H3 resolution focus
h3_processing:
  resolutions: [8, 9, 10]  # Focus on core resolutions
  primary_resolution: 9    # Higher resolution analysis

# Adjust actualization parameters  
actualization:
  generation:
    method: diffusion        # Advanced synthetic generation
    latent_dimensions: 32    # Higher dimensional learning
```

### Error Handling & Debugging
```bash
# Debug export issues
python scripts/gee/export_cascadia_alphaearth.py --year 2023 --dry_run

# Check processing logs
tail -f logs/processing_*.log

# Validate intermediate outputs
python scripts/validation/check_data_integrity.py --comprehensive
```

## Important Notes

### Data Management
- **Storage Requirements:** ~500GB for complete dataset
- **Backup Strategy:** Keep raw exports + processed H3 data
- **Version Control:** Track data lineage through processing pipeline

### Computational Resources
- **Memory:** 32GB+ recommended for full resolution processing
- **GPU:** CUDA-enabled GPU for synthetic generation
- **Network:** Stable connection for GEE exports

### Collaboration & Sharing
- **Data Sharing:** Processed datasets compatible with GEO-INFER standards
- **Code Reusability:** Modular architecture supports adaptation
- **Documentation:** Comprehensive logging enables reproducibility

---

This experiment represents a novel approach to satellite data analysis through actualization - using AI to understand and fill gaps in Earth observation data, enabling more complete agricultural and environmental analysis for policy making.

## Modular Scripts Architecture

### Preparation Pipeline
1. **setup_regions.py**: Creates H3 regions for any city/area
2. **setup_density.py**: Calculates building density using shapefiles
3. **setup_fsi_filter.py**: Filters regions by FSI (percentile or absolute)
4. **setup_hierarchical_graphs.py**: Generates accessibility graphs

### Study Area Filtering System
- **StudyAreaFilter**: Advanced filtering with bioregional categories and adaptive resolution
- **Configuration-driven**: YAML-based study area definitions with validation
- **CLI tools**: Interactive creation, listing, and validation of study areas

```bash
# Create new study area interactively
python scripts/study_areas/create_study_area.py

# List all available study areas
python scripts/study_areas/list_study_areas.py --detailed

# Use study area in experiments
python scripts/experiments/run_experiment.py --study_area willamette_valley_agriculture
```

### Experiment System
- **run_experiment.py**: Orchestrates complete experiments
- All scripts accept CLI arguments and are parameterizable
- Output directories organize by experiment name

### Common Parameter Patterns
```bash
# All preprocessing scripts support:
--city_name CITY
--input_dir DIR
--output_dir DIR
--resolutions 8,9,10

# FSI filtering options:
--fsi_percentile 95      # Top 5% densest areas
--fsi_threshold 0.1      # Absolute FSI >= 0.1

# Study area filtering:
--study_area AREA_NAME   # Use predefined study area
--bioregion agriculture  # Filter by bioregion type

# Graph generation options:
--cutoff_time 300        # Max travel time
--percentile_threshold 90  # Edge filtering
--fsi_threshold 0.1      # Active hexagon threshold
```

## Study Area Examples

### Bioregional Study Areas (GEO-INFER Aligned)
- **willamette_valley_agriculture**: Specialty crops, regenerative farming
- **coast_range_forestry**: Sustainable timber, carbon sequestration  
- **klamath_watershed**: Cross-border water management, drought resilience
- **eastern_oregon_rangelands**: Extensive grazing, rangeland management
- **north_coast_fog_belt**: Redwood conservation, climate adaptation

### Key Features
- **Adaptive resolution**: Higher detail for complex/important areas
- **Memory-aware chunking**: Prevents computational resource exhaustion
- **Bioregional context**: Agricultural/ecological management focus
- **Computational efficiency**: Variable processing depth based on area characteristics
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
- always be transparent in your systems engineering. do NOT leave me out of the loop in detailed architectural designs where they are ameniable to my style of geometric operations research thinking###