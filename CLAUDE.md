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

Core libraries: PyTorch, PyTorch Geometric, GeoPandas, H3, OSMnx, scikit-learn, WandB, **SRAI** (for H3 operations)

---

# CASCADIA COASTAL FORESTS PROCESSING

**Status:** ACTIVE (January 2025)
**Purpose:** Process Cascadia coastal forest AlphaEarth satellite embeddings to H3 hexagons
**Location:** `experiments/del_norte_exploratory/` (legacy folder name)
**Focus:** Forested coastal band west of -121° (excludes eastern prairies/valleys)

## Quick Start

### Run Modular Processor
```bash
cd experiments/del_norte_exploratory

# Test with 2 tiles
python run_modular_processor.py --max-tiles 2 --clean-start

# Resume from checkpoint
python run_modular_processor.py --resume

# Full processing
python run_modular_processor.py
```

### Monitor Progress
```bash
# Check current progress
python monitor_progress.py

# View checkpoint status
cat data/checkpoints/modular_progress.json
```

## Architecture: Modular Tile Processing with SRAI

### Key Innovation: Pre-regionalization with SRAI
- **Pre-compute all H3 hexagons** for Cascadia study area (436,944 hexagons at resolution 8)
- **Use SRAI's H3Regionalizer** for proper H3 operations (no API issues)
- **Build spatial KDTree index** for fast pixel-to-hexagon mapping
- **Process subtiles** (256×256) and map pixels to pre-existing hexagon buckets

### Study Area: Cascadia Coastal Forests
- **Full data extent**: -124.70°W to -117.35°E, 38.67°N to 43.37°N (~426,000 sq km)
- **Focused study area**: West of -121°W (~185,000 sq km)
- **Ecosystems**: Coast Range, Klamath Mountains, Coastal Douglas-fir, Coastal Redwood
- **Rationale**: Excludes eastern Oregon/California prairies, focuses on forested ecosystems
- **Tiles**: ~612 out of 968 AlphaEarth TIFFs fall within coastal area (63% of dataset)

### Directory Structure
```
experiments/del_norte_exploratory/
├── config.yaml                 # Main configuration
├── run_modular_processor.py    # Main runner with archiving
├── simple_local_srai.py        # Baseline SRAI approach
├── scripts/
│   ├── modular_tiff_processor.py  # Core modular processor
│   ├── srai_rioxarray_processor.py # SRAI+rioxarray hybrid
│   ├── monitor_progress.py     # Progress monitoring
│   └── [visualization scripts]
├── data/
│   ├── h3_2021_res8_modular/  # Final H3 outputs
│   ├── intermediate/           # Subtile results
│   ├── checkpoints/            # Processing state
│   ├── archive/                # Completed run archives
│   └── progress/               # Progress tracking
└── logs/
    └── modular_processing.log  # Detailed logs
```

### Configuration (config.yaml)
```yaml
experiment:
  name: del_norte_modular_2021
  h3_resolution: 8
  processing_mode: modular

processing:
  subtile_size: 256       # Chunk size
  subtiles_per_batch: 10  # Checkpoint frequency
  min_pixels_per_hex: 5   # Quality threshold
  checkpoint_enabled: true
  resume_from_checkpoint: true
```

### Processing Workflow

1. **Pre-regionalization**: Generate H3 hexagons for coastal forest area (west of -121°) using SRAI
2. **Spatial Filtering**: Only process tiles that intersect the coastal forest study area
3. **Spatial Index**: Build KDTree for O(log n) pixel-to-hexagon lookup (~105k hexagons)
4. **Tile Loading**: Open 3072×3072×64 band TIFF files
5. **Subtiling**: Split into 12×12 grid of 256×256 chunks
6. **Pixel Mapping**: Find nearest hexagon for each pixel using KDTree
7. **Aggregation**: Average pixel values within each hexagon bucket
8. **Checkpointing**: Save progress after each batch
9. **Merging**: Combine results into final coastal forest dataset

### Performance Metrics
- **Subtile processing**: 5-10 seconds
- **Full tile**: 12-24 minutes
- **Daily throughput**: 50-100 tiles
- **Complete dataset**: 3-6 days for 288 tiles

### Resumability Features
- JSON checkpoint with completed tiles/subtiles
- Intermediate results saved per subtile
- Automatic resume on restart
- Archive completed runs with metadata

### Memory Management
- Fixed 256×256 chunk size (vs full 3072×3072)
- Garbage collection every 5 tiles
- Batch processing with controlled memory
- No GPU memory requirements

---

# CASCADIA COASTAL FORESTS EXPERIMENT  

**Status:** ACTIVE (January 2025)  
**Purpose:** Modular TIFF processing workflow for AlphaEarth satellite embeddings analysis  
**Location:** `experiments/cascadia_exploratory/`

## Quick Start Guide

### Prerequisites
```bash
# 1. Install dependencies
pip install -e .

# 2. Navigate to experiment directory
cd experiments/cascadia_exploratory
```

### Two-Stage Processing Workflow
```bash
# Stage 1: TIFF → Intermediate JSONs (parallel processing)
python run_coastal_processing.py --workers 6

# Monitor progress
python scripts/monitor_modular_progress.py --continuous

# Stage 2: Stitch intermediate JSONs → Final Parquet
python stitch_results.py
```

## Project Overview

### Focus: Cascadia Coastal Forests
**Spatial Filtering:** West of -121° longitude to focus on forested coastal ecosystems, excluding eastern prairies/valleys

### Geographic Scope  
- **Region:** Cascadia Coastal Band (Coast Range, Klamath Mountains)
- **Bounds:** West of -121°, from Northern California to Southern Oregon
- **Coverage:** ~592 out of 968 AlphaEarth tiles (~185,000 km²)
- **H3 Resolution:** 8 (~223,904 hexagons pre-regionalized)

### Processing Pipeline
```
AlphaEarth TIFFs (968 files) → Coastal Filter → Stage 1: Parallel → Intermediate JSONs → Stage 2: Stitch → Final Parquet
```

## Data Sources & Configuration

### AlphaEarth Satellite Embeddings
- **Source:** Local TIFFs from Google Drive (`G:/My Drive/AlphaEarth_Cascadia`)
- **Dimensions:** 64 embedding features (bands A00-A63)
- **Resolution:** 10 meters native  
- **Tile Format:** 3072×3072 pixel TIFFs
- **Year:** 2021 (single year for coastal forest analysis)
- **Coverage:** 968 total tiles, ~592 coastal tiles after filtering

### H3 Hexagonal Processing via SRAI
- **Resolution:** 8 (fixed for coastal analysis)
- **Pre-regionalization:** 223,904 hexagons covering coastal area
- **Spatial Library:** SRAI (Spatial Representations for AI)
- **Aggregation:** Mean pooling of pixels within each hexagon
- **Overlap Handling:** Multiple tile contributions averaged during stitching

## Current Implementation Status

### ✅ COMPLETED (January 2025)
1. **Experiment Restructuring**
   - Renamed from del_norte_exploratory to cascadia_exploratory
   - Moved all utility scripts to scripts/ folder
   - Created comprehensive scripts/README.md documentation
   - Implemented two-stage processing architecture

2. **Core Processing Components**
   - **ModularTiffProcessor:** SRAI-based H3 operations with spatial filtering
   - **run_coastal_processing.py:** Stage 1 orchestrator with parallel workers
   - **stitch_results.py:** Stage 2 final assembly with overlap resolution
   - **Spatial filtering:** Coast-only processing (west of -121°)
   - **Pre-regionalization:** 223,904 H3 hexagons cached for efficiency

3. **Configuration & Architecture**
   - Updated config.yaml for coastal forest focus
   - Intermediate storage system for resumable processing  
   - Comprehensive logging and progress monitoring
   - Error handling and checkpointing

### 🔄 READY TO RUN
- **Stage 1:** Parallel TIFF processing to intermediate JSONs
- **Stage 2:** Final dataset stitching with overlap handling
- **Monitoring:** Real-time progress tracking available

### ⏳ FUTURE ENHANCEMENTS
1. **Advanced Analytics:** Clustering and spatial analysis of coastal forests
2. **Multi-Year Processing:** Extend to temporal analysis
3. **Visualization Pipeline:** Interactive coastal forest mapping

## Key Scripts & Usage

### Core Processing Pipeline
```bash
# Stage 1: TIFF → Intermediate JSONs (main processing)
python run_coastal_processing.py --workers 6            # Standard run
python run_coastal_processing.py --max-tiles 50 --workers 4  # Test run
python run_coastal_processing.py --clean-start --workers 8   # Fresh start

# Stage 2: Stitch intermediate JSONs → Final Parquet
python stitch_results.py                                # Standard stitching
python stitch_results.py --cleanup                      # Archive intermediates
python stitch_results.py --output-name custom_name      # Custom output
```

### Monitoring & Utilities  
```bash
# Real-time progress monitoring
python scripts/monitor_modular_progress.py --continuous

# Test processing on small batch
python scripts/test_modular.py --max-tiles 5

# Check processing status
python scripts/check_progress.py
```

### Visualization & Analysis
```bash  
# Generate spatial visualizations
python scripts/visualizations.py --method kmeans --clusters 10

# SRAI-specific visualizations
python scripts/srai_visualizations.py --resolution 8

# Load and explore AlphaEarth data
python scripts/load_alphaearth.py --explore --year 2021
```

### Configuration & Setup
```bash
# Validate configuration
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"

# Test SRAI dependencies
python scripts/test_dependencies.py

# Benchmark different processors
python scripts/benchmark_processors.py
```

## Immediate Next Steps

### Run Complete Processing Pipeline
1. **Execute Stage 1 Processing**
   ```bash
   # Navigate to experiment directory
   cd experiments/cascadia_exploratory
   
   # Run main processing with 6 workers
   python run_coastal_processing.py --workers 6
   ```

2. **Monitor Progress**
   ```bash
   # In separate terminal, monitor progress
   python scripts/monitor_modular_progress.py --continuous
   ```

3. **Execute Stage 2 Stitching**
   ```bash
   # After Stage 1 completes, stitch results
   python stitch_results.py
   ```

### Expected Outcomes
- **Processing Time:** ~4 hours for 592 coastal tiles with 6 workers
- **Intermediate Storage:** ~2-5 GB JSON files in `data/intermediate/`
- **Final Dataset:** Parquet file with ~223,904 H3 hexagons × 64 embedding dimensions
- **Coverage:** Cascadia coastal forests west of -121° longitude

### Data Quality & Validation
1. **Automatic Validation**
   - Spatial bounds checking during tile filtering
   - H3 hexagon overlap resolution during stitching
   - Embedding dimension consistency (64 bands A00-A63)

2. **Manual Quality Checks**
   ```bash
   # Check final dataset statistics
   python -c "import pandas as pd; df=pd.read_parquet('data/h3_2021_res8_coastal_forests/cascadia_coastal_forests_2021_res8_final.parquet'); print(f'Hexagons: {len(df)}, Columns: {df.columns.tolist()}')"
   ```

## Future Development Directions

### Multi-Resolution Analysis
1. **Cross-Scale Processing**
   - Extend to H3 resolutions 9-10 for detailed coastal analysis
   - Hierarchical hexagon relationships for multi-scale insights
   - Scale-adaptive feature aggregation

2. **Temporal Extensions**
   - Multi-year coastal change detection (2017-2024)
   - Seasonal forest dynamics analysis
   - Climate impact on coastal ecosystems

### Advanced Analytics
1. **Coastal Forest Clustering**
   ```bash
   # Future implementation
   python scripts/forest_analytics.py --method hierarchical --clusters 15
   python scripts/temporal_analysis.py --years 2021,2023 --change_detection
   ```

2. **Ecological Insights**
   - Old-growth vs second-growth forest distinction
   - Coastal fog influence on forest health
   - Fire recovery pattern analysis

### Integration Opportunities
1. **Multi-Modal Enhancement**
   - Integrate LiDAR data for canopy structure
   - Combine with climate data for environmental modeling
   - Cross-validate with field survey data

2. **Policy Applications**
   - Carbon sequestration mapping
   - Biodiversity corridor identification
   - Sustainable forestry planning support

## Technical Architecture

### Key Configuration Files
- **Primary Config:** `experiments/cascadia_exploratory/config.yaml`
- **Study Area:** Coastal filtering bounds and H3 regionalization
- **Processing:** Parallel workers, checkpointing, memory management
- **Output:** Directory structure and file naming conventions

### Customization Options
```yaml
# Key config modifications in config.yaml:

# Adjust spatial filtering
study_area:
  bounds:
    east: -120.0           # Extend further inland
    west: -125.0           # Extend further offshore
    
# Modify processing parameters
processing:
  subtile_size: 512        # Larger chunks for more memory
  min_pixels_per_hex: 10   # Stricter hexagon inclusion

# Change output structure
output:
  modular_dir: "data/custom_output"
  log_level: "DEBUG"       # More detailed logging
```

### Debugging & Monitoring
```bash
# Check processing logs
tail -f logs/coastal_processing_*.log

# Validate configuration
python scripts/test_modular.py --dry-run

# Monitor system resources during processing
python scripts/monitor_modular_progress.py --system-stats
```

## System Requirements & Notes

### Hardware Recommendations
- **Memory:** 16GB+ RAM for 6-worker processing (32GB optimal)
- **Storage:** ~10-20GB for intermediate files, ~5GB for final dataset
- **CPU:** Multi-core processor (6+ cores recommended for parallel processing)
- **Network:** Stable connection for large TIFF file access from Google Drive

### Processing Characteristics
- **Resumable:** Checkpointing allows recovery from interruptions
- **Scalable:** Worker count adjustable based on system capabilities
- **Memory-Efficient:** Subtile chunking prevents memory exhaustion
- **Fault-Tolerant:** Individual tile failures don't stop entire pipeline

### Data Outputs
- **Intermediate:** JSON files per tile (~592 files, 2-5GB total)
- **Final:** Single Parquet file (~223k hexagons × 64 dimensions)
- **Metadata:** Processing logs, statistics, and run archives
- **Format:** Standards-compatible H3 + AlphaEarth embeddings

---

**Clean, focused coastal forest processing pipeline ready for execution with two-stage architecture, parallel processing, and comprehensive monitoring.**

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