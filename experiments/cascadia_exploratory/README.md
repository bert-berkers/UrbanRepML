# Cascadia Exploratory Experiment

**Status:** ACTIVE (January 2025)
**Purpose:** A modular TIFF processing workflow for analyzing AlphaEarth satellite embeddings of the Cascadia bioregion. This experiment serves as a large-scale testbed for the `UrbanRepML` toolkit and as a data provider for the [GEO-INFER](https://github.com/ActiveInferenceInstitute/GEO-INFER) project.

## 🚀 Quick Start Guide

### Prerequisites
Ensure you have installed the repository's dependencies:
```bash
# From the repository root
pip install -e .
```

### Two-Stage Processing Workflow
This experiment uses a two-stage pipeline to process a large number of TIFF files efficiently.

```bash
# From within the experiments/cascadia_exploratory/ directory

# Stage 1: Process raw TIFFs into intermediate JSON files (in parallel)
python run_coastal_processing.py --workers 6

# Monitor progress in a separate terminal
python scripts/monitor_modular_progress.py --continuous

# Stage 2: Stitch the intermediate JSONs into a final Parquet file
python stitch_results.py
```

## 🎯 Project Overview

### Focus: Cascadia Coastal Forests
This experiment focuses on the forested coastal ecosystems of the Cascadia bioregion. To achieve this, the analysis is spatially filtered to the area west of -121° longitude, which excludes the eastern prairies and valleys.

### Geographic Scope
- **Region:** Cascadia Coastal Band (Coast Range, Klamath Mountains)
- **Bounds:** West of -121°, from Northern California to Southern Oregon
- **Coverage:** Approximately 592 out of 968 AlphaEarth tiles (~185,000 km²)
- **H3 Resolution:** 8 (pre-regionalized to ~223,904 hexagons)

### Processing Pipeline
The data flows through the following stages:
`AlphaEarth TIFFs → Coastal Filter → Stage 1 (Parallel Processing) → Intermediate JSONs → Stage 2 (Stitching) → Final Parquet File`

## 📊 Data Sources & Configuration

### AlphaEarth Satellite Embeddings
- **Source:** Local TIFFs from Google Drive (`G:/My Drive/AlphaEarth_Cascadia`)
- **Dimensions:** 64 embedding features (bands A00-A63)
- **Resolution:** 10 meters native
- **Tile Format:** 3072×3072 pixel TIFFs
- **Year:** 2021 (for this specific coastal forest analysis)

### H3 Hexagonal Processing via SRAI
- **Resolution:** 8 (fixed for this analysis)
- **Pre-regionalization:** The ~224k hexagons covering the study area are generated upfront for efficiency.
- **Spatial Library:** [SRAI (Spatial Representations for AI)](https://github.com/kraina-ai/srai)
- **Aggregation:** Pixel values are aggregated into their corresponding H3 hexagons using mean pooling.
- **Overlap Handling:** Contributions from multiple tiles to the same hexagon are averaged during the stitching phase.

### Configuration (`config.yaml`)
The main configuration for this experiment is in `config.yaml`. Key parameters include:
```yaml
experiment:
  name: del_norte_modular_2021 # Note: Legacy name
  h3_resolution: 8
  processing_mode: modular

processing:
  subtile_size: 256       # Chunk size for memory management
  subtiles_per_batch: 10  # Checkpoint frequency
  min_pixels_per_hex: 5   # Quality control threshold
  checkpoint_enabled: true
  resume_from_checkpoint: true
```

## 🏛️ Architecture & Implementation

This experiment uses a modular, two-stage architecture designed for resilience and efficiency when processing large datasets.

### Stage 1: Modular TIFF Processor (`run_coastal_processing.py`)
- The `ModularTiffProcessor` uses SRAI for robust H3 operations.
- It pre-regionalizes the study area and builds a spatial KDTree for fast pixel-to-hexagon mapping.
- Large TIFF tiles are broken down into smaller subtiles (e.g., 256x256 pixels) to keep memory usage low.
- Processing is parallelized across multiple workers.
- Progress is checkpointed, allowing the process to be resumed if interrupted.

### Stage 2: Stitching (`stitch_results.py`)
- After all tiles are processed, the stitching script combines the intermediate JSON files.
- It correctly handles hexagons that span multiple tiles by averaging the values.
- The final output is a single, clean Parquet file containing the H3 index and the 64 embedding dimensions for each hexagon.

## 🛠️ Key Scripts & Usage

### Core Pipeline
```bash
# Stage 1: Run with 6 workers (adjust based on your CPU cores)
python run_coastal_processing.py --workers 6

# Stage 1: Test run with a small number of tiles
python run_coastal_processing.py --max-tiles 10 --workers 4

# Stage 1: Start fresh, clearing any intermediate results
python run_coastal_processing.py --clean-start --workers 8

# Stage 2: Standard stitching
python stitch_results.py

# Stage 2: Stitch and then archive the intermediate files
python stitch_results.py --cleanup
```

### Monitoring & Utilities
```bash
# Monitor progress in real-time
python scripts/monitor_modular_progress.py --continuous

# Get a snapshot of the current progress
python scripts/check_progress.py

# Run a quick test of the processing logic on a few tiles
python scripts/test_modular.py --max-tiles 5
```

### Visualization & Analysis
```bash
# Generate spatial visualizations of the results
python scripts/visualizations.py --method kmeans --clusters 10

# Generate visualizations specific to SRAI
python scripts/srai_visualizations.py --resolution 8
```

## 📈 Expected Outcomes & Performance

- **Processing Time:** ~4 hours for 592 coastal tiles with 6 workers.
- **Intermediate Storage:** ~2-5 GB of JSON files in `data/intermediate/`.
- **Final Dataset:** A single Parquet file (`~224k rows x 65 columns`).
- **Coverage:** Cascadia coastal forests west of -121° longitude.
- **Resilience:** The process is resumable and fault-tolerant. Individual tile failures will not stop the entire pipeline.
