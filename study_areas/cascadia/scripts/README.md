# Cascadia Coastal Forests - Processing Scripts

## Core Processing Scripts

### `modular_tiff_processor.py` 
**Main processor** - Converts AlphaEarth TIFFs to H3 hexagons using SRAI
- Pre-regionalizes Cascadia coastal area (west of -121°) 
- Processes TIFFs in parallel with spatial filtering
- Saves intermediate JSON results per tile
- Uses 223k H3 hexagons at resolution 8

### `run_coastal_processing.py` (Main Directory)
**Orchestrator script** - Runs the complete processing pipeline
- Handles configuration, logging, and archiving
- Supports parallel workers, checkpointing, resume
- Two-stage approach: tiles→intermediate, then stitch→final

### `stitch_results.py` (Main Directory) 
**Final assembly** - Combines intermediate JSONs into final Parquet
- Merges overlapping hexagons by averaging
- Creates final dataset with 64-band embeddings
- Fast operation (separate from heavy processing)

## Monitoring & Utilities

### `monitor_modular_progress.py`
Real-time progress monitoring with ETA calculations and tile-level tracking

### `capture_current_progress.py` / `check_progress.py` 
Legacy progress utilities (kept for reference)

### `monitor_progress.py`
General progress monitoring (legacy)

### `test_modular.py`
Testing script for small-batch validation

## Visualization Scripts

### Processing-Related Visualizations
- `srai_rioxarray_processor.py` - Legacy SRAI+rioxarray approach
- `benchmark_processors.py` - Performance comparisons 
- `load_alphaearth.py` - Data loading utilities

### Spatial Analysis & Visualization
- `visualizations.py` - General plotting utilities
- `srai_visualizations.py` - SRAI-specific plots
- `quick_res10_viz.py` - Quick resolution 10 visualization
- `full_res10_srai_viz.py` - Full resolution 10 SRAI visualization

### Processor Variants (Reference Only)
- `gpu_multicore_processor.py` - GPU/multicore variant
- `pytorch_tiff_processor.py` - PyTorch-based processor
- `rioxarray_processor.py` - Pure rioxarray approach

## Usage

```bash
# Main processing embeddings (from main directory)
python run_coastal_processing.py --workers 6

# Monitor progress (from scripts directory)  
python monitor_modular_progress.py --continuous

# Final stitching (from main directory)
python stitch_results.py
```

## Data Flow

1. **AlphaEarth TIFFs** (968 files) → Spatial filter → **~592 coastal tiles**
2. **Coastal tiles** → Parallel processing → **Intermediate JSONs** (`data/intermediate/`)  
3. **Intermediate JSONs** → Stitching → **Final Parquet** (`data/final/`)