# Del Norte Modular Processing System

## Overview
This is a clean, modular approach to processing AlphaEarth TIFF files to H3 hexagons, designed for reliability and resumability.

## Quick Start

```bash
# Test with 2 tiles
python run_modular_processor.py --max-tiles 2

# Full processing
python run_modular_processor.py

# Monitor progress
python monitor_modular_progress.py --continuous
```

## Key Features

### 1. Subtile Processing
- Splits 3072×3072 TIFFs into 144 manageable 256×256 chunks
- Processes in controlled batches with checkpointing
- Fixed memory usage regardless of dataset size

### 2. Checkpoint System
- Saves progress after every 10 subtiles
- Tracks completed tiles and subtiles separately
- Automatic resume from exact interruption point
- JSON-based checkpoint for transparency

### 3. Archive System
Each run creates an archive with:
- Complete configuration snapshot
- Run metadata (start/end time, duration)
- Processing statistics
- Enables perfect reproducibility

## Directory Structure

```
data/
├── h3_2021_res8_modular/      # Final outputs
│   ├── *_h3_res8.json         # Per-tile H3 data
│   └── complete.parquet       # Merged dataset
├── checkpoints/               # Processing state
│   └── modular_progress.json  # Current checkpoint
├── intermediate/              # Subtile results (optional)
├── archive/                   # Completed runs
│   └── run_YYYYMMDD_HHMMSS/
│       └── run_metadata.json
└── progress/                  # Legacy progress tracking

logs/
├── modular_processing.log     # Main log
└── modular_run_*.log         # Per-run logs
```

## Configuration

Key parameters in `config.yaml`:

```yaml
processing:
  subtile_size: 256          # Chunk size (don't change)
  subtiles_per_batch: 10     # Checkpoint frequency
  min_pixels_per_hex: 5      # Quality threshold
  memory_cleanup_interval: 5  # GC frequency
  checkpoint_enabled: true   # Enable checkpointing
```

## Performance

Expected processing times:
- Per subtile: 5-10 seconds
- Per tile (144 subtiles): 12-24 minutes  
- Daily throughput: 50-100 tiles
- Full dataset (288 tiles): 3-6 days

Memory usage:
- Peak: ~2-3GB per tile
- Steady state: ~1GB
- No GPU memory required

## Monitoring

### Real-time Progress
```bash
python monitor_modular_progress.py --continuous
```

Shows:
- Overall progress percentage
- Tiles completed/in-progress
- Subtile-level progress for active tiles
- Time estimation
- Output file count

### Quick Summary
```bash
python monitor_modular_progress.py --summary
```

### Check Checkpoint
```bash
cat data/checkpoints/modular_progress.json | python -m json.tool
```

## Troubleshooting

### Resume After Interruption
The processor automatically resumes from checkpoint:
```bash
python run_modular_processor.py --resume
```

### Force Clean Start
To ignore existing checkpoints:
```bash
python run_modular_processor.py --clean-start
```

### Memory Issues
If encountering memory issues:
1. Reduce `subtiles_per_batch` in config
2. Decrease `memory_cleanup_interval` 
3. Process fewer tiles at once with `--max-tiles`

### Verify Outputs
```bash
# Count output files
ls data/h3_2021_res8_modular/*.json | wc -l

# Check file sizes
du -h data/h3_2021_res8_modular/*.json | head
```

## Design Philosophy

This system prioritizes:
1. **Reliability** - Checkpoints prevent data loss
2. **Simplicity** - No complex scheduling or distribution
3. **Transparency** - Clear progress and logging
4. **Resumability** - Seamless recovery from failures
5. **Archival** - Complete run history for reproducibility

The modular approach ensures that even with limited resources, the full Del Norte dataset can be processed reliably over several days.