# AlphaEarth Processing Scripts

Consolidated scripts for querying and processing AlphaEarth satellite embeddings.

## Overview

AlphaEarth provides 64-dimensional embedding vectors from satellite imagery at 10m resolution. These scripts handle:
1. Querying and exporting data from Google Earth Engine
2. Processing local TIFF files to H3 hexagons
3. Handling large tiled datasets with memory efficiency

## Scripts

### `query_gee.py` - Google Earth Engine Query & Export

Query and export AlphaEarth data for any geographic region.

```bash
# Export by bounding box
python query_gee.py --bbox -124.7 38.6 -117.3 43.4 --year 2023

# Export by region name
python query_gee.py --region "Netherlands" --years 2021 2022 2023

# Check availability only
python query_gee.py --region "Oregon" --check-only

# Test with limited tiles
python query_gee.py --bbox -124 42 -123 43 --max-tiles 5 --dry-run
```

### `process_tiff_to_h3.py` - TIFF to H3 Processor

Process AlphaEarth GeoTIFF files and aggregate to H3 hexagons.

```bash
# Process all TIFFs from Google Drive
python process_tiff_to_h3.py --source-dir "G:/My Drive/AlphaEarth_Cascadia"

# Process with specific H3 resolution
python process_tiff_to_h3.py --source-dir ./data/tiffs --h3-res 9

# Resume from checkpoint
python process_tiff_to_h3.py --source-dir ./data/tiffs --resume

# Test with limited tiles
python process_tiff_to_h3.py --source-dir ./data/tiffs --max-tiles 5
```

## Configuration

Edit `config.yaml` to customize:
- Google Earth Engine project settings
- H3 resolutions (7-10)
- Processing parameters (chunk size, memory limits)
- Study area definitions

## Workflow

### 1. Export from Earth Engine
```bash
# Authenticate once
earthengine authenticate --project=boreal-union-296021

# Export region
python query_gee.py --region "Del Norte" --year 2023
```

Monitor exports at: https://code.earthengine.google.com/tasks

### 2. Sync to Local
Sync Google Drive folder to local machine:
- Windows: Use Google Drive desktop app
- Linux: Use rclone or gdrive CLI

### 3. Process to H3
```bash
python process_tiff_to_h3.py \
  --source-dir "G:/My Drive/AlphaEarth_Export" \
  --h3-res 8
```

## Features

### Memory Efficiency
- Processes large TIFFs in 256×256 pixel chunks
- Automatic garbage collection
- Configurable memory limits

### Checkpointing
- Saves progress after each batch
- Resume from interruptions
- Intermediate results cached

### Spatial Operations
- SRAI-based H3 pre-regionalization
- KDTree indexing for fast pixel-to-hexagon mapping
- Overlap handling for tiled datasets

### Output Format
- Parquet files with H3 index and 64 embedding dimensions
- Columns: `h3_index`, `A00`-`A63`, `pixel_count`, `tile_count`
- Optional geometry column for GeoDataFrame

## H3 Resolution Guide

| Resolution | Hex Area | Use Case |
|------------|----------|----------|
| 7 | ~5.16 km² | Regional analysis |
| 8 | ~0.74 km² | City-level analysis |
| 9 | ~0.11 km² | Neighborhood analysis |
| 10 | ~0.015 km² | Block-level analysis |

## Requirements

```bash
pip install ee rioxarray srai geopandas scipy h3 pyyaml
```

## Troubleshooting

### Earth Engine Authentication
```bash
# If authentication fails
earthengine authenticate --project=YOUR_PROJECT_ID
```

### Memory Issues
Adjust `subtile_size` in config.yaml (smaller = less memory)

### Missing SRAI
```bash
pip install srai[all]
```