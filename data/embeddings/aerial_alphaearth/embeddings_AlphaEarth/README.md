# AlphaEarth Aerial Embeddings Processing

This directory contains the workflow for processing Google's AlphaEarth satellite embeddings for the Netherlands region. The workflow exports data from Google Earth Engine to Google Drive, which you then sync locally and process into H3 hexagon format for use in the UrbanRepML pipeline.

## Overview

AlphaEarth provides 64-dimensional embeddings derived from satellite imagery at 10m resolution. These embeddings capture semantic information about land use, urban features, and environmental characteristics from aerial imagery.

## ⚠️ Important Workflow Understanding

This is a **three-step process** that many first-time users find confusing:

1. **Export to Cloud** (`export_to_drive.py`): Tells Google Earth Engine to process and export data to your Google Drive
2. **Manual Sync**: You manually sync the Google Drive folder to your local machine
3. **Local Processing** (`process_alphaearth_2023.py`): Process the synced files into H3 format

**Key Point**: The `export_to_drive.py` script does NOT download files to your computer! It initiates a cloud-to-cloud transfer (Earth Engine → Google Drive). The actual download happens when you sync Google Drive.

## Workflow Details

### 1. Cloud Export (`export_to_drive.py`)

**What this script actually does**:
- Submits export tasks to Google Earth Engine's processing servers
- These tasks run in the cloud and can take hours for large areas
- Results are automatically saved to your Google Drive (not your local machine!)
- The script can monitor task progress, but the actual processing happens on Google's servers

**How to use**:
```bash
python export_to_drive.py
```

**What happens next**:
1. Script creates export tasks in Google Earth Engine
2. Tasks appear in the Earth Engine Code Editor's **Tasks tab** (right panel)
3. You can monitor progress in three ways:
   - In the script itself (if you choose monitoring option)
   - In the [Earth Engine Code Editor Tasks tab](https://code.earthengine.google.com/)
   - In the [Google Cloud Console Tasks page](https://console.cloud.google.com/earth-engine/tasks)

**Task Management**:
- Tasks show as `READY` → `RUNNING` → `COMPLETED` or `FAILED`
- Each task has an ID like `3DNU363IM57LNU4SDTMB6I33`
- Large exports can take 1-4 hours depending on area size
- Failed tasks will retry automatically up to 5 times
- You'll receive an email when tasks complete (if notifications enabled)

**Requirements**:
- Google Earth Engine account
- First-time setup: `earthengine authenticate`
- Google Cloud project: `boreal-union-296021`

### 2. Google Drive Sync (Manual Step)

**This is the actual "download" step!**

After Earth Engine completes the export:
1. Files appear in your Google Drive: `My Drive/EarthEngine_Exports_Corrected/`
2. You must sync this folder to your local machine
3. Options for syncing:
   - **Google Drive Desktop app** (recommended): Automatically syncs the folder
   - **Manual download**: Download each .tif file from Google Drive web interface
   - **rclone**: Command-line tool for syncing Google Drive

**Expected files**:
- Format: `Netherlands_Embedding_YYYY_Mosaic-*.tif`
- Size: Multiple GB per file (prepare ~50GB disk space for one year)
- Count: ~100+ tiles for full Netherlands coverage

**Local sync location** (configure in processing script):
```
G:\My Drive\EarthEngine_Exports_Corrected\  # Example with Google Drive Desktop
```

### 3. Data Processing (`process_alphaearth_2023.py`)

**Purpose**: Convert raw GeoTIFF tiles into H3 hexagon format for integration with UrbanRepML.

**What it does**:
- Loads large .tif files one by one (memory efficient processing)
- Extracts 64-dimensional embeddings from each 10m×10m pixel
- Maps pixels to H3 resolution 10 hexagons (~15m edge length)
- Aggregates multiple pixels per hexagon using mean averaging
- Saves processed data as parquet files
- Stitches all tiles together into final dataset

**Processing steps**:
1. **Tile Processing**: Each .tif file is processed individually to manage memory
2. **Coordinate Transformation**: Converts from Dutch CRS to WGS84 lat/lon
3. **H3 Mapping**: Maps each pixel to its corresponding H3 cell
4. **Aggregation**: Combines pixels within same H3 cell using mean
5. **Stitching**: Combines all processed tiles into single dataset

**Output**:
- Individual tile files: `processed/h3_aggregated/*.parquet`
- Final dataset: `processed/netherlands_2023_h3_res10.parquet`
- Statistics: `processed/netherlands_2023_h3_res10_stats.json`

## File Structure

```
embeddings_AlphaEarth/
│
├── README.md                          # This documentation
├── export_to_drive.py                 # Google Earth Engine export script
├── process_alphaearth_2023.py         # Main processing pipeline
│
└── processed/                         # Output directory
    ├── h3_aggregated/                 # Individual processed tiles
    │   ├── Netherlands_*_h3.parquet   # H3 data per tile
    │   └── Netherlands_*_metadata.json # Tile metadata
    ├── netherlands_2023_h3_res10.parquet # Final stitched dataset
    └── netherlands_2023_h3_res10_stats.json # Processing statistics
```

## Usage Summary

### Step 1: Export from Google Earth Engine to Google Drive
```bash
python export_to_drive.py
# This starts cloud processing tasks - check Tasks tab in Earth Engine Code Editor
```

### Step 2: Wait and Monitor
- Check task progress at https://code.earthengine.google.com/ (Tasks tab)
- Or at https://console.cloud.google.com/earth-engine/tasks
- Tasks typically take 1-4 hours for Netherlands

### Step 3: Sync Google Drive folder locally
- Use Google Drive Desktop app to sync `EarthEngine_Exports_Corrected` folder
- Or manually download .tif files from Google Drive web interface

### Step 4: Process the synced data
```bash
python process_alphaearth_2023.py
# Make sure to update INPUT_DIR in the script to your local sync path
```

## Configuration

### Export Configuration (`export_to_drive.py`)
- **Google Cloud Project**: `boreal-union-296021`
- **Export Folder**: `EarthEngine_Exports_Corrected` (in Google Drive)
- **Years**: 2020, 2021, 2022, 2023
- **Resolution**: 10 meters (native AlphaEarth resolution)
- **CRS**: EPSG:28992 (Amersfoort / RD New - Netherlands standard)

### Processing Configuration (`process_alphaearth_2023.py`)
- **Input Directory**: `G:\My Drive\EarthEngine_Exports_Corrected`
- **Output Directory**: `processed/`
- **H3 Resolution**: 10 (~15m hexagon edge length)
- **Processing**: Memory-efficient tile-by-tile approach

## Data Specifications

### AlphaEarth Embeddings
- **Dimensions**: 64 features per pixel
- **Resolution**: 10m × 10m pixels
- **Source**: Google Satellite Embedding V1 Annual
- **Coverage**: Netherlands boundary (from FAO GAUL dataset)

### H3 Processing
- **Target Resolution**: H3 level 10
- **Hexagon Size**: ~15m edge length
- **Aggregation**: Mean of all pixels within hexagon
- **Coordinate System**: WGS84 (EPSG:4326)

## Integration with UrbanRepML

The processed data integrates seamlessly with the main UrbanRepML pipeline:

1. **H3 Alignment**: Uses same H3 resolution 10 as other features
2. **Format**: Parquet files with standardized schema
3. **Geometry**: Proper H3 hexagon geometries for spatial analysis
4. **Features**: 64 embedding dimensions as `embed_0` to `embed_63`

## Memory and Performance

### Processing Requirements
- **RAM**: 16GB+ recommended for large tiles
- **Storage**: ~50GB for Netherlands 2023 data
- **Processing Time**: 2-4 hours for full Netherlands dataset

### Optimizations
- Tile-by-tile processing prevents memory overflow
- Efficient H3 aggregation using numpy operations
- Parquet format for fast I/O
- Garbage collection between tiles

## Data Quality

### Validation
- Pixel count tracking per H3 cell
- Metadata preservation (bounds, dimensions, CRS)
- Statistics generation for quality assessment

### Quality Checks
- NoData handling (skips null/zero embeddings)
- Coordinate transformation validation
- H3 cell coverage verification

## Troubleshooting

### Common Issues
1. **Authentication**: Run `earthengine authenticate` if export fails
2. **Google Drive Sync**: Ensure local folder is properly synced
3. **Memory Errors**: Reduce batch size in processing script
4. **Missing Files**: Check Google Drive sync status

### File Locations
- **Google Drive**: `G:\My Drive\EarthEngine_Exports_Corrected\`
- **Local Processing**: `C:\Users\...\UrbanRepML\data\embeddings_AlphaEarth\processed\`

## Notes

- **No Need for**: The original `download.py`, `earth_engine_script.js`, and `exploratory_data_analysis.py` are no longer needed
- **Experiments**: Exploratory analysis has been moved to separate experiments directory
- **Robustness**: The workflow is designed to handle interruptions and resume processing
- **Caching**: Processed tiles are cached to avoid reprocessing on subsequent runs