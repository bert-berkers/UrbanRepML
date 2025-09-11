# UrbanRepML Data Catalog

## Overview
Complete inventory of all data assets, processing status, and file locations within the UrbanRepML project.

## Netherlands Dataset

### Raw Data Sources

#### AlphaEarth Satellite Imagery
- **Location**: `G:/My Drive/AlphaEarth_Netherlands/`
- **Format**: GeoTIFF (64-band embeddings)
- **Years Available**: 2020, 2021, 2022, 2023
- **Files per year**: 99 tiles
- **Total files**: 396 TIFFs
- **Tile specification**: 3072x3072 pixels at 10m resolution
- **Coordinate system**: EPSG:28992 (Dutch RD)
- **Coverage area**: Northern Netherlands (~20% of country)
- **Geographic extent**: 
  - Longitude: 3.258° to 7.244°
  - Latitude: 52.997° to 53.560°
- **File size range**: 4MB - 1.3GB per tile
- **Total raw data size**: ~190GB

### Processed Data

#### AlphaEarth Embeddings

##### Resolution 8 (2022) - COMPLETED
- **File**: `data/processed/embeddings/alphaearth/netherlands_res8_2022.parquet`
- **Created**: 2025-08-30 12:26
- **Processing time**: 7 minutes
- **Hexagons**: 58,127
- **Embedding dimensions**: 64 (A00-A63)
- **Spatial resolution**: ~0.61 km² per hexagon
- **Coverage**: Northern Netherlands
- **File size**: 34MB
- **Additional data**: pixel_count, tile_count per hexagon
- **K-means clustering**: Completed for K=8,10,12
- **Silhouette scores**: K8=0.207, K10=0.178, K12=0.177

##### Resolution 10 (2022) - PENDING
- **Target file**: `data/processed/embeddings/alphaearth/netherlands_res10_2022.parquet`
- **Expected hexagons**: 200,000-500,000
- **Spatial resolution**: ~0.013 km² per hexagon
- **Estimated processing time**: 15-30 minutes
- **Status**: Ready to process

#### H3 Regional Boundaries
- **Location**: `data/processed/h3_regions/netherlands/`
- **Boundary file**: `netherlands_boundary.geojson`
- **Coverage file**: `alphaearth_coverage.geojson`

##### H3 Hexagon Grids (All Resolutions)
| Resolution | File | Hexagon Count | Area per Hex | Total Coverage |
|------------|------|---------------|--------------|----------------|
| 5 | `netherlands_h3_res5.parquet` | 431 | 215.90 km² | 93,054 km² |
| 6 | `netherlands_h3_res6.parquet` | 2,812 | 29.75 km² | 83,650 km² |
| 7 | `netherlands_h3_res7.parquet` | 19,139 | 4.54 km² | 86,873 km² |
| 8 | `netherlands_h3_res8.parquet` | 132,603 | 0.61 km² | 80,796 km² |
| 9 | `netherlands_h3_res9.parquet` | 924,348 | 0.09 km² | 81,264 km² |
| 10 | `netherlands_h3_res10.parquet` | 6,460,557 | 0.013 km² | 81,796 km² |

- **Created**: 2025-08-30 11:54
- **Generation method**: SRAI H3Regionalizer
- **Coordinate system**: EPSG:4326 (WGS84)
- **File format**: Parquet (efficient storage)

### Results and Visualizations

#### K-means Clustering Visualizations (Resolution 8, 2022)
- **Location**: `results/plots/netherlands/`
- **Files**:
  - `alphaearth_clusters_k8_2022.png` (4.2MB)
  - `alphaearth_clusters_k10_2022.png` (3.9MB)
  - `alphaearth_clusters_k12_2022.png` (4.0MB)
- **Format**: High-resolution PNG (300 DPI)
- **Projection**: Dutch RD (EPSG:28992)
- **Features**: North arrow, scale bar, legend, coordinate grid

#### Clustered Embeddings
- **File**: `results/embeddings/netherlands/alphaearth_res8_clustered_2022.parquet`
- **Content**: Embeddings + cluster assignments for K=8,10,12
- **Size**: 34MB

## Cascadia Dataset (Historical)

### Processed Data
- **Location**: `data/processed/embeddings/alphaearth/`
- **Files**:
  - `alphaearth_h3_res8_20250826_201012.parquet` (588KB)
  - `alphaearth_h3_res8_20250827_192159.parquet` (113MB)
  - `cluster_assignments_k10.parquet` (1.2MB)

## Data Quality Metrics

### Netherlands AlphaEarth Coverage Assessment
- **Expected Netherlands area**: 41,543 km²
- **AlphaEarth coverage**: ~8,500 km² (20.5%)
- **Missing coverage**: Southern and central Netherlands (79.5%)
- **Recommendation**: Queue Earth Engine downloads for complete coverage

### File Size Distribution
| Data Type | Size Range | Storage Format |
|-----------|------------|----------------|
| Raw TIFFs | 4MB - 1.3GB | GeoTIFF |
| H3 Hexagons | 37KB - 737MB | Parquet |
| Embeddings | 34MB - 113MB | Parquet |
| Visualizations | 1MB - 4.2MB | PNG (300 DPI) |

## Processing Capability Status

### Working Processors
- **AlphaEarth**: Fully functional
  - Tested: Netherlands 2022 at resolution 8
  - Performance: 58,127 hexagons in 7 minutes
  - Ready for: Any region with GeoTIFF data

### Other Modalities (Untested)
- **POI**: Processor exists (`modalities/poi/`)
- **GTFS**: Processor exists (`modalities/gtfs/`)
- **Roads**: Processor exists (`modalities/roads/`)
- **Buildings**: Processor exists (`modalities/buildings/`)
- **Streetview**: Processor exists (`modalities/streetview/`)

## File Organization Standards

### Data Directory Structure
```
data/
├── processed/
│   ├── embeddings/
│   │   └── alphaearth/
│   │       ├── {region}_res{N}_{year}.parquet        # Main embeddings
│   │       └── {region}_res{N}_{year}_metadata.json  # Processing metadata
│   ├── h3_regions/
│   │   └── {region}/
│   │       ├── {region}_boundary.geojson
│   │       ├── {region}_h3_res{N}.parquet
│   │       └── h3_summary_stats.csv
│   └── checkpoints/
│       └── {process_type}_{year}/
│           └── intermediate_*.json
```

### Results Directory Structure
```
results/
├── plots/
│   └── {region}/
│       └── {modality}_clusters_k{N}_{year}.png
├── embeddings/
│   └── {region}/
│       └── {modality}_res{N}_clustered_{year}.parquet
└── statistics/
    └── {region}/
        └── {modality}_processing_summary_{year}.json
```

### Logging Directory Structure
```
logs/
├── alphaearth/
│   └── netherlands_res{N}_{year}_YYYYMMDD_HHMMSS.log
├── clustering/
└── regions/
```

## Data Processing Guidelines

### Naming Conventions
- **Regions**: Use lowercase with underscores (e.g., `netherlands`, `south_holland`)
- **Years**: 4-digit format (e.g., `2022`)
- **Resolutions**: Use `res{N}` format (e.g., `res8`, `res10`)
- **Clustering**: Use `k{N}` format (e.g., `k8`, `k10`)

### File Formats
- **Spatial data**: Parquet for efficiency, GeoJSON for boundaries
- **Visualizations**: PNG at 300 DPI for publication quality
- **Metadata**: JSON for structured information
- **Logs**: Plain text with timestamps

### Processing Configuration Standards
- **subtile_size**: 512 for res 8, 256 for res 10
- **min_pixels_per_hex**: 5 for res 8, 3 for res 10
- **max_workers**: 10 (adjust based on system)
- **checkpoint_interval**: Every 10 tiles processed

## Quality Assurance Checklist

### Before Processing
- [ ] Verify raw data exists and is accessible
- [ ] Check coordinate system (should be known CRS)
- [ ] Validate year filter matches available files
- [ ] Ensure output directories exist

### After Processing
- [ ] Validate H3 indices are all valid
- [ ] Check for missing embedding values (NaN)
- [ ] Verify hexagon count is reasonable for area
- [ ] Test file loading and basic operations
- [ ] Generate processing summary statistics

### Documentation Updates
- [ ] Update DATA_CATALOG.md with new files
- [ ] Add entry to PROCESSING_LOG.md
- [ ] Update MODALITY_STATUS.md if needed
- [ ] Log session details in DEVELOPMENT_LOG.md

---

*Last updated: 2025-08-30*
*Data catalog version: 1.0*