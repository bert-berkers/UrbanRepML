# AlphaEarth Intermediate Processing Files

This directory contains intermediate processing artifacts from AlphaEarth TIFF processing.

## Directory Structure

```
alphaearth_intermediate/
├── cascadia_2021/           # Cascadia coastal forests 2021 processing
│   ├── checkpoints/         # Progress tracking
│   └── *.json              # Individual tile results (705 files)
└── [other_study_areas]/     # Future processing runs
```

## Cascadia 2021 Processing

- **705 JSON files**: Individual tile processing results
- **Study area**: Cascadia coastal forests (west of -121° longitude)
- **Processing date**: August 2025
- **H3 resolution**: 8
- **Coverage**: ~592 coastal TIFF files processed

## File Format

Each JSON file contains H3 hexagon data for one AlphaEarth tile:
```json
{
  "h3_index_string": {
    "embedding": [64 float values],
    "pixel_count": integer,
    "tile_count": integer
  }
}
```

## Usage

These intermediate files can be:
1. **Reprocessed**: Combined into different final datasets
2. **Filtered**: By geographic bounds or quality thresholds  
3. **Analyzed**: For processing quality and coverage
4. **Resumed**: If processing is interrupted

## Final Outputs

Intermediate files are merged into final datasets in:
- `data/processed/embeddings/alphaearth/`

## Preservation

⚠️ **Do not delete these files** - they represent hours of computational work and enable:
- Re-processing with different parameters
- Quality analysis and debugging
- Incremental updates when new tiles are added