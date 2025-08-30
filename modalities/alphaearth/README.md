# AlphaEarth Modality

**Satellite Imagery Embeddings Processor**

## ✅ Status: Complete

This modality processes pre-downloaded AlphaEarth GeoTIFF files into H3 hexagon-based embeddings. AlphaEarth provides 64-dimensional embeddings derived from satellite imagery. This processor is optimized for handling large, tiled datasets with memory efficiency.

## Features
- Processes local AlphaEarth GeoTIFF files.
- Aggregates 64-dimensional pixel embeddings into H3 hexagons by calculating the mean.
- Handles multi-tile datasets, correctly averaging data in overlapping regions.
- **Memory Efficient**: Uses chunked (subtile) processing and parallel workers to handle large raster files.
- **Checkpointing**: Creates intermediate files for each tile, allowing processing to be resumed.

## Generated Features
- `A00` - `A63`: The 64 dimensions of the AlphaEarth embedding.
- `pixel_count`: The number of source pixels that contributed to the hexagon's embedding.
- `tile_count`: The number of source TIFF files that contributed to the hexagon's embedding.

## Example Usage
```python
from modalities.alphaearth import AlphaEarthProcessor
from pathlib import Path

# Configuration for the processor
config = {
    # Directory containing the source AlphaEarth .tif files
    'source_dir': 'path/to/your/alphaearth_tiffs',
    'output_dir': 'data/processed/embeddings/alphaearth',
    'max_workers': 4, # Number of parallel processes
}

# Initialize the processor
processor = AlphaEarthProcessor(config)

# 1. Locate the raw data
# For AlphaEarth, this step just validates the source_dir from the config
raw_data_path = processor.download(study_area="cascadia")

# 2. Process the raw TIFFs into an H3-indexed GeoDataFrame
# This is the most intensive step
h3_gdf = processor.process(
    raw_data_path=raw_data_path,
    h3_resolution=10,
    year_filter='2023' # Optional: filter files by year in filename
)

# 3. Save the final embeddings to a parquet file
output_path = processor.save_embeddings(
    h3_gdf,
    filename=f"alphaearth_embeddings_res10.parquet"
)

print(f"AlphaEarth embeddings saved to: {output_path}")
```

## Data Sources
- Pre-downloaded AlphaEarth GeoTIFF files. This processor does not handle downloading from Google Earth Engine.
