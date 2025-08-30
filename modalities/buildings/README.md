# Buildings Modality

**Building Footprint Density and Morphology**

## ✅ Status: Complete

This modality provides building density and morphology features, primarily the Floor Space Index (FSI).

Unlike other modalities, this functionality is not implemented as a `ModalityProcessor` class. Instead, it is provided by a standalone preprocessing script: `scripts/preprocessing/setup_density.py`. This script is intended to be run as part of the initial data setup for a study area.

## Features
- Calculates building density (FSI) for H3 hexagons.
- Uses a shapefile of building footprints as the data source.
- Spatially joins building data with H3 regions to calculate metrics.
- Saves the calculated density data to be used in later processing steps.

## Generated Features
- `FSI_24`: The calculated Floor Space Index, representing building density.
- `building_volume`: The total building volume within a hexagon.
- `total_area`: The total area of building footprints within a hexagon.

## Example Usage

The building density calculation is run via the command line:

```bash
python scripts/preprocessing/setup_density.py \
  --city_name south_holland \
  --input_dir data/preprocessed/south_holland_base \
  --output_dir data/preprocessed/south_holland_base \
  --building_data "data/preprocessed [TODO SORT & CLEAN UP]/density/PV28__00_Basis_Bouwblok.shp" \
  --resolutions 8,9,10
```

This script reads the H3 regions for a city, calculates the FSI for each hexagon, and saves the results back to the study area's directory.

## Data Sources
- Building footprint shapefiles (e.g., `PV28__00_Basis_Bouwblok.shp` for the Netherlands).
