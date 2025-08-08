# Study Area Management Tools

This directory contains CLI tools for creating, managing, and validating study area configurations for the UrbanRepML filtering system.

## Overview

The study area filtering system enables flexible definition of geographic regions with adaptive H3 resolution based on bioregional characteristics and computational constraints. This is particularly valuable for the Cascadia GEO-INFER dataset where different regions require different levels of analysis detail.

## Available Tools

### `create_study_area.py`
Interactive CLI tool for creating new study area configurations.

```bash
# Interactive creation
python create_study_area.py

# Create from template
python create_study_area.py --template agricultural
```

**Features:**
- Interactive guided configuration
- Bioregional context setup (agriculture, forestry, watershed, conservation)
- Custom resolution rules based on area characteristics
- Validation of geographic bounds and processing parameters

### `list_study_areas.py`
List and analyze existing study area configurations.

```bash
# Basic listing
python list_study_areas.py

# Detailed view with validation
python list_study_areas.py --detailed --validate

# Filter by bioregion type
python list_study_areas.py --bioregion agriculture

# Export as JSON
python list_study_areas.py --format json
```

**Features:**
- Tabular and detailed display formats
- Configuration validation
- Bioregion filtering and summary statistics
- JSON export for programmatic use

## Study Area Configuration Schema

Study areas are defined using YAML files with the following structure:

```yaml
name: study_area_name
description: "Description of the study area..."

# Geographic boundaries
geographic_bounds:
  bounds_type: bbox  # or counties, circle, polygon, shapefile
  definition: 
    north: 45.0
    south: 44.0
    east: -122.0
    west: -123.0
  buffer_km: 1.0

# Bioregional context (optional but recommended)
bioregional_context:
  bioregion_type: agriculture
  primary_ecosystem: agricultural
  management_focus: [crops, water, sustainability]
  primary_crops: [hazelnuts, wine_grapes]
  farming_type: mixed

# Adaptive resolution rules
resolution_rules:
  - name: intensive_agriculture
    condition: "FSI_24 >= 0.1"
    resolution: 9
    priority: 90
    description: "High-density agricultural areas"

# Processing configuration
default_resolution: 8
max_memory_gb: 16.0
max_hexagons_per_chunk: 100000
enable_chunking: true
```

## Bioregional Categories

The system supports five main bioregional categories aligned with GEO-INFER goals:

### 🌾 Agriculture
- **Focus**: Crop production, sustainable farming practices
- **Ecosystems**: Agricultural lands, mixed farming systems
- **Management**: Crops, water efficiency, carbon sequestration
- **Examples**: Willamette Valley specialty crops, Central Valley agriculture

### 🌲 Forestry  
- **Focus**: Timber management, conservation, ecosystem services
- **Ecosystems**: Conifer forests, mixed forests, oak woodlands
- **Management**: Sustainable timber, carbon storage, biodiversity
- **Examples**: Coast Range forests, Cascade timber lands

### 💧 Watershed
- **Focus**: Water resource management across land uses
- **Ecosystems**: Mixed use areas with water features
- **Management**: Water quality, drought resilience, flood control
- **Examples**: Klamath Basin, Columbia River watershed

### 🦅 Conservation
- **Focus**: Ecosystem protection and biodiversity preservation
- **Ecosystems**: Old-growth forests, wetlands, pristine areas
- **Management**: Habitat protection, climate adaptation
- **Examples**: Redwood groves, wilderness areas

### 🔄 Mixed Use
- **Focus**: Integrated management of multiple land uses
- **Ecosystems**: Agricultural-forest interfaces, suburban-rural transitions
- **Management**: Multi-objective optimization, sustainable development
- **Examples**: Urban-agriculture interfaces, rural development areas

## Resolution Rules

Resolution rules determine the appropriate H3 resolution (7-11) for different areas based on characteristics:

- **Higher resolution (9-10)**: Complex areas needing detailed analysis
  - Intensive agriculture with infrastructure
  - Forest-agriculture interfaces
  - Conservation priority areas
  - Water management infrastructure

- **Standard resolution (8)**: Typical areas with moderate complexity
  - Extensive agriculture
  - Managed forests
  - Rural communities

- **Lower resolution (7)**: Sparse areas for computational efficiency
  - Extensive rangelands
  - Wilderness areas with minimal human impact
  - Large homogeneous areas

## Usage Examples

### Creating an Agricultural Study Area

```bash
python create_study_area.py
# Follow interactive prompts:
# - Name: "salinas_valley_agriculture" 
# - Bioregion: agriculture
# - Bounds: bbox around Salinas Valley
# - Crops: lettuce, strawberries, wine_grapes
# - Resolution rules for intensive vs extensive agriculture
```

### Listing Areas by Type

```bash
# Show all agricultural study areas
python list_study_areas.py --bioregion agriculture --detailed

# Validate all configurations
python list_study_areas.py --validate
```

### Integration with Experiments

```bash
# Use study area in processing pipeline
python ../experiments/run_experiment.py \
  --study_area willamette_valley_agriculture \
  --run_training \
  --epochs 200
```

## File Organization

```
config/study_areas/
├── willamette_valley_agriculture.yaml
├── coast_range_forestry.yaml  
├── klamath_watershed.yaml
├── eastern_oregon_rangelands.yaml
├── north_coast_fog_belt.yaml
└── [custom_study_areas].yaml
```

## Next Steps

1. **Create custom study areas** for your specific analysis needs
2. **Validate configurations** before running large experiments
3. **Integrate with existing pipelines** using the `--study_area` parameter
4. **Monitor computational requirements** and adjust chunking parameters as needed

This system transforms the challenge of processing large geographic datasets into manageable, scientifically meaningful analysis units that align with real-world ecological and agricultural management boundaries.