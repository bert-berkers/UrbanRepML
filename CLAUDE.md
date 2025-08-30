# CLAUDE.md - Developer Instructions

Instructions for Claude Code and developers working with the UrbanRepML project.

## 🏗️ Project Architecture

### Core Components

```
modalities/          # Data processing pipelines (one per data source)
├── alphaearth/     # Satellite imagery → H3 embeddings
├── poi/            # Points of interest → count, diversity & contextual embeddings
├── gtfs/           # Transit data → accessibility features (PLANNED)
├── roads/          # OSM networks → connectivity & centrality metrics
├── buildings/      # Footprints → FSI density metrics (via script)
└── streetview/     # Street imagery → visual features (PLANNED)

data/               # ALL data storage (no data in code directories)
├── raw/            # Original downloads
├── processed/      # H3 hexagon embeddings + networks
└── cache/          # Temporary processing files

urban_embedding/    # ML pipeline (Python scripts only)
├── pipeline.py     # Multi-modal fusion + training
├── model.py        # UrbanUNet GNN architecture
└── analytics.py    # Clustering + visualization

study_areas/        # Geospatial research areas
├── configs/        # YAML boundary definitions
├── cascadia/       # Coastal forest study
└── netherlands/    # Urban density studies
```

## 🔧 Development Practices

### Environment Setup

```bash
# Clone and setup
git clone https://github.com/bertberkers/UrbanRepML.git
cd UrbanRepML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install all optional dependencies
pip install -e ".[dev,viz,ml]"
```

### Development Commands

```bash
# Run tests
python -m pytest tests/ -v

# Format code (before committing)
black urban_embedding/ modalities/ --line-length 100

# Lint code
flake8 urban_embedding/ modalities/ --max-line-length 100

# Type checking
mypy urban_embedding/ modalities/ --ignore-missing-imports

# Generate documentation
sphinx-build -b html docs/source docs/build
```

## 📁 Data Management

### File Locations

**ALWAYS store data in these locations:**
- **Raw data**: `data/raw/{modality}/`
- **Processed embeddings**: `data/processed/embeddings/{modality}/`
- **OSM networks**: `data/processed/networks/`
- **H3 regions**: `data/processed/h3_regions/`
- **Temporary files**: `data/cache/`

**NEVER store data in:**
- `urban_embedding/` (scripts only)
- `modalities/` (processing code only)
- Project root directory
- Inside study area code directories

### Processing Large Datasets

```python
# Use chunked processing for large files
from modalities.alphaearth import AlphaEarthProcessor

processor = AlphaEarthProcessor(config={
    'chunk_size': 1000,  # Process 1000 hexagons at a time
    'max_workers': 10,   # Parallel processing threads
    'memory_limit': '8GB'  # Set memory constraints
})

# Process with progress tracking
with tqdm(total=total_tiles) as pbar:
    processor.run_pipeline(
        progress_callback=lambda x: pbar.update(1)
    )
```

## 🧩 Implementing New Modalities

### 1. Create Modality Structure

```python
# modalities/new_modality/__init__.py
from .processor import NewModalityProcessor

__all__ = ['NewModalityProcessor']
```

### 2. Implement ModalityProcessor Interface

```python
# modalities/new_modality/processor.py
from modalities.base import ModalityProcessor
import pandas as pd
import geopandas as gpd

class NewModalityProcessor(ModalityProcessor):
    """Process new data type into H3 embeddings."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.validate_config()
    
    def validate_config(self):
        """Validate required configuration parameters."""
        required = ['source_dir', 'output_dir']
        for param in required:
            if param not in self.config:
                raise ValueError(f"Missing required config: {param}")
    
    def load_data(self, study_area: str) -> gpd.GeoDataFrame:
        """Load raw data for study area."""
        # Implementation here
        pass
    
    def process_to_h3(self, data: gpd.GeoDataFrame, 
                     h3_resolution: int) -> pd.DataFrame:
        """Convert data to H3 hexagon embeddings."""
        # Implementation here
        pass
    
    def run_pipeline(self, study_area: str, 
                    h3_resolution: int,
                    output_dir: str) -> str:
        """Execute complete processing pipeline."""
        data = self.load_data(study_area)
        embeddings = self.process_to_h3(data, h3_resolution)
        output_path = self.save_embeddings(embeddings, output_dir)
        return output_path
```

### 3. Add Configuration

```yaml
# modalities/new_modality/config.yaml
default:
  chunk_size: 1000
  max_workers: 10
  
data_sources:
  primary: "https://data-source.com/api"
  fallback: "local/cache"

processing:
  normalize: true
  aggregation: "mean"
  missing_value_strategy: "interpolate"
```

## 🌍 Study Area Management

### Creating New Study Areas

```python
# study_areas/tools/create_study_area.py
from study_areas.tools import StudyAreaManager

manager = StudyAreaManager()

# Define study area with multiple boundaries
manager.create_study_area(
    name='pacific_northwest',
    boundaries={
        'main': (-125.0, 42.0, -117.0, 49.0),
        'urban_cores': [
            ('seattle', (-122.5, 47.4, -122.2, 47.8)),
            ('portland', (-122.8, 45.4, -122.5, 45.6))
        ]
    },
    h3_resolutions=[8, 9, 10],
    description='Pacific Northwest urban-forest interface'
)
```

### Study Area Configuration

```yaml
# study_areas/configs/pacific_northwest.yaml
name: pacific_northwest
bioregion: temperate_rainforest
boundaries:
  bbox: [-125.0, 42.0, -117.0, 49.0]
  filter: "longitude < -121"  # West of Cascade crest

h3_resolutions:
  regional: 8
  local: 9
  detailed: 10

modalities:
  alphaearth:
    enabled: true
    years: [2021, 2023, 2024]
  poi:
    enabled: true
    categories: ['natural', 'amenity', 'landuse']
  gtfs:
    enabled: false  # Rural areas lack transit
```

## 🐛 Debugging & Troubleshooting

### Common Issues

1. **Memory errors with large rasters**
```python
# Use windowed reading
import rasterio
from rasterio.windows import Window

with rasterio.open('large_file.tif') as src:
    for window in src.block_windows():
        data = src.read(window=window)
        # Process chunk
```

2. **H3 resolution mismatches**
```python
# Always validate H3 cells
import h3

def validate_h3_cells(cells):
    return all(h3.h3_is_valid(cell) for cell in cells)
```

3. **Coordinate system issues**
```python
# Always work in WGS84 for H3
gdf = gdf.to_crs('EPSG:4326')
```

### Performance Optimization

```python
# Use multiprocessing for CPU-bound tasks
from multiprocessing import Pool
from functools import partial

def process_tile(tile_path, config):
    # Processing logic
    pass

with Pool(processes=10) as pool:
    process_func = partial(process_tile, config=config)
    results = pool.map(process_func, tile_paths)
```

## 🔄 Git Workflow

### Branch Strategy
- `main`: Stable releases
- `develop`: Active development
- `feature/*`: New features
- `fix/*`: Bug fixes
- `experiment/*`: Research branches

### Commit Messages
```bash
# Format: <type>(<scope>): <subject>
feat(alphaearth): add multi-temporal processing
fix(h3): correct resolution conversion bug
docs(readme): update installation instructions
refactor(pipeline): simplify data loading
test(poi): add integration tests
```

## 📝 Code Style Guidelines

### Python Style
- Follow PEP 8 with 100 char line limit
- Use type hints for all functions
- Docstrings in Google style
- No inline comments unless critical

### Example Function
```python
def process_h3_embeddings(
    data: pd.DataFrame,
    h3_resolution: int,
    aggregation: str = 'mean'
) -> pd.DataFrame:
    """Process data into H3 hexagon embeddings.
    
    Args:
        data: Input dataframe with geometry column
        h3_resolution: H3 resolution level (0-15)
        aggregation: Aggregation method for multiple points
        
    Returns:
        DataFrame with H3 index and aggregated features
        
    Raises:
        ValueError: If h3_resolution is out of valid range
    """
    if not 0 <= h3_resolution <= 15:
        raise ValueError(f"Invalid H3 resolution: {h3_resolution}")
    
    # Processing implementation
    return processed_data
```

## 🚀 Deployment

### Docker Support
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "-m", "urban_embedding"]
```

### Environment Variables
```bash
# .env.example
URBANREPML_DATA_DIR=/data
URBANREPML_CACHE_DIR=/tmp/urbanrepml
URBANREPML_LOG_LEVEL=INFO
WANDB_API_KEY=your_key_here
H3_RESOLUTION_DEFAULT=9
MAX_WORKERS=10
```

## 📊 Monitoring & Logging

```python
import logging
from urban_embedding.utils import setup_logging

# Setup project-wide logging
logger = setup_logging(
    level='INFO',
    log_file='urban_embedding.log'
)

# Use in modules
logger.info(f"Processing {len(data)} records")
logger.warning(f"Missing data for H3 cells: {missing_cells}")
logger.error(f"Failed to process: {error}")
```

## 🔗 Useful Resources

- **H3 Documentation**: https://h3geo.org/
- **SRAI Framework**: https://github.com/kraina-ai/srai
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **GeoPandas**: https://geopandas.org/
- **Rasterio**: https://rasterio.readthedocs.io/

## 📌 Important Notes

- **AlphaEarth TIFFs**: ~3GB each (968 files for Cascadia study area)
- **Processing**: Uses multiprocessing (adjust `max_workers` in configs)
- **H3 Operations**: Use SRAI library (not raw h3-py) for advanced operations
- **Coordinates**: Always use WGS84 (EPSG:4326) for H3 operations
- **Memory**: Critical for large study areas - use chunking and windowing
- **Testing**: Always test with small subset before full processing

---

*Last updated: January 2025*