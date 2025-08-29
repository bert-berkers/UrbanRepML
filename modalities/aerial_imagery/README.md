# Aerial Imagery Modality

High-resolution RGB aerial imagery processing with DINOv3 encoding and hierarchical H3 aggregation.

## Overview

This modality fetches free aerial RGB images from PDOK (Netherlands) and encodes them using Meta's DINOv3 vision transformer, including specialized remote sensing variants. The key innovation is **hierarchical aggregation** that implements nested spatial scales - from fine image patches to coarse H3 hexagons.

## Key Features

- 🇳🇱 **Free PDOK Data**: Access to high-resolution aerial imagery of Netherlands
- 🧠 **DINOv3 Encoding**: State-of-the-art self-supervised vision features
- 🛰️ **Remote Sensing Variant**: DINOv3 fine-tuned specifically for satellite/aerial imagery
- 📐 **Hierarchical Aggregation**: Multi-scale processing from H3 res 12→10
- 🎯 **Active Inference**: Fisher information for dynamic attention weighting
- ⚡ **Efficient Processing**: Batched fetching and encoding with caching

## Architecture

```
┌─ PDOK API ─┐    ┌─ DINOv3 Encoder ─┐    ┌─ Hierarchical Aggregation ─┐
│ RGB Images │ -> │ Vision Features  │ -> │ H3 res 12 -> H3 res 10     │
│ 512×512px  │    │ 768-dim vectors  │    │ Attention-weighted pooling  │
└────────────┘    └──────────────────┘    └─────────────────────────────┘
```

### Nested Structure (Thermodynamic Analogy)

The processing implements a nested hierarchical structure where:

1. **Micro Level**: Individual image patches at fine H3 resolution (res 12-13)
2. **Meso Level**: DINOv3 encoding captures spatial relationships
3. **Macro Level**: Hierarchical aggregation to target resolution (res 10)

This follows thermodynamic principles where we "coarse grain" fine-scale information while preserving essential patterns - taking the "macro of the micro."

## Usage

### Basic Processing

```python
from modalities.aerial_imagery import AerialImageryProcessor

# Configuration
config = {
    'study_area': 'rotterdam_aerial',
    'pdok_year': 'current',
    'model_name': 'dinov3_rs_base',
    'target_h3_resolution': 10,
    'fine_h3_resolution': 12,
    'hierarchical_levels': 2
}

# Initialize processor
processor = AerialImageryProcessor(config)

# Run pipeline
embeddings_path = processor.run_pipeline(
    study_area='rotterdam_aerial',
    h3_resolution=10,
    output_dir='data/processed/embeddings/aerial_imagery'
)
```

### Multi-Modal Integration

```python
# Combine with AlphaEarth satellite data
from urban_embedding import UrbanEmbeddingPipeline

config = {
    'study_area': 'rotterdam_aerial',
    'modalities': ['aerial_imagery', 'alphaearth', 'poi'],
    'h3_resolution': 10
}

pipeline = UrbanEmbeddingPipeline(config)
results = pipeline.run()
```

### Active Inference Example

```python
from modalities.aerial_imagery.dinov3_encoder import DINOv3Encoder
import numpy as np

# Initialize with hierarchical extraction
encoder = DINOv3Encoder(
    model_name='dinov3_rs_base',
    extract_hierarchical=True
)

# Process image with attention dynamics
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
result = encoder.encode_image(image, return_attention=True)

# Compute Fisher information for natural gradients
fisher = encoder.compute_fisher_information(result.patch_features)
print(f"Information geometry: {fisher.shape}")
```

## Configuration

### Model Variants

| Model | Size | Performance | Use Case |
|-------|------|-------------|----------|
| `dinov3_small` | 21M params | Fast | Development/testing |
| `dinov3_base` | 86M params | Balanced | General use |
| `dinov3_large` | 304M params | High quality | Research |
| `dinov3_rs_base` | 86M params | **Recommended** | Remote sensing optimized |
| `dinov3_rs_large` | 304M params | Best quality | High-end applications |

### PDOK Years Available

- `current`: Most recent imagery (updated regularly)
- `2023`: 2023 imagery
- `2022`: 2022 imagery  
- `2021`: 2021 imagery
- `2020`: 2020 imagery

### H3 Resolutions

| Resolution | Area | Use Case |
|------------|------|----------|
| 8 | ~0.7 km² | Regional overview |
| 9 | ~0.1 km² | Neighborhoods |
| 10 | ~0.01 km² | **Target resolution** |
| 11 | ~0.001 km² | Buildings |
| 12 | ~0.0001 km² | **Fine fetching** |

## API Reference

### PDOKClient

```python
class PDOKClient:
    def __init__(self, year='current', image_size=512)
    def fetch_image_for_h3(self, h3_cell: str) -> ImageTile
    def fetch_images_for_hexagons(self, h3_cells: List[str]) -> Dict[str, ImageTile]
```

### DINOv3Encoder

```python
class DINOv3Encoder:
    def __init__(self, model_name='dinov3_rs_base', extract_hierarchical=True)
    def encode_image(self, image: np.ndarray) -> EncodingResult
    def compute_fisher_information(self, features: torch.Tensor) -> torch.Tensor
```

### AerialImageryProcessor

```python
class AerialImageryProcessor(ModalityProcessor):
    def __init__(self, config: Dict[str, Any])
    def run_pipeline(self, study_area: str, h3_resolution: int, output_dir: str) -> str
    def hierarchical_aggregation(self, fine_embeddings: Dict, target_resolution: int) -> Dict
```

## Data Sources

### PDOK (Netherlands)
- **URL**: https://www.pdok.nl/introductie/-/article/pdok-luchtfoto-rgb-open
- **Coverage**: Complete Netherlands territory
- **Resolution**: Up to 10cm per pixel
- **Update**: Yearly
- **License**: Open data (CC0)

### DINOv3 Models
- **Source**: Meta AI Research
- **URL**: https://ai.meta.com/dinov3/
- **License**: Apache 2.0
- **Remote Sensing**: Fine-tuned on satellite imagery

## Technical Details

### Coordinate Systems
- **Input boundaries**: WGS84 (EPSG:4326)
- **PDOK processing**: Dutch RD New (EPSG:28992)
- **H3 hexagons**: WGS84 (EPSG:4326)

### Processing Pipeline
1. Load study area boundaries
2. Generate H3 cells at fine resolution (12-13)
3. Fetch RGB images from PDOK WMS
4. Encode images with DINOv3
5. Hierarchical aggregation with Fisher information weighting
6. Output embeddings at target resolution (10)

### Memory & Performance
- **Image encoding**: ~200ms per 512×512 image (GPU)
- **Memory usage**: ~2GB VRAM for base model
- **Batch processing**: Recommended batch size 6-8
- **Rate limiting**: 0.1-0.2s between PDOK requests

## Examples

See `examples/aerial_imagery_example.py` for complete usage examples including:
- Standalone processing
- Multi-modal integration with AlphaEarth
- Hierarchical active inference demonstration

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
requests>=2.28.0
pillow>=9.0.0
geopandas>=0.13.0
h3>=3.7.0
```

## Limitations

- **Geographic coverage**: Currently Netherlands only (PDOK)
- **Historical data**: Limited to years 2020-present
- **Processing time**: ~1-2 seconds per hexagon including network requests
- **Rate limits**: PDOK has fair-use limits (be respectful)

## Future Enhancements

- [ ] Support for other European aerial imagery services
- [ ] Multi-temporal change detection
- [ ] Attention visualization tools
- [ ] Integration with Sentinel-2 data
- [ ] Real-time processing capabilities