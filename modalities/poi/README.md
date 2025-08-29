# POI Modality

**Points of Interest Embeddings using Hex2Vec**

## Status: Not Yet Implemented

This modality will process Points of Interest (POI) data into H3 hexagon-based embeddings.

## Planned Features
- OSM POI extraction and categorization
- Hex2Vec categorical embeddings
- Density and diversity metrics
- Commercial/residential/recreational classification

## Expected Interface
```python
from modalities import load_modality_processor

processor = load_modality_processor('poi', {
    'osm_cache_dir': 'data/cache/osm',
    'categories': ['commercial', 'amenity', 'leisure', 'shop']
})

embeddings = processor.run_pipeline(
    study_area='cascadia',
    h3_resolution=10,
    output_dir='data/processed/embeddings/poi'
)
```

## Data Sources
- OpenStreetMap POI data
- Local business directories
- Municipal datasets

## Implementation Status
- [ ] ModalityProcessor interface
- [ ] OSM data extraction
- [ ] POI categorization
- [ ] Hex2Vec embeddings
- [ ] H3 aggregation
- [ ] Testing and validation