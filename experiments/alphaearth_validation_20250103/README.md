# AlphaEarth Validation Experiment

**Date:** January 3, 2025  
**Experiment ID:** alphaearth_validation_20250103

## Objective
Validate AlphaEarth 2023 satellite-derived embeddings against existing urban representation embeddings for the Netherlands.

## Data Sources
- **AlphaEarth 2023**: Satellite-derived embeddings at H3 resolution 10
- **Baseline Embeddings**: GTFS, aerial imagery (fine-tuned), POI, and road network embeddings

## Structure
```
├── config.yaml              # Experiment configuration
├── data/
│   ├── alphaearth_processed/ # AlphaEarth embeddings
│   └── baseline_embeddings/  # Existing embeddings for comparison
├── analysis/                 # Analysis results and metrics
├── plots/                    # All visualizations
└── logs/                     # Processing logs
```

## Key Questions
1. How well do AlphaEarth embeddings correlate with urban function embeddings?
2. Can AlphaEarth embeddings identify similar urban typologies?
3. What are the strengths/weaknesses compared to existing embeddings?

## Next Steps
- [ ] Load and align all embedding datasets
- [ ] Perform spatial correlation analysis
- [ ] Compare clustering results
- [ ] Generate validation report