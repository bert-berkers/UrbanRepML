---
paths:
  - "stage1_modalities/**"
  - "stage2_fusion/**"
  - "stage3_analysis/**"
  - "scripts/processing_modalities/**"
  - "scripts/accessibility/**"
  - "scripts/netherlands/**"
  - "utils/**"
---

# SRAI-First Spatial Convention

**This project uses SRAI (Spatial Representations for AI) as the primary spatial interface.**

## Allowed h3-py Usage (ONLY these)

- `h3.cell_to_parent()`, `h3.cell_to_children()`, `h3.cell_to_center_child()`
- `h3.get_resolution()`, `h3.cell_to_local_ij()` and similar introspection

## NEVER use h3 for

- Tessellation (`h3.polyfill`) -- use `srai.regionalizers.H3Regionalizer`
- Converting cells to geometry (`h3.cell_to_boundary`) -- use `SpatialDB` for bulk queries,
  `srai.h3.h3_to_geoseries` for ad-hoc fallback
- Neighborhood queries (`h3.grid_disk`, `h3.grid_ring`) -- use `srai.neighbourhoods.H3Neighbourhood`
- Spatial joins or regionalization -- use SRAI's built-in tools

## Correct Import Pattern

```python
# CORRECT:
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood

# WRONG -- never use h3 for these:
import h3
h3.polyfill(...)     # Use H3Regionalizer
h3.grid_disk(...)    # Use H3Neighbourhood
```

## Audit Checklist

When modifying stage code, check:
1. No new `import h3` for tessellation/neighborhood operations
2. No `h3.polyfill`, `h3.grid_disk`, `h3.grid_ring`, `h3.cell_to_boundary` calls
3. GeoDataFrames indexed by `region_id` (not `h3_index`) in Stage 2+
4. SRAI regionalizer used for any new tessellation needs
