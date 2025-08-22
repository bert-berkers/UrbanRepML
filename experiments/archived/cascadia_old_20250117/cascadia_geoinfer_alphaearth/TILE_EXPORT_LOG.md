# Cascadia AlphaEarth Export Tile Tracking Log

**Export Started:** January 8, 2025  
**Total Expected:** 5,152 tiles (8 years × 644 tiles per year)  
**Export Folder:** AlphaEarth_Cascadia (Google Drive)

## Export Progress by Year

### 2017: Export Started (70/644 tiles queued)
- **Status:** EXPORTING
- **Expected Tiles:** 2017_0000 through 2017_0643
- **Completed:** 0
- **Failed:** 0
- **Running:** 0
- **Queued:** 70+ (READY state)

### 2018: Export Started (77/644 tiles queued) 
- **Status:** EXPORTING
- **Expected Tiles:** 2018_0000 through 2018_0643
- **Completed:** 0
- **Failed:** 0
- **Running:** 0
- **Queued:** 77+ (READY state)

### 2019: Export Started (80+/644 tiles submitting)
- **Status:** EXPORTING
- **Expected Tiles:** 2019_0000 through 2019_0643
- **Completed:** 0
- **Failed:** 0
- **Running:** 0
- **Queued:** 80+ tiles successfully submitted

### 2020: Not Started (0/644 tiles)
- **Status:** PENDING
- **Expected Tiles:** 2020_0000 through 2020_0643
- **Completed:** 0
- **Failed:** 0
- **Running:** 0

### 2021: Not Started (0/644 tiles)
- **Status:** PENDING
- **Expected Tiles:** 2021_0000 through 2021_0643
- **Completed:** 0
- **Failed:** 0
- **Running:** 0

### 2022: Not Started (0/644 tiles)
- **Status:** PENDING
- **Expected Tiles:** 2022_0000 through 2022_0643
- **Completed:** 0
- **Failed:** 0
- **Running:** 0

### 2023: Partial Export (12/644 tiles)
- **Status:** IN PROGRESS
- **Expected Tiles:** 2023_0000 through 2023_0643
- **Completed:** 12
- **Failed:** 4
- **Running:** 4
- **Pending:** 58
- **Notes:** Export started but incomplete

### 2024: Not Started (0/644 tiles)
- **Status:** PENDING
- **Expected Tiles:** 2024_0000 through 2024_0643
- **Completed:** 0
- **Failed:** 0
- **Running:** 0

## Export Commands Used

### Initial 2023 Export
```bash
cd experiments/cascadia_geoinfer_alphaearth
python scripts/gee/export_cascadia_alphaearth.py --year 2023
```

## Tile Naming Convention
- Format: `{YEAR}_{TILE_NUMBER:04d}`
- Examples: 2023_0000, 2023_0001, 2023_0643
- Each tile covers approximately 0.372° × 0.277° 
- Tile size: 3072×3072 pixels at 10m resolution

## Google Earth Engine Project
- Project ID: `boreal-union-296021`
- Collection: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- Export Format: GeoTIFF with 64 bands (A00-A63)
- Scale: 10 meters
- CRS: EPSG:4326 (WGS84)

## Region Coverage
- **Cascadia Bioregion:** Northern California (16 counties) + Oregon (36 counties)
- **Bounds:** [-124.70, 38.67] to [-116.46, 46.30]
- **Total Area:** ~8.2° × 7.6° = ~62.3 square degrees

## Export Status Summary (Updated: Jan 8, 2025 21:46 UTC)

**✅ MAJOR PROGRESS:** Export campaigns successfully launched for most years!

### Currently Running Exports:
1. ✅ **2017:** Export running (70+ tiles submitted)
2. ✅ **2018:** Export running (569+ tiles submitted)
3. ✅ **2019:** Export running (81+ tiles submitted) 
4. ✅ **2020:** Export running (294+ tiles submitted)
5. ✅ **2021:** Export running (75+ tiles submitting)
6. ⏳ **2022:** Ready to start (644 tiles)
7. ✅ **2023:** Partial export in progress (69 tiles total)
8. ⏳ **2024:** Ready to start (644 tiles)

### Statistics:
- **Tasks Queued:** 1,093+ tasks across multiple years
- **Status Distribution:**
  - READY (queued): 1,045 tiles (95.6%)
  - RUNNING: 5 tiles (0.5%)
  - COMPLETED: 16 tiles (1.5%)
  - FAILED: 27 tiles (2.5%)

**Total Progress:** 1,093 / 5,152 tiles (21.2%) queued or in progress

---

*This log will be updated automatically as exports progress. Check the Google Earth Engine Tasks page for real-time status: https://code.earthengine.google.com/tasks?project=boreal-union-296021*