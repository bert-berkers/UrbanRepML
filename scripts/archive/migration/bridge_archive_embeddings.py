"""
Bridge legacy archive embeddings to StudyAreaPaths-compatible locations.

Source archives:
  - data/archive/embeddings_legacy_20250914/poi/poi_embeddings_res10.parquet
    (6.46M rows, 30 cols: h3_index, 28 count features, h3_resolution)
  - data/archive/embeddings_legacy_20250914/roads/roads_embeddings_res10.parquet
    (1.82M rows, 66 cols: h3_index, 64 float features, h3_resolution)

Targets (StudyAreaPaths.embedding_file convention):
  - data/study_areas/netherlands/stage1_unimodal/poi/netherlands_res10_2022.parquet
  - data/study_areas/netherlands/stage1_unimodal/roads/netherlands_res10_2022.parquet

Transformations applied:
  - Set h3_index as index, renamed to region_id (matches StudyAreaPaths convention)
  - POI feature columns renamed to P00-P27 (MODALITY_PREFIXES["poi"] = "P")
  - Roads feature columns renamed to R00-R63 (MODALITY_PREFIXES["roads"] = "R")
  - h3_resolution column dropped (implicit from file name: res10)

Run with:
    uv run python scripts/one_off/bridge_archive_embeddings.py
"""

from pathlib import Path

import pandas as pd

from utils.paths import StudyAreaPaths

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE_BASE = PROJECT_ROOT / "data" / "archive" / "embeddings_legacy_20250914"
PATHS = StudyAreaPaths("netherlands", project_root=PROJECT_ROOT)

RESOLUTION = 10
YEAR = 2022


# ---------------------------------------------------------------------------
# POI
# ---------------------------------------------------------------------------

def bridge_poi() -> None:
    src = ARCHIVE_BASE / "poi" / "poi_embeddings_res10.parquet"
    dst = PATHS.embedding_file("poi", RESOLUTION, YEAR)

    print(f"[POI] Reading {src} ...")
    df = pd.read_parquet(src)
    print(f"[POI] Loaded: {df.shape}, columns: {list(df.columns)}")

    # h3_index holds the hex string cell IDs.
    # The remaining 28 columns are POI category counts.
    # h3_resolution is metadata — dropped (encoded in filename).
    feature_cols = [
        c for c in df.columns if c not in ("h3_index", "h3_resolution")
    ]
    n_features = len(feature_cols)
    print(f"[POI] Feature columns ({n_features}): {feature_cols}")

    # Rename features to P00, P01, ..., Pxx
    rename_map = {
        old: f"P{i:02d}" for i, old in enumerate(feature_cols)
    }
    print(f"[POI] Renaming: {rename_map}")

    out = df[["h3_index"] + feature_cols].copy()
    out = out.rename(columns=rename_map)
    out = out.set_index("h3_index")
    out.index.name = "region_id"

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst)
    print(f"[POI] Written: {dst}")
    print(f"[POI] Final shape: {out.shape}, index={out.index.name}, cols[:5]={list(out.columns[:5])}")


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------

def bridge_roads() -> None:
    src = ARCHIVE_BASE / "roads" / "roads_embeddings_res10.parquet"
    dst = PATHS.embedding_file("roads", RESOLUTION, YEAR)

    print(f"\n[Roads] Reading {src} ...")
    df = pd.read_parquet(src)
    print(f"[Roads] Loaded: {df.shape}, columns: {list(df.columns[:6])} ...")

    # Feature columns are named '0', '1', ..., '63' (stringified integers)
    feature_cols = [
        c for c in df.columns if c not in ("h3_index", "h3_resolution")
    ]
    n_features = len(feature_cols)
    print(f"[Roads] Feature columns ({n_features}): {feature_cols[:6]} ...")

    # Rename to R00, R01, ..., R63
    rename_map = {
        old: f"R{int(old):02d}" for old in feature_cols
    }
    print(f"[Roads] Renaming first 5: { {k: rename_map[k] for k in feature_cols[:5]} } ...")

    out = df[["h3_index"] + feature_cols].copy()
    out = out.rename(columns=rename_map)
    out = out.set_index("h3_index")
    out.index.name = "region_id"

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst)
    print(f"[Roads] Written: {dst}")
    print(f"[Roads] Final shape: {out.shape}, index={out.index.name}, cols[:5]={list(out.columns[:5])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bridge_poi()
    bridge_roads()
    print("\nDone. Verify with:")
    print("  uv run python -c \"import pandas as pd; df = pd.read_parquet('data/study_areas/netherlands/stage1_unimodal/poi/netherlands_res10_2022.parquet'); print(f'POI: {df.shape}, index={df.index.name}, cols={list(df.columns[:5])}')\"")
    print("  uv run python -c \"import pandas as pd; df = pd.read_parquet('data/study_areas/netherlands/stage1_unimodal/roads/netherlands_res10_2022.parquet'); print(f'Roads: {df.shape}, index={df.index.name}, cols={list(df.columns[:5])}')\"")
