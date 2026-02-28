#!/usr/bin/env python3
"""
Run Roads processor on full Netherlands with Highway2Vec embeddings.
Windows-compatible version that disables QuackOSM's Unicode-breaking progress output.

The Netherlands PBF (1.3 GB) caused a QuackOSM OOM when using OSMPbfLoader because
it calls convert_pbf_to_geodataframe(), which loads all road geometries into RAM at
once.  This script bypasses that by using PbfFileReader directly:

  1. PbfFileReader.convert_pbf_to_parquet()  -- writes to disk, no RAM spike
  2. gpd.read_parquet()                       -- streaming, <2 GB RAM for NL roads
  3. H3Regionalizer (SRAI)                    -- tessellate study area at res 10
  4. IntersectionJoiner (SRAI)                -- spatially join roads to hexagons
  5. Highway2VecEmbedder.fit_transform()      -- train autoencoder on GPU, get embeddings
  6. Write to StudyAreaPaths.embedding_file() -- region_id index, R00-R63 columns

Output: data/study_areas/netherlands/stage1_unimodal/roads/netherlands_res10_2022.parquet
  ~5.1M rows x 64 columns (R00-R63), region_id index

Memory usage (estimates):
  - PBF parse: DuckDB with 8 GB limit + disk spilling for overflow
  - Roads GDF: ~1.5 GB for ~1.5M NL road segments
  - regions_gdf: ~1 GB for 5.1M hexagons
  - joint_gdf: ~2 GB multi-index
  - Highway2Vec training: GPU (RTX 3090), minuscule CPU RAM
"""

import logging
import sys
import warnings
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

# SRAI imports  -- SRAI-first per project rules
from srai.regionalizers import H3Regionalizer
from srai.joiners import IntersectionJoiner
from srai.embedders import Highway2VecEmbedder

# h3-py only for hierarchy (not used here; present for documentation clarity)

from utils import StudyAreaPaths
from stage1_modalities import MODALITY_PREFIXES

# Disable rich progress output (Windows Unicode issue)
os.environ["QUACKOSM_PROGRESS"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STUDY_AREA = "netherlands"
H3_RESOLUTION = 10
YEAR = 2022
PBF_PATH = Path("data/raw/osm/netherlands-latest.osm.pbf")

# Highway2Vec hyperparameters -- match the existing South Holland run (R00-R63 => 64d)
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 512

# Road types to include (same as DEFAULT_ROAD_TYPES in processor.py)
ROAD_TAGS: dict = {
    "highway": [
        "motorway", "motorway_link",
        "trunk", "trunk_link",
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link",
        "unclassified", "residential",
        "living_street", "service", "road",
    ]
}

# DuckDB memory config: allow 8 GB in-memory + disk spill for the rest
# Netherlands PBF extracts ~1.5 M road ways; peak DuckDB memory is ~6-10 GB
DUCKDB_MEMORY_LIMIT = "8GB"
DUCKDB_TEMP_DIR = str(Path("data/raw/osm/duckdb_tmp").absolute())


# ---------------------------------------------------------------------------
# Step 1: Load road segments from PBF via PbfFileReader
# ---------------------------------------------------------------------------

def load_roads_from_pbf(
    area_gdf: gpd.GeoDataFrame,
    pbf_path: Path,
    working_dir: str = "files",
) -> gpd.GeoDataFrame:
    """
    Parse road segments from PBF using QuackOSM's PbfFileReader directly.

    Unlike OSMPbfLoader (which calls convert_pbf_to_geodataframe and OOMs),
    this uses convert_pbf_to_parquet() to write to disk first, then reads
    back the parquet file.  QuackOSM caches the parquet result so re-runs
    are fast.

    Args:
        area_gdf: Study area boundary GeoDataFrame (EPSG:4326).
        pbf_path: Path to the netherlands-latest.osm.pbf file.
        working_dir: Directory for QuackOSM intermediate and cache files.

    Returns:
        GeoDataFrame of road segments with LineString/MultiLineString geometries.
    """
    from quackosm import PbfFileReader
    from rq_geo_toolkit.duckdb import DuckDBConnKwargs

    log.info("Step 1: Loading road segments from PBF via PbfFileReader (disk-first)...")
    log.info(f"  PBF: {pbf_path} ({pbf_path.stat().st_size / 1024**3:.2f} GB)")
    log.info(f"  DuckDB memory limit: {DUCKDB_MEMORY_LIMIT}")

    # Ensure temp dir exists
    Path(DUCKDB_TEMP_DIR).mkdir(parents=True, exist_ok=True)

    # Build DuckDB connection kwargs with memory limit for disk spilling
    conn_kwargs: DuckDBConnKwargs = {
        "config_kwargs": {
            "memory_limit": DUCKDB_MEMORY_LIMIT,
            "temp_directory": DUCKDB_TEMP_DIR,
            "threads": 8,
        }
    }

    # geometry_filter: use the union of the study area polygon
    geometry_filter = area_gdf.union_all()

    reader = PbfFileReader(
        tags_filter=ROAD_TAGS,
        geometry_filter=geometry_filter,
        working_directory=working_dir,
        verbosity_mode="silent",  # Disable verbose output to avoid rich Unicode issues
        duckdb_conn_kwargs=conn_kwargs,
    )

    # Write roads to geoparquet (cached by hash; fast on re-run)
    log.info("  Writing road features to geoparquet (QuackOSM caches by hash)...")
    roads_parquet_path = reader.convert_pbf_to_parquet(
        pbf_path=str(pbf_path),
        ignore_cache=False,   # use cache if it already exists
    )
    log.info(f"  Roads parquet: {roads_parquet_path} "
             f"({roads_parquet_path.stat().st_size / 1024**2:.0f} MB)")

    # Read back from parquet — streaming, ~1-2 GB RAM for NL roads
    log.info("  Reading roads from parquet into GeoDataFrame...")
    roads_gdf = gpd.read_parquet(roads_parquet_path)

    log.info(f"  Loaded {len(roads_gdf):,} road features")
    log.info(f"  CRS: {roads_gdf.crs}")
    log.info(f"  Geometry types: {roads_gdf.geometry.geom_type.value_counts().to_dict()}")

    # Filter to line geometries only (drop nodes/polygons that may slip through)
    initial = len(roads_gdf)
    roads_gdf = roads_gdf[roads_gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if len(roads_gdf) < initial:
        log.info(f"  Filtered to line geometries: {len(roads_gdf):,} "
                 f"(dropped {initial - len(roads_gdf):,})")

    # Ensure WGS84
    if roads_gdf.crs is None:
        roads_gdf = roads_gdf.set_crs("EPSG:4326")
    elif roads_gdf.crs.to_epsg() != 4326:
        roads_gdf = roads_gdf.to_crs("EPSG:4326")

    return roads_gdf


# ---------------------------------------------------------------------------
# Step 2-4: SRAI tessellation, spatial join, feature prep
# ---------------------------------------------------------------------------

def prepare_highway2vec_features(
    roads_gdf: gpd.GeoDataFrame,
    joint_gdf: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of numerical road features for Highway2Vec.

    Highway2VecEmbedder.fit() strips geometry and trains an autoencoder
    on the feature matrix.  The index must match joint_gdf.index.names[1]
    (the feature level of the MultiIndex produced by IntersectionJoiner).

    Args:
        roads_gdf: Road segments with highway tag columns.
        joint_gdf: IntersectionJoiner output with (region_id, feature_id) MultiIndex.

    Returns:
        GeoDataFrame with numerical columns + geometry, indexed by feature_id.
    """
    log.info("Step 4: Preparing numerical features for Highway2Vec...")

    features = roads_gdf.copy()

    # 1. One-hot encode top highway types
    if "highway" in features.columns:
        top_types = features["highway"].value_counts().head(12).index
        for ht in top_types:
            features[f"highway_{ht}"] = (features["highway"] == ht).astype("float32")
        log.info(f"  One-hot encoded {len(top_types)} highway types")

    # 2. Geometric features
    if features.crs is not None and features.crs.to_epsg() != 4326:
        features = features.to_crs("EPSG:4326")
    features["road_length"] = features.geometry.length.astype("float32")
    features["road_complexity"] = (
        features.geometry
        .apply(lambda g: len(g.coords) if hasattr(g, "coords") else 1)
        .astype("float32")
    )

    # 3. Select only numerical + geometry columns
    numerical_cols = [c for c in features.columns
                      if c.startswith("highway_") or c in ("road_length", "road_complexity")]
    features = features[numerical_cols + ["geometry"]].copy()

    # 4. Fill NaN
    for col in numerical_cols:
        features[col] = features[col].fillna(0.0)

    # 5. Align index name to joint_gdf second level (Highway2Vec validation requirement)
    if isinstance(joint_gdf.index, pd.MultiIndex):
        expected_name = joint_gdf.index.names[1]
        if features.index.name != expected_name:
            log.info(f"  Aligning features index name: {features.index.name!r} -> {expected_name!r}")
            features.index.name = expected_name

    log.info(f"  Feature matrix: {features.shape[0]:,} rows x {len(numerical_cols)} numerical cols")
    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> int:
    """Execute full roads embedding pipeline for the Netherlands."""
    paths = StudyAreaPaths(STUDY_AREA)

    log.info("=" * 72)
    log.info("Roads Pipeline: Full Netherlands -- Highway2Vec Embeddings")
    log.info("=" * 72)

    # ---- Load study area boundary ----------------------------------------
    area_gdf = gpd.read_parquet(paths.area_gdf_file(fmt="parquet"))
    log.info(f"Study area loaded: {area_gdf.shape}, CRS: {area_gdf.crs.to_epsg()}")

    # ---- Step 1: Load roads from PBF (disk-first, avoids RAM OOM) --------
    roads_gdf = load_roads_from_pbf(area_gdf, PBF_PATH)
    log.info(f"Roads loaded: {len(roads_gdf):,} segments")

    # ---- Step 2: H3 Tessellation (SRAI) ----------------------------------
    log.info("Step 2: H3 tessellation at resolution 10 (SRAI H3Regionalizer)...")
    regionalizer = H3Regionalizer(resolution=H3_RESOLUTION)
    regions_gdf = regionalizer.transform(area_gdf)
    log.info(f"  Regions: {len(regions_gdf):,} hexagons (index: {regions_gdf.index.name!r})")

    # ---- Step 3: Spatial join (SRAI IntersectionJoiner) ------------------
    log.info("Step 3: Spatial join (SRAI IntersectionJoiner)...")
    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(regions=regions_gdf, features=roads_gdf)
    log.info(f"  Joint pairs: {len(joint_gdf):,}")

    if len(joint_gdf) == 0:
        log.error("IntersectionJoiner returned an empty join — no roads intersect any hexagon.")
        return 1

    # ---- Step 4: Feature preparation ------------------------------------
    features_gdf = prepare_highway2vec_features(roads_gdf, joint_gdf)

    # ---- Step 5: Highway2Vec training on GPU ----------------------------
    log.info("Step 5: Training Highway2Vec autoencoder...")
    log.info(f"  Architecture: input -> {HIDDEN_SIZE} -> {EMBEDDING_SIZE} -> {HIDDEN_SIZE} -> output")
    log.info(f"  Epochs: {EPOCHS}, batch_size: {BATCH_SIZE}")

    embedder = Highway2VecEmbedder(
        hidden_size=HIDDEN_SIZE,
        embedding_size=EMBEDDING_SIZE,
    )

    embeddings_df = embedder.fit_transform(
        regions_gdf=regions_gdf,
        features_gdf=features_gdf,
        joint_gdf=joint_gdf,
        trainer_kwargs={
            "accelerator": "auto",   # RTX 3090 if available
            "devices": 1,
            "max_epochs": EPOCHS,
            "enable_progress_bar": True,
        },
        dataloader_kwargs={
            "batch_size": BATCH_SIZE,
        },
    )

    log.info(f"  Highway2Vec output: {embeddings_df.shape}")

    # Drop geometry column if Highway2Vec returned one
    if "geometry" in embeddings_df.columns:
        embeddings_df = embeddings_df.drop(columns=["geometry"])

    # ---- Step 6: Rename columns to R00-R63 convention ------------------
    prefix = MODALITY_PREFIXES["roads"]   # "R"
    rename_map = {}
    for col in embeddings_df.columns:
        if isinstance(col, int) or (isinstance(col, str) and str(col).lstrip("-").isdigit()):
            rename_map[col] = f"{prefix}{int(col):02d}"
    if rename_map:
        embeddings_df = embeddings_df.rename(columns=rename_map)
        log.info(f"  Renamed {len(rename_map)} columns with '{prefix}' prefix")

    # Keep only R-prefixed embedding columns
    embedding_cols = [c for c in embeddings_df.columns
                      if c.startswith(prefix) and c[len(prefix):].isdigit()]
    embeddings_df = embeddings_df[embedding_cols]
    log.info(f"  Embedding columns: {len(embedding_cols)} ({embedding_cols[0]} .. {embedding_cols[-1]})")

    # ---- Step 7: Enforce region_id index contract ----------------------
    embeddings_df.index = embeddings_df.index.astype(str)
    embeddings_df.index.name = "region_id"

    log.info(f"  Final embeddings: {embeddings_df.shape[0]:,} hexagons x {len(embedding_cols)} dims")

    # ---- Step 8: Save to canonical path --------------------------------
    out_path = paths.embedding_file("roads", H3_RESOLUTION, YEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_parquet(out_path)
    log.info(f"Saved: {out_path}")
    log.info(f"  Shape: {embeddings_df.shape}")
    log.info(f"  Index: {embeddings_df.index.name!r}")

    # ---- Verification --------------------------------------------------
    log.info("Verification...")
    check = pd.read_parquet(out_path)
    assert check.index.name == "region_id", f"Bad index name: {check.index.name}"
    assert all(c.startswith(prefix) for c in check.columns), "Not all columns start with 'R'"
    expected_rows = len(regions_gdf)
    if check.shape[0] < expected_rows * 0.8:
        log.warning(
            f"Output has {check.shape[0]:,} rows but expected ~{expected_rows:,}. "
            "Some hexagons may have no road intersections (OK for water/rural areas)."
        )
    else:
        log.info(f"  OK: {check.shape[0]:,} / {expected_rows:,} hexagons have embeddings")

    log.info("=" * 72)
    log.info("COMPLETE")
    log.info(f"  Output: {out_path}")
    log.info(f"  Shape:  {check.shape}")
    log.info("=" * 72)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing
    if sys.platform == "win32":
        # Required for PyTorch DataLoader on Windows
        multiprocessing.set_start_method("spawn", force=True)

    try:
        sys.exit(run_pipeline())
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(130)
    except Exception:
        log.exception("Pipeline failed with unhandled exception.")
        sys.exit(1)
