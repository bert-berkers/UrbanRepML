#!/usr/bin/env python3
"""
Roads pipeline using DuckDB directly (bypass QuackOSM PBF loader issues).

This approach:
1. Uses osm2py to extract roads from PBF to GeoDataFrame directly
2. Skips QuackOSM's problematic progress output on Windows
3. Falls back to QuackOSM only for caching if needed

If osm2py not available, falls back to reading pre-existing cached parquet.
"""

import logging
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

# SRAI imports
from srai.regionalizers import H3Regionalizer
from srai.joiners import IntersectionJoiner
from srai.embedders import Highway2VecEmbedder

from utils import StudyAreaPaths
from stage1_modalities import MODALITY_PREFIXES

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

EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 512

ROAD_TAGS = {
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


def load_roads_from_pbf_osmnx(
    area_gdf: gpd.GeoDataFrame,
    pbf_path: Path,
) -> gpd.GeoDataFrame:
    """Load roads using osmnx (simpler than QuackOSM, avoids progress issues)."""
    try:
        import osmnx as ox
        log.info("Using osmnx to extract roads...")

        # Create a bounding box from the study area
        bounds = area_gdf.total_bounds  # minx, miny, maxx, maxy

        # Download roads (this will use the local PBF if available)
        try:
            # Try from local PBF first via osm2py or direct path
            log.info(f"Attempting to load from PBF: {pbf_path}")
            # osmnx doesn't directly support PBF, but we can use osm2py
            from osm2py import OSM
            osm = OSM(pbf_path)
            roads_gdf = osm.to_geopandas(tags=ROAD_TAGS)
            log.info(f"Loaded {len(roads_gdf):,} roads from PBF via osm2py")
            return roads_gdf
        except Exception as e:
            log.warning(f"osm2py failed: {e}, falling back to osmnx online API...")
            raise
    except ImportError:
        log.info("osmnx/osm2py not available, will try QuackOSM cache...")
        return None


def load_roads_from_quackosm_cache() -> gpd.GeoDataFrame:
    """Load from QuackOSM cache if it exists."""
    log.info("Checking QuackOSM cache...")
    cache_dir = Path("files")

    # Look for any roads-like parquet in cache
    parquets = list(cache_dir.glob("*exploded_sorted.parquet"))
    for pq in parquets:
        try:
            df = pd.read_parquet(pq)
            # Check if it looks like roads (has geometry, highway tag)
            if "geometry" in df.columns and "highway" in df.columns:
                gdf = gpd.GeoDataFrame(df, geometry="geometry")
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                log.info(f"Found cached roads in {pq.name}: {len(gdf):,} features")
                return gdf
        except Exception as e:
            log.debug(f"Cache {pq.name} not usable: {e}")

    return None


def prepare_highway2vec_features(
    roads_gdf: gpd.GeoDataFrame,
    joint_gdf: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Build numerical features for Highway2Vec."""
    log.info("Step 4: Preparing numerical features for Highway2Vec...")

    features = roads_gdf.copy()

    if "highway" in features.columns:
        top_types = features["highway"].value_counts().head(12).index
        for ht in top_types:
            features[f"highway_{ht}"] = (features["highway"] == ht).astype("float32")
        log.info(f"  One-hot encoded {len(top_types)} highway types")

    if features.crs is not None and features.crs.to_epsg() != 4326:
        features = features.to_crs("EPSG:4326")
    features["road_length"] = features.geometry.length.astype("float32")
    features["road_complexity"] = (
        features.geometry
        .apply(lambda g: len(g.coords) if hasattr(g, "coords") else 1)
        .astype("float32")
    )

    numerical_cols = [c for c in features.columns
                      if c.startswith("highway_") or c in ("road_length", "road_complexity")]
    features = features[numerical_cols + ["geometry"]].copy()

    for col in numerical_cols:
        features[col] = features[col].fillna(0.0)

    if isinstance(joint_gdf.index, pd.MultiIndex):
        expected_name = joint_gdf.index.names[1]
        if features.index.name != expected_name:
            log.info(f"  Aligning features index name: {features.index.name!r} -> {expected_name!r}")
            features.index.name = expected_name

    log.info(f"  Feature matrix: {features.shape[0]:,} rows x {len(numerical_cols)} numerical cols")
    return features


def run_pipeline() -> int:
    """Execute full roads embedding pipeline."""
    paths = StudyAreaPaths(STUDY_AREA)

    log.info("=" * 72)
    log.info("Roads Pipeline: Full Netherlands -- DuckDB Direct Approach")
    log.info("=" * 72)

    # Load study area
    area_gdf = gpd.read_parquet(paths.area_gdf_file(fmt="parquet"))
    log.info(f"Study area loaded: {area_gdf.shape}, CRS: {area_gdf.crs.to_epsg()}")

    # Step 1: Try to load roads
    roads_gdf = load_roads_from_pbf_osmnx(area_gdf, PBF_PATH)

    if roads_gdf is None:
        log.info("osmnx/osm2py failed, checking QuackOSM cache...")
        roads_gdf = load_roads_from_quackosm_cache()

    if roads_gdf is None:
        log.error("Could not load roads from any source")
        return 1

    log.info(f"Roads loaded: {len(roads_gdf):,} segments")

    # Step 2: H3 Tessellation
    log.info("Step 2: H3 tessellation at resolution 10 (SRAI H3Regionalizer)...")
    regionalizer = H3Regionalizer(resolution=H3_RESOLUTION)
    regions_gdf = regionalizer.transform(area_gdf)
    log.info(f"  Regions: {len(regions_gdf):,} hexagons (index: {regions_gdf.index.name!r})")

    # Step 3: Spatial join
    log.info("Step 3: Spatial join (SRAI IntersectionJoiner)...")
    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(regions=regions_gdf, features=roads_gdf)
    log.info(f"  Joint pairs: {len(joint_gdf):,}")

    if joint_gdf.empty:
        log.error("IntersectionJoiner returned empty result")
        return 1

    # Step 4: Feature preparation
    features_gdf = prepare_highway2vec_features(roads_gdf, joint_gdf)

    # Step 5: Highway2Vec training
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
            "accelerator": "auto",
            "devices": 1,
            "max_epochs": EPOCHS,
            "enable_progress_bar": True,
        },
        dataloader_kwargs={
            "batch_size": BATCH_SIZE,
        },
    )

    log.info(f"  Highway2Vec output: {embeddings_df.shape}")

    if "geometry" in embeddings_df.columns:
        embeddings_df = embeddings_df.drop(columns=["geometry"])

    # Step 6: Rename columns
    prefix = MODALITY_PREFIXES["roads"]
    rename_map = {}
    for col in embeddings_df.columns:
        if isinstance(col, int) or (isinstance(col, str) and str(col).lstrip("-").isdigit()):
            rename_map[col] = f"{prefix}{int(col):02d}"
    if rename_map:
        embeddings_df = embeddings_df.rename(columns=rename_map)
        log.info(f"  Renamed {len(rename_map)} columns with '{prefix}' prefix")

    embedding_cols = [c for c in embeddings_df.columns
                      if c.startswith(prefix) and c[len(prefix):].isdigit()]
    embeddings_df = embeddings_df[embedding_cols]
    log.info(f"  Embedding columns: {len(embedding_cols)} ({embedding_cols[0]} .. {embedding_cols[-1]})")

    # Step 7: Enforce index contract
    embeddings_df.index = embeddings_df.index.astype(str)
    embeddings_df.index.name = "region_id"

    log.info(f"  Final embeddings: {embeddings_df.shape[0]:,} hexagons x {len(embedding_cols)} dims")

    # Step 8: Save
    out_path = paths.embedding_file("roads", H3_RESOLUTION, YEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_parquet(out_path)
    log.info(f"Saved: {out_path}")
    log.info(f"  Shape: {embeddings_df.shape}")
    log.info(f"  Index: {embeddings_df.index.name!r}")

    # Verification
    log.info("Verification...")
    check = pd.read_parquet(out_path)
    assert check.index.name == "region_id", f"Bad index name: {check.index.name}"
    assert all(c.startswith(prefix) for c in check.columns), "Not all columns start with 'R'"
    expected_rows = len(regions_gdf)
    if check.shape[0] < expected_rows * 0.8:
        log.warning(
            f"Output has {check.shape[0]:,} rows but expected ~{expected_rows:,}. "
            "Some hexagons may have no road intersections."
        )
    else:
        log.info(f"  OK: {check.shape[0]:,} / {expected_rows:,} hexagons have embeddings")

    log.info("=" * 72)
    log.info("COMPLETE")
    log.info(f"  Output: {out_path}")
    log.info(f"  Shape:  {check.shape}")
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    import multiprocessing
    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn", force=True)

    try:
        sys.exit(run_pipeline())
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(130)
    except Exception:
        log.exception("Pipeline failed with unhandled exception.")
        sys.exit(1)
