"""Extract POI nodes from OSM history (.osh.pbf) with time-filter for a cutoff date.

Reads the full-history .osh.pbf using pyosmium's SimpleHandler.apply_file() (C++ fast
path). For each NODE with POI-relevant tags and timestamp <= cutoff, keeps the latest
visible version. Outputs a GeoDataFrame matching SRAI OSMPbfLoader/OSMOnlineLoader
format: feature_id index, geometry column, one column per HEX2VEC_FILTER key.

NOTE: Only processes NODES for now. Ways and relations (which account for ~95% of OSM
features like buildings and landuse polygons) are not included. To add way/relation
POIs, you would need to resolve way node references into geometries (LineString/Polygon)
and relation members into MultiPolygon, which requires a two-pass approach or an
external node location index. This could be added as a follow-up.

Lifetime: temporary
Stage: stage1
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import osmium
from tqdm import tqdm

# Ensure project root is on sys.path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.paths import StudyAreaPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

PROGRESS_INTERVAL = 5_000_000

# Lazy-load HEX2VEC_FILTER to avoid import overhead at module level.
# The filter is a dict mapping OSM keys -> list of valid tag values.
_hex2vec_filter = None


def _get_hex2vec_filter():
    """Load SRAI's HEX2VEC_FILTER (15 OSM keys, 725 sub-tags)."""
    global _hex2vec_filter
    if _hex2vec_filter is None:
        from srai.loaders.osm_loaders.filters import HEX2VEC_FILTER
        _hex2vec_filter = HEX2VEC_FILTER
    return _hex2vec_filter


def _build_tag_lookup(poi_filter: dict) -> dict:
    """Build a fast lookup: {osm_key: set(valid_values)} from HEX2VEC_FILTER.

    Each key maps to a set of allowed tag values. A node matches if it has
    at least one tag where key is in the lookup AND value is in the set.
    """
    lookup = {}
    for key, values in poi_filter.items():
        if isinstance(values, list):
            lookup[key] = set(values)
        elif isinstance(values, bool) and values:
            # True means accept any value for this key
            lookup[key] = None  # None means "accept all"
        elif isinstance(values, str):
            lookup[key] = {values}
        else:
            lookup[key] = set(values)
    return lookup


def _node_matches_filter(tags: dict, tag_lookup: dict) -> bool:
    """Check if a node's tags match any of the POI filter criteria."""
    for key, value in tags.items():
        if key in tag_lookup:
            allowed = tag_lookup[key]
            if allowed is None:
                # Accept any value
                return True
            if value in allowed:
                return True
    return False


class POIExtractHandler(osmium.SimpleHandler):
    """SimpleHandler that extracts POI nodes from .osh.pbf with time filtering.

    The .osh.pbf is sorted by (type, id, version). For each (node, id) group,
    we keep the latest version with timestamp <= cutoff. If that version has
    visible=False, we skip it.

    Stores matched nodes as dicts with {id, lon, lat, tags} for later
    conversion to GeoDataFrame.
    """

    def __init__(self, cutoff: datetime, tag_lookup: dict, poi_keys: list,
                 file_size_bytes: int = 0):
        super().__init__()
        self.cutoff = cutoff
        self.tag_lookup = tag_lookup
        self.poi_keys = poi_keys  # ordered list of HEX2VEC_FILTER keys

        # Current node tracking (history is sorted by type, id, version)
        self._cur_id = None
        self._best = None  # dict or None
        self._best_visible = False
        self._past_cutoff = False

        # Collected POI nodes
        self.pois = []  # list of dicts: {id, lon, lat, tags}

        # Stats
        self.total_nodes_read = 0
        self.total_non_nodes_skipped = 0
        self.total_deleted = 0
        self.total_no_match = 0
        self._t0 = time.monotonic()

        # tqdm progress bar (estimate ~100 bytes/object for .osh.pbf)
        est_total = file_size_bytes // 100 if file_size_bytes > 0 else None
        self.pbar = tqdm(
            total=est_total, unit="obj", unit_scale=True,
            desc="Extracting POIs", dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] POIs={postfix}",
        )

    def _flush(self):
        """Process the buffered best candidate for the current node group."""
        if self._best is not None and self._best_visible:
            tags = self._best["tags"]
            if _node_matches_filter(tags, self.tag_lookup):
                self.pois.append(self._best)
            else:
                self.total_no_match += 1
        elif self._best is not None:
            self.total_deleted += 1

        self._best = None
        self._best_visible = False
        self._past_cutoff = False

    def _progress(self):
        self.pbar.update(1)
        if self.total_nodes_read % PROGRESS_INTERVAL == 0 and self.total_nodes_read > 0:
            self.pbar.set_postfix_str(f"{len(self.pois):,}")

    def node(self, n):
        self.total_nodes_read += 1

        node_id = n.id

        # Entity boundary: new node id -> flush previous
        if node_id != self._cur_id:
            self._flush()
            self._cur_id = node_id

        # Skip if we already found a version past the cutoff for this node
        if self._past_cutoff:
            self._progress()
            return

        # Check timestamp
        if n.timestamp > self.cutoff:
            self._past_cutoff = True
            self._progress()
            return

        # This version is <= cutoff. Extract to Python dict.
        if n.visible:
            tags = {t.k: t.v for t in n.tags}
            loc = n.location
            if loc.valid():
                self._best = {
                    "id": node_id,
                    "lon": loc.lon,
                    "lat": loc.lat,
                    "tags": tags,
                }
                self._best_visible = True
            else:
                # Invalid location -- treat as not visible
                self._best = {"id": node_id}
                self._best_visible = False
        else:
            # Deleted version
            self._best = {"id": node_id}
            self._best_visible = False

        self._progress()

    def way(self, w):
        self.pbar.update(1)
        self.total_non_nodes_skipped += 1
        # Optimization: ways come after nodes in PBF. Once we see a way,
        # all nodes have been processed. We could abort here, but
        # SimpleHandler doesn't support early exit easily. The handler
        # will naturally skip ways/relations quickly since we only
        # implement node().
        # Actually, since we override way() and relation(), they get called.
        # But since we do nothing, the C++ layer is still fast.

    def relation(self, r):
        self.pbar.update(1)
        self.total_non_nodes_skipped += 1

    def finalize(self):
        """Flush the last buffered node and print summary stats."""
        self._flush()
        self.pbar.close()
        elapsed = time.monotonic() - self._t0
        log.info("--- Extraction complete ---")
        log.info("Total nodes read:         %d", self.total_nodes_read)
        log.info("POI nodes matched:        %d", len(self.pois))
        log.info("Deleted at cutoff:        %d", self.total_deleted)
        log.info("No filter match:          %d", self.total_no_match)
        log.info("Non-nodes skipped:        %d", self.total_non_nodes_skipped)
        log.info("Elapsed: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


def build_geodataframe(pois: list, poi_keys: list):
    """Convert list of POI dicts to a GeoDataFrame matching SRAI loader format.

    Output format:
        Index: feature_id (e.g. "node/12345")
        Columns: geometry (Point), aeroway, amenity, building, ..., waterway
        Values: tag value string or NaN

    This matches the output of SRAI's OSMPbfLoader and OSMOnlineLoader, which
    the POI processor's load_data() method returns.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point

    log.info("Building GeoDataFrame from %d POI nodes...", len(pois))

    if not pois:
        # Return empty GeoDataFrame with correct schema
        gdf = gpd.GeoDataFrame(
            {key: pd.Series(dtype="object") for key in poi_keys},
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        )
        gdf.index.name = "feature_id"
        return gdf

    # Build arrays for vectorized construction
    feature_ids = []
    geometries = []
    tag_columns = {key: [] for key in poi_keys}

    for poi in pois:
        feature_ids.append(f"node/{poi['id']}")
        geometries.append(Point(poi["lon"], poi["lat"]))

        tags = poi["tags"]
        for key in poi_keys:
            tag_columns[key].append(tags.get(key, np.nan))

    # Build DataFrame
    df = pd.DataFrame(tag_columns, index=pd.Index(feature_ids, name="feature_id"))

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

    log.info("GeoDataFrame shape: %s", gdf.shape)
    log.info("Columns: %s", gdf.columns.tolist())

    # Report tag coverage
    for key in poi_keys:
        non_null = gdf[key].notna().sum()
        if non_null > 0:
            log.info("  %s: %d nodes (%.1f%%)", key, non_null, 100 * non_null / len(gdf))

    return gdf


def main():
    parser = argparse.ArgumentParser(
        description="Extract POI nodes from OSM history file with time filtering",
    )
    parser.add_argument(
        "--study-area", default="netherlands",
        help="Study area name (default: netherlands)",
    )
    parser.add_argument(
        "--cutoff-date", default="2022-01-01",
        help="Cutoff date in YYYY-MM-DD format (default: 2022-01-01). "
             "Keeps latest node version with timestamp <= this date.",
    )
    parser.add_argument(
        "--output-path", default=None,
        help="Override output parquet path. If omitted, saves to "
             "intermediate/pois_gdf/{area}_res{res}_{year}_pois.parquet",
    )
    parser.add_argument(
        "--resolution", type=int, default=10,
        help="H3 resolution for filename convention (default: 10). "
             "Does NOT affect extraction (no spatial filtering).",
    )
    args = parser.parse_args()

    # Parse cutoff date
    cutoff_parts = args.cutoff_date.split("-")
    cutoff_year = int(cutoff_parts[0])
    cutoff_month = int(cutoff_parts[1]) if len(cutoff_parts) > 1 else 1
    cutoff_day = int(cutoff_parts[2]) if len(cutoff_parts) > 2 else 1
    cutoff_ts = datetime(cutoff_year, cutoff_month, cutoff_day, tzinfo=timezone.utc)

    # Paths
    paths = StudyAreaPaths(args.study_area)
    input_path = paths.osm_history_pbf()

    if not input_path.exists():
        log.error("Input .osh.pbf not found: %s", input_path)
        log.error(
            "Expected: data/study_areas/%s/osm/%s-internal.osh.pbf",
            args.study_area, args.study_area,
        )
        sys.exit(1)

    # Output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = (
            paths.intermediate("poi")
            / "pois_gdf"
            / f"{args.study_area}_res{args.resolution}_{cutoff_year}_pois.parquet"
        )

    log.info("Input:  %s (%.1f GB)", input_path, input_path.stat().st_size / 1e9)
    log.info("Output: %s", output_path)
    log.info("Cutoff: %s", cutoff_ts.isoformat())

    # Load POI filter
    poi_filter = _get_hex2vec_filter()
    poi_keys = sorted(poi_filter.keys())
    tag_lookup = _build_tag_lookup(poi_filter)

    log.info("POI filter: %d OSM keys (%s)", len(poi_keys), ", ".join(poi_keys))
    total_values = sum(len(v) if v else 0 for v in tag_lookup.values())
    log.info("Total tag values tracked: %d", total_values)

    # Run extraction
    file_size = input_path.stat().st_size
    handler = POIExtractHandler(cutoff_ts, tag_lookup, poi_keys, file_size_bytes=file_size)
    log.info("Starting extraction (nodes only, ways/relations skipped)...")
    handler.apply_file(str(input_path))
    handler.finalize()

    if not handler.pois:
        log.warning("No POI nodes extracted! Check cutoff date and filter.")
        sys.exit(1)

    # Build GeoDataFrame
    gdf = build_geodataframe(handler.pois, poi_keys)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path)
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("Saved to %s (%.1f MB)", output_path, output_size_mb)
    log.info("Total POI nodes: %d", len(gdf))


if __name__ == "__main__":
    main()
