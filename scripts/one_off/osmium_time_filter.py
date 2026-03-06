"""Extract a point-in-time OSM snapshot from a full-history .osh.pbf file.

Uses pyosmium SimpleHandler with apply() for maximum speed (C++ callback path).
Stores candidate objects as pure Python dicts to avoid C++ memory invalidation.

Algorithm:
    The .osh.pbf is sorted by (type, id, version). For each entity group,
    we track the latest version <= cutoff. When the entity changes, we write
    the buffered candidate (if visible) via SimpleWriter.

Lifetime: temporary
Stage: stage1
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone

import osmium
from tqdm import tqdm

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from utils.paths import StudyAreaPaths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

STUDY_AREA = "netherlands"
CUTOFF_DATE = "2022-01-01"
CUTOFF_TS = datetime(2022, 1, 1, tzinfo=timezone.utc)
PROGRESS_INTERVAL = 5_000_000


class TimeFilterHandler(osmium.SimpleHandler):
    """SimpleHandler that filters OSM history to a point-in-time snapshot.

    Uses pure Python dicts to buffer the best candidate for each entity,
    avoiding any reference to invalidated C++ osmium objects.
    """

    def __init__(self, writer: osmium.SimpleWriter, file_size_bytes: int):
        super().__init__()
        self.writer = writer
        self.cutoff = CUTOFF_TS

        # Current entity tracking
        self._cur_type = None  # 'n', 'w', 'r'
        self._cur_id = None
        self._best = None  # dict with entity data, or None
        self._best_visible = False
        self._past_cutoff = False

        # Stats
        self.total_read = 0
        self.total_written = 0
        self.total_deleted = 0
        self._t0 = time.monotonic()

        # tqdm progress bar (estimate ~100 bytes/object for .osh.pbf)
        est_total = file_size_bytes // 100
        self.pbar = tqdm(
            total=est_total, unit="obj", unit_scale=True,
            desc="Time-filtering", dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] written={postfix}",
        )

    def _flush(self):
        """Write the buffered best candidate (if visible) and reset."""
        if self._best is not None and self._best_visible:
            d = self._best
            if d["type"] == "n":
                self.writer.add_node(
                    osmium.osm.mutable.Node(
                        id=d["id"],
                        version=d["version"],
                        visible=True,
                        changeset=d["changeset"],
                        timestamp=d["timestamp"],
                        uid=d["uid"],
                        tags=d["tags"],
                        location=d["location"],
                    )
                )
            elif d["type"] == "w":
                self.writer.add_way(
                    osmium.osm.mutable.Way(
                        id=d["id"],
                        version=d["version"],
                        visible=True,
                        changeset=d["changeset"],
                        timestamp=d["timestamp"],
                        uid=d["uid"],
                        tags=d["tags"],
                        nodes=d["nodes"],
                    )
                )
            elif d["type"] == "r":
                self.writer.add_relation(
                    osmium.osm.mutable.Relation(
                        id=d["id"],
                        version=d["version"],
                        visible=True,
                        changeset=d["changeset"],
                        timestamp=d["timestamp"],
                        uid=d["uid"],
                        tags=d["tags"],
                        members=d["members"],
                    )
                )
            self.total_written += 1
        elif self._best is not None:
            self.total_deleted += 1

        self._best = None
        self._best_visible = False
        self._past_cutoff = False

    def _progress(self):
        self.pbar.update(1)
        if self.total_read % PROGRESS_INTERVAL == 0:
            self.pbar.set_postfix_str(f"{self.total_written:,}")

    def _process(self, obj, obj_type_char: str):
        """Process one object version (node, way, or relation)."""
        self.total_read += 1

        obj_id = obj.id

        # Entity boundary check
        if obj_type_char != self._cur_type or obj_id != self._cur_id:
            self._flush()
            self._cur_type = obj_type_char
            self._cur_id = obj_id

        if self._past_cutoff:
            self._progress()
            return

        if obj.timestamp > self.cutoff:
            self._past_cutoff = True
            self._progress()
            return

        # This version is <= cutoff. Extract to pure Python dict.
        if obj.visible:
            tags = {t.k: t.v for t in obj.tags}
            d = {
                "type": obj_type_char,
                "id": obj_id,
                "version": obj.version,
                "changeset": obj.changeset,
                "timestamp": obj.timestamp,
                "uid": obj.uid,
                "tags": tags,
            }
            if obj_type_char == "n":
                loc = obj.location
                d["location"] = (loc.lon, loc.lat) if loc.valid() else (0.0, 0.0)
            elif obj_type_char == "w":
                d["nodes"] = [n.ref for n in obj.nodes]
            elif obj_type_char == "r":
                d["members"] = [(m.type, m.ref, m.role) for m in obj.members]
            self._best = d
            self._best_visible = True
        else:
            # Deleted version — just mark as not visible, minimal data
            self._best = {"type": obj_type_char, "id": obj_id}
            self._best_visible = False

        self._progress()

    def node(self, n):
        self._process(n, "n")

    def way(self, w):
        self._process(w, "w")

    def relation(self, r):
        self._process(r, "r")

    def finalize(self):
        """Flush the last buffered entity."""
        self._flush()
        self.pbar.close()
        elapsed = time.monotonic() - self._t0
        log.info("--- Done ---")
        log.info("Total objects read:     %d", self.total_read)
        log.info("Total entities written: %d", self.total_written)
        log.info("Skipped (deleted):      %d", self.total_deleted)
        log.info("Elapsed: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


def run_time_filter() -> None:
    paths = StudyAreaPaths(STUDY_AREA)
    input_path = paths.osm_history_pbf()
    output_path = paths.osm_snapshot_pbf(CUTOFF_DATE)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    file_size = input_path.stat().st_size
    log.info("Input:  %s (%.1f GB)", input_path, file_size / 1e9)
    log.info("Output: %s", output_path)
    log.info("Cutoff: %s", CUTOFF_TS.isoformat())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = osmium.SimpleWriter(str(output_path), overwrite=True)
    handler = TimeFilterHandler(writer, file_size)

    handler.apply_file(str(input_path))
    handler.finalize()

    writer.close()

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        log.info("Output file: %s (%.1f MB)", output_path, size_mb)


if __name__ == "__main__":
    run_time_filter()
