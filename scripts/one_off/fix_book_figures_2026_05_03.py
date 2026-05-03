"""Fix-wave driver: regenerate 3 QAQC-flagged figures in The Book of Netherlands.

Imports the figure-rendering helpers from ``build_the_book_2026_05_03.py`` and
runs only the three affected sub-blocks:

  * Ch1.1b  hex_grid_teaser_res9_amsterdam.png  (was uniform-grey rectangle)
  * Ch1.1c  tessellation_density_multires.png   (was 5 ghost-panel outlines)
  * Ch2.2c  roads_density_res9.png              (was washed-out cividis)

Then regenerates ``ch8_closing/book_provenance.yaml`` aggregator.

Lifetime: temporary (30-day shelf — paired with the W3.5 fix-wave).
Stage: stage3 visualization.
"""

from __future__ import annotations

import gc
import logging
import sys
import traceback
from datetime import date
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project root on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import the build script so we get _save_with_provenance, helpers, and
# critically the *updated* ch1 + ch2 sub-block code paths via direct copy
# below. We could call chapter_1_frontispiece() to get all three Ch1 figures,
# which is fine — re-rendering the AE RGB cover is acceptable (it's a
# replayable PCA RGB at the same source artefact + config hash).
from scripts.one_off.build_the_book_2026_05_03 import (  # noqa: E402
    BOOK_ROOT,
    DPI,
    RASTER_H,
    RASTER_W,
    STUDY_AREA,
    _ensure_dir,
    _join_to_regions,
    _emb_columns,
    _load_regions_metric,
    _pca_to,
    _regenerate_book_aggregate,
    _save_clean_figure,
    _save_with_provenance,
    chapter_1_frontispiece,
)
from utils.paths import StudyAreaPaths  # noqa: E402
from utils.visualization import (  # noqa: E402
    load_boundary,
    rasterize_continuous_voronoi,
    voronoi_params_for_resolution,
)

logging.basicConfig(
    level=logging.INFO,
    format="[fix-book] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fix-book")


def regen_ch2_roads_only(paths: StudyAreaPaths, boundary: gpd.GeoDataFrame) -> int:
    """Re-render only the roads_density_res9.png panel with percentile clipping."""
    log.info("Ch2.2c — roads_density_res9 (clipped)")
    out_dir = BOOK_ROOT / "ch2_modalities"
    _ensure_dir(out_dir)
    regions = _load_regions_metric(paths, 9)
    roads_path = paths.root / "stage1_unimodal" / "roads" / "netherlands_res9_latest.parquet"
    try:
        roads = pd.read_parquet(roads_path)
        cx, cy, extent_m, joined = _join_to_regions(roads, regions)
        emb_cols = _emb_columns(joined)
        pcs = _pca_to(joined, emb_cols, n_components=1)
        values = pcs[:, 0]
        finite = np.isfinite(values)
        vmin = float(np.nanpercentile(values[finite], 5))
        vmax = float(np.nanpercentile(values[finite], 90))
        log.info("  roads PC1: clip vmin=%.4f vmax=%.4f (p5/p90)", vmin, vmax)
        pixel_m, max_dist_m = voronoi_params_for_resolution(9)
        img, ext_xy = rasterize_continuous_voronoi(
            cx, cy, values, extent_m,
            cmap="inferno", vmin=vmin, vmax=vmax,
            pixel_m=pixel_m, max_dist_m=max_dist_m,
            bg_color=(0.10, 0.10, 0.10),
        )
        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        fp = out_dir / "roads_density_res9.png"
        _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fp)
        _save_with_provenance(
            fig, fp,
            plot_config={
                "modality": "roads", "year": "latest", "resolution": 9,
                "mode": "continuous_pc1_clipped",
                "cmap": "inferno", "vmin": vmin, "vmax": vmax,
                "clip_percentiles": [5, 90],
                "bg_color": [0.10, 0.10, 0.10],
                "pixel_m": pixel_m, "max_dist_m": max_dist_m,
            },
            source_artifacts=[roads_path],
            note=("Roads embeddings -> PC1, p5/p90 clipped on inferno over dark "
                  "background — reveals A-network corridors"),
        )
        del roads, joined, pcs, img
        gc.collect()
        return 1
    except Exception as e:
        log.error("  roads regen failed: %s", e)
        traceback.print_exc()
        return 0


def main() -> int:
    paths = StudyAreaPaths(STUDY_AREA)
    log.info("Loading boundary...")
    boundary = load_boundary(paths)
    if boundary is None:
        log.error("Boundary not found — aborting")
        return 1

    n_ch1 = chapter_1_frontispiece(paths, boundary)
    log.info("Ch1 produced %d figures", n_ch1)

    n_roads = regen_ch2_roads_only(paths, boundary)
    log.info("Ch2 roads produced %d figures", n_roads)

    n_agg = _regenerate_book_aggregate()
    log.info("Aggregate regenerated (%d)", n_agg)

    log.info("=" * 60)
    log.info("FIX WAVE COMPLETE: ch1=%d, ch2_roads=%d, agg=%d",
             n_ch1, n_roads, n_agg)
    log.info("Build date: %s", date.today())
    return 0


if __name__ == "__main__":
    sys.exit(main())
