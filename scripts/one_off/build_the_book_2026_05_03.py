"""Build "The Book of Netherlands" — coffee-table report of Voronoi-rasterized plots.

A fuckton of plots collated into a cool book to read on the sofa.
Renders ~25 PNGs across 7 chapters using the Voronoi rasterizer toolkit
(W-shipped 2026-05-02). Every figure gets a sibling ``*.provenance.yaml``
sidecar via :class:`utils.provenance.SidecarWriter`.

Lifetime: temporary (30-day shelf).
Stage: stage3 visualization.
Plan: ``.claude/plans/2026-05-03-make-nice-plots-using-yesterday-s-voronoi-rasterizer.md``.

Output layout:
    reports/2026-05-03-book/
        ch1_frontispiece/
        ch2_modalities/
        ch3_voronoi_showcase/
        ch4_hierarchy/
        ch6_clusters/
        ch7_liveability/
        ch8_closing/

Chapter 5 ("The Three Embeddings") is handled by a sibling agent (Wave 1
dispatch 2). This script intentionally skips it.
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
import yaml
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# Ensure project root is on sys.path so utils.* imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.paths import StudyAreaPaths  # noqa: E402
from utils.provenance import SidecarWriter, compute_config_hash  # noqa: E402
from utils.visualization import (  # noqa: E402
    load_boundary,
    rasterize_categorical_voronoi,
    rasterize_continuous_voronoi,
    rasterize_labels,
    rasterize_rgb_voronoi,
    voronoi_params_for_resolution,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUDY_AREA = "netherlands"
BOOK_ROOT = _PROJECT_ROOT / "reports" / "2026-05-03-book"
PROVENANCE_INDEX: dict[str, dict] = {}  # collected at __exit__ for ch8 colophon

# Memory feedback: full raster per panel, never shrink
RASTER_W = 2000
RASTER_H = 2400
DPI = 150

# Coffee-table aesthetic
BG_COLOR = "white"
FALLBACK_GREY = (0.94, 0.94, 0.94)  # "#f0f0f0" — outside-cutoff fill
BOUNDARY_EDGE = "#404040"
BOUNDARY_LW = 0.6

logging.basicConfig(
    level=logging.INFO,
    format="[book] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("book")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_clean_figure(fig, ax, image, extent_xy, boundary_gdf, *, fig_path: Path,
                       title: str = "", aspect_w: float = RASTER_W / DPI,
                       aspect_h: float = RASTER_H / DPI) -> None:
    """Render a single rasterized image with minimal chrome (no titles, no ticks)."""
    if boundary_gdf is not None:
        boundary_gdf.plot(
            ax=ax, facecolor="none", edgecolor=BOUNDARY_EDGE,
            linewidth=BOUNDARY_LW, zorder=3,
        )
    minx, maxx, miny, maxy = extent_xy
    ax.imshow(
        image, extent=[minx, maxx, miny, maxy], origin="lower",
        aspect="equal", interpolation="nearest", zorder=2,
    )
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=14, pad=8)


def _save_with_provenance(
    fig, fig_path: Path, *, plot_config: dict, source_artifacts: list[Path],
    note: str = "",
) -> None:
    """Save figure + write {fig_path}.provenance.yaml sidecar."""
    _ensure_dir(fig_path.parent)
    # SidecarWriter writes a *.run.yaml alongside; we ALSO want the
    # *.provenance.yaml sibling for the book's colophon aggregator.
    cfg_hash = compute_config_hash(plot_config)
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

    # Write a lightweight sidecar matching the figure-provenance schema.
    sidecar_path = fig_path.with_suffix(fig_path.suffix + ".provenance.yaml")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc = {
        "figure_path": fig_path.name,
        "created_at": now,
        "plot_config": plot_config,
        "plot_config_hash": cfg_hash,
        "source_artifacts": [str(p.relative_to(_PROJECT_ROOT)) if _PROJECT_ROOT in p.parents else str(p) for p in source_artifacts],
        "producer_script": "scripts/one_off/build_the_book_2026_05_03.py",
        "note": note,
        "schema_version": "1.0",
    }
    with sidecar_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
    PROVENANCE_INDEX[str(fig_path.relative_to(_PROJECT_ROOT))] = {
        "plot_config_hash": cfg_hash,
        "created_at": now,
        "note": note,
    }
    log.info("    wrote %s", fig_path.name)


def _load_regions_metric(paths: StudyAreaPaths, resolution: int) -> gpd.GeoDataFrame:
    """Load regions GeoDataFrame at resolution, reprojected to RD New (28992)."""
    rgdf_path = paths.root / "regions_gdf" / f"{STUDY_AREA}_res{resolution}.parquet"
    gdf = gpd.read_parquet(rgdf_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    if gdf.crs.to_epsg() != 28992:
        gdf = gdf.to_crs(28992)
    return gdf


def _join_to_regions(emb_df: pd.DataFrame, regions_gdf: gpd.GeoDataFrame
                     ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], pd.DataFrame]:
    """Inner-join embeddings to regions; return centroids + extent + joined df."""
    # Normalize index — some Stage 1 files have h3_index column instead of region_id index
    if emb_df.index.name != "region_id":
        if "region_id" in emb_df.columns:
            emb_df = emb_df.set_index("region_id")
        elif "h3_index" in emb_df.columns:
            emb_df = emb_df.rename(columns={"h3_index": "region_id"}).set_index("region_id")
    # Inner join on region_id
    joined = regions_gdf.join(emb_df, how="inner")
    centroids = joined.geometry.centroid
    cx = centroids.x.to_numpy()
    cy = centroids.y.to_numpy()
    minx, miny, maxx, maxy = joined.total_bounds
    return cx, cy, (float(minx), float(miny), float(maxx), float(maxy)), joined


def _emb_columns(df: pd.DataFrame) -> list[str]:
    """Detect numeric embedding columns (exclude weight/count/geometry/etc)."""
    exclude = {"geometry", "resolution", "weight_sum", "n_grid_cells",
               "h3_index", "region_id", "cluster_id", "h3_resolution"}
    cols = [c for c in df.columns if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def _pca_to(df: pd.DataFrame, cols: list[str], n_components: int = 1,
            random_state: int = 42) -> np.ndarray:
    """PCA of df[cols] to n_components; returns (N, n_components)."""
    X = df[cols].to_numpy(dtype=np.float32)
    # Standardize to avoid scale dominance
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(Xs)


# ---------------------------------------------------------------------------
# Chapter 1 — Frontispiece
# ---------------------------------------------------------------------------


def chapter_1_frontispiece(paths: StudyAreaPaths, boundary: gpd.GeoDataFrame) -> int:
    log.info("Chapter 1 — Frontispiece")
    out_dir = BOOK_ROOT / "ch1_frontispiece"
    _ensure_dir(out_dir)
    n_done = 0

    # --- 1a. Cover: AlphaEarth RGB-stitched at res9 ---
    try:
        regions = _load_regions_metric(paths, 9)
        ae_path = paths.embedding_file("alphaearth", 9, 2022)
        ae = pd.read_parquet(ae_path)
        cx, cy, extent_m, joined = _join_to_regions(ae, regions)
        emb_cols = [c for c in joined.columns if c.startswith("A") and c[1:].isdigit()]
        log.info("  AE shape after join: %d hexes, %d cols", len(joined), len(emb_cols))
        pcs = _pca_to(joined, emb_cols, n_components=3)
        # Normalize each PC independently to [0,1] using 2-98 percentiles
        rgb = np.zeros_like(pcs, dtype=np.float32)
        for i in range(3):
            v = pcs[:, i]
            lo, hi = np.percentile(v, [2, 98])
            rgb[:, i] = np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        pixel_m, max_dist_m = voronoi_params_for_resolution(9)
        img, ext_xy = rasterize_rgb_voronoi(cx, cy, rgb, extent_m,
                                            pixel_m=pixel_m, max_dist_m=max_dist_m)
        # Apply white background (where alpha=0 -> white)
        out = img.copy()
        outside = out[..., 3] <= 0.0
        out[outside, :3] = 1.0
        out[outside, 3] = 1.0

        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        _save_clean_figure(fig, ax, out, ext_xy, boundary,
                           fig_path=out_dir / "cover_alphaearth_rgb_res9.png")
        _save_with_provenance(
            fig, out_dir / "cover_alphaearth_rgb_res9.png",
            plot_config={
                "modality": "alphaearth", "resolution": 9, "year": 2022,
                "mode": "rgb_pca3", "pixel_m": pixel_m, "max_dist_m": max_dist_m,
                "rgb_norm": "p2_p98",
            },
            source_artifacts=[ae_path],
            note="THE COVER — AlphaEarth PC1/PC2/PC3 -> RGB",
        )
        n_done += 1
        del joined, ae, pcs, rgb, img, out
        gc.collect()
    except Exception as e:
        log.error("  ch1.1a cover failed: %s", e)
        traceback.print_exc()

    # --- 1b. Hex grid teaser: Amsterdam metro ---
    # Draw actual hex polygon boundaries (faint grey edges on white) so the
    # tessellation mesh is visible. Earlier attempt used voronoi rasterization
    # with a single colour, which produced a uniform grey rectangle (no edges).
    try:
        regions = _load_regions_metric(paths, 9)
        # Amsterdam center: lon 4.9041, lat 52.3676 -> RD ~120800, 487300
        # Crop a ~30km box
        ax_x = 120800
        ax_y = 487300
        box = 15000
        crop = regions.cx[ax_x - box: ax_x + box, ax_y - box: ax_y + box]
        log.info("  Amsterdam crop: %d hexes", len(crop))
        minx, miny, maxx, maxy = crop.total_bounds

        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        # Faint grey hex polygon edges, no fill
        crop.boundary.plot(
            ax=ax, color="#888888", linewidth=0.35, zorder=2,
        )
        # Optional Amsterdam outline if boundary covers the area; subset to
        # crop bbox so we don't draw a tiny full-Netherlands silhouette inside
        # this close-up.
        if boundary is not None:
            boundary.plot(
                ax=ax, facecolor="none", edgecolor=BOUNDARY_EDGE,
                linewidth=BOUNDARY_LW, zorder=3,
            )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fp = out_dir / "hex_grid_teaser_res9_amsterdam.png"
        _save_with_provenance(
            fig, fp,
            plot_config={
                "subregion": "amsterdam_metro_30km", "resolution": 9,
                "mode": "hex_polygon_boundaries",
                "edge_color": "#888888", "edge_linewidth": 0.35,
                "n_hexes": int(len(crop)),
            },
            source_artifacts=[paths.root / "regions_gdf" / f"{STUDY_AREA}_res9.parquet"],
            note="H3 res9 hex tessellation around Amsterdam — faint grey hex boundaries on white",
        )
        n_done += 1
        del crop
        gc.collect()
    except Exception as e:
        log.error("  ch1.1b teaser failed: %s", e)
        traceback.print_exc()

    # --- 1c. Tessellation density across resolutions ---
    # Fill each hex with grey at increasing opacity by resolution so the
    # progression is visible as densifying texture (rather than near-empty
    # outline panels at coarse res). Each panel rendered via
    # rasterize_categorical_voronoi with a single grey colour map.
    try:
        resolutions = [5, 6, 7, 8, 9]
        # Grey RGB constant; opacity is encoded by darkness (lower value = darker).
        # res5 = lightest, res9 = darkest.
        grey_per_res = {
            5: (0.62, 0.62, 0.62),
            6: (0.52, 0.52, 0.52),
            7: (0.42, 0.42, 0.42),
            8: (0.32, 0.32, 0.32),
            9: (0.22, 0.22, 0.22),
        }
        # voronoi_params_for_resolution(<7) falls back to (250, 300) — way
        # too small for res5/res6 cells (~8 km / ~3 km edges). Scale
        # explicitly so the voronoi fill covers each hex fully.
        params_override = {
            5: (4000.0, 12000.0),
            6: (1500.0, 4500.0),
        }
        # 5 panels horizontal, full raster width per panel.
        fig, axes = plt.subplots(
            1, 5, figsize=(5 * RASTER_W / DPI, RASTER_H / DPI),
        )
        hex_counts = []
        for ax_, res in zip(axes, resolutions):
            rgdf = _load_regions_metric(paths, res)
            n = len(rgdf)
            hex_counts.append(n)
            log.info("  res%d: %d hexes", res, n)
            cx = rgdf.geometry.centroid.x.to_numpy()
            cy = rgdf.geometry.centroid.y.to_numpy()
            minx, miny, maxx, maxy = rgdf.total_bounds
            labels = np.ones(n, dtype=np.int64)
            pixel_m, max_dist_m = params_override.get(
                res, voronoi_params_for_resolution(res)
            )
            img, ext_xy = rasterize_categorical_voronoi(
                cx, cy, labels, (minx, miny, maxx, maxy),
                color_map={1: grey_per_res[res]},
                pixel_m=pixel_m, max_dist_m=max_dist_m,
                bg_color=(1.0, 1.0, 1.0),
            )
            if boundary is not None:
                boundary.plot(ax=ax_, facecolor="none", edgecolor=BOUNDARY_EDGE,
                              linewidth=BOUNDARY_LW, zorder=3)
            mnx, mxx, mny, mxy = ext_xy
            ax_.imshow(img, extent=[mnx, mxx, mny, mxy], origin="lower",
                       aspect="equal", interpolation="nearest", zorder=2)
            ax_.set_xlim(mnx, mxx)
            ax_.set_ylim(mny, mxy)
            ax_.set_xticks([])
            ax_.set_yticks([])
            for s in ax_.spines.values():
                s.set_visible(False)
            # Hex count annotation in lower-left
            ax_.text(
                0.04, 0.04,
                f"res {res}\n{n:,} hexes",
                transform=ax_.transAxes,
                fontsize=28,
                color="#111111",
                weight="bold",
                ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="#444444",
                          linewidth=0.6, alpha=0.92, pad=10),
                zorder=4,
            )
            del rgdf, img
            gc.collect()
        fig.tight_layout()
        fp = out_dir / "tessellation_density_multires.png"
        _save_with_provenance(
            fig, fp,
            plot_config={
                "resolutions": resolutions,
                "hex_counts": hex_counts,
                "mode": "categorical_grey_voronoi_5panel",
                "grey_levels": {str(k): list(v) for k, v in grey_per_res.items()},
            },
            source_artifacts=[paths.root / "regions_gdf" / f"{STUDY_AREA}_res{r}.parquet"
                              for r in resolutions],
            note="Resolution scaling — 5 panels (res5-9), grey-fill voronoi at increasing darkness",
        )
        n_done += 1
    except Exception as e:
        log.error("  ch1.1c multires failed: %s", e)
        traceback.print_exc()

    return n_done


# ---------------------------------------------------------------------------
# Chapter 2 — The Modalities
# ---------------------------------------------------------------------------


def _render_continuous_panel(paths: StudyAreaPaths, boundary, regions_metric,
                             emb_df, fig_path, *, cmap, plot_config_extra, source_arts,
                             note):
    cx, cy, extent_m, joined = _join_to_regions(emb_df, regions_metric)
    emb_cols = _emb_columns(joined)
    pcs = _pca_to(joined, emb_cols, n_components=1)
    values = pcs[:, 0]
    pixel_m, max_dist_m = voronoi_params_for_resolution(9)
    img, ext_xy = rasterize_continuous_voronoi(
        cx, cy, values, extent_m,
        cmap=cmap, pixel_m=pixel_m, max_dist_m=max_dist_m,
        bg_color=FALLBACK_GREY,
    )
    fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
    _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fig_path)
    pc = {"resolution": 9, "mode": "continuous_pc1", "cmap": cmap,
          "pixel_m": pixel_m, "max_dist_m": max_dist_m}
    pc.update(plot_config_extra)
    _save_with_provenance(fig, fig_path, plot_config=pc,
                          source_artifacts=source_arts, note=note)


def chapter_2_modalities(paths: StudyAreaPaths, boundary: gpd.GeoDataFrame) -> int:
    log.info("Chapter 2 — The Modalities")
    out_dir = BOOK_ROOT / "ch2_modalities"
    _ensure_dir(out_dir)
    n_done = 0
    regions = _load_regions_metric(paths, 9)

    # 2a. AlphaEarth PC1
    try:
        ae_path = paths.embedding_file("alphaearth", 9, 2022)
        ae = pd.read_parquet(ae_path)
        _render_continuous_panel(
            paths, boundary, regions, ae,
            out_dir / "alphaearth_pc1_res9.png",
            cmap="viridis",
            plot_config_extra={"modality": "alphaearth", "year": 2022},
            source_arts=[ae_path],
            note="AlphaEarth PC1 over Netherlands at res9",
        )
        n_done += 1
        del ae
        gc.collect()
    except Exception as e:
        log.error("  ch2.2a AE failed: %s", e); traceback.print_exc()

    # 2b. POI hex2vec -> kmeans k=10 categorical
    try:
        poi_path = paths.root / "stage1_unimodal" / "poi" / "hex2vec" / "netherlands_res9_latest.parquet"
        poi = pd.read_parquet(poi_path)
        cx, cy, extent_m, joined = _join_to_regions(poi, regions)
        emb_cols = _emb_columns(joined)
        X = joined[emb_cols].to_numpy(dtype=np.float32)
        log.info("  hex2vec -> kmeans k=10 on %d hexes", len(X))
        km = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=4096, n_init=3)
        labels = km.fit_predict(X).astype(np.int64)
        pixel_m, max_dist_m = voronoi_params_for_resolution(9)
        img, ext_xy = rasterize_categorical_voronoi(
            cx, cy, labels, extent_m,
            n_clusters=10, cmap="tab10",
            pixel_m=pixel_m, max_dist_m=max_dist_m,
            bg_color=FALLBACK_GREY,
        )
        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        fp = out_dir / "poi_kmeans_res9.png"
        _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fp)
        _save_with_provenance(
            fig, fp,
            plot_config={"modality": "poi_hex2vec", "resolution": 9, "year": "latest",
                         "mode": "categorical_kmeans", "k": 10, "random_state": 42,
                         "cmap": "tab10", "pixel_m": pixel_m, "max_dist_m": max_dist_m},
            source_artifacts=[poi_path],
            note="POI hex2vec embeddings -> kmeans k=10 categorical",
        )
        n_done += 1
        del poi, X, labels
        gc.collect()
    except Exception as e:
        log.error("  ch2.2b POI failed: %s", e); traceback.print_exc()

    # 2c. Roads PC1 continuous
    # Long-tailed road density distribution: a few dense urban hexes saturate
    # the colormap, washing out the A-network. Fix: percentile-clip (5/90) so
    # the highway corridors pop against the rural background.
    try:
        roads_path = paths.root / "stage1_unimodal" / "roads" / "netherlands_res9_latest.parquet"
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
        n_done += 1
        del roads, joined, pcs, img
        gc.collect()
    except Exception as e:
        log.error("  ch2.2c roads failed: %s", e); traceback.print_exc()

    # 2d. GTFS PC1 continuous
    try:
        gtfs_path = paths.root / "stage1_unimodal" / "gtfs" / "netherlands_res9_latest.parquet"
        gtfs = pd.read_parquet(gtfs_path)
        _render_continuous_panel(
            paths, boundary, regions, gtfs,
            out_dir / "gtfs_accessibility_res9.png",
            cmap="magma",
            plot_config_extra={"modality": "gtfs", "year": "latest"},
            source_arts=[gtfs_path],
            note="GTFS embeddings -> PC1 (sparse — fallback grey for non-stop hexes)",
        )
        n_done += 1
        del gtfs
        gc.collect()
    except Exception as e:
        log.error("  ch2.2d gtfs failed: %s", e); traceback.print_exc()

    # 2e. 2x2 collage of all four modalities
    try:
        from PIL import Image
        panel_files = [
            out_dir / "alphaearth_pc1_res9.png",
            out_dir / "poi_kmeans_res9.png",
            out_dir / "roads_density_res9.png",
            out_dir / "gtfs_accessibility_res9.png",
        ]
        if all(p.exists() for p in panel_files):
            tiles = [Image.open(p).convert("RGB") for p in panel_files]
            # Resize each tile to RASTER_W//2 x RASTER_H//2 for a clean 2x2
            tw, th = RASTER_W, RASTER_H
            resized = [t.resize((tw, th), Image.LANCZOS) for t in tiles]
            collage = Image.new("RGB", (tw * 2, th * 2), "white")
            collage.paste(resized[0], (0, 0))
            collage.paste(resized[1], (tw, 0))
            collage.paste(resized[2], (0, th))
            collage.paste(resized[3], (tw, th))
            fp = out_dir / "four_modalities_2x2.png"
            collage.save(fp, dpi=(DPI, DPI))
            # Provenance via plain yaml (no matplotlib figure)
            cfg = {"mode": "2x2_collage", "panels": [str(p.name) for p in panel_files]}
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            doc = {
                "figure_path": fp.name, "created_at": now,
                "plot_config": cfg, "plot_config_hash": compute_config_hash(cfg),
                "source_artifacts": [str(p.name) for p in panel_files],
                "producer_script": "scripts/one_off/build_the_book_2026_05_03.py",
                "note": "4-modality 2x2 collage at 4000x4800 px",
                "schema_version": "1.0",
            }
            with fp.with_suffix(".png.provenance.yaml").open("w", encoding="utf-8") as fh:
                yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
            PROVENANCE_INDEX[str(fp.relative_to(_PROJECT_ROOT))] = {
                "plot_config_hash": compute_config_hash(cfg),
                "created_at": now, "note": "2x2 collage",
            }
            log.info("    wrote %s", fp.name)
            n_done += 1
        else:
            missing = [p.name for p in panel_files if not p.exists()]
            log.warning("  ch2.2e collage skipped — missing tiles: %s", missing)
    except Exception as e:
        log.error("  ch2.2e collage failed: %s", e); traceback.print_exc()

    return n_done


# ---------------------------------------------------------------------------
# Chapter 3 — Voronoi Showcase
# ---------------------------------------------------------------------------


def chapter_3_voronoi_showcase(paths: StudyAreaPaths, boundary: gpd.GeoDataFrame) -> int:
    log.info("Chapter 3 — Voronoi Showcase")
    out_dir = BOOK_ROOT / "ch3_voronoi_showcase"
    _ensure_dir(out_dir)
    n_done = 0
    regions = _load_regions_metric(paths, 9)

    # 3a. Continuous mode — UNet PC1
    try:
        unet_path = paths.fused_embedding_file("unet", 9, "20mix")
        unet = pd.read_parquet(unet_path)
        _render_continuous_panel(
            paths, boundary, regions, unet,
            out_dir / "mode_continuous.png",
            cmap="viridis",
            plot_config_extra={"source": "unet_20mix", "demonstrates": "rasterize_continuous_voronoi"},
            source_arts=[unet_path],
            note="Continuous mode demo — UNet 20mix res9 PC1",
        )
        n_done += 1
        # Keep unet in scope below — del it after we use it twice
    except Exception as e:
        log.error("  ch3.3a continuous failed: %s", e); traceback.print_exc()

    # 3b. Categorical — kmeans k=10 on UNet
    try:
        unet_path = paths.fused_embedding_file("unet", 9, "20mix")
        unet = pd.read_parquet(unet_path)
        cx, cy, extent_m, joined = _join_to_regions(unet, regions)
        emb_cols = _emb_columns(joined)
        X = joined[emb_cols].to_numpy(dtype=np.float32)
        km = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=4096, n_init=3)
        labels = km.fit_predict(X).astype(np.int64)
        pixel_m, max_dist_m = voronoi_params_for_resolution(9)
        img, ext_xy = rasterize_categorical_voronoi(
            cx, cy, labels, extent_m, n_clusters=10, cmap="tab10",
            pixel_m=pixel_m, max_dist_m=max_dist_m,
            bg_color=FALLBACK_GREY,
        )
        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        fp = out_dir / "mode_categorical.png"
        _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fp)
        _save_with_provenance(
            fig, fp,
            plot_config={"source": "unet_20mix", "mode": "categorical_kmeans", "k": 10,
                         "random_state": 42, "demonstrates": "rasterize_categorical_voronoi",
                         "cmap": "tab10", "pixel_m": pixel_m, "max_dist_m": max_dist_m},
            source_artifacts=[unet_path],
            note="Categorical mode demo — kmeans k=10 on UNet 20mix",
        )
        n_done += 1
        # Stash kmeans for later reuse if helpful
        _stash_unet_kmeans10 = labels
        del unet, X
        gc.collect()
    except Exception as e:
        log.error("  ch3.3b categorical failed: %s", e); traceback.print_exc()

    # 3c. Binary — LBM >= median threshold
    try:
        lbm_path = paths.root / "target" / "leefbaarometer" / "leefbaarometer_h3res9_2022.parquet"
        lbm = pd.read_parquet(lbm_path)
        # Inner-join: hexes that have LBM
        cx, cy, extent_m, joined = _join_to_regions(lbm[["lbm"]], regions)
        thr = float(np.nanmedian(joined["lbm"].values))
        labels_bin = (joined["lbm"].values >= thr).astype(np.int64)
        pixel_m, max_dist_m = voronoi_params_for_resolution(9)
        # 1 = above-median (orange), 0 = below (light grey)
        img, ext_xy = rasterize_categorical_voronoi(
            cx, cy, labels_bin, extent_m,
            color_map={0: (0.85, 0.85, 0.85), 1: (0.85, 0.40, 0.18)},
            fallback_color=(0.85, 0.85, 0.85),
            pixel_m=pixel_m, max_dist_m=max_dist_m,
            bg_color=(1.0, 1.0, 1.0),
        )
        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        fp = out_dir / "mode_binary.png"
        _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fp)
        _save_with_provenance(
            fig, fp,
            plot_config={"source": "lbm_2022", "mode": "binary_threshold",
                         "threshold": thr, "demonstrates": "rasterize_binary_voronoi (via categorical_voronoi)",
                         "pixel_m": pixel_m, "max_dist_m": max_dist_m},
            source_artifacts=[lbm_path],
            note=f"Binary mode demo — LBM >= median ({thr:.4f})",
        )
        n_done += 1
        del lbm
        gc.collect()
    except Exception as e:
        log.error("  ch3.3c binary failed: %s", e); traceback.print_exc()

    # 3d. RGB — concat embeddings PCA top-3
    try:
        concat_path = paths.fused_embedding_file("concat", 9, "20mix")
        concat = pd.read_parquet(concat_path)
        cx, cy, extent_m, joined = _join_to_regions(concat, regions)
        emb_cols = _emb_columns(joined)
        log.info("  concat 20mix: %d cols", len(emb_cols))
        pcs = _pca_to(joined, emb_cols, n_components=3)
        rgb = np.zeros_like(pcs, dtype=np.float32)
        for i in range(3):
            v = pcs[:, i]
            lo, hi = np.percentile(v, [2, 98])
            rgb[:, i] = np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        pixel_m, max_dist_m = voronoi_params_for_resolution(9)
        img, ext_xy = rasterize_rgb_voronoi(cx, cy, rgb, extent_m,
                                            pixel_m=pixel_m, max_dist_m=max_dist_m)
        out = img.copy()
        outside = out[..., 3] <= 0.0
        out[outside, :3] = 1.0
        out[outside, 3] = 1.0
        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        fp = out_dir / "mode_rgb.png"
        _save_clean_figure(fig, ax, out, ext_xy, boundary, fig_path=fp)
        _save_with_provenance(
            fig, fp,
            plot_config={"source": "concat_20mix", "mode": "rgb_pca3",
                         "demonstrates": "rasterize_rgb_voronoi",
                         "rgb_norm": "p2_p98", "pixel_m": pixel_m, "max_dist_m": max_dist_m},
            source_artifacts=[concat_path],
            note="RGB mode demo — concat 20mix PCA top-3 -> RGB",
        )
        n_done += 1
        del concat, pcs, rgb, img, out
        gc.collect()
    except Exception as e:
        log.error("  ch3.3d rgb failed: %s", e); traceback.print_exc()

    return n_done


# ---------------------------------------------------------------------------
# Chapter 4 — Resolution Hierarchy
# ---------------------------------------------------------------------------


def chapter_4_hierarchy(paths: StudyAreaPaths, boundary: gpd.GeoDataFrame) -> int:
    log.info("Chapter 4 — Resolution Hierarchy")
    out_dir = BOOK_ROOT / "ch4_hierarchy"
    _ensure_dir(out_dir)
    n_done = 0

    for res in (7, 8, 9):
        try:
            unet_path = paths.fused_embedding_file("unet", res, "20mix")
            if not unet_path.exists():
                log.warning("  ch4 unet res%d missing: %s — SKIP", res, unet_path)
                continue
            regions_res = _load_regions_metric(paths, res)
            unet = pd.read_parquet(unet_path)
            cx, cy, extent_m, joined = _join_to_regions(unet, regions_res)
            emb_cols = _emb_columns(joined)
            pcs = _pca_to(joined, emb_cols, n_components=1)
            values = pcs[:, 0]
            pixel_m, max_dist_m = voronoi_params_for_resolution(res)
            img, ext_xy = rasterize_continuous_voronoi(
                cx, cy, values, extent_m, cmap="viridis",
                pixel_m=pixel_m, max_dist_m=max_dist_m, bg_color=FALLBACK_GREY,
            )
            fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
            fp = out_dir / f"unet_pc1_res{res}.png"
            _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fp)
            _save_with_provenance(
                fig, fp,
                plot_config={"model": "unet", "year": "20mix", "resolution": res,
                             "mode": "continuous_pc1", "cmap": "viridis",
                             "pixel_m": pixel_m, "max_dist_m": max_dist_m},
                source_artifacts=[unet_path],
                note=f"UNet PC1 at res{res} (auto-tuned Voronoi params)",
            )
            n_done += 1
            del unet, regions_res, joined, pcs
            gc.collect()
        except Exception as e:
            log.error("  ch4 res%d failed: %s", res, e); traceback.print_exc()

    # 4d. multiscale_avg vs multiscale_concat side-by-side at res9
    try:
        regions_9 = _load_regions_metric(paths, 9)
        avg_path = paths.model_embeddings("unet") / "netherlands_res9_multiscale_avg_20mix.parquet"
        concat_path = paths.model_embeddings("unet") / "netherlands_res9_multiscale_concat_20mix.parquet"
        if avg_path.exists() and concat_path.exists():
            fig, axes = plt.subplots(1, 2, figsize=(2 * RASTER_W / DPI, RASTER_H / DPI))
            for ax_, label, p in zip(axes, ("multiscale_avg", "multiscale_concat"),
                                     (avg_path, concat_path)):
                df = pd.read_parquet(p)
                cx, cy, extent_m, joined = _join_to_regions(df, regions_9)
                emb_cols = _emb_columns(joined)
                pcs = _pca_to(joined, emb_cols, n_components=1)
                pixel_m, max_dist_m = voronoi_params_for_resolution(9)
                img, ext_xy = rasterize_continuous_voronoi(
                    cx, cy, pcs[:, 0], extent_m, cmap="viridis",
                    pixel_m=pixel_m, max_dist_m=max_dist_m, bg_color=FALLBACK_GREY,
                )
                if boundary is not None:
                    boundary.plot(ax=ax_, facecolor="none", edgecolor=BOUNDARY_EDGE,
                                  linewidth=BOUNDARY_LW, zorder=3)
                minx, maxx, miny, maxy = ext_xy
                ax_.imshow(img, extent=[minx, maxx, miny, maxy], origin="lower",
                           aspect="equal", interpolation="nearest", zorder=2)
                ax_.set_xlim(minx, maxx); ax_.set_ylim(miny, maxy)
                ax_.set_xticks([]); ax_.set_yticks([])
                for s in ax_.spines.values(): s.set_visible(False)
                ax_.set_title(label, fontsize=14, pad=8)
                del df, joined, pcs
                gc.collect()
            fig.tight_layout()
            fp = out_dir / "multiscale_avg_vs_concat_res9.png"
            _save_with_provenance(
                fig, fp,
                plot_config={"model": "unet_multiscale", "year": "20mix",
                             "resolution": 9, "panels": ["avg", "concat"],
                             "mode": "continuous_pc1_2panel"},
                source_artifacts=[avg_path, concat_path],
                note="UNet multiscale_avg vs multiscale_concat side-by-side",
            )
            n_done += 1
        else:
            log.warning("  ch4.4d multiscale files missing — SKIP")
    except Exception as e:
        log.error("  ch4.4d multiscale failed: %s", e); traceback.print_exc()

    return n_done


# ---------------------------------------------------------------------------
# Chapter 6 — Clusters of the Land
# ---------------------------------------------------------------------------


def chapter_6_clusters(paths: StudyAreaPaths, boundary: gpd.GeoDataFrame) -> int:
    log.info("Chapter 6 — Clusters of the Land")
    out_dir = BOOK_ROOT / "ch6_clusters"
    _ensure_dir(out_dir)
    n_done = 0
    regions = _load_regions_metric(paths, 9)
    unet_path = paths.fused_embedding_file("unet", 9, "20mix")
    try:
        unet = pd.read_parquet(unet_path)
        cx, cy, extent_m, joined = _join_to_regions(unet, regions)
        emb_cols = _emb_columns(joined)
        X = joined[emb_cols].to_numpy(dtype=np.float32)
        log.info("  Ch6 base: %d hexes x %d cols", len(X), len(emb_cols))
    except Exception as e:
        log.error("  ch6 base load failed: %s", e); traceback.print_exc()
        return n_done

    for k, cmap in ((5, "tab10"), (10, "tab10"), (20, "tab20")):
        try:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=3)
            labels = km.fit_predict(X).astype(np.int64)
            pixel_m, max_dist_m = voronoi_params_for_resolution(9)
            img, ext_xy = rasterize_categorical_voronoi(
                cx, cy, labels, extent_m, n_clusters=k, cmap=cmap,
                pixel_m=pixel_m, max_dist_m=max_dist_m, bg_color=FALLBACK_GREY,
            )
            fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
            fp = out_dir / f"clusters_k{k}_voronoi.png"
            _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fp)
            _save_with_provenance(
                fig, fp,
                plot_config={"source": "unet_20mix", "resolution": 9,
                             "mode": "categorical_kmeans", "k": k,
                             "random_state": 42, "cmap": cmap,
                             "pixel_m": pixel_m, "max_dist_m": max_dist_m},
                source_artifacts=[unet_path],
                note=f"kmeans k={k} on UNet 20mix res9, Voronoi-rasterized",
            )
            n_done += 1
        except Exception as e:
            log.error("  ch6 k=%d failed: %s", k, e); traceback.print_exc()

    del unet, X, joined
    gc.collect()
    return n_done


# ---------------------------------------------------------------------------
# Chapter 7 — Liveability
# ---------------------------------------------------------------------------


def chapter_7_liveability(paths: StudyAreaPaths, boundary: gpd.GeoDataFrame) -> int:
    log.info("Chapter 7 — Liveability")
    out_dir = BOOK_ROOT / "ch7_liveability"
    _ensure_dir(out_dir)
    n_done = 0
    regions = _load_regions_metric(paths, 9)
    lbm_path = paths.root / "target" / "leefbaarometer" / "leefbaarometer_h3res9_2022.parquet"

    try:
        lbm = pd.read_parquet(lbm_path)
    except Exception as e:
        log.error("  ch7 load lbm failed: %s", e); return 0

    # 7a. LBM target raw
    try:
        cx, cy, extent_m, joined = _join_to_regions(lbm[["lbm"]], regions)
        values = joined["lbm"].to_numpy()
        pixel_m, max_dist_m = voronoi_params_for_resolution(9)
        img, ext_xy = rasterize_continuous_voronoi(
            cx, cy, values, extent_m, cmap="RdYlGn",
            pixel_m=pixel_m, max_dist_m=max_dist_m, bg_color=FALLBACK_GREY,
        )
        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        fp = out_dir / "lbm_target_res9.png"
        _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fp)
        _save_with_provenance(
            fig, fp,
            plot_config={"target": "leefbaarometer", "year": 2022,
                         "resolution": 9, "mode": "continuous", "cmap": "RdYlGn",
                         "pixel_m": pixel_m, "max_dist_m": max_dist_m},
            source_artifacts=[lbm_path],
            note="Leefbaarometer 2022 raw target (RdYlGn diverging)",
        )
        n_done += 1
    except Exception as e:
        log.error("  ch7.7a target failed: %s", e); traceback.print_exc()

    # 7b. LBM prediction — inline Ridge on UNet ring_agg
    try:
        # Try ring_agg first (per CLAUDE.md "current best performer"), fall back to UNet
        ring_agg_path = paths.fused_embedding_file("ring_agg", 9, "20mix")
        emb_path = ring_agg_path if ring_agg_path.exists() else paths.fused_embedding_file("unet", 9, "20mix")
        emb = pd.read_parquet(emb_path)
        # Align with LBM via region_id intersection
        if emb.index.name != "region_id":
            if "region_id" in emb.columns:
                emb = emb.set_index("region_id")
            elif "h3_index" in emb.columns:
                emb = emb.rename(columns={"h3_index": "region_id"}).set_index("region_id")
        common = emb.index.intersection(lbm.index)
        log.info("  ch7.7b: %d shared region_ids (emb=%d, lbm=%d)", len(common), len(emb), len(lbm))
        emb_cols = _emb_columns(emb)
        X = emb.loc[common, emb_cols].to_numpy(dtype=np.float32)
        y = lbm.loc[common, "lbm"].to_numpy(dtype=np.float32)
        # Drop NaN
        good = np.isfinite(y) & np.isfinite(X).all(axis=1)
        Xg, yg = X[good], y[good]
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(Xg, yg)
        y_pred = np.full(len(common), np.nan, dtype=np.float32)
        y_pred[good] = ridge.predict(Xg).astype(np.float32)
        r2 = ridge.score(Xg, yg)
        log.info("  ch7 inline Ridge R^2 = %.3f", r2)

        # Build a dataframe to render
        pred_df = pd.DataFrame({"pred": y_pred, "target": y, "resid": y_pred - y},
                               index=common)
        # 7b plot — predictions
        cx, cy, extent_m, joined = _join_to_regions(pred_df[["pred"]], regions)
        pixel_m, max_dist_m = voronoi_params_for_resolution(9)
        img, ext_xy = rasterize_continuous_voronoi(
            cx, cy, joined["pred"].to_numpy(), extent_m, cmap="RdYlGn",
            pixel_m=pixel_m, max_dist_m=max_dist_m, bg_color=FALLBACK_GREY,
        )
        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        fp = out_dir / "lbm_prediction_res9.png"
        _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fp)
        _save_with_provenance(
            fig, fp,
            plot_config={"target": "leefbaarometer", "year": 2022,
                         "resolution": 9, "mode": "continuous_prediction",
                         "model": "ridge_inline", "alpha": 1.0,
                         "embedding_source": str(emb_path.relative_to(_PROJECT_ROOT)),
                         "r2_in_sample": float(r2),
                         "cmap": "RdYlGn",
                         "pixel_m": pixel_m, "max_dist_m": max_dist_m},
            source_artifacts=[emb_path, lbm_path],
            note=f"Inline Ridge prediction of LBM (in-sample R^2={r2:.3f})",
        )
        n_done += 1

        # 7c residuals
        cx, cy, extent_m, joined = _join_to_regions(pred_df[["resid"]], regions)
        resid = joined["resid"].to_numpy()
        # Symmetric range around 0 using p2-p98
        finite = np.isfinite(resid)
        amp = float(np.nanpercentile(np.abs(resid[finite]), 95))
        img, ext_xy = rasterize_continuous_voronoi(
            cx, cy, resid, extent_m, cmap="RdBu_r",
            vmin=-amp, vmax=amp,
            pixel_m=pixel_m, max_dist_m=max_dist_m, bg_color=FALLBACK_GREY,
        )
        fig, ax = plt.subplots(figsize=(RASTER_W / DPI, RASTER_H / DPI))
        fp = out_dir / "lbm_residuals_res9.png"
        _save_clean_figure(fig, ax, img, ext_xy, boundary, fig_path=fp)
        _save_with_provenance(
            fig, fp,
            plot_config={"target": "leefbaarometer", "year": 2022,
                         "resolution": 9, "mode": "continuous_residual",
                         "model": "ridge_inline", "alpha": 1.0,
                         "embedding_source": str(emb_path.relative_to(_PROJECT_ROOT)),
                         "vrange": [-amp, amp], "cmap": "RdBu_r (diverging)",
                         "pixel_m": pixel_m, "max_dist_m": max_dist_m},
            source_artifacts=[emb_path, lbm_path],
            note="Ridge LBM residuals (prediction - target), diverging RdBu_r centred on 0",
        )
        n_done += 1
        del emb, X, y, y_pred, pred_df
        gc.collect()
    except Exception as e:
        log.error("  ch7.7b/c probe failed: %s", e); traceback.print_exc()

    return n_done


# ---------------------------------------------------------------------------
# Chapter 8 — Closing / Colophon
# ---------------------------------------------------------------------------


def chapter_8_closing(paths: StudyAreaPaths, boundary: gpd.GeoDataFrame) -> int:
    log.info("Chapter 8 — Closing / Colophon")
    out_dir = BOOK_ROOT / "ch8_closing"
    _ensure_dir(out_dir)
    n_done = 0

    # 8a. Best-of: re-render AE RGB cover at 4000 x 4800
    try:
        regions = _load_regions_metric(paths, 9)
        ae_path = paths.embedding_file("alphaearth", 9, 2022)
        ae = pd.read_parquet(ae_path)
        cx, cy, extent_m, joined = _join_to_regions(ae, regions)
        emb_cols = [c for c in joined.columns if c.startswith("A") and c[1:].isdigit()]
        pcs = _pca_to(joined, emb_cols, n_components=3)
        rgb = np.zeros_like(pcs, dtype=np.float32)
        for i in range(3):
            v = pcs[:, i]
            lo, hi = np.percentile(v, [2, 98])
            rgb[:, i] = np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        # Tighter pixel_m for higher fidelity at poster scale
        pixel_m, max_dist_m = 150.0, 220.0
        img, ext_xy = rasterize_rgb_voronoi(cx, cy, rgb, extent_m,
                                            pixel_m=pixel_m, max_dist_m=max_dist_m)
        out = img.copy()
        outside = out[..., 3] <= 0.0
        out[outside, :3] = 1.0
        out[outside, 3] = 1.0
        # Larger figure: 4000 x 4800 px
        big_w_in = 4000 / DPI
        big_h_in = 4800 / DPI
        fig, ax = plt.subplots(figsize=(big_w_in, big_h_in))
        _save_clean_figure(fig, ax, out, ext_xy, boundary,
                           fig_path=out_dir / "best_of_high_res.png")
        _save_with_provenance(
            fig, out_dir / "best_of_high_res.png",
            plot_config={
                "source": "alphaearth_2022_res9",
                "mode": "rgb_pca3_high_res",
                "pixel_m": pixel_m, "max_dist_m": max_dist_m,
                "raster_px": [4000, 4800],
            },
            source_artifacts=[ae_path],
            note="Poster-grade re-render of cover (4000x4800)",
        )
        n_done += 1
        del joined, ae, pcs, rgb, img, out
        gc.collect()
    except Exception as e:
        log.error("  ch8.8a best-of failed: %s", e); traceback.print_exc()

    # 8b. Aggregate provenance YAML
    try:
        n_done += _regenerate_book_aggregate()
    except Exception as e:
        log.error("  ch8.8b aggregate failed: %s", e); traceback.print_exc()

    return n_done


def _regenerate_book_aggregate() -> int:
    """Walk BOOK_ROOT for *.provenance.yaml and rebuild book_provenance.yaml.

    Extracted so it can run as a standalone fix-wave action after partial
    figure regenerations without re-rendering best_of_high_res.
    """
    out_dir = BOOK_ROOT / "ch8_closing"
    _ensure_dir(out_dir)
    agg_path = out_dir / "book_provenance.yaml"
    agg = {}
    for sidecar in BOOK_ROOT.rglob("*.provenance.yaml"):
        try:
            doc = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
            key = str(sidecar.parent.relative_to(_PROJECT_ROOT) / sidecar.stem.replace(".png", ""))
            agg[key] = {
                "plot_config_hash": doc.get("plot_config_hash"),
                "created_at": doc.get("created_at"),
                "parent_run_id": doc.get("parent_run_id"),
                "note": doc.get("note", ""),
                "source_artifacts": doc.get("source_artifacts", []),
            }
        except Exception:
            pass
    with agg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"book_title": "The Book of Netherlands",
             "build_date": str(date.today()),
             "n_figures": len(agg),
             "figures": agg},
            fh, sort_keys=False, default_flow_style=False,
        )
    log.info("  wrote %s with %d figures", agg_path, len(agg))
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _chapter_funcs() -> dict:
    return {
        "ch1": chapter_1_frontispiece,
        "ch2": chapter_2_modalities,
        "ch3": chapter_3_voronoi_showcase,
        "ch4": chapter_4_hierarchy,
        "ch6": chapter_6_clusters,
        "ch7": chapter_7_liveability,
        "ch8": chapter_8_closing,
    }


def _parse_chapters_arg(argv: list[str]) -> list[str] | None:
    """Parse `--chapters ch1,ch2,ch8` from argv. Returns ordered list or None."""
    for i, tok in enumerate(argv):
        if tok == "--chapters" and i + 1 < len(argv):
            return [c.strip() for c in argv[i + 1].split(",") if c.strip()]
        if tok.startswith("--chapters="):
            return [c.strip() for c in tok.split("=", 1)[1].split(",") if c.strip()]
    return None


def main() -> int:
    paths = StudyAreaPaths(STUDY_AREA)
    _ensure_dir(BOOK_ROOT)

    # Standalone aggregator path: rebuild book_provenance.yaml from existing
    # sidecars without rerunning any chapter renders.
    if "--aggregate-only" in sys.argv[1:]:
        log.info("Aggregate-only mode")
        _regenerate_book_aggregate()
        return 0

    log.info("Loading boundary...")
    boundary = load_boundary(paths)
    if boundary is None:
        log.error("Boundary not found — aborting")
        return 1

    cfg = {
        "book_title": "The Book of Netherlands",
        "study_area": STUDY_AREA,
        "build_date": str(date.today()),
        "raster_px": [RASTER_W, RASTER_H],
        "dpi": DPI,
    }

    counts: dict[str, int] = {}
    funcs = _chapter_funcs()
    requested = _parse_chapters_arg(sys.argv[1:])
    if requested is None:
        chapters_to_run = list(funcs.keys())
    else:
        unknown = [c for c in requested if c not in funcs]
        if unknown:
            log.error("Unknown chapter(s): %s. Valid: %s", unknown, list(funcs.keys()))
            return 2
        chapters_to_run = requested
        log.info("Partial build: chapters %s", chapters_to_run)

    with SidecarWriter(
        artifact_path=BOOK_ROOT / "build.book",
        config=cfg,
        input_paths=[],
        producer_script="scripts/one_off/build_the_book_2026_05_03.py",
        study_area=STUDY_AREA,
        stage="stage3",
        seed=42,
    ) as sw:
        sw.output_paths.append(BOOK_ROOT)

        for ch in chapters_to_run:
            counts[ch] = funcs[ch](paths, boundary)

        sw.extra["counts"] = counts
        sw.extra["total_pngs"] = sum(counts.values())
        if requested is not None:
            sw.extra["partial_build"] = True
            sw.extra["chapters_run"] = chapters_to_run

    total = sum(counts.values())
    log.info("=" * 60)
    log.info("BOOK BUILD COMPLETE: %d PNGs", total)
    for ch, n in counts.items():
        log.info("  %s: %d", ch, n)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
