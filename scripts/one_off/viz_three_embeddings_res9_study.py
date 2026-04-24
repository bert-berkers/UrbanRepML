"""High-resolution comparative visual study: concat vs ring_agg vs unet at res9 NL.

Lifetime: temporary (30-day shelf life, expires ~2026-05-24).
Stage: stage3 (visualisation production -- no model/probe work).

Inputs:
  data/study_areas/netherlands/stage2_multimodal/concat/embeddings/netherlands_res9_20mix.parquet
  data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet
  data/study_areas/netherlands/stage2_multimodal/unet/embeddings/netherlands_res9_20mix.parquet

Outputs (all under data/study_areas/netherlands/stage3_analysis/2026-04-24/panels/):
  concat/{clusters_tab10, clusters_dark2, pcrgb_rank, pc1_turbo}.png
  ring_agg/{clusters_tab10, clusters_dark2, pcrgb_rank, pc1_turbo}.png
  unet/{clusters_tab10, clusters_dark2, pcrgb_rank, pc1_turbo}.png
  comparison/{clusters_tab10_3way, pcrgb_rank_3way, pc1_turbo_3way}.png
  control/province_boundaries.png
  stats/{concat, ring_agg, unet}.json   (cluster sizes, centroids, PCA var)

All rasters use KDTree-Voronoi nearest-neighbour assignment in EPSG:28992
(RD New) with PIXEL_M=150 (sharper than the 250 m gallery, closer to native
res9 hex edge ~174 m), MAX_DIST_M=300 m for the cutoff.

Per-panel figsize=(8, 9.6), dpi=250 -> ~2000x2400 px per panel.

Production-only: titles are factual ("Concat | clusters tab10 k=10 | 208D")
without interpretive language. The coordinator analyses the resulting images
for what they reveal about Dutch urban geography; this script's job is purely
to render faithful panels.

KMeans uses random_state=42 across all three embeddings (cluster numbers
will still be permuted across embeddings since the seed only freezes the
initialisation -- arbitrary cluster labels are expected). PCA is fit
PER EMBEDDING (PCs are not the same axis across different embedding spaces).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import load_boundary

# Re-use the canonical Voronoi helpers from the wave-3 gallery script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_ring_agg_res9_grid import (  # type: ignore[import-not-found]
    gather_rgba,
    hsv_pc_rgb,  # noqa: F401  (kept for parity if needed downstream)
    rank_rgb,
    rank_unit,
    voronoi_indices,
)

STUDY_AREA = "netherlands"
RES = 9
K_CLUSTERS = 10
KMEANS_SEED = 42  # Frozen across all 3 embeddings per task spec.
PIXEL_M = 150.0   # Sharper than the 250 m gallery -- closer to native res9 hex edge ~174 m.
MAX_DIST_M = 300.0
GAMMA = 0.75
BG = "#23262e"

# Per-panel figure metadata.
PANEL_FIGSIZE = (8, 9.6)
PANEL_DPI = 250

# Three-way comparison figure metadata: 1 row x 3 cols, each panel ~2000x2400.
THREEWAY_FIGSIZE = (24, 9.6)
THREEWAY_DPI = 200  # 24 * 200 = 4800 px wide, 9.6 * 200 = 1920 px tall.

EMBEDDINGS: list[tuple[str, str, str]] = [
    # (key, parquet_relpath, descriptor)
    ("concat",
     "stage2_multimodal/concat/embeddings/netherlands_res9_20mix.parquet",
     "Concat (late fusion, no smoothing)"),
    ("ring_agg",
     "stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet",
     "Ring-Agg k=10 (zero-param spatial smoothing)"),
    ("unet",
     "stage2_multimodal/unet/embeddings/netherlands_res9_20mix.parquet",
     "U-Net (learned)"),
]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _styled_axes(ax) -> None:
    """Apply slate background + no spines/ticks/grid to a single Axes."""
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")


def _save_panel(
    img: np.ndarray,
    extent_xy: tuple[float, float, float, float],
    title: str,
    out_path: Path,
    figsize: tuple[float, float] = PANEL_FIGSIZE,
    dpi: int = PANEL_DPI,
) -> Path:
    """Save a single Voronoi raster panel to disk.

    Title is purely factual (embedding name + dim + encoding) -- no
    interpretive language so the coordinator's downstream visual analysis
    isn't primed.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG)
    _styled_axes(ax)
    ax.imshow(img, extent=extent_xy, origin="lower", interpolation="nearest")
    ax.set_title(title, color="white", fontsize=14, pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_threeway(
    panels: list[tuple[np.ndarray, str]],
    extent_xy: tuple[float, float, float, float],
    suptitle: str,
    out_path: Path,
) -> Path:
    """Save a 1x3 side-by-side comparison panel.

    Args:
        panels: list of ``(img, title)`` triples, in display order
            (concat, ring_agg, unet).
        extent_xy: shared imshow extent.
        suptitle: factual figure title.
        out_path: destination PNG path.
    """
    fig, axes = plt.subplots(1, 3, figsize=THREEWAY_FIGSIZE, dpi=THREEWAY_DPI)
    fig.patch.set_facecolor(BG)
    for ax, (img, title) in zip(axes, panels):
        _styled_axes(ax)
        ax.imshow(img, extent=extent_xy, origin="lower", interpolation="nearest")
        ax.set_title(title, color="white", fontsize=14, pad=10)
    fig.suptitle(suptitle, color="white", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=THREEWAY_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Per-embedding pipeline
# ---------------------------------------------------------------------------


def _build_per_hex_colour_tables(
    pcs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the four per-hex (N, 3) RGB tables for the four encodings.

    Args:
        pcs: ``(N, 3)`` PCA components.
        labels: ``(N,)`` integer cluster labels in ``[0, K)``.

    Returns:
        dict with keys ``clusters_tab10``, ``clusters_dark2``,
        ``pcrgb_rank``, ``pc1_turbo``.
    """
    tab10 = colormaps["tab10"]
    dark2 = colormaps["Dark2"]

    # Dark2 has only 8 colours -- modulo K (10) means clusters 8 and 9
    # repeat colours 0 and 1; that's fine for an "alternative aesthetic"
    # encoding and matches the wave-3 gallery convention.
    return {
        "clusters_tab10": tab10(labels % 10)[:, :3].astype(np.float32),
        "clusters_dark2": dark2(labels % dark2.N)[:, :3].astype(np.float32),
        "pcrgb_rank": rank_rgb(pcs, gamma=GAMMA),
        "pc1_turbo": colormaps["turbo"](
            rank_unit(pcs[:, 0], 0.0, 1.0)
        )[:, :3].astype(np.float32),
    }


def _cluster_stats(
    labels: np.ndarray,
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    pca: PCA,
) -> dict:
    """Compute per-cluster size + centroid + PCA variance summary.

    Centroids are mean (cx_m, cy_m) of member-hex centroids in EPSG:28992.
    """
    n = len(labels)
    stats: dict = {
        "n_hexes": int(n),
        "k": int(K_CLUSTERS),
        "kmeans_seed": int(KMEANS_SEED),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_explained_variance_total": float(
            pca.explained_variance_ratio_.sum()
        ),
        "clusters": [],
    }
    for cluster_id in range(K_CLUSTERS):
        mask = labels == cluster_id
        m = int(mask.sum())
        if m == 0:
            stats["clusters"].append({
                "cluster_id": int(cluster_id),
                "n_hexes": 0,
                "fraction_of_total": 0.0,
                "centroid_rd_x": None,
                "centroid_rd_y": None,
            })
            continue
        stats["clusters"].append({
            "cluster_id": int(cluster_id),
            "n_hexes": m,
            "fraction_of_total": float(m / n),
            "centroid_rd_x": float(cx_m[mask].mean()),
            "centroid_rd_y": float(cy_m[mask].mean()),
        })
    return stats


def process_embedding(
    key: str,
    src_path: Path,
    descriptor: str,
    nearest_idx: np.ndarray,
    inside: np.ndarray,
    extent_xy: tuple[float, float, float, float],
    cx_m: np.ndarray,
    cy_m: np.ndarray,
    panels_root: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run PCA + KMeans + render 4 panels for one embedding.

    Args:
        key: short embedding key (``concat`` / ``ring_agg`` / ``unet``).
        src_path: parquet path.
        descriptor: human-readable embedding label for panel titles.
        nearest_idx: shared Voronoi index array (H, W).
        inside: shared alpha mask (H, W).
        extent_xy: shared imshow extent.
        cx_m, cy_m: shared per-hex centroid arrays in EPSG:28992
            (must align row-wise with the embedding's index).
        panels_root: ``.../panels/`` directory; per-embedding subdir is
            created inside.

    Returns:
        ``(per_hex_rgb_tables, stats)``.
        ``per_hex_rgb_tables`` is a dict of 4 ``(N, 3)`` colour arrays
        keyed by encoding name (``clusters_tab10`` / ``clusters_dark2`` /
        ``pcrgb_rank`` / ``pc1_turbo``); the threeway comparison step
        re-uses these to gather pixel rasters without recomputing PCA/KMeans.
        ``stats`` is the JSON-serialisable cluster + PCA summary.
    """
    print(f"\n=== {key} ===")
    print(f"[load] {src_path}")
    t0 = time.time()
    df = pd.read_parquet(src_path)
    X = df.to_numpy(dtype=np.float32)
    print(f"  -> {X.shape} in {time.time() - t0:.1f}s")
    dim = X.shape[1]

    print("[pca] 3 components")
    t0 = time.time()
    pca = PCA(n_components=3, random_state=0)
    pcs = pca.fit_transform(X)
    var_total = pca.explained_variance_ratio_.sum()
    print(f"  -> explained var: "
          f"{pca.explained_variance_ratio_.round(3).tolist()} "
          f"(sum={var_total:.3f}) in {time.time() - t0:.1f}s")

    print(f"[kmeans] k={K_CLUSTERS} seed={KMEANS_SEED}")
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=K_CLUSTERS,
        random_state=KMEANS_SEED,
        batch_size=4096,
        n_init=4,
    )
    labels = km.fit_predict(X)
    print(f"  -> labels in {time.time() - t0:.1f}s")

    rgb_tables = _build_per_hex_colour_tables(pcs, labels)
    stats = _cluster_stats(labels, cx_m, cy_m, pca)

    print(f"[gather+save] 4 panels for {key}")
    t0 = time.time()
    out_dir = panels_root / key
    encodings_titles = {
        "clusters_tab10": (
            f"{descriptor} | KMeans k={K_CLUSTERS} | tab10 | dim={dim}"
        ),
        "clusters_dark2": (
            f"{descriptor} | KMeans k={K_CLUSTERS} | Dark2 | dim={dim}"
        ),
        "pcrgb_rank": (
            f"{descriptor} | PCA(3) -> rank-norm RGB | dim={dim} | "
            f"var={var_total:.1%}"
        ),
        "pc1_turbo": (
            f"{descriptor} | PC1 -> turbo | dim={dim} | "
            f"PC1 var={pca.explained_variance_ratio_[0]:.1%}"
        ),
    }
    for enc_name, rgb_per_hex in rgb_tables.items():
        img = gather_rgba(nearest_idx, inside, rgb_per_hex)
        _save_panel(
            img,
            extent_xy,
            encodings_titles[enc_name],
            out_dir / f"{enc_name}.png",
        )
    print(f"  -> 4 panels in {time.time() - t0:.1f}s")

    return rgb_tables, stats


# ---------------------------------------------------------------------------
# Control panel: province boundaries (or country outline fallback)
# ---------------------------------------------------------------------------


def render_control_boundaries(
    extent_xy: tuple[float, float, float, float],
    out_path: Path,
) -> tuple[Path, str]:
    """Render an admin-boundaries control image at the same extent.

    Tries gemeente (municipality) boundaries from the leefbaarometer
    geometrie GeoPackage first (Option C), since those are already on
    disk in EPSG:28992. Falls back to the study area country outline
    (Option B) if gemeente data isn't available.

    Returns:
        ``(out_path, source_descriptor)``. ``source_descriptor`` is logged
        in the scratchpad to make it explicit which reference the
        coordinator is looking at.
    """
    paths = StudyAreaPaths(STUDY_AREA)
    minx, maxx, miny, maxy = extent_xy

    gpkg = (
        paths.root
        / "target" / "leefbaarometer"
        / "geometrie-lbm3-2024" / "geometrie-lbm3-2024" / "gemeente 2024.gpkg"
    )
    boundary = load_boundary(paths, crs=28992)

    fig, ax = plt.subplots(1, 1, figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    fig.patch.set_facecolor(BG)
    _styled_axes(ax)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    layers_drawn: list[str] = []

    if boundary is not None and not boundary.empty:
        boundary.boundary.plot(
            ax=ax, color="white", linewidth=1.2, alpha=0.9,
        )
        layers_drawn.append("country outline")

    source: str
    if gpkg.exists():
        gem = gpd.read_file(gpkg)
        if gem.crs is None:
            gem = gem.set_crs(28992)
        elif gem.crs.to_epsg() != 28992:
            gem = gem.to_crs(28992)
        # Thin sub-pixel lines so 342 polygons don't crowd into a white blob.
        gem.boundary.plot(ax=ax, color="white", linewidth=0.35, alpha=0.7)
        layers_drawn.append(f"{len(gem)} gemeenten 2024")
        source = "gemeente 2024 (NL municipalities, 342 polygons)"
    else:
        source = (
            "country outline only (gemeente 2024 GeoPackage not found)"
        )

    title = "NL admin boundaries (control) | EPSG:28992 | " + source
    ax.set_title(title, color="white", fontsize=12, pad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PANEL_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> control: {' + '.join(layers_drawn) or '(empty)'} -> {out_path}")
    return out_path, source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    paths = StudyAreaPaths(STUDY_AREA)
    out_root = paths.root / "stage3_analysis" / "2026-04-24" / "panels"
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------- Sanity-check the three embedding parquets -----------------
    indices: dict[str, pd.Index] = {}
    src_paths: dict[str, Path] = {}
    for key, rel, _descriptor in EMBEDDINGS:
        src = paths.root / rel
        if not src.exists():
            raise SystemExit(f"Missing {key} parquet: {src}")
        df = pd.read_parquet(src, columns=[])
        indices[key] = df.index
        src_paths[key] = src
        print(f"[sanity] {key}: {len(df.index):,} rows ({src.name})")

    if not (
        indices["concat"].equals(indices["ring_agg"])
        and indices["concat"].equals(indices["unet"])
    ):
        # Project task expects identity. Don't push through silently.
        common = indices["concat"].intersection(
            indices["ring_agg"]
        ).intersection(indices["unet"])
        diff_concat = indices["concat"].difference(common)
        diff_ring = indices["ring_agg"].difference(common)
        diff_unet = indices["unet"].difference(common)
        raise SystemExit(
            "Hex sets differ across embeddings -- spec asserts they should "
            "be identical. Diffs (vs intersection of all three): "
            f"concat={len(diff_concat)} ring_agg={len(diff_ring)} "
            f"unet={len(diff_unet)} common={len(common)}"
        )
    print(f"[sanity] hex set identical across all 3 embeddings "
          f"({len(indices['concat']):,} hexes)")

    # ---------- Shared centroids + Voronoi (compute once) ----------------
    hex_ids = indices["concat"].to_numpy()
    print(f"\n[centroids] SpatialDB res={RES} crs=28992")
    t0 = time.time()
    db = SpatialDB.for_study_area(STUDY_AREA)
    cx_m, cy_m = db.centroids(list(hex_ids), resolution=RES, crs=28992)
    cx_m = np.asarray(cx_m, dtype=np.float64)
    cy_m = np.asarray(cy_m, dtype=np.float64)
    print(f"  -> centroids in {time.time() - t0:.1f}s "
          f"(x {cx_m.min():.0f}..{cx_m.max():.0f}, "
          f"y {cy_m.min():.0f}..{cy_m.max():.0f})")

    pad_m = 2_000.0
    extent_m = (
        float(cx_m.min() - pad_m),
        float(cy_m.min() - pad_m),
        float(cx_m.max() + pad_m),
        float(cy_m.max() + pad_m),
    )

    print(f"\n[voronoi] KDTree query at PIXEL_M={PIXEL_M:.0f}, "
          f"MAX_DIST_M={MAX_DIST_M:.0f}")
    t0 = time.time()
    nearest_idx, inside, extent_xy = voronoi_indices(
        cx_m, cy_m, extent_m,
        pixel_m=PIXEL_M, max_dist_m=MAX_DIST_M,
    )
    h, w = nearest_idx.shape
    pct_inside = float(inside.sum()) / inside.size
    print(f"  -> {w} x {h} indices in {time.time() - t0:.1f}s "
          f"({pct_inside:.1%} inside cutoff)")

    # ---------- Per-embedding rendering ----------------------------------
    all_rgb_tables: dict[str, dict[str, np.ndarray]] = {}
    all_stats: dict[str, dict] = {}
    for key, _rel, descriptor in EMBEDDINGS:
        rgb_tables, stats = process_embedding(
            key=key,
            src_path=src_paths[key],
            descriptor=descriptor,
            nearest_idx=nearest_idx,
            inside=inside,
            extent_xy=extent_xy,
            cx_m=cx_m,
            cy_m=cy_m,
            panels_root=out_root,
        )
        all_rgb_tables[key] = rgb_tables
        all_stats[key] = stats

    # ---------- Three-way comparisons ------------------------------------
    print("\n[comparison] 1x3 side-by-side panels")
    t0 = time.time()
    comp_dir = out_root / "comparison"
    encodings_for_comparison = {
        "clusters_tab10": (
            f"KMeans k={K_CLUSTERS} | tab10 (labels are arbitrary across embeddings)"
        ),
        "pcrgb_rank": "PCA(3) -> rank-norm RGB (PCs are per-embedding, not aligned across)",
        "pc1_turbo": "PC1 -> turbo (PC1 axis is per-embedding, not aligned across)",
    }
    for enc_name, suptitle in encodings_for_comparison.items():
        panels = []
        for key, _rel, descriptor in EMBEDDINGS:
            img = gather_rgba(
                nearest_idx, inside, all_rgb_tables[key][enc_name]
            )
            panels.append((img, descriptor))
        _save_threeway(
            panels, extent_xy,
            f"{suptitle}",
            comp_dir / f"{enc_name}_3way.png",
        )
    print(f"  -> 3 comparison panels in {time.time() - t0:.1f}s")

    # ---------- Control panel --------------------------------------------
    print("\n[control] admin boundaries")
    t0 = time.time()
    control_path, control_source = render_control_boundaries(
        extent_xy, out_root / "control" / "province_boundaries.png"
    )
    print(f"  -> control in {time.time() - t0:.1f}s")

    # ---------- Stats sidecars -------------------------------------------
    print("\n[stats] writing per-embedding JSON sidecars")
    stats_dir = out_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for key, stats in all_stats.items():
        stats_path = stats_dir / f"{key}.json"
        # Add provenance metadata inline so the json is self-describing.
        stats["_meta"] = {
            "study_area": STUDY_AREA,
            "resolution": RES,
            "year_label": "20mix",
            "pixel_m": PIXEL_M,
            "max_dist_m": MAX_DIST_M,
            "raster_size_px": [int(w), int(h)],
            "extent_rd_xyxy": [
                float(extent_xy[0]), float(extent_xy[2]),
                float(extent_xy[1]), float(extent_xy[3]),
            ],
            "control_source": control_source,
        }
        stats_path.write_text(json.dumps(stats, indent=2))
        print(f"  -> {stats_path}")

    # ---------- Final summary --------------------------------------------
    total_files = sum(
        1 for _ in out_root.rglob("*.png")
    ) + sum(1 for _ in stats_dir.glob("*.json"))
    print(f"\n[done] {total_files} files under {out_root}")
    for sub in ["concat", "ring_agg", "unet", "comparison", "control", "stats"]:
        sub_dir = out_root / sub
        if not sub_dir.exists():
            continue
        files = sorted(sub_dir.iterdir())
        sizes_mb = [f.stat().st_size / 1e6 for f in files]
        print(f"  {sub}/: {len(files)} files, "
              f"{sum(sizes_mb):.2f} MB total "
              f"(range {min(sizes_mb):.2f}-{max(sizes_mb):.2f} MB)")


if __name__ == "__main__":
    main()
