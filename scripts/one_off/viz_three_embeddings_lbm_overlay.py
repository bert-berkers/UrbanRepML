"""W5 follow-up: leefbaarometer probe-target overlay + cluster-centroid annotations.

Lifetime: temporary (30-day shelf life, expires ~2026-05-24).
Stage: stage3 (visualisation production -- closes the descriptive -> supervised loop).

Inputs (all res9, 20mix where applicable):
  data/study_areas/netherlands/target/leefbaarometer/leefbaarometer_h3res9_2022.parquet
  data/study_areas/netherlands/stage2_multimodal/{concat,ring_agg,unet}/embeddings/netherlands_res9_20mix.parquet
  data/study_areas/netherlands/stage3_analysis/2026-04-24/panels/stats/{concat,ring_agg,unet}.json

Outputs (all under data/study_areas/netherlands/stage3_analysis/2026-04-24/panels/):
  target/leefbaarometer.png
  ring_agg/clusters_tab10_annotated.png
  stats/leefbaarometer_per_cluster.json

Re-uses the Voronoi machinery from viz_ring_agg_res9_grid.py:
  voronoi_indices(), gather_rgba(), rank_unit().
Re-uses centroid + extent computation from viz_three_embeddings_res9_study.py
(here we recompute them locally to avoid importing main()).

KMeans is re-fit per embedding with random_state=42 to align cluster IDs
with the existing W4 panels (the W4 stats sidecars store cluster
fractions and centroids that will match these labels exactly).

LBM panel uses viridis (sequential, perceptually uniform) over the LBM
score's empirical range (~3.4 - 5.0). Hexes without LBM data are left
transparent so the rural-water gaps are visible.

Annotation overlay uses white-filled circles outlined in black, with
marker size = sqrt(cluster_fraction) * SCALE so a 25%-cluster marker
isn't 5x the visual area of a 5%-cluster marker. Numeric cluster IDs
are drawn over each marker and a legend lists ID -> %hexes -> mean LBM.

If LBM file or matching hex set is unavailable for any panel, that panel
is skipped, a placeholder PNG is rendered, and the failure is reported
to stderr (not silently swallowed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.patches import Circle
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB

# Re-use the canonical Voronoi helpers from the wave-3 gallery script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_ring_agg_res9_grid import (  # type: ignore[import-not-found]
    gather_rgba,
    rank_unit,
    voronoi_indices,
)

STUDY_AREA = "netherlands"
RES = 9
K_CLUSTERS = 10
KMEANS_SEED = 42
PIXEL_M = 150.0
MAX_DIST_M = 300.0
BG = "#23262e"

PANEL_FIGSIZE = (8, 9.6)
PANEL_DPI = 250


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _styled_axes(ax) -> None:
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal")


def _save_placeholder(out_path: Path, message: str) -> None:
    """Render a slate panel with `target not available` style text."""
    fig, ax = plt.subplots(1, 1, figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    fig.patch.set_facecolor(BG)
    _styled_axes(ax)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        color="white", fontsize=18,
        transform=ax.transAxes,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PANEL_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> placeholder: {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 1: leefbaarometer target panel
# ---------------------------------------------------------------------------


def render_lbm_target_panel(
    lbm_path: Path,
    hex_index: pd.Index,
    nearest_idx: np.ndarray,
    inside: np.ndarray,
    extent_xy: tuple[float, float, float, float],
    out_path: Path,
) -> tuple[bool, dict]:
    """Render the LBM score map at the same Voronoi extent as the embedding panels.

    Args:
        lbm_path: parquet with `lbm` score column, indexed by region_id.
        hex_index: the embedding hex index (must align row-wise with `nearest_idx`).
        nearest_idx, inside, extent_xy: shared Voronoi state.
        out_path: destination PNG.

    Returns:
        ``(ok, lbm_meta)``. ``ok`` is False if the file or hex alignment failed
        and a placeholder was rendered. ``lbm_meta`` contains the score range,
        coverage stats, and aligned per-hex score array (or empty if not ok).
    """
    if not lbm_path.exists():
        _save_placeholder(out_path, "leefbaarometer target not available")
        return False, {}

    print(f"[lbm] loading {lbm_path.name}")
    t0 = time.time()
    lbm_df = pd.read_parquet(lbm_path, columns=["lbm", "weight_sum"])
    print(f"  -> {len(lbm_df):,} rows in {time.time() - t0:.1f}s "
          f"(score range {lbm_df['lbm'].min():.3f}..{lbm_df['lbm'].max():.3f})")

    # Align with embedding hex index. Hexes without LBM data become NaN.
    aligned = lbm_df["lbm"].reindex(hex_index)
    n_total = len(aligned)
    n_with_data = int(aligned.notna().sum())
    coverage_pct = n_with_data / n_total
    print(f"  -> {n_with_data:,}/{n_total:,} hexes have LBM "
          f"({coverage_pct:.1%} coverage)")

    if n_with_data == 0:
        _save_placeholder(
            out_path,
            "leefbaarometer indices don't match embedding hex set",
        )
        return False, {}

    score = aligned.to_numpy(dtype=np.float32)
    has_score = aligned.notna().to_numpy()

    # Build per-hex RGB via viridis on rank-normalised scores so the colormap's
    # full dynamic range is used and the visualisation isn't squashed by the
    # narrow LBM range (3.4-5.0). We separately store the raw range in the title
    # so the reader knows what the colour scale represents.
    lbm_min = float(np.nanmin(score))
    lbm_max = float(np.nanmax(score))
    lbm_mean = float(np.nanmean(score))
    lbm_std = float(np.nanstd(score))

    # Use linear (not rank) normalisation so colours reflect score magnitude,
    # not just rank. Rank would visually amplify minor differences.
    norm = np.where(
        has_score,
        (score - lbm_min) / max(lbm_max - lbm_min, 1e-9),
        0.0,
    )
    viridis = colormaps["viridis"]
    rgb_per_hex = viridis(norm)[:, :3].astype(np.float32)

    # Hexes without LBM data: mask them out by setting alpha to zero in the
    # gather. We do that by zeroing their rgb and using inside_with_data.
    inside_with_data = inside.copy()
    # nearest_idx is per-pixel hex index. A pixel is "missing" if its hex
    # has no LBM score. Build a lookup mask aligned with hex_index.
    missing_hex_mask = ~has_score
    pixel_missing = missing_hex_mask[nearest_idx]
    inside_with_data &= ~pixel_missing

    img = gather_rgba(nearest_idx, inside_with_data, rgb_per_hex)

    fig, ax = plt.subplots(1, 1, figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    fig.patch.set_facecolor(BG)
    _styled_axes(ax)
    ax.imshow(img, extent=extent_xy, origin="lower", interpolation="nearest")
    title = (
        f"Leefbaarometer score (target) | LBM-2022 | viridis | "
        f"range {lbm_min:.2f}-{lbm_max:.2f} | "
        f"coverage {coverage_pct:.0%}"
    )
    ax.set_title(title, color="white", fontsize=12, pad=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PANEL_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out_path}")

    return True, {
        "score": score,
        "has_score": has_score,
        "lbm_min": lbm_min,
        "lbm_max": lbm_max,
        "lbm_mean": lbm_mean,
        "lbm_std": lbm_std,
        "n_with_data": n_with_data,
        "coverage_pct": coverage_pct,
    }


# ---------------------------------------------------------------------------
# Step 2: annotated ring_agg cluster panel
# ---------------------------------------------------------------------------


def render_annotated_ring_agg(
    ring_agg_path: Path,
    stats_path: Path,
    nearest_idx: np.ndarray,
    inside: np.ndarray,
    extent_xy: tuple[float, float, float, float],
    out_path: Path,
    per_cluster_lbm: dict[int, dict] | None = None,
) -> None:
    """Re-render ring_agg clusters with white circles + cluster IDs at centroids.

    Loads the ring_agg embedding, refits KMeans with the same seed used in W4
    (cluster IDs match the W4 stats sidecar), gathers the cluster RGB raster,
    then overlays per-cluster centroid markers sized by sqrt(fraction).
    """
    print(f"[annotated] loading {ring_agg_path.name}")
    t0 = time.time()
    df = pd.read_parquet(ring_agg_path)
    X = df.to_numpy(dtype=np.float32)
    print(f"  -> {X.shape} in {time.time() - t0:.1f}s")

    print(f"[kmeans] k={K_CLUSTERS} seed={KMEANS_SEED}")
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=K_CLUSTERS, random_state=KMEANS_SEED,
        batch_size=4096, n_init=4,
    )
    labels = km.fit_predict(X)
    print(f"  -> labels in {time.time() - t0:.1f}s")

    tab10 = colormaps["tab10"]
    rgb_per_hex = tab10(labels % 10)[:, :3].astype(np.float32)
    img = gather_rgba(nearest_idx, inside, rgb_per_hex)

    # Load cluster stats for centroid coordinates and fractions.
    stats = json.loads(stats_path.read_text())
    clusters = stats["clusters"]

    fig, ax = plt.subplots(1, 1, figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    fig.patch.set_facecolor(BG)
    _styled_axes(ax)
    ax.imshow(img, extent=extent_xy, origin="lower", interpolation="nearest")

    # Marker sizing: sqrt of fraction so a 25%-cluster doesn't dominate.
    # In data units (metres), use radius_m = sqrt(frac) * scale.
    minx, maxx, miny, maxy = extent_xy
    map_width_m = maxx - minx
    # Anchor radius: for a 0.20 (20%) cluster, want radius ~6 km. So
    # scale = 6000 / sqrt(0.20) ~= 13420.
    radius_scale_m = 13_420.0

    legend_lines: list[str] = []

    for c in clusters:
        if c["centroid_rd_x"] is None:
            continue
        cid = c["cluster_id"]
        frac = c["fraction_of_total"]
        cx = c["centroid_rd_x"]
        cy = c["centroid_rd_y"]
        radius_m = np.sqrt(max(frac, 0.001)) * radius_scale_m

        circle = Circle(
            (cx, cy), radius=radius_m,
            facecolor="white", edgecolor="black",
            linewidth=1.2, alpha=0.92, zorder=5,
        )
        ax.add_patch(circle)
        # Cluster ID label inside the circle in black bold.
        ax.text(
            cx, cy, str(cid),
            ha="center", va="center",
            color="black", fontsize=11, fontweight="bold",
            zorder=6,
        )

        # Build a legend line: ID | %hexes | mean LBM (if available).
        line = f"{cid}: {frac:.1%} of hexes"
        if per_cluster_lbm and cid in per_cluster_lbm:
            lbm_info = per_cluster_lbm[cid]
            if lbm_info["mean"] is not None:
                line += f" | LBM mean {lbm_info['mean']:.3f}"
        legend_lines.append(line)

    title = (
        "Ring-Agg k=10 | tab10 | dim=208 | annotated centroids "
        "(marker area ∝ cluster fraction)"
    )
    ax.set_title(title, color="white", fontsize=12, pad=10)

    # Legend: text block in upper-right corner, white on slate.
    legend_text = "\n".join(legend_lines)
    ax.text(
        0.98, 0.98, legend_text,
        ha="right", va="top",
        color="white", fontsize=8, family="monospace",
        transform=ax.transAxes,
        bbox=dict(facecolor=BG, edgecolor="white", linewidth=0.5, alpha=0.85,
                  boxstyle="round,pad=0.4"),
        zorder=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PANEL_DPI, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> wrote {out_path}")


# ---------------------------------------------------------------------------
# Step 3: per-cluster LBM correlation table
# ---------------------------------------------------------------------------


def compute_lbm_per_cluster(
    embeddings: list[tuple[str, str, str]],
    paths: StudyAreaPaths,
    hex_index: pd.Index,
    lbm_meta: dict,
) -> dict:
    """For each embedding x cluster, compute mean LBM + std + n_hexes + percentile.

    Returns a JSON-serialisable dict:
        { embedding_key: { cluster_id: { mean, std, n_hexes, lbm_pct_rank } } }
    where lbm_pct_rank is the cluster mean's percentile within the LBM
    nationwide distribution (1.0 = highest scoring cluster nationally).
    """
    if not lbm_meta:
        return {}

    score = lbm_meta["score"]
    has_score = lbm_meta["has_score"]
    # Country-wide LBM distribution (only over hexes that have data).
    lbm_country_sorted = np.sort(score[has_score])

    out: dict[str, dict] = {}

    for key, rel, _descriptor in embeddings:
        src = paths.root / rel
        if not src.exists():
            print(f"[per_cluster] skip {key}: missing {src}", file=sys.stderr)
            continue
        print(f"\n[per_cluster] {key}: re-fit KMeans for cluster IDs")
        t0 = time.time()
        df = pd.read_parquet(src)
        # Sanity check the index aligns with hex_index from the W4 panels.
        if not df.index.equals(hex_index):
            print(f"[per_cluster] {key}: index mismatch with W4 hex set, skipping",
                  file=sys.stderr)
            continue

        X = df.to_numpy(dtype=np.float32)
        km = MiniBatchKMeans(
            n_clusters=K_CLUSTERS, random_state=KMEANS_SEED,
            batch_size=4096, n_init=4,
        )
        labels = km.fit_predict(X)
        print(f"  -> labels in {time.time() - t0:.1f}s")

        per_cluster: dict[str, dict] = {}
        for cid in range(K_CLUSTERS):
            cluster_mask = labels == cid
            with_data = cluster_mask & has_score
            n = int(with_data.sum())
            if n == 0:
                per_cluster[str(cid)] = {
                    "mean": None, "std": None,
                    "n_hexes": int(cluster_mask.sum()),
                    "n_hexes_with_lbm": 0,
                    "lbm_pct_rank": None,
                }
                continue
            cluster_scores = score[with_data]
            mean = float(cluster_scores.mean())
            std = float(cluster_scores.std())
            # Percentile of this cluster's mean within the country-wide score
            # distribution: fraction of country hexes with LBM <= cluster_mean.
            pct_rank = float(
                np.searchsorted(lbm_country_sorted, mean, side="right")
                / len(lbm_country_sorted)
            )
            per_cluster[str(cid)] = {
                "mean": mean,
                "std": std,
                "n_hexes": int(cluster_mask.sum()),
                "n_hexes_with_lbm": n,
                "lbm_pct_rank": pct_rank,
            }
        out[key] = per_cluster

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    paths = StudyAreaPaths(STUDY_AREA)
    out_root = paths.root / "stage3_analysis" / "2026-04-24" / "panels"
    out_root.mkdir(parents=True, exist_ok=True)

    embeddings = [
        ("concat",
         "stage2_multimodal/concat/embeddings/netherlands_res9_20mix.parquet",
         "Concat"),
        ("ring_agg",
         "stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet",
         "Ring-Agg"),
        ("unet",
         "stage2_multimodal/unet/embeddings/netherlands_res9_20mix.parquet",
         "U-Net"),
    ]

    # ---------- Build the shared hex_index + Voronoi state ---------------
    # We use ring_agg's index as canonical (W4 confirmed all 3 embeddings
    # have identical hex sets).
    canonical_src = paths.root / embeddings[1][1]
    hex_index = pd.read_parquet(canonical_src, columns=[]).index
    hex_ids = hex_index.to_numpy()
    print(f"[setup] {len(hex_ids):,} hexes (canonical from ring_agg)")

    print(f"[centroids] SpatialDB res={RES} crs=28992")
    t0 = time.time()
    db = SpatialDB.for_study_area(STUDY_AREA)
    cx_m, cy_m = db.centroids(list(hex_ids), resolution=RES, crs=28992)
    cx_m = np.asarray(cx_m, dtype=np.float64)
    cy_m = np.asarray(cy_m, dtype=np.float64)
    print(f"  -> centroids in {time.time() - t0:.1f}s")

    pad_m = 2_000.0
    extent_m = (
        float(cx_m.min() - pad_m), float(cy_m.min() - pad_m),
        float(cx_m.max() + pad_m), float(cy_m.max() + pad_m),
    )

    print(f"[voronoi] PIXEL_M={PIXEL_M:.0f} MAX_DIST_M={MAX_DIST_M:.0f}")
    t0 = time.time()
    nearest_idx, inside, extent_xy = voronoi_indices(
        cx_m, cy_m, extent_m, pixel_m=PIXEL_M, max_dist_m=MAX_DIST_M,
    )
    print(f"  -> Voronoi in {time.time() - t0:.1f}s "
          f"({nearest_idx.shape[1]}x{nearest_idx.shape[0]} px)")

    # ---------- Step 1: LBM target panel ----------------------------------
    lbm_path = (
        paths.root / "target" / "leefbaarometer"
        / "leefbaarometer_h3res9_2022.parquet"
    )
    print(f"\n=== Step 1: LBM target panel ===")
    lbm_ok, lbm_meta = render_lbm_target_panel(
        lbm_path, hex_index, nearest_idx, inside, extent_xy,
        out_root / "target" / "leefbaarometer.png",
    )

    # ---------- Step 3: per-cluster LBM correlation -----------------------
    # Compute this BEFORE step 2 so the annotated panel can include LBM
    # means in its legend.
    print(f"\n=== Step 3: per-cluster LBM correlations ===")
    if lbm_ok:
        per_cluster_lbm_all = compute_lbm_per_cluster(
            embeddings, paths, hex_index, lbm_meta,
        )
        stats_out = out_root / "stats" / "leefbaarometer_per_cluster.json"
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lbm_source": str(lbm_path.relative_to(paths.root)),
            "lbm_year": 2022,
            "embedding_year_label": "20mix",
            "lbm_country_stats": {
                "mean": lbm_meta["lbm_mean"],
                "std": lbm_meta["lbm_std"],
                "min": lbm_meta["lbm_min"],
                "max": lbm_meta["lbm_max"],
                "n_hexes_with_data": int(lbm_meta["n_with_data"]),
                "coverage_pct": lbm_meta["coverage_pct"],
            },
            "per_embedding": per_cluster_lbm_all,
            "_meta": {
                "k": K_CLUSTERS,
                "kmeans_seed": KMEANS_SEED,
                "note": (
                    "Cluster IDs match those in panels/stats/{embedding}.json "
                    "(same KMeans seed)."
                ),
            },
        }
        stats_out.write_text(json.dumps(payload, indent=2))
        print(f"  -> wrote {stats_out}")
    else:
        per_cluster_lbm_all = {}
        print("  -> skipped: LBM not available", file=sys.stderr)

    # ---------- Step 2: annotated ring_agg cluster panel ------------------
    print(f"\n=== Step 2: annotated ring_agg cluster panel ===")
    ring_agg_lbm = (
        {int(k): v for k, v in per_cluster_lbm_all["ring_agg"].items()}
        if "ring_agg" in per_cluster_lbm_all else None
    )
    render_annotated_ring_agg(
        paths.root / embeddings[1][1],
        out_root / "stats" / "ring_agg.json",
        nearest_idx, inside, extent_xy,
        out_root / "ring_agg" / "clusters_tab10_annotated.png",
        per_cluster_lbm=ring_agg_lbm,
    )

    print("\n[done]")


if __name__ == "__main__":
    main()
