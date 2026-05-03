"""
Book v2 — Chapter 9 — Score-aware clustering (W3.A, extended sweep).

Single formulation: joint MiniBatchKMeans on concat(z(emb_208D), lambda * z(lbm_6D))
across lambda in {0, 0.5, 1, 2, 3, 5, 8, 12, 20}. Lambda=0 is the vanilla
embedding-only baseline. W3.A extends W2.C's 3-point sweep to 9 points to
characterise the full cohesion/separation tradeoff curve.

Inputs
------
- Embedding: stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet
  (~397K rows, 208D)
- LBM target: target/leefbaarometer/leefbaarometer_h3res9_2022.parquet
  Columns: lbm, fys, onv, soc, vrz, won  (sub-scores already z-scored;
  `lbm` is z-scored independently here before stacking).
- Inner-join on region_id => fit on the ~130K LBM-covered subset.

Outputs
-------
1. Cluster assignments per lambda (separate dirs, do NOT touch ring_agg_k10):
     stage3_analysis/cluster_results/score_aware_lambda{0,1,3}/assignments.parquet
   Columns: cluster_id (Int64), strength (float). Index: region_id.
2. Results CSV:
     reports/2026-05-03-book/v2/ch9_score_aware/results.csv
   Columns: lambda, within_cluster_lbm_var, silhouette_emb, ari_vs_vanilla,
            spatial_coherence, n_hex_lbm_subset
3. Figures (in same dir):
     - tradeoff.png            — silhouette vs LBM-variance, points labeled by lambda
     - lbm_signature_compare.png — vanilla vs best-lambda 10-cluster x 6-dim LBM means
     - cluster_map_best.png    — spatial map of best-lambda clustering (boundary=None)
4. Append "## Chapter 9 — Score-aware clustering" to
     reports/2026-05-03-book/THE_BOOK.md

Spec rules
----------
- No NL outline anywhere (boundary=None to plot_spatial_map).
- SRAI for H3 neighbours (not h3-py).
- StudyAreaPaths for all paths.
- Random seed = 42 throughout.

Lifetime: temporary (book v2 chapter, expires ~2026-06-03)
Stage: 3 (post-training analysis)

Usage
-----
    python scripts/one_off/book_v2_ch9_score_aware.py
    python scripts/one_off/book_v2_ch9_score_aware.py --study-area netherlands \
        --resolution 9 --year 20mix --k 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# Project root on sys.path so we can import utils
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.paths import StudyAreaPaths
from utils.visualization import (
    plot_spatial_map,
    rasterize_continuous_voronoi,
    rasterize_categorical_voronoi,
)

LBM_COLS = ["lbm", "fys", "onv", "soc", "vrz", "won"]
SEED = 42

logger = logging.getLogger("book_v2_ch9_score_aware")


# --------------------------------------------------------------------------- #
# Load + prep
# --------------------------------------------------------------------------- #


def load_inputs(
    paths: StudyAreaPaths, resolution: int, year: str
) -> tuple[pd.DataFrame, pd.DataFrame, "pd.DataFrame"]:
    """Load embedding, LBM target, and regions GDF."""
    import geopandas as gpd

    emb_path = (
        paths.model_embeddings("ring_agg")
        / f"{paths.study_area}_res{resolution}_{year}.parquet"
    )
    lbm_path = paths.target_file("leefbaarometer", resolution, 2022)
    regions_path = paths.region_file(resolution)

    logger.info("Loading embedding: %s", emb_path)
    emb = pd.read_parquet(emb_path)
    logger.info("  shape=%s, index=%s", emb.shape, emb.index.name)

    logger.info("Loading leefbaarometer: %s", lbm_path)
    lbm = pd.read_parquet(lbm_path)
    logger.info("  shape=%s, cols=%s", lbm.shape, list(lbm.columns))
    missing = [c for c in LBM_COLS if c not in lbm.columns]
    if missing:
        raise ValueError(f"LBM file missing columns: {missing}")
    lbm = lbm[LBM_COLS].copy()

    logger.info("Loading regions: %s", regions_path)
    regions = gpd.read_parquet(regions_path)
    logger.info("  shape=%s, crs=%s", regions.shape, regions.crs)
    return emb, lbm, regions


def prepare_features(
    emb: pd.DataFrame, lbm: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Inner-join on region_id, z-score embedding (208D) and LBM (6D).

    Returns:
        joined: DataFrame on the LBM-covered subset, original (un-scaled)
            columns, indexed by region_id.
        Z_emb: z-scored embedding matrix, shape (n, 208).
        Z_lbm: z-scored LBM matrix, shape (n, 6).
        emb_cols: list of embedding column names (stable order).
    """
    emb_cols = list(emb.columns)
    joined = emb.join(lbm, how="inner")
    if len(joined) == 0:
        raise RuntimeError("Inner join produced empty frame")
    logger.info("Joined frame: %d hexagons (LBM coverage)", len(joined))

    # Z-score embedding (per dim) — protects against any one modality dominating.
    emb_scaler = StandardScaler()
    Z_emb = emb_scaler.fit_transform(joined[emb_cols].values).astype(np.float32)

    # Z-score LBM (per dim). The 5 sub-scores are already roughly z-scored, but
    # `lbm` itself uses different scale; z-score all six together for parity.
    lbm_scaler = StandardScaler()
    Z_lbm = lbm_scaler.fit_transform(joined[LBM_COLS].values).astype(np.float32)

    logger.info(
        "Z_emb shape=%s mean=%.3g std=%.3g",
        Z_emb.shape, float(Z_emb.mean()), float(Z_emb.std()),
    )
    logger.info(
        "Z_lbm shape=%s mean=%.3g std=%.3g",
        Z_lbm.shape, float(Z_lbm.mean()), float(Z_lbm.std()),
    )
    return joined, Z_emb, Z_lbm, emb_cols


# --------------------------------------------------------------------------- #
# Clustering + metrics
# --------------------------------------------------------------------------- #


def fit_joint_kmeans(
    Z_emb: np.ndarray, Z_lbm: np.ndarray, lam: float, k: int
) -> tuple[np.ndarray, MiniBatchKMeans, np.ndarray]:
    """Fit MiniBatchKMeans on concat(Z_emb, lam * Z_lbm).

    Returns:
        labels: (n,) int cluster ids in [0, k).
        model: fitted MiniBatchKMeans.
        X: the stacked feature matrix (kept so we can re-evaluate metrics).
    """
    X = np.concatenate([Z_emb, lam * Z_lbm], axis=1).astype(np.float32)
    mbk = MiniBatchKMeans(
        n_clusters=k,
        random_state=SEED,
        batch_size=4096,
        n_init=10,
        max_iter=300,
        reassignment_ratio=0.01,
    )
    labels = mbk.fit_predict(X)
    return labels, mbk, X


def within_cluster_lbm_variance(Z_lbm: np.ndarray, labels: np.ndarray) -> float:
    """Sum across 6 z-scored LBM dims of cluster-weighted within-cluster variance.

    For each dim d, compute var_d = sum_k (n_k / N) * Var(Z_lbm[labels==k, d]).
    Sum over d. Smaller = clustering separates LBM better.
    """
    total = 0.0
    n = len(labels)
    if n == 0:
        return float("nan")
    for k in np.unique(labels):
        mask = labels == k
        n_k = int(mask.sum())
        if n_k < 2:
            continue
        var_per_dim = Z_lbm[mask].var(axis=0, ddof=0)
        total += float((n_k / n) * var_per_dim.sum())
    return total


def silhouette_emb(
    Z_emb: np.ndarray, labels: np.ndarray, sample: int = 5000
) -> float:
    """Silhouette of `labels` in embedding space (sampled for speed)."""
    n = len(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    rng = np.random.default_rng(SEED)
    if n > sample:
        idx = rng.choice(n, size=sample, replace=False)
        Z_s, lab_s = Z_emb[idx], labels[idx]
    else:
        Z_s, lab_s = Z_emb, labels
    if len(np.unique(lab_s)) < 2:
        return float("nan")
    return float(silhouette_score(Z_s, lab_s, random_state=SEED))


def spatial_coherence(
    region_ids: pd.Index, labels: np.ndarray, regions_gdf
) -> float:
    """Mean per-cluster fraction of res9 H3 neighbours sharing the same cluster.

    Uses srai.neighbourhoods.H3Neighbourhood (NOT h3-py).
    """
    from srai.neighbourhoods import H3Neighbourhood

    # H3Neighbourhood operates on a regions_gdf indexed by region_id.
    # Restrict to the clustered subset for neighbour lookup parity.
    sub = regions_gdf.loc[regions_gdf.index.intersection(region_ids)]
    n = H3Neighbourhood(regions_gdf=sub, include_center=False)
    label_lookup = pd.Series(labels, index=region_ids)

    same_per_cluster: dict[int, list[float]] = {}
    for rid, lab in label_lookup.items():
        nbrs = n.get_neighbours(rid)
        if not nbrs:
            continue
        nbr_labels = label_lookup.reindex(list(nbrs)).dropna()
        if len(nbr_labels) == 0:
            continue
        frac = float((nbr_labels.values == lab).mean())
        same_per_cluster.setdefault(int(lab), []).append(frac)

    if not same_per_cluster:
        return float("nan")
    cluster_means = [float(np.mean(v)) for v in same_per_cluster.values() if v]
    return float(np.mean(cluster_means)) if cluster_means else float("nan")


def cluster_lbm_means(
    Z_lbm: np.ndarray, labels: np.ndarray, k: int
) -> np.ndarray:
    """(k, 6) matrix of mean z-scored LBM vector per cluster (ordered 0..k-1)."""
    out = np.zeros((k, Z_lbm.shape[1]), dtype=np.float64)
    for c in range(k):
        mask = labels == c
        if mask.any():
            out[c] = Z_lbm[mask].mean(axis=0)
        else:
            out[c] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


def save_assignments(
    region_ids: pd.Index, labels: np.ndarray, lam: float, paths: StudyAreaPaths
) -> Path:
    """Save cluster assignments parquet for one lambda variant.

    Path: stage3_analysis/cluster_results/score_aware_lambda{tag}/assignments.parquet
    Tag is the integer-rendered lambda for stable directory names per spec
    ({0,1,3}).
    """
    tag = _lambda_tag(lam)
    out_dir = paths.cluster_results(f"score_aware_lambda{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "assignments.parquet"
    df = pd.DataFrame(
        {"cluster_id": pd.array(labels, dtype="Int64"),
         "lambda_score": float(lam)},
        index=region_ids,
    )
    df.index.name = "region_id"
    df.to_parquet(out_path)
    logger.info("Wrote %d assignments to %s", len(df), out_path)
    return out_path


def _lambda_tag(lam: float) -> str:
    """Map lambda to dir-name tag per W3.A spec.

    Integer lambdas → str(int): 0, 1, 2, 3, 5, 8, 12, 20.
    Non-integer (e.g. 0.5) → '0_5' (decimal point becomes underscore).
    """
    if abs(lam - round(lam)) < 1e-9:
        return str(int(round(lam)))
    return f"{lam:g}".replace(".", "_")


def write_results_csv(
    rows: list[dict], out_path: Path
) -> None:
    df = pd.DataFrame(rows)
    cols = [
        "lambda", "within_cluster_lbm_var", "silhouette_emb",
        "ari_vs_vanilla", "spatial_coherence", "n_hex_lbm_subset",
    ]
    df = df[cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Wrote results CSV: %s", out_path)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_tradeoff(rows: list[dict], out_path: Path) -> None:
    """Tradeoff plot: silhouette vs within-cluster LBM variance.

    Each point colored by λ on a perceptually-uniform gradient (viridis).
    Points connected by a faint line in λ order to show the trajectory.
    Realistic axis bounds (not zoomed to data extents) so the trajectory is
    readable against the natural scale, not chasing pixels. Annotates each
    point with its λ value and includes a colorbar mapping color to λ.
    """
    import matplotlib.colors as mcolors
    from matplotlib import cm

    # Sort by λ so the trajectory line connects points in order.
    rows_sorted = sorted(rows, key=lambda r: r["lambda"])
    xs = np.array([r["within_cluster_lbm_var"] for r in rows_sorted])
    ys = np.array([r["silhouette_emb"] for r in rows_sorted])
    lams = np.array([r["lambda"] for r in rows_sorted])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Gradient color mapped to λ (linear). Use Normalize so 0 → start of cmap,
    # max λ → end of cmap.
    norm = mcolors.Normalize(vmin=float(lams.min()), vmax=float(lams.max()))
    cmap = cm.viridis
    colors = cmap(norm(lams))

    # Trajectory line (faint, behind points).
    ax.plot(xs, ys, color="0.4", linewidth=1.2, alpha=0.55, zorder=2,
            linestyle="--")

    # Points colored by λ.
    sc = ax.scatter(
        xs, ys, c=lams, cmap=cmap, norm=norm,
        s=160, edgecolor="black", linewidth=0.9, zorder=3,
    )

    # Annotate each point with its λ value.
    for x, y, lam in zip(xs, ys, lams):
        ax.annotate(
            f"λ={lam:g}", (x, y), xytext=(9, 7), textcoords="offset points",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="0.7", alpha=0.85, linewidth=0.5),
            zorder=4,
        )

    # Realistic axis bounds — don't tightly zoom to data range.
    # LBM variance: span 0.0 to ~4.5 with padding (max possible ≈ 6 z-dims at
    # variance 1 each = 6.0 if no clustering separation; vanilla ≈ 4.0 covers
    # ~67% of that ceiling). Show 0 to 4.5 to give natural scale.
    # Silhouette: full theoretical range is [-1, 1]; observed values are
    # 0.0–0.08 territory. Show a realistic interpretable window centered on 0:
    # -0.05 to 0.12 so the trajectory is visible against the natural scale.
    x_min = 0.0
    x_max = max(4.5, float(xs.max()) * 1.05)
    y_min = -0.05
    y_max = max(0.12, float(ys.max()) + 0.03)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Reference lines.
    ax.axhline(0.0, color="0.6", linewidth=0.6, linestyle=":", zorder=1)

    ax.set_xlabel(
        "Within-cluster LBM variance\n"
        "(sum over 6 z-dims; smaller = better LBM separation)",
        fontsize=10,
    )
    ax.set_ylabel(
        "Embedding silhouette (sampled n=5K)\n"
        "(larger = better structural cohesion)",
        fontsize=10,
    )
    ax.set_title(
        "Score-aware kmeans tradeoff: structural cohesion vs LBM separation\n"
        f"({len(rows_sorted)}-point λ sweep on {rows_sorted[0]['n_hex_lbm_subset']:,} LBM-covered hexagons)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    # Colorbar mapping color to λ.
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("λ (LBM weight in joint kmeans objective)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def fig_signature_compare(
    means_vanilla: np.ndarray,
    means_best: np.ndarray,
    best_lambda: float,
    out_path: Path,
) -> None:
    """Side-by-side heatmap of (k x 6) cluster-mean LBM signatures."""
    k = means_vanilla.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    vmax = float(np.nanmax(np.abs(np.concatenate([means_vanilla, means_best]))))
    for ax, M, title in zip(
        axes,
        [means_vanilla, means_best],
        [f"Vanilla (λ=0)", f"Best (λ={best_lambda:g})"],
    ):
        im = ax.imshow(
            M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
        )
        ax.set_xticks(range(len(LBM_COLS)))
        ax.set_xticklabels(LBM_COLS, fontsize=10)
        ax.set_yticks(range(k))
        ax.set_yticklabels([f"C{i}" for i in range(k)], fontsize=9)
        ax.set_title(title, fontsize=11)
        for i in range(k):
            for j in range(len(LBM_COLS)):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > vmax * 0.55 else "black",
                    )
    fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02,
                 label="Cluster-mean z-LBM")
    fig.suptitle(
        "Cluster x dim LBM signature: vanilla vs best-λ score-aware",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def fig_cluster_map(
    regions_gdf,
    region_ids: pd.Index,
    labels_full: np.ndarray,
    lam: float,
    k: int,
    out_path: Path,
) -> None:
    """Spatial cluster map for the full set of hexagons (boundary=None per spec)."""
    sub = regions_gdf.loc[regions_gdf.index.intersection(region_ids)].copy()
    label_series = pd.Series(labels_full, index=region_ids).astype(np.int64)
    sub["cluster_id"] = label_series.reindex(sub.index).astype(np.int64)
    # Rasterize via GDF -> metric centroids (utility handles CRS reproj).
    sub_metric = sub.to_crs(28992)
    cents = sub_metric.geometry.centroid
    cx = np.asarray(cents.x.to_numpy(), dtype=np.float64)
    cy = np.asarray(cents.y.to_numpy(), dtype=np.float64)
    minx, miny, maxx, maxy = sub_metric.total_bounds
    img, extent_xy = rasterize_categorical_voronoi(
        cx, cy, sub["cluster_id"].to_numpy(),
        (float(minx), float(miny), float(maxx), float(maxy)),
        n_clusters=k, cmap="tab20",
        pixel_m=250.0, max_dist_m=300.0,
    )
    fig, ax = plt.subplots(figsize=(8, 9.5))
    plot_spatial_map(
        ax, img, extent_xy, boundary_gdf=None,
        title=(
            f"Score-aware clustering (λ={lam:g}, k={k}) — "
            f"{len(sub):,} hexagons"
        ),
        show_rd_grid=False,
        disable_rd_grid=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# --------------------------------------------------------------------------- #
# THE_BOOK chapter prose
# --------------------------------------------------------------------------- #


def append_chapter9(
    book_path: Path,
    rows: list[dict],
    best_row: dict,
    runtime_s: float,
) -> None:
    """Append Chapter 9 markdown to THE_BOOK.md (idempotent: replaces if present)."""
    book_path.parent.mkdir(parents=True, exist_ok=True)
    text = book_path.read_text(encoding="utf-8") if book_path.exists() else ""

    chapter_marker = "## Chapter 9 — Score-aware clustering"
    if chapter_marker in text:
        # Trim at the existing marker so we replace cleanly.
        text = text.split(chapter_marker, 1)[0].rstrip() + "\n\n"

    table_rows = [
        "| λ | within-cluster LBM var | silhouette (emb) | ARI vs vanilla | spatial coherence |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        table_rows.append(
            "| {lam:g} | {wcv:.4f} | {sil:.4f} | {ari:.4f} | {sc:.4f} |".format(
                lam=r["lambda"],
                wcv=r["within_cluster_lbm_var"],
                sil=r["silhouette_emb"],
                ari=r["ari_vs_vanilla"],
                sc=r["spatial_coherence"],
            )
        )
    table = "\n".join(table_rows)

    d_sil = best_row["silhouette_emb"] - rows[0]["silhouette_emb"]
    d_wcv = best_row["within_cluster_lbm_var"] - rows[0]["within_cluster_lbm_var"]
    d_sc = best_row["spatial_coherence"] - rows[0]["spatial_coherence"]
    pct_wcv = 100.0 * d_wcv / rows[0]["within_cluster_lbm_var"]
    chapter = f"""{chapter_marker}

Joint MiniBatchKMeans (k=10, n_init=10, seed=42) on the per-dimension z-scored ring-aggregation embedding (208D) concatenated with λ × z(LBM 6D), where LBM = (lbm, fys, onv, soc, vrz, won). At λ=0 the partition ignores LBM entirely (vanilla baseline); at λ>0 LBM enters the kmeans objective with weight controlled by λ. Fit on the {best_row['n_hex_lbm_subset']:,}-hexagon subset where LBM is observed.

{table}

Best variant: **λ={best_row['lambda']:g}** — minimizes within-cluster LBM variance ({best_row['within_cluster_lbm_var']:.4f}, Δ={d_wcv:+.4f} = {pct_wcv:+.1f}% vs vanilla) at the cost of embedding silhouette (Δsil = {d_sil:+.4f}) and spatial coherence (Δsc = {d_sc:+.4f}, vanilla 0.5119 → 0.4872). ARI vs vanilla = {best_row['ari_vs_vanilla']:.4f} confirms the partition rotates rather than collapses. Rank by within-cluster LBM variance: λ=3 < λ=1 < λ=0; rank by silhouette: λ=1 > λ=0 > λ=3. Total runtime: {runtime_s:.1f}s.

![Score-aware tradeoff](v2/ch9_score_aware/tradeoff.png)
*Silhouette vs within-cluster LBM variance, points labeled by λ. Moving from λ=0 to λ={best_row['lambda']:g} traces the cohesion/separation tradeoff frontier.*

![LBM signature comparison](v2/ch9_score_aware/lbm_signature_compare.png)
*Per-cluster mean z-LBM (10 clusters × 6 dims), vanilla vs best-λ. Stronger row contrast at λ={best_row['lambda']:g} indicates clusters now align with LBM directions, not just embedding directions.*

![Spatial cluster map (best λ)](v2/ch9_score_aware/cluster_map_best.png)
*The 10-cluster score-aware partition projected over the LBM-covered subset.*

---
"""
    new_text = text.rstrip() + "\n\n" + chapter
    book_path.write_text(new_text, encoding="utf-8")
    logger.info("Appended Chapter 9 to %s", book_path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--study-area", default="netherlands")
    parser.add_argument("--resolution", type=int, default=9)
    parser.add_argument("--year", default="20mix")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--lambdas", type=float, nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
        help=(
            "Lambda sweep values (default W3.A: 0 0.5 1 2 3 5 8 12 20)."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    warnings.filterwarnings("ignore", category=FutureWarning)

    t0 = time.time()
    paths = StudyAreaPaths(args.study_area)

    # 1. Load + prep
    emb, lbm, regions = load_inputs(paths, args.resolution, args.year)
    joined, Z_emb, Z_lbm, _ = prepare_features(emb, lbm)
    region_ids = joined.index

    # 2. For full-set spatial map: fit nearest centroid in embedding space later.
    full_emb_scaler = StandardScaler()
    full_emb_scaler.fit(emb.values)
    Z_emb_full = full_emb_scaler.transform(emb.values).astype(np.float32)

    # 3. Sweep lambdas. Vanilla = lam=0.
    rows: list[dict] = []
    labels_per_lam: dict[float, np.ndarray] = {}
    centroids_emb_per_lam: dict[float, np.ndarray] = {}
    means_per_lam: dict[float, np.ndarray] = {}

    vanilla_labels: np.ndarray | None = None

    out_book_dir = (
        _project_root / "reports" / "2026-05-03-book" / "v2" / "ch9_score_aware"
    )
    out_book_dir.mkdir(parents=True, exist_ok=True)

    for lam in args.lambdas:
        t_lam = time.time()
        logger.info("=== lambda = %.3f ===", lam)
        labels, model, _X = fit_joint_kmeans(Z_emb, Z_lbm, lam, args.k)
        labels_per_lam[lam] = labels

        # Centroids in embedding space only (first 208 dims of the joint model).
        centroids_emb_per_lam[lam] = model.cluster_centers_[:, : Z_emb.shape[1]]

        if abs(lam) < 1e-12:
            vanilla_labels = labels.copy()

        wcv = within_cluster_lbm_variance(Z_lbm, labels)
        sil = silhouette_emb(Z_emb, labels, sample=5000)
        ari = (
            float(adjusted_rand_score(vanilla_labels, labels))
            if vanilla_labels is not None
            else float("nan")
        )
        sc = spatial_coherence(region_ids, labels, regions)
        rows.append({
            "lambda": float(lam),
            "within_cluster_lbm_var": float(wcv),
            "silhouette_emb": float(sil),
            "ari_vs_vanilla": float(ari),
            "spatial_coherence": float(sc),
            "n_hex_lbm_subset": int(len(region_ids)),
        })
        means_per_lam[lam] = cluster_lbm_means(Z_lbm, labels, args.k)
        save_assignments(region_ids, labels, lam, paths)
        logger.info(
            "  wcv=%.4f sil=%.4f ari=%.4f sc=%.4f  (%.1fs)",
            wcv, sil, ari, sc, time.time() - t_lam,
        )

    # 4. Pick best by W3.A composite criterion:
    #    score = (var_reduction_ratio vs vanilla) * max(silhouette, 0)
    #    This rewards LBM separation but penalises silhouette collapse.
    #    Vanilla (λ=0) always scores 0 (no var reduction). Tie-break: lowest
    #    LBM variance among silhouette > 0 candidates.
    vanilla_var = next(r["within_cluster_lbm_var"] for r in rows
                       if abs(r["lambda"]) < 1e-12)

    def _composite(r: dict) -> float:
        var_red = max(0.0, (vanilla_var - r["within_cluster_lbm_var"]) / vanilla_var)
        sil_pos = max(0.0, r["silhouette_emb"])
        return var_red * sil_pos

    composite_scores = {r["lambda"]: _composite(r) for r in rows}
    logger.info("Composite scores per λ: %s",
                {f"{k:g}": f"{v:.5f}" for k, v in composite_scores.items()})

    # Best candidates with silhouette > 0; if none, fall back to lowest LBM var.
    pos_sil = [r for r in rows if r["silhouette_emb"] > 0]
    if pos_sil:
        best_row = max(pos_sil, key=_composite)
    else:
        best_row = min(rows, key=lambda r: r["within_cluster_lbm_var"])
    best_lam = best_row["lambda"]
    logger.info("Best λ (composite, silhouette>0): %g", best_lam)

    # 5. Write CSV
    write_results_csv(rows, out_book_dir / "results.csv")

    # 6. Figures
    fig_tradeoff(rows, out_book_dir / "tradeoff.png")
    fig_signature_compare(
        means_per_lam[0.0], means_per_lam[best_lam],
        best_lam, out_book_dir / "lbm_signature_compare.png",
    )

    # 7. Cluster map: predict labels for the FULL set via nearest centroid in
    #    the embedding subspace of the best-lambda model.
    cents_best = centroids_emb_per_lam[best_lam]  # (k, 208) in z-space
    # Compute nearest-centroid labels for full embedding set.
    # Use chunked computation to keep memory reasonable.
    full_labels = _nearest_centroid_predict(Z_emb_full, cents_best)
    full_region_ids = emb.index
    fig_cluster_map(
        regions, full_region_ids, full_labels, best_lam, args.k,
        out_book_dir / "cluster_map_best.png",
    )

    # 8. W3.A: do NOT touch THE_BOOK.md — W4 owns all chapter prose.
    #    Artifact-only run: results.csv + 3 PNGs + 9 cluster assignment parquets.
    runtime_s = time.time() - t0

    logger.info("=== DONE in %.1fs ===", runtime_s)
    logger.info("Best lambda = %g", best_lam)
    for r in rows:
        logger.info(
            "  λ=%g  wcv=%.4f  sil=%.4f  ari=%.4f  sc=%.4f",
            r["lambda"], r["within_cluster_lbm_var"], r["silhouette_emb"],
            r["ari_vs_vanilla"], r["spatial_coherence"],
        )
    return 0


def _nearest_centroid_predict(
    Z: np.ndarray, centroids: np.ndarray, chunk: int = 50000
) -> np.ndarray:
    """Assign each row of Z to the nearest centroid (Euclidean), chunked."""
    n = Z.shape[0]
    out = np.empty(n, dtype=np.int64)
    # Pre-compute c·c for centroids
    cc = (centroids ** 2).sum(axis=1)  # (k,)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = Z[s:e]
        # ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2; ||x||^2 doesn't change argmin.
        d = -2.0 * block @ centroids.T + cc[None, :]
        out[s:e] = d.argmin(axis=1)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
