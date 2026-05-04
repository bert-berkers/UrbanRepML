"""
Book v2 — Chapter 9 — Score-aware clustering (W2.C proof-of-concept).

Two formulations of clustering that incorporate the leefbaarometer score
into the partitioning objective itself:

  Formulation 1 (joint): MiniBatchKMeans on concat(embedding_z208,
      lambda * lbm_z6) for lambda in {0.5, 1.0, 2.0, 5.0}.

  Formulation 2 (re-assignment): vanilla KMeans on embedding_z208,
      then iterative reassignment minimizing
          alpha * ||emb_i - mu_emb_k||^2 + (1-alpha) * ||lbm_i - mu_lbm_k||^2
      for alpha in {0.95, 0.85, 0.7, 0.5}.

Vanilla baseline = MiniBatchKMeans on embedding_z208 alone.

For each variant we report:
  - within-cluster LBM variance (sum across 6 dims, z-scored)
  - embedding silhouette in original 208D space
  - ARI vs vanilla baseline
  - spatial coherence (mean per-cluster fraction of same-cluster H3 neighbours)
  - per-cluster mean LBM vector

Vanilla baseline is fitted on the LBM-covered ~130K subset for fair
comparison; predictions are extended to the full ~397K hexagon set by
nearest centroid in embedding space (for the spatial figure only).

Lifetime: temporary (book v2 chapter)
Stage: 3 (post-training analysis)

Usage:
    python scripts/one_off/book_v2_ch9_score_aware_clustering.py \
        --study-area netherlands --resolution 9 --year 20mix --k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# Project root on sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.paths import StudyAreaPaths
from utils.spatial_db import SpatialDB
from utils.visualization import (
    plot_spatial_map,
    rasterize_categorical_voronoi,
    voronoi_params_for_resolution,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Reproducibility
SEED = 42
DPI = 200

LBM_COLS = ["lbm", "fys", "onv", "soc", "vrz", "won"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(paths: StudyAreaPaths, resolution: int, year: str | int) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load embedding (208D) and LBM target, return (full_emb, joined_emb, joined_lbm).

    Returns:
        full_emb:  ring-agg embeddings indexed by region_id (~397K rows).
        joined_emb: subset where LBM is available (~130K rows).
        joined_lbm: leefbaarometer 6D vector for the joined subset.
    """
    emb_path = (
        paths.root
        / "stage2_multimodal" / "ring_agg" / "embeddings"
        / f"{paths.study_area}_res{resolution}_{year}.parquet"
    )
    full_emb = pd.read_parquet(emb_path)
    if full_emb.index.name != "region_id" and "region_id" in full_emb.columns:
        full_emb = full_emb.set_index("region_id")
    logger.info("Embedding: %s rows x %sD", *full_emb.shape)

    lbm_path = paths.target_file("leefbaarometer", resolution, 2022)
    lbm = pd.read_parquet(lbm_path)
    if lbm.index.name != "region_id" and "region_id" in lbm.columns:
        lbm = lbm.set_index("region_id")
    lbm = lbm[LBM_COLS]
    logger.info("LBM target: %s rows", len(lbm))

    # Inner join on region_id
    common = full_emb.index.intersection(lbm.index)
    joined_emb = full_emb.loc[common]
    joined_lbm = lbm.loc[common]
    logger.info("Joined: %s rows (LBM-covered subset)", len(common))

    return full_emb, joined_emb, joined_lbm


# ---------------------------------------------------------------------------
# Variant runners (each returns dict with metrics + labels)
# ---------------------------------------------------------------------------


def fit_vanilla(emb_z: np.ndarray, k: int) -> np.ndarray:
    """MiniBatchKMeans on z-scored embedding alone."""
    km = MiniBatchKMeans(
        n_clusters=k, random_state=SEED, batch_size=4096,
        n_init=10, max_iter=300,
    )
    return km.fit_predict(emb_z)


def fit_joint(emb_z: np.ndarray, lbm_z: np.ndarray, lam: float, k: int) -> np.ndarray:
    """Joint kmeans on concat(emb_z, lam * lbm_z)."""
    features = np.hstack([emb_z, lam * lbm_z]).astype(np.float32)
    km = MiniBatchKMeans(
        n_clusters=k, random_state=SEED, batch_size=4096,
        n_init=10, max_iter=300,
    )
    return km.fit_predict(features)


def fit_reassign(
    emb_z: np.ndarray, lbm_z: np.ndarray, alpha: float, k: int,
    max_iter: int = 20,
) -> np.ndarray:
    """Score-variance regularised k-means (Formulation 2).

    Initialise from vanilla labels, then iterate:
      labels[i] = argmin_k  alpha * ||emb_i - mu_emb_k||^2
                          + (1-alpha) * ||lbm_i - mu_lbm_k||^2
    where centroids are recomputed in both spaces each step.

    Stops when assignments stop changing or max_iter is hit.
    """
    n = emb_z.shape[0]
    rng = np.random.RandomState(SEED)

    # Initialise from vanilla
    labels = fit_vanilla(emb_z, k)

    for it in range(max_iter):
        # Recompute centroids in both spaces
        mu_emb = np.zeros((k, emb_z.shape[1]), dtype=np.float32)
        mu_lbm = np.zeros((k, lbm_z.shape[1]), dtype=np.float32)
        sizes = np.zeros(k, dtype=np.int64)
        for j in range(k):
            mask = labels == j
            sz = int(mask.sum())
            sizes[j] = sz
            if sz == 0:
                # Re-seed empty cluster from a random point
                idx = rng.randint(0, n)
                mu_emb[j] = emb_z[idx]
                mu_lbm[j] = lbm_z[idx]
                continue
            mu_emb[j] = emb_z[mask].mean(axis=0)
            mu_lbm[j] = lbm_z[mask].mean(axis=0)

        # Compute distances in chunks (avoid (n, k, d) tensors)
        # Use the identity ||x - mu||^2 = ||x||^2 - 2 x.mu + ||mu||^2
        # Constant ||x||^2 is irrelevant for argmin so drop it.
        emb_dist = -2.0 * (emb_z @ mu_emb.T) + (mu_emb ** 2).sum(axis=1)[None, :]
        lbm_dist = -2.0 * (lbm_z @ mu_lbm.T) + (mu_lbm ** 2).sum(axis=1)[None, :]
        total = alpha * emb_dist + (1.0 - alpha) * lbm_dist
        new_labels = total.argmin(axis=1)

        n_changed = int((new_labels != labels).sum())
        labels = new_labels
        if n_changed == 0:
            logger.info("    reassign(alpha=%.2f) converged at iter %d", alpha, it + 1)
            break
    else:
        logger.info("    reassign(alpha=%.2f) hit max_iter=%d, last changed=%d",
                    alpha, max_iter, n_changed)
    return labels


def run_variant(
    name: str, kind: str, param: float | None,
    emb_z: np.ndarray, lbm_z: np.ndarray, k: int,
) -> tuple[str, np.ndarray, float]:
    """Fit one variant. Returns (name, labels, runtime_s)."""
    t0 = time.time()
    if kind == "vanilla":
        labels = fit_vanilla(emb_z, k)
    elif kind == "joint":
        labels = fit_joint(emb_z, lbm_z, lam=param, k=k)
    elif kind == "reassign":
        labels = fit_reassign(emb_z, lbm_z, alpha=param, k=k)
    else:
        raise ValueError(kind)
    return name, labels, time.time() - t0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def within_cluster_lbm_variance(labels: np.ndarray, lbm_z: np.ndarray) -> float:
    """Sum across 6 LBM dims of weighted within-cluster variance.

    Weighted by cluster size (i.e. equivalent to the WCSS / n form in LBM space).
    Lower = clusters more homogeneous in score.
    """
    total = 0.0
    n = lbm_z.shape[0]
    for j in np.unique(labels):
        mask = labels == j
        if mask.sum() < 2:
            continue
        v = lbm_z[mask].var(axis=0).sum()  # sum across 6 dims
        total += v * mask.sum() / n
    return float(total)


def embedding_silhouette(labels: np.ndarray, emb: np.ndarray, sample_n: int = 20_000) -> float:
    """Silhouette in original 208D embedding space (subsampled for speed)."""
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return float("nan")
    if len(emb) > sample_n:
        idx = np.random.RandomState(SEED).choice(len(emb), sample_n, replace=False)
        return float(silhouette_score(emb[idx], labels[idx]))
    return float(silhouette_score(emb, labels))


def spatial_coherence(
    labels: np.ndarray, region_ids: pd.Index, sample_n: int = 15_000,
) -> float:
    """Mean per-cluster fraction of cell neighbours that share the cluster.

    Uses SRAI's H3Neighbourhood. Subsamples cells for tractability.
    """
    from srai.neighbourhoods import H3Neighbourhood

    n = len(region_ids)
    if n > sample_n:
        idx = np.random.RandomState(SEED).choice(n, sample_n, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)

    cells = region_ids.to_numpy()
    cell_set = set(cells.tolist())
    label_map = dict(zip(cells, labels))

    nb = H3Neighbourhood()
    same = 0
    total = 0
    for i in idx:
        cell = cells[i]
        my_label = label_map[cell]
        try:
            neighs = nb.get_neighbours(cell)
        except Exception:
            continue
        for nbr in neighs:
            if nbr in cell_set:
                total += 1
                if label_map[nbr] == my_label:
                    same += 1
    if total == 0:
        return float("nan")
    return same / total


# ---------------------------------------------------------------------------
# Cluster assignment writer (parquet)
# ---------------------------------------------------------------------------


def write_assignments(
    paths: StudyAreaPaths, variant_id: str, labels: np.ndarray,
    region_ids: pd.Index, k: int,
) -> Path:
    out_dir = paths.cluster_results_root() / "score_aware" / variant_id
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "approach": f"score_aware__{variant_id}",
        "k": int(k),
        "region_id": region_ids.astype(str).to_numpy(),
        "cluster_label": labels.astype(np.int32),
    })
    out = out_dir / "assignments.parquet"
    df.to_parquet(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_tradeoff(results: pd.DataFrame, out_path: Path) -> None:
    """X = within-cluster LBM variance; Y = embedding silhouette.

    Two series (lambda joint, alpha reassign) + vanilla baseline marker.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=DPI)
    fig.set_facecolor("white")

    base = results[results["family"] == "vanilla"].iloc[0]
    ax.scatter(
        base["within_cluster_lbm_var"], base["silhouette_emb"],
        s=200, c="black", marker="X", zorder=5, label="Vanilla baseline (lambda=0)",
    )

    for fam, color, marker in [("joint", "#1f77b4", "o"), ("reassign", "#d62728", "s")]:
        sub = results[results["family"] == fam].sort_values("strength")
        if sub.empty:
            continue
        ax.plot(
            sub["within_cluster_lbm_var"], sub["silhouette_emb"],
            color=color, marker=marker, ms=8, lw=1.5, alpha=0.8,
            label=f"{fam} (param sweep)",
        )
        for _, row in sub.iterrows():
            tag = (
                f"λ={row['strength']:.1f}" if fam == "joint"
                else f"α={row['strength']:.2f}"
            )
            ax.annotate(
                tag, (row["within_cluster_lbm_var"], row["silhouette_emb"]),
                textcoords="offset points", xytext=(7, 4), fontsize=8,
                color=color,
            )

    ax.set_xlabel("Within-cluster LBM variance (sum across 6 dims, z-scored)")
    ax.set_ylabel("Embedding silhouette (208D)")
    ax.set_title(
        "Score-aware clustering: LBM homogeneity vs embedding coherence\n"
        "(left = better score homogeneity; up = better embedding cohesion)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_signature_comparison(
    vanilla_sig: pd.DataFrame, best_sig: pd.DataFrame,
    best_name: str, out_path: Path,
) -> None:
    """Side-by-side heatmap of per-cluster mean LBM vectors."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=DPI, sharey=True)
    fig.set_facecolor("white")

    vmin = min(vanilla_sig.values.min(), best_sig.values.min())
    vmax = max(vanilla_sig.values.max(), best_sig.values.max())

    for ax, sig, title in [
        (axes[0], vanilla_sig, "Vanilla baseline"),
        (axes[1], best_sig, f"Best variant: {best_name}"),
    ]:
        im = ax.imshow(sig.values, aspect="auto", cmap="RdBu_r",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(sig.columns)))
        ax.set_xticklabels(sig.columns, rotation=0, fontsize=10)
        ax.set_yticks(range(len(sig.index)))
        ax.set_yticklabels([f"C{i}" for i in sig.index], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("LBM dimension")
        ax.set_ylabel("Cluster (sorted by lbm)")
        # Annotate
        for i in range(sig.shape[0]):
            for j in range(sig.shape[1]):
                ax.text(j, i, f"{sig.values[i, j]:+.2f}",
                        ha="center", va="center", fontsize=7,
                        color="black")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Mean (z-scored)", fontsize=10)
    fig.suptitle("Per-cluster LBM signatures: vanilla vs best score-aware variant",
                 fontsize=12, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cluster_map(
    paths: StudyAreaPaths, region_ids: pd.Index, labels: np.ndarray,
    k: int, resolution: int, title: str, out_path: Path,
) -> None:
    """Spatial cluster map. NO NL outline (boundary=None)."""
    db = SpatialDB.for_study_area(paths.study_area)
    cx, cy = db.centroids(region_ids, resolution=resolution, crs=28992)
    minx = float(cx.min()); maxx = float(cx.max())
    miny = float(cy.min()); maxy = float(cy.max())
    pad = (maxx - minx) * 0.03
    extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    pixel_m, max_dist_m = voronoi_params_for_resolution(resolution)
    image, _ = rasterize_categorical_voronoi(
        cx, cy, labels, extent,
        n_clusters=k, cmap="tab10",
        pixel_m=pixel_m, max_dist_m=max_dist_m,
    )

    fig, ax = plt.subplots(figsize=(10, 12), dpi=DPI)
    fig.set_facecolor("white")
    plot_spatial_map(ax, image, extent, boundary_gdf=None)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lbm_violins(
    labels: np.ndarray, lbm_z: pd.DataFrame, k: int, out_path: Path,
) -> None:
    """Per-LBM-dim violin grid for the best variant.

    6 panels (one per LBM dim), each with k violins (one per cluster).
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=DPI)
    fig.set_facecolor("white")
    axes = axes.flatten()
    palette = plt.get_cmap("tab10")
    colors = [palette(i / max(k - 1, 1)) for i in range(k)]

    for ax, dim in zip(axes, LBM_COLS):
        data = [lbm_z.loc[labels == j, dim].values for j in range(k)]
        parts = ax.violinplot(
            data, showmeans=True, showmedians=False, widths=0.85,
        )
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c)
            body.set_alpha(0.7)
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
        ax.set_xticks(range(1, k + 1))
        ax.set_xticklabels([f"C{i}" for i in range(k)], fontsize=8)
        ax.set_title(f"{dim} (z-scored)", fontsize=10, fontweight="bold")
        ax.axhline(0.0, color="grey", lw=0.5, ls="--", alpha=0.5)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Per-cluster LBM distributions (best score-aware variant)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reorder_by_lbm(labels: np.ndarray, lbm_composite: np.ndarray) -> np.ndarray:
    """Reorder cluster IDs so 0 = lowest mean lbm, k-1 = highest."""
    df = pd.DataFrame({"label": labels, "lbm": lbm_composite})
    rank = df.groupby("label")["lbm"].mean().sort_values()
    rank_map = {old: new for new, old in enumerate(rank.index)}
    return np.array([rank_map[v] for v in labels])


def cluster_signature(labels: np.ndarray, lbm_z: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster mean LBM vector (k x 6), sorted by index."""
    df = lbm_z.copy()
    df["__c"] = labels
    sig = df.groupby("__c").mean()
    sig.index.name = "cluster"
    return sig


def predict_full(emb_full: np.ndarray, emb_train: np.ndarray, labels_train: np.ndarray,
                 k: int) -> np.ndarray:
    """Assign full-set hexes to nearest centroid in embedding space."""
    centroids = np.zeros((k, emb_full.shape[1]), dtype=np.float32)
    for j in range(k):
        m = labels_train == j
        if m.sum() == 0:
            continue
        centroids[j] = emb_train[m].mean(axis=0)
    # Chunked NN
    BATCH = 50_000
    out = np.empty(len(emb_full), dtype=np.int32)
    for s in range(0, len(emb_full), BATCH):
        e = min(s + BATCH, len(emb_full))
        d = -2.0 * (emb_full[s:e] @ centroids.T) + (centroids ** 2).sum(axis=1)[None, :]
        out[s:e] = d.argmin(axis=1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-area", default="netherlands")
    ap.add_argument("--resolution", type=int, default=9)
    ap.add_argument("--year", default="20mix")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0],
    )
    ap.add_argument(
        "--alphas", type=float, nargs="+", default=[0.95, 0.85, 0.7, 0.5],
    )
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = StudyAreaPaths(args.study_area)
    out_root = Path(args.output_dir) if args.output_dir else (
        project_root / "reports" / "2026-05-03-book" / "v2" / "ch9_score_aware"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_root)

    # -----------------------------------------------------------------
    # 1. Load + prep
    # -----------------------------------------------------------------
    full_emb, joined_emb, joined_lbm = load_data(paths, args.resolution, args.year)

    # z-score embedding (independently per column) — ring-agg already
    # normalised but cheap to redo
    emb_scaler = StandardScaler()
    emb_z_train = emb_scaler.fit_transform(joined_emb.values).astype(np.float32)
    emb_z_full = emb_scaler.transform(full_emb.values).astype(np.float32)

    # z-score LBM independently (sub-scores already z-scored upstream;
    # we still standardise to ensure unit variance per column on this subset)
    lbm_scaler = StandardScaler()
    lbm_z_arr = lbm_scaler.fit_transform(joined_lbm.values).astype(np.float32)
    lbm_z_df = pd.DataFrame(lbm_z_arr, index=joined_emb.index, columns=LBM_COLS)

    logger.info("emb_z_train shape: %s, lbm_z shape: %s",
                emb_z_train.shape, lbm_z_arr.shape)

    # -----------------------------------------------------------------
    # 2. Run all variants in parallel
    # -----------------------------------------------------------------
    jobs: list[tuple[str, str, float | None]] = [("vanilla", "vanilla", None)]
    for lam in args.lambdas:
        jobs.append((f"joint_lam{lam:g}", "joint", lam))
    for a in args.alphas:
        jobs.append((f"reassign_a{a:g}", "reassign", a))

    logger.info("Running %d variants in parallel...", len(jobs))
    t0 = time.time()
    raw = Parallel(n_jobs=-1, backend="loky", verbose=5)(
        delayed(run_variant)(name, kind, p, emb_z_train, lbm_z_arr, args.k)
        for name, kind, p in jobs
    )
    logger.info("All variants done in %.1fs", time.time() - t0)

    # raw is list of (name, labels, runtime). Reorder by lbm composite.
    lbm_composite = joined_lbm["lbm"].values
    fitted = {}
    for name, labels, rt in raw:
        labels_r = reorder_by_lbm(labels, lbm_composite)
        fitted[name] = {"labels": labels_r, "runtime_s": rt}

    # -----------------------------------------------------------------
    # 3. Compute metrics for each variant
    # -----------------------------------------------------------------
    vanilla_labels = fitted["vanilla"]["labels"]
    rows = []
    for name, kind, p in jobs:
        labels = fitted[name]["labels"]
        wcv = within_cluster_lbm_variance(labels, lbm_z_arr)
        sil = embedding_silhouette(labels, joined_emb.values.astype(np.float32))
        ari = (
            1.0 if name == "vanilla"
            else float(adjusted_rand_score(vanilla_labels, labels))
        )
        coh = spatial_coherence(labels, joined_emb.index)
        rows.append({
            "variant": name,
            "family": kind,
            "strength": float(p) if p is not None else 0.0,
            "within_cluster_lbm_var": wcv,
            "silhouette_emb": sil,
            "ari_vs_vanilla": ari,
            "spatial_coherence": coh,
            "runtime_s": fitted[name]["runtime_s"],
            "n_hex_lbm_subset": int(len(joined_emb)),
        })
        logger.info("  %-18s wcv=%.4f sil=%.4f ari=%.3f coh=%.3f rt=%.1fs",
                    name, wcv, sil, ari, coh, fitted[name]["runtime_s"])

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["family", "strength"], kind="stable",
    ).reset_index(drop=True)
    csv_path = out_root / "results.csv"
    results.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    # -----------------------------------------------------------------
    # 4. Persist cluster assignments
    # -----------------------------------------------------------------
    for name, _kind, _p in jobs:
        write_assignments(
            paths, name, fitted[name]["labels"],
            joined_emb.index, args.k,
        )
    logger.info("Wrote assignments to %s", paths.cluster_results_root() / "score_aware")

    # -----------------------------------------------------------------
    # 5. Pick best variant
    # -----------------------------------------------------------------
    # Best = lowest WCV among non-vanilla, with sil >= 0.5 * vanilla sil.
    vanilla_row = results[results["family"] == "vanilla"].iloc[0]
    sil_floor = 0.5 * vanilla_row["silhouette_emb"]
    candidates = results[
        (results["family"] != "vanilla")
        & (results["silhouette_emb"] >= sil_floor)
    ]
    if len(candidates) == 0:
        candidates = results[results["family"] != "vanilla"]
    best = candidates.sort_values("within_cluster_lbm_var").iloc[0]
    best_name = str(best["variant"])
    logger.info("Best variant: %s (wcv=%.4f sil=%.4f ari=%.3f)",
                best_name, best["within_cluster_lbm_var"],
                best["silhouette_emb"], best["ari_vs_vanilla"])

    # -----------------------------------------------------------------
    # 6. Figures
    # -----------------------------------------------------------------
    # Figure 1: tradeoff
    plot_tradeoff(results, out_root / "fig1_tradeoff.png")
    logger.info("Wrote fig1_tradeoff.png")

    # Figure 2: signature comparison
    vanilla_sig = cluster_signature(vanilla_labels, lbm_z_df)
    best_sig = cluster_signature(fitted[best_name]["labels"], lbm_z_df)
    plot_signature_comparison(
        vanilla_sig, best_sig, best_name,
        out_root / "fig2_signature_comparison.png",
    )
    logger.info("Wrote fig2_signature_comparison.png")

    # Figure 3: cluster map of best variant on FULL set
    full_labels = predict_full(
        emb_z_full, emb_z_train, fitted[best_name]["labels"], k=args.k,
    )
    plot_cluster_map(
        paths, full_emb.index, full_labels, k=args.k,
        resolution=args.resolution,
        title=(
            f"Score-aware clusters — best variant: {best_name}\n"
            f"k={args.k}, train-on-LBM-subset (n={len(joined_emb):,}), "
            f"extended to full set (n={len(full_emb):,}) by nearest centroid"
        ),
        out_path=out_root / "fig3_cluster_map_best.png",
    )
    logger.info("Wrote fig3_cluster_map_best.png")

    # Figure 4: violin grid for best variant
    plot_lbm_violins(
        fitted[best_name]["labels"], lbm_z_df, k=args.k,
        out_path=out_root / "fig4_lbm_violins_best.png",
    )
    logger.info("Wrote fig4_lbm_violins_best.png")

    # Also write a vanilla cluster map for visual reference
    full_labels_van = predict_full(
        emb_z_full, emb_z_train, vanilla_labels, k=args.k,
    )
    plot_cluster_map(
        paths, full_emb.index, full_labels_van, k=args.k,
        resolution=args.resolution,
        title=(
            f"Vanilla baseline clusters\n"
            f"k={args.k}, embedding-only kmeans on LBM-subset, "
            f"extended to full set"
        ),
        out_path=out_root / "fig5_cluster_map_vanilla.png",
    )
    logger.info("Wrote fig5_cluster_map_vanilla.png")

    # -----------------------------------------------------------------
    # 7. Summary JSON for later report-building
    # -----------------------------------------------------------------
    summary = {
        "study_area": args.study_area,
        "resolution": args.resolution,
        "year": args.year,
        "k": args.k,
        "n_hex_full": int(len(full_emb)),
        "n_hex_lbm_subset": int(len(joined_emb)),
        "lambdas": args.lambdas,
        "alphas": args.alphas,
        "best_variant": best_name,
        "best_metrics": {
            k: float(v) for k, v in best.to_dict().items()
            if k not in ("variant", "family")
        },
        "vanilla_metrics": {
            k: float(v) for k, v in vanilla_row.to_dict().items()
            if k not in ("variant", "family")
        },
        "date": date.today().isoformat(),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary.json")

    # -----------------------------------------------------------------
    # 8. Console summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS (sorted by family, strength)")
    print("=" * 70)
    cols = ["variant", "within_cluster_lbm_var", "silhouette_emb",
            "ari_vs_vanilla", "spatial_coherence", "runtime_s"]
    print(results[cols].to_string(index=False, float_format="%.4f"))
    print("\nBest variant:", best_name)
    print(f"  WCV reduction vs vanilla: "
          f"{(1 - best['within_cluster_lbm_var'] / vanilla_row['within_cluster_lbm_var']) * 100:.1f}%")
    print(f"  Silhouette retention: "
          f"{best['silhouette_emb'] / vanilla_row['silhouette_emb'] * 100:.1f}%")
    print(f"  ARI vs vanilla: {best['ari_vs_vanilla']:.3f}")
    print(f"\nArtifacts: {out_root}")


if __name__ == "__main__":
    main()
