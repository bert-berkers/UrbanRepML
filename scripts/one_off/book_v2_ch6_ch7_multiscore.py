"""W3.B — Book v2 chapter 6 (k-sweep + per-cluster signatures) and chapter 7 (per-LBM probes).

Lifetime: temporary (one-off, supports the Book of Netherlands v2 build on 2026-05-03).
Stage: Stage 3 (analysis / interpretability).

Produces, for the Netherlands study area, ring-agg-208D embedding × leefbaarometer 2022:

Ch6:
  - reports/2026-05-03-book/v2/ch6/k_sweep_metrics.png   (silhouette / CH / DB across k=2..15)
  - reports/2026-05-03-book/v2/ch6/multi_lbm_violins_k10.png (6-panel violin grid, one per LBM dim)
  - reports/2026-05-03-book/v2/ch6/lbm_var_reduction_vs_k.png  (per-dim normalized within-cluster var)
  - reports/2026-05-03-book/v2/ch6/lbm_var_vs_k.csv

Ch7:
  - reports/2026-05-03-book/v2/ch7/probe_r2_per_dim.csv  (provenance: existing artifact vs fresh ridge)
  - reports/2026-05-03-book/v2/ch7/probe_r2_bars.png
  - reports/2026-05-03-book/v2/ch7/dim_rank_correlation.png and .csv
  - reports/2026-05-03-book/v2/ch7/residuals_top10_{dim}.csv (six small files)

The cluster k used for ch6 panels is documented at runtime (k=10 if available; falls back to k=8).
No spatial maps are produced; matplotlib only, no NL outline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold

# Ensure project root is on sys.path for utils imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import StudyAreaPaths  # noqa: E402

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

STUDY_AREA = "netherlands"
RES = 9
YEAR_LABEL = "20mix"
LBM_YEAR = 2022
LBM_DIMS = ["lbm", "fys", "onv", "soc", "vrz", "won"]
K_RANGE = list(range(2, 16))
SILHOUETTE_SAMPLE = 10_000
RANDOM_STATE = 42

# Existing probe artifacts: ring_agg DNN probe predictions for all 6 dims (2026-03-07).
PROBE_ARTIFACT_DIR = (
    "data/study_areas/netherlands/stage3_analysis/dnn_probe/"
    "2026-03-07/2026-03-07_res9_stage2_fusion_ring_agg"
)

OUT_BOOK = Path("reports/2026-05-03-book/v2")
OUT_CH6 = OUT_BOOK / "ch6"
OUT_CH7 = OUT_BOOK / "ch7"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_embedding(paths: StudyAreaPaths) -> pd.DataFrame:
    fp = paths.fused_embedding_file("ring_agg", RES, YEAR_LABEL)
    log(f"Loading ring-agg embedding: {fp}")
    df = pd.read_parquet(fp)
    log(f"  shape={df.shape}, idx={df.index.name}")
    return df


def load_lbm(paths: StudyAreaPaths) -> pd.DataFrame:
    fp = paths.target_file("leefbaarometer", RES, LBM_YEAR)
    log(f"Loading LBM target: {fp}")
    df = pd.read_parquet(fp)[LBM_DIMS].copy()
    # The composite `lbm` is in raw 3.42-5.04 range; sub-scores are z-scored.
    # Z-score `lbm` independently so all 6 dims are on comparable scale.
    df["lbm"] = (df["lbm"] - df["lbm"].mean()) / df["lbm"].std()
    log(f"  shape={df.shape}; lbm z-scored to mean={df['lbm'].mean():.3e}")
    return df


def load_cluster_assignments(
    paths: StudyAreaPaths,
) -> tuple[pd.Series, int]:
    """Pick k=10 if available, else k=8, else k=5."""
    fp = paths.cluster_results("ring_agg_k10") / "assignments.parquet"
    log(f"Loading cluster assignments: {fp}")
    df = pd.read_parquet(fp)
    available_ks = sorted(df["k"].unique().tolist())
    log(f"  available k values: {available_ks}")
    for preferred in (10, 8, 5):
        if preferred in available_ks:
            chosen = preferred
            break
    else:
        chosen = available_ks[-1]
    log(f"  using k={chosen}")
    sub = df[df["k"] == chosen].set_index("region_id")["cluster_label"]
    return sub, chosen


# ---------------------------------------------------------------------
# Chapter 6: k-sweep + per-cluster signatures
# ---------------------------------------------------------------------


def k_sweep_metrics(emb: np.ndarray) -> pd.DataFrame:
    """Sweep k=2..15, MiniBatchKMeans. Sample 10K rows for silhouette."""
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(emb.shape[0], size=SILHOUETTE_SAMPLE, replace=False)

    rows = []
    for k in K_RANGE:
        t0 = time.time()
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=5,
            batch_size=4096,
        )
        labels = km.fit_predict(emb)
        # Silhouette on sampled subset (full-set is O(N^2) and infeasible at 397K).
        sample_labels = labels[sample_idx]
        sample_emb = emb[sample_idx]
        # Need >=2 unique labels in sample; with k>=2 and big sample this holds.
        sil = silhouette_score(sample_emb, sample_labels, sample_size=None)
        ch = calinski_harabasz_score(emb, labels)
        db = davies_bouldin_score(emb, labels)
        elapsed = time.time() - t0
        rows.append(
            {"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}
        )
        log(
            f"  k={k:>2}: sil={sil:.4f} CH={ch:.0f} DB={db:.4f}  ({elapsed:.1f}s)"
        )
    return pd.DataFrame(rows)


def plot_k_sweep_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(metrics_df["k"], metrics_df["silhouette"], "o-", color="#2c7fb8")
    axes[0].set_title("Silhouette score (higher = better)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("silhouette")
    axes[0].grid(alpha=0.3)

    axes[1].plot(metrics_df["k"], metrics_df["calinski_harabasz"], "o-", color="#31a354")
    axes[1].set_title("Calinski-Harabasz (higher = better)")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("CH index")
    axes[1].grid(alpha=0.3)

    axes[2].plot(metrics_df["k"], metrics_df["davies_bouldin"], "o-", color="#e6550d")
    axes[2].set_title("Davies-Bouldin (lower = better)")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("DB index")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Ring-agg-208D cluster quality vs k (Netherlands, res9)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out_path}")


def plot_multi_lbm_violins(
    lbm_df: pd.DataFrame, clusters: pd.Series, k_used: int, out_path: Path
) -> None:
    """6-panel grid: one violin plot per LBM dim, x = cluster_label."""
    df = lbm_df.join(clusters.rename("cluster"), how="inner").dropna()
    log(f"  multi_lbm_violins: joined {len(df):,} hexes")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    cluster_ids = sorted(df["cluster"].unique())
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in cluster_ids]

    for ax, dim in zip(axes.ravel(), LBM_DIMS):
        data = [df.loc[df["cluster"] == c, dim].values for c in cluster_ids]
        parts = ax.violinplot(
            data, positions=cluster_ids, showmeans=False, showmedians=True, widths=0.85
        )
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.75)
            body.set_edgecolor("black")
            body.set_linewidth(0.5)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(0.7)
        ax.set_title(f"{dim} (z-scored)")
        ax.set_xlabel("cluster")
        ax.set_ylabel(dim)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Ring-agg k={k_used} — LBM sub-score distributions per cluster (Netherlands 2022)",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out_path}")


def lbm_var_reduction_vs_k(
    emb: np.ndarray, region_ids: pd.Index, lbm_df: pd.DataFrame
) -> pd.DataFrame:
    """For each k=2..15: cluster ring-agg, compute normalized within-cluster variance per LBM dim.

    Within-cluster variance: sum over clusters of (sigma^2 * n_hex / total_n_hex), divided
    by the global variance of the dim. 1.0 = no reduction; lower is better.
    """
    # Align embedding ↔ LBM
    common = region_ids.intersection(lbm_df.index)
    log(f"  var-reduction: {len(common):,} common region_ids (emb x LBM)")
    emb_aligned = pd.DataFrame(emb, index=region_ids).loc[common].values
    lbm_aligned = lbm_df.loc[common, LBM_DIMS].values  # (n, 6)

    global_var = lbm_aligned.var(axis=0)  # (6,)

    rows = []
    for k in K_RANGE:
        km = MiniBatchKMeans(
            n_clusters=k, random_state=RANDOM_STATE, n_init=5, batch_size=4096
        )
        labels = km.fit_predict(emb_aligned)
        # Within-cluster variance via groupby-style calculation
        within = np.zeros(len(LBM_DIMS))
        n_total = lbm_aligned.shape[0]
        for c in range(k):
            mask = labels == c
            n_c = mask.sum()
            if n_c < 2:
                continue
            var_c = lbm_aligned[mask].var(axis=0)
            within += var_c * n_c
        within /= n_total
        normalized = within / global_var  # 1.0 = no reduction
        row = {"k": k}
        for dim, val in zip(LBM_DIMS, normalized):
            row[dim] = val
        rows.append(row)
        log(f"  k={k:>2} normalized within-cluster var: " +
            ", ".join(f"{d}={row[d]:.3f}" for d in LBM_DIMS))
    return pd.DataFrame(rows)


def plot_var_reduction(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10
    for i, dim in enumerate(LBM_DIMS):
        ax.plot(df["k"], df[dim], "o-", label=dim, color=cmap(i), linewidth=1.8)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Normalized within-cluster variance")
    ax.set_title(
        "LBM dim variance partition vs k\n(lower = cluster captures more of the dim)"
    )
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, label="no reduction")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out_path}")


# ---------------------------------------------------------------------
# Chapter 7: per-LBM probes
# ---------------------------------------------------------------------


def load_existing_probe_predictions(
    artifact_dir: Path,
) -> dict[str, pd.DataFrame] | None:
    """Each predictions_{dim}.parquet has columns [actual, predicted, residual] indexed by region_id."""
    out: dict[str, pd.DataFrame] = {}
    for dim in LBM_DIMS:
        fp = artifact_dir / f"predictions_{dim}.parquet"
        if not fp.exists():
            log(f"  missing artifact for dim={dim} ({fp}); will fit fresh")
            return None
        df = pd.read_parquet(fp)
        if not {"actual", "predicted", "residual"}.issubset(df.columns):
            log(f"  artifact {fp} missing required cols")
            return None
        out[dim] = df
    log(f"  loaded existing probe artifacts for all {len(LBM_DIMS)} dims from {artifact_dir}")
    return out


def fit_fresh_ridge_probe(
    emb: np.ndarray, region_ids: pd.Index, lbm_df: pd.DataFrame, dim: str
) -> pd.DataFrame:
    """5-fold CV ridge; return per-region (actual, predicted, residual)."""
    common = region_ids.intersection(lbm_df.index)
    emb_aligned = pd.DataFrame(emb, index=region_ids).loc[common].values
    y = lbm_df.loc[common, dim].values

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preds = np.zeros_like(y, dtype=float)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(emb_aligned)):
        model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        model.fit(emb_aligned[tr_idx], y[tr_idx])
        preds[te_idx] = model.predict(emb_aligned[te_idx])

    df = pd.DataFrame(
        {"actual": y, "predicted": preds, "residual": y - preds}, index=common
    )
    df.index.name = "region_id"
    return df


def write_residuals_top10(pred_df: pd.DataFrame, dim: str, out_dir: Path) -> None:
    """Top 5 over-predicted (most negative residual) and 5 under-predicted (most positive)."""
    sorted_by_resid = pred_df.sort_values("residual")
    over = sorted_by_resid.head(5).copy()
    under = sorted_by_resid.tail(5).copy()
    combined = pd.concat([under, over])
    combined = combined.reset_index()[
        ["region_id", "actual", "predicted", "residual"]
    ]
    out_path = out_dir / f"residuals_top10_{dim}.csv"
    combined.to_csv(out_path, index=False)
    log(f"  wrote {out_path}")


def plot_probe_r2_bars(r2_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(len(r2_df))]
    bars = ax.bar(r2_df["dim"], r2_df["r2"], color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, r2_df["r2"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Out-of-fold R²")
    ax.set_xlabel("LBM dimension")
    ax.set_title("Per-LBM-dim probe R² (ring-agg-208D, res9, Netherlands 2022)")
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out_path}")


def dim_rank_correlation(
    lbm_df: pd.DataFrame, clusters: pd.Series, k_used: int
) -> pd.DataFrame:
    """Spearman ρ of cluster-mean rankings between dim pairs."""
    df = lbm_df.join(clusters.rename("cluster"), how="inner").dropna()
    cluster_means = df.groupby("cluster")[LBM_DIMS].mean()  # (k_used, 6)
    log(f"  cluster_means shape={cluster_means.shape} (k_used={k_used})")
    # Rank each column, then Spearman is just Pearson on ranks
    ranks = cluster_means.rank(axis=0)
    n = len(LBM_DIMS)
    mat = np.zeros((n, n))
    for i, di in enumerate(LBM_DIMS):
        for j, dj in enumerate(LBM_DIMS):
            rho, _ = spearmanr(cluster_means[di], cluster_means[dj])
            mat[i, j] = rho
    out = pd.DataFrame(mat, index=LBM_DIMS, columns=LBM_DIMS)
    return out


def plot_rank_correlation(corr: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                    fontsize=10)
    ax.set_title("Spearman ρ of cluster-mean rankings between LBM dims")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    paths = StudyAreaPaths(STUDY_AREA)

    OUT_CH6.mkdir(parents=True, exist_ok=True)
    OUT_CH7.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    emb_df = load_embedding(paths)
    region_ids = emb_df.index
    emb_arr = emb_df.values.astype(np.float32)
    lbm_df = load_lbm(paths)
    clusters_full, k_used = load_cluster_assignments(paths)
    log(f"k chosen for ch6 panels: {k_used} "
        f"(task asked for k=10; falls back to k=8 if 10 unavailable)")

    # ---------- Chapter 6 ----------
    log("=" * 70)
    log("CH6: k-sweep silhouette / CH / DB metrics")
    metrics_df = k_sweep_metrics(emb_arr)
    plot_k_sweep_metrics(metrics_df, OUT_CH6 / "k_sweep_metrics.png")
    metrics_df.to_csv(OUT_CH6 / "k_sweep_metrics.csv", index=False)

    log("CH6: multi-LBM violins for k_used")
    plot_multi_lbm_violins(
        lbm_df, clusters_full, k_used, OUT_CH6 / "multi_lbm_violins_k10.png"
    )

    log("CH6: LBM within-cluster variance reduction vs k")
    var_df = lbm_var_reduction_vs_k(emb_arr, region_ids, lbm_df)
    var_df.to_csv(OUT_CH6 / "lbm_var_vs_k.csv", index=False)
    plot_var_reduction(var_df, OUT_CH6 / "lbm_var_reduction_vs_k.png")

    # ---------- Chapter 7 ----------
    log("=" * 70)
    log("CH7: per-LBM ridge probes (R² + residuals)")

    artifact_dir = (PROJECT_ROOT / PROBE_ARTIFACT_DIR).resolve()
    existing = load_existing_probe_predictions(artifact_dir)

    r2_rows = []
    pred_dfs: dict[str, pd.DataFrame] = {}

    if existing is not None:
        # Reuse existing artifacts. Note: these are DNN probe predictions, not pure ridge.
        # They are available for all 6 dims so we use them for R² + residual extraction.
        for dim in LBM_DIMS:
            df = existing[dim]
            r2 = r2_score(df["actual"], df["predicted"])
            r2_rows.append(
                {
                    "dim": dim,
                    "r2": r2,
                    "n_train": "N/A (full)",
                    "n_test": len(df),
                    "source": f"existing_artifact:{PROBE_ARTIFACT_DIR}/predictions_{dim}.parquet",
                }
            )
            pred_dfs[dim] = df
            log(f"  {dim}: reused R²={r2:.4f} (n={len(df):,})")
    else:
        log("  fitting fresh ridge probes (5-fold CV, alpha=1.0)")
        for dim in LBM_DIMS:
            df = fit_fresh_ridge_probe(emb_arr, region_ids, lbm_df, dim)
            r2 = r2_score(df["actual"], df["predicted"])
            n_total = len(df)
            r2_rows.append(
                {
                    "dim": dim,
                    "r2": r2,
                    "n_train": int(n_total * 4 / 5),
                    "n_test": int(n_total / 5),
                    "source": "fresh_ridge",
                }
            )
            pred_dfs[dim] = df
            log(f"  {dim}: fresh R²={r2:.4f} (n={n_total:,})")

    r2_df = pd.DataFrame(r2_rows)
    r2_df.to_csv(OUT_CH7 / "probe_r2_per_dim.csv", index=False)
    log(f"  wrote {OUT_CH7 / 'probe_r2_per_dim.csv'}")
    plot_probe_r2_bars(r2_df, OUT_CH7 / "probe_r2_bars.png")

    # Residuals top-10 per dim
    for dim in LBM_DIMS:
        write_residuals_top10(pred_dfs[dim], dim, OUT_CH7)

    # Spearman rank correlation
    log("CH7: Spearman rank correlation of cluster-mean rankings")
    corr = dim_rank_correlation(lbm_df, clusters_full, k_used)
    corr.to_csv(OUT_CH7 / "dim_rank_correlation.csv")
    plot_rank_correlation(corr, OUT_CH7 / "dim_rank_correlation.png")

    # Identify strongest off-diagonal correlations
    off_diag = []
    for i, di in enumerate(LBM_DIMS):
        for j, dj in enumerate(LBM_DIMS):
            if i < j:
                off_diag.append((di, dj, corr.iloc[i, j]))
    off_diag.sort(key=lambda x: abs(x[2]), reverse=True)
    log("Strongest LBM-dim rank correlations:")
    for di, dj, rho in off_diag[:5]:
        log(f"  {di} vs {dj}: rho={rho:+.3f}")

    elapsed = time.time() - t_start
    log("=" * 70)
    log(f"TOTAL RUNTIME: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log(f"k_used={k_used}")
    log("R² summary:")
    for row in r2_rows:
        log(f"  {row['dim']}: {row['r2']:.4f}  ({row['source'][:50]})")


if __name__ == "__main__":
    main()
