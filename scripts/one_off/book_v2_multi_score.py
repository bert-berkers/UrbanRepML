"""Book of Netherlands v2 — multi-score atlas across all 6 leefbaarometer dimensions.

Lifetime: temporary (one-off, supports W2.A book rewrite 2026-05-03)
Stage: 3 (post-training analysis on stage2 embeddings)

Produces quantitative figures + sidecar JSON/CSV stats for chapters 5, 6, 7 of
`reports/2026-05-03-book/THE_BOOK.md` v2 rewrite. Replaces the morning session's
single-score (lbm-only) analysis with full 6-dim breakdown across embedding
approaches and cluster k.

Inputs:
    - LBM target: data/study_areas/netherlands/target/leefbaarometer/leefbaarometer_h3res9_2022.parquet
    - Cluster assignments (k in {5,8,12}, no k=10 in current data — book treats k=8 as canonical):
        data/study_areas/netherlands/stage3_analysis/cluster_results/{approach}/assignments.parquet
        Approaches: ring_agg_k10, concat_zscore, raw_concat, supervised_unet_kendall
    - Embeddings (res9): stage2_multimodal/{ring_agg,concat,unet}/embeddings/netherlands_res9_20mix.parquet

Outputs (under reports/2026-05-03-book/v2/):
    - ch5/{approach}_signature_table.{csv,md}: 6-dim per-cluster table (mean ± σ + p-rank)
    - ch5/{approach}_multiscore_violins.png: 6-panel violin grid by cluster
    - ch5/{approach}_score_distance_matrix.png: 8x8 heatmap, Euclidean in 6D z-space
    - ch5/{approach}_score_dendrogram.png: Ward linkage in 6D z-space
    - ch5/cross_embedding_fstat.{csv,md,png}: ANOVA F-stat per dim per embedding
    - ch5/leefbaarometer_per_cluster_full.json: extended sidecar with all 6 dims
    - ch6/k_sweep_metrics.png: silhouette / CH / DB vs k
    - ch6/within_cluster_variance_table.{csv,md}: per-k normalized within-var
    - ch6/best_k_violins.png: 6-panel violins at best k
    - ch7/probe_r2_per_dim.{csv,md,png}: ridge probe R² per dim
    - ch7/cluster_rank_correlation.png: 6x6 Spearman heatmap
    - ch7/probe_summary.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold

from utils.paths import StudyAreaPaths

# ---- Constants ---------------------------------------------------------------

LBM_DIMS = ["lbm", "fys", "onv", "soc", "vrz", "won"]
LBM_LABELS = {
    "lbm": "Composite (lbm)",
    "fys": "Physical (fys)",
    "onv": "Safety (onv)",
    "soc": "Social (soc)",
    "vrz": "Amenities (vrz)",
    "won": "Housing (won)",
}
RNG = 42

# k=8 is the canonical cluster count for the book v2:
#  - Best silhouette for ring_agg (0.228) and concat_zscore (0.241)
#  - Spec asked for k=10, but cluster_results have only k in {5, 8, 12}
#  - 8 closest to 10 and silhouette-optimal
CANON_K = 8

APPROACH_LABELS = {
    "ring_agg_k10": "Ring Aggregation (208D)",
    "concat_zscore": "Concat z-scored (208D)",
    "supervised_unet_kendall": "Supervised UNet (128D)",
}
APPROACHES_FOR_BOOK = list(APPROACH_LABELS.keys())

EMBED_PATHS = {
    "ring_agg_k10": "data/study_areas/netherlands/stage2_multimodal/ring_agg/embeddings/netherlands_res9_20mix.parquet",
    "concat_zscore": "data/study_areas/netherlands/stage2_multimodal/concat/embeddings/netherlands_res9_20mix.parquet",
    "supervised_unet_kendall": "data/study_areas/netherlands/stage2_multimodal/unet/embeddings/netherlands_res9_20mix.parquet",
}

OUT_ROOT = Path("reports/2026-05-03-book/v2")

# ---- Loaders -----------------------------------------------------------------


def load_lbm_target() -> pd.DataFrame:
    paths = StudyAreaPaths("netherlands")
    fp = paths.target_file("leefbaarometer", resolution=9, year=2022)
    df = pd.read_parquet(fp)
    # Z-score lbm so all 6 dims are on comparable scale for distance work.
    df["lbm_z"] = (df["lbm"] - df["lbm"].mean()) / df["lbm"].std()
    return df[["lbm", "fys", "onv", "soc", "vrz", "won", "lbm_z"]].copy()


def load_clusters(approach: str, k: int) -> pd.Series:
    fp = (
        Path("data/study_areas/netherlands/stage3_analysis/cluster_results")
        / approach
        / "assignments.parquet"
    )
    df = pd.read_parquet(fp)
    df = df[df["k"] == k]
    s = df.set_index("region_id")["cluster_label"]
    s.name = f"{approach}_k{k}"
    return s


def join_clusters_lbm(approach: str, k: int, lbm: pd.DataFrame) -> pd.DataFrame:
    cl = load_clusters(approach, k)
    df = lbm.join(cl, how="inner")
    df = df.rename(columns={cl.name: "cluster"})
    return df.dropna(subset=["cluster", "lbm"])


# ---- Per-cluster signature ---------------------------------------------------


def compute_signature_table(joined: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return DataFrame: cluster x [n_hex, lbm mean±σ p_rank, fys ..., won ...]."""
    rows = []
    # Compute across-cluster means for percentile rank lookup, per dim.
    per_dim_cluster_means = {
        d: joined.groupby("cluster")[d].mean().to_dict() for d in LBM_DIMS
    }
    for c in range(k):
        sub = joined[joined["cluster"] == c]
        if sub.empty:
            continue
        row = {"cluster": c, "n_hex": len(sub)}
        for d in LBM_DIMS:
            mean = sub[d].mean()
            std = sub[d].std()
            # Percentile rank of this cluster mean within all clusters' means.
            all_means = pd.Series(per_dim_cluster_means[d])
            rank = (all_means < mean).sum() / max(1, len(all_means) - 1)
            row[f"{d}_mean"] = mean
            row[f"{d}_std"] = std
            row[f"{d}_prank"] = rank
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("lbm_mean", ascending=False).reset_index(drop=True)
    df["rank_by_lbm"] = df.index
    return df


def signature_to_markdown(sig: pd.DataFrame) -> str:
    """Compact markdown table: cluster, n_hex, then `mean ± σ (p%)` per dim."""
    cols = ["cluster", "n_hex"] + [f"{d}" for d in LBM_DIMS]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, r in sig.iterrows():
        cells = [str(int(r["cluster"])), f"{int(r['n_hex']):,}"]
        for d in LBM_DIMS:
            cells.append(
                f"{r[f'{d}_mean']:+.3f} ± {r[f'{d}_std']:.3f} (p{int(round(r[f'{d}_prank']*100))})"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- Multi-score violin grid -------------------------------------------------


def plot_multiscore_violins(joined: pd.DataFrame, approach: str, out_fp: Path) -> None:
    # Sort clusters by lbm mean for consistent x-axis across panels.
    order = (
        joined.groupby("cluster")["lbm"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, dim in zip(axes.ravel(), LBM_DIMS):
        sns.violinplot(
            data=joined,
            x="cluster",
            y=dim,
            order=order,
            ax=ax,
            inner="quartile",
            linewidth=0.8,
            cut=0,
            density_norm="width",
            palette="tab20",
        )
        ax.set_title(LBM_LABELS[dim], fontsize=10)
        ax.set_xlabel("cluster (sorted by lbm mean)" if dim in ("soc", "vrz", "won") else "")
        ax.set_ylabel(dim)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"{APPROACH_LABELS[approach]} — k={CANON_K} cluster signatures across 6 LBM dimensions",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Cluster score-profile distance matrix + dendrogram ----------------------


def cluster_centroids_zscore(joined: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Per-cluster mean vector in 6-dim z-scored score space."""
    # z-score lbm; the others are already z-scored globally.
    means = joined[LBM_DIMS].mean()
    stds = joined[LBM_DIMS].std()
    z = (joined[LBM_DIMS] - means) / stds
    z["cluster"] = joined["cluster"].values
    centroids = z.groupby("cluster").mean()
    centroids = centroids.sort_index()
    return centroids.values, centroids.index.tolist()


def plot_distance_matrix(joined: pd.DataFrame, approach: str, out_fp: Path) -> np.ndarray:
    cent, idx = cluster_centroids_zscore(joined)
    D = squareform(pdist(cent, metric="euclidean"))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        D,
        cmap="rocket_r",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        xticklabels=idx,
        yticklabels=idx,
        ax=ax,
        cbar_kws={"label": "Euclidean distance (6D z-score space)"},
    )
    ax.set_title(
        f"{APPROACH_LABELS[approach]} — cluster score-profile distance (k={CANON_K})",
        fontsize=11,
    )
    ax.set_xlabel("cluster")
    ax.set_ylabel("cluster")
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return D


def plot_dendrogram(joined: pd.DataFrame, approach: str, out_fp: Path) -> None:
    cent, idx = cluster_centroids_zscore(joined)
    Z = linkage(cent, method="ward")
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(Z, labels=idx, ax=ax, color_threshold=0.7 * Z[:, 2].max())
    ax.set_title(
        f"{APPROACH_LABELS[approach]} — cluster dendrogram (Ward, 6D z-score space)",
        fontsize=11,
    )
    ax.set_xlabel("cluster")
    ax.set_ylabel("Ward linkage distance")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Cross-embedding F-stat --------------------------------------------------


def compute_cross_embedding_fstat(
    lbm: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Per (dim, approach), one-way ANOVA F-stat across approach's k=8 clusters."""
    rows = []
    cluster_metrics: dict = {}
    for app in APPROACHES_FOR_BOOK:
        joined = join_clusters_lbm(app, CANON_K, lbm)
        # Embedding-level silhouette (computed in metrics.parquet, cite there)
        cluster_metrics[app] = {"n_hexes_with_lbm": int(len(joined))}
        for dim in LBM_DIMS:
            groups = [g[dim].values for _, g in joined.groupby("cluster") if len(g) > 1]
            if len(groups) < 2:
                continue
            f, p = f_oneway(*groups)
            rows.append(
                {
                    "embedding": app,
                    "embedding_label": APPROACH_LABELS[app],
                    "dim": dim,
                    "f_stat": float(f),
                    "p_value": float(p),
                    "n_clusters": len(groups),
                }
            )
    return pd.DataFrame(rows), cluster_metrics


def plot_fstat_heatmap(fstat_df: pd.DataFrame, out_fp: Path) -> None:
    pivot = fstat_df.pivot(index="dim", columns="embedding", values="f_stat")
    pivot = pivot.reindex(LBM_DIMS)
    pivot.columns = [APPROACH_LABELS[c] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(
        pivot,
        cmap="viridis",
        annot=True,
        fmt=".0f",
        ax=ax,
        cbar_kws={"label": "F-statistic (higher = clusters partition dim more cleanly)"},
    )
    ax.set_title(
        f"Cross-embedding ANOVA — does this embedding's k={CANON_K} clusters partition each LBM dim?",
        fontsize=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("LBM dimension")
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Chapter 6: k-sweep ------------------------------------------------------


def k_sweep(
    embed_path: Path,
    lbm: pd.DataFrame,
    k_range: range,
    sample_n: int = 60_000,
) -> pd.DataFrame:
    """Fit MiniBatchKMeans for each k on a sub-sample (for silhouette feasibility)
    and report silhouette / CH / DB plus per-dim within-cluster variance share.

    Variance share is computed on the *full* joined data set (not the sample).
    """
    print(f"  [k_sweep] loading {embed_path}")
    emb = pd.read_parquet(embed_path)
    print(f"  [k_sweep] embeddings shape={emb.shape}")
    # Restrict to hexes with LBM (the analysis target).
    common = emb.index.intersection(lbm.index)
    emb_lbm = emb.loc[common]
    lbm_full = lbm.loc[common]
    print(f"  [k_sweep] common rows with LBM: {len(common):,}")

    rng = np.random.default_rng(RNG)
    sample_idx = rng.choice(len(emb_lbm), size=min(sample_n, len(emb_lbm)), replace=False)
    Xs = emb_lbm.values[sample_idx]
    print(f"  [k_sweep] silhouette sample size: {len(Xs):,}")

    rows = []
    for k in k_range:
        t0 = time.time()
        kmeans = MiniBatchKMeans(
            n_clusters=k, random_state=RNG, batch_size=4096, n_init=3, max_iter=200
        )
        labels_full = kmeans.fit_predict(emb_lbm.values)
        labels_sample = labels_full[sample_idx]
        # Metrics on sample for tractability.
        if len(np.unique(labels_sample)) >= 2:
            sil = silhouette_score(Xs, labels_sample, sample_size=10_000, random_state=RNG)
            ch = calinski_harabasz_score(Xs, labels_sample)
            db = davies_bouldin_score(Xs, labels_sample)
        else:
            sil = ch = db = float("nan")

        # Within-cluster variance share per LBM dim, full data
        row = {"k": k, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}
        df_full = lbm_full.copy()
        df_full["cluster"] = labels_full
        for dim in LBM_DIMS:
            tot = df_full[dim].var()
            within = (
                df_full.groupby("cluster")[dim].apply(lambda s: s.var() * len(s)).sum()
                / len(df_full)
            )
            row[f"{dim}_within_share"] = within / tot if tot > 0 else float("nan")
        rows.append(row)
        print(
            f"    k={k:>2d} sil={sil:.3f} ch={ch:.0f} db={db:.3f} "
            f"lbm_within={row['lbm_within_share']:.3f} ({time.time()-t0:.1f}s)"
        )
    return pd.DataFrame(rows)


def plot_k_sweep(sweep: pd.DataFrame, out_fp: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, label, ascending_better in zip(
        axes,
        ["silhouette", "calinski_harabasz", "davies_bouldin"],
        ["Silhouette (higher better)", "Calinski-Harabasz (higher better)", "Davies-Bouldin (lower better)"],
        [True, True, False],
    ):
        ax.plot(sweep["k"], sweep[metric], "o-", color="tab:blue")
        best_k = int(sweep["k"].iloc[sweep[metric].idxmax() if ascending_better else sweep[metric].idxmin()])
        ax.axvline(best_k, ls="--", color="red", alpha=0.6, label=f"best k={best_k}")
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Ring Aggregation — k-sweep clustering metrics on 208D embedding (60k-hex silhouette sample)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


def variance_share_table_md(sweep: pd.DataFrame) -> str:
    cols = ["k"] + [f"{d}_within_share" for d in LBM_DIMS]
    sub = sweep[cols].copy()
    header = "| k | " + " | ".join(LBM_DIMS) + " |"
    sep = "|---|" + "|".join(["---"] * len(LBM_DIMS)) + "|"
    lines = [header, sep]
    for _, r in sub.iterrows():
        cells = [str(int(r["k"]))] + [f"{r[f'{d}_within_share']:.3f}" for d in LBM_DIMS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- Chapter 7: per-dim ridge probes -----------------------------------------


def fit_ridge_probes(
    embed_path: Path,
    lbm: pd.DataFrame,
    n_splits: int = 5,
    sample_n: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit ridge probes per LBM dim, return (R² table, per-dim residual table).

    Returns:
        r2_df: rows=dim, columns=[r2_mean, r2_std, n_hexes]
        resid_df: per-region residuals for the canonical dim (lbm) only — kept
                  small because saving 6 full dim residuals per hex blows up size.
    """
    print(f"  [probe] loading {embed_path}")
    emb = pd.read_parquet(embed_path)
    common = emb.index.intersection(lbm.index)
    emb_c = emb.loc[common]
    lbm_c = lbm.loc[common]
    print(f"  [probe] common rows: {len(common):,}")

    rng = np.random.default_rng(RNG)
    if len(common) > sample_n:
        sample_idx = rng.choice(len(common), size=sample_n, replace=False)
        emb_c = emb_c.iloc[sample_idx]
        lbm_c = lbm_c.iloc[sample_idx]
        print(f"  [probe] sub-sampled to {sample_n:,} for speed")

    X = emb_c.values
    rows = []
    resid_records = {}
    for dim in LBM_DIMS:
        y = lbm_c[dim].values
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RNG)
        r2s = []
        y_pred_full = np.zeros_like(y, dtype=float)
        for tr, te in kf.split(X):
            mdl = Ridge(alpha=1.0, random_state=RNG)
            mdl.fit(X[tr], y[tr])
            yp = mdl.predict(X[te])
            y_pred_full[te] = yp
            r2s.append(r2_score(y[te], yp))
        r2_mean = float(np.mean(r2s))
        r2_std = float(np.std(r2s))
        rows.append(
            {
                "dim": dim,
                "label": LBM_LABELS[dim],
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "n_hexes": int(len(y)),
            }
        )
        resid_records[dim] = pd.DataFrame(
            {
                "actual": y,
                "predicted": y_pred_full,
                "residual": y - y_pred_full,
            },
            index=emb_c.index,
        )
        print(f"    {dim}: R² = {r2_mean:.3f} ± {r2_std:.3f}")

    r2_df = pd.DataFrame(rows)
    return r2_df, resid_records


def plot_probe_r2(r2_df: pd.DataFrame, out_fp: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = sns.color_palette("rocket_r", len(r2_df))
    bars = ax.bar(
        r2_df["dim"],
        r2_df["r2_mean"],
        yerr=r2_df["r2_std"],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, r2 in zip(bars, r2_df["r2_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{r2:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("R² (5-fold CV)")
    ax.set_xlabel("LBM dimension")
    ax.set_title(
        "Ridge probe per LBM dimension on Ring Aggregation 208D embedding (α=1.0)",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(0.7, r2_df["r2_mean"].max() * 1.15))
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_rank_correlation(
    lbm: pd.DataFrame, approach: str, out_fp: Path
) -> pd.DataFrame:
    """Spearman corr of cluster ranks across LBM dims (using approach's k=8)."""
    joined = join_clusters_lbm(approach, CANON_K, lbm)
    cluster_means = joined.groupby("cluster")[LBM_DIMS].mean()
    cluster_ranks = cluster_means.rank()  # rank each dim across clusters
    rho_mat = np.zeros((len(LBM_DIMS), len(LBM_DIMS)))
    for i, di in enumerate(LBM_DIMS):
        for j, dj in enumerate(LBM_DIMS):
            rho, _ = spearmanr(cluster_ranks[di], cluster_ranks[dj])
            rho_mat[i, j] = rho
    rho_df = pd.DataFrame(rho_mat, index=LBM_DIMS, columns=LBM_DIMS)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        rho_df,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Spearman ρ of cluster-mean ranks"},
    )
    ax.set_title(
        f"{APPROACH_LABELS[approach]} — cluster-rank correlation across LBM dims (k={CANON_K})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rho_df


# ---- Top residual hexes ------------------------------------------------------


def top_residuals(resid: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    over = resid.nsmallest(n, "residual").assign(direction="over-predicted")
    under = resid.nlargest(n, "residual").assign(direction="under-predicted")
    return over.reset_index(), under.reset_index()


# ---- Top-level driver --------------------------------------------------------


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "ch5").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "ch6").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "ch7").mkdir(parents=True, exist_ok=True)

    print("== Loading LBM target ==")
    lbm = load_lbm_target()
    print(f"   {len(lbm):,} hexes with LBM data, {len(LBM_DIMS)} dims")

    # =================== CHAPTER 5 ===================
    print("\n== Chapter 5: per-embedding multi-score signature ==")
    full_sidecar: dict = {
        "lbm_year": 2022,
        "canonical_k": CANON_K,
        "k_choice_rationale": (
            "Spec asked for k=10 but cluster_results contain only k in {5,8,12}. "
            "Selected k=8: best silhouette for ring_agg (0.228) and concat_zscore (0.241), "
            "closest to spec's k=10 framing."
        ),
        "lbm_country_stats": {
            d: {
                "mean": float(lbm[d].mean()),
                "std": float(lbm[d].std()),
                "min": float(lbm[d].min()),
                "max": float(lbm[d].max()),
            }
            for d in LBM_DIMS
        },
        "n_hexes_with_lbm": int(len(lbm)),
        "approaches": {},
    }

    sigs_md: dict[str, str] = {}
    for app in APPROACHES_FOR_BOOK:
        print(f"\n-- {app} --")
        joined = join_clusters_lbm(app, CANON_K, lbm)
        print(f"   joined rows: {len(joined):,}")
        sig = compute_signature_table(joined, CANON_K)

        # CSV
        sig.to_csv(OUT_ROOT / "ch5" / f"{app}_signature_table.csv", index=False)
        # MD
        md = signature_to_markdown(sig)
        (OUT_ROOT / "ch5" / f"{app}_signature_table.md").write_text(md, encoding="utf-8")
        sigs_md[app] = md

        # Figures
        plot_multiscore_violins(joined, app, OUT_ROOT / "ch5" / f"{app}_multiscore_violins.png")
        D = plot_distance_matrix(joined, app, OUT_ROOT / "ch5" / f"{app}_score_distance_matrix.png")
        plot_dendrogram(joined, app, OUT_ROOT / "ch5" / f"{app}_score_dendrogram.png")

        # Sidecar entry
        per_cluster = {}
        for _, r in sig.iterrows():
            entry = {"n_hex": int(r["n_hex"]), "rank_by_lbm": int(r["rank_by_lbm"])}
            for d in LBM_DIMS:
                entry[d] = {
                    "mean": float(r[f"{d}_mean"]),
                    "std": float(r[f"{d}_std"]),
                    "prank": float(r[f"{d}_prank"]),
                }
            per_cluster[int(r["cluster"])] = entry
        full_sidecar["approaches"][app] = {
            "label": APPROACH_LABELS[app],
            "n_hex_with_lbm": int(len(joined)),
            "distance_matrix_max": float(D.max()),
            "distance_matrix_mean_offdiag": float(
                (D.sum() - np.trace(D)) / (D.size - len(D))
            ),
            "per_cluster": per_cluster,
        }

    # Cross-embedding F-stat
    print("\n-- Cross-embedding F-stat --")
    fstat_df, _ = compute_cross_embedding_fstat(lbm)
    fstat_df.to_csv(OUT_ROOT / "ch5" / "cross_embedding_fstat.csv", index=False)

    pivot = fstat_df.pivot(index="dim", columns="embedding", values="f_stat").reindex(LBM_DIMS)
    md_lines = ["| dim | " + " | ".join(APPROACH_LABELS[c] for c in pivot.columns) + " |"]
    md_lines.append("|" + "|".join(["---"] * (len(pivot.columns) + 1)) + "|")
    for dim in pivot.index:
        cells = [dim] + [f"{pivot.loc[dim, c]:.0f}" for c in pivot.columns]
        md_lines.append("| " + " | ".join(cells) + " |")
    (OUT_ROOT / "ch5" / "cross_embedding_fstat.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    plot_fstat_heatmap(fstat_df, OUT_ROOT / "ch5" / "cross_embedding_fstat.png")
    full_sidecar["cross_embedding_fstat"] = fstat_df.to_dict("records")

    # Sidecar JSON
    sidecar_fp = OUT_ROOT / "ch5" / "leefbaarometer_per_cluster_full.json"
    sidecar_fp.write_text(json.dumps(full_sidecar, indent=2), encoding="utf-8")
    print(f"   wrote sidecar: {sidecar_fp}")

    # =================== CHAPTER 6 ===================
    print("\n== Chapter 6: k-sweep on ring_agg ==")
    sweep = k_sweep(Path(EMBED_PATHS["ring_agg_k10"]), lbm, k_range=range(2, 16))
    sweep.to_csv(OUT_ROOT / "ch6" / "k_sweep_metrics.csv", index=False)
    plot_k_sweep(sweep, OUT_ROOT / "ch6" / "k_sweep_metrics.png")
    var_md = variance_share_table_md(sweep)
    (OUT_ROOT / "ch6" / "within_cluster_variance_table.md").write_text(var_md, encoding="utf-8")

    # Best-k violins (on the existing canonical k=8 cluster file for stability)
    joined_k8 = join_clusters_lbm("ring_agg_k10", CANON_K, lbm)
    plot_multiscore_violins(joined_k8, "ring_agg_k10", OUT_ROOT / "ch6" / "best_k_violins.png")

    # =================== CHAPTER 7 ===================
    print("\n== Chapter 7: per-dim ridge probes ==")
    r2_df, resid_records = fit_ridge_probes(
        Path(EMBED_PATHS["ring_agg_k10"]), lbm, n_splits=5, sample_n=100_000
    )
    r2_df.to_csv(OUT_ROOT / "ch7" / "probe_r2_per_dim.csv", index=False)
    md = ["| dim | label | R² mean | R² std | n |", "|---|---|---|---|---|"]
    for _, r in r2_df.iterrows():
        md.append(f"| {r['dim']} | {r['label']} | {r['r2_mean']:.3f} | {r['r2_std']:.3f} | {int(r['n_hexes']):,} |")
    (OUT_ROOT / "ch7" / "probe_r2_per_dim.md").write_text("\n".join(md), encoding="utf-8")
    plot_probe_r2(r2_df, OUT_ROOT / "ch7" / "probe_r2_per_dim.png")

    rho_df = plot_cluster_rank_correlation(lbm, "ring_agg_k10", OUT_ROOT / "ch7" / "cluster_rank_correlation.png")
    rho_df.to_csv(OUT_ROOT / "ch7" / "cluster_rank_correlation.csv")

    # Top residuals per dim, aggregated to a single MD table
    md = ["## Top residuals per LBM dim (Ring Aggregation 208D, ridge probe, 5-fold CV)", ""]
    md.append("| dim | top over-predicted (model > reality) | top under-predicted (model < reality) |")
    md.append("|---|---|---|")
    top_payload = {}
    for dim, resid in resid_records.items():
        over, under = top_residuals(resid, n=5)
        over_cells = "; ".join(
            f"{r['region_id'][:9]}… resid={r['residual']:+.3f}" for _, r in over.iterrows()
        )
        under_cells = "; ".join(
            f"{r['region_id'][:9]}… resid={r['residual']:+.3f}" for _, r in under.iterrows()
        )
        md.append(f"| {dim} | {over_cells} | {under_cells} |")
        top_payload[dim] = {
            "over": over.to_dict("records"),
            "under": under.to_dict("records"),
        }
    (OUT_ROOT / "ch7" / "top_residuals.md").write_text("\n".join(md), encoding="utf-8")

    probe_summary = {
        "embedding": "ring_agg_208D",
        "alpha": 1.0,
        "n_splits": 5,
        "sample_n": 100_000,
        "r2_per_dim": r2_df.set_index("dim")[["r2_mean", "r2_std", "n_hexes"]].to_dict("index"),
        "spearman_rank_corr_offdiag_mean": float(
            (rho_df.values.sum() - np.trace(rho_df.values)) / (rho_df.size - len(rho_df))
        ),
    }
    (OUT_ROOT / "ch7" / "probe_summary.json").write_text(json.dumps(probe_summary, indent=2), encoding="utf-8")

    print("\n== DONE ==")
    print(f"   outputs in {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
