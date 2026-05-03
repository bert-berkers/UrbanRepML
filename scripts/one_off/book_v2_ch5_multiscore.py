"""Book of Netherlands v2 — Chapter 5 multi-score atlas across 6 leefbaarometer dims.

Lifetime: temporary (W2.A1, supports book v2 chapter 5 rewrite 2026-05-03)
Stage: 3 (post-training analysis on stage2 cluster assignments + lbm target)

Extends the morning's lbm-only chapter 5 to all 6 LBM dims (lbm, fys, onv, soc, vrz, won).
Produces per-embedding cluster signature tables (mean +/- sigma + percentile rank), violin
panels per dim, score-profile distance matrix, Ward dendrogram on cluster centroids in 6D
z-score space, plus a cross-embedding ANOVA F-stat table across dims.

K-NOTE: spec asked for k=10 but cluster_results contain only k in {5, 8, 12}. Selected
k=8 -- closest to spec's k=10, silhouette-optimal for ring_agg (0.228) and
concat_zscore (0.241). Documented in sidecar JSON `_meta.k_choice_rationale`.

Inputs:
    - LBM target: data/study_areas/netherlands/target/leefbaarometer/leefbaarometer_h3res9_2022.parquet
        Columns: lbm (~3.42-5.04), fys/onv/soc/vrz/won (z-scored). lbm is z-scored
        independently before the 6D distance work so all dims share scale.
    - Cluster assignments: data/study_areas/netherlands/stage3_analysis/cluster_results/
        {ring_agg_k10,concat_zscore,supervised_unet_kendall}/assignments.parquet  (k=8)

Outputs (under reports/2026-05-03-book/v2/ch5/):
    - {approach}_signature_table.{csv,md}: 8 rows x 6 dims with mean+/-sigma + p_rank
    - {approach}_multiscore_violins.png:   6 panels (one per dim), violins per cluster
    - {approach}_score_distance_matrix.png 8x8 Euclidean dist over 6D z centroids
    - {approach}_score_dendrogram.png       Ward linkage of cluster centroids
    - cross_embedding_fstat.{csv,md,png}    F-stat per (dim, embedding) + argmax
    - leefbaarometer_per_cluster_full.json  extended sidecar (all dims x all approaches)

Does NOT touch the morning's `leefbaarometer_per_cluster.json` (preserved alongside the new
`_full.json`). No spatial maps are produced -- so the boundary outline rule does not apply.
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
from scipy.stats import f_oneway

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

# k=8 used in lieu of spec's k=10 -- only k in {5,8,12} present in cluster_results.
# 8 is silhouette-optimal for ring_agg + concat_zscore and closest to spec's 10.
CANON_K = 8

APPROACH_LABELS = {
    "concat_zscore": "Concat z-scored (208D)",
    "ring_agg_k10": "Ring Aggregation (208D)",
    "supervised_unet_kendall": "Supervised UNet (128D)",
}
APPROACHES = list(APPROACH_LABELS.keys())

OUT_DIR = Path("reports/2026-05-03-book/v2/ch5")


# ---- Data loaders ------------------------------------------------------------


def load_lbm_target() -> pd.DataFrame:
    """LBM target with lbm independently z-scored (lbm_z).

    Sub-scores (fys/onv/soc/vrz/won) are already z-scored globally; lbm is on its
    own scale (~3.42-5.04). For 6D distance work in cluster-centroid space we need
    all six on a comparable scale, so add a `lbm_z` column.
    """
    paths = StudyAreaPaths("netherlands")
    fp = paths.target_file("leefbaarometer", resolution=9, year=2022)
    df = pd.read_parquet(fp)
    df["lbm_z"] = (df["lbm"] - df["lbm"].mean()) / df["lbm"].std()
    return df[["lbm", "fys", "onv", "soc", "vrz", "won", "lbm_z"]].copy()


def load_clusters(approach: str, k: int) -> pd.Series:
    paths = StudyAreaPaths("netherlands")
    fp = paths.cluster_results(approach) / "assignments.parquet"
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
    """Per-cluster: n_hex + (mean, std, p_rank) per LBM dim.

    p_rank for cluster c on dim d = (# clusters with mean_d < cluster c's mean_d) /
    (k-1). So the lowest-mean cluster gets p0 and the highest gets p100. Uses the
    raw `lbm` (not z-scored) for the lbm row -- the table reports the user-facing
    score, not the z-transform.
    """
    rows = []
    per_dim_cluster_means = {
        d: joined.groupby("cluster")[d].mean().to_dict() for d in LBM_DIMS
    }
    for c in sorted(joined["cluster"].unique()):
        sub = joined[joined["cluster"] == c]
        if sub.empty:
            continue
        row = {"cluster": int(c), "n_hex": int(len(sub))}
        for d in LBM_DIMS:
            mean = sub[d].mean()
            std = sub[d].std()
            all_means = pd.Series(per_dim_cluster_means[d])
            rank = (all_means < mean).sum() / max(1, len(all_means) - 1)
            row[f"{d}_mean"] = float(mean)
            row[f"{d}_std"] = float(std)
            row[f"{d}_prank"] = float(rank)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("lbm_mean", ascending=False).reset_index(drop=True)
    df["rank_by_lbm"] = df.index
    return df


def signature_to_markdown(sig: pd.DataFrame) -> str:
    """Compact markdown table: cluster, n_hex, then `mean +/- std (pXX)` per dim."""
    cols = ["cluster", "n_hex"] + LBM_DIMS
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, r in sig.iterrows():
        cells = [str(int(r["cluster"])), f"{int(r['n_hex']):,}"]
        for d in LBM_DIMS:
            cells.append(
                f"{r[f'{d}_mean']:+.3f} +/- {r[f'{d}_std']:.3f} "
                f"(p{int(round(r[f'{d}_prank']*100))})"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- Multi-score violin grid -------------------------------------------------


def plot_multiscore_violins(joined: pd.DataFrame, approach: str, out_fp: Path) -> None:
    """6 subplots, violin+strip per dim, x-axis = clusters sorted by lbm mean desc."""
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
        ax.set_xlabel(
            "cluster (sorted by lbm mean)" if dim in ("soc", "vrz", "won") else ""
        )
        ax.set_ylabel(dim)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"{APPROACH_LABELS[approach]} -- k={CANON_K} cluster signatures across 6 LBM dimensions",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Cluster centroid distance matrix + dendrogram ---------------------------


def cluster_centroids_zscore(joined: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Cluster centroids in 6D z-score space (lbm z-scored independently)."""
    means = joined[LBM_DIMS].mean()
    stds = joined[LBM_DIMS].std()
    z = (joined[LBM_DIMS] - means) / stds
    z["cluster"] = joined["cluster"].values
    centroids = z.groupby("cluster").mean().sort_index()
    return centroids.values, centroids.index.tolist()


def plot_distance_matrix(
    joined: pd.DataFrame, approach: str, out_fp: Path
) -> np.ndarray:
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
        f"{APPROACH_LABELS[approach]} -- cluster score-profile distance (k={CANON_K})",
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
        f"{APPROACH_LABELS[approach]} -- cluster dendrogram (Ward, 6D z-score space)",
        fontsize=11,
    )
    ax.set_xlabel("cluster")
    ax.set_ylabel("Ward linkage distance")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Cross-embedding ANOVA F-stat --------------------------------------------


def compute_cross_embedding_fstat(lbm: pd.DataFrame) -> pd.DataFrame:
    """Per (dim, approach), one-way ANOVA F-stat across approach's k=8 clusters.

    Higher F = the embedding's clusters partition that dim more cleanly (more
    between-group variance relative to within-group).
    """
    rows = []
    for app in APPROACHES:
        joined = join_clusters_lbm(app, CANON_K, lbm)
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
                    "n_hex": int(len(joined)),
                }
            )
    return pd.DataFrame(rows)


def plot_fstat_heatmap(fstat_df: pd.DataFrame, out_fp: Path) -> None:
    pivot = fstat_df.pivot(index="dim", columns="embedding", values="f_stat").reindex(
        LBM_DIMS
    )
    # column order: keep stable
    pivot = pivot[APPROACHES]
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
        f"Cross-embedding ANOVA -- partition quality per LBM dim (k={CANON_K})",
        fontsize=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("LBM dimension")
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fstat_to_markdown(fstat_df: pd.DataFrame) -> str:
    """Markdown table: rows=dim, cols=embedding (+ argmax winner column)."""
    pivot = fstat_df.pivot(index="dim", columns="embedding", values="f_stat").reindex(
        LBM_DIMS
    )
    pivot = pivot[APPROACHES]
    cols = [APPROACH_LABELS[c] for c in pivot.columns]
    header = "| dim | " + " | ".join(cols) + " | argmax |"
    sep = "|" + "|".join(["---"] * (len(cols) + 2)) + "|"
    lines = [header, sep]
    for dim in pivot.index:
        winner_app = pivot.loc[dim].idxmax()
        winner_label = APPROACH_LABELS[winner_app]
        cells = [dim] + [f"{pivot.loc[dim, c]:.0f}" for c in pivot.columns] + [winner_label]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- Top-level driver --------------------------------------------------------


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("== Loading LBM target ==")
    lbm = load_lbm_target()
    print(f"   {len(lbm):,} hexes with LBM data, dims={LBM_DIMS}")

    full_sidecar: dict = {
        "lbm_year": 2022,
        "canonical_k": CANON_K,
        "_meta": {
            "k_choice_rationale": (
                "Spec asked for k=10 but cluster_results contain only k in {5, 8, 12}. "
                "Selected k=8: best silhouette for ring_agg (0.228) and concat_zscore "
                "(0.241), closest to spec's k=10 framing."
            ),
            "lbm_zscore_note": (
                "lbm is on its own scale (~3.42-5.04). Z-scored independently as `lbm_z` "
                "before stacking with already-z-scored sub-scores for 6D distance work."
            ),
            "kmeans_seed": RNG,
        },
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

    for app in APPROACHES:
        print(f"\n-- {app} --")
        t0 = time.time()
        joined = join_clusters_lbm(app, CANON_K, lbm)
        print(f"   joined rows: {len(joined):,}")
        sig = compute_signature_table(joined, CANON_K)

        sig.to_csv(OUT_DIR / f"{app}_signature_table.csv", index=False)
        (OUT_DIR / f"{app}_signature_table.md").write_text(
            signature_to_markdown(sig), encoding="utf-8"
        )
        plot_multiscore_violins(joined, app, OUT_DIR / f"{app}_multiscore_violins.png")
        D = plot_distance_matrix(joined, app, OUT_DIR / f"{app}_score_distance_matrix.png")
        plot_dendrogram(joined, app, OUT_DIR / f"{app}_score_dendrogram.png")

        per_cluster = {}
        for _, r in sig.iterrows():
            entry = {
                "n_hex": int(r["n_hex"]),
                "rank_by_lbm": int(r["rank_by_lbm"]),
            }
            for d in LBM_DIMS:
                entry[d] = {
                    "mean": float(r[f"{d}_mean"]),
                    "std": float(r[f"{d}_std"]),
                    "p_rank": float(r[f"{d}_prank"]),
                    "n": int(r["n_hex"]),
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
        print(f"   ({time.time()-t0:.1f}s) signature + 3 figures done")

    print("\n-- Cross-embedding F-stat --")
    fstat_df = compute_cross_embedding_fstat(lbm)
    fstat_df.to_csv(OUT_DIR / "cross_embedding_fstat.csv", index=False)
    (OUT_DIR / "cross_embedding_fstat.md").write_text(
        fstat_to_markdown(fstat_df), encoding="utf-8"
    )
    plot_fstat_heatmap(fstat_df, OUT_DIR / "cross_embedding_fstat.png")

    pivot = fstat_df.pivot(index="dim", columns="embedding", values="f_stat").reindex(
        LBM_DIMS
    )[APPROACHES]
    winners = {dim: pivot.loc[dim].idxmax() for dim in pivot.index}
    full_sidecar["cross_embedding_fstat"] = fstat_df.to_dict("records")
    full_sidecar["cross_embedding_fstat_winners"] = winners

    sidecar_fp = OUT_DIR / "leefbaarometer_per_cluster_full.json"
    sidecar_fp.write_text(json.dumps(full_sidecar, indent=2), encoding="utf-8")
    print(f"   wrote sidecar: {sidecar_fp}")

    # Summary print
    print("\n== F-stat winners per dim ==")
    win_counts: dict[str, int] = {a: 0 for a in APPROACHES}
    for dim, app in winners.items():
        print(f"   {dim:<3}  -> {APPROACH_LABELS[app]}")
        win_counts[app] += 1
    print("\n== Headline winner counts ==")
    for app in sorted(win_counts, key=lambda a: -win_counts[a]):
        print(f"   {win_counts[app]}/{len(LBM_DIMS)}  {APPROACH_LABELS[app]}")

    elapsed = time.time() - t_start
    print(f"\n== DONE in {elapsed:.1f}s ==")
    print(f"   outputs in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
