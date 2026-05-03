"""
Chapter 5 visibility-fix heatmaps for THE_BOOK v2.

Lifetime: temporary (one-off W4 visibility-fix asset)
Stage: Stage 3 (post-clustering analysis presentation)

Generates one cluster x dim heatmap per embedding (concat_zscore, ring_agg_k10,
supervised_unet_kendall) by reading the W2.A1 signature CSVs in
reports/2026-05-03-book/v2/ch5/. Output: 3 PNGs in same dir, named
{approach}_signature_heatmap.png.

Approach: rows = clusters sorted by lbm_mean descending; columns = 6 LBM dims
(lbm, fys, onv, soc, vrz, won); cell color = z-scored mean (across clusters,
per dim) with diverging RdBu_r colormap centered at 0; cell annotation = raw
mean. NO NL outline (these are not spatial maps).

Runtime: ~5s total.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CH5_DIR = Path(__file__).resolve().parents[2] / "reports" / "2026-05-03-book" / "v2" / "ch5"
APPROACHES = ["concat_zscore", "ring_agg_k10", "supervised_unet_kendall"]
DIMS = ["lbm", "fys", "onv", "soc", "vrz", "won"]
APPROACH_TITLES = {
    "concat_zscore": "Concat z-scored (208D)",
    "ring_agg_k10": "Ring Aggregation (208D)",
    "supervised_unet_kendall": "Supervised UNet Kendall (128D)",
}


def make_heatmap(approach: str) -> Path:
    csv_path = CH5_DIR / f"{approach}_signature_table.csv"
    df = pd.read_csv(csv_path)
    df = df.sort_values("lbm_mean", ascending=False).reset_index(drop=True)

    # raw means per cluster x dim
    means = df[[f"{d}_mean" for d in DIMS]].to_numpy()
    means = np.asarray(means, dtype=float)
    cluster_labels = [f"c{int(c)} (n={int(n):,})" for c, n in zip(df["cluster"], df["n_hex"])]

    # z-score per column (across clusters) for diverging color
    col_means = means.mean(axis=0, keepdims=True)
    col_stds = means.std(axis=0, keepdims=True) + 1e-9
    z = (means - col_means) / col_stds

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.5 * len(df) + 1.4)), dpi=150)
    vmax = float(np.abs(z).max())
    im = ax.imshow(z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # annotate with raw means (so the eye gets the actual signature,
    # while color carries the relative signal)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            txt_color = "white" if abs(z[i, j]) > 0.55 * vmax else "black"
            ax.text(
                j,
                i,
                f"{means[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=txt_color,
            )

    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels(DIMS)
    ax.set_yticks(range(len(cluster_labels)))
    ax.set_yticklabels(cluster_labels)
    ax.set_xlabel("LBM dimension")
    ax.set_ylabel("cluster (sorted by lbm_mean desc)")
    ax.set_title(
        f"{APPROACH_TITLES[approach]} — per-cluster LBM signature\n"
        f"cell color: z-score across clusters (diverging) — annotation: raw mean"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("z-score across clusters (per dim)")
    fig.tight_layout()

    out_path = CH5_DIR / f"{approach}_signature_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    for approach in APPROACHES:
        out = make_heatmap(approach)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
