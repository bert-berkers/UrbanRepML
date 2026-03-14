"""
DNN Probe comparison bar chart: RingAgg vs Concat-PCA vs GCN-UNet vs SAGEConv-UNet.

Purpose: Grouped bar chart showing ring aggregation as the new winner.
Lifetime: temporary
Stage: 3 (analysis/visualization)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# --- Data ---
targets = {
    "Overall\nLiveability": "lbm",
    "Physical\nEnvironment": "fys",
    "Safety": "onv",
    "Social\nCohesion": "soc",
    "Amenities": "vrz",
    "Housing\nQuality": "won",
}

# R-squared values per condition, ordered as specified
# Bar order: RingAgg | gap | Concat-64, Concat-192 | gap | GCN-64, GCN-192 | gap | SAGE-64, SAGE-192
conditions = [
    ("RingAgg-k10 64D",      [0.3040, 0.4450, 0.5230, 0.6600, 0.7740, 0.4860]),
    ("Concat-PCA 64D",       [0.2858, 0.4118, 0.5059, 0.6426, 0.7382, 0.4668]),
    ("Concat-PCA 192D",      [0.2933, 0.4129, 0.5077, 0.6433, 0.7585, 0.4662]),
    ("GCN-UNet 64D",         [0.2305, 0.3080, 0.4934, 0.6470, 0.7424, 0.4545]),
    ("GCN-UNet 192D",        [0.1689, 0.2020, 0.4722, 0.6095, 0.7499, 0.4291]),
    ("SAGEConv-UNet 64D",    [0.1570, 0.1700, 0.4150, 0.5940, 0.5550, 0.4070]),
    ("SAGEConv-UNet 192D",   [0.1750, 0.1840, 0.4380, 0.6050, 0.6730, 0.4250]),
]

# Colors: purple for RingAgg, paired light/dark for each group
colors = [
    "#9b59b6",  # purple/magenta - RingAgg (winner, stands out)
    "#6baed6",  # light blue  - Concat 64D
    "#2171b5",  # dark blue   - Concat 192D
    "#fdae6b",  # light orange - GCN 64D
    "#d94701",  # dark orange  - GCN 192D
    "#a1d99b",  # light green  - SAGE 64D
    "#31a354",  # dark green   - SAGE 192D
]

n_targets = len(targets)
n_conditions = len(conditions)
target_labels = list(targets.keys())

# --- Layout with intra-group gaps ---
# RingAgg is solo | gap | Concat pair | gap | GCN pair | gap | SAGE pair
bar_width = 0.09
inter_gap = 0.06  # gap between groups

# Build offsets: [RingAgg] gap [Concat64, Concat192] gap [GCN64, GCN192] gap [SAGE64, SAGE192]
# Groups: [0], [1,2], [3,4], [5,6]
offsets = []
pos = 0.0
group_boundaries = [0, 1, 3, 5]  # start index of each group

for i in range(n_conditions):
    offsets.append(pos)
    pos += bar_width
    # Add inter-group gap after the last bar of each group
    if i in [0, 2, 4]:  # after RingAgg, after Concat-192, after GCN-192
        pos += inter_gap

offsets = np.array(offsets)
offsets -= offsets.mean()  # center around 0

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("white")

x = np.arange(n_targets)

for i, (label, values) in enumerate(conditions):
    bars = ax.bar(
        x + offsets[i],
        values,
        width=bar_width,
        label=label,
        color=colors[i],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f".{int(val * 1000):03d}" if val < 1 else f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="medium",
            color="#333333",
            rotation=90,
        )

# Axes
ax.set_xticks(x)
ax.set_xticklabels(target_labels, fontsize=11)
ax.set_ylabel("R-squared", fontsize=12, fontweight="medium")
ax.set_ylim(0, 0.89)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

# Grid
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.grid(axis="y", which="minor", alpha=0.15, linewidth=0.3)

# Title
ax.set_title(
    "DNN Probe: Ring Agg vs Concat vs GCN-UNet vs SAGEConv-UNet",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
ax.text(
    0.5, 1.02,
    "(MLP h=256, 3 layers, SiLU, 5-fold spatial block CV, all at 64D except where noted)",
    transform=ax.transAxes,
    ha="center",
    fontsize=10,
    color="#666666",
    style="italic",
)

# Legend -- group visually
legend = ax.legend(
    loc="upper left",
    fontsize=8.5,
    framealpha=0.9,
    edgecolor="#cccccc",
    ncol=4,
    columnspacing=1.2,
    handlelength=1.2,
)

# Subtle vertical separators between target groups on x-axis
for xi in range(1, n_targets):
    ax.axvline(xi - 0.5, color="#e0e0e0", linewidth=0.5, zorder=1)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_alpha(0.3)
ax.spines["bottom"].set_alpha(0.3)

plt.tight_layout()

# --- Save ---
root = Path(r"C:\Users\Bert Berkers\PycharmProjects\UrbanRepML")

out1 = root / "reports" / "figures" / "2026-03-14-unet-vs-concat-comparison.png"
out2 = (
    root
    / "data"
    / "study_areas"
    / "netherlands"
    / "stage3_analysis"
    / "dnn_probe"
    / "2026-03-14_sageconv_comparison"
    / "probe_sageconv_comparison.png"
)

for out in [out1, out2]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

plt.close()
