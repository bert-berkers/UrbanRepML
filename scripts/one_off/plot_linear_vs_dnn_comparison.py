#!/usr/bin/env python3
"""
Recreate Linear vs DNN comparison bar chart with cleaner design.
Reads probe metrics and creates side-by-side R² and RMSE charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define paths
STUDY_AREA = "netherlands"
BASE_PATH = Path(__file__).parent.parent.parent / "data" / "study_areas" / STUDY_AREA / "stage3_analysis"

LINEAR_METRICS = BASE_PATH / "linear_probe" / "2026-02-15_default" / "metrics_summary.csv"
DNN_METRICS = BASE_PATH / "dnn_probe" / "2026-02-15_default" / "metrics_summary.csv"

OUTPUT_PATH = Path(__file__).parent.parent.parent / "docs" / "images" / "dnn_vs_linear_comparison.png"

# Read data
linear_df = pd.read_csv(LINEAR_METRICS)
dnn_df = pd.read_csv(DNN_METRICS)

# Extract target names and metrics
targets = linear_df["target_name"].values
linear_r2 = linear_df["overall_r2"].values
linear_rmse = linear_df["overall_rmse"].values
dnn_r2 = dnn_df["overall_r2"].values
dnn_rmse = dnn_df["overall_rmse"].values

# Calculate differences
r2_diff = dnn_r2 - linear_r2  # Positive = DNN better
rmse_diff = linear_rmse - dnn_rmse  # Positive = DNN better (lower RMSE)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
fig.suptitle("Linear vs DNN Probe: AlphaEarth Embeddings → Leefbaarometer", fontsize=14, fontweight="bold", y=0.98)

# Subplot 1: R² Scores
x_pos = np.arange(len(targets))
bar_width = 0.35
spacing = 0.1

bars1_lin = ax1.bar(x_pos - bar_width/2 - spacing/2, linear_r2, bar_width, label="Linear", color="#1f77b4", alpha=0.85)
bars1_dnn = ax1.bar(x_pos + bar_width/2 + spacing/2, dnn_r2, bar_width, label="DNN (MLP)", color="#ff7f0e", alpha=0.85)

ax1.set_xlabel("Target", fontsize=12, fontweight="bold")
ax1.set_ylabel("R² Score", fontsize=12, fontweight="bold")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(targets, rotation=32, ha="right", fontsize=11)
ax1.set_ylim(0, 0.85)
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.legend(loc="upper left", fontsize=11)

# Add difference labels above R² chart
for i, (lin, dnn, diff) in enumerate(zip(linear_r2, dnn_r2, r2_diff)):
    max_val = max(lin, dnn)
    label_y = max_val + 0.08
    color = "#2ecc71" if diff > 0 else "#e74c3c"
    ax1.text(i, label_y, f"+{diff:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)

# Subplot 2: RMSE Scores
bars2_lin = ax2.bar(x_pos - bar_width/2 - spacing/2, linear_rmse, bar_width, label="Linear", color="#1f77b4", alpha=0.85)
bars2_dnn = ax2.bar(x_pos + bar_width/2 + spacing/2, dnn_rmse, bar_width, label="DNN (MLP)", color="#ff7f0e", alpha=0.85)

ax2.set_xlabel("Target", fontsize=12, fontweight="bold")
ax2.set_ylabel("RMSE", fontsize=12, fontweight="bold")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(targets, rotation=32, ha="right", fontsize=11)
ax2.yaxis.set_major_locator(plt.MultipleLocator(0.01))
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.legend(loc="upper right", fontsize=11)

# Add difference labels above RMSE chart (lower is better, so show as reduction)
for i, (lin, dnn, diff) in enumerate(zip(linear_rmse, dnn_rmse, rmse_diff)):
    max_val = max(lin, dnn)
    label_y = max_val + 0.007
    color = "#2ecc71" if diff > 0 else "#e74c3c"
    ax2.text(i, label_y, f"{diff:+.003f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)

# Add subtitle with cross-validation note
fig.text(0.5, 0.02, "Spatial Block Cross-Validation | AlphaEarth 64-dim embeddings",
         ha="center", fontsize=10, style="italic", color="#555")

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

# Ensure output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150, facecolor="white")
print(f"Chart saved to {OUTPUT_PATH}")

# Print summary
print("\n=== Linear vs DNN Comparison Summary ===")
print(f"{'Target':<25} {'Linear R²':<12} {'DNN R²':<12} {'R² Diff':<12} {'Linear RMSE':<13} {'DNN RMSE':<13} {'RMSE Diff':<13}")
print("-" * 110)
for name, lr2, dr2, rd, lrmse, drmse, rmd in zip(targets, linear_r2, dnn_r2, r2_diff, linear_rmse, dnn_rmse, rmse_diff):
    print(f"{name:<25} {lr2:<12.4f} {dr2:<12.4f} {rd:+.4f} {lrmse:<13.6f} {drmse:<13.6f} {rmd:+.6f}")

print("\nDNN wins (R²):", sum(r2_diff > 0), "/ 6")
print("DNN wins (RMSE):", sum(rmse_diff > 0), "/ 6")
