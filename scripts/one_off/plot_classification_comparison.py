"""
Generate comparison chart for Linear vs DNN classification probe performance.
Output: docs/images/classification_hierarchical_comparison.png

Design:
- Two-subplot layout: plot above, table below with ALIGNED x-axis
- Color scheme:
  * DNN Accuracy: dark blue (#003E7E), circles, solid line
  * DNN F1 Macro: light blue (#4A90E2), squares, dashed line
  * Linear Accuracy: dark red (#C41E3A), circles, solid line
  * Linear F1 Macro: light red (#E74C3C), squares, dashed line
- Gray bars for n_classes: medium gray (#999999), alpha 0.4-0.5
- Gridlines: major at 0.1 intervals, minor at 0.05 intervals
- Table: columns aligned with plot x-axis (levels 1-7), colored text matching lines
- X-axis alignment: both axes share xlim(0.5, 7.5) so table cells line up with bars/markers
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np

# Read data
linear_csv = "data/study_areas/netherlands/stage3_analysis/classification_probe/2026-02-20_default/metrics_summary.csv"
dnn_csv = "data/study_areas/netherlands/stage3_analysis/dnn_classification_probe/2026-02-20_default/metrics_summary.csv"

linear_df = pd.read_csv(linear_csv)
dnn_df = pd.read_csv(dnn_csv)

# Extract hierarchy levels (1-7)
hierarchy_levels = np.arange(1, 8)

# Extract metrics
linear_acc = linear_df['overall_accuracy'].values
linear_f1 = linear_df['overall_f1_macro'].values
linear_classes = linear_df['n_classes'].values

dnn_acc = dnn_df['overall_accuracy'].values
dnn_f1 = dnn_df['overall_f1_macro'].values
dnn_classes = dnn_df['n_classes'].values

# Use DNN classes for background bars
n_classes = dnn_classes

# Create figure with two subplots: plot above, table below
# Use GridSpec to control height ratios
fig = plt.figure(figsize=(14, 9.5), dpi=150)
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], hspace=0.15)
ax_plot = fig.add_subplot(gs[0])
ax_table = fig.add_subplot(gs[1])

# Color definitions
dnn_acc_color = '#003E7E'      # Dark blue for DNN Accuracy
dnn_f1_color = '#4A90E2'       # Light blue for DNN F1 Macro
linear_acc_color = '#C41E3A'   # Dark red for Linear Accuracy
linear_f1_color = '#E74C3C'    # Light red for Linear F1 Macro
bar_color = '#999999'          # Medium gray for background bars

# === PLOT SECTION ===

# Create secondary y-axis for n_classes
ax_plot2 = ax_plot.twinx()

# Plot background bars (n_classes on right axis) — more visible
bar_width = 0.4
ax_plot2.bar(hierarchy_levels, n_classes, width=bar_width, alpha=0.45,
             color=bar_color, label='Number of Classes', zorder=1)

# Plot lines on primary axis (scores)
# DNN Accuracy: solid dark blue line, filled circles
ax_plot.plot(hierarchy_levels, dnn_acc, marker='o', linestyle='-', linewidth=2.5,
             markersize=8, markerfacecolor=dnn_acc_color, markeredgecolor=dnn_acc_color,
             color=dnn_acc_color, label='DNN Accuracy', zorder=3)

# DNN F1 Macro: dashed light blue line, filled squares
ax_plot.plot(hierarchy_levels, dnn_f1, marker='s', linestyle='--', linewidth=2.5,
             markersize=7.5, markerfacecolor=dnn_f1_color, markeredgecolor=dnn_f1_color,
             color=dnn_f1_color, label='DNN F1 Macro', zorder=3)

# Linear Accuracy: solid dark red line, filled circles
ax_plot.plot(hierarchy_levels, linear_acc, marker='o', linestyle='-', linewidth=2.5,
             markersize=8, markerfacecolor=linear_acc_color, markeredgecolor=linear_acc_color,
             color=linear_acc_color, label='Linear Accuracy', zorder=3)

# Linear F1 Macro: dashed light red line, filled squares
ax_plot.plot(hierarchy_levels, linear_f1, marker='s', linestyle='--', linewidth=2.5,
             markersize=7.5, markerfacecolor=linear_f1_color, markeredgecolor=linear_f1_color,
             color=linear_f1_color, label='Linear F1 Macro', zorder=3)

# Configure primary y-axis (scores) with major and minor gridlines
ax_plot.set_xlabel('Hierarchy Level', fontsize=12, fontweight='bold')
ax_plot.set_ylabel('Score', fontsize=12, fontweight='bold', color='black')
ax_plot.set_ylim(0, 1.0)
ax_plot.set_xlim(0.5, 7.5)
ax_plot.set_xticks(hierarchy_levels)

# Major gridlines at 0.1 intervals
ax_plot.set_yticks(np.arange(0, 1.1, 0.1))
ax_plot.grid(axis='y', which='major', alpha=0.35, linestyle='-', linewidth=0.9,
             zorder=0, color='#CCCCCC')

# Minor gridlines at 0.05 intervals
ax_plot.set_yticks(np.arange(0, 1.1, 0.05), minor=True)
ax_plot.grid(axis='y', which='minor', alpha=0.15, linestyle='-', linewidth=0.5,
             zorder=0, color='#DDDDDD')

ax_plot.tick_params(axis='y', labelcolor='black')
ax_plot.set_axisbelow(True)

# Configure secondary y-axis (n_classes)
ax_plot2.set_ylabel('Number of Classes', fontsize=12, fontweight='bold', color='#777777')
ax_plot2.tick_params(axis='y', labelcolor='#777777')
max_classes = max(n_classes) + 15
ax_plot2.set_ylim(0, max_classes)

# Title and subtitle
fig.suptitle('Hierarchical Classification: Linear vs DNN Probes',
             fontsize=14, fontweight='bold', y=0.97)
ax_plot.text(0.5, 0.93, 'Performance vs taxonomy granularity',
             transform=fig.transFigure, ha='center', fontsize=11,
             style='italic', color='#666666')

# Legend in upper right corner of plot
lines1, labels1 = ax_plot.get_legend_handles_labels()
lines2, labels2 = ax_plot2.get_legend_handles_labels()
ax_plot.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10,
               framealpha=0.96, edgecolor='#CCCCCC', fancybox=False)

# === TABLE SECTION ===
# Key: ax_table shares the SAME x-limits as ax_plot for data columns (1-7).
# Row labels extend to the left via clip_on=False.

# Total rows: 1 header + 5 data = 6
n_rows = 6
row_height = 1.0

# Set up table axes — same x range as plot, y range fits all rows
ax_table.set_xlim(0.5, 7.5)  # CRITICAL: match ax_plot xlim
ax_table.set_ylim(-0.5, n_rows + 0.5)

# Remove all spines and ticks
for spine in ax_table.spines.values():
    spine.set_visible(False)
ax_table.set_xticks([])
ax_table.set_yticks([])

# Row y-positions (top to bottom): header at top, data rows descending
# header_y is the bottom edge of the header row
header_y = n_rows - 1  # = 5

# --- Header row ---
# Label cell (left side, extends beyond xlim)
label_rect = Rectangle((-1.3, header_y), 1.6, row_height,
                       linewidth=0.8, edgecolor='#555555', facecolor='#4A4A4A',
                       clip_on=False, zorder=2)
ax_table.add_patch(label_rect)
ax_table.text(-0.5, header_y + 0.5, 'Metric', fontsize=9, fontweight='bold',
             ha='center', va='center', color='white', clip_on=False)

# Header data cells for levels 1-7
for level in hierarchy_levels:
    rect = Rectangle((level - 0.45, header_y), 0.9, row_height,
                     linewidth=0.8, edgecolor='#555555', facecolor='#4A4A4A', zorder=2)
    ax_table.add_patch(rect)
    ax_table.text(level, header_y + 0.5, f'Level {level}', fontsize=9, fontweight='bold',
                 ha='center', va='center', color='white')

# --- Data rows ---
metric_rows = [
    ('n_classes',        [str(int(n)) for n in n_classes],       '#333333'),
    ('DNN Accuracy',     [f'{v:.3f}' for v in dnn_acc],          dnn_acc_color),
    ('DNN F1 Macro',     [f'{v:.3f}' for v in dnn_f1],           dnn_f1_color),
    ('Linear Accuracy',  [f'{v:.3f}' for v in linear_acc],       linear_acc_color),
    ('Linear F1 Macro',  [f'{v:.3f}' for v in linear_f1],        linear_f1_color),
]

for row_idx, (metric_name, values, text_color) in enumerate(metric_rows):
    row_y = header_y - (row_idx + 1) * row_height
    bg_color = '#F0F0F0' if row_idx % 2 == 0 else '#FAFAFA'

    # Metric label cell (extends left of xlim)
    label_rect = Rectangle((-1.3, row_y), 1.6, row_height,
                           linewidth=0.5, edgecolor='#CCCCCC', facecolor='#E0E0E0',
                           clip_on=False, zorder=2)
    ax_table.add_patch(label_rect)
    ax_table.text(-0.5, row_y + 0.5, metric_name, fontsize=9, fontweight='bold',
                 ha='center', va='center', color=text_color, clip_on=False)

    # Value cells at x = 1, 2, ..., 7
    for level, value in zip(hierarchy_levels, values):
        cell_rect = Rectangle((level - 0.45, row_y), 0.9, row_height,
                              linewidth=0.5, edgecolor='#DDDDDD', facecolor=bg_color, zorder=2)
        ax_table.add_patch(cell_rect)
        ax_table.text(level, row_y + 0.5, value, fontsize=9,
                     ha='center', va='center', color=text_color, fontweight='medium')

# Save figure
output_path = "docs/images/classification_hierarchical_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Chart saved to {output_path}")
plt.close()
