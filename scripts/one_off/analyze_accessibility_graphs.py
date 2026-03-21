"""Accessibility graph analysis and visualization.

Lifetime: temporary (one-off analysis of walk/bike/drive accessibility graphs).
Stage: 3 (post-hoc analysis of stage2 accessibility graph outputs).

Produces:
  - Summary stats (printed + written to reports/accessibility_graph_analysis.md)
  - Histograms of travel time and gravity weight (walk_res9)
  - Rasterized maps of node degree and mean gravity weight (walk_res9)
  - Cross-mode comparison bar charts
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.spatial_db import SpatialDB

# ── Paths ──────────────────────────────────────────────────────────────
ACC_DIR = PROJECT_ROOT / "data" / "study_areas" / "netherlands" / "accessibility"
FIG_DIR = PROJECT_ROOT / "reports" / "figures" / "accessibility" / "2026-03-21"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = PROJECT_ROOT / "reports" / "accessibility_graph_analysis.md"

GRAPHS = {
    "walk_res9": {"file": "walk_res9.parquet", "res": 9},
    "bike_res8": {"file": "bike_res8.parquet", "res": 8},
    "drive_res7": {"file": "drive_res7.parquet", "res": 7},
}

# ── Load all graphs ───────────────────────────────────────────────────
print("Loading graphs...")
dfs = {}
for name, meta in GRAPHS.items():
    path = ACC_DIR / meta["file"]
    df = pd.read_parquet(path)
    dfs[name] = df
    print(f"  {name}: {len(df):,} edges, columns={list(df.columns)}")

# ── 1. Basic stats ────────────────────────────────────────────────────
report_lines = ["# Accessibility Graph Analysis (2026-03-21)", ""]

all_stats = {}

for name, df in dfs.items():
    res = GRAPHS[name]["res"]
    print(f"\n{'='*60}")
    print(f"  {name} (res{res})")
    print(f"{'='*60}")

    n_edges = len(df)
    all_nodes = set(df["origin_hex"].unique()) | set(df["dest_hex"].unique())
    n_nodes = len(all_nodes)

    # Degree stats via networkx
    G = nx.from_pandas_edgelist(df, "origin_hex", "dest_hex", create_using=nx.DiGraph())
    degrees = dict(G.degree())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    deg_vals = np.array(list(degrees.values()))

    # Connected components (on undirected version)
    G_undir = G.to_undirected()
    components = list(nx.connected_components(G_undir))
    n_components = len(components)
    largest_cc = max(len(c) for c in components)
    isolated = sum(1 for d in deg_vals if d == 0)

    # Travel time stats
    tt = df["travel_time_s"]
    tt_stats = {
        "min": tt.min(), "max": tt.max(), "mean": tt.mean(),
        "median": tt.median(), "std": tt.std(),
        "p5": tt.quantile(0.05), "p25": tt.quantile(0.25),
        "p75": tt.quantile(0.75), "p95": tt.quantile(0.95),
    }

    # Gravity weight stats
    gw = df["gravity_weight"]
    gw_stats = {
        "min": gw.min(), "max": gw.max(), "mean": gw.mean(),
        "median": gw.median(), "std": gw.std(),
        "p5": gw.quantile(0.05), "p25": gw.quantile(0.25),
        "p75": gw.quantile(0.75), "p95": gw.quantile(0.95),
        "pct_zero": (gw == 0).mean() * 100,
    }

    stats = {
        "n_edges": n_edges, "n_nodes": n_nodes,
        "mean_degree": deg_vals.mean(), "median_degree": np.median(deg_vals),
        "max_degree": deg_vals.max(), "min_degree": deg_vals.min(),
        "n_components": n_components, "largest_cc": largest_cc,
        "isolated": isolated,
        "tt": tt_stats, "gw": gw_stats,
    }
    all_stats[name] = stats

    # Print
    print(f"  Edges: {n_edges:,}")
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Degree: mean={deg_vals.mean():.1f}, median={np.median(deg_vals):.0f}, "
          f"max={deg_vals.max()}, min={deg_vals.min()}")
    print(f"  Components: {n_components:,} (largest: {largest_cc:,})")
    print(f"  Isolated nodes: {isolated}")
    print(f"  Travel time (s): min={tt_stats['min']:.0f}, max={tt_stats['max']:.0f}, "
          f"mean={tt_stats['mean']:.0f}, median={tt_stats['median']:.0f}, "
          f"p5={tt_stats['p5']:.0f}, p95={tt_stats['p95']:.0f}")
    print(f"  Gravity weight: min={gw_stats['min']:.4f}, max={gw_stats['max']:.4f}, "
          f"mean={gw_stats['mean']:.4f}, median={gw_stats['median']:.4f}, "
          f"pct_zero={gw_stats['pct_zero']:.1f}%")

    # Write to report
    report_lines.extend([
        f"## {name} (H3 resolution {res})", "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Edges | {n_edges:,} |",
        f"| Nodes | {n_nodes:,} |",
        f"| Mean degree | {deg_vals.mean():.1f} |",
        f"| Median degree | {np.median(deg_vals):.0f} |",
        f"| Max degree | {deg_vals.max()} |",
        f"| Connected components | {n_components:,} |",
        f"| Largest component | {largest_cc:,} ({largest_cc/n_nodes*100:.1f}%) |",
        f"| Isolated nodes | {isolated} |",
        "",
        f"### Travel time (seconds)",
        f"| Stat | Value |",
        f"|------|-------|",
    ])
    for k, v in tt_stats.items():
        report_lines.append(f"| {k} | {v:.1f} |")
    report_lines.extend([
        "",
        f"### Gravity weight",
        f"| Stat | Value |",
        f"|------|-------|",
    ])
    for k, v in gw_stats.items():
        report_lines.append(f"| {k} | {v:.6f} |")
    report_lines.append("")

# ── 2a. Walk histograms ──────────────────────────────────────────────
print("\nPlotting walk_res9 histograms...")
walk_df = dfs["walk_res9"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Travel time histogram
axes[0].hist(walk_df["travel_time_s"], bins=100, color="#2196F3", edgecolor="none", alpha=0.85)
axes[0].set_xlabel("Travel time (seconds)")
axes[0].set_ylabel("Edge count")
axes[0].set_title("Walk graph (res9): Travel time distribution")
axes[0].axvline(walk_df["travel_time_s"].median(), color="red", linestyle="--",
                label=f'Median: {walk_df["travel_time_s"].median():.0f}s')
axes[0].legend()

# Gravity weight histogram (log scale)
gw_nonzero = walk_df["gravity_weight"][walk_df["gravity_weight"] > 0]
axes[1].hist(gw_nonzero, bins=100, color="#FF9800", edgecolor="none", alpha=0.85)
axes[1].set_xlabel("Gravity weight")
axes[1].set_ylabel("Edge count (log scale)")
axes[1].set_yscale("log")
axes[1].set_title("Walk graph (res9): Gravity weight distribution (nonzero)")
axes[1].axvline(gw_nonzero.median(), color="red", linestyle="--",
                label=f'Median: {gw_nonzero.median():.4f}')
axes[1].legend()

plt.tight_layout()
fig.savefig(FIG_DIR / "walk_res9_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'walk_res9_distributions.png'}")

# ── 2b. Cross-mode comparison bar chart ──────────────────────────────
print("\nPlotting cross-mode comparison...")
modes = list(all_stats.keys())
labels = ["walk (res9)", "bike (res8)", "drive (res7)"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Edge count
edge_counts = [all_stats[m]["n_edges"] for m in modes]
bars = axes[0].bar(labels, edge_counts, color=["#2196F3", "#4CAF50", "#FF5722"])
axes[0].set_ylabel("Edge count")
axes[0].set_title("Total edges")
for bar, val in zip(bars, edge_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

# Mean degree
mean_degs = [all_stats[m]["mean_degree"] for m in modes]
bars = axes[1].bar(labels, mean_degs, color=["#2196F3", "#4CAF50", "#FF5722"])
axes[1].set_ylabel("Mean degree")
axes[1].set_title("Mean node degree")
for bar, val in zip(bars, mean_degs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# Mean travel time
mean_tt = [all_stats[m]["tt"]["mean"] for m in modes]
bars = axes[2].bar(labels, mean_tt, color=["#2196F3", "#4CAF50", "#FF5722"])
axes[2].set_ylabel("Mean travel time (s)")
axes[2].set_title("Mean travel time")
for bar, val in zip(bars, mean_tt):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / "cross_mode_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'cross_mode_comparison.png'}")

# ── 2c. Walk res9 spatial maps (rasterized) ─────────────────────────
print("\nComputing walk_res9 per-node metrics...")
walk_df = dfs["walk_res9"]

# Node degree (total = in + out)
origin_counts = walk_df["origin_hex"].value_counts()
dest_counts = walk_df["dest_hex"].value_counts()
all_hexes = set(origin_counts.index) | set(dest_counts.index)
degree_series = pd.Series(0, index=list(all_hexes))
degree_series = degree_series.add(origin_counts, fill_value=0).add(dest_counts, fill_value=0)

# Mean incoming gravity weight per destination hex
mean_gw = walk_df.groupby("dest_hex")["gravity_weight"].mean()

print(f"  Unique nodes: {len(all_hexes):,}")
print(f"  Nodes with degree data: {len(degree_series):,}")
print(f"  Nodes with gravity data: {len(mean_gw):,}")

# Use SpatialDB for centroids — rasterize with scatter
print("\nRendering rasterized maps via SpatialDB centroids...")
db = SpatialDB.for_study_area("netherlands")

# Get centroids for all walk nodes
hex_ids = pd.Index(sorted(all_hexes))
cx, cy = db.centroids(hex_ids, resolution=9)

# Build DataFrame for plotting
plot_df = pd.DataFrame({"cx": cx, "cy": cy}, index=hex_ids)
plot_df["degree"] = degree_series.reindex(hex_ids).fillna(0)
plot_df["mean_gw"] = mean_gw.reindex(hex_ids).fillna(0)
# Drop NaN centroids (hexes not in regions_gdf)
plot_df = plot_df.dropna(subset=["cx", "cy"])
print(f"  Plottable nodes: {len(plot_df):,}")

# Map 1: Node degree
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
scatter = ax.scatter(
    plot_df["cx"], plot_df["cy"],
    c=plot_df["degree"], cmap="viridis",
    s=0.05, marker=".", rasterized=True,
    vmin=plot_df["degree"].quantile(0.01),
    vmax=plot_df["degree"].quantile(0.99),
)
plt.colorbar(scatter, ax=ax, label="Degree (edges per node)", shrink=0.7)
ax.set_title("Walk accessibility graph (res9): Node degree")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")
ax.set_facecolor("#1a1a2e")
fig.savefig(FIG_DIR / "walk_res9_degree_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'walk_res9_degree_map.png'}")

# Map 2: Mean gravity weight
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
# Only plot nodes with nonzero gravity
mask = plot_df["mean_gw"] > 0
scatter = ax.scatter(
    plot_df.loc[mask, "cx"], plot_df.loc[mask, "cy"],
    c=plot_df.loc[mask, "mean_gw"], cmap="inferno",
    s=0.05, marker=".", rasterized=True,
    vmin=0,
    vmax=plot_df.loc[mask, "mean_gw"].quantile(0.95),
)
plt.colorbar(scatter, ax=ax, label="Mean incoming gravity weight", shrink=0.7)
ax.set_title("Walk accessibility graph (res9): Mean gravity weight per destination")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")
ax.set_facecolor("#1a1a2e")
fig.savefig(FIG_DIR / "walk_res9_gravity_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {FIG_DIR / 'walk_res9_gravity_map.png'}")

# ── 3. Connectivity check ────────────────────────────────────────────
print("\n=== Connectivity check ===")

# Load total res9 regions
regions_path = PROJECT_ROOT / "data" / "study_areas" / "netherlands" / "regions_gdf" / "res9"
import geopandas as gpd

# Try to find the parquet
region_files = list(regions_path.glob("*.parquet")) if regions_path.exists() else []
if not region_files:
    # Try alternative path
    alt = PROJECT_ROOT / "data" / "study_areas" / "netherlands" / "regions_gdf" / "netherlands_res9.parquet"
    if alt.exists():
        region_files = [alt]

if region_files:
    regions = gpd.read_parquet(region_files[0])
    total_res9 = len(regions)
    walk_nodes = set(walk_df["origin_hex"].unique()) | set(walk_df["dest_hex"].unique())
    covered = len(walk_nodes & set(regions.index))
    coverage_pct = covered / total_res9 * 100

    print(f"  Total res9 hexagons: {total_res9:,}")
    print(f"  Hexagons in walk graph: {len(walk_nodes):,}")
    print(f"  Hexagons in walk graph AND in regions_gdf: {covered:,}")
    print(f"  Coverage: {coverage_pct:.1f}%")

    # Find hexes NOT in walk graph — are they rural/water?
    uncovered = set(regions.index) - walk_nodes
    print(f"  Uncovered hexagons: {len(uncovered):,} ({100 - coverage_pct:.1f}%)")

    report_lines.extend([
        "## Connectivity check (walk_res9 vs total res9 regions)", "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total res9 hexagons | {total_res9:,} |",
        f"| Hexagons in walk graph | {len(walk_nodes):,} |",
        f"| Coverage | {coverage_pct:.1f}% |",
        f"| Uncovered hexagons | {len(uncovered):,} |",
        "",
    ])
else:
    print("  Could not find res9 regions parquet — skipping coverage check")
    report_lines.extend(["## Connectivity check", "", "Skipped: res9 regions not found.", ""])

# ── Write report ─────────────────────────────────────────────────────
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\nReport written to: {REPORT_PATH}")
print(f"Figures saved to: {FIG_DIR}")
print("Done.")
