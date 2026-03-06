"""
Diagnostic visualizations for res9 2022 Stage 1 embeddings.

Lifetime: temporary (one-off diagnostic)
Stage: 3 (analysis/visualization of stage1 outputs)

Produces 6 plot types: coverage overlap, PCA spatial maps (hex2vec, highway2vec),
leefbaarometer spatial map, feature distributions (KDE), explained variance curves.
All plots use rasterization and centroid-based spatial rendering for speed.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import h3

# Paths
BASE = r"C:\Users\Bert Berkers\PycharmProjects\UrbanRepML"
DATA = os.path.join(BASE, "data", "study_areas", "netherlands")
OUT = os.path.join(DATA, "stage1_unimodal", "plots", "res9_diagnostics")
os.makedirs(OUT, exist_ok=True)

DPI = 150
plt.style.use('seaborn-v0_8-darkgrid')

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")

ae = pd.read_parquet(os.path.join(DATA, "stage1_unimodal/alphaearth/netherlands_res9_2022.parquet"))
ae = ae.set_index("h3_index")
ae.index.name = "region_id"
ae_emb_cols = [c for c in ae.columns if c.startswith("A")]

poi_count = pd.read_parquet(os.path.join(DATA, "stage1_unimodal/poi/netherlands_res9_2022.parquet"))
poi_h2v = pd.read_parquet(os.path.join(DATA, "stage1_unimodal/poi/hex2vec/netherlands_res9_2022.parquet"))
roads = pd.read_parquet(os.path.join(DATA, "stage1_unimodal/roads/netherlands_res9_2022.parquet"))
lbm = pd.read_parquet(os.path.join(DATA, "target/leefbaarometer/leefbaarometer_h3res9_2022.parquet"))

print(f"  AlphaEarth: {len(ae):,} hexagons, {len(ae_emb_cols)} dims")
print(f"  POI count:  {len(poi_count):,} hexagons, {poi_count.shape[1]} features")
print(f"  POI hex2vec:{len(poi_h2v):,} hexagons, {poi_h2v.shape[1]} dims")
print(f"  Roads:      {len(roads):,} hexagons, {roads.shape[1]} dims")
print(f"  LBM:        {len(lbm):,} hexagons, {lbm.shape[1]} cols")


# ── Helper: get lat/lng arrays from h3 index ──────────────────────────────
def get_centroids(index):
    """Return (lat, lng) arrays from h3 index using h3.cell_to_latlng."""
    lats, lngs = [], []
    for h in index:
        lat, lng = h3.cell_to_latlng(h)
        lats.append(lat)
        lngs.append(lng)
    return np.array(lats), np.array(lngs)


# ══════════════════════════════════════════════════════════════════════════
# PLOT 1: Coverage overlap bar chart
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/6] Coverage overlap...")

sets = {
    "AlphaEarth": set(ae.index),
    "POI (count)": set(poi_count.index),
    "POI (hex2vec)": set(poi_h2v.index),
    "Roads": set(roads.index),
    "Leefbaarometer": set(lbm.index),
}
all_hexes = set().union(*sets.values())

# Membership matrix
modality_names = list(sets.keys())
membership = np.zeros((len(all_hexes), len(modality_names)), dtype=bool)
all_hexes_list = list(all_hexes)
for j, name in enumerate(modality_names):
    s = sets[name]
    for i, h in enumerate(all_hexes_list):
        membership[i, j] = h in s

# Per-modality counts
counts_per_mod = {name: len(s) for name, s in sets.items()}

# Pairwise overlap counts
n_mod = len(modality_names)
overlap_matrix = np.zeros((n_mod, n_mod), dtype=int)
for i in range(n_mod):
    for j in range(n_mod):
        overlap_matrix[i, j] = int(np.sum(membership[:, i] & membership[:, j]))

# How many modalities per hex
n_modalities_per_hex = membership.sum(axis=1)
coverage_dist = {k: int((n_modalities_per_hex == k).sum()) for k in range(1, n_mod + 1)}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel A: per-modality count
ax = axes[0]
colors_mod = plt.cm.Set2(np.linspace(0, 1, n_mod))
bars = ax.barh(modality_names, [counts_per_mod[n] for n in modality_names], color=colors_mod)
ax.set_xlabel("Number of hexagons")
ax.set_title("Hexagons per modality")
for bar, val in zip(bars, [counts_per_mod[n] for n in modality_names]):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
            f"{val:,}", va='center', fontsize=9)

# Panel B: overlap heatmap
ax = axes[1]
im = ax.imshow(overlap_matrix, cmap='YlOrRd')
ax.set_xticks(range(n_mod))
ax.set_yticks(range(n_mod))
short_names = ["AE", "POI-cnt", "POI-h2v", "Roads", "LBM"]
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short_names, fontsize=9)
for i in range(n_mod):
    for j in range(n_mod):
        ax.text(j, i, f"{overlap_matrix[i,j]//1000}k", ha='center', va='center', fontsize=7,
                color='white' if overlap_matrix[i,j] > overlap_matrix.max()*0.6 else 'black')
ax.set_title("Pairwise overlap (hexagons)")
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel C: coverage depth distribution
ax = axes[2]
depths = sorted(coverage_dist.keys())
vals = [coverage_dist[d] for d in depths]
ax.bar(depths, vals, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(depths))))
ax.set_xlabel("Number of modalities present")
ax.set_ylabel("Number of hexagons")
ax.set_title("Coverage depth")
ax.set_xticks(depths)
for d, v in zip(depths, vals):
    ax.text(d, v + 2000, f"{v:,}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_coverage_overlap.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved 01_coverage_overlap.png")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 2: Hex2vec PCA spatial map (RGB from first 3 PCs)
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/6] Hex2vec PCA spatial map...")

h2v_cols = [c for c in poi_h2v.columns if c.startswith("hex2vec_")]
h2v_vals = poi_h2v[h2v_cols].values

pca_h2v = PCA(n_components=3, random_state=42)
pcs_h2v = pca_h2v.fit_transform(h2v_vals)
# Normalize each PC to [0, 1] for RGB
scaler = MinMaxScaler()
rgb_h2v = scaler.fit_transform(pcs_h2v)

# Subsample for speed: use all if < 500k, else sample
MAX_POINTS = 500_000
if len(poi_h2v) > MAX_POINTS:
    rng = np.random.RandomState(42)
    idx = rng.choice(len(poi_h2v), MAX_POINTS, replace=False)
    idx.sort()
    plot_index = poi_h2v.index[idx]
    plot_rgb = rgb_h2v[idx]
else:
    plot_index = poi_h2v.index
    plot_rgb = rgb_h2v

lats, lngs = get_centroids(plot_index)

fig, ax = plt.subplots(figsize=(10, 12))
ax.scatter(lngs, lats, c=plot_rgb, s=0.05, marker='.', rasterized=True, linewidths=0)
ax.set_aspect('equal')
ax.set_title(f"Hex2vec PCA (RGB = PC1/2/3)\n"
             f"Var explained: {pca_h2v.explained_variance_ratio_[:3].sum():.1%} "
             f"({pca_h2v.explained_variance_ratio_[0]:.1%}, "
             f"{pca_h2v.explained_variance_ratio_[1]:.1%}, "
             f"{pca_h2v.explained_variance_ratio_[2]:.1%})")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_hex2vec_pca_spatial.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved 02_hex2vec_pca_spatial.png")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 3: Highway2vec PCA spatial map
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/6] Highway2vec PCA spatial map...")

road_cols = [c for c in roads.columns if c.startswith("R")]
road_vals = roads[road_cols].values

pca_roads = PCA(n_components=3, random_state=42)
pcs_roads = pca_roads.fit_transform(road_vals)
rgb_roads = MinMaxScaler().fit_transform(pcs_roads)

lats_r, lngs_r = get_centroids(roads.index)

fig, ax = plt.subplots(figsize=(10, 12))
ax.scatter(lngs_r, lats_r, c=rgb_roads, s=0.1, marker='.', rasterized=True, linewidths=0)
ax.set_aspect('equal')
ax.set_title(f"Highway2vec PCA (RGB = PC1/2/3)\n"
             f"Var explained: {pca_roads.explained_variance_ratio_[:3].sum():.1%} "
             f"({pca_roads.explained_variance_ratio_[0]:.1%}, "
             f"{pca_roads.explained_variance_ratio_[1]:.1%}, "
             f"{pca_roads.explained_variance_ratio_[2]:.1%})")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_highway2vec_pca_spatial.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved 03_highway2vec_pca_spatial.png")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 4: Leefbaarometer spatial map
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/6] Leefbaarometer spatial map...")

lbm_score = lbm['lbm'].values
lats_l, lngs_l = get_centroids(lbm.index)

fig, ax = plt.subplots(figsize=(10, 12))
# Clip outliers for better color range
vmin, vmax = np.percentile(lbm_score, [2, 98])
sc = ax.scatter(lngs_l, lats_l, c=lbm_score, s=0.3, marker='.', rasterized=True,
                linewidths=0, cmap='RdYlGn', vmin=vmin, vmax=vmax)
ax.set_aspect('equal')
ax.set_title(f"Leefbaarometer (LBM) overall score\n"
             f"N={len(lbm):,} hexagons, median={np.median(lbm_score):.3f}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
cbar = plt.colorbar(sc, ax=ax, shrink=0.7, label="LBM score")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_leefbaarometer_spatial.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved 04_leefbaarometer_spatial.png")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 5: Feature distributions (KDE)
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/6] Feature distributions (KDE)...")

from scipy.stats import gaussian_kde

def kde_plot(ax, data, label, color):
    """Plot KDE of data on axis, handling edge cases."""
    data = data[np.isfinite(data)]
    if len(data) < 10:
        return
    # Subsample for KDE speed
    if len(data) > 50000:
        data = np.random.RandomState(42).choice(data, 50000, replace=False)
    try:
        kde = gaussian_kde(data, bw_method='scott')
        x = np.linspace(np.percentile(data, 0.5), np.percentile(data, 99.5), 200)
        ax.fill_between(x, kde(x), alpha=0.4, color=color, label=label)
        ax.plot(x, kde(x), color=color, linewidth=1)
    except Exception:
        pass

# Row 1: PCA PCs for each embedding type (3 cols x 1 row for each modality)
# Row 2: LBM target variables (6 cols)

fig, axes = plt.subplots(4, 3, figsize=(15, 14))

# AlphaEarth PCs
pca_ae = PCA(n_components=3, random_state=42)
pcs_ae = pca_ae.fit_transform(ae[ae_emb_cols].values)
colors_pc = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(3):
    kde_plot(axes[0, i], pcs_ae[:, i],
             f"PC{i+1} ({pca_ae.explained_variance_ratio_[i]:.1%})", colors_pc[i])
    axes[0, i].set_title(f"AlphaEarth PC{i+1}")
    axes[0, i].legend(fontsize=8)

# Hex2vec PCs (already computed)
for i in range(3):
    kde_plot(axes[1, i], pcs_h2v[:, i],
             f"PC{i+1} ({pca_h2v.explained_variance_ratio_[i]:.1%})", colors_pc[i])
    axes[1, i].set_title(f"Hex2vec PC{i+1}")
    axes[1, i].legend(fontsize=8)

# Roads PCs (already computed)
for i in range(3):
    kde_plot(axes[2, i], pcs_roads[:, i],
             f"PC{i+1} ({pca_roads.explained_variance_ratio_[i]:.1%})", colors_pc[i])
    axes[2, i].set_title(f"Highway2vec PC{i+1}")
    axes[2, i].legend(fontsize=8)

# LBM targets (6 variables across bottom row + extra)
lbm_vars = ['lbm', 'fys', 'onv', 'soc', 'vrz', 'won']
lbm_colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
for i, var in enumerate(lbm_vars[:3]):
    kde_plot(axes[3, i], lbm[var].values, var, lbm_colors[i])
    axes[3, i].set_title(f"LBM: {var}")
    axes[3, i].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "05a_distributions_pcs_lbm.png"), dpi=DPI, bbox_inches='tight')
plt.close()

# Second figure for remaining LBM vars
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, var in enumerate(lbm_vars[3:]):
    kde_plot(axes[i], lbm[var].values, var, lbm_colors[i+3])
    axes[i].set_title(f"LBM: {var}")
    axes[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05b_distributions_lbm_remaining.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved 05a/05b distribution plots")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 6: PCA explained variance curves
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/6] PCA explained variance curves...")

fig, ax = plt.subplots(figsize=(10, 6))

modalities_pca = {
    f"AlphaEarth ({len(ae_emb_cols)}D)": (ae[ae_emb_cols].values, '#1f77b4'),
    f"Hex2vec ({len(h2v_cols)}D)": (h2v_vals, '#ff7f0e'),
    f"Highway2vec ({len(road_cols)}D)": (road_vals, '#2ca02c'),
}

for name, (vals, color) in modalities_pca.items():
    n_comp = min(vals.shape[1], 50)
    pca_full = PCA(n_components=n_comp, random_state=42)
    pca_full.fit(vals)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    ax.plot(range(1, n_comp + 1), cumvar, '-o', markersize=3, label=name, color=color, linewidth=2)
    # Mark 90% threshold
    idx90 = np.searchsorted(cumvar, 0.9)
    if idx90 < n_comp:
        ax.axvline(idx90 + 1, color=color, linestyle='--', alpha=0.4, linewidth=1)
        ax.text(idx90 + 1.5, cumvar[idx90] - 0.03, f"90%@{idx90+1}",
                fontsize=8, color=color)

ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.95, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel("Number of principal components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("Embedding dimensionality: PCA explained variance")
ax.legend(fontsize=10)
ax.set_ylim(0, 1.02)
ax.set_xlim(0.5, 50.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_pca_explained_variance.png"), dpi=DPI, bbox_inches='tight')
plt.close()
print("  Saved 06_pca_explained_variance.png")

print(f"\nAll plots saved to: {OUT}")
