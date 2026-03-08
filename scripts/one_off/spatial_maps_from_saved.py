"""Generate spatial maps from saved DNN probe predictions.

Lifetime: temporary (one_off)
Stage: 3
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from shapely import get_geometry, get_num_geometries

from utils.spatial_db import SpatialDB
from utils.paths import StudyAreaPaths

STUDY_AREA = "netherlands"
H3_RESOLUTION = 9
BASE = project_root / "data/study_areas/netherlands/stage3_analysis/dnn_probe/2026-03-08_probe_multiscale_comparison"
CONCAT_DIR = BASE / "concat_208D"
UNET_DIR = BASE / "unet_ms_192D"
TARGET_NAMES = {
    "lbm": "Overall Liveability", "fys": "Physical Environment",
    "onv": "Safety", "soc": "Social Cohesion",
    "vrz": "Amenities", "won": "Housing Quality",
}
TARGET_COLS = ["lbm", "fys", "onv", "soc", "vrz", "won"]

# Load boundary
paths = StudyAreaPaths(STUDY_AREA)
boundary_gdf = gpd.read_file(paths.area_gdf_file())
if boundary_gdf.crs is None:
    boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
boundary_gdf = boundary_gdf.to_crs(epsg=28992)
geom = boundary_gdf.geometry.iloc[0]
n_parts = get_num_geometries(geom)
if n_parts > 1:
    euro_geom = max((get_geometry(geom, i) for i in range(n_parts)), key=lambda g: g.area)
    boundary_gdf = gpd.GeoDataFrame(geometry=[euro_geom], crs=boundary_gdf.crs)

ext = boundary_gdf.total_bounds
minx, miny, maxx, maxy = ext
pad = (maxx - minx) * 0.03
extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

db = SpatialDB.for_study_area(STUDY_AREA)


def rasterize_continuous(cx, cy, values, extent, cmap_name="RdBu", vcenter=0.0,
                         vmin=None, vmax=None, W=2000, H=2400):
    minx, miny, maxx, maxy = extent
    mask = (cx >= minx) & (cx <= maxx) & (cy >= miny) & (cy <= maxy)
    cx_m, cy_m, val_m = cx[mask], cy[mask], values[mask]
    px = ((cx_m - minx) / (maxx - minx) * (W - 1)).astype(int)
    py = ((cy_m - miny) / (maxy - miny) * (H - 1)).astype(int)
    np.clip(px, 0, W - 1, out=px)
    np.clip(py, 0, H - 1, out=py)
    if vmin is None:
        vmin = float(np.nanquantile(val_m, 0.02))
    if vmax is None:
        vmax = float(np.nanquantile(val_m, 0.98))
    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap_name)
    rgb = colormap(norm(val_m))[:, :3].astype(np.float32)
    image = np.zeros((H, W, 4), dtype=np.float32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            sy = np.clip(py + dy, 0, H - 1)
            sx = np.clip(px + dx, 0, W - 1)
            image[sy, sx, :3] = rgb
            image[sy, sx, 3] = 1.0
    return image, norm, colormap


# === Spatial improvement map (mean over all 6 targets) ===
print("Computing spatial improvement...")
all_improvements = {}
for t in TARGET_COLS:
    c = pd.read_parquet(CONCAT_DIR / f"predictions_{t}.parquet")
    u = pd.read_parquet(UNET_DIR / f"predictions_{t}.parquet")
    merged = c[["actual", "predicted"]].join(
        u[["predicted"]].rename(columns={"predicted": "u_pred"}), how="inner"
    )
    merged["c_abs"] = np.abs(merged["actual"] - merged["predicted"])
    merged["u_abs"] = np.abs(merged["actual"] - merged["u_pred"])
    all_improvements[t] = merged["c_abs"] - merged["u_abs"]

imp_df = pd.DataFrame(all_improvements)
mean_imp = imp_df.mean(axis=1)
hex_ids = mean_imp.index
cx, cy = db.centroids(hex_ids, resolution=H3_RESOLUTION, crs=28992)

vals = mean_imp.values
vmax_imp = float(np.abs(vals).clip(0, np.quantile(np.abs(vals), 0.98)).max())
if vmax_imp == 0:
    vmax_imp = 0.01

image, norm, colormap = rasterize_continuous(
    cx, cy, vals, extent, cmap_name="RdBu", vcenter=0.0, vmin=-vmax_imp, vmax=vmax_imp
)

fig, ax = plt.subplots(figsize=(12, 14))
fig.set_facecolor("white")
boundary_gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)
ax.imshow(
    image, extent=[extent[0], extent[2], extent[1], extent[3]],
    origin="lower", aspect="equal", interpolation="nearest", zorder=2,
)
ax.set_xlim(extent[0], extent[2])
ax.set_ylim(extent[1], extent[3])
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, shrink=0.7,
             label="|Concat resid| - |UNet resid| (mean 6 targets)\n(+ = UNet better)")
n_u = int((vals > 0).sum())
n_c = int((vals < 0).sum())
ax.set_title(
    f"Spatial Improvement: UNet MS 192D vs Concat 208D\n"
    f"Blue = UNet better ({n_u:,}) | Red = Concat better ({n_c:,})\n"
    f"Mean across 6 targets | {len(vals):,} hexagons",
    fontsize=11, pad=10,
)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
fig.savefig(BASE / "spatial_improvement.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved spatial_improvement.png (UNet better: {n_u:,}, Concat better: {n_c:,})")


# === Prediction + residual maps for lbm and vrz ===
for tc in ["lbm", "vrz"]:
    c = pd.read_parquet(CONCAT_DIR / f"predictions_{tc}.parquet")
    u = pd.read_parquet(UNET_DIR / f"predictions_{tc}.parquet")
    merged = c[["actual", "predicted"]].join(
        u[["predicted"]].rename(columns={"predicted": "u_pred"}), how="inner"
    )
    merged["c_resid"] = merged["actual"] - merged["predicted"]
    merged["u_resid"] = merged["actual"] - merged["u_pred"]
    hids = merged.index
    ccx, ccy = db.centroids(hids, resolution=H3_RESOLUTION, crs=28992)
    tname = TARGET_NAMES.get(tc, tc)

    # Predictions
    vmin_p = float(merged["actual"].quantile(0.02))
    vmax_p = float(merged["actual"].quantile(0.98))
    fig, axes = plt.subplots(1, 2, figsize=(22, 14))
    fig.set_facecolor("white")
    for ax_idx, (col, label) in enumerate([
        ("predicted", "Concat 208D"), ("u_pred", "UNet MS 192D")
    ]):
        ax = axes[ax_idx]
        img, _, cmap_p = rasterize_continuous(
            ccx, ccy, merged[col].values, extent,
            cmap_name="viridis", vcenter=None, vmin=vmin_p, vmax=vmax_p,
        )
        boundary_gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)
        ax.imshow(
            img, extent=[extent[0], extent[2], extent[1], extent[3]],
            origin="lower", aspect="equal", interpolation="nearest", zorder=2,
        )
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_title(f"{label} predictions", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    sm = cm.ScalarMappable(
        cmap=plt.get_cmap("viridis"), norm=plt.Normalize(vmin=vmin_p, vmax=vmax_p)
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.5, label=f"{tname} predicted")
    fig.suptitle(f"Predicted {tname} ({tc})", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(BASE / f"regression_spatial_pred_{tc}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved regression_spatial_pred_{tc}.png")

    # Residuals
    all_resid = np.concatenate([merged["c_resid"].values, merged["u_resid"].values])
    vmax_r = float(np.quantile(np.abs(all_resid), 0.98))
    fig, axes = plt.subplots(1, 2, figsize=(22, 14))
    fig.set_facecolor("white")
    for ax_idx, (col, label) in enumerate([
        ("c_resid", "Concat 208D"), ("u_resid", "UNet MS 192D")
    ]):
        ax = axes[ax_idx]
        img, _, cmap_r = rasterize_continuous(
            ccx, ccy, merged[col].values, extent,
            cmap_name="RdBu", vcenter=0.0, vmin=-vmax_r, vmax=vmax_r,
        )
        boundary_gdf.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)
        ax.imshow(
            img, extent=[extent[0], extent[2], extent[1], extent[3]],
            origin="lower", aspect="equal", interpolation="nearest", zorder=2,
        )
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_title(f"{label} residuals", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    sm = cm.ScalarMappable(
        cmap=plt.get_cmap("RdBu"),
        norm=TwoSlopeNorm(vmin=-vmax_r, vcenter=0, vmax=vmax_r),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.5, label="Residual (actual - predicted)")
    fig.suptitle(f"Residuals: {tname} ({tc})", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(BASE / f"regression_spatial_residual_{tc}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved regression_spatial_residual_{tc}.png")

print("All spatial maps done.")
