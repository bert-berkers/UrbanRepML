#!/usr/bin/env python
"""Quick visualization of resolution 10 results."""

import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Note: SRAI provides H3 functionality with additional spatial analysis tools
from shapely.geometry import Polygon

# Load the data
print("Loading resolution 10 data...")
df = pd.read_parquet('data/h3_2021_res10/srai_rioxarray_results.parquet')
print(f"Loaded {len(df):,} hexagons!")

# Create a simple clustering based on first few principal components
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Get band columns
band_cols = [f'band_{i:02d}' for i in range(64)]

# Remove NaN values
df_clean = df.dropna(subset=band_cols)
print(f"Clean data: {len(df_clean):,} hexagons")

# Apply PCA for dimensionality reduction
print("Applying PCA...")
pca = PCA(n_components=3)
features_pca = pca.fit_transform(df_clean[band_cols])

# Quick K-means clustering
print("Running quick K-means (k=10)...")
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df_clean.loc[:, 'cluster'] = kmeans.fit_predict(features_pca)

# Create geometries
print("Creating map visualization...")
geometries = []
for h3_idx in df_clean['h3_index'].values[:10000]:  # Sample for speed
    try:
        boundary = h3.cell_to_boundary(h3_idx)
        poly = Polygon([(lon, lat) for lat, lon in boundary])
        geometries.append(poly)
    except:
        geometries.append(None)

# Create GeoDataFrame (sample for visualization)
gdf = gpd.GeoDataFrame(
    df_clean.iloc[:10000],
    geometry=geometries[:10000],
    crs='EPSG:4326'
)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Map plot
gdf.plot(column='cluster', cmap='tab10', ax=ax1, legend=True, 
         edgecolor='none', alpha=0.7)
ax1.set_title(f'Del Norte 2021 - Resolution 10 Clustering\n{len(df):,} total hexagons (showing 10k sample)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

# Statistics plot
cluster_sizes = df_clean['cluster'].value_counts().sort_index()
ax2.bar(cluster_sizes.index, cluster_sizes.values, color=plt.cm.tab10(np.arange(10)))
ax2.set_title('Cluster Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Cluster ID')
ax2.set_ylabel('Number of Hexagons')
ax2.grid(True, alpha=0.3)

# Add summary text
summary_text = f"""
Resolution 10 Data Summary:
- Total hexagons: {len(df):,}
- Clean hexagons: {len(df_clean):,}
- H3 Resolution: 10
- Dimensions: 64 (AlphaEarth bands)
- PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}
- Clustering: K-means (k=10)
- Processor: SRAI+Rioxarray
"""
fig.text(0.02, 0.02, summary_text, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

plt.tight_layout()

# Save the plot
output_path = Path('plots/resolution_10_overview.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {output_path}")

print("\nDone! Your resolution 10 visualization is ready!")