#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pure SRAI visualization for Cascadia Coastal Forests
Uses SRAI's native plotting functions designed for H3 data
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from srai.regionalizers import H3Regionalizer
from srai.plotting import plot_regions
from shapely.geometry import Polygon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_pure_srai_plots():
    """Create plots using pure SRAI functionality"""
    
    # Load corrected dataset
    data_path = Path("data/h3_2021_res8_coastal_forests/cascadia_coastal_forests_2021_res8_final.parquet")
    logger.info(f"Loading corrected data from {data_path}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} hexagons")
    
    # Create H3 regionalizer
    regionalizer = H3Regionalizer(resolution=8)
    
    # Create SRAI H3 regions using the regionalizer
    logger.info("Creating SRAI H3 regions...")
    
    # Use SRAI to get the actual H3 geometries for the study area
    bbox_polygon = Polygon([
        (-124.7, 38.5), (-121.0, 38.5), (-121.0, 43.5), (-124.7, 43.5), (-124.7, 38.5)
    ])
    area_gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs='EPSG:4326')
    full_regions = regionalizer.transform(area_gdf)
    
    logger.info(f"SRAI created {len(full_regions):,} total H3 regions")
    
    # Get only the regions we have data for
    data_h3_indices = set(df['h3_index'].unique())
    regions_gdf = full_regions.loc[full_regions.index.intersection(data_h3_indices)].copy()
    logger.info(f"Using {len(regions_gdf):,} H3 regions with data")
    
    # Output directory
    output_dir = Path("plots/coastal_2021/srai_native")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load available clustering results
    assignments_dir = Path("results/coastal_2021/assignments")
    if not assignments_dir.exists():
        logger.error("No clustering results found! Run clustering first.")
        return
    
    clustering_files = list(assignments_dir.glob("*.parquet"))
    logger.info(f"Found {len(clustering_files)} clustering results")
    
    # Create plots for each clustering method
    for i, cluster_file in enumerate(clustering_files[:5]):  # First 5 methods
        method_name = cluster_file.stem
        logger.info(f"\nCreating SRAI plot {i+1}/{min(5, len(clustering_files))}: {method_name}")
        
        try:
            # Load cluster assignments
            clusters_df = pd.read_parquet(cluster_file)
            if 'h3_index' in clusters_df.columns:
                clusters_df = clusters_df.set_index('h3_index')
            
            # Add cluster data to regions
            plotting_gdf = regions_gdf.copy()
            plotting_gdf['cluster'] = clusters_df.loc[plotting_gdf.index, 'cluster']
            
            logger.info(f"  Plotting {len(plotting_gdf):,} regions with {plotting_gdf['cluster'].nunique()} clusters")
            
            # Use SRAI's native plot_regions function - creates interactive Folium map
            logger.info(f"  Creating interactive Folium map...")
            
            # SRAI plot_regions is designed specifically for H3 data and creates clean visualizations
            folium_map = plot_regions(
                regions_gdf=plotting_gdf,
                tiles_style='OpenStreetMap',
                show_borders=False,  # Clean appearance without hexagon borders
                colormap=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                         '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
            )
            
            # Save interactive HTML map
            output_path = output_dir / f"{method_name}_srai_interactive.html"
            folium_map.save(str(output_path))
            logger.info(f"  ✅ Saved interactive map to {output_path}")
            
        except Exception as e:
            logger.error(f"  ❌ Failed {method_name}: {e}")
            continue
    
    logger.info(f"\n🎉 SRAI native plotting complete!")
    logger.info(f"Clean visualizations saved to: {output_dir}")

if __name__ == "__main__":
    create_pure_srai_plots()