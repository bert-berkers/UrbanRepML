#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Static SRAI Visualization for Cascadia Coastal Forests
Following AlphaEarth validation success pattern for lightweight, publication-ready maps
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import logging
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# SRAI imports
try:
    from srai.regionalizers import H3Regionalizer
    SRAI_AVAILABLE = True
except ImportError:
    SRAI_AVAILABLE = False
    raise ImportError("SRAI required for professional H3 visualization")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoastalStaticSRAIVisualizer:
    """Create professional static visualizations using SRAI + GeoPandas following AlphaEarth validation pattern"""
    
    def __init__(self):
        """Initialize the visualizer with AlphaEarth validation settings"""
        
        # Paths
        self.data_path = Path("data/h3_2021_res8_coastal_forests/cascadia_coastal_forests_2021_res8_final.parquet")
        self.results_dir = Path("results/coastal_2021")
        self.output_dir = Path("plots/coastal_2021/srai_static_professional")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # AlphaEarth validation styling
        self.figure_size = (25.6, 14.4)  # Full 1440p canvas
        self.background_color = '#fdfdf9'  # Cream white background
        self.web_dpi = 100  # Web viewing
        self.print_dpi = 300  # Publication quality
        
        # Cascadia coastal bounds (west of -121°)
        self.coastal_bounds = box(-124.7, 38.5, -121.0, 43.5)
        
        # Initialize SRAI regionalizer
        self.regionalizer = H3Regionalizer(resolution=8)
        
        self.data = None
        self.h3_regions_gdf = None
    
    def load_data(self):
        """Load corrected coastal forest dataset"""
        logger.info(f"Loading corrected coastal forest data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(self.data):,} hexagons")
        
        # Ensure h3_index column
        if 'h3_index' in self.data.columns:
            self.data = self.data.set_index('h3_index')
        
        return self.data
    
    def create_h3_geometries(self):
        """Create H3 geometries using SRAI following AlphaEarth validation pattern"""
        logger.info("Creating H3 geometries with SRAI...")
        
        # Create study area GeoDataFrame
        area_gdf = gpd.GeoDataFrame({'geometry': [self.coastal_bounds]}, crs='EPSG:4326')
        
        # Use SRAI to generate all H3 regions for coastal area
        full_regions_gdf = self.regionalizer.transform(area_gdf)
        logger.info(f"SRAI generated {len(full_regions_gdf):,} H3 regions for coastal area")
        
        # Filter to regions with data
        data_indices = set(self.data.index)
        self.h3_regions_gdf = full_regions_gdf.loc[
            full_regions_gdf.index.intersection(data_indices)
        ].copy()
        
        logger.info(f"Using {len(self.h3_regions_gdf):,} H3 regions with data")
        return self.h3_regions_gdf
    
    def load_clustering_results(self, method: str) -> Optional[gpd.GeoDataFrame]:
        """Load clustering results and merge with H3 geometries"""
        
        # Find clustering file
        assignments_path = self.results_dir / f"assignments/{method}.parquet"
        
        if not assignments_path.exists():
            logger.warning(f"Clustering results not found: {assignments_path}")
            return None
        
        logger.info(f"Loading clustering results: {method}")
        clusters_df = pd.read_parquet(assignments_path)
        
        # Set h3_index as index if needed
        if 'h3_index' in clusters_df.columns:
            clusters_df = clusters_df.set_index('h3_index')
        
        # Merge clustering results with H3 geometries
        merged_indices = self.h3_regions_gdf.index.intersection(clusters_df.index)
        
        final_gdf = gpd.GeoDataFrame(
            clusters_df.loc[merged_indices],
            geometry=self.h3_regions_gdf.loc[merged_indices].geometry,
            crs='EPSG:4326'
        )
        
        logger.info(f"Merged {len(final_gdf):,} hexagons with {final_gdf['cluster'].nunique()} clusters")
        return final_gdf
    
    def get_cluster_colors(self, n_clusters: int) -> np.ndarray:
        """Get appropriate colors for cluster count following AlphaEarth validation"""
        
        if n_clusters <= 20:
            # Use tab20 for up to 20 clusters
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        else:
            # For more clusters, combine colormaps like AlphaEarth validation
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.tab20b(np.linspace(0, 1, min(20, n_clusters-20)))
            if n_clusters > 40:
                colors3 = plt.cm.tab20c(np.linspace(0, 1, n_clusters-40))
                colors = np.vstack([colors1, colors2, colors3])
            else:
                colors = np.vstack([colors1, colors2])
        
        return colors
    
    def create_professional_static_map(self, gdf: gpd.GeoDataFrame, method: str):
        """Create professional static map following AlphaEarth validation pattern"""
        
        logger.info(f"Creating professional static map for {method}...")
        
        # Get cluster info
        n_clusters = gdf['cluster'].nunique()
        colors = self.get_cluster_colors(n_clusters)
        
        # Full 1440p canvas following AlphaEarth validation
        fig = plt.figure(figsize=self.figure_size, facecolor=self.background_color)
        
        # Single axis covering ENTIRE page - no margins!
        ax = fig.add_axes([0, 0, 1, 1])
        
        # Plot all clusters with professional styling
        logger.info(f"Plotting {n_clusters} clusters with professional styling...")
        for cluster_id in range(n_clusters):
            cluster_data = gdf[gdf['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                cluster_data.plot(
                    ax=ax,
                    color=colors[cluster_id],
                    alpha=0.85,  # AlphaEarth validation alpha
                    edgecolor='none',  # Clean, no outlines
                    linewidth=0
                )
        
        # Set bounds to data extent
        bounds = gdf.total_bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_aspect('equal')
        
        # AlphaEarth validation clean styling
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Very subtle grid following AlphaEarth validation
        ax.grid(True, alpha=0.02, linestyle='-', linewidth=0.1)
        
        # MINIMAL text - tiny corner info only (AlphaEarth validation style)
        fig.text(0.99, 0.01, f'Cascadia•Coastal•{method}', ha='right', va='bottom', 
                 fontsize=8, fontweight='200', color='#888888', alpha=0.5)
        
        # Save both web and print versions
        method_clean = method.replace('_', '-')
        
        # Web version (100 DPI)
        output_web = self.output_dir / f"{method_clean}_static_web.png"
        plt.savefig(output_web, dpi=self.web_dpi, bbox_inches='tight', pad_inches=0,
                    facecolor=self.background_color, edgecolor='none')
        logger.info(f"Saved web version: {output_web}")
        
        # Print version (300 DPI)
        output_print = self.output_dir / f"{method_clean}_static_print.png"
        plt.savefig(output_print, dpi=self.print_dpi, bbox_inches='tight', pad_inches=0,
                    facecolor=self.background_color, edgecolor='none')
        logger.info(f"Saved print version: {output_print}")
        
        plt.close()
        
        # Save method statistics
        cluster_stats = gdf['cluster'].value_counts().sort_index()
        stats = {
            'method': method,
            'total_hexagons': len(gdf),
            'clusters': n_clusters,
            'coastal_area_km2': len(gdf) * 0.737,  # Approximate H3 res8 area
            'bounds': bounds.tolist(),
            'cluster_size_stats': {
                'mean': float(cluster_stats.mean()),
                'median': float(cluster_stats.median()),
                'min': int(cluster_stats.min()),
                'max': int(cluster_stats.max()),
                'std': float(cluster_stats.std())
            }
        }
        
        stats_path = self.output_dir / f"{method_clean}_stats.json"
        with open(stats_path, 'w') as f:
            import json
            json.dump(stats, f, indent=2)
        
        return output_web, output_print
    
    def create_all_static_maps(self):
        """Create professional static maps for all available clustering results"""
        
        logger.info("="*80)
        logger.info("CREATING PROFESSIONAL STATIC SRAI VISUALIZATIONS")
        logger.info("Following AlphaEarth validation success pattern")
        logger.info("="*80)
        
        # Load data and create geometries
        self.load_data()
        self.create_h3_geometries()
        
        # Find all available clustering results
        assignments_dir = self.results_dir / "assignments"
        if not assignments_dir.exists():
            logger.error(f"Clustering results directory not found: {assignments_dir}")
            return
        
        clustering_files = list(assignments_dir.glob("*.parquet"))
        logger.info(f"Found {len(clustering_files)} clustering result files")
        
        created_maps = []
        
        for cluster_file in clustering_files:
            method = cluster_file.stem
            logger.info(f"\nProcessing {method}...")
            
            try:
                # Load clustering results with H3 geometries
                gdf_clustered = self.load_clustering_results(method)
                
                if gdf_clustered is None:
                    continue
                
                # Create professional static map
                web_path, print_path = self.create_professional_static_map(gdf_clustered, method)
                created_maps.append({
                    'method': method,
                    'web_path': web_path,
                    'print_path': print_path,
                    'clusters': gdf_clustered['cluster'].nunique(),
                    'hexagons': len(gdf_clustered)
                })
                
            except Exception as e:
                logger.error(f"Failed to create map for {method}: {e}")
                continue
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("PROFESSIONAL STATIC VISUALIZATION COMPLETE!")
        logger.info(f"Created {len(created_maps)} high-quality static maps")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*80)
        
        logger.info("\nCreated maps:")
        for map_info in created_maps:
            logger.info(f"  {map_info['method']}: {map_info['clusters']} clusters, {map_info['hexagons']:,} hexagons")
        
        logger.info(f"\nFiles are lightweight (~2-5MB) vs 80MB interactive HTML")
        logger.info(f"Publication-ready quality with AlphaEarth validation styling")
        
        return created_maps


def main():
    """Main execution function"""
    
    if not SRAI_AVAILABLE:
        print("ERROR: SRAI is required for professional H3 visualization")
        return
    
    visualizer = CoastalStaticSRAIVisualizer()
    visualizer.create_all_static_maps()


if __name__ == "__main__":
    main()