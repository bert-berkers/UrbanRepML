# D:\Projects\UrbanRepML\stage2_fusion\analytics.py

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans  # Changed from AgglomerativeClustering to KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, Optional, List
import torch
from datetime import datetime
from shapely.ops import unary_union

from utils import StudyAreaPaths
from utils.paths import write_run_info

logger = logging.getLogger(__name__)

class UrbanEmbeddingAnalyzer:
    """Analytics and visualization for urban embeddings."""

    def __init__(
            self,
            output_dir: Path,
            city_name: str,
            cmap: str = 'tab20b',
            dpi: int = 600,
            figsize: tuple = (12, 12),
            paths: Optional[StudyAreaPaths] = None,
            run_descriptor: str = "default",
    ):
        """
        Initialize the analyzer.

        Args:
            output_dir: Directory for saving results [old 2024]
            city_name: Name of the city being analyzed
            cmap: Matplotlib colormap name
            dpi: Figure resolution
            figsize: Figure size (width, height)
            paths: Optional StudyAreaPaths for centralized path management.
                   If provided, overrides output_dir-based path construction
                   for embeddings and analysis output.
            run_descriptor: Run descriptor for provenance. When non-empty and
                   paths is provided, output goes to a dated run directory
                   under stage3_analysis/analytics/{run_id}/.
        """
        self.city_name = city_name
        self.cmap = cmap
        self.dpi = dpi
        self.figsize = figsize
        self.paths = paths
        self.run_id: Optional[str] = None

        # When paths and run_descriptor are set, route output to a run directory
        if paths is not None and run_descriptor:
            self.run_id = paths.create_run_id(run_descriptor)
            self.output_dir = paths.stage3_run("analytics", self.run_id)
        else:
            self.output_dir = output_dir

        self.plot_dir = self.output_dir / "plots" / city_name
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def save_embeddings(
            self,
            embeddings: Dict[int, torch.Tensor],
            hex_indices_by_res: Dict[int, List[str]],
            modes: Dict[int, str]
    ) -> Dict[int, Path]:
        """
        Save embeddings to parquet files.

        Args:
            embeddings: Dictionary of embeddings by resolution
            hex_indices_by_res: Dictionary of hex indices by resolution
            modes: Dictionary mapping resolutions to transport modes

        Returns:
            Dictionary of saved file paths by resolution
        """
        if self.paths is not None:
            emb_dir = self.paths.model_embeddings(self.city_name)
        else:
            emb_dir = self.output_dir / 'embeddings' / self.city_name
        emb_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for res in embeddings:
            mode = modes[res]
            emb = embeddings[res].detach().cpu().numpy()

            # Validate embeddings
            n_nans = np.isnan(emb).sum()
            if n_nans > 0:
                logger.warning(f"Found {n_nans} NaN values in {mode} embeddings")

            n_inf = np.isinf(emb).sum()
            if n_inf > 0:
                logger.warning(f"Found {n_inf} infinite values in {mode} embeddings")

            # Log statistics
            logger.info(f"Embeddings stats for {mode}:")
            logger.info(f"  Mean: {np.mean(emb):.3f}")
            logger.info(f"  Std: {np.std(emb):.3f}")
            logger.info(f"  Min: {np.min(emb):.3f}")
            logger.info(f"  Max: {np.max(emb):.3f}")

            # Create DataFrame
            df = pd.DataFrame(
                emb,
                index=hex_indices_by_res[res],
                columns=[f'emb_{i}' for i in range(emb.shape[1])]
            )

            # Save to parquet
            output_path = emb_dir / f'urban_embeddings_{mode}_unet.parquet'
            df.to_parquet(output_path)
            saved_paths[res] = output_path
            logger.info(f"Saved embeddings for resolution {res} to {output_path}")

        # Write run-level provenance when using a run directory
        if self.run_id is not None and self.paths is not None:
            write_run_info(
                self.output_dir,
                stage="stage3",
                study_area=self.paths.study_area,
                config={
                    "city_name": self.city_name,
                    "resolutions": list(embeddings.keys()),
                    "cmap": self.cmap,
                    "dpi": self.dpi,
                },
            )
            logger.info(f"Saved run_info.json to {self.output_dir / 'run_info.json'}")

        return saved_paths


    def plot_clusters(
            self,
            area_gdf: gpd.GeoDataFrame,
            regions_by_res: Dict[int, gpd.GeoDataFrame],
            embeddings_by_res: Dict[int, pd.DataFrame],
            n_clusters: Dict[int, int],
            timestamp: Optional[str] = None
    ) -> List[Path]:
        """
        Create separate visualizations of urban clusters for each resolution.

        Args:
            area_gdf: GeoDataFrame with study area boundary
            regions_by_res: Dictionary of region GeoDataFrames by resolution
            embeddings_by_res: Dictionary of embedding DataFrames by resolution
            n_clusters: Dictionary mapping each resolution to the number of clusters
            timestamp: Optional timestamp for filename

        Returns:
            List of Paths to saved visualizations
        """
        try:
            # Add timestamp to title
            timestamp = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M')

            output_paths = []

            # Create plots for each resolution
            for resolution in sorted(regions_by_res.keys()):
                embeddings = embeddings_by_res[resolution]
                regions = regions_by_res[resolution]

                # Filter regions and embeddings to only include valid hexagons
                valid_regions = regions[regions['in_study_area']]
                valid_indices = valid_regions.index
                valid_embeddings = embeddings.loc[valid_indices]

                if valid_embeddings.empty:
                    logger.warning(f"No valid embeddings for resolution {resolution}. Skipping plotting.")
                    continue

                # Determine number of clusters for current resolution
                n_clusters_res = n_clusters.get(resolution, 5)  # Default to 5 if not specified

                # Perform kmeans_clustering_1layer
                kmeans = KMeans(n_clusters=n_clusters_res, random_state=42)
                labels = kmeans.fit_predict(valid_embeddings)

                # Add cluster labels to valid_regions
                valid_regions = valid_regions.copy()
                valid_regions['cluster'] = labels

                # Merge cluster labels back to all regions
                regions = regions.copy()
                regions = regions.merge(
                    valid_regions[['cluster']],
                    how='left',
                    left_index=True,
                    right_index=True
                )

                # Create colormap
                colormap = plt.get_cmap(self.cmap, n_clusters_res)
                custom_cmap = ListedColormap(colormap(np.arange(n_clusters_res)))

                # Create figure
                fig, ax = plt.subplots(figsize=self.figsize, constrained_layout=True)
                fig.suptitle(f'Urban Clustering (KMeans) - {timestamp} - h3.{resolution}', size=16, y=1.02)

                # Plot all hexagons with cluster colors
                regions.plot(
                    column='cluster',
                    ax=ax,
                    cmap=custom_cmap,
                    alpha=0.7,
                )

                # Outline connected in_study_area regions
                built_regions = regions[regions['in_study_area']].copy()
                if not built_regions.empty:
                    built_outline = built_regions.unary_union
                    built_outline_gdf = gpd.GeoDataFrame(geometry=[built_outline], crs=regions.crs)
                    built_outline_gdf = built_outline_gdf.to_crs('EPSG:4326')  # Ensure correct CRS

                    built_outline_gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)

                # Plot the study area boundary
                area_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5)

                ax.set_title(f'Resolution h3.{resolution} - {n_clusters_res} Clusters')
                ax.axis('off')

                # Save output
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.plot_dir / f'urban_clusters_h3_{resolution}_{timestamp_str}.png'

                plt.savefig(
                    output_path,
                    dpi=self.dpi,  # High resolution
                    bbox_inches='tight',
                    pad_inches=0.5
                )
                plt.close()

                output_paths.append(output_path)
                logger.info(f"Successfully saved visualization to {output_path}")

            return output_paths

        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            plt.close()
            raise


    def compute_cluster_statistics(
            self,
            embeddings: Dict[int, np.ndarray],
            regions_by_res: Dict[int, gpd.GeoDataFrame],
            n_clusters: int = 8
    ) -> pd.DataFrame:
        """
        Compute statistics for each cluster at each resolution.

        Args:
            embeddings: Dictionary of embeddings by resolution
            regions_by_res: Dictionary of region GeoDataFrames by resolution
            n_clusters: Number of clusters

        Returns:
            DataFrame with cluster statistics
        """
        stats_list = []

        for res in embeddings:
            emb = embeddings[res]
            regions = regions_by_res[res]

            # Filter valid regions using 'in_study_area'
            valid_regions = regions[regions['in_study_area'] == True]
            valid_embeddings = emb[valid_regions.index]

            # Perform kmeans_clustering_1layer using KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(valid_embeddings)

            # Compute statistics for each cluster
            for cluster in range(n_clusters):
                mask = labels == cluster
                cluster_regions = valid_regions.iloc[mask]

                stats = {
                    'resolution': res,
                    'cluster': cluster,
                    'n_regions': len(cluster_regions),
                    'mean_fsi': cluster_regions['FSI_24'].mean(),
                    'std_fsi': cluster_regions['FSI_24'].std(),
                    'total_area': cluster_regions.to_crs('EPSG:28992').geometry.area.sum(),
                    'mean_area': cluster_regions.to_crs('EPSG:28992').geometry.area.mean()
                }

                stats_list.append(stats)

        stats_df = pd.DataFrame(stats_list)

        # Save statistics
        if self.paths is not None:
            output_path = self.paths.stage3("kmeans_clustering_1layer") / 'cluster_statistics.parquet'
        else:
            output_path = self.output_dir / 'analysis' / self.city_name / 'cluster_statistics.parquet'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_parquet(output_path)

        return stats_df
