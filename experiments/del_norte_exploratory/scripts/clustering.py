#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering analysis for 64-dimensional AlphaEarth embeddings.
Implements K-means, Hierarchical, and Gaussian Mixture Models.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import joblib
import json

logger = logging.getLogger(__name__)


class MultiMethodClusterer:
    """Perform clustering using multiple methods on 64-dim AlphaEarth data."""
    
    def __init__(self, config: dict):
        """Initialize clusterer with configuration."""
        self.config = config
        self.clustering_config = config['clustering']
        self.scaler = StandardScaler()
        
        # Setup logging
        logging.basicConfig(
            level=config['output']['log_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def load_h3_data(self, file_path: Path) -> pd.DataFrame:
        """Load H3 hexagon data with 64-dim embeddings."""
        logger.info(f"Loading H3 data from {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} hexagons with {sum(1 for c in df.columns if c.startswith('band_'))} dimensions")
        return df
    
    def prepare_features(self, df: pd.DataFrame, standardize: bool = True) -> np.ndarray:
        """Extract and optionally standardize 64-dim features."""
        # Get band columns
        band_cols = [f'band_{i:02d}' for i in range(64)]
        features = df[band_cols].values
        
        # Handle NaN values
        logger.info(f"Features before cleaning: {features.shape}")
        nan_mask = np.isnan(features).any(axis=1)
        if nan_mask.sum() > 0:
            logger.info(f"Removing {nan_mask.sum()} rows with NaN values")
            features = features[~nan_mask]
            # Also filter the dataframe for later use
            df_clean = df[~nan_mask].copy()
        else:
            df_clean = df.copy()
        
        # Handle infinite values
        inf_mask = np.isinf(features).any(axis=1)
        if inf_mask.sum() > 0:
            logger.info(f"Removing {inf_mask.sum()} rows with infinite values")
            features = features[~inf_mask]
            df_clean = df_clean[~inf_mask]
        
        if standardize:
            logger.info("Standardizing features...")
            features = self.scaler.fit_transform(features)
            
        logger.info(f"Prepared features shape: {features.shape}")
        return features, df_clean
    
    def kmeans_clustering(self, features: np.ndarray) -> Dict[int, Dict]:
        """Perform K-means clustering with different K values."""
        results = {}
        kmeans_config = self.clustering_config['kmeans']
        
        for n_clusters in tqdm(kmeans_config['n_clusters'], desc="K-means clustering"):
            logger.info(f"Running K-means with {n_clusters} clusters...")
            
            # Fit K-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=kmeans_config['n_init'],
                max_iter=kmeans_config['max_iter'],
                random_state=kmeans_config['random_state']
            )
            
            labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            metrics = self.calculate_metrics(features, labels)
            
            # Store results
            results[n_clusters] = {
                'labels': labels,
                'centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'metrics': metrics,
                'model': kmeans
            }
            
            logger.info(f"K={n_clusters}: Silhouette={metrics['silhouette']:.3f}, "
                       f"Davies-Bouldin={metrics['davies_bouldin']:.3f}")
        
        return results
    
    def hierarchical_clustering(self, features: np.ndarray) -> Dict[str, Dict]:
        """Perform hierarchical clustering with different linkage methods."""
        results = {}
        hier_config = self.clustering_config['hierarchical']
        
        for linkage in hier_config['linkage']:
            for n_clusters in tqdm(hier_config['n_clusters'], 
                                  desc=f"Hierarchical-{linkage}"):
                logger.info(f"Running hierarchical clustering ({linkage}) with {n_clusters} clusters...")
                
                # Fit hierarchical clustering
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage
                )
                
                labels = clusterer.fit_predict(features)
                
                # Calculate metrics
                metrics = self.calculate_metrics(features, labels)
                
                # Store results
                key = f"{linkage}_{n_clusters}"
                results[key] = {
                    'labels': labels,
                    'linkage': linkage,
                    'n_clusters': n_clusters,
                    'metrics': metrics,
                    'model': clusterer
                }
                
                logger.info(f"{linkage}-{n_clusters}: Silhouette={metrics['silhouette']:.3f}, "
                           f"Davies-Bouldin={metrics['davies_bouldin']:.3f}")
        
        return results
    
    def gmm_clustering(self, features: np.ndarray) -> Dict[int, Dict]:
        """Perform Gaussian Mixture Model clustering."""
        results = {}
        gmm_config = self.clustering_config['gmm']
        
        for n_components in tqdm(gmm_config['n_components'], desc="GMM clustering"):
            logger.info(f"Running GMM with {n_components} components...")
            
            # Fit GMM
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=gmm_config['covariance_type'],
                max_iter=gmm_config['max_iter'],
                random_state=gmm_config['random_state'],
                n_init=3
            )
            
            labels = gmm.fit_predict(features)
            
            # Calculate metrics
            metrics = self.calculate_metrics(features, labels)
            
            # Add GMM-specific metrics
            metrics['bic'] = gmm.bic(features)
            metrics['aic'] = gmm.aic(features)
            metrics['log_likelihood'] = gmm.score(features)
            
            # Store results
            results[n_components] = {
                'labels': labels,
                'means': gmm.means_,
                'covariances': gmm.covariances_,
                'weights': gmm.weights_,
                'metrics': metrics,
                'model': gmm
            }
            
            logger.info(f"GMM-{n_components}: Silhouette={metrics['silhouette']:.3f}, "
                       f"BIC={metrics['bic']:.1f}, AIC={metrics['aic']:.1f}")
        
        return results
    
    def calculate_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        # Only calculate if we have more than 1 cluster
        n_clusters = len(np.unique(labels))
        if n_clusters > 1:
            metrics['silhouette'] = silhouette_score(features, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
        else:
            metrics['silhouette'] = -1
            metrics['davies_bouldin'] = np.inf
            metrics['calinski_harabasz'] = 0
        
        # Add cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique.tolist(), counts.tolist()))
        metrics['n_clusters'] = n_clusters
        
        return metrics
    
    def save_clustering_results(self, df: pd.DataFrame, results: Dict, method: str):
        """Save clustering results with proper naming convention."""
        output_dir = Path("results/clusters")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for key, result in results.items():
            # Create output filename
            res = self.config['data']['h3_resolution']
            if isinstance(key, int):
                filename = f"{method}_2021_res{res}_k{key}.parquet"
            else:
                filename = f"{method}_2021_res{res}_{key}.parquet"
            
            # Add cluster labels to dataframe
            result_df = df.copy()
            result_df['cluster'] = result['labels']
            
            # Save to parquet
            output_path = output_dir / filename
            result_df.to_parquet(output_path)
            logger.info(f"Saved {method} results to {output_path}")
            
            # Save metrics
            metrics_path = output_path.with_suffix('.json')
            metrics_data = {
                'method': method,
                'config_key': str(key),
                'metrics': result['metrics'],
                'year': 2021,
                'h3_resolution': self.config['data']['h3_resolution'],
                'n_dimensions': 64
            }
            
            # Convert numpy types to Python types for JSON
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                else:
                    return obj
            
            metrics_data = convert_types(metrics_data)
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Optionally save the model
            if self.config['performance']['cache_clusters']:
                res = self.config['data']['h3_resolution']
                model_path = output_dir / f"{method}_2021_res{res}_{key}_model.pkl"
                joblib.dump(result['model'], model_path)
    
    def run_all_clustering(self, df: pd.DataFrame) -> Dict:
        """Run all clustering methods on the data."""
        # Prepare features
        features, df_clean = self.prepare_features(df, standardize=True)
        
        all_results = {}
        
        # K-means
        logger.info("Starting K-means clustering...")
        kmeans_results = self.kmeans_clustering(features)
        self.save_clustering_results(df_clean, kmeans_results, 'kmeans')
        all_results['kmeans'] = kmeans_results
        
        # Hierarchical
        logger.info("Starting hierarchical clustering...")
        hier_results = self.hierarchical_clustering(features)
        self.save_clustering_results(df_clean, hier_results, 'hierarchical')
        all_results['hierarchical'] = hier_results
        
        # GMM
        logger.info("Starting GMM clustering...")
        gmm_results = self.gmm_clustering(features)
        self.save_clustering_results(df_clean, gmm_results, 'gmm')
        all_results['gmm'] = gmm_results
        
        # Save summary statistics
        self.save_summary_stats(all_results)
        
        return all_results
    
    def save_summary_stats(self, all_results: Dict):
        """Save summary statistics for all clustering methods."""
        stats_dir = Path("results/stats")
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'year': 2021,
            'h3_resolution': self.config['data']['h3_resolution'],
            'n_dimensions': 64,
            'methods': {}
        }
        
        for method, results in all_results.items():
            method_summary = {}
            for key, result in results.items():
                method_summary[str(key)] = {
                    'silhouette': result['metrics']['silhouette'],
                    'davies_bouldin': result['metrics']['davies_bouldin'],
                    'calinski_harabasz': result['metrics']['calinski_harabasz'],
                    'n_clusters': result['metrics']['n_clusters']
                }
            summary['methods'][method] = method_summary
        
        # Save summary
        res = self.config['data']['h3_resolution']
        summary_path = stats_dir / f"clustering_summary_2021_res{res}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved clustering summary to {summary_path}")


def main():
    """Main entry point for clustering analysis."""
    import yaml
    from pathlib import Path
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize clusterer
    clusterer = MultiMethodClusterer(config)
    
    # Load H3 data
    data_path = Path(f"data/h3_2021_res11/{config['data']['output_file']}")
    if not data_path.exists():
        logger.error(f"H3 data not found at {data_path}. Run load_alphaearth.py first!")
        return None
    
    df = clusterer.load_h3_data(data_path)
    
    # Run all clustering methods
    logger.info("Starting clustering analysis on 64-dimensional AlphaEarth data...")
    results = clusterer.run_all_clustering(df)
    
    logger.info("Clustering analysis complete!")
    return results


if __name__ == "__main__":
    main()