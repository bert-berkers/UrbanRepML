#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering analysis for Cascadia Coastal Forests dataset.
Processes 160,730 H3 hexagons with 64-dimensional AlphaEarth embeddings.
Saves models, assignments, and metadata for reuse.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import joblib
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoastalForestClusterer:
    """Clustering pipeline for Cascadia Coastal Forest H3 hexagons."""
    
    def __init__(self, data_path: str = None):
        """Initialize clusterer with data path."""
        if data_path is None:
            data_path = "data/h3_2021_res8_coastal_forests/cascadia_coastal_forests_2021_res8_final.parquet"
        
        self.data_path = Path(data_path)
        self.output_dir = Path("results/coastal_2021")
        
        # Create output directories
        for subdir in ['models', 'assignments', 'metadata']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.data = None
        self.features = None
        self.features_scaled = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the 160,730 coastal forest hexagons."""
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_parquet(self.data_path)
        
        # Extract AlphaEarth embeddings (A00-A63)
        embedding_cols = [f'A{i:02d}' for i in range(64)]
        
        # Check if columns exist, otherwise try band_ prefix
        if 'A00' not in self.data.columns:
            embedding_cols = [f'band_{i:02d}' for i in range(64)]
            if 'band_00' not in self.data.columns:
                # Try to find the actual column names
                embedding_cols = [col for col in self.data.columns if col not in ['h3_index', 'geometry']][:64]
        
        logger.info(f"Loaded {len(self.data)} hexagons")
        logger.info(f"Using columns: {embedding_cols[:3]}...{embedding_cols[-3:]}")
        
        self.features = self.data[embedding_cols].values
        
        # Handle NaN values - replace with column means
        col_means = np.nanmean(self.features, axis=0)
        nan_mask = np.isnan(self.features)
        n_nans = nan_mask.sum()
        
        if n_nans > 0:
            logger.warning(f"Found {n_nans} NaN values ({n_nans/(self.features.size)*100:.2f}%), replacing with column means")
            # Replace NaNs with column means
            for col_idx in range(self.features.shape[1]):
                self.features[nan_mask[:, col_idx], col_idx] = col_means[col_idx]
        
        # Check if still have NaNs (in case entire column was NaN)
        if np.isnan(self.features).any():
            logger.warning("Some columns are entirely NaN, replacing with 0")
            self.features = np.nan_to_num(self.features, nan=0.0)
        
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        logger.info(f"Features shape: {self.features.shape}")
        logger.info(f"Features scaled with mean={self.features_scaled.mean():.3f}, std={self.features_scaled.std():.3f}")
        
        return self.data
    
    def compute_metrics(self, labels: np.ndarray) -> Dict:
        """Compute clustering quality metrics."""
        metrics = {}
        
        # Only compute if we have valid clusters
        n_clusters = len(np.unique(labels))
        if n_clusters > 1 and n_clusters < len(labels):
            try:
                metrics['silhouette'] = float(silhouette_score(self.features_scaled, labels, sample_size=10000))
            except:
                metrics['silhouette'] = None
            
            try:
                metrics['davies_bouldin'] = float(davies_bouldin_score(self.features_scaled, labels))
            except:
                metrics['davies_bouldin'] = None
            
            try:
                metrics['calinski_harabasz'] = float(calinski_harabasz_score(self.features_scaled, labels))
            except:
                metrics['calinski_harabasz'] = None
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = {int(k): int(v) for k, v in zip(unique, counts)}
        metrics['n_clusters'] = int(n_clusters)
        
        return metrics
    
    def run_kmeans(self, k_values: List[int] = None):
        """Run K-means clustering for multiple k values."""
        if k_values is None:
            k_values = [5, 8, 10, 12, 15, 20]
        
        logger.info(f"Running K-means clustering for k={k_values}")
        
        for k in tqdm(k_values, desc="K-means"):
            logger.info(f"  K-means with k={k}")
            
            # Check if already exists
            model_path = self.output_dir / f"models/kmeans_k{k}.pkl"
            if model_path.exists():
                logger.info(f"    Loading existing model from {model_path}")
                model = joblib.load(model_path)
                labels = model.predict(self.features_scaled)
            else:
                # Run clustering
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(self.features_scaled)
                
                # Save model
                joblib.dump(model, model_path)
                logger.info(f"    Saved model to {model_path}")
            
            # Save assignments
            assignments_df = pd.DataFrame({
                'h3_index': self.data.index if self.data.index.name else self.data.iloc[:, 0],
                'cluster': labels
            })
            assignments_path = self.output_dir / f"assignments/kmeans_k{k}.parquet"
            assignments_df.to_parquet(assignments_path)
            
            # Compute and save metrics
            metrics = self.compute_metrics(labels)
            metrics['method'] = 'kmeans'
            metrics['k'] = k
            metrics['timestamp'] = datetime.now().isoformat()
            
            metadata_path = self.output_dir / f"metadata/kmeans_k{k}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"    Silhouette: {metrics.get('silhouette', 'N/A'):.3f}" if metrics.get('silhouette') else "    Silhouette: N/A")
            logger.info(f"    Saved to {assignments_path}")
    
    def run_hierarchical(self, k_values: List[int] = None, linkage: str = 'ward'):
        """Run Hierarchical clustering."""
        if k_values is None:
            k_values = [8, 10, 12]
        
        logger.info(f"Running Hierarchical clustering ({linkage}) for k={k_values}")
        
        # Sample for hierarchical (memory intensive)
        sample_size = min(50000, len(self.features_scaled))
        if sample_size < len(self.features_scaled):
            logger.info(f"  Sampling {sample_size} points for hierarchical clustering")
            sample_idx = np.random.choice(len(self.features_scaled), sample_size, replace=False)
            features_sample = self.features_scaled[sample_idx]
        else:
            features_sample = self.features_scaled
            sample_idx = np.arange(len(self.features_scaled))
        
        for k in tqdm(k_values, desc=f"Hierarchical-{linkage}"):
            logger.info(f"  Hierarchical {linkage} with k={k}")
            
            model_path = self.output_dir / f"models/hierarchical_{linkage}_k{k}.pkl"
            
            if model_path.exists():
                logger.info(f"    Loading existing model")
                model = joblib.load(model_path)
                labels_sample = model.labels_
            else:
                # Run clustering on sample
                model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                labels_sample = model.fit_predict(features_sample)
                
                # Save model
                joblib.dump(model, model_path)
            
            # If sampled, assign remaining points to nearest cluster center
            if sample_size < len(self.features_scaled):
                # Compute cluster centers from sample
                centers = np.array([features_sample[labels_sample == i].mean(axis=0) 
                                   for i in range(k)])
                
                # Assign all points to nearest center
                from sklearn.metrics.pairwise import euclidean_distances
                distances = euclidean_distances(self.features_scaled, centers)
                labels = distances.argmin(axis=1)
            else:
                labels = labels_sample
            
            # Save assignments
            assignments_df = pd.DataFrame({
                'h3_index': self.data.index if self.data.index.name else self.data.iloc[:, 0],
                'cluster': labels
            })
            assignments_path = self.output_dir / f"assignments/hierarchical_{linkage}_k{k}.parquet"
            assignments_df.to_parquet(assignments_path)
            
            # Compute and save metrics
            metrics = self.compute_metrics(labels)
            metrics['method'] = f'hierarchical_{linkage}'
            metrics['k'] = k
            metrics['sample_size'] = sample_size
            metrics['timestamp'] = datetime.now().isoformat()
            
            metadata_path = self.output_dir / f"metadata/hierarchical_{linkage}_k{k}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"    Saved to {assignments_path}")
    
    def run_gmm(self, k_values: List[int] = None):
        """Run Gaussian Mixture Model clustering."""
        if k_values is None:
            k_values = [5, 8, 10, 12, 15]
        
        logger.info(f"Running GMM clustering for k={k_values}")
        
        for k in tqdm(k_values, desc="GMM"):
            logger.info(f"  GMM with k={k}")
            
            model_path = self.output_dir / f"models/gmm_k{k}.pkl"
            
            if model_path.exists():
                logger.info(f"    Loading existing model")
                model = joblib.load(model_path)
                labels = model.predict(self.features_scaled)
            else:
                # Run clustering
                model = GaussianMixture(n_components=k, random_state=42, covariance_type='diag')
                labels = model.fit_predict(self.features_scaled)
                
                # Save model
                joblib.dump(model, model_path)
            
            # Save assignments
            assignments_df = pd.DataFrame({
                'h3_index': self.data.index if self.data.index.name else self.data.iloc[:, 0],
                'cluster': labels
            })
            assignments_path = self.output_dir / f"assignments/gmm_k{k}.parquet"
            assignments_df.to_parquet(assignments_path)
            
            # Compute and save metrics
            metrics = self.compute_metrics(labels)
            metrics['method'] = 'gmm'
            metrics['k'] = k
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Add GMM-specific metrics
            metrics['log_likelihood'] = float(model.score(self.features_scaled))
            metrics['aic'] = float(model.aic(self.features_scaled))
            metrics['bic'] = float(model.bic(self.features_scaled))
            
            metadata_path = self.output_dir / f"metadata/gmm_k{k}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"    AIC: {metrics['aic']:.1f}, BIC: {metrics['bic']:.1f}")
            logger.info(f"    Saved to {assignments_path}")
    
    def run_all(self):
        """Run all clustering methods."""
        logger.info("="*80)
        logger.info("CASCADIA COASTAL FORESTS CLUSTERING")
        logger.info("="*80)
        
        # Load data
        self.load_data()
        
        # Run all methods
        logger.info("\n" + "="*40)
        self.run_kmeans()
        
        logger.info("\n" + "="*40)
        self.run_hierarchical(linkage='ward')
        
        logger.info("\n" + "="*40)
        self.run_gmm()
        
        logger.info("\n" + "="*80)
        logger.info("CLUSTERING COMPLETE!")
        logger.info(f"Results saved to {self.output_dir}")
        logger.info("="*80)


def main():
    """Main execution function."""
    clusterer = CoastalForestClusterer()
    clusterer.run_all()


if __name__ == "__main__":
    main()