#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick clustering for corrected Cascadia coastal forest dataset
Run clustering on the clean data without tile boundary artifacts
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run clustering on corrected coastal forest data"""
    
    # Load corrected dataset
    data_path = Path("data/h3_2021_res8_coastal_forests/cascadia_coastal_forests_2021_res8_final.parquet")
    logger.info(f"Loading corrected dataset from {data_path}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} hexagons")
    
    # Extract AlphaEarth embeddings (A00-A63)
    embedding_cols = [f'A{i:02d}' for i in range(64)]
    embeddings = df[embedding_cols].values
    logger.info(f"Extracted embeddings: {embeddings.shape}")
    
    # Standardize features
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Output directories
    results_dir = Path("results/coastal_2021")
    assignments_dir = results_dir / "assignments"
    models_dir = results_dir / "models"
    metadata_dir = results_dir / "metadata"
    
    for dir_path in [assignments_dir, models_dir, metadata_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Run clustering methods
    methods = [
        ('kmeans_k5', KMeans(n_clusters=5, random_state=42, n_init=10)),
        ('kmeans_k8', KMeans(n_clusters=8, random_state=42, n_init=10)),
        ('kmeans_k10', KMeans(n_clusters=10, random_state=42, n_init=10)),
        ('kmeans_k12', KMeans(n_clusters=12, random_state=42, n_init=10)),
        ('kmeans_k15', KMeans(n_clusters=15, random_state=42, n_init=10)),
        ('gmm_k5', GaussianMixture(n_components=5, random_state=42)),
        ('gmm_k8', GaussianMixture(n_components=8, random_state=42)),
        ('gmm_k10', GaussianMixture(n_components=10, random_state=42)),
        ('gmm_k12', GaussianMixture(n_components=12, random_state=42)),
        ('gmm_k15', GaussianMixture(n_components=15, random_state=42)),
    ]
    
    for method_name, model in methods:
        logger.info(f"\nRunning {method_name}...")
        
        # Fit model
        try:
            if 'kmeans' in method_name:
                labels = model.fit_predict(embeddings_scaled)
            else:  # GMM
                model.fit(embeddings_scaled)
                labels = model.predict(embeddings_scaled)
            
            logger.info(f"  Completed clustering with {len(np.unique(labels))} clusters")
            
            # Save results
            assignments_df = pd.DataFrame({
                'h3_index': df['h3_index'],
                'cluster': labels
            })
            assignments_df.to_parquet(assignments_dir / f"{method_name}.parquet", index=False)
            
            # Save model
            joblib.dump(model, models_dir / f"{method_name}.pkl")
            
            # Save metadata
            metadata = {
                'method': method_name,
                'n_clusters': len(np.unique(labels)),
                'n_samples': len(labels),
                'unique_clusters': np.unique(labels).tolist()
            }
            
            with open(metadata_dir / f"{method_name}.json", 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            logger.info(f"  Saved results for {method_name}")
            
        except Exception as e:
            logger.error(f"  Failed {method_name}: {e}")
            continue
    
    logger.info(f"\n✅ Clustering complete! Results saved to {results_dir}")
    logger.info("Ready to generate clean visualizations without tile boundaries")

if __name__ == "__main__":
    main()