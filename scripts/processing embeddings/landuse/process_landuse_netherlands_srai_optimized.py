"""
Optimized landuse processing for Netherlands using SRAI.
Uses more specific tags and better error handling.
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import time

# Use non-interactive backend to avoid Tkinter issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# SRAI imports
from srai import regionalizers, loaders, joiners, embedders, neighbourhoods
from srai.plotting import plot_regions, plot_numeric_data
from shapely.geometry import Polygon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def create_netherlands_boundary():
    """Create Netherlands study area boundary using SRAI."""
    logger.info("Creating Netherlands study area boundary...")
    
    # Define Netherlands bounding box (approximate)
    # Netherlands roughly: 3.3°E to 7.2°E, 50.7°N to 53.6°N
    min_lon, max_lon = 3.3, 7.2
    min_lat, max_lat = 50.7, 53.6
    
    # Create a polygon for the bounding box
    bbox = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat)
    ])
    
    # Create GeoDataFrame
    area_gdf = gpd.GeoDataFrame(
        {'name': ['Netherlands'], 'geometry': [bbox]},
        crs='EPSG:4326'
    )
    
    logger.info(f"Created boundary: {area_gdf.total_bounds}")
    return area_gdf


def load_or_create_regionalization(area_gdf, resolution=10):
    """Load existing regionalization or create new one."""
    region_path = Path(f"data/processed/h3_regions/netherlands_res{resolution}.parquet")
    
    if region_path.exists():
        logger.info(f"Loading existing regionalization from {region_path}")
        regions_gdf = gpd.read_parquet(region_path)
        regionalizer = regionalizers.H3Regionalizer(resolution=resolution)
        logger.info(f"Loaded {len(regions_gdf)} H3 hexagons at resolution {resolution}")
    else:
        logger.info(f"Creating new regionalization at H3 resolution {resolution}...")
        regionalizer = regionalizers.H3Regionalizer(resolution=resolution)
        regions_gdf = regionalizer.transform(area_gdf)
        logger.info(f"Created {len(regions_gdf)} H3 hexagons at resolution {resolution}")
        
        # Save regionalization
        region_path.parent.mkdir(parents=True, exist_ok=True)
        regions_gdf.to_parquet(region_path)
        logger.info(f"Saved regionalization to {region_path}")
    
    return regions_gdf, regionalizer


def download_poi_data_optimized(area_gdf):
    """Download POI data with specific landuse categories."""
    logger.info("Downloading POI data from OpenStreetMap...")
    
    # Use more specific tags to reduce download size
    # Focus on major landuse categories relevant for urban analysis
    tags = {
        'landuse': [
            'residential', 'commercial', 'industrial', 'retail',
            'farmland', 'forest', 'meadow', 'grass', 'greenfield',
            'recreation_ground', 'park', 'cemetery', 
            'construction', 'brownfield'
        ],
        'natural': [
            'water', 'wetland', 'wood', 'grassland', 
            'heath', 'scrub', 'beach', 'sand'
        ]
    }
    
    logger.info(f"Tags to download: {tags}")
    
    # Create OSM loader
    loader = loaders.OSMOnlineLoader()
    
    # Load features with timeout handling
    logger.info("Starting optimized download...")
    start_time = time.time()
    
    try:
        features_gdf = loader.load(area_gdf, tags)
        download_time = time.time() - start_time
        logger.info(f"Downloaded {len(features_gdf)} features in {download_time:.1f} seconds")
        
        # Show feature types distribution
        if 'feature_id' in features_gdf.columns:
            feature_counts = features_gdf['feature_id'].value_counts()
            logger.info(f"Top 10 feature types:\n{feature_counts.head(10)}")
        
        return features_gdf
        
    except Exception as e:
        logger.error(f"Error during download: {e}")
        logger.info("Trying with even more restricted tags...")
        
        # Fallback to most essential tags only
        simple_tags = {
            'landuse': ['residential', 'commercial', 'industrial', 'forest', 'farmland'],
            'natural': ['water', 'wood']
        }
        
        features_gdf = loader.load(area_gdf, simple_tags)
        logger.info(f"Downloaded {len(features_gdf)} features with restricted tags")
        return features_gdf


def save_intermediate_data(features_gdf, joint_gdf, stage_name):
    """Save intermediate processing results."""
    output_dir = Path("data/intermediate/landuse")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features
    if features_gdf is not None and len(features_gdf) > 0:
        features_path = output_dir / f"netherlands_{stage_name}_features.parquet"
        features_gdf.to_parquet(features_path)
        logger.info(f"Saved {len(features_gdf)} features to {features_path}")
    
    # Save joint data
    if joint_gdf is not None and len(joint_gdf) > 0:
        joint_path = output_dir / f"netherlands_{stage_name}_joint.parquet"
        joint_gdf.to_parquet(joint_path)
        logger.info(f"Saved joint data ({joint_gdf.shape}) to {joint_path}")


def intersect_with_hexagons(features_gdf, regions_gdf):
    """Intersect POI features with H3 hexagons."""
    logger.info("Intersecting features with hexagons...")
    logger.info(f"Input: {len(features_gdf)} features, {len(regions_gdf)} hexagons")
    
    # Create intersection joiner
    joiner = joiners.IntersectionJoiner()
    
    # Perform intersection
    joint_gdf = joiner.transform(regions_gdf, features_gdf)
    
    logger.info(f"Created joint dataset with shape: {joint_gdf.shape}")
    if len(joint_gdf) > 0:
        logger.info(f"Columns: {joint_gdf.columns.tolist()}")
        logger.info(f"Non-empty hexagons: {joint_gdf.index.nunique()}")
    
    return joint_gdf


def train_embeddings_optimized(joint_gdf, regions_gdf, features_gdf):
    """Train embeddings with optimized settings."""
    logger.info("Training embeddings...")
    
    embeddings = {}
    
    # 1. Count Embedder (basic counts)
    logger.info("Generating count embeddings...")
    try:
        count_embedder = embedders.CountEmbedder()
        count_embeddings = count_embedder.transform(regions_gdf, features_gdf, joint_gdf)
        embeddings['counts'] = count_embeddings
        logger.info(f"Count embeddings shape: {count_embeddings.shape}")
        logger.info(f"Non-zero hexagons: {(count_embeddings.sum(axis=1) > 0).sum()}")
    except Exception as e:
        logger.error(f"Count embedding failed: {e}")
    
    # 2. Hex2Vec Embedder (if available and we have enough data)
    if len(joint_gdf) > 1000:  # Only train if we have enough data
        try:
            from srai.embedders import Hex2VecEmbedder
            logger.info("Training Hex2Vec embeddings...")
            logger.info("Configuration: batch_size=5000, epochs=10, embedding_size=32")
            
            # Create neighborhood for Hex2Vec
            h3_neighbourhood = neighbourhoods.H3Neighbourhood(regions_gdf)
            
            # Configure Hex2Vec with large batch size
            hex2vec = embedders.Hex2VecEmbedder(
                embedder_size=32,
                batch_size=5000,  # Large batch size
                epochs=10,         # Low epochs for speed
                learning_rate=0.001,
                skip_distance=2,
            )
            
            # Fit and transform
            logger.info("Fitting Hex2Vec model...")
            hex2vec.fit(regions_gdf, features_gdf, joint_gdf, h3_neighbourhood)
            
            logger.info("Transforming to get embeddings...")
            hex2vec_embeddings = hex2vec.transform(regions_gdf, features_gdf, joint_gdf)
            embeddings['hex2vec'] = hex2vec_embeddings
            logger.info(f"Hex2Vec embeddings shape: {hex2vec_embeddings.shape}")
            
        except ImportError:
            logger.warning("Hex2Vec not available. Install with: pip install 'srai[torch]'")
        except Exception as e:
            logger.error(f"Hex2Vec training failed: {e}")
    else:
        logger.warning(f"Not enough data for Hex2Vec training (only {len(joint_gdf)} joints)")
    
    return embeddings


def save_embeddings(embeddings, resolution=10):
    """Save final embeddings to disk."""
    output_dir = Path("data/processed/embeddings/landuse")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    for embed_type, embed_df in embeddings.items():
        output_path = output_dir / f"netherlands_res{resolution}_{embed_type}.parquet"
        
        # Ensure H3 index is saved
        if embed_df.index.name:
            embed_df = embed_df.reset_index()
        
        embed_df.to_parquet(output_path)
        saved_paths[embed_type] = output_path
        logger.info(f"Saved {embed_type} embeddings to {output_path}")
    
    return saved_paths


def create_summary_visualization(embeddings, regions_gdf):
    """Create a summary visualization of the results."""
    output_dir = Path("data/processed/landuse/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'counts' in embeddings and len(embeddings['counts']) > 0:
        # Get total counts per hexagon
        count_df = embeddings['counts']
        total_counts = count_df.sum(axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Sample hexagons with data for visualization
        hexagons_with_data = total_counts[total_counts > 0].index
        sample_size = min(10000, len(hexagons_with_data))
        
        if sample_size > 0:
            sample_indices = np.random.choice(hexagons_with_data, size=sample_size, replace=False)
            sample_regions = regions_gdf.loc[sample_indices].copy()
            sample_regions['counts'] = total_counts.loc[sample_indices]
            
            # Plot
            sample_regions.plot(column='counts', cmap='YlOrRd', 
                               legend=True, ax=ax, 
                               legend_kwds={'label': 'Feature Count'})
            
            ax.set_title('Landuse Feature Density in Netherlands\n(Sample of hexagons with data)',
                        fontsize=16, fontweight='bold')
            ax.set_xlabel("Longitude", fontsize=12)
            ax.set_ylabel("Latitude", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Save
            fig.savefig(output_dir / "netherlands_landuse_density.png", dpi=150, bbox_inches='tight')
            logger.info(f"Saved density visualization")
        
        plt.close(fig)


def main():
    """Main execution pipeline."""
    logger.info("="*60)
    logger.info("Starting Optimized Netherlands Landuse Processing with SRAI")
    logger.info("="*60)
    
    try:
        # Step 1: Create Netherlands study area
        logger.info("\nStep 1: Creating Netherlands study area...")
        area_gdf = create_netherlands_boundary()
        
        # Step 2: Load or create regionalization
        logger.info("\nStep 2: Loading/creating H3 regionalization...")
        regions_gdf, regionalizer = load_or_create_regionalization(area_gdf, resolution=10)
        
        # Step 3: Download POI data with optimized tags
        logger.info("\nStep 3: Downloading POI data (optimized)...")
        features_gdf = download_poi_data_optimized(area_gdf)
        
        if len(features_gdf) == 0:
            logger.error("No features downloaded. Exiting.")
            return
        
        # Step 4: Intersect with hexagons
        logger.info("\nStep 4: Intersecting features with hexagons...")
        joint_gdf = intersect_with_hexagons(features_gdf, regions_gdf)
        
        # Save intermediate results
        logger.info("\nSaving intermediate results...")
        save_intermediate_data(features_gdf, joint_gdf, "res10_optimized")
        
        # Step 5: Train embeddings
        logger.info("\nStep 5: Training embeddings...")
        embeddings = train_embeddings_optimized(joint_gdf, regions_gdf, features_gdf)
        
        if len(embeddings) == 0:
            logger.error("No embeddings generated. Check the data.")
            return
        
        # Step 6: Save final embeddings
        logger.info("\nStep 6: Saving final embeddings...")
        saved_paths = save_embeddings(embeddings, resolution=10)
        
        # Step 7: Create visualization
        logger.info("\nStep 7: Creating summary visualization...")
        create_summary_visualization(embeddings, regions_gdf)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info("\nSummary:")
        logger.info(f"  - Study area: Netherlands")
        logger.info(f"  - H3 Resolution: 10")
        logger.info(f"  - Total hexagons: {len(regions_gdf):,}")
        logger.info(f"  - Total features: {len(features_gdf):,}")
        logger.info(f"  - Joint data points: {len(joint_gdf):,}")
        logger.info(f"  - Embeddings generated: {list(embeddings.keys())}")
        logger.info("\nOutput files:")
        for embed_type, path in saved_paths.items():
            logger.info(f"  - {embed_type}: {path}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()