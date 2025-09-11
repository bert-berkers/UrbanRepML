"""
Process landuse data for all of Netherlands using SRAI library.
This script uses SRAI's functions for creating study areas, regionalizing, 
and generating embeddings with optimized batch processing.
"""

import logging
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
import numpy as np

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


def visualize_area(area_gdf, title="Netherlands Study Area"):
    """Visualize the study area boundary."""
    fig, ax = plt.subplots(figsize=(10, 10))
    area_gdf.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.7)
    
    # Add labels
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add boundary coordinates as text
    bounds = area_gdf.total_bounds
    ax.text(0.02, 0.98, f"Bounds:\nWest: {bounds[0]:.2f}°E\nSouth: {bounds[1]:.2f}°N\n"
                        f"East: {bounds[2]:.2f}°E\nNorth: {bounds[3]:.2f}°N",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("data/processed/landuse/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "netherlands_boundary.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved boundary visualization to {output_dir / 'netherlands_boundary.png'}")
    
    # Close figure to free memory
    plt.close(fig)


def regionalize_h3(area_gdf, resolution=10):
    """Regionalize the area using H3 hexagons."""
    logger.info(f"Regionalizing Netherlands at H3 resolution {resolution}...")
    
    # Create H3 regionalizer
    regionalizer = regionalizers.H3Regionalizer(resolution=resolution)
    
    # Transform to get H3 hexagons
    regions_gdf = regionalizer.transform(area_gdf)
    
    logger.info(f"Created {len(regions_gdf)} H3 hexagons at resolution {resolution}")
    logger.info(f"Sample H3 indices: {regions_gdf.index[:5].tolist()}")
    
    return regions_gdf, regionalizer


def visualize_regionalization(regions_gdf, title="Netherlands H3 Regionalization"):
    """Visualize the H3 regionalization."""
    # Sample for visualization (too many hexagons to plot all)
    sample_size = min(5000, len(regions_gdf))
    sample_regions = regions_gdf.sample(n=sample_size, random_state=42)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot regions
    sample_regions.plot(ax=ax, color='lightgreen', edgecolor='darkgreen', 
                        alpha=0.6, linewidth=0.5)
    
    # Add title and labels
    ax.set_title(f"{title}\n(Showing {sample_size:,} of {len(regions_gdf):,} hexagons)",
                 fontsize=16, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(0.02, 0.98, f"Total hexagons: {len(regions_gdf):,}\n"
                        f"H3 Resolution: 10\n"
                        f"Avg hexagon area: ~0.015 km²",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("data/processed/landuse/figures")
    fig.savefig(output_dir / "netherlands_h3_regionalization.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved regionalization visualization to {output_dir / 'netherlands_h3_regionalization.png'}")
    
    # Close figure to free memory
    plt.close(fig)


def save_regionalization(regions_gdf, resolution=10):
    """Save the regionalization to disk."""
    output_dir = Path("data/processed/h3_regions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"netherlands_res{resolution}.parquet"
    regions_gdf.to_parquet(output_path)
    logger.info(f"Saved regionalization to {output_path}")
    
    # Also save a small sample as GeoJSON for inspection
    sample_path = output_dir / f"netherlands_res{resolution}_sample.geojson"
    regions_gdf.head(1000).to_file(sample_path, driver='GeoJSON')
    logger.info(f"Saved sample to {sample_path}")
    
    return output_path


def download_poi_data(area_gdf):
    """Download POI data for landuse and natural features."""
    logger.info("Downloading POI data from OpenStreetMap...")
    
    # Define tags for landuse and natural features
    tags = {
        'landuse': True,  # Get all landuse types
        'natural': True,  # Get all natural features
    }
    
    logger.info(f"Tags to download: {tags}")
    
    # Create OSM loader
    loader = loaders.OSMOnlineLoader()
    
    # Load features
    logger.info("Starting download (this may take several minutes for all of Netherlands)...")
    features_gdf = loader.load(area_gdf, tags)
    
    logger.info(f"Downloaded {len(features_gdf)} features")
    
    # Show feature types distribution
    if 'feature_id' in features_gdf.columns:
        feature_counts = features_gdf['feature_id'].value_counts()
        logger.info(f"Top 10 feature types:\n{feature_counts.head(10)}")
    
    return features_gdf


def save_intermediate_data(features_gdf, joint_gdf, stage_name):
    """Save intermediate processing results."""
    output_dir = Path("data/intermediate/landuse")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features
    if features_gdf is not None:
        features_path = output_dir / f"netherlands_{stage_name}_features.parquet"
        features_gdf.to_parquet(features_path)
        logger.info(f"Saved features to {features_path}")
    
    # Save joint data
    if joint_gdf is not None:
        joint_path = output_dir / f"netherlands_{stage_name}_joint.parquet"
        joint_gdf.to_parquet(joint_path)
        logger.info(f"Saved joint data to {joint_path}")


def intersect_with_hexagons(features_gdf, regions_gdf):
    """Intersect POI features with H3 hexagons."""
    logger.info("Intersecting features with hexagons...")
    
    # Create intersection joiner
    joiner = joiners.IntersectionJoiner()
    
    # Perform intersection
    joint_gdf = joiner.transform(regions_gdf, features_gdf)
    
    logger.info(f"Created joint dataset with shape: {joint_gdf.shape}")
    logger.info(f"Columns: {joint_gdf.columns.tolist()}")
    
    return joint_gdf


def train_embeddings(joint_gdf, regions_gdf, features_gdf):
    """Train embeddings using SRAI embedders with large batch size."""
    logger.info("Training embeddings...")
    
    embeddings = {}
    
    # 1. Count Embedder (basic counts)
    logger.info("Generating count embeddings...")
    count_embedder = embedders.CountEmbedder()
    count_embeddings = count_embedder.transform(regions_gdf, features_gdf, joint_gdf)
    embeddings['counts'] = count_embeddings
    logger.info(f"Count embeddings shape: {count_embeddings.shape}")
    
    # 2. Hex2Vec Embedder (neural embeddings)
    try:
        logger.info("Training Hex2Vec embeddings...")
        logger.info("Configuration: batch_size=5000, epochs=10, embedding_size=32")
        
        # Create neighborhood for Hex2Vec
        h3_neighbourhood = neighbourhoods.H3Neighbourhood(regions_gdf)
        
        # Configure Hex2Vec
        hex2vec = embedders.Hex2VecEmbedder(
            embedder_size=32,
            batch_size=5000,  # Large batch size as requested
            epochs=10,         # Low epochs as requested
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
        
    except Exception as e:
        logger.warning(f"Could not train Hex2Vec: {e}")
        logger.info("Install torch dependencies with: pip install 'srai[torch]'")
    
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


def main():
    """Main execution pipeline."""
    logger.info("="*60)
    logger.info("Starting Netherlands Landuse Processing with SRAI")
    logger.info("="*60)
    
    try:
        # Step 1: Create Netherlands study area
        logger.info("\nStep 1: Creating Netherlands study area...")
        area_gdf = create_netherlands_boundary()
        visualize_area(area_gdf)
        
        # Step 2: Regionalize at H3 resolution 10
        logger.info("\nStep 2: Regionalizing at H3 resolution 10...")
        regions_gdf, regionalizer = regionalize_h3(area_gdf, resolution=10)
        visualize_regionalization(regions_gdf)
        
        # Step 3: Save regionalization
        logger.info("\nStep 3: Saving regionalization...")
        save_regionalization(regions_gdf, resolution=10)
        
        # Step 4: Download POI data
        logger.info("\nStep 4: Downloading POI data...")
        features_gdf = download_poi_data(area_gdf)
        
        # Step 5: Intersect with hexagons
        logger.info("\nStep 5: Intersecting features with hexagons...")
        joint_gdf = intersect_with_hexagons(features_gdf, regions_gdf)
        
        # Save intermediate results
        logger.info("\nSaving intermediate results...")
        save_intermediate_data(features_gdf, joint_gdf, "res10")
        
        # Step 6: Train embeddings
        logger.info("\nStep 6: Training embeddings...")
        embeddings = train_embeddings(joint_gdf, regions_gdf, features_gdf)
        
        # Step 7: Save final embeddings
        logger.info("\nStep 7: Saving final embeddings...")
        saved_paths = save_embeddings(embeddings, resolution=10)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info("\nSummary:")
        logger.info(f"  - Study area: Netherlands")
        logger.info(f"  - H3 Resolution: 10")
        logger.info(f"  - Total hexagons: {len(regions_gdf):,}")
        logger.info(f"  - Total features: {len(features_gdf):,}")
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