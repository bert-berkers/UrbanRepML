"""
Process landuse data for South Holland province using SRAI.
South Holland (Zuid-Holland) includes Rotterdam, The Hague, and other major Dutch cities.
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


def create_south_holland_boundary():
    """Create South Holland province boundary using SRAI."""
    logger.info("Creating South Holland province boundary...")
    
    # South Holland approximate bounds
    # Rotterdam area: ~4.3-4.6°E, 51.8-52.0°N
    # The Hague area: ~4.1-4.5°E, 51.9-52.2°N
    # Full province roughly: 3.9°E to 4.9°E, 51.7°N to 52.3°N
    min_lon, max_lon = 3.9, 4.9
    min_lat, max_lat = 51.7, 52.3
    
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
        {'name': ['South Holland'], 'geometry': [bbox]},
        crs='EPSG:4326'
    )
    
    logger.info(f"Created boundary: {area_gdf.total_bounds}")
    logger.info(f"Area covers major cities: Rotterdam, The Hague, Delft, Leiden")
    return area_gdf


def visualize_area(area_gdf, title="South Holland Province"):
    """Visualize the study area boundary."""
    fig, ax = plt.subplots(figsize=(10, 10))
    area_gdf.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.7)
    
    # Add city markers for reference
    cities = {
        'Rotterdam': (4.4777, 51.9244),
        'The Hague': (4.3007, 52.0705),
        'Delft': (4.3571, 52.0116),
        'Leiden': (4.4937, 52.1601),
        'Dordrecht': (4.6900, 51.8133)
    }
    
    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, 'ro', markersize=8)
        ax.annotate(city, (lon, lat), xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add labels
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add boundary info
    bounds = area_gdf.total_bounds
    ax.text(0.02, 0.98, f"South Holland Province\nWest: {bounds[0]:.2f}°E\nSouth: {bounds[1]:.2f}°N\n"
                        f"East: {bounds[2]:.2f}°E\nNorth: {bounds[3]:.2f}°N",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("data/processed/landuse/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "south_holland_boundary.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved boundary visualization to {output_dir / 'south_holland_boundary.png'}")
    
    plt.close(fig)


def regionalize_h3(area_gdf, resolution=10):
    """Regionalize the area using H3 hexagons."""
    logger.info(f"Regionalizing South Holland at H3 resolution {resolution}...")
    
    # Create H3 regionalizer
    regionalizer = regionalizers.H3Regionalizer(resolution=resolution)
    
    # Transform to get H3 hexagons
    regions_gdf = regionalizer.transform(area_gdf)
    
    logger.info(f"Created {len(regions_gdf):,} H3 hexagons at resolution {resolution}")
    logger.info(f"Sample H3 indices: {regions_gdf.index[:5].tolist()}")
    
    # Calculate approximate area
    area_km2 = len(regions_gdf) * 0.015  # Each res-10 hexagon is ~0.015 km²
    logger.info(f"Approximate area covered: {area_km2:.1f} km²")
    
    return regions_gdf, regionalizer


def visualize_regionalization(regions_gdf, title="South Holland H3 Regionalization"):
    """Visualize the H3 regionalization."""
    # Sample for visualization
    sample_size = min(10000, len(regions_gdf))
    sample_regions = regions_gdf.sample(n=sample_size, random_state=42)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot regions
    sample_regions.plot(ax=ax, color='lightgreen', edgecolor='darkgreen', 
                        alpha=0.6, linewidth=0.5)
    
    # Add city markers
    cities = {
        'Rotterdam': (4.4777, 51.9244),
        'The Hague': (4.3007, 52.0705),
        'Delft': (4.3571, 52.0116)
    }
    
    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, 'ro', markersize=10, zorder=5)
        ax.annotate(city, (lon, lat), xytext=(5, 5), textcoords='offset points', 
                   fontsize=11, fontweight='bold', zorder=5,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add title and labels
    ax.set_title(f"{title}\n(Showing {sample_size:,} of {len(regions_gdf):,} hexagons)",
                 fontsize=16, fontweight='bold')
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
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
    fig.savefig(output_dir / "south_holland_h3_regionalization.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved regionalization visualization")
    
    plt.close(fig)


def save_regionalization(regions_gdf, name="south_holland", resolution=10):
    """Save the regionalization to disk."""
    output_dir = Path("data/processed/h3_regions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{name}_res{resolution}.parquet"
    regions_gdf.to_parquet(output_path)
    logger.info(f"Saved regionalization to {output_path}")
    
    # Also save a small sample as GeoJSON for inspection
    sample_path = output_dir / f"{name}_res{resolution}_sample.geojson"
    regions_gdf.head(1000).to_file(sample_path, driver='GeoJSON')
    logger.info(f"Saved sample to {sample_path}")
    
    return output_path


def download_poi_data(area_gdf):
    """Download POI data for landuse and natural features."""
    logger.info("Downloading POI data from OpenStreetMap for South Holland...")
    
    # Use focused tags for urban/suburban analysis
    tags = {
        'landuse': [
            'residential', 'commercial', 'industrial', 'retail',
            'farmland', 'forest', 'grass', 'park', 'recreation_ground'
        ],
        'natural': [
            'water', 'wetland', 'wood', 'grassland', 'beach'
        ]
    }
    
    logger.info(f"Tags to download: {tags}")
    
    # Create OSM loader
    loader = loaders.OSMOnlineLoader()
    
    # Load features
    logger.info("Starting download (should be quick for province-sized area)...")
    start_time = time.time()
    
    try:
        features_gdf = loader.load(area_gdf, tags)
        download_time = time.time() - start_time
        logger.info(f"Downloaded {len(features_gdf):,} features in {download_time:.1f} seconds")
        
        # Show feature types distribution
        if 'feature_id' in features_gdf.columns:
            feature_counts = features_gdf['feature_id'].value_counts()
            logger.info(f"\nTop 10 feature types:")
            for feat, count in feature_counts.head(10).items():
                logger.info(f"  {feat}: {count:,}")
        
        return features_gdf
        
    except Exception as e:
        logger.error(f"Error during download: {e}")
        logger.info("Retrying with minimal tags...")
        
        # Fallback to most essential tags
        simple_tags = {
            'landuse': ['residential', 'commercial', 'industrial', 'farmland'],
            'natural': ['water']
        }
        
        features_gdf = loader.load(area_gdf, simple_tags)
        logger.info(f"Downloaded {len(features_gdf):,} features with minimal tags")
        return features_gdf


def save_intermediate_data(features_gdf, joint_gdf, stage_name):
    """Save intermediate processing results."""
    output_dir = Path("data/intermediate/landuse")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features
    if features_gdf is not None and len(features_gdf) > 0:
        features_path = output_dir / f"south_holland_{stage_name}_features.parquet"
        features_gdf.to_parquet(features_path)
        logger.info(f"Saved {len(features_gdf):,} features to {features_path}")
    
    # Save joint data
    if joint_gdf is not None and len(joint_gdf) > 0:
        joint_path = output_dir / f"south_holland_{stage_name}_joint.parquet"
        joint_gdf.to_parquet(joint_path)
        logger.info(f"Saved joint data ({joint_gdf.shape}) to {joint_path}")


def intersect_with_hexagons(features_gdf, regions_gdf):
    """Intersect POI features with H3 hexagons."""
    logger.info("Intersecting features with hexagons...")
    logger.info(f"Input: {len(features_gdf):,} features, {len(regions_gdf):,} hexagons")
    
    # Create intersection joiner
    joiner = joiners.IntersectionJoiner()
    
    # Perform intersection
    joint_gdf = joiner.transform(regions_gdf, features_gdf)
    
    logger.info(f"Created joint dataset with shape: {joint_gdf.shape}")
    if len(joint_gdf) > 0:
        logger.info(f"Non-empty hexagons: {joint_gdf.index.nunique():,}")
    
    return joint_gdf


def train_embeddings(joint_gdf, regions_gdf, features_gdf):
    """Train embeddings using SRAI embedders with large batch size."""
    logger.info("Training embeddings...")
    
    embeddings = {}
    
    # 1. Count Embedder (basic counts)
    logger.info("\n1. Generating count embeddings...")
    try:
        count_embedder = embedders.CountEmbedder()
        count_embeddings = count_embedder.transform(regions_gdf, features_gdf, joint_gdf)
        embeddings['counts'] = count_embeddings
        logger.info(f"   Count embeddings shape: {count_embeddings.shape}")
        logger.info(f"   Non-zero hexagons: {(count_embeddings.sum(axis=1) > 0).sum():,}")
    except Exception as e:
        logger.error(f"Count embedding failed: {e}")
    
    # 2. Hex2Vec Embedder (neural embeddings)
    if len(joint_gdf) > 1000:
        try:
            from srai.embedders import Hex2VecEmbedder
            logger.info("\n2. Training Hex2Vec embeddings...")
            logger.info("   Configuration: batch_size=5000, epochs=10, embedding_size=32")
            
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
            logger.info("   Fitting Hex2Vec model...")
            hex2vec.fit(regions_gdf, features_gdf, joint_gdf, h3_neighbourhood)
            
            logger.info("   Transforming to get embeddings...")
            hex2vec_embeddings = hex2vec.transform(regions_gdf, features_gdf, joint_gdf)
            embeddings['hex2vec'] = hex2vec_embeddings
            logger.info(f"   Hex2Vec embeddings shape: {hex2vec_embeddings.shape}")
            
        except ImportError:
            logger.warning("Hex2Vec not available. Install with: pip install 'srai[torch]'")
        except Exception as e:
            logger.error(f"Hex2Vec training failed: {e}")
    else:
        logger.warning(f"Not enough data for Hex2Vec training (only {len(joint_gdf)} joints)")
    
    return embeddings


def save_embeddings(embeddings, name="south_holland", resolution=10):
    """Save final embeddings to disk."""
    output_dir = Path("data/processed/embeddings/landuse")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    for embed_type, embed_df in embeddings.items():
        output_path = output_dir / f"{name}_res{resolution}_{embed_type}.parquet"
        
        # Ensure H3 index is saved
        if embed_df.index.name:
            embed_df = embed_df.reset_index()
        
        embed_df.to_parquet(output_path)
        saved_paths[embed_type] = output_path
        logger.info(f"Saved {embed_type} embeddings to {output_path}")
    
    return saved_paths


def create_density_visualization(embeddings, regions_gdf):
    """Create a density visualization of landuse features."""
    output_dir = Path("data/processed/landuse/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'counts' in embeddings and len(embeddings['counts']) > 0:
        # Get total counts per hexagon
        count_df = embeddings['counts']
        total_counts = count_df.sum(axis=1)
        
        # Filter to hexagons with data
        hexagons_with_data = total_counts[total_counts > 0]
        
        if len(hexagons_with_data) > 0:
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Get regions with data
            regions_with_data = regions_gdf.loc[hexagons_with_data.index].copy()
            regions_with_data['counts'] = hexagons_with_data.values
            
            # Plot with color scale
            regions_with_data.plot(column='counts', cmap='YlOrRd', 
                                   legend=True, ax=ax,
                                   legend_kwds={'label': 'Feature Count', 'shrink': 0.8})
            
            # Add city markers
            cities = {
                'Rotterdam': (4.4777, 51.9244),
                'The Hague': (4.3007, 52.0705),
                'Delft': (4.3571, 52.0116),
                'Leiden': (4.4937, 52.1601)
            }
            
            for city, (lon, lat) in cities.items():
                ax.plot(lon, lat, 'ko', markersize=8, zorder=5)
                ax.annotate(city, (lon, lat), xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='black', zorder=5,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.set_title('Landuse Feature Density in South Holland Province',
                        fontsize=16, fontweight='bold')
            ax.set_xlabel("Longitude", fontsize=12)
            ax.set_ylabel("Latitude", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            ax.text(0.02, 0.98, f"Total hexagons with data: {len(hexagons_with_data):,}\n"
                               f"Max features per hexagon: {hexagons_with_data.max():.0f}\n"
                               f"Mean features per hexagon: {hexagons_with_data.mean():.1f}",
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save
            fig.savefig(output_dir / "south_holland_landuse_density.png", dpi=150, bbox_inches='tight')
            logger.info(f"Saved density visualization")
            
            plt.close(fig)


def main():
    """Main execution pipeline."""
    logger.info("="*60)
    logger.info("Starting South Holland Landuse Processing with SRAI")
    logger.info("="*60)
    
    try:
        # Step 1: Create South Holland study area
        logger.info("\nStep 1: Creating South Holland study area...")
        area_gdf = create_south_holland_boundary()
        visualize_area(area_gdf)
        
        # Step 2: Regionalize at H3 resolution 10
        logger.info("\nStep 2: Regionalizing at H3 resolution 10...")
        regions_gdf, regionalizer = regionalize_h3(area_gdf, resolution=10)
        visualize_regionalization(regions_gdf)
        
        # Step 3: Save regionalization
        logger.info("\nStep 3: Saving regionalization...")
        save_regionalization(regions_gdf, "south_holland", resolution=10)
        
        # Step 4: Download POI data
        logger.info("\nStep 4: Downloading POI data...")
        features_gdf = download_poi_data(area_gdf)
        
        if len(features_gdf) == 0:
            logger.error("No features downloaded. Exiting.")
            return
        
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
        saved_paths = save_embeddings(embeddings, "south_holland", resolution=10)
        
        # Step 8: Create visualizations
        logger.info("\nStep 8: Creating visualizations...")
        create_density_visualization(embeddings, regions_gdf)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info("\nSummary:")
        logger.info(f"  - Study area: South Holland Province")
        logger.info(f"  - Major cities: Rotterdam, The Hague, Delft, Leiden")
        logger.info(f"  - H3 Resolution: 10")
        logger.info(f"  - Total hexagons: {len(regions_gdf):,}")
        logger.info(f"  - Total features: {len(features_gdf):,}")
        logger.info(f"  - Joint data points: {len(joint_gdf):,}")
        logger.info(f"  - Embeddings generated: {list(embeddings.keys())}")
        logger.info("\nOutput files:")
        for embed_type, path in saved_paths.items():
            logger.info(f"  - {embed_type}: {path}")
        
        logger.info("\nVisualization saved to: data/processed/landuse/figures/")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()