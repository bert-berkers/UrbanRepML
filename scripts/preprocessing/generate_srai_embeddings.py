"""
Generate SRAI embeddings (POI, Road Network, GTFS) for the Netherlands at H3 resolution 10.
This script processes the entire Netherlands in spatial batches to manage memory efficiently.
"""

import logging
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import h3
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import srai
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf
from srai.embedders import (
    GTFS2VecEmbedder,
    CountEmbedder,
    ContextualCountEmbedder
)
from srai.loaders import OSMWayLoader, OSMTileLoader, GTFSLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SRAIEmbeddingGenerator:
    """Generate SRAI embeddings for large regions using spatial batching."""
    
    def __init__(
        self,
        resolution: int = 10,
        batch_size: int = 10000,  # Process hexagons in batches
        output_dir: Path = None,
        cache_dir: Path = None
    ):
        """
        Initialize SRAI embedding generator.
        
        Args:
            resolution: H3 resolution level (default 10)
            batch_size: Number of hexagons to process per batch
            output_dir: Directory for output embeddings
            cache_dir: Directory for caching intermediate results
        """
        self.resolution = resolution
        self.batch_size = batch_size
        
        # Setup directories
        self.output_dir = output_dir or Path("experiments/netherlands/data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = cache_dir or Path("experiments/netherlands/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SRAI embedding generator:")
        logger.info(f"  Resolution: {resolution}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Output dir: {self.output_dir}")
        
    def create_netherlands_regions(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Create H3 hexagons covering the Netherlands.
        
        Returns:
            Tuple of (boundary GeoDataFrame, regions GeoDataFrame)
        """
        logger.info("Creating Netherlands H3 regions...")
        
        # Try to load cached regions
        cache_path = self.cache_dir / f"netherlands_res{self.resolution}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached regions from {cache_path}")
            regions_gdf = gpd.read_parquet(cache_path)
            
            # Also load boundary
            boundary_path = self.cache_dir / "netherlands_boundary.parquet"
            if boundary_path.exists():
                boundary_gdf = gpd.read_parquet(boundary_path)
            else:
                # Create boundary from regions
                boundary_gdf = gpd.GeoDataFrame(
                    [{"geometry": regions_gdf.unary_union}],
                    crs="EPSG:4326"
                )
                boundary_gdf.to_parquet(boundary_path)
                
            return boundary_gdf, regions_gdf
        
        # Create new regions for Netherlands
        logger.info("Geocoding Netherlands boundary...")
        netherlands_gdf = geocode_to_region_gdf("Netherlands")
        
        # Save boundary
        boundary_path = self.cache_dir / "netherlands_boundary.parquet"
        netherlands_gdf.to_parquet(boundary_path)
        
        # Create H3 hexagons
        logger.info(f"Creating H3 hexagons at resolution {self.resolution}...")
        regionalizer = H3Regionalizer(self.resolution)
        regions_gdf = regionalizer.transform(netherlands_gdf)
        
        # Add hex_id column
        regions_gdf.index.name = 'hex_id'
        regions_gdf = regions_gdf.reset_index()
        regions_gdf.set_index('hex_id', inplace=True)
        
        # Cache regions
        regions_gdf.to_parquet(cache_path)
        logger.info(f"Created {len(regions_gdf)} hexagons for Netherlands")
        logger.info(f"Cached regions to {cache_path}")
        
        return netherlands_gdf, regions_gdf
    
    def create_spatial_batches(self, regions_gdf: gpd.GeoDataFrame) -> List[List[str]]:
        """
        Create spatial batches of hexagons for processing.
        Uses H3 parent cells to group nearby hexagons together.
        
        Args:
            regions_gdf: GeoDataFrame with H3 hexagons
            
        Returns:
            List of hexagon ID batches
        """
        logger.info("Creating spatial batches...")
        
        # Get parent hexagons at lower resolution for spatial grouping
        parent_resolution = max(self.resolution - 2, 4)  # Use 2 levels up or min res 4
        
        hex_ids = list(regions_gdf.index)
        parent_to_children = {}
        
        # Group hexagons by parent
        for hex_id in hex_ids:
            parent_id = h3.cell_to_parent(hex_id, parent_resolution)
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append(hex_id)
        
        # Create batches from parent groups
        batches = []
        current_batch = []
        
        for parent_id, children in parent_to_children.items():
            if len(current_batch) + len(children) > self.batch_size:
                if current_batch:
                    batches.append(current_batch)
                current_batch = children
            else:
                current_batch.extend(children)
        
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} spatial batches")
        logger.info(f"Average batch size: {np.mean([len(b) for b in batches]):.0f}")
        
        return batches
    
    def generate_poi_embeddings(
        self,
        regions_gdf: gpd.GeoDataFrame,
        batches: List[List[str]]
    ) -> pd.DataFrame:
        """
        Generate POI embeddings using OSM data.
        
        Args:
            regions_gdf: H3 regions GeoDataFrame
            batches: List of hexagon ID batches
            
        Returns:
            DataFrame with POI embeddings
        """
        logger.info("Generating POI embeddings...")
        
        # Check cache
        cache_path = self.cache_dir / f"poi_embeddings_res{self.resolution}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached POI embeddings from {cache_path}")
            return pd.read_parquet(cache_path)
        
        # Define POI categories to extract
        poi_tags = {
            'amenity': ['restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'hospital', 
                       'clinic', 'school', 'university', 'bank', 'pharmacy', 'parking'],
            'shop': True,  # Get all shops
            'leisure': ['park', 'playground', 'sports_centre', 'stadium'],
            'tourism': ['hotel', 'museum', 'attraction'],
            'office': True,  # Get all offices
        }
        
        # Initialize loader and embedder  
        poi_loader = OSMTileLoader()
        
        # Use ContextualCountEmbedder for richer POI embeddings
        poi_embedder = ContextualCountEmbedder(
            neighbourhood=None,  # Will be set per batch
            neighbourhood_distance=300,  # 300m radius
            concatenate_vectors=True
        )
        
        all_embeddings = []
        
        for batch_idx, batch_hex_ids in enumerate(tqdm(batches, desc="Processing POI batches")):
            try:
                # Get batch regions
                batch_regions = regions_gdf.loc[batch_hex_ids]
                
                # Load POI data for batch area
                batch_boundary = batch_regions.unary_union
                pois_gdf = poi_loader.load(batch_boundary, poi_tags)
                
                if len(pois_gdf) == 0:
                    logger.warning(f"No POIs found for batch {batch_idx}")
                    # Create zero embeddings
                    batch_embeddings = pd.DataFrame(
                        0, 
                        index=batch_hex_ids,
                        columns=[f"poi_{i}" for i in range(20)]  # Default POI dimensions
                    )
                else:
                    # Generate embeddings
                    poi_embedder.neighbourhood = batch_regions
                    batch_embeddings = poi_embedder.transform(
                        batch_regions,
                        pois_gdf,
                        poi_tags
                    )
                
                all_embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error processing POI batch {batch_idx}: {str(e)}")
                # Create zero embeddings for failed batch
                batch_embeddings = pd.DataFrame(
                    0,
                    index=batch_hex_ids,
                    columns=[f"poi_{i}" for i in range(20)]
                )
                all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        poi_embeddings = pd.concat(all_embeddings, axis=0)
        
        # Ensure all hexagons are included
        missing_hexes = set(regions_gdf.index) - set(poi_embeddings.index)
        if missing_hexes:
            logger.warning(f"Adding zero embeddings for {len(missing_hexes)} missing hexagons")
            missing_df = pd.DataFrame(
                0,
                index=list(missing_hexes),
                columns=poi_embeddings.columns
            )
            poi_embeddings = pd.concat([poi_embeddings, missing_df], axis=0)
        
        # Cache embeddings
        poi_embeddings.to_parquet(cache_path)
        logger.info(f"Generated POI embeddings with shape {poi_embeddings.shape}")
        logger.info(f"Cached to {cache_path}")
        
        return poi_embeddings
    
    def generate_road_network_embeddings(
        self,
        regions_gdf: gpd.GeoDataFrame,
        batches: List[List[str]]
    ) -> pd.DataFrame:
        """
        Generate road network embeddings using OSM street data.
        
        Args:
            regions_gdf: H3 regions GeoDataFrame
            batches: List of hexagon ID batches
            
        Returns:
            DataFrame with road network embeddings
        """
        logger.info("Generating road network embeddings...")
        
        # Check cache
        cache_path = self.cache_dir / f"road_embeddings_res{self.resolution}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached road embeddings from {cache_path}")
            return pd.read_parquet(cache_path)
        
        # Initialize loader for road network
        network_loader = OSMWayLoader()
        
        # Use CountEmbedder for road statistics
        road_embedder = CountEmbedder(
            expected_output_features=[
                'primary_length', 'secondary_length', 'residential_length',
                'motorway_length', 'total_length', 'intersection_count'
            ]
        )
        
        all_embeddings = []
        
        for batch_idx, batch_hex_ids in enumerate(tqdm(batches, desc="Processing road batches")):
            try:
                # Get batch regions
                batch_regions = regions_gdf.loc[batch_hex_ids]
                
                # Load road network for batch area
                batch_boundary = batch_regions.unary_union
                roads_gdf = network_loader.load(batch_boundary)
                
                if len(roads_gdf) == 0:
                    logger.warning(f"No roads found for batch {batch_idx}")
                    # Create zero embeddings
                    batch_embeddings = pd.DataFrame(
                        0,
                        index=batch_hex_ids,
                        columns=[f"road_{i}" for i in range(10)]
                    )
                else:
                    # Process road features
                    road_features = self._extract_road_features(roads_gdf, batch_regions)
                    batch_embeddings = road_features
                
                all_embeddings.append(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error processing road batch {batch_idx}: {str(e)}")
                # Create zero embeddings for failed batch
                batch_embeddings = pd.DataFrame(
                    0,
                    index=batch_hex_ids,
                    columns=[f"road_{i}" for i in range(10)]
                )
                all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        road_embeddings = pd.concat(all_embeddings, axis=0)
        
        # Ensure all hexagons are included
        missing_hexes = set(regions_gdf.index) - set(road_embeddings.index)
        if missing_hexes:
            logger.warning(f"Adding zero embeddings for {len(missing_hexes)} missing hexagons")
            missing_df = pd.DataFrame(
                0,
                index=list(missing_hexes),
                columns=road_embeddings.columns
            )
            road_embeddings = pd.concat([road_embeddings, missing_df], axis=0)
        
        # Cache embeddings
        road_embeddings.to_parquet(cache_path)
        logger.info(f"Generated road network embeddings with shape {road_embeddings.shape}")
        logger.info(f"Cached to {cache_path}")
        
        return road_embeddings
    
    def generate_gtfs_embeddings(
        self,
        regions_gdf: gpd.GeoDataFrame,
        gtfs_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generate GTFS transit embeddings.
        
        Args:
            regions_gdf: H3 regions GeoDataFrame
            gtfs_path: Path to GTFS data (will download if not provided)
            
        Returns:
            DataFrame with GTFS embeddings
        """
        logger.info("Generating GTFS embeddings...")
        
        # Check cache
        cache_path = self.cache_dir / f"gtfs_embeddings_res{self.resolution}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached GTFS embeddings from {cache_path}")
            return pd.read_parquet(cache_path)
        
        try:
            # Initialize GTFS loader
            gtfs_loader = GTFSLoader()
            
            if gtfs_path and gtfs_path.exists():
                logger.info(f"Loading GTFS data from {gtfs_path}")
                # Load local GTFS data
                gtfs_data = gtfs_loader.load(gtfs_path)
            else:
                logger.info("Downloading Netherlands GTFS data...")
                # Download Netherlands GTFS data
                # This would need actual GTFS feed URLs for Netherlands
                gtfs_data = gtfs_loader.load_from_url(
                    "https://gtfs.ovapi.nl/nl/gtfs-nl.zip"  # Example URL
                )
            
            # Initialize GTFS embedder
            gtfs_embedder = GTFS2VecEmbedder()
            
            # Generate embeddings for all regions at once
            gtfs_embeddings = gtfs_embedder.transform(regions_gdf, gtfs_data)
            
        except Exception as e:
            logger.error(f"Error generating GTFS embeddings: {str(e)}")
            logger.info("Creating zero GTFS embeddings as fallback")
            # Create zero embeddings
            gtfs_embeddings = pd.DataFrame(
                0,
                index=regions_gdf.index,
                columns=[f"gtfs_{i}" for i in range(10)]
            )
        
        # Cache embeddings
        gtfs_embeddings.to_parquet(cache_path)
        logger.info(f"Generated GTFS embeddings with shape {gtfs_embeddings.shape}")
        logger.info(f"Cached to {cache_path}")
        
        return gtfs_embeddings
    
    def _extract_road_features(
        self,
        roads_gdf: gpd.GeoDataFrame,
        regions_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Extract road network features for regions.
        
        Args:
            roads_gdf: GeoDataFrame with road network
            regions_gdf: GeoDataFrame with H3 regions
            
        Returns:
            DataFrame with road features
        """
        features = []
        
        for hex_id in regions_gdf.index:
            hex_geom = regions_gdf.loc[hex_id, 'geometry']
            
            # Find roads intersecting this hexagon
            hex_roads = roads_gdf[roads_gdf.intersects(hex_geom)]
            
            # Calculate features
            feature_dict = {
                'total_length': 0,
                'primary_length': 0,
                'secondary_length': 0,
                'residential_length': 0,
                'motorway_length': 0,
                'intersection_count': 0,
                'avg_lanes': 0,
                'road_density': 0,
                'connectivity': 0,
                'road_types': 0
            }
            
            if len(hex_roads) > 0:
                # Calculate road lengths by type
                for _, road in hex_roads.iterrows():
                    road_length = road.geometry.length
                    feature_dict['total_length'] += road_length
                    
                    # Check road type from tags
                    if 'highway' in road:
                        road_type = road['highway']
                        if 'primary' in road_type:
                            feature_dict['primary_length'] += road_length
                        elif 'secondary' in road_type:
                            feature_dict['secondary_length'] += road_length
                        elif 'residential' in road_type:
                            feature_dict['residential_length'] += road_length
                        elif 'motorway' in road_type:
                            feature_dict['motorway_length'] += road_length
                
                # Calculate density and connectivity metrics
                hex_area = hex_geom.area
                feature_dict['road_density'] = feature_dict['total_length'] / hex_area if hex_area > 0 else 0
                feature_dict['road_types'] = len(hex_roads['highway'].unique()) if 'highway' in hex_roads else 0
            
            features.append(pd.Series(feature_dict, name=hex_id))
        
        return pd.DataFrame(features)
    
    def combine_embeddings(
        self,
        poi_embeddings: pd.DataFrame,
        road_embeddings: pd.DataFrame,
        gtfs_embeddings: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine all SRAI embeddings into a single DataFrame.
        
        Args:
            poi_embeddings: POI embeddings
            road_embeddings: Road network embeddings
            gtfs_embeddings: GTFS embeddings
            
        Returns:
            Combined embeddings DataFrame
        """
        logger.info("Combining SRAI embeddings...")
        
        # Ensure same index
        common_index = poi_embeddings.index.intersection(road_embeddings.index).intersection(gtfs_embeddings.index)
        
        # Prefix columns for clarity
        poi_embeddings = poi_embeddings.loc[common_index].add_prefix('poi_')
        road_embeddings = road_embeddings.loc[common_index].add_prefix('road_')
        gtfs_embeddings = gtfs_embeddings.loc[common_index].add_prefix('gtfs_')
        
        # Combine
        combined = pd.concat([poi_embeddings, road_embeddings, gtfs_embeddings], axis=1)
        
        logger.info(f"Combined embeddings shape: {combined.shape}")
        
        return combined
    
    def run(self, gtfs_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
        """
        Run the complete SRAI embedding generation pipeline.
        
        Args:
            gtfs_path: Optional path to local GTFS data
            
        Returns:
            Dictionary with embedding DataFrames
        """
        logger.info("Starting SRAI embedding generation for Netherlands...")
        start_time = datetime.now()
        
        # Create regions
        boundary_gdf, regions_gdf = self.create_netherlands_regions()
        
        # Create spatial batches
        batches = self.create_spatial_batches(regions_gdf)
        
        # Generate embeddings
        poi_embeddings = self.generate_poi_embeddings(regions_gdf, batches)
        road_embeddings = self.generate_road_network_embeddings(regions_gdf, batches)
        gtfs_embeddings = self.generate_gtfs_embeddings(regions_gdf, gtfs_path)
        
        # Combine embeddings
        combined_embeddings = self.combine_embeddings(
            poi_embeddings,
            road_embeddings,
            gtfs_embeddings
        )
        
        # Save outputs
        output_files = {
            'poi': self.output_dir / f"netherlands_embeddings_POI_{self.resolution}.parquet",
            'road': self.output_dir / f"netherlands_embeddings_roadnetwork_{self.resolution}.parquet",
            'gtfs': self.output_dir / f"netherlands_embeddings_GTFS_{self.resolution}.parquet",
            'combined': self.output_dir / f"netherlands_embeddings_combined_srai_{self.resolution}.parquet"
        }
        
        poi_embeddings.to_parquet(output_files['poi'])
        road_embeddings.to_parquet(output_files['road'])
        gtfs_embeddings.to_parquet(output_files['gtfs'])
        combined_embeddings.to_parquet(output_files['combined'])
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'resolution': self.resolution,
            'num_hexagons': len(regions_gdf),
            'num_batches': len(batches),
            'batch_size': self.batch_size,
            'poi_dims': poi_embeddings.shape[1],
            'road_dims': road_embeddings.shape[1],
            'gtfs_dims': gtfs_embeddings.shape[1],
            'combined_dims': combined_embeddings.shape[1],
            'processing_time': str(datetime.now() - start_time),
            'output_files': {k: str(v) for k, v in output_files.items()}
        }
        
        with open(self.output_dir / 'netherlands_srai_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"SRAI embedding generation completed in {datetime.now() - start_time}")
        logger.info(f"Outputs saved to {self.output_dir}")
        
        return {
            'poi': poi_embeddings,
            'road': road_embeddings,
            'gtfs': gtfs_embeddings,
            'combined': combined_embeddings,
            'regions': regions_gdf
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SRAI embeddings for Netherlands"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=10,
        help="H3 resolution level (default: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Number of hexagons per batch (default: 10000)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments/netherlands/data"),
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("experiments/netherlands/cache"),
        help="Cache directory"
    )
    parser.add_argument(
        "--gtfs_path",
        type=Path,
        default=None,
        help="Path to local GTFS data (optional)"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SRAIEmbeddingGenerator(
        resolution=args.resolution,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    # Run generation
    embeddings = generator.run(gtfs_path=args.gtfs_path)
    
    # Print summary
    print("\nEmbedding Generation Summary:")
    print(f"POI embeddings: {embeddings['poi'].shape}")
    print(f"Road embeddings: {embeddings['road'].shape}")
    print(f"GTFS embeddings: {embeddings['gtfs'].shape}")
    print(f"Combined embeddings: {embeddings['combined'].shape}")
    print(f"Total hexagons: {len(embeddings['regions'])}")


if __name__ == "__main__":
    main()