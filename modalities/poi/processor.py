"""
Points of Interest (POI) Modality Processor

Processes OpenStreetMap POI data into H3 hexagon embeddings using SRAI.
Generates count-based and contextual embeddings for various POI categories.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import h3
from tqdm.auto import tqdm

# SRAI imports
from srai.loaders import OSMPbfLoader, OSMOnlineLoader
from srai.regionalizers import H3Regionalizer
from srai.embedders import CountEmbedder, Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.neighbourhoods import H3Neighbourhood

# Import base class
from ..base import ModalityProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class POIProcessor(ModalityProcessor):
    """Process POI data into H3 hexagon embeddings."""
    
    # Default POI categories and their tags
    DEFAULT_POI_CATEGORIES = {
        'amenity': [
            'restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'food_court',
            'hospital', 'clinic', 'pharmacy', 'doctors', 'dentist',
            'school', 'university', 'college', 'kindergarten', 'library',
            'bank', 'atm', 'post_office', 'police', 'fire_station',
            'place_of_worship', 'community_centre', 'theatre', 'cinema',
            'parking', 'fuel', 'charging_station', 'bicycle_parking'
        ],
        'shop': [
            'supermarket', 'convenience', 'bakery', 'butcher', 'greengrocer',
            'clothes', 'fashion', 'shoes', 'jewelry', 'department_store',
            'electronics', 'mobile_phone', 'computer', 'furniture', 'hardware',
            'sports', 'bicycle', 'car', 'car_repair', 'hairdresser', 'beauty'
        ],
        'leisure': [
            'park', 'playground', 'sports_centre', 'stadium', 'swimming_pool',
            'fitness_centre', 'golf_course', 'marina', 'nature_reserve'
        ],
        'tourism': [
            'hotel', 'motel', 'hostel', 'guest_house', 'apartment',
            'museum', 'gallery', 'attraction', 'viewpoint', 'information'
        ],
        'landuse': [
            'retail', 'commercial', 'industrial', 'residential',
            'forest', 'meadow', 'farmland', 'cemetery'
        ],
        'natural': [
            'water', 'wetland', 'beach', 'wood', 'tree', 'peak'
        ],
        'public_transport': [
            'station', 'stop_position', 'platform', 'stop_area'
        ]
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize POI processor with configuration."""
        super().__init__(config)
        self.validate_config()
        
        # Set up configuration
        self.data_source = config.get('data_source', 'osm_online')
        self.pbf_path = config.get('pbf_path', None)
        self.poi_categories = config.get('poi_categories', self.DEFAULT_POI_CATEGORIES)
        self.use_hex2vec = config.get('use_hex2vec', True)
        self.hex2vec_dimensions = config.get('hex2vec_dimensions', 64)
        self.compute_diversity_metrics = config.get('compute_diversity_metrics', True)
        self.chunk_size = config.get('chunk_size', 1000)
        self.max_workers = config.get('max_workers', 4)
        
        logger.info(f"Initialized POIProcessor with data source: {self.data_source}")
    
    def validate_config(self):
        """Validate configuration parameters."""
        required = ['output_dir']
        for param in required:
            if param not in self.config:
                raise ValueError(f"Missing required config parameter: {param}")
        
        # Validate data source
        if self.config.get('data_source') == 'pbf' and not self.config.get('pbf_path'):
            raise ValueError("PBF data source requires 'pbf_path' in config")
    
    def load_data(self, study_area: Union[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """Load POI data for the study area."""
        logger.info(f"Loading POI data using {self.data_source}")
        
        # Convert study area to GeoDataFrame if needed
        if isinstance(study_area, str):
            area_gdf = gpd.read_file(study_area)
        else:
            area_gdf = study_area
        
        # Ensure WGS84 projection
        if area_gdf.crs != 'EPSG:4326':
            area_gdf = area_gdf.to_crs('EPSG:4326')
        
        # Load POIs based on data source
        if self.data_source == 'pbf' and self.pbf_path:
            pois_gdf = self._load_from_pbf(area_gdf)
        else:
            pois_gdf = self._load_from_osm_online(area_gdf)
        
        logger.info(f"Loaded {len(pois_gdf)} POIs")
        return pois_gdf
    
    def _load_from_pbf(self, area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Load POIs from PBF file."""
        loader = OSMPbfLoader()
        
        # Build tags filter from categories
        tags = {}
        for category, values in self.poi_categories.items():
            if isinstance(values, list):
                tags[category] = values
            else:
                tags[category] = True
        
        # Load from PBF
        pois_gdf = loader.load(
            self.pbf_path,
            tags=tags,
            area=area_gdf
        )
        
        # Add category labels
        pois_gdf = self._categorize_pois(pois_gdf)
        
        return pois_gdf
    
    def _load_from_osm_online(self, area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Load POIs from OSM online API."""
        loader = OSMOnlineLoader()
        
        all_pois = []
        
        # Load each category separately to avoid API limits
        for category, values in self.poi_categories.items():
            try:
                logger.info(f"Loading {category} POIs...")
                
                if isinstance(values, list) and len(values) > 0:
                    # Load specific values
                    tags = {category: values[:10]}  # Limit to avoid API issues
                else:
                    tags = {category: True}
                
                category_pois = loader.load(area_gdf, tags=tags)
                
                if not category_pois.empty:
                    category_pois['main_category'] = category
                    all_pois.append(category_pois)
                    logger.info(f"  Found {len(category_pois)} {category} POIs")
                
            except Exception as e:
                logger.warning(f"Could not load {category} POIs: {e}")
        
        # Combine all POIs
        if all_pois:
            pois_gdf = pd.concat(all_pois, ignore_index=True)
            pois_gdf = gpd.GeoDataFrame(pois_gdf, crs='EPSG:4326')
        else:
            pois_gdf = gpd.GeoDataFrame([], crs='EPSG:4326')
        
        # Add detailed categorization
        if not pois_gdf.empty:
            pois_gdf = self._categorize_pois(pois_gdf)
        
        return pois_gdf
    
    def _categorize_pois(self, pois_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add detailed category information to POIs."""
        if 'main_category' not in pois_gdf.columns:
            pois_gdf['main_category'] = 'unknown'
        
        # Add subcategory based on specific tags
        pois_gdf['subcategory'] = 'other'
        
        for category, values in self.poi_categories.items():
            if isinstance(values, list):
                for value in values:
                    mask = pois_gdf.get(category, '') == value
                    pois_gdf.loc[mask, 'main_category'] = category
                    pois_gdf.loc[mask, 'subcategory'] = value
        
        return pois_gdf
    
    def process_to_h3(self, data: gpd.GeoDataFrame, h3_resolution: int) -> pd.DataFrame:
        """Process POI data into H3 hexagon embeddings."""
        logger.info(f"Processing POIs to H3 resolution {h3_resolution}")
        
        # Create H3 hexagons for the area
        bounds = data.total_bounds
        area_polygon = Polygon([
            (bounds[0], bounds[1]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[0], bounds[3])
        ])
        area_gdf = gpd.GeoDataFrame([1], geometry=[area_polygon], crs='EPSG:4326')
        
        # Generate H3 hexagons
        regionalizer = H3Regionalizer(resolution=h3_resolution)
        hexagons_gdf = regionalizer.transform(area_gdf)
        
        logger.info(f"Created {len(hexagons_gdf)} H3 hexagons")
        
        # Create count-based embeddings
        embeddings_df = self._create_count_embeddings(data, hexagons_gdf)
        
        # Add diversity metrics
        if self.compute_diversity_metrics:
            diversity_df = self._calculate_diversity_metrics(data, hexagons_gdf)
            embeddings_df = embeddings_df.merge(diversity_df, on='h3_index', how='left')
        
        # Create contextual embeddings with Hex2Vec if requested
        if self.use_hex2vec and len(hexagons_gdf) > 10:  # Need enough hexagons
            contextual_embeddings = self._create_hex2vec_embeddings(
                data, hexagons_gdf, embeddings_df
            )
            if contextual_embeddings is not None:
                embeddings_df = embeddings_df.merge(
                    contextual_embeddings, 
                    on='h3_index', 
                    how='left'
                )
        
        # Add metadata
        embeddings_df['h3_resolution'] = h3_resolution
        
        return embeddings_df
    
    def _create_count_embeddings(self, pois_gdf: gpd.GeoDataFrame, 
                                 hexagons_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Create count-based POI embeddings."""
        logger.info("Creating count-based embeddings")
        
        # Use SRAI's CountEmbedder
        embedder = CountEmbedder()
        joiner = IntersectionJoiner()
        
        # Join POIs with hexagons
        joint_gdf = joiner.transform(hexagons_gdf, pois_gdf)
        
        # Create embeddings for main categories
        if 'main_category' in pois_gdf.columns:
            count_embeddings = embedder.transform(
                regions_gdf=hexagons_gdf,
                features_gdf=pois_gdf,
                joint_gdf=joint_gdf,
                aggregation_column='main_category'
            )
        else:
            count_embeddings = embedder.transform(
                regions_gdf=hexagons_gdf,
                features_gdf=pois_gdf,
                joint_gdf=joint_gdf
            )
        
        # Convert to DataFrame with h3_index
        embeddings_df = count_embeddings.reset_index()
        embeddings_df = embeddings_df.rename(columns={'index': 'h3_index'})
        
        # Add total POI count
        total_counts = joint_gdf.groupby(joint_gdf.index).size()
        embeddings_df['total_poi_count'] = embeddings_df['h3_index'].map(
            lambda x: total_counts.get(x, 0)
        )
        
        # Calculate POI density (POIs per km²)
        embeddings_df['poi_density'] = embeddings_df.apply(
            lambda row: row['total_poi_count'] / h3.cell_area(row['h3_index'], 'km^2'),
            axis=1
        )
        
        # Fill NaN values with 0
        embeddings_df = embeddings_df.fillna(0)
        
        return embeddings_df
    
    def _calculate_diversity_metrics(self, pois_gdf: gpd.GeoDataFrame,
                                    hexagons_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate POI diversity metrics for each hexagon."""
        logger.info("Calculating diversity metrics")
        
        joiner = IntersectionJoiner()
        joint_gdf = joiner.transform(hexagons_gdf, pois_gdf)
        
        diversity_metrics = []
        
        for hex_id in hexagons_gdf.index:
            hex_pois = joint_gdf[joint_gdf.index == hex_id]
            
            if hex_pois.empty:
                diversity_metrics.append({
                    'h3_index': hex_id,
                    'poi_shannon_entropy': 0,
                    'poi_simpson_diversity': 0,
                    'poi_richness': 0,
                    'poi_evenness': 0
                })
            else:
                # Get category counts
                if 'main_category' in hex_pois.columns:
                    category_counts = hex_pois['main_category'].value_counts()
                else:
                    category_counts = pd.Series([len(hex_pois)])
                
                # Calculate diversity indices
                total = category_counts.sum()
                proportions = category_counts / total
                
                # Shannon entropy
                shannon_entropy = -sum(p * np.log(p) if p > 0 else 0 for p in proportions)
                
                # Simpson diversity
                simpson_diversity = 1 - sum(p**2 for p in proportions)
                
                # Richness (number of unique categories)
                richness = len(category_counts)
                
                # Evenness (Shannon entropy / max entropy)
                max_entropy = np.log(richness) if richness > 1 else 1
                evenness = shannon_entropy / max_entropy if max_entropy > 0 else 0
                
                diversity_metrics.append({
                    'h3_index': hex_id,
                    'poi_shannon_entropy': shannon_entropy,
                    'poi_simpson_diversity': simpson_diversity,
                    'poi_richness': richness,
                    'poi_evenness': evenness
                })
        
        return pd.DataFrame(diversity_metrics)
    
    def _create_hex2vec_embeddings(self, pois_gdf: gpd.GeoDataFrame,
                                   hexagons_gdf: gpd.GeoDataFrame,
                                   count_embeddings: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create contextual embeddings using Hex2Vec."""
        try:
            logger.info("Creating Hex2Vec contextual embeddings")
            
            # Get H3 neighborhood
            neighbourhood = H3Neighbourhood()
            neighbours_df = neighbourhood.get_neighbours(hexagons_gdf)
            
            # Prepare features for Hex2Vec
            feature_columns = [col for col in count_embeddings.columns 
                             if col not in ['h3_index', 'h3_resolution']]
            
            if not feature_columns:
                logger.warning("No features available for Hex2Vec")
                return None
            
            # Set index for merging
            count_embeddings_indexed = count_embeddings.set_index('h3_index')
            
            # Initialize Hex2Vec embedder
            hex2vec_embedder = Hex2VecEmbedder(
                dimensions=self.hex2vec_dimensions,
                walk_length=30,
                num_walks=100,
                window_size=5,
                p=1,
                q=1,
                skip_gram=True
            )
            
            # Train and transform
            hex2vec_embeddings = hex2vec_embedder.fit_transform(
                regions_gdf=hexagons_gdf,
                features_gdf=pois_gdf,
                neighbourhood=neighbours_df,
                base_embeddings=count_embeddings_indexed[feature_columns]
            )
            
            # Format output
            hex2vec_df = hex2vec_embeddings.reset_index()
            hex2vec_df = hex2vec_df.rename(columns={'index': 'h3_index'})
            
            # Rename columns to indicate they're Hex2Vec embeddings
            embedding_cols = [col for col in hex2vec_df.columns if col != 'h3_index']
            rename_dict = {col: f'hex2vec_{col}' for col in embedding_cols}
            hex2vec_df = hex2vec_df.rename(columns=rename_dict)
            
            logger.info(f"Created {self.hex2vec_dimensions}-dimensional Hex2Vec embeddings")
            return hex2vec_df
            
        except Exception as e:
            logger.warning(f"Could not create Hex2Vec embeddings: {e}")
            return None
    
    def run_pipeline(self, study_area: Union[str, gpd.GeoDataFrame], 
                    h3_resolution: int,
                    output_dir: str = None) -> str:
        """Execute complete POI processing pipeline."""
        logger.info(f"Starting POI pipeline for resolution {h3_resolution}")
        
        # Use configured output dir if not specified
        if output_dir is None:
            output_dir = self.config.get('output_dir', 'data/processed/embeddings/poi')
        
        # Load POI data
        pois_gdf = self.load_data(study_area)
        
        if pois_gdf.empty:
            logger.warning("No POI data found for study area")
            return None
        
        # Process to H3
        embeddings_df = self.process_to_h3(pois_gdf, h3_resolution)
        
        # Save embeddings
        output_path = self.save_embeddings(
            embeddings_df,
            output_dir,
            filename=f"poi_embeddings_res{h3_resolution}.parquet"
        )
        
        logger.info(f"POI embeddings saved to {output_path}")
        logger.info(f"Processed {len(embeddings_df)} hexagons with {embeddings_df.shape[1]} features")
        
        # Log summary statistics
        if 'total_poi_count' in embeddings_df.columns:
            total_pois = embeddings_df['total_poi_count'].sum()
            avg_density = embeddings_df['poi_density'].mean()
            logger.info(f"Total POIs: {total_pois:.0f}, Average density: {avg_density:.2f} POIs/km²")
        
        return output_path