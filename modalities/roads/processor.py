"""
Roads Network Modality Processor

Processes OpenStreetMap road network data into H3 hexagon embeddings using SRAI.
Extracts network topology, connectivity metrics, and road type distributions.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import warnings

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
import h3
from tqdm.auto import tqdm

# SRAI imports
from srai.loaders import OSMWayLoader, OSMPbfLoader, OSMOnlineLoader
from srai.regionalizers import H3Regionalizer
from srai.joiners import IntersectionJoiner

# Network analysis
import networkx as nx
import osmnx as ox

# Import base class
from ..base import ModalityProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class RoadsProcessor(ModalityProcessor):
    """Process road network data into H3 hexagon embeddings."""
    
    # Road type hierarchy for classification
    ROAD_HIERARCHY = {
        'motorway': 1.0,
        'motorway_link': 0.9,
        'trunk': 0.85,
        'trunk_link': 0.8,
        'primary': 0.75,
        'primary_link': 0.7,
        'secondary': 0.65,
        'secondary_link': 0.6,
        'tertiary': 0.55,
        'tertiary_link': 0.5,
        'unclassified': 0.4,
        'residential': 0.35,
        'living_street': 0.3,
        'service': 0.25,
        'pedestrian': 0.2,
        'track': 0.15,
        'footway': 0.1,
        'cycleway': 0.1,
        'path': 0.05
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize roads processor with configuration."""
        super().__init__(config)
        self.validate_config()
        
        # Set up data source
        self.data_source = config.get('data_source', 'osm_online')
        self.pbf_path = config.get('pbf_path', None)
        self.road_types = config.get('road_types', list(self.ROAD_HIERARCHY.keys()))
        self.compute_network_metrics = config.get('compute_network_metrics', True)
        self.chunk_size = config.get('chunk_size', 1000)
        self.max_workers = config.get('max_workers', 4)
        
        logger.info(f"Initialized RoadsProcessor with data source: {self.data_source}")
    
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
        """Load road network data for the study area."""
        logger.info(f"Loading road network data using {self.data_source}")
        
        # Convert study area to GeoDataFrame if needed
        if isinstance(study_area, str):
            # Assume it's a file path
            area_gdf = gpd.read_file(study_area)
        else:
            area_gdf = study_area
        
        # Ensure WGS84 projection
        if area_gdf.crs != 'EPSG:4326':
            area_gdf = area_gdf.to_crs('EPSG:4326')
        
        # Load roads based on data source
        if self.data_source == 'pbf' and self.pbf_path:
            roads_gdf = self._load_from_pbf(area_gdf)
        else:
            roads_gdf = self._load_from_osm_online(area_gdf)
        
        logger.info(f"Loaded {len(roads_gdf)} road segments")
        return roads_gdf
    
    def _load_from_pbf(self, area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Load roads from PBF file."""
        loader = OSMPbfLoader()
        
        # Create highway tags filter
        tags = {'highway': self.road_types}
        
        # Load from PBF
        roads_gdf = loader.load(
            self.pbf_path,
            tags=tags,
            area=area_gdf
        )
        
        return roads_gdf
    
    def _load_from_osm_online(self, area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Load roads from OSM online API."""
        loader = OSMWayLoader(
            way_type='highway',
            osm_way_filter={'highway': self.road_types}
        )
        
        # Load from OSM API
        roads_gdf = loader.load(area_gdf)
        
        # Filter to study area if needed
        if not roads_gdf.empty:
            roads_gdf = gpd.sjoin(
                roads_gdf,
                area_gdf,
                how='inner',
                predicate='intersects'
            )
        
        return roads_gdf
    
    def process_to_h3(self, data: gpd.GeoDataFrame, h3_resolution: int) -> pd.DataFrame:
        """Process road network into H3 hexagon embeddings."""
        logger.info(f"Processing roads to H3 resolution {h3_resolution}")
        
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
        
        # Calculate road features per hexagon
        embeddings = self._calculate_road_embeddings(data, hexagons_gdf, h3_resolution)
        
        return embeddings
    
    def _calculate_road_embeddings(self, roads_gdf: gpd.GeoDataFrame, 
                                  hexagons_gdf: gpd.GeoDataFrame,
                                  h3_resolution: int) -> pd.DataFrame:
        """Calculate road network embeddings for each hexagon."""
        logger.info("Calculating road embeddings per hexagon")
        
        # Spatial join roads with hexagons
        joiner = IntersectionJoiner()
        joined_gdf = joiner.transform(hexagons_gdf, roads_gdf)
        
        # Initialize embedding DataFrame
        embeddings_list = []
        
        # Process each hexagon
        for hex_id in tqdm(hexagons_gdf.index, desc="Processing hexagons"):
            hex_roads = joined_gdf[joined_gdf.index == hex_id]
            
            if hex_roads.empty:
                # No roads in this hexagon
                embedding = self._get_empty_embedding(hex_id)
            else:
                embedding = self._calculate_hex_embedding(hex_id, hex_roads)
            
            embeddings_list.append(embedding)
        
        # Combine all embeddings
        embeddings_df = pd.DataFrame(embeddings_list)
        embeddings_df['h3_resolution'] = h3_resolution
        
        # Add network metrics if requested
        if self.compute_network_metrics:
            logger.info("Computing network-wide metrics")
            network_metrics = self._compute_network_metrics(roads_gdf, hexagons_gdf)
            embeddings_df = embeddings_df.merge(
                network_metrics,
                on='h3_index',
                how='left'
            )
        
        return embeddings_df
    
    def _calculate_hex_embedding(self, hex_id: str, hex_roads: gpd.GeoDataFrame) -> Dict:
        """Calculate embedding for a single hexagon."""
        embedding = {'h3_index': hex_id}
        
        # Road type distribution
        for road_type, importance in self.ROAD_HIERARCHY.items():
            type_roads = hex_roads[hex_roads.get('highway', '') == road_type]
            if not type_roads.empty:
                # Calculate total length
                type_length = type_roads.geometry.length.sum()
                embedding[f'road_{road_type}_length'] = type_length
                embedding[f'road_{road_type}_count'] = len(type_roads)
            else:
                embedding[f'road_{road_type}_length'] = 0
                embedding[f'road_{road_type}_count'] = 0
        
        # Overall statistics
        embedding['total_road_length'] = hex_roads.geometry.length.sum()
        embedding['road_count'] = len(hex_roads)
        embedding['road_density'] = embedding['total_road_length'] / h3.cell_area(hex_id, 'km^2')
        
        # Road hierarchy score (weighted by importance)
        hierarchy_score = 0
        total_length = 0
        for _, road in hex_roads.iterrows():
            road_type = road.get('highway', 'unclassified')
            importance = self.ROAD_HIERARCHY.get(road_type, 0.1)
            length = road.geometry.length
            hierarchy_score += importance * length
            total_length += length
        
        embedding['road_hierarchy_score'] = hierarchy_score / total_length if total_length > 0 else 0
        
        # Connectivity metrics
        unique_intersections = set()
        for geom in hex_roads.geometry:
            if hasattr(geom, 'coords'):
                coords = list(geom.coords)
                unique_intersections.add(coords[0])  # Start point
                unique_intersections.add(coords[-1])  # End point
        
        embedding['intersection_count'] = len(unique_intersections)
        embedding['connectivity_index'] = len(unique_intersections) / (len(hex_roads) + 1) if len(hex_roads) > 0 else 0
        
        return embedding
    
    def _get_empty_embedding(self, hex_id: str) -> Dict:
        """Get empty embedding for hexagon with no roads."""
        embedding = {'h3_index': hex_id}
        
        # Initialize all road type features to 0
        for road_type in self.ROAD_HIERARCHY.keys():
            embedding[f'road_{road_type}_length'] = 0
            embedding[f'road_{road_type}_count'] = 0
        
        # Overall statistics
        embedding['total_road_length'] = 0
        embedding['road_count'] = 0
        embedding['road_density'] = 0
        embedding['road_hierarchy_score'] = 0
        embedding['intersection_count'] = 0
        embedding['connectivity_index'] = 0
        
        return embedding
    
    def _compute_network_metrics(self, roads_gdf: gpd.GeoDataFrame, 
                                hexagons_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Compute network-level metrics using graph analysis."""
        try:
            # Build network graph
            logger.info("Building road network graph")
            
            # Create nodes from road endpoints
            nodes = []
            edges = []
            
            for idx, road in roads_gdf.iterrows():
                if hasattr(road.geometry, 'coords'):
                    coords = list(road.geometry.coords)
                    start = coords[0]
                    end = coords[-1]
                    
                    nodes.extend([start, end])
                    edges.append((start, end, {
                        'weight': road.geometry.length,
                        'highway': road.get('highway', 'unclassified')
                    }))
            
            # Create graph
            G = nx.Graph()
            G.add_nodes_from(set(nodes))
            G.add_edges_from(edges)
            
            logger.info(f"Network graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Calculate centrality metrics for nodes
            if G.number_of_nodes() > 0:
                degree_centrality = nx.degree_centrality(G)
                
                # Limited betweenness centrality (sample for performance)
                if G.number_of_nodes() > 100:
                    k = min(100, G.number_of_nodes())
                    betweenness_centrality = nx.betweenness_centrality(G, k=k)
                else:
                    betweenness_centrality = nx.betweenness_centrality(G)
                
                # Assign metrics to hexagons
                hex_metrics = []
                for hex_id in hexagons_gdf.index:
                    hex_geom = hexagons_gdf.loc[hex_id, 'geometry']
                    
                    # Find nodes within hexagon
                    hex_degree = []
                    hex_betweenness = []
                    
                    for node in G.nodes():
                        point = Point(node)
                        if hex_geom.contains(point):
                            hex_degree.append(degree_centrality.get(node, 0))
                            hex_betweenness.append(betweenness_centrality.get(node, 0))
                    
                    hex_metrics.append({
                        'h3_index': hex_id,
                        'avg_degree_centrality': np.mean(hex_degree) if hex_degree else 0,
                        'max_degree_centrality': np.max(hex_degree) if hex_degree else 0,
                        'avg_betweenness_centrality': np.mean(hex_betweenness) if hex_betweenness else 0,
                        'max_betweenness_centrality': np.max(hex_betweenness) if hex_betweenness else 0
                    })
                
                return pd.DataFrame(hex_metrics)
            
        except Exception as e:
            logger.warning(f"Could not compute network metrics: {e}")
        
        # Return empty metrics if computation fails
        return pd.DataFrame({
            'h3_index': hexagons_gdf.index,
            'avg_degree_centrality': 0,
            'max_degree_centrality': 0,
            'avg_betweenness_centrality': 0,
            'max_betweenness_centrality': 0
        })
    
    def run_pipeline(self, study_area: Union[str, gpd.GeoDataFrame], 
                    h3_resolution: int,
                    output_dir: str = None) -> str:
        """Execute complete road processing pipeline."""
        logger.info(f"Starting roads pipeline for resolution {h3_resolution}")
        
        # Use configured output dir if not specified
        if output_dir is None:
            output_dir = self.config.get('output_dir', 'data/processed/embeddings/roads')
        
        # Load road data
        roads_gdf = self.load_data(study_area)
        
        if roads_gdf.empty:
            logger.warning("No road data found for study area")
            return None
        
        # Process to H3
        embeddings_df = self.process_to_h3(roads_gdf, h3_resolution)
        
        # Save embeddings
        output_path = self.save_embeddings(
            embeddings_df, 
            output_dir,
            filename=f"roads_embeddings_res{h3_resolution}.parquet"
        )
        
        logger.info(f"Roads embeddings saved to {output_path}")
        logger.info(f"Processed {len(embeddings_df)} hexagons with {embeddings_df.shape[1]} features")
        
        return output_path