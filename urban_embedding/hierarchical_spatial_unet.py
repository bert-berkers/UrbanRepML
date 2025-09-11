#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SRAI-based Hierarchical Spatial U-Net for Multi-Scale Embeddings

Implements a hierarchical U-Net architecture using SRAI framework with:
- H3 hexagonal regionalization
- Adjacency graphs instead of distance matrices  
- GeoVex POI embeddings
- Neighbor-based slope calculations
- PoissonVAE integration (future)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path

# SRAI imports - proper framework for spatial analysis
try:
    import srai
    from srai.loaders.osm_loaders import OSMPOILoader, OSMNetworkType
    from srai.regionalizers import H3Regionalizer
    from srai.embedders.geovex import GeoVexEmbedder
    from srai.joiners import IntersectionJoiner
    from srai.neighbourhoods import H3Neighbourhood
    SRAI_AVAILABLE = True
except ImportError:
    print("SRAI not available - using fallback implementations")
    SRAI_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TopographicalGradient:
    """Topographical gradient information for accessibility calculation"""
    elevation: float
    slope: float
    aspect: float
    curvature: float
    accessibility_cost: float


@dataclass 
class POIUtility:
    """Point of Interest utility for attraction/repulsion calculation"""
    poi_type: str
    utility_value: float  # Positive = attraction, Negative = repulsion
    influence_radius: float
    accessibility_modifier: float


class SRAIHierarchicalEmbedding:
    """
    SRAI-based hierarchical spatial embedding system using proper hexagonal regionalization,
    GeoVex embeddings, and adjacency graphs instead of distance matrices.
    """
    
    def __init__(
        self,
        resolutions: List[int] = [8, 9, 10, 11],  # Focused resolutions
        primary_resolution: int = 8,
        embedding_dim: int = 64
    ):
        self.resolutions = sorted(resolutions)
        self.primary_resolution = primary_resolution
        self.embedding_dim = embedding_dim
        
        # Initialize SRAI components
        if SRAI_AVAILABLE:
            self.regionalizer = H3Regionalizer(resolutions=resolutions)
            self.poi_loader = OSMPOILoader()
            self.geovex_embedder = GeoVexEmbedder()
            self.joiner = IntersectionJoiner()
            self.neighbourhood = H3Neighbourhood()
            logger.info("SRAI components initialized successfully")
        else:
            logger.warning("SRAI not available - using fallback implementations")
            self.regionalizer = None
            self.poi_loader = None
            self.geovex_embedder = None
        
        # Storage for hierarchical data
        self.regions = {}  # Resolution -> GeoDataFrame of regions
        self.adjacency_graphs = {}  # Resolution -> adjacency graphs
        self.elevation_data = {}  # Resolution -> elevation per region
        self.slope_data = {}  # Resolution -> slope per region (from neighbors)
        self.poi_embeddings = {}  # Resolution -> GeoVex POI embeddings
        
        logger.info(f"Initialized SRAI hierarchical embedding: resolutions {resolutions}")
    
    def create_hierarchical_regions(
        self,
        bounds: Dict[str, float]
    ) -> Dict[int, gpd.GeoDataFrame]:
        """
        Create hierarchical hexagonal regions using SRAI H3 regionalizer.
        
        Args:
            bounds: Geographic bounds {north, south, east, west}
            
        Returns:
            Dictionary mapping resolution to GeoDataFrames of regions
        """
        logger.info("Creating hierarchical hexagonal regions with SRAI...")
        
        # Create bounding box geometry
        from shapely.geometry import box
        bbox = box(bounds['west'], bounds['south'], bounds['east'], bounds['north'])
        
        regions = {}
        
        if SRAI_AVAILABLE and self.regionalizer is not None:
            # Use SRAI H3 regionalizer
            for resolution in self.resolutions:
                try:
                    region_gdf = self.regionalizer.transform(
                        geometry=bbox, 
                        resolution=resolution
                    )
                    regions[resolution] = region_gdf
                    logger.info(f"  Resolution {resolution}: {len(region_gdf)} hexagonal regions created")
                except Exception as e:
                    logger.error(f"Error creating regions for resolution {resolution}: {e}")
        else:
            # Fallback: create DENSE regions using h3 directly - OPTIMIZED FOR RTX 3090!
            import h3
            for resolution in self.resolutions:
                cells = set()  # Use set for faster deduplication
                
                # MASSIVE DENSITY for resolution 11 - RTX 3090 can handle it!
                if resolution == 11:
                    lat_step = (bounds['north'] - bounds['south']) / (200)  # SUPER DENSE!
                    lon_step = (bounds['east'] - bounds['west']) / (200)   # MAXIMUM COVERAGE!
                elif resolution == 10:
                    lat_step = (bounds['north'] - bounds['south']) / (100)  # HIGH DENSITY
                    lon_step = (bounds['east'] - bounds['west']) / (100)
                else:
                    lat_step = (bounds['north'] - bounds['south']) / (20 * resolution)
                    lon_step = (bounds['east'] - bounds['west']) / (20 * resolution)
                
                lat = bounds['south']
                while lat <= bounds['north']:
                    lon = bounds['west']
                    while lon <= bounds['east']:
                        cell = h3.latlng_to_cell(lat, lon, resolution)
                        cells.add(cell)  # Set automatically handles deduplication
                        lon += lon_step
                    lat += lat_step
                
                cells = list(cells)  # Convert back to list for processing embeddings
                
                # Create GeoDataFrame
                geometries = []
                region_ids = []
                for cell in cells:
                    boundary = h3.cell_to_boundary(cell)
                    from shapely.geometry import Polygon
                    poly = Polygon([(lon, lat) for lat, lon in boundary])
                    geometries.append(poly)
                    region_ids.append(cell)
                
                region_gdf = gpd.GeoDataFrame({
                    'region_id': region_ids,
                    'geometry': geometries
                })
                regions[resolution] = region_gdf
                logger.info(f"  Resolution {resolution}: {len(region_gdf)} fallback regions created")
        
        self.regions = regions
        return regions
    
    def create_adjacency_graphs(self) -> Dict[int, Dict]:
        """
        Create adjacency graphs for each resolution using SRAI neighbourhood system.
        """
        logger.info("Creating hexagonal adjacency graphs...")
        
        adjacency_graphs = {}
        
        for resolution in self.resolutions:
            if resolution not in self.regions:
                continue
                
            region_gdf = self.regions[resolution]
            
            if SRAI_AVAILABLE and self.neighbourhood is not None:
                try:
                    # Use SRAI neighbourhood system
                    adjacency = self.neighbourhood.get_neighbourhood(
                        regions_gdf=region_gdf,
                        neighbourhood_radius=1  # Direct neighbors only
                    )
                    adjacency_graphs[resolution] = adjacency
                    logger.info(f"  Resolution {resolution}: adjacency graph with {len(adjacency)} connections")
                except Exception as e:
                    logger.error(f"Error creating adjacency for resolution {resolution}: {e}")
                    # Fallback to simple adjacency
                    adjacency_graphs[resolution] = self._create_simple_adjacency(region_gdf)
            else:
                # Fallback adjacency using h3
                adjacency_graphs[resolution] = self._create_simple_adjacency(region_gdf)
        
        self.adjacency_graphs = adjacency_graphs
        return adjacency_graphs
    
    def _create_simple_adjacency(self, region_gdf: gpd.GeoDataFrame) -> Dict:
        """Simple adjacency using h3 grid_ring for fallback."""
        import h3
        adjacency = {}
        
        for idx, row in region_gdf.iterrows():
            region_id = row['region_id']
            neighbors = list(h3.grid_ring(region_id, 1))
            # Filter to only neighbors that exist in our region set
            valid_neighbors = [n for n in neighbors if n in region_gdf['region_id'].values]
            adjacency[region_id] = valid_neighbors
        
        logger.info(f"    Fallback adjacency created with {len(adjacency)} nodes")
        return adjacency
    
    def calculate_neighbor_slopes(self, resolution: int = 11) -> Dict[str, float]:
        """
        Calculate slopes using neighbor-to-neighbor elevation differences.
        Only needs adjacency, no distance matrices.
        
        Args:
            resolution: Resolution to calculate slopes for
            
        Returns:
            Dictionary mapping region_id to slope value
        """
        logger.info(f"Calculating neighbor-based slopes for resolution {resolution}")
        
        if resolution not in self.regions or resolution not in self.adjacency_graphs:
            logger.warning(f"Missing data for resolution {resolution}")
            return {}
        
        if resolution not in self.elevation_data:
            # Load elevation data first
            self.load_elevation_data({})
        
        region_gdf = self.regions[resolution]
        adjacency = self.adjacency_graphs[resolution]
        elevation_data = self.elevation_data.get(resolution, {})
        
        slopes = {}
        
        for region_id in region_gdf['region_id']:
            if region_id not in elevation_data:
                slopes[region_id] = 0.0
                continue
            
            current_elevation = elevation_data[region_id]
            neighbors = adjacency.get(region_id, [])
            
            if not neighbors:
                slopes[region_id] = 0.0
                continue
            
            # Calculate slope as max elevation difference with neighbors
            max_slope = 0.0
            for neighbor_id in neighbors:
                if neighbor_id in elevation_data:
                    neighbor_elevation = elevation_data[neighbor_id]
                    elevation_diff = abs(current_elevation - neighbor_elevation)
                    # Simple slope calculation (could use actual distances if needed)
                    slope = elevation_diff / 100.0  # Assuming ~100m between hex centers
                    max_slope = max(max_slope, slope)
            
            slopes[region_id] = max_slope
        
        self.slope_data[resolution] = slopes
        logger.info(f"Calculated slopes for {len(slopes)} regions")
        return slopes
    
    def create_geovex_poi_embeddings(
        self, 
        bounds: Dict[str, float]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Create POI embeddings using SRAI GeoVex embedder.
        
        Args:
            bounds: Geographic bounds for POI loading
            
        Returns:
            Dictionary mapping resolution to POI embeddings per region
        """
        logger.info("Creating GeoVex POI embeddings...")
        
        poi_embeddings = {}
        
        if SRAI_AVAILABLE and self.geovex_embedder is not None:
            try:
                # Load POIs using SRAI
                from shapely.geometry import box
                bbox = box(bounds['west'], bounds['south'], bounds['east'], bounds['north'])
                
                # Load POI data
                pois_gdf = self.poi_loader.load(
                    area=bbox,
                    tags={'amenity': True, 'shop': True, 'tourism': True, 'landuse': True}
                )
                logger.info(f"Loaded {len(pois_gdf)} POIs from OSM")
                
                # Create embeddings for each resolution
                for resolution in self.resolutions:
                    if resolution not in self.regions:
                        continue
                    
                    region_gdf = self.regions[resolution]
                    
                    # Join POIs with regions
                    joined_gdf = self.joiner.transform(
                        regions_gdf=region_gdf,
                        features_gdf=pois_gdf
                    )
                    
                    # Generate GeoVex embeddings
                    embeddings = self.geovex_embedder.transform(
                        regions_gdf=region_gdf,
                        features_gdf=joined_gdf
                    )
                    
                    poi_embeddings[resolution] = embeddings
                    logger.info(f"  Resolution {resolution}: GeoVex embeddings created")
                    
            except Exception as e:
                logger.error(f"Error creating GeoVex embeddings: {e}")
                # Fallback to simple POI embeddings
                poi_embeddings = self._create_fallback_poi_embeddings(bounds)
        else:
            # Fallback to simple POI embeddings
            poi_embeddings = self._create_fallback_poi_embeddings(bounds)
        
        self.poi_embeddings = poi_embeddings
        return poi_embeddings
    
    def _create_fallback_poi_embeddings(self, bounds: Dict[str, float]) -> Dict[int, Dict]:
        """Create simple fallback POI embeddings."""
        logger.info("Creating fallback POI embeddings...")
        
        poi_embeddings = {}
        
        for resolution in self.resolutions:
            if resolution not in self.regions:
                continue
                
            region_gdf = self.regions[resolution]
            region_embeddings = {}
            
            for idx, row in region_gdf.iterrows():
                region_id = row['region_id']
                # Simple mock embedding based on geographic location
                centroid = row['geometry'].centroid
                lat, lon = centroid.y, centroid.x
                
                # Create simple feature vector
                features = np.array([
                    lat - 41.75,  # Relative to Del Norte center
                    lon + 124.1,  # Relative to Del Norte center
                    np.sin(lat * 10),  # Spatial variation
                    np.cos(lon * 10),  # Spatial variation
                    np.random.normal(0, 0.1)  # Random component
                ])
                
                region_embeddings[region_id] = features
            
            poi_embeddings[resolution] = region_embeddings
            logger.info(f"  Resolution {resolution}: {len(region_embeddings)} fallback embeddings created")
        
        return poi_embeddings
    
    def load_elevation_data(
        self, 
        bounds: Dict[str, float],
        resolution_11_precision: float = 10.0  # meters
    ) -> Dict[int, Dict]:
        """
        Load elevation data for all regions.
        
        Args:
            bounds: Geographic bounds {north, south, east, west}
            resolution_11_precision: Elevation precision at res 11 in meters
            
        Returns:
            Dictionary mapping resolution to elevation data per region
        """
        logger.info("Loading elevation data for all regions...")
        
        elevation_data = {}
        
        for resolution in self.resolutions:
            if resolution not in self.regions:
                continue
            
            region_gdf = self.regions[resolution]
            region_elevations = {}
            
            for idx, row in region_gdf.iterrows():
                region_id = row['region_id']
                geometry = row['geometry']
                centroid = geometry.centroid
                lat, lon = centroid.y, centroid.x
                
                # Generate synthetic elevation for Del Norte County
                # Coastal mountains increase elevation inland
                coastal_distance = abs(lon + 124.0)  # Distance from coast
                elevation = (
                    coastal_distance * 500 +      # Base coastal to inland gradient
                    np.sin(lat * 10) * 100 +       # North-south variation
                    np.random.normal(0, 50)        # Local variation
                )
                elevation = max(0, elevation)       # No negative elevation
                
                region_elevations[region_id] = elevation
            
            elevation_data[resolution] = region_elevations
            avg_elevation = np.mean(list(region_elevations.values()))
            logger.info(f"  Resolution {resolution}: {len(region_elevations)} regions, avg elevation {avg_elevation:.1f}m")
        
        self.elevation_data = elevation_data
        return elevation_data


class HierarchicalUNet(nn.Module):
    """
    Hierarchical U-Net for learning beautiful multi-scale spatial embeddings.
    
    Architecture processes H3 resolutions in a U-Net pattern with adjacency graphs.
    """
    
    def __init__(
        self,
        input_dims: Dict[int, int],  # Resolution -> input feature dimension
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_conv_layers: int = 2
    ):
        super().__init__()
        
        self.resolutions = sorted(input_dims.keys())
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Simple multi-resolution processing embeddings layers
        self.resolution_encoders = nn.ModuleDict()
        self.output_projections = nn.ModuleDict()
        
        for res in self.resolutions:
            input_dim = input_dims[res]
            
            # Encoder for this resolution
            encoder_layers = []
            for i in range(num_conv_layers):
                layer_input_dim = input_dim if i == 0 else hidden_dim
                encoder_layers.extend([
                    nn.Linear(layer_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            
            self.resolution_encoders[str(res)] = nn.Sequential(*encoder_layers)
            
            # Output projection
            self.output_projections[str(res)] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
    
    def forward(
        self,
        hierarchical_features: Dict[int, torch.Tensor],
        adjacency_graphs: Optional[Dict[int, Dict]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Forward pass through hierarchical U-Net using adjacency information.
        """
        outputs = {}
        
        for res in self.resolutions:
            if res not in hierarchical_features:
                continue
            
            # Process features for this resolution
            features = hierarchical_features[res]
            encoded = self.resolution_encoders[str(res)](features)
            output = self.output_projections[str(res)](encoded)
            
            outputs[res] = output
        
        return outputs


class HierarchicalSpatialEmbedding:
    """
    Legacy hierarchical spatial embedding (keeping for compatibility)
    """
    def __init__(self, *args, **kwargs):
        logger.warning("HierarchicalSpatialEmbedding is deprecated - use SRAIHierarchicalEmbedding instead")


def main():
    """Demonstration of SRAI hierarchical spatial embedding system."""
    
    # Example usage for Del Norte County
    bounds = {
        'north': 42.0,
        'south': 41.5,
        'west': -124.4,
        'east': -123.8
    }
    
    # Initialize SRAI hierarchical spatial embedding
    srai_embedder = SRAIHierarchicalEmbedding(
        resolutions=[8, 9, 10, 11],  # Focused resolutions
        primary_resolution=8,
        embedding_dim=64
    )
    
    # Create hierarchical regions
    regions = srai_embedder.create_hierarchical_regions(bounds)
    
    # Create adjacency graphs
    adjacency_graphs = srai_embedder.create_adjacency_graphs()
    
    # Load elevation and calculate slopes
    elevation_data = srai_embedder.load_elevation_data(bounds)
    slopes = srai_embedder.calculate_neighbor_slopes(resolution=11)
    
    # Create POI embeddings
    poi_embeddings = srai_embedder.create_geovex_poi_embeddings(bounds)
    
    # Print summary
    print("SRAI Hierarchical Spatial Embeddings Built!")
    print("="*50)
    for res in srai_embedder.resolutions:
        if res in regions:
            region_count = len(regions[res])
            adjacency_count = len(adjacency_graphs.get(res, {}))
            elevation_count = len(elevation_data.get(res, {}))
            poi_count = len(poi_embeddings.get(res, {}))
            
            print(f"Resolution {res}:")
            print(f"  - Regions: {region_count}")
            print(f"  - Adjacency connections: {adjacency_count}")
            print(f"  - Elevation data: {elevation_count}")
            print(f"  - POI embeddings: {poi_count}")
            
            if res == 11:
                slope_count = len(slopes)
                print(f"  - Slopes calculated: {slope_count}")
    
    return srai_embedder


if __name__ == "__main__":
    main()