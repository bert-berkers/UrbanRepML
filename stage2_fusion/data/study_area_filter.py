"""
Study Area Filtering System for Multi-Resolution Urban Analysis

This module provides flexible filtering capabilities for defining study areas within
large geographic datasets. It supports:

1. Geographic filtering (counties, custom polygons, circular regions)
2. Density-based adaptive resolution (variable H3 depth based on building density)
3. Computational workload management (memory-aware chunking)
4. Multi-level epistemic depth (different resolution depths for different regions)

Usage:
    # Define study area using configuration
    filter_config = StudyAreaConfig.from_yaml('study_areas/my_area.yaml')
    area_filter = StudyAreaFilter(filter_config)
    
    # Apply filtering to dataset
    filtered_data = area_filter.filter_dataset(h3_data_dict)
    
    # Get processing_modalities chunks for memory management
    chunks = area_filter.get_processing_chunks(memory_limit_gb=16)
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import yaml

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
# MIGRATION: Replaced direct h3 import with SRAI (per CLAUDE.md)
from srai.regionalizers import H3Regionalizer
from srai.neighbourhoods import H3Neighbourhood
# Note: SRAI provides H3 functionality with additional spatial analysis tools

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ResolutionRule:
    """Rule for determining H3 resolution based on area characteristics."""
    name: str
    condition: str  # Python expression evaluated against hex properties
    resolution: int
    priority: int = 0  # Higher priority rules override lower priority
    description: str = ""


@dataclass
class GeographicBounds:
    """Geographic bounds definition for study area."""
    bounds_type: str  # 'bbox', 'counties', 'polygon', 'circle', 'shapefile'
    definition: Union[Dict, List, str]  # Bounds definition data
    buffer_km: float = 0.0  # Buffer around bounds in kilometers
    crs: str = "EPSG:4326"  # Coordinate reference system


@dataclass
class BioregionalContext:
    """Bioregional and ecological context for study area."""
    bioregion_type: str  # 'forestry', 'agriculture', 'mixed_use', 'conservation', 'watershed'
    primary_ecosystem: str  # 'conifer_forest', 'oak_woodland', 'grassland', 'riparian', 'agricultural'
    management_focus: List[str] = field(default_factory=list)  # ['timber', 'carbon', 'biodiversity', 'water', 'crops']
    climate_zone: Optional[str] = None  # 'mediterranean', 'temperate_oceanic', 'semi_arid'
    conservation_priority: str = 'standard'  # 'high', 'standard', 'low'
    
    # Agricultural characteristics
    primary_crops: List[str] = field(default_factory=list)
    farming_type: Optional[str] = None  # 'conventional', 'organic', 'regenerative', 'mixed'
    water_source: Optional[str] = None  # 'rainfed', 'irrigated', 'mixed'
    
    # Forestry characteristics  
    forest_type: Optional[str] = None  # 'old_growth', 'second_growth', 'plantation', 'mixed'
    timber_management: Optional[str] = None  # 'sustainable', 'intensive', 'conservation', 'none'


@dataclass
class StudyAreaConfig:
    """Configuration for study area filtering."""
    name: str
    description: str
    geographic_bounds: GeographicBounds
    bioregional_context: Optional[BioregionalContext] = None
    resolution_rules: List[ResolutionRule] = field(default_factory=list)
    default_resolution: int = 8
    
    # Density filtering
    density_filtering: Dict[str, Any] = field(default_factory=dict)
    
    # Bioregional filtering
    bioregional_filtering: Dict[str, Any] = field(default_factory=dict)
    
    # Computational constraints
    max_memory_gb: float = 16.0
    max_hexagons_per_chunk: int = 100000
    enable_chunking: bool = True
    
    # Quality thresholds
    min_coverage_threshold: float = 0.8
    min_density_threshold: float = 0.0
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'StudyAreaConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Parse geographic bounds
        bounds_data = config_data['geographic_bounds']
        geographic_bounds = GeographicBounds(**bounds_data)
        
        # Parse bioregional context
        bioregional_context = None
        if 'bioregional_context' in config_data:
            bioregional_context = BioregionalContext(**config_data['bioregional_context'])
        
        # Parse resolution rules
        resolution_rules = []
        for rule_data in config_data.get('resolution_rules', []):
            resolution_rules.append(ResolutionRule(**rule_data))
        
        return cls(
            name=config_data['name'],
            description=config_data['description'],
            geographic_bounds=geographic_bounds,
            bioregional_context=bioregional_context,
            resolution_rules=resolution_rules,
            **{k: v for k, v in config_data.items() 
               if k not in ['name', 'description', 'geographic_bounds', 'bioregional_context', 'resolution_rules']}
        )
    
    def save_yaml(self, yaml_path: Union[str, Path]):
        """Save configuration to YAML file."""
        config_dict = {
            'name': self.name,
            'description': self.description,
            'geographic_bounds': {
                'bounds_type': self.geographic_bounds.bounds_type,
                'definition': self.geographic_bounds.definition,
                'buffer_km': self.geographic_bounds.buffer_km,
                'crs': self.geographic_bounds.crs
            }
        }
        
        # Add bioregional context if present
        if self.bioregional_context:
            config_dict['bioregional_context'] = {
                'bioregion_type': self.bioregional_context.bioregion_type,
                'primary_ecosystem': self.bioregional_context.primary_ecosystem,
                'management_focus': self.bioregional_context.management_focus,
                'climate_zone': self.bioregional_context.climate_zone,
                'conservation_priority': self.bioregional_context.conservation_priority,
                'primary_crops': self.bioregional_context.primary_crops,
                'farming_type': self.bioregional_context.farming_type,
                'water_source': self.bioregional_context.water_source,
                'forest_type': self.bioregional_context.forest_type,
                'timber_management': self.bioregional_context.timber_management
            }
        
        config_dict.update({
            'resolution_rules': [
                {
                    'name': rule.name,
                    'condition': rule.condition,
                    'resolution': rule.resolution,
                    'priority': rule.priority,
                    'description': rule.description
                } for rule in self.resolution_rules
            ],
            'default_resolution': self.default_resolution,
            'density_filtering': self.density_filtering,
            'bioregional_filtering': self.bioregional_filtering,
            'max_memory_gb': self.max_memory_gb,
            'max_hexagons_per_chunk': self.max_hexagons_per_chunk,
            'enable_chunking': self.enable_chunking,
            'min_coverage_threshold': self.min_coverage_threshold,
            'min_density_threshold': self.min_density_threshold
        })
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, indent=2, default_flow_style=False)


class GeographicFilter(ABC):
    """Abstract base class for geographic filtering strategies."""
    
    @abstractmethod
    def create_study_area(self, definition: Any, buffer_km: float = 0.0, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """Create study area geometry from definition."""
        pass


class BoundingBoxFilter(GeographicFilter):
    """Filter by bounding box coordinates."""
    
    def create_study_area(self, definition: Dict, buffer_km: float = 0.0, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Create study area from bounding box.
        
        Args:
            definition: Dict with keys 'north', 'south', 'east', 'west'
            buffer_km: Buffer around bounds in kilometers
            crs: Coordinate reference system
        """
        bbox = box(definition['west'], definition['south'], 
                  definition['east'], definition['north'])
        
        gdf = gpd.GeoDataFrame([{'geometry': bbox}], crs=crs)
        
        if buffer_km > 0:
            # Convert to UTM for accurate buffering
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            gdf_utm['geometry'] = gdf_utm.buffer(buffer_km * 1000)  # Convert km to meters
            gdf = gdf_utm.to_crs(crs)
        
        return gdf


class CountyFilter(GeographicFilter):
    """Filter by county names."""
    
    def create_study_area(self, definition: List[str], buffer_km: float = 0.0, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Create study area from county list.
        
        Args:
            definition: List of county names
            buffer_km: Buffer around bounds in kilometers  
            crs: Coordinate reference system
        """
        # Note: This would need access to county boundary data
        # For now, placeholder implementation
        logger.warning("County filtering requires county boundary data - implement based on your data sources")
        
        # Return empty GeoDataFrame with proper structure
        return gpd.GeoDataFrame(columns=['geometry', 'county_name'], crs=crs)


class CircularFilter(GeographicFilter):
    """Filter by circular region (center + radius)."""
    
    def create_study_area(self, definition: Dict, buffer_km: float = 0.0, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Create study area from center point and radius.
        
        Args:
            definition: Dict with 'center' [lon, lat] and 'radius_km'
            buffer_km: Additional buffer in kilometers
            crs: Coordinate reference system
        """
        center_lon, center_lat = definition['center']
        radius_km = definition['radius_km'] + buffer_km
        
        # Create point and buffer
        center_point = Point(center_lon, center_lat)
        gdf = gpd.GeoDataFrame([{'geometry': center_point}], crs=crs)
        
        # Convert to UTM for accurate buffering
        utm_crs = gdf.estimate_utm_crs()
        gdf_utm = gdf.to_crs(utm_crs)
        gdf_utm['geometry'] = gdf_utm.buffer(radius_km * 1000)  # Convert km to meters
        gdf = gdf_utm.to_crs(crs)
        
        return gdf


class PolygonFilter(GeographicFilter):
    """Filter by custom polygon geometry."""
    
    def create_study_area(self, definition: Union[str, Dict], buffer_km: float = 0.0, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Create study area from polygon definition.
        
        Args:
            definition: WKT string or dict with coordinates
            buffer_km: Buffer around polygon in kilometers
            crs: Coordinate reference system
        """
        if isinstance(definition, str):
            # WKT string
            from shapely import wkt
            polygon = wkt.loads(definition)
        else:
            # Coordinates dict
            coords = definition['coordinates']
            polygon = Polygon(coords)
        
        gdf = gpd.GeoDataFrame([{'geometry': polygon}], crs=crs)
        
        if buffer_km > 0:
            # Convert to UTM for accurate buffering
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            gdf_utm['geometry'] = gdf_utm.buffer(buffer_km * 1000)
            gdf = gdf_utm.to_crs(crs)
        
        return gdf


class ShapefileFilter(GeographicFilter):
    """Filter by shapefile boundaries."""
    
    def create_study_area(self, definition: str, buffer_km: float = 0.0, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Create study area from shapefile.
        
        Args:
            definition: Path to shapefile
            buffer_km: Buffer around geometries in kilometers
            crs: Target coordinate reference system
        """
        gdf = gpd.read_file(definition)
        
        if buffer_km > 0:
            # Convert to UTM for accurate buffering
            utm_crs = gdf.estimate_utm_crs()
            gdf_utm = gdf.to_crs(utm_crs)
            gdf_utm['geometry'] = gdf_utm.buffer(buffer_km * 1000)
            gdf = gdf_utm.to_crs(crs)
        
        # Ensure target CRS
        if gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        
        return gdf


class StudyAreaFilter:
    """Main class for filtering study areas with adaptive resolution."""
    
    def __init__(self, config: StudyAreaConfig):
        """
        Initialize study area filter.
        
        Args:
            config: Study area configuration
        """
        self.config = config
        self.study_area_geometry = None
        self._setup_geographic_filter()
        
        logger.info(f"Initialized StudyAreaFilter: {config.name}")
        logger.info(f"Geographic bounds: {config.geographic_bounds.bounds_type}")
        logger.info(f"Resolution rules: {len(config.resolution_rules)}")
    
    def _setup_geographic_filter(self):
        """Setup the appropriate geographic filter based on bounds type."""
        bounds_type = self.config.geographic_bounds.bounds_type
        
        filter_map = {
            'bbox': BoundingBoxFilter(),
            'counties': CountyFilter(),
            'circle': CircularFilter(),
            'polygon': PolygonFilter(),
            'shapefile': ShapefileFilter()
        }
        
        if bounds_type not in filter_map:
            raise ValueError(f"Unsupported bounds type: {bounds_type}")
        
        self.geographic_filter = filter_map[bounds_type]
    
    def create_study_area(self) -> gpd.GeoDataFrame:
        """Create study area geometry from configuration."""
        if self.study_area_geometry is None:
            self.study_area_geometry = self.geographic_filter.create_study_area(
                self.config.geographic_bounds.definition,
                self.config.geographic_bounds.buffer_km,
                self.config.geographic_bounds.crs
            )
        
        return self.study_area_geometry
    
    def determine_resolution(self, hex_properties: Dict[str, Any]) -> int:
        """
        Determine H3 resolution for a hexagon based on its properties.
        
        Args:
            hex_properties: Dictionary of hexagon properties (FSI_24, building_volume, etc.)
        
        Returns:
            Appropriate H3 resolution for this hexagon
        """
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.config.resolution_rules, key=lambda x: x.priority, reverse=True)
        
        # Evaluate rules in priority order
        for rule in sorted_rules:
            try:
                # Create safe evaluation context
                eval_context = {
                    **hex_properties,
                    'np': np,  # Allow numpy functions
                    '__builtins__': {}  # Restrict built-ins for security
                }
                
                if eval(rule.condition, eval_context):
                    logger.debug(f"Applied rule '{rule.name}': resolution {rule.resolution}")
                    return rule.resolution
                    
            except Exception as e:
                logger.warning(f"Error evaluating rule '{rule.name}': {e}")
                continue
        
        # Return default resolution if no rules match
        return self.config.default_resolution
    
    def filter_dataset(self, 
                      h3_data_dict: Dict[int, gpd.GeoDataFrame],
                      apply_density_filter: bool = True) -> Dict[int, gpd.GeoDataFrame]:
        """
        Filter H3 dataset to study area with adaptive resolution.
        
        Args:
            h3_data_dict: Dictionary mapping resolution to GeoDataFrame
            apply_density_filter: Whether to apply density-based filtering
        
        Returns:
            Filtered H3 data with adaptive resolution
        """
        study_area = self.create_study_area()
        logger.info(f"Filtering to study area: {len(study_area)} polygons")
        
        filtered_data = {}
        resolution_assignments = {}  # Track which hexagons get which resolution
        
        # First pass: determine spatial intersection
        base_resolution = min(h3_data_dict.keys())  # Use lowest resolution as base
        base_data = h3_data_dict[base_resolution]
        
        # Spatial intersection with study area
        intersecting_hexes = gpd.sjoin(
            base_data.reset_index(),
            study_area,
            how='inner',
            predicate='intersects'
        ).set_index('region_id' if 'region_id' in base_data.columns else base_data.index.name or 'index')
        
        logger.info(f"Found {len(intersecting_hexes)} hexagons intersecting study area")
        
        if len(intersecting_hexes) == 0:
            logger.warning("No hexagons intersect with study area")
            return {}
        
        # Second pass: apply density filtering and resolution rules
        for resolution, gdf in h3_data_dict.items():
            filtered_hexes = set()
            
            # Get hexagons that intersect study area at this resolution
            if resolution == base_resolution:
                candidate_hexes = intersecting_hexes
            else:
                # For higher resolutions, get children of intersecting base hexes
                candidate_hexes = []
                for base_hex in intersecting_hexes.index:
                    if resolution > base_resolution:
                        # Get children
                        children = list(h3.cell_to_children(base_hex, resolution))
                        candidate_hexes.extend([h for h in children if h in gdf.index])
                    elif resolution < base_resolution:
                        # Get parent
                        parent = h3.cell_to_parent(base_hex, resolution)
                        if parent in gdf.index:
                            candidate_hexes.append(parent)
                
                candidate_hexes = gdf.loc[gdf.index.isin(candidate_hexes)]
            
            # Apply resolution rules to each candidate hexagon
            for hex_id, hex_row in candidate_hexes.iterrows():
                hex_properties = hex_row.to_dict()
                
                # Apply density filtering if enabled
                if apply_density_filter and 'FSI_24' in hex_properties:
                    if hex_properties['FSI_24'] < self.config.min_density_threshold:
                        continue
                
                # Determine appropriate resolution for this hexagon
                target_resolution = self.determine_resolution(hex_properties)
                
                # If this resolution matches the target, include it
                if resolution == target_resolution:
                    filtered_hexes.add(hex_id)
                    resolution_assignments[hex_id] = resolution
            
            if filtered_hexes:
                filtered_data[resolution] = gdf.loc[gdf.index.isin(filtered_hexes)].copy()
                logger.info(f"Resolution {resolution}: {len(filtered_data[resolution])} hexagons")
        
        # Log resolution assignment statistics
        if resolution_assignments:
            resolution_counts = pd.Series(list(resolution_assignments.values())).value_counts()
            logger.info("Resolution assignments:")
            for res in sorted(resolution_counts.index):
                logger.info(f"  Resolution {res}: {resolution_counts[res]} hexagons")
        
        return filtered_data
    
    def get_processing_chunks(self, 
                            filtered_data: Dict[int, gpd.GeoDataFrame],
                            memory_limit_gb: Optional[float] = None) -> List[Dict]:
        """
        Create processing_modalities chunks for memory-aware computation.
        
        Args:
            filtered_data: Filtered H3 data
            memory_limit_gb: Memory limit for each chunk
        
        Returns:
            List of chunk definitions
        """
        if not self.config.enable_chunking:
            return [{'chunk_id': 0, 'data': filtered_data, 'hexagon_count': sum(len(gdf) for gdf in filtered_data.values())}]
        
        memory_limit = memory_limit_gb or self.config.max_memory_gb
        max_hexagons = self.config.max_hexagons_per_chunk
        
        # Estimate memory usage per hexagon (rough heuristic)
        bytes_per_hexagon = 1024  # Approximate memory usage
        max_hexagons_by_memory = int((memory_limit * 1024**3) / bytes_per_hexagon)
        effective_max_hexagons = min(max_hexagons, max_hexagons_by_memory)
        
        total_hexagons = sum(len(gdf) for gdf in filtered_data.values())
        
        if total_hexagons <= effective_max_hexagons:
            return [{'chunk_id': 0, 'data': filtered_data, 'hexagon_count': total_hexagons}]
        
        # Create chunks by spatial proximity
        chunks = []
        chunk_id = 0
        
        # Use lowest resolution as chunking basis
        base_resolution = min(filtered_data.keys())
        base_data = filtered_data[base_resolution]
        
        # Simple grid-based chunking
        bounds = base_data.total_bounds
        n_chunks = int(np.ceil(total_hexagons / effective_max_hexagons))
        grid_size = int(np.ceil(np.sqrt(n_chunks)))
        
        x_step = (bounds[2] - bounds[0]) / grid_size
        y_step = (bounds[3] - bounds[1]) / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                chunk_bounds = [
                    bounds[0] + i * x_step,
                    bounds[1] + j * y_step,
                    bounds[0] + (i + 1) * x_step,
                    bounds[1] + (j + 1) * y_step
                ]
                
                chunk_data = {}
                chunk_hexagon_count = 0
                
                for resolution, gdf in filtered_data.items():
                    # Filter to chunk bounds
                    chunk_gdf = gdf.cx[chunk_bounds[0]:chunk_bounds[2], 
                                       chunk_bounds[1]:chunk_bounds[3]]
                    
                    if len(chunk_gdf) > 0:
                        chunk_data[resolution] = chunk_gdf
                        chunk_hexagon_count += len(chunk_gdf)
                
                if chunk_data:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'bounds': chunk_bounds,
                        'data': chunk_data,
                        'hexagon_count': chunk_hexagon_count
                    })
                    chunk_id += 1
        
        logger.info(f"Created {len(chunks)} processing_modalities chunks")
        logger.info(f"Average hexagons per chunk: {total_hexagons / len(chunks):.0f}")
        
        return chunks
    
    def get_statistics(self, filtered_data: Dict[int, gpd.GeoDataFrame]) -> Dict[str, Any]:
        """
        Get statistics about the filtered study area.
        
        Args:
            filtered_data: Filtered H3 data
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'study_area_name': self.config.name,
            'total_hexagons': sum(len(gdf) for gdf in filtered_data.values()),
            'resolutions': {res: len(gdf) for res, gdf in filtered_data.items()},
            'geographic_bounds_type': self.config.geographic_bounds.bounds_type,
            'resolution_rules_count': len(self.config.resolution_rules)
        }
        
        # Calculate area coverage if geometry available
        study_area = self.create_study_area()
        if len(study_area) > 0:
            # Convert to appropriate CRS for area calculation
            utm_crs = study_area.estimate_utm_crs()
            study_area_utm = study_area.to_crs(utm_crs)
            total_area_km2 = study_area_utm.area.sum() / 1e6  # Convert m² to km²
            stats['study_area_km2'] = total_area_km2
        
        # Density statistics if available
        for resolution, gdf in filtered_data.items():
            if 'FSI_24' in gdf.columns:
                fsi_values = gdf['FSI_24']
                stats[f'resolution_{resolution}_density'] = {
                    'min_fsi': float(fsi_values.min()),
                    'max_fsi': float(fsi_values.max()),
                    'mean_fsi': float(fsi_values.mean()),
                    'median_fsi': float(fsi_values.median())
                }
        
        return stats