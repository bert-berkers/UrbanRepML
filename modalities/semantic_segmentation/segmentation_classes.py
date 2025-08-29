"""
Semantic segmentation classes for Netherlands land use/land cover.

Defines the categorical variables for segmentation outputs,
optimized for Dutch urban and rural landscapes.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


class NetherlandsLandCover(Enum):
    """
    Land cover classes for Netherlands semantic segmentation.
    
    Based on Dutch land use statistics (CBS) and optimized for urban analysis.
    """
    # Water bodies
    WATER = 0
    CANALS = 1
    
    # Urban areas
    RESIDENTIAL_DENSE = 2
    RESIDENTIAL_SPARSE = 3
    COMMERCIAL = 4
    INDUSTRIAL = 5
    INFRASTRUCTURE = 6
    
    # Green spaces
    PARKS = 7
    FORESTS = 8
    AGRICULTURE_ARABLE = 9
    AGRICULTURE_GRASSLAND = 10
    GREENHOUSES = 11
    
    # Transportation
    ROADS_MAJOR = 12
    ROADS_MINOR = 13
    RAILWAYS = 14
    AIRPORTS = 15
    
    # Specialized Dutch features
    POLDERS = 16
    DIKES = 17
    WIND_FARMS = 18
    PORT_AREAS = 19
    
    # Natural/semi-natural
    BEACHES = 20
    DUNES = 21
    WETLANDS = 22
    
    # Other
    CONSTRUCTION = 23
    UNKNOWN = 24


@dataclass
class ClassMetadata:
    """Metadata for each segmentation class."""
    id: int
    name: str
    color: Tuple[int, int, int]  # RGB color for visualization
    description: str
    priority: int  # Higher priority classes take precedence in conflicts
    is_urban: bool


class SegmentationClasses:
    """
    Management class for semantic segmentation categories.
    
    Handles class definitions, hierarchical relationships,
    and conversion between different representations.
    """
    
    # Color palette for visualization (RGB)
    CLASS_COLORS = {
        NetherlandsLandCover.WATER: (0, 100, 200),
        NetherlandsLandCover.CANALS: (50, 150, 220),
        NetherlandsLandCover.RESIDENTIAL_DENSE: (200, 100, 100),
        NetherlandsLandCover.RESIDENTIAL_SPARSE: (220, 150, 150),
        NetherlandsLandCover.COMMERCIAL: (150, 50, 150),
        NetherlandsLandCover.INDUSTRIAL: (100, 100, 100),
        NetherlandsLandCover.INFRASTRUCTURE: (80, 80, 80),
        NetherlandsLandCover.PARKS: (100, 200, 100),
        NetherlandsLandCover.FORESTS: (50, 150, 50),
        NetherlandsLandCover.AGRICULTURE_ARABLE: (200, 200, 100),
        NetherlandsLandCover.AGRICULTURE_GRASSLAND: (150, 200, 100),
        NetherlandsLandCover.GREENHOUSES: (200, 150, 200),
        NetherlandsLandCover.ROADS_MAJOR: (60, 60, 60),
        NetherlandsLandCover.ROADS_MINOR: (120, 120, 120),
        NetherlandsLandCover.RAILWAYS: (80, 40, 20),
        NetherlandsLandCover.AIRPORTS: (150, 150, 200),
        NetherlandsLandCover.POLDERS: (180, 180, 150),
        NetherlandsLandCover.DIKES: (100, 80, 60),
        NetherlandsLandCover.WIND_FARMS: (200, 200, 200),
        NetherlandsLandCover.PORT_AREAS: (100, 120, 140),
        NetherlandsLandCover.BEACHES: (220, 200, 150),
        NetherlandsLandCover.DUNES: (200, 180, 120),
        NetherlandsLandCover.WETLANDS: (100, 150, 120),
        NetherlandsLandCover.CONSTRUCTION: (150, 100, 50),
        NetherlandsLandCover.UNKNOWN: (128, 128, 128)
    }
    
    # Class priorities for conflict resolution
    CLASS_PRIORITIES = {
        NetherlandsLandCover.WATER: 10,
        NetherlandsLandCover.CANALS: 9,
        NetherlandsLandCover.ROADS_MAJOR: 8,
        NetherlandsLandCover.RAILWAYS: 8,
        NetherlandsLandCover.INFRASTRUCTURE: 7,
        NetherlandsLandCover.RESIDENTIAL_DENSE: 6,
        NetherlandsLandCover.COMMERCIAL: 6,
        NetherlandsLandCover.INDUSTRIAL: 6,
        NetherlandsLandCover.RESIDENTIAL_SPARSE: 5,
        NetherlandsLandCover.PARKS: 4,
        NetherlandsLandCover.FORESTS: 4,
        NetherlandsLandCover.AGRICULTURE_ARABLE: 3,
        NetherlandsLandCover.AGRICULTURE_GRASSLAND: 3,
        NetherlandsLandCover.GREENHOUSES: 5,
        NetherlandsLandCover.ROADS_MINOR: 4,
        NetherlandsLandCover.AIRPORTS: 7,
        NetherlandsLandCover.POLDERS: 2,
        NetherlandsLandCover.DIKES: 8,
        NetherlandsLandCover.WIND_FARMS: 3,
        NetherlandsLandCover.PORT_AREAS: 6,
        NetherlandsLandCover.BEACHES: 3,
        NetherlandsLandCover.DUNES: 3,
        NetherlandsLandCover.WETLANDS: 4,
        NetherlandsLandCover.CONSTRUCTION: 5,
        NetherlandsLandCover.UNKNOWN: 1
    }
    
    # Urban vs non-urban classification
    URBAN_CLASSES = {
        NetherlandsLandCover.RESIDENTIAL_DENSE,
        NetherlandsLandCover.RESIDENTIAL_SPARSE,
        NetherlandsLandCover.COMMERCIAL,
        NetherlandsLandCover.INDUSTRIAL,
        NetherlandsLandCover.INFRASTRUCTURE,
        NetherlandsLandCover.ROADS_MAJOR,
        NetherlandsLandCover.ROADS_MINOR,
        NetherlandsLandCover.RAILWAYS,
        NetherlandsLandCover.CONSTRUCTION
    }
    
    @classmethod
    def get_class_metadata(cls) -> Dict[NetherlandsLandCover, ClassMetadata]:
        """Get complete metadata for all classes."""
        metadata = {}
        
        for land_cover in NetherlandsLandCover:
            metadata[land_cover] = ClassMetadata(
                id=land_cover.value,
                name=land_cover.name.lower().replace('_', ' '),
                color=cls.CLASS_COLORS[land_cover],
                description=cls._get_class_description(land_cover),
                priority=cls.CLASS_PRIORITIES[land_cover],
                is_urban=land_cover in cls.URBAN_CLASSES
            )
        
        return metadata
    
    @classmethod
    def _get_class_description(cls, land_cover: NetherlandsLandCover) -> str:
        """Get detailed description for a land cover class."""
        descriptions = {
            NetherlandsLandCover.WATER: "Natural water bodies, lakes, rivers",
            NetherlandsLandCover.CANALS: "Artificial waterways, urban canals",
            NetherlandsLandCover.RESIDENTIAL_DENSE: "High-density urban housing",
            NetherlandsLandCover.RESIDENTIAL_SPARSE: "Low-density suburban housing", 
            NetherlandsLandCover.COMMERCIAL: "Shopping, offices, services",
            NetherlandsLandCover.INDUSTRIAL: "Manufacturing, warehouses, industry",
            NetherlandsLandCover.INFRASTRUCTURE: "Utilities, facilities, civic buildings",
            NetherlandsLandCover.PARKS: "Urban parks, recreational areas",
            NetherlandsLandCover.FORESTS: "Natural and planted forests",
            NetherlandsLandCover.AGRICULTURE_ARABLE: "Crop fields, arable farming",
            NetherlandsLandCover.AGRICULTURE_GRASSLAND: "Pastures, meadows, grazing",
            NetherlandsLandCover.GREENHOUSES: "Horticultural greenhouse complexes",
            NetherlandsLandCover.ROADS_MAJOR: "Highways, main arterials", 
            NetherlandsLandCover.ROADS_MINOR: "Local streets, residential roads",
            NetherlandsLandCover.RAILWAYS: "Train tracks, rail infrastructure",
            NetherlandsLandCover.AIRPORTS: "Airports, aviation facilities",
            NetherlandsLandCover.POLDERS: "Reclaimed land below sea level",
            NetherlandsLandCover.DIKES: "Sea defenses, flood barriers",
            NetherlandsLandCover.WIND_FARMS: "Wind energy installations",
            NetherlandsLandCover.PORT_AREAS: "Harbors, shipping facilities",
            NetherlandsLandCover.BEACHES: "Coastal sandy areas",
            NetherlandsLandCover.DUNES: "Coastal dune systems",
            NetherlandsLandCover.WETLANDS: "Natural marshes, wet meadows",
            NetherlandsLandCover.CONSTRUCTION: "Active construction sites",
            NetherlandsLandCover.UNKNOWN: "Unclassified or mixed areas"
        }
        
        return descriptions.get(land_cover, "No description available")
    
    @classmethod
    def get_num_classes(cls) -> int:
        """Get total number of segmentation classes."""
        return len(NetherlandsLandCover)
    
    @classmethod
    def get_class_weights(cls, distribution: Dict[int, float] = None) -> np.ndarray:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            distribution: Optional class distribution from training data
            
        Returns:
            Array of class weights
        """
        num_classes = cls.get_num_classes()
        
        if distribution is None:
            # Default weights based on expected Netherlands distribution
            default_weights = {
                NetherlandsLandCover.WATER.value: 1.2,
                NetherlandsLandCover.CANALS.value: 3.0,
                NetherlandsLandCover.RESIDENTIAL_DENSE.value: 1.5,
                NetherlandsLandCover.RESIDENTIAL_SPARSE.value: 1.0,
                NetherlandsLandCover.COMMERCIAL.value: 2.0,
                NetherlandsLandCover.INDUSTRIAL.value: 1.8,
                NetherlandsLandCover.INFRASTRUCTURE.value: 3.5,
                NetherlandsLandCover.PARKS.value: 2.5,
                NetherlandsLandCover.FORESTS.value: 1.5,
                NetherlandsLandCover.AGRICULTURE_ARABLE.value: 0.8,
                NetherlandsLandCover.AGRICULTURE_GRASSLAND.value: 0.6,
                NetherlandsLandCover.GREENHOUSES.value: 4.0,
                NetherlandsLandCover.ROADS_MAJOR.value: 5.0,
                NetherlandsLandCover.ROADS_MINOR.value: 3.0,
                NetherlandsLandCover.RAILWAYS.value: 8.0,
                NetherlandsLandCover.AIRPORTS.value: 10.0,
                NetherlandsLandCover.POLDERS.value: 1.0,
                NetherlandsLandCover.DIKES.value: 6.0,
                NetherlandsLandCover.WIND_FARMS.value: 12.0,
                NetherlandsLandCover.PORT_AREAS.value: 4.0,
                NetherlandsLandCover.BEACHES.value: 3.0,
                NetherlandsLandCover.DUNES.value: 2.5,
                NetherlandsLandCover.WETLANDS.value: 3.5,
                NetherlandsLandCover.CONSTRUCTION.value: 5.0,
                NetherlandsLandCover.UNKNOWN.value: 0.1
            }
            weights = np.array([default_weights[i] for i in range(num_classes)])
        else:
            # Calculate inverse frequency weights
            total_samples = sum(distribution.values())
            weights = np.zeros(num_classes)
            for class_id, count in distribution.items():
                weights[class_id] = total_samples / (num_classes * count)
        
        return weights
    
    @classmethod
    def create_colormap(cls) -> np.ndarray:
        """Create colormap for visualization."""
        num_classes = cls.get_num_classes()
        colormap = np.zeros((num_classes, 3), dtype=np.uint8)
        
        for land_cover in NetherlandsLandCover:
            colormap[land_cover.value] = cls.CLASS_COLORS[land_cover]
        
        return colormap
    
    @classmethod
    def get_hierarchical_groups(cls) -> Dict[str, List[NetherlandsLandCover]]:
        """Get hierarchical groupings of classes."""
        return {
            'water': [
                NetherlandsLandCover.WATER,
                NetherlandsLandCover.CANALS,
                NetherlandsLandCover.WETLANDS
            ],
            'urban': [
                NetherlandsLandCover.RESIDENTIAL_DENSE,
                NetherlandsLandCover.RESIDENTIAL_SPARSE,
                NetherlandsLandCover.COMMERCIAL,
                NetherlandsLandCover.INDUSTRIAL,
                NetherlandsLandCover.INFRASTRUCTURE
            ],
            'green': [
                NetherlandsLandCover.PARKS,
                NetherlandsLandCover.FORESTS,
                NetherlandsLandCover.AGRICULTURE_ARABLE,
                NetherlandsLandCover.AGRICULTURE_GRASSLAND,
                NetherlandsLandCover.GREENHOUSES
            ],
            'transport': [
                NetherlandsLandCover.ROADS_MAJOR,
                NetherlandsLandCover.ROADS_MINOR,
                NetherlandsLandCover.RAILWAYS,
                NetherlandsLandCover.AIRPORTS
            ],
            'coastal': [
                NetherlandsLandCover.BEACHES,
                NetherlandsLandCover.DUNES,
                NetherlandsLandCover.DIKES
            ],
            'specialized': [
                NetherlandsLandCover.POLDERS,
                NetherlandsLandCover.WIND_FARMS,
                NetherlandsLandCover.PORT_AREAS,
                NetherlandsLandCover.CONSTRUCTION
            ]
        }
    
    @classmethod
    def class_id_to_name(cls, class_id: int) -> str:
        """Convert class ID to human-readable name."""
        try:
            land_cover = NetherlandsLandCover(class_id)
            return land_cover.name.lower().replace('_', ' ')
        except ValueError:
            return "unknown"
    
    @classmethod
    def apply_to_segmentation_map(cls, seg_map: np.ndarray) -> np.ndarray:
        """
        Apply class definitions to a segmentation map.
        
        Args:
            seg_map: Raw segmentation output (H, W) with class IDs
            
        Returns:
            Processed segmentation map with proper class assignments
        """
        # Ensure values are within valid range
        seg_map = np.clip(seg_map, 0, cls.get_num_classes() - 1)
        
        # Convert unknown/invalid classes to UNKNOWN
        invalid_mask = seg_map >= cls.get_num_classes()
        seg_map[invalid_mask] = NetherlandsLandCover.UNKNOWN.value
        
        return seg_map