#!/usr/bin/env python3
"""
Interactive CLI tool to create new study area configurations.

This script provides an interactive interface for creating study area
configurations for the UrbanRepML filtering system, with support for
bioregional categories aligned with GEO-INFER goals.

Usage:
    python create_study_area.py
    python create_study_area.py --template agricultural
    python create_study_area.py --batch --config batch_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
import inquirer
from inquirer import questions

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from urban_embedding.study_area_filter import StudyAreaConfig, GeographicBounds, BioregionalContext, ResolutionRule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudyAreaCreator:
    """Interactive study area configuration creator."""
    
    # Available bioregion types and their characteristics
    BIOREGION_TYPES = {
        'agriculture': {
            'description': 'Agricultural areas focused on crop production and sustainable farming',
            'typical_ecosystems': ['agricultural', 'mixed_use'],
            'management_focus': ['crops', 'water', 'carbon', 'biodiversity']
        },
        'forestry': {
            'description': 'Forest management areas for timber, conservation, and ecosystem services',
            'typical_ecosystems': ['conifer_forest', 'mixed_forest', 'oak_woodland'],
            'management_focus': ['timber', 'carbon', 'biodiversity', 'water']
        },
        'watershed': {
            'description': 'Water management areas spanning multiple land uses',
            'typical_ecosystems': ['mixed_use', 'riparian'],
            'management_focus': ['water', 'biodiversity', 'flood_control', 'drought_resilience']
        },
        'conservation': {
            'description': 'High-priority conservation areas and protected ecosystems',
            'typical_ecosystems': ['old_growth', 'wetland', 'prairie', 'alpine'],
            'management_focus': ['biodiversity', 'carbon', 'climate_adaptation']
        },
        'mixed_use': {
            'description': 'Areas with multiple land uses requiring integrated management',
            'typical_ecosystems': ['mixed_use', 'agricultural', 'forest_edge'],
            'management_focus': ['integration', 'sustainability', 'multi_use']
        }
    }
    
    # Predefined region templates
    REGION_TEMPLATES = {
        'cascadia_counties': {
            'california_counties': ['Butte', 'Colusa', 'Del Norte', 'Glenn', 'Humboldt', 'Lake', 
                                  'Lassen', 'Mendocino', 'Modoc', 'Nevada', 'Plumas', 'Shasta', 
                                  'Sierra', 'Siskiyou', 'Tehama', 'Trinity'],
            'oregon_counties': 'all'  # All 36 Oregon counties
        }
    }
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize study area creator.
        
        Args:
            output_dir: Directory to save study area configurations
        """
        self.output_dir = output_dir or (project_root / "config" / "study_areas")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Study area creator initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_interactive(self) -> StudyAreaConfig:
        """Run interactive study area creation process."""
        print("\n" + "="*60)
        print("🌍 UrbanRepML Study Area Configuration Creator")
        print("="*60)
        print("Create a new study area configuration for bioregional analysis")
        print()
        
        # Basic information
        config = self._collect_basic_info()
        
        # Geographic bounds
        config.geographic_bounds = self._collect_geographic_bounds()
        
        # Bioregional context
        if self._ask_yes_no("Include bioregional context (recommended for GEO-INFER)?"):
            config.bioregional_context = self._collect_bioregional_context()
        
        # Resolution rules
        if self._ask_yes_no("Add custom resolution rules?"):
            config.resolution_rules = self._collect_resolution_rules()
        
        # Computational settings
        config = self._collect_computational_settings(config)
        
        # Save configuration
        self._save_configuration(config)
        
        return config
    
    def _collect_basic_info(self) -> StudyAreaConfig:
        """Collect basic study area information."""
        print("\n📋 Basic Information")
        print("-" * 20)
        
        name = input("Study area name (lowercase with underscores): ").strip()
        while not name or not name.replace('_', '').replace('-', '').isalnum():
            print("❌ Name must be alphanumeric with underscores/hyphens only")
            name = input("Study area name: ").strip()
        
        description = input("Description: ").strip()
        while not description:
            description = input("Description (required): ").strip()
        
        return StudyAreaConfig(
            name=name,
            description=description,
            geographic_bounds=None  # Will be set later
        )
    
    def _collect_geographic_bounds(self) -> GeographicBounds:
        """Collect geographic bounds information."""
        print("\n🗺️  Geographic Bounds")
        print("-" * 20)
        
        bounds_type = inquirer.list_input(
            "Select bounds type",
            choices=[
                ('Bounding box (lat/lon coordinates)', 'bbox'),
                ('Counties', 'counties'),
                ('Circular region (center + radius)', 'circle'),
                ('Custom polygon', 'polygon'),
                ('Shapefile', 'shapefile')
            ]
        )
        
        definition = self._collect_bounds_definition(bounds_type)
        
        buffer_km = float(input("Buffer distance (km, default 0): ") or "0")
        
        return GeographicBounds(
            bounds_type=bounds_type,
            definition=definition,
            buffer_km=buffer_km,
            crs="EPSG:4326"
        )
    
    def _collect_bounds_definition(self, bounds_type: str) -> Any:
        """Collect bounds definition based on type."""
        if bounds_type == 'bbox':
            print("Enter bounding box coordinates:")
            return {
                'north': float(input("  North latitude: ")),
                'south': float(input("  South latitude: ")),
                'east': float(input("  East longitude: ")),
                'west': float(input("  West longitude: "))
            }
        
        elif bounds_type == 'counties':
            print("Available county templates:")
            for name, template in self.REGION_TEMPLATES.items():
                print(f"  - {name}")
            
            use_template = input("Use template (enter name) or custom counties? [template/custom]: ")
            
            if use_template in self.REGION_TEMPLATES:
                ca_counties = self.REGION_TEMPLATES[use_template].get('california_counties', [])
                or_counties = self.REGION_TEMPLATES[use_template].get('oregon_counties', [])
                
                counties = []
                if ca_counties:
                    counties.extend(ca_counties)
                if or_counties == 'all':
                    counties.append('all_oregon')
                elif or_counties:
                    counties.extend(or_counties)
                    
                return counties
            else:
                counties_str = input("Enter county names (comma-separated): ")
                return [c.strip() for c in counties_str.split(',')]
        
        elif bounds_type == 'circle':
            center_lon = float(input("Center longitude: "))
            center_lat = float(input("Center latitude: "))
            radius_km = float(input("Radius (km): "))
            return {
                'center': [center_lon, center_lat],
                'radius_km': radius_km
            }
        
        elif bounds_type == 'polygon':
            wkt_string = input("Enter WKT polygon string: ")
            return wkt_string
        
        elif bounds_type == 'shapefile':
            shapefile_path = input("Enter shapefile path: ")
            return shapefile_path
        
        else:
            raise ValueError(f"Unsupported bounds type: {bounds_type}")
    
    def _collect_bioregional_context(self) -> BioregionalContext:
        """Collect bioregional context information."""
        print("\n🌱 Bioregional Context")
        print("-" * 20)
        
        # Bioregion type
        bioregion_type = inquirer.list_input(
            "Select bioregion type",
            choices=[(f"{k}: {v['description']}", k) for k, v in self.BIOREGION_TYPES.items()]
        )
        
        template = self.BIOREGION_TYPES[bioregion_type]
        
        # Primary ecosystem
        ecosystem_choices = template['typical_ecosystems'] + ['other']
        primary_ecosystem = inquirer.list_input(
            "Primary ecosystem",
            choices=ecosystem_choices
        )
        
        if primary_ecosystem == 'other':
            primary_ecosystem = input("Enter custom ecosystem type: ")
        
        # Management focus
        available_focus = template['management_focus'] + ['other']
        management_focus = inquirer.checkbox(
            "Management focus areas (select multiple)",
            choices=available_focus
        )
        
        if 'other' in management_focus:
            management_focus.remove('other')
            custom_focus = input("Enter custom management focus: ")
            management_focus.append(custom_focus)
        
        # Agricultural details if relevant
        primary_crops = []
        farming_type = None
        water_source = None
        
        if bioregion_type in ['agriculture', 'mixed_use']:
            if self._ask_yes_no("Specify agricultural details?"):
                crops_str = input("Primary crops (comma-separated): ")
                primary_crops = [c.strip() for c in crops_str.split(',') if c.strip()]
                
                farming_type = inquirer.list_input(
                    "Farming type",
                    choices=['conventional', 'organic', 'regenerative', 'mixed']
                )
                
                water_source = inquirer.list_input(
                    "Water source",
                    choices=['rainfed', 'irrigated', 'mixed']
                )
        
        # Forest details if relevant
        forest_type = None
        timber_management = None
        
        if bioregion_type in ['forestry', 'conservation', 'mixed_use']:
            if self._ask_yes_no("Specify forest details?"):
                forest_type = inquirer.list_input(
                    "Forest type",
                    choices=['old_growth', 'second_growth', 'plantation', 'mixed']
                )
                
                timber_management = inquirer.list_input(
                    "Timber management",
                    choices=['sustainable', 'intensive', 'conservation', 'none']
                )
        
        return BioregionalContext(
            bioregion_type=bioregion_type,
            primary_ecosystem=primary_ecosystem,
            management_focus=management_focus,
            primary_crops=primary_crops,
            farming_type=farming_type,
            water_source=water_source,
            forest_type=forest_type,
            timber_management=timber_management
        )
    
    def _collect_resolution_rules(self) -> List[ResolutionRule]:
        """Collect custom resolution rules."""
        print("\n⚙️  Resolution Rules")
        print("-" * 20)
        print("Define rules for adaptive H3 resolution based on area characteristics")
        print("Available variables: FSI_24, building_volume, in_study_area")
        print("Example condition: 'FSI_24 >= 0.1 and building_volume > 1000'")
        print()
        
        rules = []
        
        while True:
            print(f"\nRule #{len(rules) + 1}")
            
            name = input("Rule name: ").strip()
            if not name:
                break
            
            condition = input("Condition (Python expression): ").strip()
            if not condition:
                break
            
            try:
                resolution = int(input("H3 resolution (7-11): "))
                if not 7 <= resolution <= 11:
                    print("❌ Resolution must be between 7 and 11")
                    continue
            except ValueError:
                print("❌ Resolution must be an integer")
                continue
            
            try:
                priority = int(input("Priority (higher = more important, default 50): ") or "50")
            except ValueError:
                priority = 50
            
            description = input("Description (optional): ").strip()
            
            rules.append(ResolutionRule(
                name=name,
                condition=condition,
                resolution=resolution,
                priority=priority,
                description=description
            ))
            
            if not self._ask_yes_no("Add another rule?"):
                break
        
        return rules
    
    def _collect_computational_settings(self, config: StudyAreaConfig) -> StudyAreaConfig:
        """Collect computational constraint settings."""
        print("\n💾 Computational Settings")
        print("-" * 20)
        
        try:
            config.max_memory_gb = float(input("Max memory (GB, default 16): ") or "16")
        except ValueError:
            config.max_memory_gb = 16.0
        
        try:
            config.max_hexagons_per_chunk = int(input("Max hexagons per chunk (default 100000): ") or "100000")
        except ValueError:
            config.max_hexagons_per_chunk = 100000
        
        config.enable_chunking = self._ask_yes_no("Enable chunking for large areas?", default=True)
        
        try:
            config.min_coverage_threshold = float(input("Min coverage threshold (0-1, default 0.8): ") or "0.8")
        except ValueError:
            config.min_coverage_threshold = 0.8
        
        try:
            config.min_density_threshold = float(input("Min density threshold (default 0.0): ") or "0.0")
        except ValueError:
            config.min_density_threshold = 0.0
        
        return config
    
    def _ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask yes/no question with default."""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{question} [{default_str}]: ").strip().lower()
        
        if not response:
            return default
        
        return response in ['y', 'yes', 'true', '1']
    
    def _save_configuration(self, config: StudyAreaConfig):
        """Save configuration to YAML file."""
        output_path = self.output_dir / f"{config.name}.yaml"
        
        # Check if file exists
        if output_path.exists():
            overwrite = self._ask_yes_no(f"File {output_path} exists. Overwrite?")
            if not overwrite:
                print("❌ Configuration not saved")
                return
        
        config.save_yaml(output_path)
        
        print(f"\n✅ Study area configuration saved to: {output_path}")
        print(f"📁 Use this configuration with: --study_area {config.name}")
    
    def create_from_template(self, template_type: str) -> StudyAreaConfig:
        """Create study area from predefined template."""
        templates = {
            'agricultural': self._create_agricultural_template,
            'forestry': self._create_forestry_template,
            'watershed': self._create_watershed_template,
            'conservation': self._create_conservation_template
        }
        
        if template_type not in templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        return templates[template_type]()
    
    def _create_agricultural_template(self) -> StudyAreaConfig:
        """Create agricultural study area template."""
        name = input("Study area name: ").strip()
        
        return StudyAreaConfig(
            name=name,
            description=f"Agricultural study area: {name}",
            geographic_bounds=GeographicBounds(
                bounds_type='bbox',
                definition={'north': 45.0, 'south': 44.0, 'west': -123.0, 'east': -122.0},
                buffer_km=1.0
            ),
            bioregional_context=BioregionalContext(
                bioregion_type='agriculture',
                primary_ecosystem='agricultural',
                management_focus=['crops', 'water', 'sustainability'],
                farming_type='mixed',
                water_source='mixed'
            ),
            resolution_rules=[
                ResolutionRule(
                    name='intensive_agriculture',
                    condition='FSI_24 >= 0.1',
                    resolution=9,
                    priority=90,
                    description='Intensive agricultural areas'
                )
            ],
            default_resolution=8,
            max_memory_gb=12.0,
            max_hexagons_per_chunk=50000
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create study area configurations")
    parser.add_argument('--template', choices=['agricultural', 'forestry', 'watershed', 'conservation'],
                      help="Create from template")
    parser.add_argument('--output-dir', type=Path, help="Output directory for configurations")
    
    args = parser.parse_args()
    
    creator = StudyAreaCreator(output_dir=args.output_dir)
    
    try:
        if args.template:
            config = creator.create_from_template(args.template)
        else:
            config = creator.run_interactive()
        
        print(f"\n🎉 Successfully created study area: {config.name}")
        
    except KeyboardInterrupt:
        print("\n\n❌ Configuration creation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error creating study area: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()