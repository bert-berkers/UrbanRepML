#!/usr/bin/env python3
"""
List and analyze available study area configurations.

This script lists all available study area configurations and provides
summary statistics and analysis capabilities.

Usage:
    python list_study_areas.py
    python list_study_areas.py --detailed
    python list_study_areas.py --bioregion agriculture
    python list_study_areas.py --validate
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json

import yaml
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from stage2_fusion.study_area_filter import StudyAreaConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudyAreaLister:
    """List and analyze study area configurations."""
    
    def __init__(self, config_dir: Path = None):
        """
        Initialize study area lister.
        
        Args:
            config_dir: Directory containing study area configurations
        """
        self.config_dir = config_dir or (project_root / "config" / "study_areas")
        
        if not self.config_dir.exists():
            logger.error(f"Study area config directory not found: {self.config_dir}")
            sys.exit(1)
        
        logger.info(f"Scanning study areas in: {self.config_dir}")
    
    def list_study_areas(self, 
                        bioregion_filter: Optional[str] = None,
                        detailed: bool = False,
                        validate: bool = False) -> List[Dict]:
        """
        List all study area configurations.
        
        Args:
            bioregion_filter: Filter by bioregion type
            detailed: Include detailed information
            validate: Validate configurations
        
        Returns:
            List of study area information dictionaries
        """
        yaml_files = list(self.config_dir.glob("*.yaml"))
        yaml_files.extend(self.config_dir.glob("*.yml"))
        
        study_areas = []
        
        for config_file in sorted(yaml_files):
            try:
                config = StudyAreaConfig.from_yaml(config_file)
                
                # Apply bioregion filter
                if bioregion_filter:
                    if not config.bioregional_context:
                        continue
                    if config.bioregional_context.bioregion_type != bioregion_filter:
                        continue
                
                # Validate if requested
                is_valid = True
                validation_errors = []
                if validate:
                    is_valid, validation_errors = self._validate_configuration(config)
                
                study_area_info = {
                    'name': config.name,
                    'description': config.description[:80] + "..." if len(config.description) > 80 else config.description,
                    'bounds_type': config.geographic_bounds.bounds_type,
                    'bioregion_type': config.bioregional_context.bioregion_type if config.bioregional_context else 'None',
                    'primary_ecosystem': config.bioregional_context.primary_ecosystem if config.bioregional_context else 'None',
                    'resolution_rules': len(config.resolution_rules),
                    'default_resolution': config.default_resolution,
                    'max_memory_gb': config.max_memory_gb,
                    'file': config_file.name,
                    'is_valid': is_valid,
                    'validation_errors': validation_errors
                }
                
                if detailed:
                    study_area_info.update({
                        'full_description': config.description,
                        'management_focus': config.bioregional_context.management_focus if config.bioregional_context else [],
                        'primary_crops': config.bioregional_context.primary_crops if config.bioregional_context else [],
                        'farming_type': config.bioregional_context.farming_type if config.bioregional_context else None,
                        'forest_type': config.bioregional_context.forest_type if config.bioregional_context else None,
                        'chunking_enabled': config.enable_chunking,
                        'max_hexagons_per_chunk': config.max_hexagons_per_chunk,
                        'min_coverage_threshold': config.min_coverage_threshold,
                        'min_density_threshold': config.min_density_threshold
                    })
                
                study_areas.append(study_area_info)
                
            except Exception as e:
                logger.warning(f"Failed to load {config_file}: {e}")
                study_areas.append({
                    'name': config_file.stem,
                    'description': f"ERROR: {str(e)}",
                    'bounds_type': 'ERROR',
                    'bioregion_type': 'ERROR',
                    'primary_ecosystem': 'ERROR',
                    'resolution_rules': 0,
                    'default_resolution': 0,
                    'max_memory_gb': 0,
                    'file': config_file.name,
                    'is_valid': False,
                    'validation_errors': [str(e)]
                })
        
        return study_areas
    
    def _validate_configuration(self, config: StudyAreaConfig) -> tuple:
        """
        Validate a study area configuration.
        
        Args:
            config: Study area configuration to validate
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate geographic bounds
        bounds = config.geographic_bounds
        if bounds.bounds_type == 'bbox':
            if not isinstance(bounds.definition, dict):
                errors.append("Bounding box definition must be a dictionary")
            else:
                required_keys = {'north', 'south', 'east', 'west'}
                if not required_keys.issubset(bounds.definition.keys()):
                    errors.append(f"Bounding box missing required keys: {required_keys - set(bounds.definition.keys())}")
                
                # Check coordinate validity
                try:
                    if bounds.definition['north'] <= bounds.definition['south']:
                        errors.append("North coordinate must be greater than south coordinate")
                    if bounds.definition['east'] <= bounds.definition['west']:
                        errors.append("East coordinate must be greater than west coordinate")
                except (KeyError, TypeError):
                    errors.append("Invalid coordinate values in bounding box")
        
        # Validate resolution rules
        for i, rule in enumerate(config.resolution_rules):
            if not 7 <= rule.resolution <= 11:
                errors.append(f"Rule {i+1} ({rule.name}): Resolution must be between 7-11, got {rule.resolution}")
            
            # Try to compile the condition (basic validation)
            try:
                compile(rule.condition, '<string>', 'eval')
            except SyntaxError as e:
                errors.append(f"Rule {i+1} ({rule.name}): Invalid condition syntax - {e}")
        
        # Validate memory settings
        if config.max_memory_gb <= 0:
            errors.append("max_memory_gb must be positive")
        
        if config.max_hexagons_per_chunk <= 0:
            errors.append("max_hexagons_per_chunk must be positive")
        
        # Validate thresholds
        if not 0 <= config.min_coverage_threshold <= 1:
            errors.append("min_coverage_threshold must be between 0 and 1")
        
        if config.min_density_threshold < 0:
            errors.append("min_density_threshold must be non-negative")
        
        return len(errors) == 0, errors
    
    def print_summary_table(self, study_areas: List[Dict], detailed: bool = False):
        """Print summary table of study areas."""
        if not study_areas:
            print("No study areas found.")
            return
        
        if detailed:
            headers = ['Name', 'Bioregion', 'Ecosystem', 'Bounds', 'Rules', 'Memory', 'Valid', 'File']
            table_data = []
            for sa in study_areas:
                status = "✅" if sa['is_valid'] else "❌"
                table_data.append([
                    sa['name'],
                    sa['bioregion_type'],
                    sa['primary_ecosystem'],
                    sa['bounds_type'],
                    sa['resolution_rules'],
                    f"{sa['max_memory_gb']}GB",
                    status,
                    sa['file']
                ])
        else:
            headers = ['Name', 'Description', 'Bioregion', 'Bounds', 'File']
            table_data = []
            for sa in study_areas:
                table_data.append([
                    sa['name'],
                    sa['description'],
                    sa['bioregion_type'],
                    sa['bounds_type'],
                    sa['file']
                ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def print_detailed_info(self, study_areas: List[Dict]):
        """Print detailed information for each study area."""
        for i, sa in enumerate(study_areas):
            if i > 0:
                print("\n" + "="*80)
            
            print(f"\n📍 {sa['name']}")
            print(f"📁 File: {sa['file']}")
            print(f"✅ Valid: {'Yes' if sa['is_valid'] else 'No'}")
            
            if not sa['is_valid']:
                print("❌ Validation Errors:")
                for error in sa['validation_errors']:
                    print(f"   • {error}")
            
            print(f"\n📖 Description:")
            print(f"   {sa.get('full_description', sa['description'])}")
            
            print(f"\n🗺️  Geographic Configuration:")
            print(f"   Bounds Type: {sa['bounds_type']}")
            
            print(f"\n🌱 Bioregional Context:")
            print(f"   Bioregion Type: {sa['bioregion_type']}")
            print(f"   Primary Ecosystem: {sa['primary_ecosystem']}")
            
            if 'management_focus' in sa and sa['management_focus']:
                print(f"   Management Focus: {', '.join(sa['management_focus'])}")
            
            if 'primary_crops' in sa and sa['primary_crops']:
                print(f"   Primary Crops: {', '.join(sa['primary_crops'])}")
            
            if 'farming_type' in sa and sa['farming_type']:
                print(f"   Farming Type: {sa['farming_type']}")
            
            if 'forest_type' in sa and sa['forest_type']:
                print(f"   Forest Type: {sa['forest_type']}")
            
            print(f"\n⚙️ Processing Configuration:")
            print(f"   Resolution Rules: {sa['resolution_rules']}")
            print(f"   Default Resolution: {sa['default_resolution']}")
            print(f"   Max Memory: {sa['max_memory_gb']}GB")
            
            if 'chunking_enabled' in sa:
                print(f"   Chunking: {'Enabled' if sa['chunking_enabled'] else 'Disabled'}")
                if sa['chunking_enabled']:
                    print(f"   Max Hexagons/Chunk: {sa['max_hexagons_per_chunk']:,}")
            
            if 'min_coverage_threshold' in sa:
                print(f"   Min Coverage: {sa['min_coverage_threshold']:.1%}")
                print(f"   Min Density: {sa['min_density_threshold']}")
    
    def print_bioregion_summary(self, study_areas: List[Dict]):
        """Print summary by bioregion type."""
        bioregion_counts = {}
        ecosystem_counts = {}
        
        for sa in study_areas:
            bioregion = sa['bioregion_type']
            ecosystem = sa['primary_ecosystem']
            
            bioregion_counts[bioregion] = bioregion_counts.get(bioregion, 0) + 1
            ecosystem_counts[ecosystem] = ecosystem_counts.get(ecosystem, 0) + 1
        
        print("\n🌍 Bioregion Summary:")
        print("=" * 30)
        for bioregion, count in sorted(bioregion_counts.items()):
            print(f"  {bioregion}: {count}")
        
        print("\n🌿 Ecosystem Summary:")
        print("=" * 30)
        for ecosystem, count in sorted(ecosystem_counts.items()):
            print(f"  {ecosystem}: {count}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="List study area configurations")
    parser.add_argument('--config-dir', type=Path, help="Study area config directory")
    parser.add_argument('--bioregion', choices=['agriculture', 'forestry', 'watershed', 'conservation', 'mixed_use'],
                      help="Filter by bioregion type")
    parser.add_argument('--detailed', action='store_true', help="Show detailed information")
    parser.add_argument('--validate', action='store_true', help="Validate configurations")
    parser.add_argument('--summary', action='store_true', help="Show bioregion summary")
    parser.add_argument('--format', choices=['table', 'detailed', 'json'], default='table',
                      help="Output format")
    
    args = parser.parse_args()
    
    lister = StudyAreaLister(config_dir=args.config_dir)
    
    try:
        study_areas = lister.list_study_areas(
            bioregion_filter=args.bioregion,
            detailed=args.detailed or args.format == 'detailed',
            validate=args.validate
        )
        
        if args.format == 'json':
            print(json.dumps(study_areas, indent=2))
        elif args.format == 'detailed':
            lister.print_detailed_info(study_areas)
        else:
            lister.print_summary_table(study_areas, detailed=args.validate)
        
        if args.summary:
            lister.print_bioregion_summary(study_areas)
        
        print(f"\nFound {len(study_areas)} study area configurations")
        
    except Exception as e:
        logger.error(f"Error listing study areas: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()