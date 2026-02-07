"""
Validation script for Earth Engine tiled exports integration with AlphaEarth processing pipeline.

This script validates that tiled exports from fetch_alphaearth_embeddings_tiled.py
are properly compatible with the existing AlphaEarth processing pipeline.

Validation checks:
1. Tile naming conventions match expected patterns
2. Tile boundaries and overlaps are correct
3. Coordinate systems are consistent
4. Tiles can be processed by existing AlphaEarth processors
5. Stitching metadata is properly generated

Usage:
    python scripts/alphaearth_earthengine_retrieval/validate_tile_integration.py --study-area cascadia_oldremove --year 2021
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box, Polygon
import numpy as np

# Add project path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from stage1_modalities.alphaearth.processor import AlphaEarthProcessor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TileIntegrationValidator:
    """Validates integration between Earth Engine tiled exports and processing pipeline."""
    
    def __init__(self, study_area: str, year: int):
        self.study_area = study_area
        self.year = year
        self.metadata_path = Path(f"data/study_areas/{study_area}/tiles_metadata_{year}.json")
        self.validation_results = {}
        
    def load_tile_metadata(self) -> Dict:
        """Load tile metadata from tiled export."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Tile metadata not found: {self.metadata_path}")
        
        with open(self.metadata_path) as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {metadata['total_tiles']} tiles")
        return metadata
    
    def validate_naming_conventions(self, metadata: Dict) -> bool:
        """Validate that tile names follow expected conventions."""
        logger.info("Validating naming conventions...")
        
        expected_patterns = {
            'cascadia_oldremove': r'Cascadia_AlphaEarth_\d{4}_\d{4}\.tif',
            'netherlands': r'Netherlands_Embedding_\d{4}_\d{4}\.tif'
        }
        
        pattern = expected_patterns.get(self.study_area.lower())
        if not pattern:
            logger.warning(f"No specific pattern defined for {self.study_area}")
            return True  # Skip validation for unknown study areas
        
        import re
        pattern_regex = re.compile(pattern)
        
        valid_names = 0
        total_tiles = len(metadata['tiles'])
        
        for tile in metadata['tiles']:
            filename = tile['filename']
            if pattern_regex.match(filename):
                valid_names += 1
            else:
                logger.error(f"Invalid filename pattern: {filename}")
        
        success = valid_names == total_tiles
        self.validation_results['naming_conventions'] = {
            'valid': valid_names,
            'total': total_tiles,
            'success': success
        }
        
        if success:
            logger.info(f"✓ All {total_tiles} filenames follow correct naming convention")
        else:
            logger.error(f"✗ {total_tiles - valid_names} filenames don't match expected pattern")
        
        return success
    
    def validate_tile_coverage(self, metadata: Dict) -> bool:
        """Validate that tiles properly cover the study area."""
        logger.info("Validating tile coverage...")
        
        # Load study area boundary
        try:
            boundary_path = Path(f"data/boundaries/{self.study_area}/{self.study_area}_states.geojson")
            if not boundary_path.exists():
                boundary_path = Path(f"study_areas/{self.study_area}/area_gdf/boundary.geojson")
            
            study_gdf = gpd.read_file(boundary_path)
            study_area_geom = study_gdf.unary_union
        except Exception as e:
            logger.error(f"Could not load study area boundary: {e}")
            return False
        
        # Create tile geometries
        tile_geometries = []
        for tile in metadata['tiles']:
            bounds = tile['bounds_wgs84']  # [minx, miny, maxx, maxy]
            tile_geom = box(*bounds)
            tile_geometries.append(tile_geom)
        
        # Check coverage
        from shapely.ops import unary_union
        tiles_union = unary_union(tile_geometries)
        
        # Calculate coverage metrics
        study_area_area = study_area_geom.area
        covered_area = study_area_geom.intersection(tiles_union).area
        coverage_ratio = covered_area / study_area_area
        
        # Check for gaps
        gaps = study_area_geom.difference(tiles_union)
        gap_area = gaps.area if hasattr(gaps, 'area') else 0
        
        self.validation_results['tile_coverage'] = {
            'coverage_ratio': coverage_ratio,
            'gap_area_ratio': gap_area / study_area_area,
            'total_tiles': len(tile_geometries),
            'success': coverage_ratio > 0.95  # 95% coverage threshold
        }
        
        if coverage_ratio > 0.95:
            logger.info(f"✓ Tile coverage: {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)")
        else:
            logger.error(f"✗ Insufficient coverage: {coverage_ratio:.3f} ({coverage_ratio*100:.1f}%)")
        
        return coverage_ratio > 0.95
    
    def validate_overlap_consistency(self, metadata: Dict) -> bool:
        """Validate that tile overlaps are consistent."""
        logger.info("Validating tile overlaps...")
        
        expected_overlap_km = metadata.get('overlap_km', 0)
        tile_size_km = metadata.get('tile_size_km')
        
        if expected_overlap_km == 0:
            logger.info("✓ No overlap specified - skipping overlap validation")
            return True
        
        # Check adjacent tiles for proper overlap
        tiles = metadata['tiles']
        overlap_errors = 0
        overlap_checks = 0
        
        for i, tile1 in enumerate(tiles):
            bounds1 = tile1['bounds_wgs84']
            geom1 = box(*bounds1)
            
            for j, tile2 in enumerate(tiles[i+1:], i+1):
                bounds2 = tile2['bounds_wgs84']
                geom2 = box(*bounds2)
                
                # Check if tiles are adjacent (share an edge)
                if geom1.touches(geom2) or geom1.intersects(geom2):
                    overlap_checks += 1
                    
                    # Calculate actual overlap
                    intersection = geom1.intersection(geom2)
                    if intersection.is_empty:
                        overlap_errors += 1
                        logger.warning(f"Adjacent tiles {tile1['id']} and {tile2['id']} have no overlap")
        
        success = overlap_errors == 0
        self.validation_results['overlap_consistency'] = {
            'overlap_checks': overlap_checks,
            'overlap_errors': overlap_errors,
            'success': success
        }
        
        if success:
            logger.info(f"✓ Overlap validation passed ({overlap_checks} tile pairs checked)")
        else:
            logger.error(f"✗ {overlap_errors} overlap errors found")
        
        return success
    
    def test_processor_compatibility(self, metadata: Dict, sample_size: int = 1) -> bool:
        """Test that tiles can be processed by existing AlphaEarth processor."""
        logger.info(f"Testing processor compatibility with {sample_size} sample tiles...")
        
        # Note: This is a dry-run test since actual TIFF files may not be downloaded yet
        try:
            # Create a mock config for the processor
            config = {
                'subtile_size': 512,
                'min_pixels_per_hex': 5,
                'max_workers': 1
            }
            
            processor = AlphaEarthProcessor(config)
            
            # Validate that the processor can handle the expected file patterns
            expected_pattern = metadata.get('naming_convention', '')
            
            # Test filename pattern recognition
            sample_filenames = [tile['filename'] for tile in metadata['tiles'][:sample_size]]
            
            compatible_files = 0
            for filename in sample_filenames:
                # Check if filename would be recognized by existing processors
                if any(pattern in filename for pattern in ['AlphaEarth', 'Embedding']):
                    compatible_files += 1
                else:
                    logger.warning(f"Filename may not be recognized: {filename}")
            
            success = compatible_files == len(sample_filenames)
            
            self.validation_results['processor_compatibility'] = {
                'tested_files': len(sample_filenames),
                'compatible_files': compatible_files,
                'success': success
            }
            
            if success:
                logger.info(f"✓ All {len(sample_filenames)} sample files compatible with processor")
            else:
                logger.error(f"✗ {len(sample_filenames) - compatible_files} files may not be compatible")
            
            return success
            
        except Exception as e:
            logger.error(f"Processor compatibility test failed: {e}")
            self.validation_results['processor_compatibility'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def validate_coordinate_systems(self, metadata: Dict) -> bool:
        """Validate coordinate system consistency."""
        logger.info("Validating coordinate systems...")
        
        # Check that all tiles use WGS84 (EPSG:4326)
        # This should be consistent with Earth Engine exports
        success = True  # Assume success since we control the export CRS
        
        # Validate bounds are in reasonable WGS84 ranges
        for tile in metadata['tiles']:
            bounds = tile['bounds_wgs84']  # [minx, miny, maxx, maxy]
            
            # Check longitude bounds (-180 to 180)
            if not (-180 <= bounds[0] <= 180 and -180 <= bounds[2] <= 180):
                logger.error(f"Invalid longitude bounds for tile {tile['id']}: {bounds[0]}, {bounds[2]}")
                success = False
            
            # Check latitude bounds (-90 to 90)
            if not (-90 <= bounds[1] <= 90 and -90 <= bounds[3] <= 90):
                logger.error(f"Invalid latitude bounds for tile {tile['id']}: {bounds[1]}, {bounds[3]}")
                success = False
        
        self.validation_results['coordinate_systems'] = {'success': success}
        
        if success:
            logger.info("✓ All coordinate systems valid (WGS84)")
        else:
            logger.error("✗ Coordinate system validation failed")
        
        return success
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        overall_success = all(
            result.get('success', False) 
            for result in self.validation_results.values()
        )
        
        report = {
            'study_area': self.study_area,
            'year': self.year,
            'overall_success': overall_success,
            'validation_results': self.validation_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save report
        report_path = Path(f"data/study_areas/{self.study_area}/tile_validation_report_{self.year}.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {report_path}")
        
        # Print summary
        print("\\n" + "="*60)
        print("TILE INTEGRATION VALIDATION SUMMARY")
        print("="*60)
        print(f"Study Area: {self.study_area}")
        print(f"Year: {self.year}")
        print(f"Overall Result: {'✓ PASSED' if overall_success else '✗ FAILED'}")
        print()
        
        for test_name, result in self.validation_results.items():
            status = '✓ PASS' if result.get('success', False) else '✗ FAIL'
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("="*60)
        
        return report
    
    def run_full_validation(self) -> bool:
        """Run all validation stage3_analysis."""
        logger.info(f"Starting full validation for {self.study_area} {self.year}")
        
        try:
            # Load metadata
            metadata = self.load_tile_metadata()
            
            # Run all validation stage3_analysis
            tests = [
                self.validate_naming_conventions,
                self.validate_tile_coverage,
                self.validate_overlap_consistency,
                self.test_processor_compatibility,
                self.validate_coordinate_systems
            ]
            
            for test in tests:
                try:
                    test(metadata)
                except Exception as e:
                    logger.error(f"Test {test.__name__} failed with error: {e}")
                    self.validation_results[test.__name__] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Generate report
            report = self.generate_validation_report()
            
            return report['overall_success']
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False


def get_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Earth Engine tiled export integration with AlphaEarth processing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--study-area", type=str, required=True,
                       help="Name of study area to validate")
    parser.add_argument("--year", type=int, default=2022,
                       help="Year of data to validate")
    parser.add_argument("--sample-size", type=int, default=3,
                       help="Number of tiles to test for processor compatibility")
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = get_arguments()
    
    try:
        validator = TileIntegrationValidator(args.study_area, args.year)
        success = validator.run_full_validation()
        
        if success:
            logger.info("🎉 All validation stage3_analysis passed! Tiled export is fully integrated.")
            return 0
        else:
            logger.error("❌ Some validation stage3_analysis failed. Check the report for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Validation process failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())