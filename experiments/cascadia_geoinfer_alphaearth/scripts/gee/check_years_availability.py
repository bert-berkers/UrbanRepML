"""
Check AlphaEarth data availability across years 2017-2024.

This script verifies which years have AlphaEarth data available in Google Earth Engine,
checks data quality, coverage, and generates an availability report.

Usage:
    python check_years_availability.py
    python check_years_availability.py --save_report
"""

import ee
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
import logging
import os
import sys

# Initialize Earth Engine
try:
    ee.Initialize(project='boreal-union-296021')
    print("Google Earth Engine initialized successfully with project boreal-union-296021")
except Exception as e:
    print(f"Failed to initialize Earth Engine: {e}")
    print("Please authenticate using: earthengine authenticate --project=boreal-union-296021")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class AlphaEarthAvailabilityChecker:
    """Check AlphaEarth data availability across years."""
    
    # Known AlphaEarth collection patterns
    COLLECTION_PATTERNS = [
        "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",  # Official AlphaEarth collection
        "projects/sat-io/open-datasets/AlphaEarth/{year}",
        "projects/ee-sat-io/open-datasets/AlphaEarth/{year}",
        "projects/alphaearth/embeddings/{year}",
    ]
    
    # Expected specifications
    EXPECTED_BANDS = 64  # AlphaEarth has 64-dimensional embeddings
    EXPECTED_RESOLUTION = 10  # 10 meters
    
    def __init__(self):
        """Initialize availability checker."""
        self.availability_report = {
            'check_date': datetime.now().isoformat(),
            'years_checked': [],
            'years_available': [],
            'years_missing': [],
            'details': {}
        }
        logger.info("Initialized AlphaEarth Availability Checker")
    
    def check_collection_exists(self, collection_id: str) -> Dict[str, Any]:
        """
        Check if a collection exists and get its properties.
        
        Args:
            collection_id: Earth Engine collection ID
            
        Returns:
            Dict with collection information or None if not exists
        """
        try:
            collection = ee.ImageCollection(collection_id)
            size = collection.size().getInfo()
            
            if size > 0:
                # Get first image for detailed info
                first_image = ee.Image(collection.first())
                
                # Get properties
                bands = first_image.bandNames().getInfo()
                projection = first_image.projection().getInfo()
                
                # Get spatial extent
                geometry = first_image.geometry()
                bounds = geometry.bounds().coordinates().get(0).getInfo()
                
                # Get temporal info
                dates = collection.aggregate_array('system:time_start').getInfo()
                if dates:
                    min_date = datetime.fromtimestamp(min(dates)/1000).strftime('%Y-%m-%d')
                    max_date = datetime.fromtimestamp(max(dates)/1000).strftime('%Y-%m-%d')
                else:
                    min_date = max_date = "Unknown"
                
                return {
                    'exists': True,
                    'collection_id': collection_id,
                    'image_count': size,
                    'band_count': len(bands),
                    'band_names': bands[:5] if len(bands) > 5 else bands,  # Sample bands
                    'resolution': projection.get('nominalScale', 'Unknown'),
                    'crs': projection.get('crs', 'Unknown'),
                    'temporal_range': f"{min_date} to {max_date}",
                    'spatial_bounds': bounds
                }
            else:
                return {
                    'exists': True,
                    'collection_id': collection_id,
                    'image_count': 0,
                    'message': 'Collection exists but is empty'
                }
                
        except Exception as e:
            return {
                'exists': False,
                'collection_id': collection_id,
                'error': str(e)
            }
    
    def check_year_availability(self, year: int) -> Dict[str, Any]:
        """
        Check AlphaEarth availability for a specific year.
        
        Args:
            year: Year to check
            
        Returns:
            Dict with availability information
        """
        logger.info(f"\nChecking year {year}...")
        
        year_info = {
            'year': year,
            'available': False,
            'collections_checked': [],
            'valid_collection': None,
            'data_quality': {}
        }
        
        # Try different collection patterns
        for pattern in self.COLLECTION_PATTERNS:
            if "{year}" in pattern:
                collection_id = pattern.format(year=year)
            else:
                collection_id = pattern
            
            logger.info(f"  Trying collection: {collection_id}")
            
            result = self.check_collection_exists(collection_id)
            year_info['collections_checked'].append(collection_id)
            
            if result['exists'] and result.get('image_count', 0) > 0:
                # For the official collection, check if it has data for this specific year
                if collection_id == "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL":
                    year_coverage = self.check_year_in_collection(collection_id, year)
                    if not year_coverage['has_year']:
                        logger.info(f"  ✗ Collection found but no data for year {year}")
                        continue
                    result.update(year_coverage)
                
                year_info['available'] = True
                year_info['valid_collection'] = collection_id
                year_info['collection_info'] = result
                
                # Check data quality
                year_info['data_quality'] = self.assess_data_quality(result)
                
                logger.info(f"  ✓ Found valid data in {collection_id}")
                logger.info(f"    Images: {result['image_count']}")
                logger.info(f"    Bands: {result['band_count']}")
                logger.info(f"    Resolution: {result['resolution']}m")
                
                break
            else:
                if result['exists']:
                    logger.info(f"  ✗ Collection exists but is empty")
                else:
                    logger.info(f"  ✗ Collection not found")
        
        if not year_info['available']:
            logger.warning(f"  No AlphaEarth data found for year {year}")
        
        return year_info
    
    def check_year_in_collection(self, collection_id: str, year: int) -> Dict[str, Any]:
        """
        Check if a specific year has data in the collection.
        
        Args:
            collection_id: Earth Engine collection ID
            year: Year to check
            
        Returns:
            Dict with year availability information
        """
        try:
            collection = ee.ImageCollection(collection_id)
            
            # Filter by year
            start_date = f"{year}-01-01"
            end_date = f"{year+1}-01-01"
            
            yearly_collection = collection.filterDate(start_date, end_date)
            yearly_size = yearly_collection.size().getInfo()
            
            if yearly_size > 0:
                # Get sample image from this year
                sample_image = ee.Image(yearly_collection.first())
                
                return {
                    'has_year': True,
                    'year_image_count': yearly_size,
                    'year_date_range': f"{start_date} to {end_date}",
                    'sample_image_date': ee.Date(sample_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                }
            else:
                return {
                    'has_year': False,
                    'year_image_count': 0,
                    'message': f'No images found for year {year}'
                }
                
        except Exception as e:
            return {
                'has_year': False,
                'error': f'Error checking year {year}: {str(e)}'
            }
    
    def assess_data_quality(self, collection_info: Dict) -> Dict[str, Any]:
        """
        Assess the quality of available data.
        
        Args:
            collection_info: Collection information dict
            
        Returns:
            Dict with quality assessment
        """
        quality = {
            'meets_specifications': True,
            'issues': [],
            'warnings': []
        }
        
        # Check band count
        if collection_info['band_count'] != self.EXPECTED_BANDS:
            quality['warnings'].append(
                f"Expected {self.EXPECTED_BANDS} bands, found {collection_info['band_count']}"
            )
        
        # Check resolution
        if collection_info['resolution'] != self.EXPECTED_RESOLUTION:
            quality['warnings'].append(
                f"Expected {self.EXPECTED_RESOLUTION}m resolution, found {collection_info['resolution']}m"
            )
        
        # Check image count (warning if very few)
        if collection_info['image_count'] < 10:
            quality['warnings'].append(
                f"Low image count ({collection_info['image_count']}), may have incomplete coverage"
            )
        
        if quality['warnings']:
            quality['meets_specifications'] = False
        
        return quality
    
    def check_cascadia_coverage(self, year: int, collection_id: str) -> Dict[str, Any]:
        """
        Check if the collection covers the Cascadia region.
        
        Args:
            year: Year to check
            collection_id: Valid collection ID
            
        Returns:
            Dict with coverage information
        """
        logger.info(f"  Checking Cascadia coverage for {year}...")
        
        try:
            # Define Cascadia bounds (approximate)
            cascadia_bounds = ee.Geometry.Rectangle([-124.6, 39.0, -116.5, 46.3])
            
            # Load collection
            collection = ee.ImageCollection(collection_id)
            
            # Get mosaic footprint
            mosaic = collection.mosaic()
            footprint = mosaic.geometry()
            
            # Check intersection
            intersection = footprint.intersection(cascadia_bounds, 1000)
            intersection_area = intersection.area(1000)
            cascadia_area = cascadia_bounds.area(1000)
            
            coverage_percent = (intersection_area.divide(cascadia_area).multiply(100)).getInfo()
            
            coverage_info = {
                'covers_cascadia': coverage_percent > 90,
                'coverage_percent': round(coverage_percent, 2),
                'fully_covered': coverage_percent >= 99
            }
            
            logger.info(f"    Cascadia coverage: {coverage_percent:.1f}%")
            
            return coverage_info
            
        except Exception as e:
            logger.error(f"    Failed to check coverage: {e}")
            return {
                'covers_cascadia': None,
                'error': str(e)
            }
    
    def run_full_check(self, years: List[int] = None) -> Dict[str, Any]:
        """
        Run full availability check for all years.
        
        Args:
            years: List of years to check (default: 2017-2024)
            
        Returns:
            Complete availability report
        """
        if years is None:
            years = list(range(2017, 2025))
        
        logger.info("="*60)
        logger.info("AlphaEarth Availability Check")
        logger.info("="*60)
        
        self.availability_report['years_checked'] = years
        
        for year in years:
            year_info = self.check_year_availability(year)
            
            # If available, check Cascadia coverage
            if year_info['available'] and year_info['valid_collection']:
                coverage = self.check_cascadia_coverage(year, year_info['valid_collection'])
                year_info['cascadia_coverage'] = coverage
            
            # Update report
            self.availability_report['details'][str(year)] = year_info
            
            if year_info['available']:
                self.availability_report['years_available'].append(year)
            else:
                self.availability_report['years_missing'].append(year)
        
        # Generate summary
        self.generate_summary()
        
        return self.availability_report
    
    def generate_summary(self):
        """Generate summary statistics for the report."""
        total_years = len(self.availability_report['years_checked'])
        available_years = len(self.availability_report['years_available'])
        
        self.availability_report['summary'] = {
            'total_years_checked': total_years,
            'years_with_data': available_years,
            'years_missing': total_years - available_years,
            'availability_rate': f"{(available_years/total_years)*100:.1f}%",
            'continuous_coverage': self.check_continuous_coverage(),
            'recommendations': self.generate_recommendations()
        }
        
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Years checked: {total_years}")
        logger.info(f"Years available: {available_years}")
        logger.info(f"Years missing: {total_years - available_years}")
        logger.info(f"Availability rate: {(available_years/total_years)*100:.1f}%")
    
    def check_continuous_coverage(self) -> Dict[str, Any]:
        """Check for continuous year coverage."""
        available = sorted(self.availability_report['years_available'])
        
        if not available:
            return {'has_continuous': False, 'gaps': []}
        
        gaps = []
        for i in range(len(available) - 1):
            if available[i+1] - available[i] > 1:
                gap_years = list(range(available[i]+1, available[i+1]))
                gaps.extend(gap_years)
        
        return {
            'has_continuous': len(gaps) == 0,
            'gaps': gaps,
            'longest_continuous': self.find_longest_continuous(available)
        }
    
    def find_longest_continuous(self, years: List[int]) -> Dict[str, Any]:
        """Find longest continuous period of available years."""
        if not years:
            return {'start': None, 'end': None, 'length': 0}
        
        years = sorted(years)
        longest = []
        current = [years[0]]
        
        for i in range(1, len(years)):
            if years[i] - years[i-1] == 1:
                current.append(years[i])
            else:
                if len(current) > len(longest):
                    longest = current
                current = [years[i]]
        
        if len(current) > len(longest):
            longest = current
        
        return {
            'start': longest[0] if longest else None,
            'end': longest[-1] if longest else None,
            'length': len(longest)
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on availability."""
        recommendations = []
        
        available = self.availability_report['years_available']
        missing = self.availability_report['years_missing']
        
        if len(available) >= 5:
            recommendations.append("Sufficient temporal coverage for trend analysis")
        else:
            recommendations.append("Limited temporal coverage; consider focusing on available years")
        
        if missing:
            recommendations.append(f"Missing years {missing} - consider interpolation methods")
        
        # Check for recent data
        if 2024 in available or 2023 in available:
            recommendations.append("Recent data available for current conditions analysis")
        
        return recommendations
    
    def save_report(self, filename: str = None):
        """
        Save availability report to JSON file.
        
        Args:
            filename: Output filename (default: availability_report.json)
        """
        if filename is None:
            filename = "../../analysis/availability_report.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.availability_report, f, indent=2)
        
        logger.info(f"\nReport saved to: {filename}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Check AlphaEarth data availability")
    parser.add_argument('--years', nargs='+', type=int, 
                       help='Specific years to check')
    parser.add_argument('--save_report', action='store_true',
                       help='Save report to JSON file')
    parser.add_argument('--output', type=str,
                       help='Output filename for report')
    
    args = parser.parse_args()
    
    # Create checker
    checker = AlphaEarthAvailabilityChecker()
    
    # Run check
    years = args.years if args.years else list(range(2017, 2025))
    report = checker.run_full_check(years)
    
    # Display available years
    if report['years_available']:
        logger.info(f"\n✓ Available years: {sorted(report['years_available'])}")
    if report['years_missing']:
        logger.info(f"✗ Missing years: {sorted(report['years_missing'])}")
    
    # Save report if requested
    if args.save_report:
        checker.save_report(args.output)


if __name__ == "__main__":
    main()