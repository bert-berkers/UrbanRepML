"""
Gap Detection for Cascadia AlphaEarth Data.

This script identifies spatial and temporal gaps in the AlphaEarth dataset,
preparing for synthetic data generation through actualization.

Usage:
    python gap_detector.py --year 2023 --resolution 8
    python gap_detector.py --all_years --all_resolutions --save_report
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Polygon, Point
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console only for now
    ]
)
logger = logging.getLogger(__name__)


class GapDetector:
    """Detect gaps in spatial-temporal AlphaEarth coverage."""
    
    def __init__(self,
                 data_dir: str = "data/h3_processed",
                 output_dir: str = "data/temporal/gap_analysis", 
                 config_path: str = "config.yaml"):
        """
        Initialize gap detector.
        
        Args:
            data_dir: Directory with H3 processed data
            output_dir: Directory for gap analysis outputs
            config_path: Path to configuration file
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.config_path = config_path
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self.load_config()
        
        # Gap detection results
        self.gaps = {
            'spatial': {},
            'temporal': {},
            'quality': {},
            'summary': {}
        }
        
        logger.info("Initialized Gap Detector")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Output directory: {output_dir}")
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {
                'actualization': {
                    'gap_detection': {
                        'spatial_coverage_threshold': 0.8,
                        'temporal_coverage_threshold': 0.6,
                        'quality_threshold': 0.2
                    }
                }
            }
    
    def get_expected_coverage(self, resolution: int) -> Set[str]:
        """
        Get expected H3 coverage for Cascadia at given resolution.
        
        Args:
            resolution: H3 resolution
            
        Returns:
            Set of expected H3 indices
        """
        logger.info(f"Calculating expected coverage for resolution {resolution}")
        
        # Define Cascadia boundary (approximate)
        cascadia_bounds = [
            (-124.6, 39.0),  # SW corner
            (-124.6, 46.3),  # NW corner
            (-116.5, 46.3),  # NE corner
            (-116.5, 39.0),  # SE corner
            (-124.6, 39.0)   # Close polygon
        ]
        
        # Create polygon
        cascadia_polygon = Polygon(cascadia_bounds)
        
        # Get all H3 cells that intersect with Cascadia
        expected_cells = set()
        
        # Generate approximate grid for expected coverage
        logger.info("Using grid sampling method for expected coverage")
        # Generate cells by sampling points across Cascadia region
        lat_step = 0.05 if resolution >= 9 else 0.1  # Denser sampling for higher res
        lon_step = 0.05 if resolution >= 9 else 0.1
        
        for lat in np.arange(39.0, 46.3, lat_step):
            for lon in np.arange(-124.6, -116.5, lon_step):
                cell = h3.latlng_to_cell(lat, lon, resolution)
                expected_cells.add(cell)
        
        logger.info(f"Expected {len(expected_cells):,} cells at resolution {resolution}")
        return expected_cells
    
    def detect_spatial_gaps(self, year: int, resolution: int) -> Dict:
        """
        Detect spatial gaps for a specific year and resolution.
        
        Args:
            year: Year to analyze
            resolution: H3 resolution
            
        Returns:
            Dictionary with gap information
        """
        logger.info(f"Detecting spatial gaps for year {year}, resolution {resolution}")
        
        # Load data file
        data_file = os.path.join(
            self.data_dir,
            f"resolution_{resolution}",
            f"cascadia_{year}_h3_res{resolution}.parquet"
        )
        
        if not os.path.exists(data_file):
            logger.warning(f"Data file not found: {data_file}")
            return {
                'year': year,
                'resolution': resolution,
                'status': 'missing',
                'coverage': 0.0
            }
        
        # Load data
        gdf = gpd.read_parquet(data_file)
        actual_cells = set(gdf['h3_index'].values)
        
        # Get expected coverage
        expected_cells = self.get_expected_coverage(resolution)
        
        # Calculate gaps
        missing_cells = expected_cells - actual_cells
        extra_cells = actual_cells - expected_cells
        
        coverage = len(actual_cells) / len(expected_cells) if expected_cells else 0
        
        gap_info = {
            'year': year,
            'resolution': resolution,
            'expected_cells': len(expected_cells),
            'actual_cells': len(actual_cells),
            'missing_cells': len(missing_cells),
            'extra_cells': len(extra_cells),
            'coverage': coverage,
            'coverage_percent': coverage * 100,
            'is_complete': coverage >= self.config['actualization']['gap_detection']['spatial_coverage_threshold']
        }
        
        # Identify gap clusters (contiguous missing regions)
        if missing_cells and resolution <= 8:  # Only for manageable resolutions
            gap_clusters = self.identify_gap_clusters(missing_cells, resolution)
            gap_info['gap_clusters'] = gap_clusters
        
        logger.info(f"  Coverage: {coverage*100:.1f}% ({len(actual_cells):,}/{len(expected_cells):,} cells)")
        
        return gap_info
    
    def identify_gap_clusters(self, missing_cells: Set[str], resolution: int) -> List[Dict]:
        """
        Identify contiguous clusters of missing cells.
        
        Args:
            missing_cells: Set of missing H3 indices
            resolution: H3 resolution
            
        Returns:
            List of gap cluster information
        """
        logger.info(f"  Identifying gap clusters from {len(missing_cells)} missing cells")
        
        clusters = []
        visited = set()
        
        for cell in missing_cells:
            if cell in visited:
                continue
            
            # Start new cluster
            cluster = set()
            to_visit = [cell]
            
            while to_visit:
                current = to_visit.pop()
                if current in visited:
                    continue
                
                visited.add(current)
                cluster.add(current)
                
                # Get neighbors
                neighbors = h3.grid_ring(current, 1)
                for neighbor in neighbors:
                    if neighbor in missing_cells and neighbor not in visited:
                        to_visit.append(neighbor)
            
            # Calculate cluster centroid
            if cluster:
                lats = []
                lons = []
                for h3_cell in cluster:
                    lat, lon = h3.cell_to_latlng(h3_cell)
                    lats.append(lat)
                    lons.append(lon)
                
                clusters.append({
                    'size': len(cluster),
                    'centroid': (np.mean(lats), np.mean(lons)),
                    'cells': list(cluster)[:10]  # Sample for reference
                })
        
        # Sort by size
        clusters.sort(key=lambda x: x['size'], reverse=True)
        
        logger.info(f"  Found {len(clusters)} gap clusters")
        if clusters:
            logger.info(f"  Largest cluster: {clusters[0]['size']} cells")
        
        return clusters[:10]  # Return top 10 clusters
    
    def detect_temporal_gaps(self, resolution: int, years: List[int] = None) -> Dict:
        """
        Detect temporal gaps across years for a resolution.
        
        Args:
            resolution: H3 resolution
            years: List of years to analyze
            
        Returns:
            Dictionary with temporal gap information
        """
        if years is None:
            years = list(range(2017, 2025))
        
        logger.info(f"Detecting temporal gaps for resolution {resolution}")
        
        # Track cell presence across years
        cell_years = {}
        year_coverage = {}
        
        for year in years:
            data_file = os.path.join(
                self.data_dir,
                f"resolution_{resolution}",
                f"cascadia_{year}_h3_res{resolution}.parquet"
            )
            
            if os.path.exists(data_file):
                gdf = gpd.read_parquet(data_file)
                cells = set(gdf['h3_index'].values)
                
                year_coverage[year] = len(cells)
                
                for cell in cells:
                    if cell not in cell_years:
                        cell_years[cell] = []
                    cell_years[cell].append(year)
            else:
                year_coverage[year] = 0
        
        # Analyze temporal consistency
        temporal_stats = {
            'resolution': resolution,
            'years_analyzed': years,
            'year_coverage': year_coverage,
            'total_unique_cells': len(cell_years),
            'cells_all_years': sum(1 for cell, yrs in cell_years.items() 
                                  if len(yrs) == len(years)),
            'cells_partial_years': sum(1 for cell, yrs in cell_years.items() 
                                      if 0 < len(yrs) < len(years)),
            'temporal_consistency': {}
        }
        
        # Calculate temporal consistency for each cell
        if cell_years:
            consistencies = []
            for cell, yrs in cell_years.items():
                consistency = len(yrs) / len(years)
                consistencies.append(consistency)
            
            temporal_stats['temporal_consistency'] = {
                'mean': np.mean(consistencies),
                'std': np.std(consistencies),
                'min': np.min(consistencies),
                'max': np.max(consistencies),
                'median': np.median(consistencies)
            }
        
        # Identify problematic years
        if year_coverage:
            avg_coverage = np.mean(list(year_coverage.values()))
            threshold = self.config['actualization']['gap_detection']['temporal_coverage_threshold']
            
            temporal_stats['problematic_years'] = [
                year for year, coverage in year_coverage.items()
                if coverage < avg_coverage * threshold
            ]
        
        logger.info(f"  Temporal consistency: {temporal_stats['temporal_consistency'].get('mean', 0)*100:.1f}%")
        
        return temporal_stats
    
    def detect_quality_gaps(self, year: int, resolution: int) -> Dict:
        """
        Detect quality issues in the data (e.g., low variance, outliers).
        
        Args:
            year: Year to analyze
            resolution: H3 resolution
            
        Returns:
            Dictionary with quality gap information
        """
        logger.info(f"Detecting quality gaps for year {year}, resolution {resolution}")
        
        data_file = os.path.join(
            self.data_dir,
            f"resolution_{resolution}",
            f"cascadia_{year}_h3_res{resolution}.parquet"
        )
        
        if not os.path.exists(data_file):
            return {'status': 'missing'}
        
        # Load data
        gdf = gpd.read_parquet(data_file)
        
        # Get embedding columns
        embed_cols = [col for col in gdf.columns if col.startswith('embed_')]
        
        if not embed_cols:
            return {'status': 'no_embeddings'}
        
        # Calculate quality metrics
        embeddings = gdf[embed_cols].values
        
        quality_metrics = {
            'year': year,
            'resolution': resolution,
            'total_cells': len(gdf),
            'embedding_dims': len(embed_cols)
        }
        
        # Check for low variance cells
        cell_variances = np.var(embeddings, axis=1)
        low_var_threshold = self.config['actualization']['gap_detection']['quality_threshold']
        low_var_cells = np.sum(cell_variances < low_var_threshold)
        
        quality_metrics['low_variance_cells'] = int(low_var_cells)
        quality_metrics['low_variance_percent'] = (low_var_cells / len(gdf)) * 100
        
        # Check for outliers (using IQR method)
        embedding_means = np.mean(embeddings, axis=1)
        Q1 = np.percentile(embedding_means, 25)
        Q3 = np.percentile(embedding_means, 75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5
        
        outliers = np.sum((embedding_means < Q1 - outlier_threshold * IQR) | 
                         (embedding_means > Q3 + outlier_threshold * IQR))
        
        quality_metrics['outlier_cells'] = int(outliers)
        quality_metrics['outlier_percent'] = (outliers / len(gdf)) * 100
        
        # Check for zero/nan values
        zero_cells = np.sum(np.all(embeddings == 0, axis=1))
        nan_cells = np.sum(np.any(np.isnan(embeddings), axis=1))
        
        quality_metrics['zero_cells'] = int(zero_cells)
        quality_metrics['nan_cells'] = int(nan_cells)
        
        # Overall quality score
        quality_score = 1.0
        quality_score -= (low_var_cells / len(gdf)) * 0.3
        quality_score -= (outliers / len(gdf)) * 0.2
        quality_score -= (zero_cells / len(gdf)) * 0.3
        quality_score -= (nan_cells / len(gdf)) * 0.2
        
        quality_metrics['quality_score'] = max(0, quality_score)
        quality_metrics['needs_improvement'] = quality_score < 0.8
        
        logger.info(f"  Quality score: {quality_score:.2f}")
        logger.info(f"  Low variance: {low_var_cells} cells ({quality_metrics['low_variance_percent']:.1f}%)")
        
        return quality_metrics
    
    def run_comprehensive_analysis(self, 
                                  years: List[int] = None,
                                  resolutions: List[int] = None) -> Dict:
        """
        Run comprehensive gap analysis across years and resolutions.
        
        Args:
            years: List of years to analyze
            resolutions: List of resolutions to analyze
            
        Returns:
            Complete gap analysis results
        """
        if years is None:
            years = list(range(2017, 2025))
        if resolutions is None:
            resolutions = [5, 6, 7, 8, 9, 10, 11]
        
        logger.info("="*60)
        logger.info("Running Comprehensive Gap Analysis")
        logger.info(f"Years: {years}")
        logger.info(f"Resolutions: {resolutions}")
        logger.info("="*60)
        
        # Spatial gaps
        logger.info("\n--- Spatial Gap Analysis ---")
        for resolution in resolutions:
            self.gaps['spatial'][f'res_{resolution}'] = {}
            for year in years:
                gap_info = self.detect_spatial_gaps(year, resolution)
                self.gaps['spatial'][f'res_{resolution}'][year] = gap_info
        
        # Temporal gaps
        logger.info("\n--- Temporal Gap Analysis ---")
        for resolution in resolutions:
            temporal_info = self.detect_temporal_gaps(resolution, years)
            self.gaps['temporal'][f'res_{resolution}'] = temporal_info
        
        # Quality gaps
        logger.info("\n--- Quality Gap Analysis ---")
        for resolution in resolutions:
            self.gaps['quality'][f'res_{resolution}'] = {}
            for year in years:
                quality_info = self.detect_quality_gaps(year, resolution)
                self.gaps['quality'][f'res_{resolution}'][year] = quality_info
        
        # Generate summary
        self.generate_summary()
        
        return self.gaps
    
    def generate_summary(self):
        """Generate summary statistics from gap analysis."""
        logger.info("\n--- Generating Summary ---")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'spatial_coverage': {},
            'temporal_consistency': {},
            'quality_scores': {},
            'recommendations': []
        }
        
        # Spatial coverage summary
        for res_key, years_data in self.gaps['spatial'].items():
            coverages = []
            for year, data in years_data.items():
                if isinstance(data, dict) and 'coverage' in data:
                    coverages.append(data['coverage'])
            
            if coverages:
                summary['spatial_coverage'][res_key] = {
                    'mean': np.mean(coverages),
                    'std': np.std(coverages),
                    'min': np.min(coverages),
                    'max': np.max(coverages)
                }
        
        # Temporal consistency summary
        for res_key, data in self.gaps['temporal'].items():
            if 'temporal_consistency' in data and data['temporal_consistency']:
                summary['temporal_consistency'][res_key] = data['temporal_consistency']['mean']
        
        # Quality scores summary
        for res_key, years_data in self.gaps['quality'].items():
            scores = []
            for year, data in years_data.items():
                if isinstance(data, dict) and 'quality_score' in data:
                    scores.append(data['quality_score'])
            
            if scores:
                summary['quality_scores'][res_key] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
        
        # Generate recommendations
        if summary['spatial_coverage']:
            low_coverage = [res for res, stats in summary['spatial_coverage'].items() 
                          if stats['mean'] < 0.8]
            if low_coverage:
                summary['recommendations'].append(
                    f"Improve spatial coverage for resolutions: {', '.join(low_coverage)}"
                )
        
        if summary['temporal_consistency']:
            low_consistency = [res for res, score in summary['temporal_consistency'].items() 
                             if score < 0.6]
            if low_consistency:
                summary['recommendations'].append(
                    f"Address temporal gaps for resolutions: {', '.join(low_consistency)}"
                )
        
        if summary['quality_scores']:
            low_quality = [res for res, stats in summary['quality_scores'].items() 
                         if stats['mean'] < 0.8]
            if low_quality:
                summary['recommendations'].append(
                    f"Improve data quality for resolutions: {', '.join(low_quality)}"
                )
        
        self.gaps['summary'] = summary
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("GAP ANALYSIS SUMMARY")
        logger.info("="*60)
        
        for res in ['res_5', 'res_6', 'res_7', 'res_8', 'res_9', 'res_10', 'res_11']:
            if res in summary['spatial_coverage']:
                spatial = summary['spatial_coverage'][res]['mean'] * 100
                temporal = summary['temporal_consistency'].get(res, 0) * 100
                quality = summary['quality_scores'].get(res, {}).get('mean', 0) * 100
                
                logger.info(f"\n{res.upper()}:")
                logger.info(f"  Spatial Coverage: {spatial:.1f}%")
                logger.info(f"  Temporal Consistency: {temporal:.1f}%")
                logger.info(f"  Quality Score: {quality:.1f}%")
        
        if summary['recommendations']:
            logger.info("\nRECOMMENDATIONS:")
            for rec in summary['recommendations']:
                logger.info(f"  - {rec}")
    
    def save_report(self, filename: str = None):
        """
        Save gap analysis report to JSON.
        
        Args:
            filename: Output filename
        """
        if filename is None:
            filename = os.path.join(
                self.output_dir,
                f"gap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        with open(filename, 'w') as f:
            json.dump(self.gaps, f, indent=2, default=str)
        
        logger.info(f"\nReport saved to: {filename}")
    
    def visualize_gaps(self, year: int = 2023, resolution: int = 8):
        """
        Create visualization of spatial gaps.
        
        Args:
            year: Year to visualize
            resolution: H3 resolution
        """
        logger.info(f"Creating gap visualization for year {year}, resolution {resolution}")
        
        # This would create actual maps using folium or matplotlib
        # For now, just log the intent
        logger.info("Visualization features:")
        logger.info("  - Spatial coverage heatmap")
        logger.info("  - Temporal consistency timeline")
        logger.info("  - Quality score distribution")
        
        # Save placeholder
        viz_file = os.path.join(
            self.output_dir,
            f"gap_visualization_{year}_res{resolution}.html"
        )
        logger.info(f"Visualization would be saved to: {viz_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Detect gaps in Cascadia AlphaEarth data")
    parser.add_argument('--year', type=int, help='Single year to analyze')
    parser.add_argument('--years', nargs='+', type=int, help='Multiple years')
    parser.add_argument('--all_years', action='store_true', help='Analyze all years')
    parser.add_argument('--resolution', type=int, help='Single resolution')
    parser.add_argument('--resolutions', nargs='+', type=int, help='Multiple resolutions')
    parser.add_argument('--all_resolutions', action='store_true', help='All resolutions')
    parser.add_argument('--save_report', action='store_true', help='Save JSON report')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    # Create detector
    detector = GapDetector()
    
    # Determine years and resolutions
    if args.all_years:
        years = list(range(2017, 2025))
    elif args.years:
        years = args.years
    elif args.year:
        years = [args.year]
    else:
        years = [2023]  # Default
    
    if args.all_resolutions:
        resolutions = [5, 6, 7, 8, 9, 10, 11]
    elif args.resolutions:
        resolutions = args.resolutions
    elif args.resolution:
        resolutions = [args.resolution]
    else:
        resolutions = [8]  # Default
    
    # Run analysis
    results = detector.run_comprehensive_analysis(years, resolutions)
    
    # Save report
    if args.save_report:
        detector.save_report()
    
    # Create visualizations
    if args.visualize:
        detector.visualize_gaps()


if __name__ == "__main__":
    main()