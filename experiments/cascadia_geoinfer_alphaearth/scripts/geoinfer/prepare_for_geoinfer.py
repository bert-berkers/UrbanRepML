"""
Prepare Cascadia AlphaEarth data for GEO-INFER integration.

This script formats the processed AlphaEarth data to be compatible with the
GEO-INFER agricultural analysis framework, ensuring proper structure and metadata.

Usage:
    python prepare_for_geoinfer.py --year 2023
    python prepare_for_geoinfer.py --all_years --include_synthetic
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console only for now
    ]
)
logger = logging.getLogger(__name__)


class GeoInferPreparator:
    """Prepare data for GEO-INFER integration."""
    
    # GEO-INFER Cascadia Counties
    CALIFORNIA_COUNTIES = {
        "Butte": "06007", "Colusa": "06011", "Del Norte": "06015", "Glenn": "06021",
        "Humboldt": "06023", "Lake": "06033", "Lassen": "06035", "Mendocino": "06045",
        "Modoc": "06049", "Nevada": "06057", "Plumas": "06063", "Shasta": "06089",
        "Sierra": "06091", "Siskiyou": "06093", "Tehama": "06103", "Trinity": "06105"
    }
    
    # Oregon counties (all 36) - placeholder FIPS codes
    OREGON_COUNTIES = {
        f"OR_County_{i:02d}": f"41{i:03d}" for i in range(1, 37)
    }
    
    def __init__(self,
                 data_dir: str = "data/h3_processed",
                 synthetic_dir: str = "data/synthetic/generated",
                 output_dir: str = "analysis/geoinfer_readiness",
                 config_path: str = "config.yaml"):
        """
        Initialize preparator.
        
        Args:
            data_dir: Directory with H3 processed data
            synthetic_dir: Directory with synthetic data
            output_dir: Output directory for GEO-INFER data
            config_path: Configuration file
        """
        self.data_dir = data_dir
        self.synthetic_dir = synthetic_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # County boundaries (would be loaded from actual shapefiles)
        self.county_boundaries = None
        
        # Processing statistics
        self.stats = {
            'years_processed': [],
            'counties_covered': 0,
            'total_hexagons': 0,
            'agricultural_areas': 0,
            'data_quality_score': 0
        }
        
        logger.info("Initialized GEO-INFER Preparator")
        logger.info(f"Output directory: {output_dir}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {
                'geoinfer_alignment': {
                    'output_format': 'parquet',
                    'coordinate_system': 'EPSG:4326',
                    'metadata': {
                        'include_county_id': True,
                        'include_state_id': True,
                        'include_agricultural_zones': True
                    }
                }
            }
    
    def load_county_boundaries(self) -> gpd.GeoDataFrame:
        """
        Load county boundaries for spatial joins.
        In practice, this would load actual shapefiles.
        
        Returns:
            GeoDataFrame with county boundaries
        """
        logger.info("Loading county boundaries...")
        
        # Placeholder - in practice would load from Census TIGER files
        counties_data = []
        
        # California counties
        for county, fips in self.CALIFORNIA_COUNTIES.items():
            counties_data.append({
                'county_name': county,
                'county_fips': fips,
                'state_name': 'California',
                'state_fips': '06',
                'state_abbr': 'CA'
            })
        
        # Oregon counties
        for county, fips in self.OREGON_COUNTIES.items():
            counties_data.append({
                'county_name': county.replace('OR_County_', 'County_'),
                'county_fips': fips,
                'state_name': 'Oregon',
                'state_fips': '41',
                'state_abbr': 'OR'
            })
        
        # Create placeholder geometries (would be actual county polygons)
        from shapely.geometry import box
        
        for i, county in enumerate(counties_data):
            # Create placeholder bounding boxes
            lat_offset = (i % 6) * 0.5
            lon_offset = (i // 6) * 0.5
            
            county['geometry'] = box(
                -124.5 + lon_offset,
                39.5 + lat_offset,
                -124.0 + lon_offset,
                40.0 + lat_offset
            )
        
        county_gdf = gpd.GeoDataFrame(counties_data, crs='EPSG:4326')
        logger.info(f"Loaded {len(county_gdf)} county boundaries")
        
        return county_gdf
    
    def assign_county_ids(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assign county IDs to H3 hexagons through spatial joins.
        
        Args:
            gdf: GeoDataFrame with H3 hexagons
            
        Returns:
            GeoDataFrame with county assignments
        """
        logger.info("Assigning county IDs to hexagons...")
        
        if self.county_boundaries is None:
            self.county_boundaries = self.load_county_boundaries()
        
        # Spatial join to assign counties
        # Note: This is computationally expensive for many hexagons
        logger.info("Performing spatial join (this may take time)...")
        
        # Sample points from hexagon centroids for faster processing
        gdf_points = gdf.copy()
        gdf_points['geometry'] = gdf_points['geometry'].centroid
        
        # Spatial join
        joined = gpd.sjoin(
            gdf_points, 
            self.county_boundaries,
            how='left',
            predicate='within'
        )
        
        # Restore original geometry and clean up
        joined['geometry'] = gdf['geometry']
        
        # Fill missing values
        joined['county_fips'] = joined['county_fips'].fillna('unknown')
        joined['state_fips'] = joined['state_fips'].fillna('unknown')
        joined['county_name'] = joined['county_name'].fillna('unknown')
        joined['state_name'] = joined['state_name'].fillna('unknown')
        joined['state_abbr'] = joined['state_abbr'].fillna('unknown')
        
        # Remove spatial join artifacts
        cols_to_keep = [col for col in joined.columns 
                       if col not in ['index_right']]
        joined = joined[cols_to_keep]
        
        logger.info(f"Assigned counties to {len(joined)} hexagons")
        
        return joined
    
    def identify_agricultural_areas(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Identify and flag agricultural areas using embedding patterns.
        
        Args:
            gdf: GeoDataFrame with embeddings
            
        Returns:
            GeoDataFrame with agricultural flags
        """
        logger.info("Identifying agricultural areas...")
        
        embed_cols = [col for col in gdf.columns if col.startswith('embed_')]
        
        if not embed_cols:
            logger.warning("No embedding columns found")
            gdf['agricultural_likelihood'] = 0.0
            gdf['is_agricultural'] = False
            return gdf
        
        # Simple agricultural classification using embedding patterns
        # In practice, this would use trained models
        
        embeddings = gdf[embed_cols].values
        
        # Calculate agricultural indicators
        # These are placeholder heuristics - real implementation would use ML
        
        # Agricultural areas often have:
        # - Regular patterns (low variance in certain dimensions)
        # - Specific spectral signatures (certain embedding ranges)
        # - Seasonal patterns (if multi-temporal)
        
        # Placeholder agricultural scoring
        embedding_means = np.mean(embeddings, axis=1)
        embedding_stds = np.std(embeddings, axis=1)
        
        # Simple heuristic: agricultural areas have moderate mean, low variance
        agricultural_scores = np.zeros(len(embeddings))
        
        for i in range(len(embeddings)):
            score = 0.0
            
            # Moderate embedding values (0.2-0.8 range after normalization)
            normalized_mean = (embedding_means[i] - embedding_means.min()) / (embedding_means.max() - embedding_means.min())
            if 0.2 <= normalized_mean <= 0.8:
                score += 0.4
            
            # Low variability (consistent patterns)
            if embedding_stds[i] < np.percentile(embedding_stds, 30):
                score += 0.3
            
            # Spatial clustering (agricultural areas cluster together)
            # This would require neighbor analysis - placeholder
            score += 0.3
            
            agricultural_scores[i] = score
        
        gdf['agricultural_likelihood'] = agricultural_scores
        gdf['is_agricultural'] = agricultural_scores > 0.6
        
        agricultural_count = gdf['is_agricultural'].sum()
        logger.info(f"Identified {agricultural_count} agricultural hexagons ({agricultural_count/len(gdf)*100:.1f}%)")
        
        return gdf
    
    def add_geoinfer_metadata(self, gdf: gpd.GeoDataFrame, year: int) -> gpd.GeoDataFrame:
        """
        Add metadata required for GEO-INFER integration.
        
        Args:
            gdf: GeoDataFrame to enhance
            year: Data year
            
        Returns:
            Enhanced GeoDataFrame
        """
        logger.info("Adding GEO-INFER metadata...")
        
        # Primary key for GEO-INFER
        gdf['geoinfer_id'] = gdf['h3_index'].astype(str) + f"_{year}"
        
        # Temporal information
        gdf['data_year'] = year
        gdf['data_timestamp'] = datetime(year, 6, 15).isoformat()  # Mid-year
        
        # Spatial metadata
        gdf['h3_resolution'] = gdf.get('resolution', 8)
        
        # Calculate hexagon centroids
        gdf['centroid_lat'] = gdf['geometry'].centroid.y
        gdf['centroid_lon'] = gdf['geometry'].centroid.x
        
        # Hexagon area (constant for each resolution)
        gdf['area_km2'] = gdf['h3_resolution'].apply(lambda r: {
            5: 252.9, 6: 31.0, 7: 3.65, 8: 0.46, 
            9: 0.054, 10: 0.0063, 11: 0.00074
        }.get(r, 0.46))
        
        # Data source flags
        gdf['is_synthetic'] = gdf.get('synthetic', False)
        gdf['data_source'] = gdf.apply(lambda row: 
            'synthetic' if row.get('synthetic', False) else 'alphaearth', axis=1)
        
        # Quality indicators
        embed_cols = [col for col in gdf.columns if col.startswith('embed_')]
        if embed_cols:
            embeddings = gdf[embed_cols].values
            gdf['embedding_magnitude'] = np.linalg.norm(embeddings, axis=1)
            gdf['embedding_variance'] = np.var(embeddings, axis=1)
        else:
            gdf['embedding_magnitude'] = 0.0
            gdf['embedding_variance'] = 0.0
        
        # Agricultural classification confidence
        if 'agricultural_likelihood' in gdf.columns:
            gdf['agricultural_confidence'] = np.where(
                gdf['agricultural_likelihood'] > 0.8, 'high',
                np.where(gdf['agricultural_likelihood'] > 0.4, 'medium', 'low')
            )
        else:
            gdf['agricultural_confidence'] = 'unknown'
        
        logger.info(f"Added metadata for {len(gdf)} hexagons")
        
        return gdf
    
    def create_geoinfer_schema(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Restructure data to match GEO-INFER schema requirements.
        
        Args:
            gdf: Source GeoDataFrame
            
        Returns:
            GeoDataFrame with GEO-INFER schema
        """
        logger.info("Creating GEO-INFER schema...")
        
        # Define required columns for GEO-INFER
        geoinfer_columns = [
            'geoinfer_id',
            'h3_index', 
            'h3_resolution',
            'data_year',
            'data_timestamp',
            'county_fips',
            'state_fips',
            'county_name',
            'state_name',
            'state_abbr',
            'centroid_lat',
            'centroid_lon',
            'area_km2',
            'is_agricultural',
            'agricultural_likelihood',
            'agricultural_confidence',
            'is_synthetic',
            'data_source',
            'embedding_magnitude',
            'embedding_variance'
        ]
        
        # Add embedding columns
        embed_cols = [col for col in gdf.columns if col.startswith('embed_')]
        geoinfer_columns.extend(embed_cols)
        
        # Add geometry
        geoinfer_columns.append('geometry')
        
        # Select and reorder columns
        available_cols = [col for col in geoinfer_columns if col in gdf.columns]
        missing_cols = [col for col in geoinfer_columns if col not in gdf.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        geoinfer_gdf = gdf[available_cols].copy()
        
        # Ensure proper data types
        if 'data_year' in geoinfer_gdf.columns:
            geoinfer_gdf['data_year'] = geoinfer_gdf['data_year'].astype(int)
        if 'h3_resolution' in geoinfer_gdf.columns:
            geoinfer_gdf['h3_resolution'] = geoinfer_gdf['h3_resolution'].astype(int)
        if 'is_agricultural' in geoinfer_gdf.columns:
            geoinfer_gdf['is_agricultural'] = geoinfer_gdf['is_agricultural'].astype(bool)
        if 'is_synthetic' in geoinfer_gdf.columns:
            geoinfer_gdf['is_synthetic'] = geoinfer_gdf['is_synthetic'].astype(bool)
        
        logger.info(f"Created GEO-INFER schema with {len(geoinfer_gdf)} records")
        logger.info(f"Schema columns: {len(available_cols)}")
        
        return geoinfer_gdf
    
    def process_year(self, year: int, include_synthetic: bool = False) -> Optional[gpd.GeoDataFrame]:
        """
        Process a single year of data for GEO-INFER.
        
        Args:
            year: Year to process
            include_synthetic: Whether to include synthetic data
            
        Returns:
            Processed GeoDataFrame or None if no data
        """
        logger.info(f"\nProcessing year {year} for GEO-INFER")
        logger.info(f"Include synthetic: {include_synthetic}")
        
        # Load H3 resolution 8 data (GEO-INFER standard)
        resolution = 8
        data_file = os.path.join(
            self.data_dir,
            f"resolution_{resolution}",
            f"cascadia_{year}_h3_res{resolution}.parquet"
        )
        
        if not os.path.exists(data_file):
            logger.warning(f"Data file not found: {data_file}")
            return None
        
        # Load main data
        gdf = gpd.read_parquet(data_file)
        logger.info(f"Loaded {len(gdf)} hexagons from {data_file}")
        
        # Load and merge synthetic data if requested
        if include_synthetic:
            synthetic_files = [
                os.path.join(self.synthetic_dir, f"synthetic_{year}_res{resolution}_vae.parquet"),
                os.path.join(self.synthetic_dir, f"synthetic_{year}_res{resolution}_gan.parquet"),
                os.path.join(self.synthetic_dir, f"synthetic_{year}_res{resolution}_interpolation.parquet")
            ]
            
            synthetic_data = []
            for syn_file in synthetic_files:
                if os.path.exists(syn_file):
                    syn_gdf = gpd.read_parquet(syn_file)
                    synthetic_data.append(syn_gdf)
                    logger.info(f"Loaded {len(syn_gdf)} synthetic hexagons from {syn_file}")
            
            if synthetic_data:
                # Combine synthetic data
                synthetic_combined = pd.concat(synthetic_data, ignore_index=True)
                
                # Remove duplicates (same H3 cells from different methods)
                synthetic_combined = synthetic_combined.drop_duplicates(
                    subset=['h3_index'], keep='first'
                )
                
                # Merge with main data
                gdf = pd.concat([gdf, synthetic_combined], ignore_index=True)
                logger.info(f"Total hexagons after synthetic merge: {len(gdf)}")
        
        # Process through pipeline
        logger.info("Processing through GEO-INFER pipeline...")
        
        # 1. Assign county IDs
        gdf = self.assign_county_ids(gdf)
        
        # 2. Identify agricultural areas
        gdf = self.identify_agricultural_areas(gdf)
        
        # 3. Add GEO-INFER metadata
        gdf = self.add_geoinfer_metadata(gdf, year)
        
        # 4. Create GEO-INFER schema
        geoinfer_gdf = self.create_geoinfer_schema(gdf)
        
        # Update statistics
        self.stats['total_hexagons'] += len(geoinfer_gdf)
        self.stats['agricultural_areas'] += geoinfer_gdf['is_agricultural'].sum()
        self.stats['counties_covered'] = len(geoinfer_gdf['county_fips'].unique())
        
        return geoinfer_gdf
    
    def validate_geoinfer_compatibility(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Validate data compatibility with GEO-INFER requirements.
        
        Args:
            gdf: GeoDataFrame to validate
            
        Returns:
            Validation results
        """
        logger.info("Validating GEO-INFER compatibility...")
        
        validation = {
            'is_compatible': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Required columns check
        required_cols = ['geoinfer_id', 'h3_index', 'county_fips', 'state_fips', 'is_agricultural']
        missing_required = [col for col in required_cols if col not in gdf.columns]
        
        if missing_required:
            validation['issues'].append(f"Missing required columns: {missing_required}")
            validation['is_compatible'] = False
        
        # Data quality checks
        if 'h3_index' in gdf.columns:
            duplicate_h3 = gdf['h3_index'].duplicated().sum()
            if duplicate_h3 > 0:
                validation['warnings'].append(f"Found {duplicate_h3} duplicate H3 indices")
        
        # Spatial coverage check
        if 'county_fips' in gdf.columns:
            unknown_counties = (gdf['county_fips'] == 'unknown').sum()
            if unknown_counties > len(gdf) * 0.1:  # > 10%
                validation['warnings'].append(f"High number of unknown counties: {unknown_counties}")
        
        # Agricultural classification check
        if 'is_agricultural' in gdf.columns:
            agricultural_ratio = gdf['is_agricultural'].mean()
            if agricultural_ratio < 0.1 or agricultural_ratio > 0.9:
                validation['warnings'].append(
                    f"Unusual agricultural ratio: {agricultural_ratio:.2f}"
                )
        
        # Embedding quality check
        embed_cols = [col for col in gdf.columns if col.startswith('embed_')]
        if embed_cols:
            embeddings = gdf[embed_cols].values
            zero_embeddings = np.all(embeddings == 0, axis=1).sum()
            if zero_embeddings > 0:
                validation['warnings'].append(f"Found {zero_embeddings} zero embeddings")
        
        # Statistics
        validation['stats'] = {
            'total_hexagons': len(gdf),
            'counties_covered': len(gdf['county_fips'].unique()) if 'county_fips' in gdf.columns else 0,
            'agricultural_hexagons': gdf['is_agricultural'].sum() if 'is_agricultural' in gdf.columns else 0,
            'synthetic_hexagons': gdf['is_synthetic'].sum() if 'is_synthetic' in gdf.columns else 0,
            'embedding_dimensions': len(embed_cols)
        }
        
        # Overall compatibility
        if len(validation['issues']) == 0:
            validation['is_compatible'] = True
        
        # Log results
        if validation['is_compatible']:
            logger.info("✓ Data is GEO-INFER compatible")
        else:
            logger.warning("✗ Data compatibility issues found")
        
        for issue in validation['issues']:
            logger.error(f"  ISSUE: {issue}")
        
        for warning in validation['warnings']:
            logger.warning(f"  WARNING: {warning}")
        
        return validation
    
    def save_geoinfer_data(self, gdf: gpd.GeoDataFrame, year: int, 
                          include_synthetic: bool = False) -> str:
        """
        Save data in GEO-INFER compatible format.
        
        Args:
            gdf: GeoDataFrame to save
            year: Data year
            include_synthetic: Whether synthetic data was included
            
        Returns:
            Output file path
        """
        # Determine filename
        suffix = "_with_synthetic" if include_synthetic else ""
        filename = f"cascadia_geoinfer_{year}{suffix}.parquet"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save as parquet (efficient for large datasets)
        gdf.to_parquet(output_path)
        
        # Also save metadata
        metadata = {
            'year': year,
            'total_hexagons': len(gdf),
            'includes_synthetic': include_synthetic,
            'counties_covered': len(gdf['county_fips'].unique()) if 'county_fips' in gdf.columns else 0,
            'agricultural_hexagons': int(gdf['is_agricultural'].sum()) if 'is_agricultural' in gdf.columns else 0,
            'created_timestamp': datetime.now().isoformat(),
            'coordinate_system': str(gdf.crs),
            'embedding_dimensions': len([col for col in gdf.columns if col.startswith('embed_')])
        }
        
        metadata_path = output_path.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved GEO-INFER data to: {output_path}")
        logger.info(f"Saved metadata to: {metadata_path}")
        
        return output_path
    
    def process_all_years(self, years: List[int] = None, 
                         include_synthetic: bool = False):
        """
        Process all years for GEO-INFER integration.
        
        Args:
            years: List of years to process
            include_synthetic: Include synthetic data
        """
        if years is None:
            years = list(range(2017, 2025))
        
        logger.info("="*60)
        logger.info("Processing all years for GEO-INFER integration")
        logger.info(f"Years: {years}")
        logger.info(f"Include synthetic: {include_synthetic}")
        logger.info("="*60)
        
        processed_files = []
        
        for year in years:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing Year {year}")
            logger.info(f"{'='*40}")
            
            # Process year
            geoinfer_gdf = self.process_year(year, include_synthetic)
            
            if geoinfer_gdf is not None:
                # Validate
                validation = self.validate_geoinfer_compatibility(geoinfer_gdf)
                
                # Save if compatible
                if validation['is_compatible']:
                    output_path = self.save_geoinfer_data(
                        geoinfer_gdf, year, include_synthetic
                    )
                    processed_files.append(output_path)
                    self.stats['years_processed'].append(year)
                else:
                    logger.error(f"Skipping year {year} due to compatibility issues")
            else:
                logger.warning(f"No data available for year {year}")
        
        # Save final statistics
        self.save_processing_stats(processed_files)
        
        logger.info("\n" + "="*60)
        logger.info("GEO-INFER preparation complete!")
        logger.info(f"Processed files: {len(processed_files)}")
        logger.info("="*60)
    
    def save_processing_stats(self, processed_files: List[str]):
        """Save processing statistics."""
        stats_file = os.path.join(
            self.output_dir,
            f"geoinfer_processing_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        self.stats['processed_files'] = processed_files
        self.stats['processing_timestamp'] = datetime.now().isoformat()
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Processing statistics saved to: {stats_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Prepare data for GEO-INFER integration")
    parser.add_argument('--year', type=int, help='Single year to process')
    parser.add_argument('--years', nargs='+', type=int, help='Multiple years')
    parser.add_argument('--all_years', action='store_true', help='Process all years')
    parser.add_argument('--include_synthetic', action='store_true', 
                       help='Include synthetic data')
    parser.add_argument('--validate_only', action='store_true',
                       help='Only run validation on existing files')
    
    args = parser.parse_args()
    
    # Create preparator
    preparator = GeoInferPreparator()
    
    # Determine years
    if args.all_years:
        years = list(range(2017, 2025))
    elif args.years:
        years = args.years
    elif args.year:
        years = [args.year]
    else:
        years = [2023]  # Default
    
    if args.validate_only:
        # Just validate existing files
        for year in years:
            filename = f"cascadia_geoinfer_{year}"
            if args.include_synthetic:
                filename += "_with_synthetic"
            filename += ".parquet"
            
            file_path = os.path.join(preparator.output_dir, filename)
            if os.path.exists(file_path):
                gdf = gpd.read_parquet(file_path)
                validation = preparator.validate_geoinfer_compatibility(gdf)
                logger.info(f"Year {year} validation: {'PASS' if validation['is_compatible'] else 'FAIL'}")
    else:
        # Process years
        preparator.process_all_years(years, args.include_synthetic)


if __name__ == "__main__":
    main()