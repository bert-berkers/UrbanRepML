#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process Del Norte County AlphaEarth embeddings to H3 hexagons.

This script processes downloaded AlphaEarth GeoTIFF files and aggregates
them to H3 hexagons at multiple resolutions for use with UrbanRepML
and GEO-INFER pipelines.

Based on successful Netherlands processing pipeline.
"""

import rasterio
import h3
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
import argparse
import json
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import information theory modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from urban_embedding.information_gain import (
    SpatialInformationCalculator,
    SpatialInformationGain,
    InformationMetrics
)


class DelNorteH3Processor:
    """Process Del Norte AlphaEarth data to H3 hexagons."""
    
    def __init__(self, 
                 input_dir: str = "data/alphaearth/del_norte",
                 output_dir: str = "data/h3_processed/del_norte",
                 h3_resolutions: List[int] = [7, 8, 9, 10],
                 calculate_info_metrics: bool = True):
        """
        Initialize processor.
        
        Args:
            input_dir: Directory containing downloaded GeoTIFF files
            output_dir: Directory for H3 processed outputs
            h3_resolutions: H3 resolutions to generate (default: 7-10)
            calculate_info_metrics: Whether to calculate information theory metrics
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.h3_resolutions = h3_resolutions
        self.calculate_info_metrics = calculate_info_metrics
        
        # Initialize information theory components if enabled
        if self.calculate_info_metrics:
            self.info_calculator = SpatialInformationCalculator(
                method='histogram',
                n_bins=50,
                normalize=True
            )
            self.spatial_info = SpatialInformationGain(
                h3_resolution=8,  # Primary resolution
                temporal_window=1
            )
            print("  Information metrics: ENABLED")
        else:
            self.info_calculator = None
            self.spatial_info = None
            print("  Information metrics: DISABLED")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for res in h3_resolutions:
            (self.output_dir / f"resolution_{res}").mkdir(exist_ok=True)
        
        print(f"Del Norte H3 Processor initialized")
        print(f"  Input dir: {self.input_dir}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  H3 resolutions: {h3_resolutions}")
    
    def load_geotiff(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load AlphaEarth GeoTIFF file.
        
        Args:
            filepath: Path to GeoTIFF file
            
        Returns:
            Tuple of (data array, metadata dict)
        """
        print(f"\nLoading: {filepath.name}")
        
        with rasterio.open(filepath) as src:
            # Read all bands (64 AlphaEarth embedding dimensions)
            data = src.read()  # Shape: (bands, height, width)
            
            # Get metadata
            metadata = {
                'crs': src.crs.to_string(),
                'transform': src.transform,
                'bounds': src.bounds,
                'shape': data.shape,
                'num_bands': src.count,
                'width': src.width,
                'height': src.height,
                'nodata': src.nodata
            }
            
            print(f"  Shape: {data.shape}")
            print(f"  Bounds: {src.bounds}")
            print(f"  CRS: {src.crs}")
            
            return data, metadata
    
    def pixel_to_h3(self, col: int, row: int, transform, resolution: int) -> str:
        """
        Convert pixel coordinates to H3 index.
        
        Args:
            col: Column index
            row: Row index
            transform: Rasterio transform
            resolution: H3 resolution
            
        Returns:
            H3 index string
        """
        # Convert pixel to geographic coordinates
        lon, lat = transform * (col, row)
        
        # Convert to H3
        return h3.latlng_to_cell(lat, lon, resolution)
    
    def aggregate_to_h3(self, 
                       data: np.ndarray, 
                       metadata: Dict, 
                       resolution: int,
                       sample_rate: float = 1.0) -> pd.DataFrame:
        """
        Aggregate AlphaEarth embeddings to H3 hexagons.
        
        Args:
            data: Array of shape (bands, height, width)
            metadata: GeoTIFF metadata
            resolution: H3 resolution
            sample_rate: Fraction of pixels to sample (for speed)
            
        Returns:
            DataFrame with H3 indices and aggregated embeddings
        """
        print(f"\nAggregating to H3 resolution {resolution}...")
        
        bands, height, width = data.shape
        transform = metadata['transform']
        
        # Dictionary to store aggregated values per H3 cell
        h3_aggregates = {}
        
        # Sample pixels for efficiency
        total_pixels = height * width
        sample_size = int(total_pixels * sample_rate)
        
        if sample_rate < 1.0:
            print(f"  Sampling {sample_size:,} of {total_pixels:,} pixels ({sample_rate*100:.1f}%)")
            rows = np.random.choice(height, size=sample_size, replace=True)
            cols = np.random.choice(width, size=sample_size, replace=True)
        else:
            print(f"  Processing all {total_pixels:,} pixels")
            rows, cols = np.meshgrid(range(height), range(width), indexing='ij')
            rows = rows.flatten()
            cols = cols.flatten()
        
        # Process each sampled pixel
        for idx, (row, col) in enumerate(zip(rows, cols)):
            if idx % 10000 == 0:
                print(f"    Processed {idx:,} / {len(rows):,} pixels", end='\r')
            
            # Get H3 index for this pixel
            h3_idx = self.pixel_to_h3(col, row, transform, resolution)
            
            # Get pixel values across all bands
            pixel_values = data[:, row, col]
            
            # Skip nodata pixels
            if np.any(np.isnan(pixel_values)) or np.all(pixel_values == 0):
                continue
            
            # Aggregate (running mean)
            if h3_idx not in h3_aggregates:
                h3_aggregates[h3_idx] = {
                    'values': pixel_values,
                    'count': 1
                }
            else:
                h3_aggregates[h3_idx]['values'] += pixel_values
                h3_aggregates[h3_idx]['count'] += 1
        
        print(f"\n  Found {len(h3_aggregates)} H3 cells at resolution {resolution}")
        
        # Convert to DataFrame
        rows = []
        for h3_idx, agg in h3_aggregates.items():
            # Calculate mean embeddings
            mean_values = agg['values'] / agg['count']
            
            row = {'h3': h3_idx, 'pixel_count': agg['count']}
            
            # Add each embedding dimension
            for band_idx in range(bands):
                row[f'A{band_idx:02d}'] = mean_values[band_idx]
            
            # Calculate information metrics if enabled
            if self.calculate_info_metrics and self.info_calculator:
                try:
                    # Calculate entropy for this embedding
                    entropy = self.info_calculator.calculate_entropy(
                        mean_values.reshape(1, -1)
                    )
                    row['entropy'] = entropy
                    
                    # Calculate embedding magnitude (information content proxy)
                    magnitude = np.linalg.norm(mean_values)
                    row['embedding_magnitude'] = magnitude
                    
                    # Calculate variance across embedding dimensions
                    variance = np.var(mean_values)
                    row['embedding_variance'] = variance
                    
                except Exception as e:
                    print(f"    Warning: Failed to calculate info metrics for {h3_idx}: {e}")
                    row['entropy'] = 0.0
                    row['embedding_magnitude'] = 0.0
                    row['embedding_variance'] = 0.0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.set_index('h3', inplace=True)
        
        # Add geometry for visualization
        df['geometry'] = df.index.map(lambda x: Polygon(h3.cell_to_boundary(x)))
        
        # Calculate spatial information metrics if enabled
        if self.calculate_info_metrics and self.spatial_info and len(df) > 1:
            print(f"  Calculating spatial information metrics...")
            
            # Convert to format needed by spatial info calculator
            embeddings_dict = {}
            embed_cols = [col for col in df.columns if col.startswith('A')]
            
            for h3_idx, row in df.iterrows():
                embeddings_dict[h3_idx] = row[embed_cols].values
            
            # Calculate spatial mutual information for each hexagon
            spatial_mi = {}
            for h3_idx in df.index:
                try:
                    mi = self.spatial_info.calculate_spatial_mutual_information(
                        embeddings_dict, h3_idx, neighbor_ring=1
                    )
                    spatial_mi[h3_idx] = mi
                except Exception as e:
                    print(f"    Warning: Failed to calculate spatial MI for {h3_idx}: {e}")
                    spatial_mi[h3_idx] = 0.0
            
            df['spatial_mutual_info'] = df.index.map(spatial_mi)
        
        # Calculate statistics
        print(f"  H3 cells: {len(df)}")
        print(f"  Mean pixels per cell: {df['pixel_count'].mean():.1f}")
        print(f"  Min/Max pixels: {df['pixel_count'].min()} / {df['pixel_count'].max()}")
        
        if 'entropy' in df.columns:
            print(f"  Mean entropy: {df['entropy'].mean():.4f}")
            print(f"  Mean embedding magnitude: {df['embedding_magnitude'].mean():.4f}")
            if 'spatial_mutual_info' in df.columns:
                print(f"  Mean spatial MI: {df['spatial_mutual_info'].mean():.4f}")
        
        return df
    
    def create_hierarchical_mapping(self, dfs: Dict[int, pd.DataFrame]) -> Dict:
        """
        Create parent-child mappings between H3 resolutions.
        
        Args:
            dfs: Dictionary mapping resolution to DataFrames
            
        Returns:
            Dictionary of hierarchical mappings
        """
        print("\nCreating hierarchical H3 mappings...")
        
        mappings = {}
        
        resolutions = sorted(dfs.keys())
        for i in range(len(resolutions) - 1):
            child_res = resolutions[i + 1]
            parent_res = resolutions[i]
            
            if child_res - parent_res != 1:
                continue
            
            print(f"  Mapping resolution {child_res} -> {parent_res}")
            
            child_to_parent = {}
            for child_h3 in dfs[child_res].index:
                parent_h3 = h3.cell_to_parent(child_h3, parent_res)
                child_to_parent[child_h3] = parent_h3
            
            mappings[f"res{child_res}_to_res{parent_res}"] = child_to_parent
            
            # Count children per parent
            parent_counts = pd.Series(child_to_parent).value_counts()
            print(f"    Mean children per parent: {parent_counts.mean():.1f}")
        
        return mappings
    
    def process_year(self, year: int, sample_rate: float = 1.0) -> Dict[int, pd.DataFrame]:
        """
        Process AlphaEarth data for a specific year.
        
        Args:
            year: Year to process
            sample_rate: Fraction of pixels to sample
            
        Returns:
            Dictionary mapping H3 resolution to DataFrames
        """
        print(f"\n{'='*60}")
        print(f"Processing year {year}")
        print(f"{'='*60}")
        
        # Find input file
        pattern = f"DelNorte_AlphaEarth_{year}*.tif"
        input_files = list(self.input_dir.glob(pattern))
        
        if not input_files:
            print(f"[WARNING] No files found matching: {pattern}")
            return {}
        
        input_file = input_files[0]
        print(f"Found input file: {input_file.name}")
        
        # Load GeoTIFF
        data, metadata = self.load_geotiff(input_file)
        
        # Process each H3 resolution
        results = {}
        for resolution in self.h3_resolutions:
            df = self.aggregate_to_h3(data, metadata, resolution, sample_rate)
            results[resolution] = df
            
            # Save to file
            output_path = self.output_dir / f"resolution_{resolution}" / f"del_norte_{year}_h3_res{resolution}.parquet"
            df.to_parquet(output_path)
            print(f"  Saved: {output_path}")
        
        # Create hierarchical mappings
        mappings = self.create_hierarchical_mapping(results)
        
        # Save mappings
        mappings_path = self.output_dir / f"hierarchical_mappings_{year}.json"
        with open(mappings_path, 'w') as f:
            # Convert to serializable format
            mappings_serializable = {
                key: {k: v for k, v in mapping.items()}
                for key, mapping in mappings.items()
            }
            json.dump(mappings_serializable, f, indent=2)
        print(f"  Saved mappings: {mappings_path}")
        
        # Save metadata with information metrics summary
        meta_path = self.output_dir / f"metadata_{year}.json"
        with open(meta_path, 'w') as f:
            meta_info = {
                'year': year,
                'source_file': input_file.name,
                'processed_date': datetime.now().isoformat(),
                'h3_resolutions': self.h3_resolutions,
                'h3_cell_counts': {res: len(df) for res, df in results.items()},
                'sample_rate': sample_rate,
                'information_metrics_enabled': self.calculate_info_metrics,
                'geotiff_metadata': {k: str(v) for k, v in metadata.items()}
            }
            
            # Add information metrics summary if calculated
            if self.calculate_info_metrics:
                info_summary = {}
                for res, df in results.items():
                    if 'entropy' in df.columns:
                        info_summary[res] = {
                            'mean_entropy': float(df['entropy'].mean()),
                            'std_entropy': float(df['entropy'].std()),
                            'mean_magnitude': float(df['embedding_magnitude'].mean()),
                            'mean_variance': float(df['embedding_variance'].mean())
                        }
                        if 'spatial_mutual_info' in df.columns:
                            info_summary[res]['mean_spatial_mi'] = float(df['spatial_mutual_info'].mean())
                
                meta_info['information_summary'] = info_summary
            
            json.dump(meta_info, f, indent=2)
        print(f"  Saved metadata: {meta_path}")
        
        return results
    
    def create_summary_report(self, years: List[int]):
        """
        Create summary report of processed data.
        
        Args:
            years: List of years processed
        """
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        
        summary = {
            'county': 'Del Norte',
            'state': 'California',
            'years_processed': years,
            'h3_resolutions': self.h3_resolutions,
            'output_directory': str(self.output_dir),
            'files_created': []
        }
        
        # Count files created
        for res in self.h3_resolutions:
            res_dir = self.output_dir / f"resolution_{res}"
            files = list(res_dir.glob("*.parquet"))
            summary['files_created'].extend([str(f) for f in files])
            print(f"  Resolution {res}: {len(files)} files")
        
        print(f"\nTotal files created: {len(summary['files_created'])}")
        
        # Save summary
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Process Del Norte AlphaEarth data to H3 hexagons'
    )
    parser.add_argument(
        '--input-dir',
        default='data/alphaearth/del_norte',
        help='Directory containing downloaded GeoTIFF files'
    )
    parser.add_argument(
        '--output-dir',
        default='data/h3_processed/del_norte',
        help='Output directory for H3 data'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=[2023],
        help='Years to process'
    )
    parser.add_argument(
        '--resolutions',
        nargs='+',
        type=int,
        default=[7, 8, 9, 10],
        help='H3 resolutions to generate'
    )
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=1.0,
        help='Fraction of pixels to sample (1.0 = all pixels)'
    )
    parser.add_argument(
        '--no-info-metrics',
        action='store_true',
        help='Disable information theory metrics calculation'
    )
    
    args = parser.parse_args()
    
    print("Del Norte AlphaEarth to H3 Processor")
    print("="*60)
    
    # Initialize processor
    processor = DelNorteH3Processor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        h3_resolutions=args.resolutions,
        calculate_info_metrics=not args.no_info_metrics
    )
    
    # Process each year
    for year in args.years:
        processor.process_year(year, sample_rate=args.sample_rate)
    
    # Create summary
    processor.create_summary_report(args.years)
    
    print("\nProcessing complete!")
    print(f"H3 data saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())