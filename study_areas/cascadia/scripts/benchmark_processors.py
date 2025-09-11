#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance benchmark comparing rasterio vs rioxarray processors.
Measures speed, memory usage, and output quality for AlphaEarth processing embeddings.
"""

import numpy as np
import pandas as pd
import time
import psutil
import gc
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import yaml

# Import all processors
from load_alphaearth import AlphaEarthToH3Converter
from rioxarray_processor import RioxarrayAlphaEarthProcessor, RIOXARRAY_AVAILABLE
from srai_rioxarray_processor import SRAIRioxarrayProcessor

logger = logging.getLogger(__name__)


class ProcessorBenchmark:
    """Benchmark different AlphaEarth processing embeddings approaches."""
    
    def __init__(self, config: dict):
        """Initialize benchmark with configuration."""
        self.config = config
        self.results = {}
        
        # Setup logging
        logging.basicConfig(
            level=config['output']['log_level'],
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_test_files(self, n_files: int = 5) -> List[Path]:
        """Get a subset of files for testing."""
        source_dir = Path(self.config['data']['source_dir'])
        pattern = self.config['data']['pattern']
        
        all_files = list(source_dir.glob(pattern))
        test_files = sorted(all_files)[:n_files]
        
        logger.info(f"Selected {len(test_files)} files for benchmark")
        return test_files
    
    def benchmark_rasterio_processor(self, test_files: List[Path]) -> Dict:
        """Benchmark the current rasterio-based processor."""
        logger.info("=" * 60)
        logger.info("BENCHMARKING RASTERIO PROCESSOR")
        logger.info("=" * 60)
        
        # Update config to use only test files
        test_config = self.config.copy()
        test_config['processing embeddings']['batch_size'] = len(test_files)
        
        # Initialize processor
        processor = AlphaEarthToH3Converter(test_config)
        
        # Override file list to use only test files
        original_get_files = processor.get_tiff_files
        processor.get_tiff_files = lambda: test_files
        
        # Measure performance
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        # Process files
        hex_df = processor.process_all()
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        # Calculate metrics
        processing_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        results = {
            'processor': 'rasterio',
            'n_files': len(test_files),
            'processing_time': processing_time,
            'memory_used_mb': memory_used,
            'n_hexagons': len(hex_df) if hex_df is not None and not hex_df.empty else 0,
            'hexagons_per_second': len(hex_df) / processing_time if hex_df is not None and not hex_df.empty and processing_time > 0 else 0,
            'mb_per_hexagon': memory_used / len(hex_df) if hex_df is not None and not hex_df.empty else 0,
            'success': hex_df is not None and not hex_df.empty
        }
        
        # Save results for comparison
        if hex_df is not None and not hex_df.empty:
            output_path = Path("data/benchmark/rasterio_results.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            hex_df.to_parquet(output_path)
            results['output_file'] = str(output_path)
        
        logger.info(f"Rasterio Results: {results}")
        return results
    
    def benchmark_rioxarray_processor(self, test_files: List[Path]) -> Dict:
        """Benchmark the rioxarray-based processor."""
        logger.info("=" * 60)
        logger.info("BENCHMARKING RIOXARRAY PROCESSOR")
        logger.info("=" * 60)
        
        if not RIOXARRAY_AVAILABLE:
            logger.error("Rioxarray not available for benchmarking")
            return {'processor': 'rioxarray', 'success': False, 'error': 'rioxarray_not_available'}
        
        # Update config to use only test files
        test_config = self.config.copy()
        test_config['processing embeddings']['batch_size'] = len(test_files)
        
        # Add rioxarray config if not present
        if 'rioxarray' not in test_config:
            test_config['rioxarray'] = {
                'chunk_size_mb': 100,
                'use_parallel': True,
                'optimize_chunks': True
            }
        
        # Initialize processor
        processor = RioxarrayAlphaEarthProcessor(test_config)
        
        # Override file list to use only test files
        original_get_files = processor.get_tiff_files
        processor.get_tiff_files = lambda: test_files
        
        # Measure performance
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        # Process files
        hex_df = processor.process_all_rioxarray()
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        # Calculate metrics
        processing_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        results = {
            'processor': 'rioxarray',
            'n_files': len(test_files),
            'processing_time': processing_time,
            'memory_used_mb': memory_used,
            'n_hexagons': len(hex_df) if hex_df is not None and not hex_df.empty else 0,
            'hexagons_per_second': len(hex_df) / processing_time if hex_df is not None and not hex_df.empty and processing_time > 0 else 0,
            'mb_per_hexagon': memory_used / len(hex_df) if hex_df is not None and not hex_df.empty else 0,
            'success': hex_df is not None and not hex_df.empty,
            'chunk_size_mb': test_config['rioxarray']['chunk_size_mb']
        }
        
        # Save results for comparison
        if hex_df is not None and not hex_df.empty:
            output_path = Path("data/benchmark/rioxarray_results.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            hex_df.to_parquet(output_path)
            results['output_file'] = str(output_path)
        
        logger.info(f"Rioxarray Results: {results}")
        return results
    
    def benchmark_srai_rioxarray_processor(self, test_files: List[Path]) -> Dict:
        """Benchmark the SRAI+Rioxarray optimized processor."""
        logger.info("=" * 60)
        logger.info("BENCHMARKING SRAI+RIOXARRAY PROCESSOR")
        logger.info("=" * 60)
        
        # Update config to use only test files
        test_config = self.config.copy()
        test_config['processing embeddings']['batch_size'] = len(test_files)
        
        # Add SRAI config if not present
        if 'srai' not in test_config:
            test_config['srai'] = {
                'use_vectorized': True,
                'cache_regionalizer': True,
                'use_neighbourhood': True,
                'parallel_h3_conversion': True,
                'combined_processing': True,
                'adaptive_sampling': True
            }
        
        # Initialize processor
        processor = SRAIRioxarrayProcessor(test_config)
        
        # Override file list to use only test files
        original_get_files = processor.get_tiff_files
        processor.get_tiff_files = lambda: test_files
        
        # Measure performance
        start_time = time.time()
        start_memory = self.measure_memory_usage()
        
        # Process files
        gdf = processor.process_all()
        
        end_time = time.time()
        end_memory = self.measure_memory_usage()
        
        # Calculate metrics
        processing_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        results = {
            'processor': 'srai_rioxarray',
            'n_files': len(test_files),
            'processing_time': processing_time,
            'memory_used_mb': memory_used,
            'n_hexagons': len(gdf) if gdf is not None and not gdf.empty else 0,
            'hexagons_per_second': len(gdf) / processing_time if gdf is not None and not gdf.empty and processing_time > 0 else 0,
            'mb_per_hexagon': memory_used / len(gdf) if gdf is not None and not gdf.empty else 0,
            'success': gdf is not None and not gdf.empty,
            'chunk_size_mb': test_config['rioxarray']['chunk_size_mb'],
            'h3_resolution': test_config['data']['h3_resolution']
        }
        
        # Save results for comparison
        if gdf is not None and not gdf.empty:
            output_path = Path("data/benchmark/srai_rioxarray_results.parquet")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Save as regular DataFrame (without geometry for parquet)
            df_to_save = pd.DataFrame(gdf.drop(columns='geometry'))
            df_to_save.to_parquet(output_path)
            results['output_file'] = str(output_path)
        
        logger.info(f"SRAI+Rioxarray Results: {results}")
        return results
    
    def compare_outputs(self, rasterio_results: Dict, rioxarray_results: Dict) -> Dict:
        """Compare the output quality between processors."""
        comparison = {
            'outputs_comparable': False,
            'hexagon_count_diff': 0,
            'spatial_overlap': 0.0,
            'feature_correlation': 0.0
        }
        
        if not (rasterio_results.get('success') and rioxarray_results.get('success')):
            logger.warning("Cannot compare outputs - one or both processors failed")
            return comparison
        
        try:
            # Load both datasets
            rasterio_df = pd.read_parquet(rasterio_results['output_file'])
            rioxarray_df = pd.read_parquet(rioxarray_results['output_file'])
            
            # Compare hexagon counts
            comparison['hexagon_count_diff'] = abs(len(rasterio_df) - len(rioxarray_df))
            
            # Find overlapping hexagons
            common_hexagons = set(rasterio_df['h3_index']).intersection(set(rioxarray_df['h3_index']))
            comparison['spatial_overlap'] = len(common_hexagons) / max(len(rasterio_df), len(rioxarray_df))
            
            # Compare feature values for common hexagons
            if common_hexagons:
                # Get common data
                rasterio_common = rasterio_df[rasterio_df['h3_index'].isin(common_hexagons)].set_index('h3_index')
                rioxarray_common = rioxarray_df[rioxarray_df['h3_index'].isin(common_hexagons)].set_index('h3_index')
                
                # Compare band values
                band_cols = [f'band_{i:02d}' for i in range(64)]
                available_bands = [col for col in band_cols if col in rasterio_common.columns and col in rioxarray_common.columns]
                
                if available_bands:
                    correlations = []
                    for band in available_bands:
                        corr = np.corrcoef(rasterio_common[band], rioxarray_common[band])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                    
                    comparison['feature_correlation'] = np.mean(correlations) if correlations else 0.0
            
            comparison['outputs_comparable'] = True
            logger.info(f"Output comparison: {comparison}")
            
        except Exception as e:
            logger.error(f"Error comparing outputs: {e}")
        
        return comparison
    
    def run_benchmark(self, n_files: int = 5) -> Dict:
        """Run complete benchmark comparing all three processors."""
        logger.info(f"Starting performance benchmark with {n_files} files...")
        
        # Get test files
        test_files = self.get_test_files(n_files)
        if not test_files:
            logger.error("No test files found!")
            return {}
        
        # Clean memory before starting
        gc.collect()
        
        # Benchmark rasterio processor
        rasterio_results = self.benchmark_rasterio_processor(test_files)
        
        # Clean memory between tests
        gc.collect()
        
        # Benchmark rioxarray processor
        rioxarray_results = self.benchmark_rioxarray_processor(test_files)
        
        # Clean memory between tests
        gc.collect()
        
        # Benchmark SRAI+rioxarray processor
        srai_rioxarray_results = self.benchmark_srai_rioxarray_processor(test_files)
        
        # Compare outputs (basic comparison between first two for now)
        comparison = self.compare_outputs(rasterio_results, rioxarray_results)
        
        # Compile final results
        benchmark_results = {
            'benchmark_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_files': [str(f) for f in test_files],
            'rasterio': rasterio_results,
            'rioxarray': rioxarray_results,
            'srai_rioxarray': srai_rioxarray_results,
            'comparison': comparison,
            'winner': self.determine_winner_three_way(rasterio_results, rioxarray_results, srai_rioxarray_results)
        }
        
        # Save benchmark results
        self.save_benchmark_results(benchmark_results)
        
        return benchmark_results
    
    def determine_winner(self, rasterio_results: Dict, rioxarray_results: Dict) -> Dict:
        """Determine which processor performed better."""
        winner = {
            'overall': 'none',
            'speed_winner': 'none',
            'memory_winner': 'none',
            'quality_winner': 'none'
        }
        
        if not (rasterio_results.get('success') and rioxarray_results.get('success')):
            if rasterio_results.get('success'):
                winner['overall'] = 'rasterio'
            elif rioxarray_results.get('success'):
                winner['overall'] = 'rioxarray'
            return winner
        
        # Speed comparison
        if rasterio_results['processing_time'] < rioxarray_results['processing_time']:
            winner['speed_winner'] = 'rasterio'
        else:
            winner['speed_winner'] = 'rioxarray'
        
        # Memory comparison
        if rasterio_results['memory_used_mb'] < rioxarray_results['memory_used_mb']:
            winner['memory_winner'] = 'rasterio'
        else:
            winner['memory_winner'] = 'rioxarray'
        
        # Quality comparison (prefer higher hexagon count as proxy for completeness)
        if rasterio_results['n_hexagons'] > rioxarray_results['n_hexagons']:
            winner['quality_winner'] = 'rasterio'
        else:
            winner['quality_winner'] = 'rioxarray'
        
        # Overall winner (weighted: speed=40%, memory=30%, quality=30%)
        rasterio_score = 0
        rioxarray_score = 0
        
        if winner['speed_winner'] == 'rasterio':
            rasterio_score += 0.4
        else:
            rioxarray_score += 0.4
        
        if winner['memory_winner'] == 'rasterio':
            rasterio_score += 0.3
        else:
            rioxarray_score += 0.3
        
        if winner['quality_winner'] == 'rasterio':
            rasterio_score += 0.3
        else:
            rioxarray_score += 0.3
        
        if rasterio_score > rioxarray_score:
            winner['overall'] = 'rasterio'
        else:
            winner['overall'] = 'rioxarray'
        
        return winner
    
    def determine_winner_three_way(self, rasterio_results: Dict, rioxarray_results: Dict, srai_results: Dict) -> Dict:
        """Determine which of the three processors performed best."""
        winner = {
            'overall': 'none',
            'speed_winner': 'none',
            'memory_winner': 'none',
            'quality_winner': 'none'
        }
        
        # Only compare successful processors
        successful_processors = []
        results_dict = {
            'rasterio': rasterio_results,
            'rioxarray': rioxarray_results,
            'srai_rioxarray': srai_results
        }
        
        for name, results in results_dict.items():
            if results.get('success'):
                successful_processors.append((name, results))
        
        if not successful_processors:
            return winner
        elif len(successful_processors) == 1:
            winner['overall'] = successful_processors[0][0]
            return winner
        
        # Speed comparison
        fastest_time = float('inf')
        fastest_processor = None
        for name, results in successful_processors:
            if results['processing_time'] < fastest_time:
                fastest_time = results['processing_time']
                fastest_processor = name
        winner['speed_winner'] = fastest_processor
        
        # Memory comparison
        lowest_memory = float('inf')
        memory_winner = None
        for name, results in successful_processors:
            if results['memory_used_mb'] < lowest_memory:
                lowest_memory = results['memory_used_mb']
                memory_winner = name
        winner['memory_winner'] = memory_winner
        
        # Quality comparison (highest hexagon count)
        highest_quality = 0
        quality_winner = None
        for name, results in successful_processors:
            if results['n_hexagons'] > highest_quality:
                highest_quality = results['n_hexagons']
                quality_winner = name
        winner['quality_winner'] = quality_winner
        
        # Overall winner (weighted scoring)
        scores = {name: 0 for name, _ in successful_processors}
        
        # Speed: 40% weight
        if fastest_processor:
            scores[fastest_processor] += 0.4
        
        # Memory: 30% weight
        if memory_winner:
            scores[memory_winner] += 0.3
        
        # Quality: 30% weight
        if quality_winner:
            scores[quality_winner] += 0.3
        
        # Find overall winner
        if scores:
            overall_winner = max(scores, key=scores.get)
            winner['overall'] = overall_winner
        
        return winner
    
    def save_benchmark_results(self, results: Dict):
        """Save benchmark results to file."""
        output_dir = Path("data/benchmark")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        import json
        json_path = output_dir / "processor_benchmark.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {json_path}")
        
        # Print summary
        self.print_benchmark_summary(results)
    
    def print_benchmark_summary(self, results: Dict):
        """Print a formatted summary of benchmark results."""
        print("\n" + "="*95)
        print("PROCESSOR BENCHMARK SUMMARY")
        print("="*95)
        
        rasterio = results.get('rasterio', {})
        rioxarray = results.get('rioxarray', {})
        srai_rioxarray = results.get('srai_rioxarray', {})
        winner = results.get('winner', {})
        
        print(f"Test Files: {len(results.get('test_files', []))}")
        print(f"Timestamp: {results.get('benchmark_timestamp', 'Unknown')}")
        print()
        
        print("PERFORMANCE METRICS:")
        print("-" * 95)
        print(f"{'Metric':<25} {'Rasterio':<15} {'Rioxarray':<15} {'SRAI+Riox':<15} {'Winner':<15}")
        print("-" * 95)
        
        # Check if all processors succeeded
        all_success = (rasterio.get('success') and 
                      rioxarray.get('success') and 
                      srai_rioxarray.get('success'))
        
        if all_success:
            print(f"{'Processing Time (s)':<25} {rasterio.get('processing_time', 0):<15.1f} {rioxarray.get('processing_time', 0):<15.1f} {srai_rioxarray.get('processing_time', 0):<15.1f} {winner.get('speed_winner', 'N/A'):<15}")
            print(f"{'Memory Used (MB)':<25} {rasterio.get('memory_used_mb', 0):<15.1f} {rioxarray.get('memory_used_mb', 0):<15.1f} {srai_rioxarray.get('memory_used_mb', 0):<15.1f} {winner.get('memory_winner', 'N/A'):<15}")
            print(f"{'Hexagons Generated':<25} {rasterio.get('n_hexagons', 0):<15} {rioxarray.get('n_hexagons', 0):<15} {srai_rioxarray.get('n_hexagons', 0):<15} {winner.get('quality_winner', 'N/A'):<15}")
            print(f"{'Hexagons/Second':<25} {rasterio.get('hexagons_per_second', 0):<15.1f} {rioxarray.get('hexagons_per_second', 0):<15.1f} {srai_rioxarray.get('hexagons_per_second', 0):<15.1f} {'N/A':<15}")
        else:
            print("Some processors failed - see detailed results")
            print(f"Rasterio: {'SUCCESS' if rasterio.get('success') else 'FAILED'}")
            print(f"Rioxarray: {'SUCCESS' if rioxarray.get('success') else 'FAILED'}")
            print(f"SRAI+Rioxarray: {'SUCCESS' if srai_rioxarray.get('success') else 'FAILED'}")
        
        print("-" * 95)
        print(f"{'OVERALL WINNER':<25} {winner.get('overall', 'N/A').upper()}")
        print("="*95)


def main():
    """Main entry point for benchmark."""
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize benchmark
    benchmark = ProcessorBenchmark(config)
    
    # Run benchmark with 5 files
    results = benchmark.run_benchmark(n_files=5)
    
    return results


if __name__ == "__main__":
    main()