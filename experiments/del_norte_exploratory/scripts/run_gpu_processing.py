#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU-Accelerated Del Norte 2021 Full Dataset Processing
Leverages RTX 3090 + 12-core CPU + NVMe SSD for maximum performance

Features:
- PyTorch GPU acceleration with mixed precision
- SRAI integration for H3 operations
- Vectorized geospatial mapping
- Progressive processing with checkpointing
- Real-time performance monitoring
"""

import os
import sys
import yaml
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gpu_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DelNorteGPUProcessor:
    """Main orchestrator for GPU-accelerated Del Norte processing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        self.config = self.load_config(config_path)
        self.setup_hardware()
        self.setup_directories()
        self.checkpoint_file = Path("data/processing_checkpoint.json")
        self.start_time = time.time()
        
    def load_config(self, config_path: str) -> dict:
        """Load and enhance configuration for GPU processing"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update for full dataset at resolution 8
        config['experiment']['h3_resolution'] = 8
        config['data']['h3_resolution'] = 8
        config['processing']['h3_resolution'] = 8
        config['experiment']['max_tiles'] = None  # Process all tiles
        
        # Add GPU-specific settings
        if 'hardware' not in config:
            config['hardware'] = {}
        
        config['hardware'].update({
            'max_cores': 12,
            'gpu_memory_gb': 20,
            'prefetch_tiles': 5,
            'gpu_chunk_size': 2048,
            'pinned_memory': True,
            'async_transfer': True,
            'use_mixed_precision': True
        })
        
        # Optimize batch sizes for GPU
        config['processing']['batch_size'] = 20  # Process more tiles at once
        config['processing']['chunk_size'] = 2048  # Larger chunks for GPU
        
        logger.info(f"Configuration loaded: H3 Resolution {config['data']['h3_resolution']}")
        logger.info(f"Processing mode: {'GPU-Accelerated' if torch.cuda.is_available() else 'CPU Fallback'}")
        
        return config
    
    def setup_hardware(self):
        """Initialize GPU and hardware resources"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"GPU Detected: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            
            # Enable TF32 for RTX 3090
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            logger.info("GPU optimizations enabled: TF32, cuDNN benchmark")
        else:
            logger.warning("No GPU detected! Falling back to CPU processing")
    
    def setup_directories(self):
        """Create necessary output directories"""
        self.output_dir = Path("data/h3_2021_res8")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path("plots/gpu_processing")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self) -> Dict:
        """Load processing checkpoint if exists"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                logger.info(f"Resuming from checkpoint: {checkpoint['processed_tiles']} tiles already processed")
                return checkpoint
        return {'processed_tiles': [], 'completed': False}
    
    def save_checkpoint(self, processed_tiles: List[str]):
        """Save processing checkpoint"""
        checkpoint = {
            'processed_tiles': processed_tiles,
            'timestamp': datetime.now().isoformat(),
            'completed': False
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def run_gpu_processor(self) -> gpd.GeoDataFrame:
        """Run the enhanced GPU processor"""
        from scripts.pytorch_tiff_processor import PyTorchTiffProcessor
        
        logger.info("Initializing PyTorch GPU processor...")
        processor = PyTorchTiffProcessor(self.config)
        
        # Load checkpoint
        checkpoint = self.load_checkpoint()
        processor.set_checkpoint(checkpoint['processed_tiles'])
        
        # Process all tiles
        logger.info("Starting GPU-accelerated processing...")
        gdf = processor.process_all_gpu_optimized()
        
        return gdf
    
    def run_fallback_processor(self) -> gpd.GeoDataFrame:
        """Run existing GPU multicore processor as fallback"""
        from scripts.gpu_multicore_processor import GPUMulticoreProcessor
        
        logger.info("Using existing GPU multicore processor...")
        processor = GPUMulticoreProcessor(self.config)
        gdf = processor.process_all_hardware_optimized()
        
        return gdf
    
    def run_clustering(self, gdf: gpd.GeoDataFrame):
        """Run clustering algorithms on processed data"""
        from scripts.clustering import run_clustering_analysis
        
        logger.info("Running clustering analysis...")
        
        # Save H3 data first
        h3_data_path = self.output_dir / "del_norte_2021_res8_gpu_full.parquet"
        df_to_save = pd.DataFrame(gdf.drop(columns='geometry'))
        df_to_save.to_parquet(h3_data_path, compression='snappy')
        
        # Run clustering
        clustering_config = self.config.get('clustering', {})
        run_clustering_analysis(df_to_save, clustering_config, self.output_dir)
    
    def run_srai_visualization(self, gdf: gpd.GeoDataFrame):
        """Generate SRAI-based visualizations"""
        from scripts.srai_visualizations import SRAIVisualizer
        
        logger.info("Generating SRAI visualizations...")
        visualizer = SRAIVisualizer(self.config)
        
        # Load clustering results
        cluster_files = list((self.output_dir.parent.parent / "results/clusters").glob("*_2021_res8_*.parquet"))
        
        for cluster_file in cluster_files[:3]:  # Visualize top 3 clustering results
            logger.info(f"Visualizing {cluster_file.name}")
            visualizer.visualize_clusters(cluster_file)
    
    def generate_report(self, gdf: gpd.GeoDataFrame):
        """Generate processing report with statistics"""
        processing_time = time.time() - self.start_time
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'processing_time_formatted': f"{processing_time/60:.1f} minutes",
            'hardware': {
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
                'cpu_cores': self.config['hardware']['max_cores']
            },
            'dataset': {
                'year': 2021,
                'h3_resolution': 8,
                'total_hexagons': len(gdf),
                'total_pixels': gdf['total_pixels'].sum() if 'total_pixels' in gdf else 0,
                'avg_pixels_per_hex': gdf['total_pixels'].mean() if 'total_pixels' in gdf else 0
            },
            'coverage': {
                'min_lat': gdf.geometry.bounds.miny.min(),
                'max_lat': gdf.geometry.bounds.maxy.max(),
                'min_lon': gdf.geometry.bounds.minx.min(),
                'max_lon': gdf.geometry.bounds.maxx.max()
            }
        }
        
        # Save report
        report_path = self.output_dir / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("="*60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Total time: {report['processing_time_formatted']}")
        logger.info(f"Total hexagons: {report['dataset']['total_hexagons']:,}")
        logger.info(f"Hardware: {report['hardware']['gpu']}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("="*60)
        
        return report
    
    def run(self):
        """Main execution pipeline"""
        try:
            logger.info("="*60)
            logger.info("DEL NORTE 2021 GPU-ACCELERATED PROCESSING")
            logger.info("="*60)
            
            # Step 1: Process TIFF files to H3
            logger.info("Step 1: Processing AlphaEarth TIFF files...")
            
            # Try PyTorch processor first, fallback to existing
            try:
                gdf = self.run_gpu_processor()
            except Exception as e:
                logger.warning(f"PyTorch processor failed: {e}")
                logger.info("Falling back to existing GPU processor...")
                gdf = self.run_fallback_processor()
            
            if gdf.empty:
                logger.error("No data processed!")
                return
            
            # Save results
            logger.info("Saving H3 hexagon data...")
            output_path = self.output_dir / "del_norte_2021_res8_gpu_full.parquet"
            df_to_save = pd.DataFrame(gdf.drop(columns='geometry'))
            df_to_save.to_parquet(output_path, compression='snappy')
            
            # Save GeoPackage
            gpkg_path = output_path.with_suffix('.gpkg')
            gdf.to_file(gpkg_path, driver='GPKG')
            logger.info(f"Saved {len(gdf)} hexagons to {output_path}")
            
            # Step 2: Run clustering
            logger.info("Step 2: Running clustering analysis...")
            self.run_clustering(gdf)
            
            # Step 3: Generate visualizations
            logger.info("Step 3: Generating visualizations...")
            self.run_srai_visualization(gdf)
            
            # Step 4: Generate report
            logger.info("Step 4: Generating final report...")
            report = self.generate_report(gdf)
            
            # Mark as complete
            checkpoint = self.load_checkpoint()
            checkpoint['completed'] = True
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    processor = DelNorteGPUProcessor()
    processor.run()


if __name__ == "__main__":
    main()