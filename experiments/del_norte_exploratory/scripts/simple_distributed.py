#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Distributed SRAI H3 Processing with Multiple Workers

Launches multiple SRAI worker processes using subprocess for better Windows compatibility.
Each worker processes TIFF files independently while a master coordinates and monitors.
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import psutil
import yaml
import pandas as pd
import geopandas as gpd
from progress_tracker import ProgressTracker, ProgressDashboard
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simple_distributed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleDistributedSRAI:
    """Simple distributed SRAI processing coordinator"""
    
    def __init__(self, config_path: str = "config.yaml", num_workers: int = 6, enable_dashboard: bool = True):
        self.config_path = config_path
        self.num_workers = num_workers
        self.worker_processes = []
        self.start_time = time.time()
        self.enable_dashboard = enable_dashboard
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directories
        self.output_dir = Path("data/h3_2021_res8_distributed_simple")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logs_dir = Path("logs/workers")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(config_path, num_workers)
        
        # Initialize dashboard
        self.dashboard = None
        self.dashboard_thread = None
        if enable_dashboard:
            self.dashboard = ProgressDashboard(self.progress_tracker)
        
        logger.info(f"Simple Distributed SRAI initialized with {num_workers} workers")
        if enable_dashboard:
            logger.info("Progress dashboard will be available at http://localhost:8080")
    
    def get_tiff_files(self) -> List[str]:
        """Get list of TIFF files to process"""
        source_dir = Path(self.config['data']['source_dir'])
        pattern = self.config['data']['pattern']
        
        tiff_files = list(source_dir.glob(pattern))
        logger.info(f"Found {len(tiff_files)} TIFF files to process")
        return [str(f) for f in tiff_files]
    
    def split_files_for_workers(self, files: List[str]) -> List[List[str]]:
        """Split files among workers"""
        chunk_size = len(files) // self.num_workers
        remainder = len(files) % self.num_workers
        
        worker_files = []
        start_idx = 0
        
        for i in range(self.num_workers):
            # Add one extra file to first 'remainder' workers
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            worker_files.append(files[start_idx:end_idx])
            start_idx = end_idx
        
        return worker_files
    
    def create_worker_script(self, worker_id: int, files_to_process: List[str]):
        """Create a worker script for processing assigned files with progress tracking"""
        worker_script = f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent paths
sys.path.append(str(Path(__file__).parent.parent.parent))

import warnings
warnings.filterwarnings('ignore')

# Setup worker logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Worker-{worker_id} - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/workers/worker_{worker_id}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Worker {worker_id} starting...")
    
    try:
        # Import SRAI processor and progress tracker
        from scripts.srai_rioxarray_processor import SRAIRioxarrayProcessor
        from progress_tracker import ProgressTracker
        import yaml
        import pandas as pd
        
        # Load config
        with open('{self.config_path}', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker('{self.config_path}', {self.num_workers})
        
        # Modify config for single worker
        config['processing']['batch_size'] = 1  # Process one file at a time
        config['experiment']['worker_id'] = {worker_id}
        
        # Initialize processor
        processor = SRAIRioxarrayProcessor(config)
        
        # Files to process
        files_to_process = {files_to_process}
        
        logger.info(f"Worker {worker_id} processing {{len(files_to_process)}} files")
        
        # Mark worker as started
        progress_tracker.worker_progress[{worker_id}].start_time = datetime.now().isoformat()
        progress_tracker.worker_progress[{worker_id}].status = "running"
        progress_tracker.save_progress()
        
        processed_count = 0
        
        for file_path in files_to_process:
            try:
                # Start processing this file
                progress_tracker.start_processing_file({worker_id}, file_path)
                
                logger.info(f"Processing {{Path(file_path).name}}")
                start_time = time.time()
                
                # Load and process file
                da = processor.load_tiff_optimized(Path(file_path))
                if da is not None:
                    gdf = processor.process_dataarray_to_h3(da, Path(file_path))
                    
                    if not gdf.empty:
                        # Complete processing with intermediate storage
                        progress_tracker.complete_processing_file(
                            {worker_id}, file_path, len(gdf), gdf
                        )
                        
                        processed_count += 1
                        processing_time = time.time() - start_time
                        
                        logger.info(f"Processed {{Path(file_path).name}}: {{len(gdf)}} hexagons in {{processing_time:.1f}}s")
                    else:
                        # No data produced
                        progress_tracker.fail_processing_file(
                            {worker_id}, file_path, "No hexagons produced"
                        )
                        logger.warning(f"No data from {{Path(file_path).name}}")
                else:
                    # Could not load file
                    progress_tracker.fail_processing_file(
                        {worker_id}, file_path, "Could not load TIFF file"
                    )
                    logger.warning(f"Could not load {{Path(file_path).name}}")
            
            except Exception as e:
                # Processing failed
                progress_tracker.fail_processing_file(
                    {worker_id}, file_path, str(e)
                )
                logger.error(f"Error processing {{Path(file_path).name}}: {{e}}")
        
        # Mark worker as completed
        progress_tracker.worker_progress[{worker_id}].status = "completed"
        progress_tracker.save_progress()
        
        logger.info(f"Worker {worker_id} completed: {{processed_count}} files processed")
    
    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {{e}}")
        import traceback
        traceback.print_exc()
        
        # Mark worker as error
        try:
            from progress_tracker import ProgressTracker
            progress_tracker = ProgressTracker('{self.config_path}', {self.num_workers})
            progress_tracker.worker_progress[{worker_id}].status = "error"
            progress_tracker.save_progress()
        except:
            pass

if __name__ == "__main__":
    main()
"""
        
        # Write worker script
        worker_script_path = self.logs_dir / f"worker_{worker_id}.py"
        with open(worker_script_path, 'w') as f:
            f.write(worker_script)
        
        return worker_script_path
    
    def start_worker(self, worker_id: int, files_to_process: List[str]) -> subprocess.Popen:
        """Start a worker process"""
        worker_script_path = self.create_worker_script(worker_id, files_to_process)
        
        # Start worker process
        cmd = [sys.executable, str(worker_script_path)]
        
        logger.info(f"Starting Worker {worker_id} with {len(files_to_process)} files")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path.cwd()
        )
        
        return process
    
    def monitor_workers(self):
        """Monitor worker processes"""
        logger.info("Monitoring worker processes...")
        
        while True:
            all_finished = True
            active_workers = 0
            
            for i, process in enumerate(self.worker_processes):
                if process.poll() is None:  # Still running
                    all_finished = False
                    active_workers += 1
                elif process.returncode != 0:  # Finished with error
                    logger.error(f"Worker {i} finished with error code {process.returncode}")
                
            logger.info(f"Active workers: {active_workers}/{len(self.worker_processes)}")
            
            if all_finished:
                break
            
            time.sleep(10)  # Check every 10 seconds
        
        logger.info("All workers finished")
    
    def monitor_workers_with_progress(self):
        """Monitor worker processes with progress tracking"""
        logger.info("Monitoring worker processes with progress tracking...")
        
        while True:
            all_finished = True
            active_workers = 0
            
            for i, process in enumerate(self.worker_processes):
                if process.poll() is None:  # Still running
                    all_finished = False
                    active_workers += 1
                elif process.returncode != 0:  # Finished with error
                    logger.error(f"Worker {i} finished with error code {process.returncode}")
            
            # Get progress summary
            progress_summary = self.progress_tracker.get_progress_summary()
            overall = progress_summary['overall']
            
            # Log progress update
            logger.info(f"Progress: {overall['completed_files']}/{overall['total_files']} files "
                       f"({overall['overall_completion_percentage']:.1f}%), "
                       f"{active_workers} workers active, "
                       f"{overall['total_hexagons']:,} hexagons, "
                       f"{overall['current_processing_rate']:.1f} files/min")
            
            if all_finished:
                break
            
            time.sleep(10)  # Check every 10 seconds
        
        logger.info("All workers finished")
    
    def save_final_results_from_intermediate(self, gdf: gpd.GeoDataFrame):
        """Save final results using intermediate data"""
        if gdf.empty:
            logger.warning("No intermediate data to save")
            return
        
        try:
            # Save as Parquet
            parquet_path = self.output_dir / "distributed_srai_final_with_progress.parquet"
            df_to_save = pd.DataFrame(gdf.drop(columns='geometry'))
            df_to_save.to_parquet(parquet_path, compression='snappy')
            
            # Save as GeoPackage
            gpkg_path = self.output_dir / "distributed_srai_final_with_progress.gpkg"
            gdf.to_file(gpkg_path, driver='GPKG')
            
            # Get final progress summary
            progress_summary = self.progress_tracker.get_progress_summary()
            
            # Create comprehensive processing report
            report = {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': time.time() - self.start_time,
                'processing_time_formatted': f"{(time.time() - self.start_time)/60:.1f} minutes",
                'num_workers': self.num_workers,
                'total_hexagons': len(gdf),
                'processor': 'distributed_srai_with_progress',
                'h3_resolution': 8,
                'year': 2021,
                'files': {
                    'parquet': parquet_path.name,
                    'geopackage': gpkg_path.name
                },
                'progress_tracking': {
                    'intermediate_storage': True,
                    'resumable': True,
                    'dashboard_enabled': self.enable_dashboard
                },
                'processing_summary': progress_summary
            }
            
            # Add band statistics if available
            band_cols = [col for col in gdf.columns if col.startswith('band_')]
            if band_cols:
                report['band_statistics'] = {}
                for col in band_cols[:5]:  # First 5 bands
                    report['band_statistics'][col] = {
                        'mean': float(gdf[col].mean()),
                        'std': float(gdf[col].std()),
                        'min': float(gdf[col].min()),
                        'max': float(gdf[col].max())
                    }
            
            report_path = self.output_dir / "distributed_processing_report_with_progress.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("="*60)
            logger.info("DISTRIBUTED SRAI PROCESSING COMPLETE!")
            logger.info("="*60)
            logger.info(f"Total time: {report['processing_time_formatted']}")
            logger.info(f"Workers: {self.num_workers}")
            logger.info(f"Hexagons: {len(gdf):,}")
            logger.info(f"Success rate: {progress_summary['statistics']['success_rate']:.1f}%")
            logger.info(f"Avg processing time: {progress_summary['statistics']['avg_processing_time']:.1f}s per file")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"Files: {parquet_path.name}, {gpkg_path.name}")
            if self.enable_dashboard:
                logger.info(f"Dashboard: http://localhost:8080")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
    
    def collect_results(self):
        """Collect results from all workers"""
        logger.info("Collecting results from workers...")
        
        all_records = []
        total_hexagons = 0
        
        for worker_id in range(self.num_workers):
            results_file = self.output_dir / f"worker_{worker_id}_results.json"
            
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        worker_results = json.load(f)
                    
                    all_records.extend(worker_results)
                    total_hexagons += len(worker_results)
                    
                    logger.info(f"Worker {worker_id}: {len(worker_results)} hexagons")
                
                except Exception as e:
                    logger.error(f"Error reading results from worker {worker_id}: {e}")
            else:
                logger.warning(f"No results file found for worker {worker_id}")
        
        if all_records:
            # Create combined DataFrame
            combined_df = pd.DataFrame(all_records)
            
            # Create geometries if needed
            if 'geometry' not in combined_df.columns and 'h3_index' in combined_df.columns:
                import h3
                from shapely.geometry import Polygon
                
                geometries = []
                for idx, row in combined_df.iterrows():
                    h3_idx = row.get('h3_index')
                    if h3_idx:
                        try:
                            boundary = h3.cell_to_boundary(h3_idx)
                            poly = Polygon([(lon, lat) for lat, lon in boundary])
                            geometries.append(poly)
                        except:
                            geometries.append(None)
                    else:
                        geometries.append(None)
                
                combined_df['geometry'] = geometries
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(combined_df, crs='EPSG:4326')
            
            # Remove duplicate H3 indices and aggregate
            if 'h3_index' in gdf.columns:
                logger.info("Aggregating duplicate H3 hexagons...")
                
                # Find band columns
                band_cols = [col for col in gdf.columns if col.startswith('band_')]
                
                if band_cols:
                    # Aggregate by H3 index
                    agg_dict = {col: 'mean' for col in band_cols}
                    agg_dict.update({
                        'lat': 'mean',
                        'lon': 'mean',
                        'pixel_count': 'sum' if 'pixel_count' in gdf.columns else 'mean'
                    })
                    
                    # Group and aggregate
                    grouped = gdf.groupby('h3_index').agg(agg_dict).reset_index()
                    
                    # Recreate geometries
                    geometries = []
                    for h3_idx in grouped['h3_index']:
                        try:
                            boundary = h3.cell_to_boundary(h3_idx)
                            poly = Polygon([(lon, lat) for lat, lon in boundary])
                            geometries.append(poly)
                        except:
                            geometries.append(None)
                    
                    gdf = gpd.GeoDataFrame(grouped, geometry=geometries, crs='EPSG:4326')
            
            # Add metadata
            gdf['year'] = 2021
            gdf['resolution'] = 8
            gdf['processor'] = 'distributed_srai_simple'
            
            logger.info(f"Combined results: {len(gdf):,} unique hexagons")
            return gdf
        
        else:
            logger.error("No results collected from any worker")
            return gpd.GeoDataFrame()
    
    def save_final_results(self, gdf: gpd.GeoDataFrame):
        """Save final combined results"""
        if gdf.empty:
            logger.warning("No data to save")
            return
        
        try:
            # Save as Parquet
            parquet_path = self.output_dir / "distributed_srai_final.parquet"
            df_to_save = pd.DataFrame(gdf.drop(columns='geometry'))
            df_to_save.to_parquet(parquet_path, compression='snappy')
            
            # Save as GeoPackage
            gpkg_path = self.output_dir / "distributed_srai_final.gpkg"
            gdf.to_file(gpkg_path, driver='GPKG')
            
            # Create processing report
            processing_time = time.time() - self.start_time
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': processing_time,
                'processing_time_formatted': f"{processing_time/60:.1f} minutes",
                'num_workers': self.num_workers,
                'total_hexagons': len(gdf),
                'processor': 'distributed_srai_simple',
                'h3_resolution': 8,
                'year': 2021,
                'files': {
                    'parquet': parquet_path.name,
                    'geopackage': gpkg_path.name
                }
            }
            
            # Add band statistics if available
            band_cols = [col for col in gdf.columns if col.startswith('band_')]
            if band_cols:
                report['band_statistics'] = {}
                for col in band_cols[:5]:  # First 5 bands
                    report['band_statistics'][col] = {
                        'mean': float(gdf[col].mean()),
                        'std': float(gdf[col].std()),
                        'min': float(gdf[col].min()),
                        'max': float(gdf[col].max())
                    }
            
            report_path = self.output_dir / "distributed_processing_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("="*60)
            logger.info("DISTRIBUTED SRAI PROCESSING COMPLETE!")
            logger.info("="*60)
            logger.info(f"Total time: {report['processing_time_formatted']}")
            logger.info(f"Workers: {self.num_workers}")
            logger.info(f"Hexagons: {len(gdf):,}")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"Files: {parquet_path.name}, {gpkg_path.name}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
    
    def start_dashboard(self):
        """Start progress dashboard in background thread"""
        if self.dashboard:
            def run_dashboard():
                self.dashboard.run()
            
            self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            self.dashboard_thread.start()
            logger.info("Progress dashboard started at http://localhost:8080")
    
    def run(self):
        """Run distributed processing with progress tracking"""
        try:
            logger.info("="*60)
            logger.info("DISTRIBUTED SRAI H3 PROCESSING WITH PROGRESS TRACKING")
            logger.info("="*60)
            logger.info(f"Workers: {self.num_workers}")
            logger.info(f"Source: {self.config['data']['source_dir']}")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"Progress tracking: Enabled")
            logger.info(f"Intermediate storage: Enabled")
            if self.enable_dashboard:
                logger.info(f"Dashboard: http://localhost:8080")
            logger.info("="*60)
            
            # Start dashboard
            if self.enable_dashboard:
                self.start_dashboard()
                time.sleep(2)  # Give dashboard time to start
            
            # Get files to process
            all_files = self.get_tiff_files()
            
            if not all_files:
                logger.error("No TIFF files found to process")
                return
            
            # Initialize progress tracking
            self.progress_tracker.initialize_tracking(all_files)
            
            # Get file assignments (excludes already completed files)
            worker_file_assignments = self.progress_tracker.assign_files_to_workers()
            
            # Log assignment summary
            pending_files = sum(len(files) for files in worker_file_assignments.values())
            completed_files = self.progress_tracker.overall_progress.completed_files
            
            logger.info(f"Files status: {completed_files} completed, {pending_files} pending")
            
            if pending_files == 0:
                logger.info("All files already completed! Using existing intermediate results.")
                final_gdf = self.progress_tracker.combine_intermediate_results()
                self.save_final_results_from_intermediate(final_gdf)
                return
            
            # Start worker processes only for workers with files
            for worker_id, files in worker_file_assignments.items():
                if files:  # Only start worker if it has files to process
                    process = self.start_worker(worker_id, files)
                    self.worker_processes.append(process)
                    logger.info(f"Started Worker {worker_id}: {len(files)} files")
            
            # Monitor workers with progress updates
            self.monitor_workers_with_progress()
            
            # Combine intermediate results
            final_gdf = self.progress_tracker.combine_intermediate_results()
            self.save_final_results_from_intermediate(final_gdf)
            
            logger.info("Distributed processing completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in distributed processing: {e}")
            import traceback
            traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed SRAI H3 Processing with Progress Tracking")
    parser.add_argument("--workers", type=int, default=6, help="Number of worker processes")
    parser.add_argument("--config", default="config.yaml", help="Configuration file")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable progress dashboard")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    
    args = parser.parse_args()
    
    # Create and run distributed processor
    processor = SimpleDistributedSRAI(
        config_path=args.config, 
        num_workers=args.workers,
        enable_dashboard=not args.no_dashboard
    )
    processor.run()


if __name__ == "__main__":
    main()