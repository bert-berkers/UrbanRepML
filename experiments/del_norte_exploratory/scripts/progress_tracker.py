#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Progress Tracking System for Distributed SRAI Processing

Features:
- Intermediate H3 result storage per TIFF file
- Resumable processing with checkpoint management
- Real-time progress monitoring
- Detailed statistics and performance metrics
- Web dashboard for live progress updates
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import geopandas as gpd
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import threading
import queue
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TiffProcessingRecord:
    """Record for a single TIFF file processing"""
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    worker_id: int
    status: str  # pending, processing, completed, failed, skipped
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    processing_time: Optional[float] = None
    hexagon_count: int = 0
    error_message: Optional[str] = None
    h3_result_file: Optional[str] = None
    retry_count: int = 0


@dataclass
class WorkerProgress:
    """Progress tracking for individual workers"""
    worker_id: int
    status: str  # starting, running, completed, error, paused
    assigned_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    current_file: Optional[str] = None
    total_hexagons: int = 0
    start_time: Optional[str] = None
    last_update: Optional[str] = None
    processing_rate: float = 0.0  # files per minute
    estimated_completion: Optional[str] = None


@dataclass
class OverallProgress:
    """Overall processing progress"""
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    processing_files: int = 0
    pending_files: int = 0
    total_hexagons: int = 0
    start_time: Optional[str] = None
    elapsed_time: float = 0.0
    estimated_remaining_time: Optional[float] = None
    overall_completion_percentage: float = 0.0
    current_processing_rate: float = 0.0  # files per minute
    peak_processing_rate: float = 0.0


class ProgressTracker:
    """Comprehensive progress tracking system"""
    
    def __init__(self, config_path: str = "config.yaml", num_workers: int = 6):
        self.config_path = config_path
        self.num_workers = num_workers
        
        # Initialize directories
        self.setup_directories()
        
        # Progress data
        self.tiff_records: Dict[str, TiffProcessingRecord] = {}
        self.worker_progress: Dict[int, WorkerProgress] = {}
        self.overall_progress = OverallProgress()
        
        # Files for persistence
        self.progress_file = self.progress_dir / "processing_progress.json"
        self.checkpoint_file = self.progress_dir / "checkpoint.json"
        self.completed_files_index = self.progress_dir / "completed_files.txt"
        
        # Load existing progress
        self.load_progress()
        
        # Statistics
        self.statistics = {
            'session_start': datetime.now().isoformat(),
            'total_sessions': 1,
            'performance_history': []
        }
        
        logger.info("Progress Tracker initialized")
    
    def setup_directories(self):
        """Setup directory structure for progress tracking"""
        # Progress tracking directory
        self.progress_dir = Path("data/progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Intermediate results directory (H3 files per TIFF)
        self.intermediate_dir = Path("data/h3_intermediate")
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint directory
        self.checkpoint_dir = Path("data/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Dashboard assets
        self.dashboard_dir = Path("dashboard")
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Progress directories created: {self.progress_dir}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Get SHA-256 hash of file for integrity checking"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()[:16]  # First 16 chars for brevity
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return "unknown"
    
    def initialize_tracking(self, all_tiff_files: List[str]):
        """Initialize tracking for all TIFF files"""
        logger.info(f"Initializing progress tracking for {len(all_tiff_files)} files")
        
        # Initialize overall progress
        self.overall_progress.total_files = len(all_tiff_files)
        self.overall_progress.start_time = datetime.now().isoformat()
        
        # Create records for all files
        for file_path in all_tiff_files:
            file_path = str(file_path)  # Ensure string
            file_name = Path(file_path).name
            
            if file_path not in self.tiff_records:
                # Check if intermediate result already exists
                h3_result_file = self.get_intermediate_h3_path(file_name)
                status = "completed" if h3_result_file.exists() else "pending"
                
                if status == "completed":
                    # Load hexagon count from existing file
                    hexagon_count = self.get_hexagon_count_from_file(h3_result_file)
                else:
                    hexagon_count = 0
                
                record = TiffProcessingRecord(
                    file_path=file_path,
                    file_name=file_name,
                    file_size=Path(file_path).stat().st_size if Path(file_path).exists() else 0,
                    file_hash=self.get_file_hash(file_path),
                    worker_id=-1,  # Unassigned
                    status=status,
                    hexagon_count=hexagon_count,
                    h3_result_file=str(h3_result_file) if status == "completed" else None
                )
                
                self.tiff_records[file_path] = record
        
        # Initialize worker progress
        for worker_id in range(self.num_workers):
            self.worker_progress[worker_id] = WorkerProgress(
                worker_id=worker_id,
                status="starting"
            )
        
        # Update counters
        self.update_progress_counters()
        self.save_progress()
        
        logger.info(f"Tracking initialized: {self.overall_progress.completed_files} already completed, "
                   f"{self.overall_progress.pending_files} pending")
    
    def get_intermediate_h3_path(self, tiff_filename: str) -> Path:
        """Get path for intermediate H3 result file"""
        # Convert TIFF filename to H3 result filename
        base_name = Path(tiff_filename).stem  # Remove .tif extension
        h3_filename = f"{base_name}_h3_res8.parquet"
        return self.intermediate_dir / h3_filename
    
    def get_hexagon_count_from_file(self, h3_file_path: Path) -> int:
        """Get hexagon count from existing H3 file"""
        try:
            if h3_file_path.exists():
                df = pd.read_parquet(h3_file_path)
                return len(df)
        except Exception as e:
            logger.warning(f"Could not read hexagon count from {h3_file_path}: {e}")
        return 0
    
    def assign_files_to_workers(self) -> Dict[int, List[str]]:
        """Assign pending files to workers, excluding already completed ones"""
        # Get only pending files
        pending_files = [
            file_path for file_path, record in self.tiff_records.items()
            if record.status == "pending"
        ]
        
        logger.info(f"Assigning {len(pending_files)} pending files to {self.num_workers} workers")
        
        # Split files among workers
        worker_assignments = {}
        files_per_worker = len(pending_files) // self.num_workers
        remainder = len(pending_files) % self.num_workers
        
        start_idx = 0
        for worker_id in range(self.num_workers):
            end_idx = start_idx + files_per_worker + (1 if worker_id < remainder else 0)
            assigned_files = pending_files[start_idx:end_idx]
            
            worker_assignments[worker_id] = assigned_files
            
            # Update worker progress
            self.worker_progress[worker_id].assigned_files = len(assigned_files)
            self.worker_progress[worker_id].status = "running" if assigned_files else "completed"
            
            # Update TIFF records with worker assignment
            for file_path in assigned_files:
                self.tiff_records[file_path].worker_id = worker_id
                self.tiff_records[file_path].status = "assigned"
            
            start_idx = end_idx
            
            logger.info(f"Worker {worker_id}: {len(assigned_files)} files assigned")
        
        self.save_progress()
        return worker_assignments
    
    def start_processing_file(self, worker_id: int, file_path: str):
        """Mark file as being processed"""
        if file_path in self.tiff_records:
            record = self.tiff_records[file_path]
            record.status = "processing"
            record.start_time = datetime.now().isoformat()
            record.worker_id = worker_id
            
            # Update worker progress
            worker = self.worker_progress[worker_id]
            worker.current_file = Path(file_path).name
            worker.last_update = datetime.now().isoformat()
            
            self.update_progress_counters()
            
            logger.info(f"Worker {worker_id} started processing: {Path(file_path).name}")
    
    def complete_processing_file(self, worker_id: int, file_path: str, 
                               hexagon_count: int, h3_data: pd.DataFrame = None):
        """Mark file as completed and store intermediate results"""
        if file_path not in self.tiff_records:
            logger.warning(f"File {file_path} not found in tracking records")
            return
        
        record = self.tiff_records[file_path]
        record.status = "completed"
        record.end_time = datetime.now().isoformat()
        record.hexagon_count = hexagon_count
        
        # Calculate processing time
        if record.start_time:
            start = datetime.fromisoformat(record.start_time)
            end = datetime.fromisoformat(record.end_time)
            record.processing_time = (end - start).total_seconds()
        
        # Save intermediate H3 result
        h3_result_path = self.get_intermediate_h3_path(record.file_name)
        if h3_data is not None and not h3_data.empty:
            try:
                # Add metadata
                h3_data_copy = h3_data.copy()
                h3_data_copy['source_file'] = record.file_name
                h3_data_copy['processing_time'] = record.processing_time
                h3_data_copy['worker_id'] = worker_id
                h3_data_copy['completed_at'] = record.end_time
                
                h3_data_copy.to_parquet(h3_result_path, compression='snappy')
                record.h3_result_file = str(h3_result_path)
                
                logger.info(f"Saved intermediate result: {h3_result_path.name} ({len(h3_data)} hexagons)")
                
            except Exception as e:
                logger.error(f"Failed to save intermediate result for {record.file_name}: {e}")
        
        # Update worker progress
        worker = self.worker_progress[worker_id]
        worker.completed_files += 1
        worker.total_hexagons += hexagon_count
        worker.current_file = None
        worker.last_update = datetime.now().isoformat()
        
        # Calculate worker processing rate
        if worker.start_time:
            start = datetime.fromisoformat(worker.start_time)
            elapsed_minutes = (datetime.now() - start).total_seconds() / 60
            if elapsed_minutes > 0:
                worker.processing_rate = worker.completed_files / elapsed_minutes
        
        # Update overall progress
        self.update_progress_counters()
        self.save_progress()
        
        logger.info(f"Worker {worker_id} completed: {record.file_name} ({hexagon_count} hexagons, "
                   f"{record.processing_time:.1f}s)")
    
    def fail_processing_file(self, worker_id: int, file_path: str, error_message: str):
        """Mark file as failed"""
        if file_path in self.tiff_records:
            record = self.tiff_records[file_path]
            record.status = "failed"
            record.end_time = datetime.now().isoformat()
            record.error_message = error_message
            record.retry_count += 1
            
            # Update worker progress
            worker = self.worker_progress[worker_id]
            worker.failed_files += 1
            worker.current_file = None
            worker.last_update = datetime.now().isoformat()
            
            self.update_progress_counters()
            self.save_progress()
            
            logger.warning(f"Worker {worker_id} failed: {Path(file_path).name} - {error_message}")
    
    def update_progress_counters(self):
        """Update overall progress counters"""
        completed = sum(1 for r in self.tiff_records.values() if r.status == "completed")
        failed = sum(1 for r in self.tiff_records.values() if r.status == "failed")
        processing = sum(1 for r in self.tiff_records.values() if r.status == "processing")
        pending = sum(1 for r in self.tiff_records.values() if r.status in ["pending", "assigned"])
        skipped = sum(1 for r in self.tiff_records.values() if r.status == "skipped")
        
        self.overall_progress.completed_files = completed
        self.overall_progress.failed_files = failed
        self.overall_progress.processing_files = processing
        self.overall_progress.pending_files = pending
        self.overall_progress.skipped_files = skipped
        
        # Calculate completion percentage
        total_processed = completed + failed + skipped
        if self.overall_progress.total_files > 0:
            self.overall_progress.overall_completion_percentage = (
                total_processed / self.overall_progress.total_files * 100
            )
        
        # Calculate total hexagons
        self.overall_progress.total_hexagons = sum(
            r.hexagon_count for r in self.tiff_records.values() if r.hexagon_count > 0
        )
        
        # Calculate elapsed time and processing rate
        if self.overall_progress.start_time:
            start = datetime.fromisoformat(self.overall_progress.start_time)
            self.overall_progress.elapsed_time = (datetime.now() - start).total_seconds()
            
            elapsed_minutes = self.overall_progress.elapsed_time / 60
            if elapsed_minutes > 0 and completed > 0:
                self.overall_progress.current_processing_rate = completed / elapsed_minutes
                
                # Update peak rate
                if self.overall_progress.current_processing_rate > self.overall_progress.peak_processing_rate:
                    self.overall_progress.peak_processing_rate = self.overall_progress.current_processing_rate
                
                # Estimate remaining time
                remaining_files = pending + processing
                if self.overall_progress.current_processing_rate > 0:
                    remaining_minutes = remaining_files / self.overall_progress.current_processing_rate
                    self.overall_progress.estimated_remaining_time = remaining_minutes * 60
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary"""
        self.update_progress_counters()
        
        # Worker summaries
        worker_summaries = {}
        for worker_id, worker in self.worker_progress.items():
            completion_pct = 0.0
            if worker.assigned_files > 0:
                completion_pct = (worker.completed_files / worker.assigned_files) * 100
            
            worker_summaries[worker_id] = {
                'status': worker.status,
                'assigned_files': worker.assigned_files,
                'completed_files': worker.completed_files,
                'failed_files': worker.failed_files,
                'current_file': worker.current_file,
                'completion_percentage': completion_pct,
                'processing_rate': worker.processing_rate,
                'total_hexagons': worker.total_hexagons
            }
        
        # Recent completions (last 10)
        recent_completions = []
        completed_records = [
            r for r in self.tiff_records.values() 
            if r.status == "completed" and r.end_time
        ]
        completed_records.sort(key=lambda x: x.end_time, reverse=True)
        
        for record in completed_records[:10]:
            recent_completions.append({
                'file_name': record.file_name,
                'worker_id': record.worker_id,
                'hexagon_count': record.hexagon_count,
                'processing_time': record.processing_time,
                'completed_at': record.end_time
            })
        
        return {
            'overall': asdict(self.overall_progress),
            'workers': worker_summaries,
            'recent_completions': recent_completions,
            'statistics': {
                'avg_hexagons_per_file': (
                    self.overall_progress.total_hexagons / max(1, self.overall_progress.completed_files)
                ),
                'avg_processing_time': self.get_average_processing_time(),
                'success_rate': self.get_success_rate(),
                'estimated_completion': self.get_estimated_completion_time()
            }
        }
    
    def get_average_processing_time(self) -> float:
        """Get average processing time per file"""
        processing_times = [
            r.processing_time for r in self.tiff_records.values()
            if r.processing_time and r.processing_time > 0
        ]
        return sum(processing_times) / len(processing_times) if processing_times else 0.0
    
    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        total_attempted = self.overall_progress.completed_files + self.overall_progress.failed_files
        if total_attempted > 0:
            return (self.overall_progress.completed_files / total_attempted) * 100
        return 0.0
    
    def get_estimated_completion_time(self) -> Optional[str]:
        """Get estimated completion time"""
        if self.overall_progress.estimated_remaining_time:
            completion_time = datetime.now() + timedelta(seconds=self.overall_progress.estimated_remaining_time)
            return completion_time.isoformat()
        return None
    
    def save_progress(self):
        """Save progress to disk"""
        try:
            progress_data = {
                'overall_progress': asdict(self.overall_progress),
                'tiff_records': {k: asdict(v) for k, v in self.tiff_records.items()},
                'worker_progress': {k: asdict(v) for k, v in self.worker_progress.items()},
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
            # Also save completed files list for quick reference
            completed_files = [
                r.file_path for r in self.tiff_records.values() 
                if r.status == "completed"
            ]
            with open(self.completed_files_index, 'w') as f:
                f.write('\n'.join(completed_files))
                
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def load_progress(self):
        """Load progress from disk"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                # Load overall progress
                if 'overall_progress' in progress_data:
                    overall_dict = progress_data['overall_progress']
                    self.overall_progress = OverallProgress(**overall_dict)
                
                # Load TIFF records
                if 'tiff_records' in progress_data:
                    for file_path, record_dict in progress_data['tiff_records'].items():
                        self.tiff_records[file_path] = TiffProcessingRecord(**record_dict)
                
                # Load worker progress
                if 'worker_progress' in progress_data:
                    for worker_id_str, worker_dict in progress_data['worker_progress'].items():
                        worker_id = int(worker_id_str)
                        self.worker_progress[worker_id] = WorkerProgress(**worker_dict)
                
                logger.info(f"Loaded progress: {len(self.tiff_records)} files tracked")
                
            except Exception as e:
                logger.error(f"Failed to load progress: {e}")
    
    def combine_intermediate_results(self) -> gpd.GeoDataFrame:
        """Combine all intermediate H3 results into final dataset"""
        logger.info("Combining intermediate results...")
        
        all_dataframes = []
        combined_hexagons = 0
        
        # Find all intermediate result files
        h3_files = list(self.intermediate_dir.glob("*_h3_res8.parquet"))
        logger.info(f"Found {len(h3_files)} intermediate H3 files")
        
        for h3_file in h3_files:
            try:
                df = pd.read_parquet(h3_file)
                all_dataframes.append(df)
                combined_hexagons += len(df)
                
            except Exception as e:
                logger.error(f"Error reading {h3_file}: {e}")
        
        if all_dataframes:
            # Combine all dataframes
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Remove duplicates by H3 index and aggregate
            if 'h3_index' in combined_df.columns:
                band_cols = [col for col in combined_df.columns if col.startswith('band_')]
                
                if band_cols:
                    agg_dict = {col: 'mean' for col in band_cols}
                    agg_dict.update({
                        'lat': 'mean',
                        'lon': 'mean',
                        'pixel_count': 'sum' if 'pixel_count' in combined_df.columns else 'mean',
                        'processing_time': 'mean',
                        'worker_id': 'first'
                    })
                    
                    combined_df = combined_df.groupby('h3_index').agg(agg_dict).reset_index()
            
            # Create geometries
            import h3
            from shapely.geometry import Polygon
            
            geometries = []
            for h3_idx in combined_df['h3_index']:
                try:
                    boundary = h3.cell_to_boundary(h3_idx)
                    poly = Polygon([(lon, lat) for lat, lon in boundary])
                    geometries.append(poly)
                except:
                    geometries.append(None)
            
            gdf = gpd.GeoDataFrame(combined_df, geometry=geometries, crs='EPSG:4326')
            
            logger.info(f"Combined {len(gdf):,} unique hexagons from {len(h3_files)} intermediate files")
            return gdf
        
        else:
            logger.warning("No intermediate results found to combine")
            return gpd.GeoDataFrame()
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files after successful combination"""
        h3_files = list(self.intermediate_dir.glob("*_h3_res8.parquet"))
        
        for h3_file in h3_files:
            try:
                h3_file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete {h3_file}: {e}")
        
        logger.info(f"Cleaned up {len(h3_files)} intermediate files")


# Progress Dashboard Server
class ProgressDashboard:
    """Real-time web dashboard for progress monitoring"""
    
    def __init__(self, progress_tracker: ProgressTracker, port: int = 8080):
        self.progress_tracker = progress_tracker
        self.port = port
        self.app = FastAPI(title="SRAI Processing Dashboard")
        self.active_connections: List[WebSocket] = []
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup dashboard routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self.get_dashboard_html()
        
        @self.app.get("/api/progress")
        async def get_progress():
            return JSONResponse(self.progress_tracker.get_progress_summary())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    # Send progress updates every 5 seconds
                    progress = self.progress_tracker.get_progress_summary()
                    await websocket.send_json(progress)
                    await asyncio.sleep(5)
            except:
                pass
            finally:
                self.active_connections.remove(websocket)
    
    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>SRAI Processing Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: #2a2a2a; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 14px; color: #aaa; margin-top: 5px; }
        .progress-bar { background: #333; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-fill { background: linear-gradient(90deg, #4CAF50, #45a049); height: 100%; transition: width 0.5s; }
        .workers-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .worker-card { background: #2a2a2a; padding: 15px; border-radius: 8px; }
        .worker-status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .status-running { background: #4CAF50; }
        .status-completed { background: #2196F3; }
        .status-error { background: #f44336; }
        .recent-completions { background: #2a2a2a; padding: 20px; border-radius: 8px; margin-top: 20px; }
        .completion-item { padding: 10px; border-bottom: 1px solid #444; }
        #status { margin: 10px 0; padding: 10px; background: #333; border-radius: 4px; }
        .connected { color: #4CAF50; }
        .disconnected { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 SRAI H3 Processing Dashboard</h1>
            <p>Real-time progress monitoring with intermediate result storage</p>
            <div id="status" class="disconnected">Connecting...</div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="completion-pct">0%</div>
                <div class="stat-label">Overall Completion</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-value" id="completed-files">0</div>
                <div class="stat-label">Files Completed</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-value" id="total-hexagons">0</div>
                <div class="stat-label">Total Hexagons</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-value" id="processing-rate">0.0</div>
                <div class="stat-label">Files/Min</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-value" id="eta">--:--</div>
                <div class="stat-label">Estimated Completion</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-value" id="success-rate">0%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
        
        <h2>Worker Status</h2>
        <div class="workers-grid" id="workers-grid">
            <!-- Workers will be populated here -->
        </div>
        
        <div class="recent-completions">
            <h2>Recent Completions</h2>
            <div id="recent-completions-list">
                <!-- Recent completions will be populated here -->
            </div>
        </div>
    </div>
    
    <script>
        const ws = new WebSocket('ws://localhost:8080/ws');
        const statusEl = document.getElementById('status');
        
        ws.onopen = function() {
            statusEl.textContent = 'Connected - Live Updates';
            statusEl.className = 'connected';
        };
        
        ws.onclose = function() {
            statusEl.textContent = 'Disconnected - Attempting to reconnect...';
            statusEl.className = 'disconnected';
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        function updateDashboard(data) {
            const overall = data.overall;
            const workers = data.workers;
            const recent = data.recent_completions;
            const stats = data.statistics;
            
            // Update overall stats
            document.getElementById('completion-pct').textContent = overall.overall_completion_percentage.toFixed(1) + '%';
            document.getElementById('progress-fill').style.width = overall.overall_completion_percentage + '%';
            document.getElementById('completed-files').textContent = overall.completed_files + '/' + overall.total_files;
            document.getElementById('total-hexagons').textContent = overall.total_hexagons.toLocaleString();
            document.getElementById('processing-rate').textContent = overall.current_processing_rate.toFixed(1);
            document.getElementById('success-rate').textContent = stats.success_rate.toFixed(1) + '%';
            
            // Update ETA
            if (stats.estimated_completion) {
                const eta = new Date(stats.estimated_completion);
                document.getElementById('eta').textContent = eta.toLocaleTimeString();
            }
            
            // Update workers
            const workersGrid = document.getElementById('workers-grid');
            workersGrid.innerHTML = '';
            
            for (const [workerId, worker] of Object.entries(workers)) {
                const workerCard = document.createElement('div');
                workerCard.className = 'worker-card';
                
                const statusClass = worker.status === 'running' ? 'status-running' : 
                                  worker.status === 'completed' ? 'status-completed' : 'status-error';
                
                workerCard.innerHTML = `
                    <h3>Worker ${workerId} <span class="worker-status ${statusClass}">${worker.status}</span></h3>
                    <p><strong>Progress:</strong> ${worker.completed_files}/${worker.assigned_files} 
                       (${worker.completion_percentage.toFixed(1)}%)</p>
                    <p><strong>Current:</strong> ${worker.current_file || 'None'}</p>
                    <p><strong>Rate:</strong> ${worker.processing_rate.toFixed(1)} files/min</p>
                    <p><strong>Hexagons:</strong> ${worker.total_hexagons.toLocaleString()}</p>
                    <p><strong>Failed:</strong> ${worker.failed_files}</p>
                `;
                
                workersGrid.appendChild(workerCard);
            }
            
            // Update recent completions
            const recentList = document.getElementById('recent-completions-list');
            recentList.innerHTML = '';
            
            recent.forEach(completion => {
                const item = document.createElement('div');
                item.className = 'completion-item';
                const completedTime = new Date(completion.completed_at).toLocaleTimeString();
                
                item.innerHTML = `
                    <strong>${completion.file_name}</strong> - 
                    Worker ${completion.worker_id} - 
                    ${completion.hexagon_count} hexagons - 
                    ${completion.processing_time.toFixed(1)}s - 
                    ${completedTime}
                `;
                
                recentList.appendChild(item);
            });
        }
        
        // Auto-refresh if WebSocket fails
        setInterval(() => {
            if (ws.readyState === WebSocket.CLOSED) {
                location.reload();
            }
        }, 30000);
    </script>
</body>
</html>
        """
    
    def run(self):
        """Run dashboard server"""
        logger.info(f"Starting progress dashboard on http://localhost:{self.port}")
        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="warning")


def main():
    """Test progress tracker"""
    tracker = ProgressTracker()
    
    # Simulate some progress
    test_files = [f"test_file_{i}.tif" for i in range(10)]
    tracker.initialize_tracking(test_files)
    
    print("Progress Summary:")
    print(json.dumps(tracker.get_progress_summary(), indent=2))


if __name__ == "__main__":
    main()