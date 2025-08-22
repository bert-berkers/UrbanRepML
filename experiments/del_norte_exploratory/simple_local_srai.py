#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Local SRAI Processing - No fancy scheduling, just efficient local processing
Claude is the scheduler using its own intelligence
"""

import json
import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import yaml

class SimpleLocalSRAI:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize simple local SRAI processor"""
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.source_dir = Path(self.config['data']['source_dir'])
        self.output_dir = Path("data/h3_2021_res8_simple")
        self.progress_dir = Path("data/progress")
        self.logs_dir = Path("logs/simple_workers")
        
        # Create directories
        for dir_path in [self.output_dir, self.progress_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Simple parameters
        self.max_workers = 4
        self.files_per_worker = 50  # Process in smaller batches
        
        # Progress tracking
        self.completed_files: Set[str] = set()
        self.worker_processes: Dict[int, subprocess.Popen] = {}
        self.start_time = time.time()
        
        # Load previous progress
        self.load_previous_progress()
        
        print(f"[SIMPLE] Simple local SRAI processor initialized")
        print(f"   Workers: {self.max_workers}")
        print(f"   Files per batch: {self.files_per_worker}")
        print(f"   Previously completed: {len(self.completed_files)} files")
    
    def load_previous_progress(self):
        """Load previously completed files"""
        
        # Load from capture data
        capture_file = self.progress_dir / "captured_progress_before_transition.json"
        if capture_file.exists():
            with open(capture_file, 'r') as f:
                data = json.load(f)
            for filename in data.get('files_completed', []):
                self.completed_files.add(filename)
        
        # Check for intermediate results
        for result_file in self.output_dir.glob("*_h3_res8.json"):
            filename = result_file.name.replace("_h3_res8.json", ".tif")
            self.completed_files.add(filename)
        
        print(f"[RESUME] Total files to skip: {len(self.completed_files)}")
    
    def get_remaining_files(self) -> List[str]:
        """Get remaining files to process"""
        
        pattern = self.config['data']['pattern']
        all_files = list(self.source_dir.glob(pattern))
        
        remaining = []
        for file_path in all_files:
            filename = file_path.name
            if filename not in self.completed_files:
                remaining.append(str(file_path))
        
        print(f"[FILES] Total: {len(all_files)}, Remaining: {len(remaining)}")
        return remaining
    
    def create_worker_script(self, worker_id: int, files_batch: List[str]) -> str:
        """Create simple worker script"""
        
        worker_script = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple Worker {worker_id} - Clean SRAI Processing"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

# Worker config
WORKER_ID = {worker_id}
OUTPUT_DIR = Path("{self.output_dir}")
LOGS_DIR = Path("{self.logs_dir}")

# Logging
log_file = LOGS_DIR / f"simple_worker_{{WORKER_ID}}.log"
def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{{timestamp}} [W{{WORKER_ID}}] {{message}}\\n")
    print(f"[W{{WORKER_ID}}] {{message}}")

log("Simple worker starting")

# Import SRAI
try:
    import geopandas as gpd
    import rioxarray as rxr
    from srai.regionalizers import H3Regionalizer
    from shapely.geometry import box
    log("SRAI components imported successfully")
except Exception as e:
    log(f"ERROR importing SRAI: {{e}}")
    sys.exit(1)

def process_file(file_path: str) -> bool:
    """Process single TIFF file"""
    
    filename = Path(file_path).name
    output_file = OUTPUT_DIR / f"{{filename.replace('.tif', '_h3_res8.json')}}"
    
    if output_file.exists():
        log(f"SKIP {{filename}} - already exists")
        return True
    
    start_time = time.time()
    
    try:
        # Open raster
        da = rxr.open_rasterio(file_path)
        
        if da is None or da.sizes.get('x', 0) == 0:
            log(f"SKIP {{filename}} - invalid raster")
            return True
        
        # Get bounds and create geometry
        minx, miny, maxx, maxy = da.rio.bounds()
        geometry = [box(minx, miny, maxx, maxy)]
        gdf = gpd.GeoDataFrame({{'id': [0]}}, geometry=geometry, crs=da.rio.crs)
        
        # H3 regionalization
        regionalizer = H3Regionalizer(resolution=8)
        regions_gdf = regionalizer.transform(gdf)
        
        if regions_gdf is None or len(regions_gdf) == 0:
            log(f"SKIP {{filename}} - no regions generated")
            return True
        
        # Save results
        results = {{}}
        for idx, row in regions_gdf.iterrows():
            results[str(idx)] = {{
                'geometry': str(row['geometry']),
                'centroid': [row['geometry'].centroid.x, row['geometry'].centroid.y],
                'processed_at': datetime.now().isoformat()
            }}
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        processing_time = time.time() - start_time
        log(f"COMPLETED {{filename}}: {{len(regions_gdf)}} hexagons in {{processing_time:.1f}}s")
        
        # Cleanup
        del da, gdf, regions_gdf
        gc.collect()
        
        return True
        
    except Exception as e:
        processing_time = time.time() - start_time
        log(f"ERROR {{filename}}: {{str(e)}} ({{processing_time:.1f}}s)")
        return False

# Main processing
def main():
    files_to_process = {json.dumps(files_batch)}
    
    log(f"Processing {{len(files_to_process)}} files")
    
    completed = 0
    errors = 0
    
    for file_path in files_to_process:
        success = process_file(file_path)
        if success:
            completed += 1
        else:
            errors += 1
        
        if (completed + errors) % 10 == 0:
            log(f"Progress: {{completed}} completed, {{errors}} errors")
    
    log(f"Worker finished: {{completed}} completed, {{errors}} errors")

if __name__ == "__main__":
    main()
'''
        
        # Write worker script
        worker_file = self.logs_dir / f"simple_worker_{worker_id}.py"
        with open(worker_file, 'w', encoding='utf-8') as f:
            f.write(worker_script)
        
        return str(worker_file)
    
    def start_worker(self, worker_id: int, files_batch: List[str]) -> bool:
        """Start simple worker"""
        
        try:
            script_path = self.create_worker_script(worker_id, files_batch)
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=str(Path.cwd()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self.worker_processes[worker_id] = process
            print(f"[WORKER] Started Worker {worker_id} (PID: {process.pid}, {len(files_batch)} files)")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start worker {worker_id}: {e}")
            return False
    
    def monitor_progress(self):
        """Simple progress monitoring"""
        
        while self.worker_processes:
            time.sleep(15)  # Check every 15 seconds
            
            # Check worker status
            finished_workers = []
            for worker_id, process in self.worker_processes.items():
                if process.poll() is not None:
                    return_code = process.returncode
                    if return_code == 0:
                        print(f"[COMPLETE] Worker {worker_id} finished successfully")
                    else:
                        print(f"[ERROR] Worker {worker_id} failed with code {return_code}")
                    finished_workers.append(worker_id)
            
            # Remove finished workers
            for worker_id in finished_workers:
                del self.worker_processes[worker_id]
            
            # Progress update
            completed_count = len(list(self.output_dir.glob("*_h3_res8.json")))
            runtime = (time.time() - self.start_time) / 60
            print(f"[PROGRESS] Runtime: {runtime:.1f}min, Completed: {completed_count} files, Active workers: {len(self.worker_processes)}")
        
        print("[MONITOR] All workers completed")
    
    def run_simple_processing(self):
        """Run simple local processing"""
        
        print("\\n" + "="*60)
        print("STARTING SIMPLE LOCAL SRAI PROCESSING")
        print("="*60)
        
        remaining_files = self.get_remaining_files()
        
        if not remaining_files:
            print("[COMPLETE] All files already processed!")
            return
        
        # Process files in batches
        batch_start = 0
        while batch_start < len(remaining_files):
            
            # Create worker batches
            workers_started = 0
            for worker_id in range(self.max_workers):
                batch_end = batch_start + self.files_per_worker
                if batch_start < len(remaining_files):
                    batch = remaining_files[batch_start:batch_end]
                    if batch:
                        if self.start_worker(worker_id, batch):
                            workers_started += 1
                        batch_start = batch_end
                    else:
                        break
            
            if workers_started == 0:
                print("[ERROR] No workers started")
                break
            
            # Monitor this batch
            self.monitor_progress()
            
            # Brief pause between batches
            time.sleep(5)
        
        # Final summary
        final_count = len(list(self.output_dir.glob("*_h3_res8.json")))
        total_runtime = (time.time() - self.start_time) / 60
        
        print("\\n" + "="*60)
        print("SIMPLE PROCESSING COMPLETE")
        print("="*60)
        print(f"Total runtime: {total_runtime:.1f} minutes")
        print(f"Total files completed: {final_count}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)

def main():
    try:
        processor = SimpleLocalSRAI()
        processor.run_simple_processing()
        
    except KeyboardInterrupt:
        print("\\n[INTERRUPT] Processing interrupted")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()