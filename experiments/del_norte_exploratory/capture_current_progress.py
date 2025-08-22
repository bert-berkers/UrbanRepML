#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Capture current progress from stuck workers before transitioning to enhanced system
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def capture_worker_progress():
    """Capture detailed progress from all worker logs"""
    
    progress_data = {
        'capture_time': datetime.now().isoformat(),
        'workers': {},
        'total_completed': 0,
        'files_completed': []
    }
    
    logs_dir = Path("logs/workers")
    
    print("="*60)
    print("CAPTURING CURRENT PROGRESS FROM STUCK WORKERS")
    print("="*60)
    
    for worker_id in range(6):
        worker_log = logs_dir / f"worker_{worker_id}.log"
        
        if worker_log.exists():
            print(f"\nAnalyzing Worker {worker_id}...")
            
            worker_data = {
                'worker_id': worker_id,
                'completed_files': [],
                'current_file': None,
                'hexagon_counts': {},
                'total_hexagons': 0
            }
            
            try:
                with open(worker_log, 'r') as f:
                    lines = f.readlines()
                
                # Extract completed files and their hexagon counts
                for line in lines:
                    if "Processed " in line and " hexagons" in line:
                        # Parse: "Processed filename.tif: 1043 hexagons"
                        parts = line.split("Processed ")[1]
                        filename = parts.split(":")[0].strip()
                        hexagon_count = int(parts.split(": ")[1].split(" hexagons")[0])
                        
                        worker_data['completed_files'].append(filename)
                        worker_data['hexagon_counts'][filename] = hexagon_count
                        worker_data['total_hexagons'] += hexagon_count
                        
                        print(f"  [OK] {filename}: {hexagon_count:,} hexagons")
                
                # Get current file being processed
                for line in reversed(lines):
                    if "Processing Cascadia_AlphaEarth" in line and "to H3" not in line:
                        worker_data['current_file'] = line.split("Processing ")[1].strip()
                        print(f"  [STUCK] Currently stuck on: {worker_data['current_file']}")
                        break
                
                print(f"  [TOTAL] Worker {worker_id} totals: {len(worker_data['completed_files'])} files, {worker_data['total_hexagons']:,} hexagons")
                
                progress_data['workers'][worker_id] = worker_data
                progress_data['total_completed'] += len(worker_data['completed_files'])
                progress_data['files_completed'].extend(worker_data['completed_files'])
                
            except Exception as e:
                print(f"  [ERROR] Error reading worker {worker_id} log: {e}")
                worker_data['error'] = str(e)
                progress_data['workers'][worker_id] = worker_data
    
    # Remove duplicates from completed files list
    progress_data['files_completed'] = list(set(progress_data['files_completed']))
    progress_data['unique_completed'] = len(progress_data['files_completed'])
    
    # Calculate totals
    total_hexagons = sum(
        worker_data.get('total_hexagons', 0) 
        for worker_data in progress_data['workers'].values()
    )
    
    print(f"\n[SUMMARY] OVERALL PROGRESS SUMMARY:")
    print(f"   Total files completed: {progress_data['unique_completed']}")
    print(f"   Total hexagons created: {total_hexagons:,}")
    print(f"   Files per worker: {progress_data['total_completed'] / 6:.1f} average")
    
    # Save progress data
    progress_file = Path("data/progress") 
    progress_file.mkdir(parents=True, exist_ok=True)
    
    capture_file = progress_file / "captured_progress_before_transition.json"
    with open(capture_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    print(f"\n[SAVED] Progress data saved to: {capture_file}")
    
    return progress_data

def check_intermediate_results():
    """Check if there are any intermediate result files"""
    
    print(f"\n[CHECK] CHECKING FOR INTERMEDIATE RESULTS...")
    
    results_dir = Path("data/h3_2021_res8_distributed_simple")
    if results_dir.exists():
        result_files = list(results_dir.glob("worker_*_results.json"))
        
        if result_files:
            total_hexagons = 0
            print(f"Found {len(result_files)} intermediate result files:")
            
            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    hexagons = len(data)
                    total_hexagons += hexagons
                    
                    mod_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                    print(f"  [FILE] {result_file.name}: {hexagons:,} hexagons (modified: {mod_time.strftime('%H:%M:%S')})")
                
                except Exception as e:
                    print(f"  [ERROR] {result_file.name}: Error reading - {e}")
            
            print(f"  [TOTAL] Total hexagons in intermediate files: {total_hexagons:,}")
            return result_files, total_hexagons
        else:
            print("  No intermediate JSON result files found")
            return [], 0
    else:
        print("  Results directory doesn't exist")
        return [], 0

def main():
    """Main capture function"""
    
    # Capture worker progress
    progress_data = capture_worker_progress()
    
    # Check intermediate results  
    result_files, intermediate_hexagons = check_intermediate_results()
    
    print("="*60)
    print("READY FOR TRANSITION TO ENHANCED SYSTEM")
    print("="*60)
    print(f"[OK] Completed files captured: {progress_data['unique_completed']}")
    print(f"[OK] Progress data preserved: captured_progress_before_transition.json")
    print(f"[OK] Ready to kill stuck processes and launch enhanced system")
    print("="*60)
    
    return progress_data

if __name__ == "__main__":
    main()