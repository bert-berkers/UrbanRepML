#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick progress checker for current SRAI processing
"""

import time
import json
from pathlib import Path
import pandas as pd

def check_current_progress():
    """Check progress of current workers"""
    
    print("="*60)
    print("SRAI PROCESSING PROGRESS CHECK")
    print("="*60)
    
    # Check worker logs
    logs_dir = Path("logs/workers")
    worker_stats = {}
    
    if logs_dir.exists():
        for worker_log in logs_dir.glob("worker_*.log"):
            worker_id = worker_log.stem.split('_')[1]
            
            try:
                with open(worker_log, 'r') as f:
                    lines = f.readlines()
                
                # Count processed files
                processed_count = sum(1 for line in lines if "Processed " in line and " hexagons" in line)
                
                # Get last activity
                last_line = lines[-1].strip() if lines else "No activity"
                
                # Extract current file being processed
                current_file = "Unknown"
                for line in reversed(lines):
                    if "Processing Cascadia_AlphaEarth" in line:
                        current_file = line.split("Processing ")[-1].strip()
                        break
                
                worker_stats[worker_id] = {
                    'processed': processed_count,
                    'current_file': current_file,
                    'last_activity': last_line
                }
                
            except Exception as e:
                worker_stats[worker_id] = {'error': str(e)}
    
    # Display worker progress
    total_processed = 0
    for worker_id in sorted(worker_stats.keys()):
        stats = worker_stats[worker_id]
        if 'processed' in stats:
            print(f"Worker {worker_id}: {stats['processed']} files completed")
            print(f"  Current: {stats['current_file']}")
            total_processed += stats['processed']
        else:
            print(f"Worker {worker_id}: Error - {stats.get('error', 'Unknown')}")
    
    print(f"\nTotal files processed across all workers: {total_processed}")
    
    # Check for any intermediate results
    intermediate_dir = Path("data/h3_2021_res8_distributed_simple")
    if intermediate_dir.exists():
        result_files = list(intermediate_dir.glob("worker_*_results.json"))
        total_hexagons = 0
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                total_hexagons += len(data)
            except:
                pass
        
        if result_files:
            print(f"Intermediate results: {len(result_files)} worker result files")
            print(f"Total hexagons so far: {total_hexagons:,}")
    
    # Estimate total files
    try:
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        source_dir = Path(config['data']['source_dir'])
        pattern = config['data']['pattern']
        total_files = len(list(source_dir.glob(pattern)))
        
        if total_processed > 0:
            completion_pct = (total_processed / total_files) * 100
            print(f"\nEstimated completion: {completion_pct:.1f}% ({total_processed}/{total_files} files)")
            
            # Estimate time remaining (very rough)
            if completion_pct > 1:  # At least 1% done
                # Assume we've been running for a while, estimate based on progress
                est_remaining_files = total_files - total_processed
                print(f"Estimated remaining: {est_remaining_files} files")
        
    except Exception as e:
        print(f"Could not estimate total: {e}")
    
    print("="*60)

if __name__ == "__main__":
    check_current_progress()