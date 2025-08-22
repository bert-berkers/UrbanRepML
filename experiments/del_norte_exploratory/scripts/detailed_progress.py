#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detailed progress checker with file timestamps and detailed analysis
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

def detailed_progress_check():
    """Detailed analysis of current processing state"""
    
    print("="*80)
    print("DETAILED SRAI PROCESSING ANALYSIS")
    print("="*80)
    
    # Check each worker's detailed status
    logs_dir = Path("logs/workers")
    
    for worker_id in range(6):
        print(f"\n--- WORKER {worker_id} ---")
        
        worker_log = logs_dir / f"worker_{worker_id}.log"
        worker_script = logs_dir / f"worker_{worker_id}.py"
        
        if worker_log.exists():
            # Get file modification time
            mod_time = datetime.fromtimestamp(worker_log.stat().st_mtime)
            time_since_mod = (datetime.now() - mod_time).total_seconds() / 60
            
            print(f"Log file: {worker_log.name}")
            print(f"Last modified: {mod_time.strftime('%H:%M:%S')} ({time_since_mod:.1f} min ago)")
            print(f"Log size: {worker_log.stat().st_size:,} bytes")
            
            # Read last few lines
            try:
                with open(worker_log, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    print("Last 3 log entries:")
                    for line in lines[-3:]:
                        print(f"  {line.strip()}")
                
                # Count completed files
                completed = sum(1 for line in lines if "Processed " in line and " hexagons" in line)
                processing = sum(1 for line in lines if "Processing Cascadia_AlphaEarth" in line and "to H3" not in line)
                
                print(f"Completed files: {completed}")
                print(f"Total processing attempts: {processing}")
                
                # Check if stuck on a file
                current_file = None
                for line in reversed(lines):
                    if "Processing Cascadia_AlphaEarth" in line and "to H3" not in line:
                        current_file = line.split("Processing ")[-1].strip()
                        break
                
                if current_file:
                    print(f"Current file: {current_file}")
                    
                    # Check how long it's been processing this file
                    for i, line in enumerate(reversed(lines)):
                        if f"Processing {current_file}" in line:
                            timestamp_str = line.split(" - ")[0]
                            try:
                                log_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                                time_on_file = (datetime.now() - log_time).total_seconds() / 60
                                print(f"Time on current file: {time_on_file:.1f} minutes")
                                
                                if time_on_file > 10:
                                    print("  ⚠️  POTENTIALLY STUCK - processing same file for >10 minutes")
                                elif time_on_file > 30:
                                    print("  ❌ LIKELY STUCK - processing same file for >30 minutes")
                                
                                break
                            except:
                                pass
                            break
                
            except Exception as e:
                print(f"Error reading log: {e}")
        else:
            print("❌ No log file found")
    
    # Check intermediate results
    print(f"\n--- INTERMEDIATE RESULTS ---")
    results_dir = Path("data/h3_2021_res8_distributed_simple")
    if results_dir.exists():
        result_files = list(results_dir.glob("worker_*_results.json"))
        if result_files:
            total_hexagons = 0
            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    hexagons = len(data)
                    total_hexagons += hexagons
                    
                    mod_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                    print(f"  {result_file.name}: {hexagons:,} hexagons (saved {mod_time.strftime('%H:%M:%S')})")
                
                except Exception as e:
                    print(f"  {result_file.name}: Error reading - {e}")
            
            print(f"  Total hexagons in intermediate files: {total_hexagons:,}")
        else:
            print("  No intermediate result files found yet")
    else:
        print("  Results directory doesn't exist yet")
    
    # System resource check
    print(f"\n--- SYSTEM RESOURCES ---")
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"CPU usage: {cpu_percent}%")
        print(f"Memory: {memory.percent}% used ({memory.used / 1024**3:.1f} GB / {memory.total / 1024**3:.1f} GB)")
        
        # Check if there are Python processes
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            try:
                if proc.info['name'].lower().startswith('python'):
                    python_processes.append(proc.info)
            except:
                pass
        
        print(f"Python processes: {len(python_processes)}")
        for proc in python_processes:
            mem_mb = proc['memory_info'].rss / 1024**2 if proc['memory_info'] else 0
            print(f"  PID {proc['pid']}: {mem_mb:.1f} MB")
            
    except ImportError:
        print("  psutil not available for detailed system info")
    except Exception as e:
        print(f"  Error getting system info: {e}")
    
    print("="*80)

if __name__ == "__main__":
    detailed_progress_check()