#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metacognitive Attention Monitor

Real-time monitoring system that watches the hierarchical experiment like a
second-order attention mechanism - monitoring the monitor, tracking progress,
resource usage, and providing metacognitive insights.
"""

import time
import psutil
import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tcl/Tk issues
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import subprocess
import os
import sys

# Setup logging for the monitor
monitor_log_dir = Path("experiments/del_norte_active_inference/logs")
monitor_log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - METACOGNITIVE - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(monitor_log_dir / 'metacognitive_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MetacognitiveMonitor:
    """
    Metacognitive attention system that monitors the hierarchical experiment.
    Provides real-time insights, resource tracking, and progress analysis.
    """
    
    def __init__(self, experiment_dir: str = "experiments/del_norte_active_inference"):
        self.experiment_dir = Path(experiment_dir)
        self.monitoring_data = []
        self.start_time = datetime.now()
        
        # Files to monitor
        self.log_files = [
            self.experiment_dir / "logs" / "hierarchical_experiment.log",
            self.experiment_dir / "logs" / "experiment.log"
        ]
        
        self.results_dir = self.experiment_dir / "hierarchical_results"
        self.viz_dir = self.experiment_dir / "hierarchical_visualizations"
        
        # Tracking state
        self.last_log_positions = {str(f): 0 for f in self.log_files}
        self.phases_detected = set()
        self.progress_markers = []
        
        logger.info("🧠 METACOGNITIVE MONITOR ACTIVATED!")
        logger.info(f"   Watching experiment in: {self.experiment_dir}")
        logger.info("   Metacognitive attention layer engaged...")
    
    def get_system_resources(self) -> Dict:
        """Get current system resource usage - RTX 3090 focus!"""
        
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU monitoring (RTX 3090 specific)
            gpu_info = {}
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_info = {
                    'memory_used_gb': gpu_memory.used / 1024**3,
                    'memory_total_gb': gpu_memory.total / 1024**3,
                    'memory_percent': (gpu_memory.used / gpu_memory.total) * 100,
                    'utilization_percent': gpu_util.gpu,
                    'temperature_c': gpu_temp
                }
                
            except Exception:
                # Fallback if pynvml not available
                gpu_info = {
                    'memory_used_gb': 0,
                    'memory_total_gb': 24.0,  # RTX 3090
                    'memory_percent': 0,
                    'utilization_percent': 0,
                    'temperature_c': 0
                }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_used_gb': memory.used / 1024**3,
                'memory_total_gb': memory.total / 1024**3,
                'memory_percent': memory.percent,
                'gpu': gpu_info
            }
            
        except Exception as e:
            logger.warning(f"Resource monitoring error: {e}")
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}
    
    def analyze_log_progress(self) -> Dict:
        """Analyze log files for experiment progress."""
        
        progress_info = {
            'phases_completed': len(self.phases_detected),
            'current_phase': 'Unknown',
            'hexagon_counts': {},
            'latest_activity': 'No activity detected',
            'error_count': 0
        }
        
        for log_file in self.log_files:
            if not log_file.exists():
                continue
            
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read new lines since last check
                    f.seek(self.last_log_positions[str(log_file)])
                    new_lines = f.readlines()
                    self.last_log_positions[str(log_file)] = f.tell()
                
                for line in new_lines:
                    line = line.strip()
                    
                    # Detect phases
                    if "PHASE 1:" in line:
                        self.phases_detected.add("Phase 1: Spatial Embedding")
                        progress_info['current_phase'] = "Phase 1: Spatial Embedding"
                    elif "PHASE 2:" in line:
                        self.phases_detected.add("Phase 2: Neural Learning")
                        progress_info['current_phase'] = "Phase 2: Neural Learning"
                    elif "PHASE 3:" in line:
                        self.phases_detected.add("Phase 3: Active Inference")
                        progress_info['current_phase'] = "Phase 3: Active Inference"
                    elif "PHASE 4:" in line:
                        self.phases_detected.add("Phase 4: Cluster Analysis")
                        progress_info['current_phase'] = "Phase 4: Cluster Analysis"
                    elif "PHASE 5:" in line:
                        self.phases_detected.add("Phase 5: Comprehensive Analysis")
                        progress_info['current_phase'] = "Phase 5: Comprehensive Analysis"
                    elif "PHASE 6:" in line:
                        self.phases_detected.add("Phase 6: Holographic Visualization")
                        progress_info['current_phase'] = "Phase 6: Holographic Visualization"
                    
                    # Extract hexagon counts
                    if "fallback regions created" in line or "hexagonal regions created" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "Resolution" in part and i+1 < len(parts):
                                try:
                                    res_num = int(parts[i+1].rstrip(':'))
                                    for j in range(i+2, len(parts)):
                                        if parts[j].isdigit():
                                            count = int(parts[j])
                                            progress_info['hexagon_counts'][res_num] = count
                                            break
                                except:
                                    pass
                    
                    # Track latest meaningful activity
                    if any(keyword in line.lower() for keyword in ['processing', 'created', 'calculated', 'saved', 'complete']):
                        progress_info['latest_activity'] = line[:100] + "..." if len(line) > 100 else line
                    
                    # Count errors
                    if "ERROR" in line or "Exception" in line:
                        progress_info['error_count'] += 1
                        
            except Exception as e:
                logger.warning(f"Error reading log {log_file}: {e}")
        
        progress_info['phases_completed'] = len(self.phases_detected)
        return progress_info
    
    def check_file_outputs(self) -> Dict:
        """Check what files have been generated."""
        
        file_status = {
            'results_files': [],
            'visualization_files': [],
            'log_files': [],
            'total_output_size_mb': 0
        }
        
        # Check results directory
        if self.results_dir.exists():
            for file_path in self.results_dir.rglob("*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / 1024**2
                    file_status['results_files'].append({
                        'name': file_path.name,
                        'size_mb': size_mb,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
                    file_status['total_output_size_mb'] += size_mb
        
        # Check visualization directory
        if self.viz_dir.exists():
            for file_path in self.viz_dir.rglob("*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / 1024**2
                    file_status['visualization_files'].append({
                        'name': file_path.name,
                        'size_mb': size_mb,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
                    file_status['total_output_size_mb'] += size_mb
        
        return file_status
    
    def generate_realtime_dashboard(self) -> None:
        """Generate a real-time monitoring dashboard."""
        
        if len(self.monitoring_data) < 2:
            return
        
        try:
            # Create dashboard plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('METACOGNITIVE ATTENTION DASHBOARD\nReal-time Hierarchical Experiment Monitoring', 
                        fontsize=16, fontweight='bold')
            
            # Extract data for plotting
            timestamps = [datetime.fromisoformat(d['timestamp']) for d in self.monitoring_data]
            cpu_data = [d.get('cpu_percent', 0) for d in self.monitoring_data]
            memory_data = [d.get('memory_percent', 0) for d in self.monitoring_data]
            gpu_memory_data = [d.get('gpu', {}).get('memory_percent', 0) for d in self.monitoring_data]
            gpu_util_data = [d.get('gpu', {}).get('utilization_percent', 0) for d in self.monitoring_data]
            
            # Plot 1: CPU and Memory
            ax1 = axes[0, 0]
            ax1.plot(timestamps, cpu_data, 'b-', label='CPU %', linewidth=2)
            ax1.plot(timestamps, memory_data, 'r-', label='Memory %', linewidth=2)
            ax1.set_title('System Resources', fontweight='bold')
            ax1.set_ylabel('Usage %')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: RTX 3090 GPU
            ax2 = axes[0, 1]
            ax2.plot(timestamps, gpu_memory_data, 'g-', label='GPU Memory %', linewidth=2)
            ax2.plot(timestamps, gpu_util_data, 'm-', label='GPU Utilization %', linewidth=2)
            ax2.set_title('RTX 3090 GPU Status', fontweight='bold')
            ax2.set_ylabel('Usage %')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Progress Timeline
            ax3 = axes[1, 0]
            progress_data = [d.get('progress', {}) for d in self.monitoring_data]
            phase_counts = [p.get('phases_completed', 0) for p in progress_data]
            error_counts = [p.get('error_count', 0) for p in progress_data]
            
            ax3.plot(timestamps, phase_counts, 'go-', label='Phases Completed', linewidth=2, markersize=6)
            ax3_twin = ax3.twinx()
            ax3_twin.plot(timestamps, error_counts, 'ro-', label='Errors', linewidth=2, markersize=4)
            
            ax3.set_title('Experiment Progress', fontweight='bold')
            ax3.set_ylabel('Phases Completed', color='green')
            ax3_twin.set_ylabel('Error Count', color='red')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Hexagon Processing Status
            ax4 = axes[1, 1]
            latest_progress = progress_data[-1] if progress_data else {}
            hexagon_counts = latest_progress.get('hexagon_counts', {})
            
            if hexagon_counts:
                resolutions = list(hexagon_counts.keys())
                counts = list(hexagon_counts.values())
                
                bars = ax4.bar(resolutions, counts, color=['blue', 'green', 'orange', 'red'])
                ax4.set_title('Hexagon Counts by Resolution', fontweight='bold')
                ax4.set_xlabel('H3 Resolution')
                ax4.set_ylabel('Hexagon Count')
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{count:,}', ha='center', va='bottom', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'Waiting for hexagon data...', ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_path = self.experiment_dir / "metacognitive_dashboard.png"
            plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Dashboard generation error: {e}")
    
    def monitor_cycle(self) -> Dict:
        """Single monitoring cycle - gather all intelligence."""
        
        logger.info("🔍 Metacognitive attention cycle...")
        
        # Gather system resources
        resources = self.get_system_resources()
        
        # Analyze experiment progress
        progress = self.analyze_log_progress()
        
        # Check file outputs
        files = self.check_file_outputs()
        
        # Compile monitoring snapshot
        monitoring_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'runtime_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'resources': resources,
            'progress': progress,
            'files': files
        }
        
        self.monitoring_data.append(monitoring_snapshot)
        
        # Log key insights
        gpu_mem = resources.get('gpu', {}).get('memory_percent', 0)
        current_phase = progress.get('current_phase', 'Unknown')
        hexagon_total = sum(progress.get('hexagon_counts', {}).values())
        
        logger.info(f"   Phase: {current_phase}")
        logger.info(f"   RTX 3090 GPU Memory: {gpu_mem:.1f}%")
        logger.info(f"   Total Hexagons: {hexagon_total:,}")
        logger.info(f"   Phases Completed: {progress['phases_completed']}/6")
        logger.info(f"   Output Files: {len(files['results_files']) + len(files['visualization_files'])}")
        
        # Generate dashboard periodically
        if len(self.monitoring_data) % 5 == 0:  # Every 5th cycle
            self.generate_realtime_dashboard()
        
        return monitoring_snapshot
    
    def run_continuous_monitoring(self, interval_seconds: int = 30, max_runtime_minutes: int = 30):
        """Run continuous metacognitive monitoring."""
        
        logger.info(f"🧠 STARTING CONTINUOUS METACOGNITIVE MONITORING")
        logger.info(f"   Monitoring interval: {interval_seconds} seconds")
        logger.info(f"   Maximum runtime: {max_runtime_minutes} minutes")
        logger.info("   Metacognitive attention engaged...")
        
        end_time = datetime.now() + timedelta(minutes=max_runtime_minutes)
        
        try:
            while datetime.now() < end_time:
                # Run monitoring cycle
                snapshot = self.monitor_cycle()
                
                # Check if experiment seems to be finished
                progress = snapshot['progress']
                if progress['phases_completed'] >= 6:
                    logger.info("🎉 EXPERIMENT APPEARS TO BE COMPLETE!")
                    logger.info("   All 6 phases detected. Continuing monitoring for final outputs...")
                
                # Check for critical issues
                gpu_temp = snapshot['resources'].get('gpu', {}).get('temperature_c', 0)
                if gpu_temp > 85:
                    logger.warning(f"RTX 3090 temperature high: {gpu_temp}°C")
                
                error_count = progress.get('error_count', 0)
                if error_count > 10:
                    logger.warning(f"High error count detected: {error_count}")
                
                # Save monitoring log
                log_path = self.experiment_dir / "metacognitive_monitoring_log.json"
                with open(log_path, 'w') as f:
                    json.dump(self.monitoring_data, f, indent=2)
                
                # Wait for next cycle
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f" Monitoring error: {e}")
        
        # Final summary
        logger.info("📊 METACOGNITIVE MONITORING SUMMARY:")
        if self.monitoring_data:
            final_snapshot = self.monitoring_data[-1]
            logger.info(f"   Total runtime: {final_snapshot['runtime_minutes']:.1f} minutes")
            logger.info(f"   Phases completed: {final_snapshot['progress']['phases_completed']}/6")
            logger.info(f"   Total hexagons processed: {sum(final_snapshot['progress']['hexagon_counts'].values()):,}")
            logger.info(f"   Output files generated: {len(final_snapshot['files']['results_files']) + len(final_snapshot['files']['visualization_files'])}")
            logger.info(f"   Total output size: {final_snapshot['files']['total_output_size_mb']:.1f} MB")
        
        logger.info("🧠 Metacognitive attention cycle complete.")


def main():
    """Run the metacognitive monitor."""
    
    monitor = MetacognitiveMonitor()
    
    # Run continuous monitoring
    monitor.run_continuous_monitoring(
        interval_seconds=30,  # Monitor every 30 seconds
        max_runtime_minutes=20  # Run for up to 20 minutes
    )


if __name__ == "__main__":
    main()