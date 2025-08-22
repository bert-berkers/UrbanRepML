#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MASTER OBSERVATORY - Universal Monitoring System
================================================

The ultimate monitoring system that acts as Claude's assistant, watching:
- All background bash processes and experiments
- Claude outputs, tool calls, and responses
- System resources (RTX 3090 GPU, memory, performance)
- Data flow, progress tracking, and error detection
- Multi-experiment coordination and optimization

This is the metacognitive layer that oversees all spatial AI operations.
"""

import time
import psutil
import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any
import subprocess
import os
import sys
import threading
import queue
import traceback
from dataclasses import dataclass
from collections import defaultdict, deque

# Setup logging
observatory_dir = Path("experiments/master_observatory")
observatory_dir.mkdir(parents=True, exist_ok=True)
log_dir = observatory_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - OBSERVATORY - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'master_observatory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessStatus:
    """Status of a monitored process."""
    process_id: str
    command: str
    status: str
    start_time: datetime
    last_output: str
    output_lines: int
    errors: List[str]
    resource_usage: Dict[str, float]


@dataclass
class ExperimentStatus:
    """Status of a running experiment."""
    name: str
    phase: str
    progress: float
    eta: Optional[datetime]
    hexagons_processed: int
    total_hexagons: int
    gpu_utilization: float
    memory_usage: float
    errors: List[str]
    outputs_generated: List[str]


class MasterObservatory:
    """
    The Ultimate Monitoring System - Claude's Assistant
    
    Provides comprehensive oversight of all spatial AI operations,
    acting as a second-order consciousness that watches everything.
    """
    
    def __init__(self):
        """Initialize the Master Observatory."""
        
        self.start_time = datetime.now()
        self.monitoring_active = True
        self.observatory_dir = observatory_dir
        self.dashboard_dir = self.observatory_dir / "dashboards"
        self.dashboard_dir.mkdir(exist_ok=True)
        
        # Monitoring data structures
        self.processes = {}  # bash_id -> ProcessStatus
        self.experiments = {}  # experiment_name -> ExperimentStatus
        self.system_metrics = deque(maxlen=1000)  # Recent system performance
        self.claude_outputs = deque(maxlen=500)  # Recent Claude outputs
        self.alerts = deque(maxlen=100)  # System alerts
        
        # Monitoring threads
        self.monitor_queue = queue.Queue()
        self.threads = []
        
        # Performance tracking
        self.gpu_history = deque(maxlen=200)
        self.memory_history = deque(maxlen=200)
        self.process_history = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("MASTER OBSERVATORY ACTIVATED")
        logger.info("Universal monitoring system initializing...")
        logger.info(f"Observatory base: {self.observatory_dir}")
        
        # Start monitoring threads
        self._start_monitoring_threads()
    
    def _start_monitoring_threads(self):
        """Start all monitoring threads."""
        
        # System resource monitor
        system_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)
        system_thread.start()
        self.threads.append(system_thread)
        
        # Process monitor
        process_thread = threading.Thread(target=self._monitor_processes, daemon=True)
        process_thread.start()
        self.threads.append(process_thread)
        
        # Experiment tracker
        experiment_thread = threading.Thread(target=self._monitor_experiments, daemon=True)
        experiment_thread.start()
        self.threads.append(experiment_thread)
        
        # Dashboard generator
        dashboard_thread = threading.Thread(target=self._generate_dashboards, daemon=True)
        dashboard_thread.start()
        self.threads.append(dashboard_thread)
        
        logger.info(f"Started {len(self.threads)} monitoring threads")
    
    def _monitor_system_resources(self):
        """Monitor system resources continuously."""
        
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU monitoring
                gpu_info = self._get_gpu_info()
                
                # Network and disk I/O
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / 1024**3,
                    'memory_total_gb': memory.total / 1024**3,
                    'gpu': gpu_info,
                    'disk_read_mb': disk_io.read_bytes / 1024**2 if disk_io else 0,
                    'disk_write_mb': disk_io.write_bytes / 1024**2 if disk_io else 0,
                    'net_sent_mb': net_io.bytes_sent / 1024**2 if net_io else 0,
                    'net_recv_mb': net_io.bytes_recv / 1024**2 if net_io else 0
                }
                
                self.system_metrics.append(metrics)
                self.gpu_history.append(gpu_info.get('utilization_percent', 0))
                self.memory_history.append(memory.percent)
                
                # Check for alerts
                self._check_resource_alerts(metrics)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
            
            time.sleep(5)  # Monitor every 5 seconds
    
    def _get_gpu_info(self) -> Dict:
        """Get RTX 3090 GPU information."""
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                'name': pynvml.nvmlDeviceGetName(handle).decode(),
                'memory_used_gb': memory_info.used / 1024**3,
                'memory_total_gb': memory_info.total / 1024**3,
                'memory_percent': (memory_info.used / memory_info.total) * 100,
                'utilization_percent': utilization.gpu,
                'memory_utilization_percent': utilization.memory,
                'temperature_c': temperature
            }
        except Exception:
            return {
                'name': 'RTX 3090 (not detected)',
                'memory_used_gb': 0,
                'memory_total_gb': 24.0,
                'memory_percent': 0,
                'utilization_percent': 0,
                'memory_utilization_percent': 0,
                'temperature_c': 0
            }
    
    def _monitor_processes(self):
        """Monitor all background bash processes."""
        
        while self.monitoring_active:
            try:
                # Check known bash processes
                for bash_id in ['bash_1', 'bash_2', 'bash_3', 'bash_4', 'bash_5']:
                    try:
                        # Get process output using Claude's BashOutput tool
                        # This would be called via the monitoring interface
                        self._update_process_status(bash_id)
                    except Exception as e:
                        logger.debug(f"Process {bash_id} not active: {e}")
                
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
            
            time.sleep(10)  # Check processes every 10 seconds
    
    def _update_process_status(self, bash_id: str):
        """Update status for a specific bash process."""
        
        # This would interface with Claude's bash monitoring
        # For now, we'll simulate process tracking
        
        if bash_id not in self.processes:
            self.processes[bash_id] = ProcessStatus(
                process_id=bash_id,
                command="Unknown",
                status="unknown",
                start_time=datetime.now(),
                last_output="",
                output_lines=0,
                errors=[],
                resource_usage={}
            )
    
    def _monitor_experiments(self):
        """Monitor running experiments and their progress."""
        
        while self.monitoring_active:
            try:
                # Check for experiment log files
                experiment_dirs = [
                    Path("experiments/del_norte_active_inference"),
                    Path("experiments/cascadia_massive_2021"),
                    Path("experiments/cascadia_geoinfer_alphaearth")
                ]
                
                for exp_dir in experiment_dirs:
                    if exp_dir.exists():
                        self._analyze_experiment_logs(exp_dir)
                
            except Exception as e:
                logger.error(f"Experiment monitoring error: {e}")
            
            time.sleep(15)  # Check experiments every 15 seconds
    
    def _analyze_experiment_logs(self, exp_dir: Path):
        """Analyze experiment logs for progress tracking."""
        
        log_files = list(exp_dir.glob("logs/*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                # Analyze recent lines for progress indicators
                recent_lines = lines[-50:] if len(lines) > 50 else lines
                
                progress_info = self._extract_progress_info(recent_lines)
                
                if progress_info:
                    exp_name = exp_dir.name
                    self.experiments[exp_name] = ExperimentStatus(
                        name=exp_name,
                        phase=progress_info.get('phase', 'Unknown'),
                        progress=progress_info.get('progress', 0.0),
                        eta=progress_info.get('eta'),
                        hexagons_processed=progress_info.get('hexagons_processed', 0),
                        total_hexagons=progress_info.get('total_hexagons', 0),
                        gpu_utilization=progress_info.get('gpu_utilization', 0.0),
                        memory_usage=progress_info.get('memory_usage', 0.0),
                        errors=progress_info.get('errors', []),
                        outputs_generated=progress_info.get('outputs', [])
                    )
                    
            except Exception as e:
                logger.debug(f"Error reading log {log_file}: {e}")
    
    def _extract_progress_info(self, lines: List[str]) -> Dict:
        """Extract progress information from log lines."""
        
        progress_info = {
            'phase': 'Unknown',
            'progress': 0.0,
            'hexagons_processed': 0,
            'total_hexagons': 0,
            'errors': [],
            'outputs': []
        }
        
        for line in lines:
            line = line.strip()
            
            # Phase detection
            if "PHASE" in line:
                progress_info['phase'] = line
            
            # Hexagon counts
            if "hexagons" in line.lower() or "regions" in line.lower():
                numbers = [int(s) for s in line.split() if s.isdigit()]
                if numbers:
                    progress_info['hexagons_processed'] = max(numbers)
            
            # Error detection
            if "ERROR" in line or "Exception" in line:
                progress_info['errors'].append(line)
            
            # Output detection
            if "saved" in line.lower() or "generated" in line.lower():
                progress_info['outputs'].append(line)
        
        return progress_info
    
    def _check_resource_alerts(self, metrics: Dict):
        """Check for resource-based alerts."""
        
        gpu = metrics.get('gpu', {})
        
        # GPU temperature alert
        temp = gpu.get('temperature_c', 0)
        if temp > 85:
            alert = f"HIGH GPU TEMPERATURE: {temp}°C"
            self.alerts.append({'timestamp': datetime.now(), 'level': 'WARNING', 'message': alert})
            logger.warning(alert)
        
        # Memory usage alert
        if metrics['memory_percent'] > 90:
            alert = f"HIGH MEMORY USAGE: {metrics['memory_percent']:.1f}%"
            self.alerts.append({'timestamp': datetime.now(), 'level': 'WARNING', 'message': alert})
            logger.warning(alert)
        
        # GPU memory alert
        gpu_mem = gpu.get('memory_percent', 0)
        if gpu_mem > 95:
            alert = f"HIGH GPU MEMORY: {gpu_mem:.1f}%"
            self.alerts.append({'timestamp': datetime.now(), 'level': 'CRITICAL', 'message': alert})
            logger.error(alert)
    
    def _generate_dashboards(self):
        """Generate monitoring dashboards."""
        
        while self.monitoring_active:
            try:
                if len(self.system_metrics) > 10:  # Wait for some data
                    self._create_system_dashboard()
                    self._create_experiment_dashboard()
                    self._create_process_dashboard()
                
            except Exception as e:
                logger.error(f"Dashboard generation error: {e}")
                logger.error(traceback.format_exc())
            
            time.sleep(30)  # Generate dashboards every 30 seconds
    
    def _create_system_dashboard(self):
        """Create system resource dashboard."""
        
        if len(self.system_metrics) < 5:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MASTER OBSERVATORY - System Resources', fontsize=16, fontweight='bold')
        
        # Extract data
        timestamps = [m['timestamp'] for m in self.system_metrics]
        cpu_data = [m['cpu_percent'] for m in self.system_metrics]
        memory_data = [m['memory_percent'] for m in self.system_metrics]
        gpu_util = [m['gpu']['utilization_percent'] for m in self.system_metrics]
        gpu_mem = [m['gpu']['memory_percent'] for m in self.system_metrics]
        
        # CPU and Memory
        axes[0, 0].plot(timestamps, cpu_data, 'b-', label='CPU %', linewidth=2)
        axes[0, 0].plot(timestamps, memory_data, 'r-', label='Memory %', linewidth=2)
        axes[0, 0].set_title('CPU & System Memory', fontweight='bold')
        axes[0, 0].set_ylabel('Usage %')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RTX 3090 GPU
        axes[0, 1].plot(timestamps, gpu_util, 'g-', label='GPU Util %', linewidth=2)
        axes[0, 1].plot(timestamps, gpu_mem, 'm-', label='GPU Memory %', linewidth=2)
        axes[0, 1].set_title('RTX 3090 GPU Performance', fontweight='bold')
        axes[0, 1].set_ylabel('Usage %')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Process count and alerts
        axes[1, 0].text(0.1, 0.8, f"Active Processes: {len(self.processes)}", transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].text(0.1, 0.6, f"Active Experiments: {len(self.experiments)}", transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].text(0.1, 0.4, f"Recent Alerts: {len(self.alerts)}", transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('Observatory Status', fontweight='bold')
        
        # Latest metrics
        if self.system_metrics:
            latest = self.system_metrics[-1]
            gpu_info = latest['gpu']
            axes[1, 1].text(0.1, 0.8, f"GPU: {gpu_info['name']}", transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.6, f"GPU Temp: {gpu_info['temperature_c']}°C", transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.4, f"GPU Memory: {gpu_info['memory_used_gb']:.1f}/{gpu_info['memory_total_gb']:.1f} GB", transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.2, f"System Memory: {latest['memory_used_gb']:.1f}/{latest['memory_total_gb']:.1f} GB", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Current Status', fontweight='bold')
        
        plt.tight_layout()
        dashboard_path = self.dashboard_dir / "system_dashboard.png"
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_experiment_dashboard(self):
        """Create experiment monitoring dashboard."""
        
        if not self.experiments:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('MASTER OBSERVATORY - Experiment Status', fontsize=16, fontweight='bold')
        
        y_pos = np.arange(len(self.experiments))
        experiment_names = list(self.experiments.keys())
        progresses = [exp.progress for exp in self.experiments.values()]
        
        bars = ax.barh(y_pos, progresses, color=['green' if p > 0.8 else 'orange' if p > 0.5 else 'red' for p in progresses])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(experiment_names)
        ax.set_xlabel('Progress %')
        ax.set_title('Experiment Progress Overview')
        
        # Add progress labels
        for i, (bar, progress) in enumerate(zip(bars, progresses)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{progress:.1f}%', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        dashboard_path = self.dashboard_dir / "experiment_dashboard.png"
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _create_process_dashboard(self):
        """Create process monitoring dashboard."""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('MASTER OBSERVATORY - Process Status', fontsize=16, fontweight='bold')
        
        if self.processes:
            process_names = list(self.processes.keys())
            statuses = [proc.status for proc in self.processes.values()]
            
            status_counts = {}
            for status in statuses:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                ax.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
                ax.set_title('Process Status Distribution')
        else:
            ax.text(0.5, 0.5, 'No active processes detected', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        plt.tight_layout()
        dashboard_path = self.dashboard_dir / "process_dashboard.png"
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def get_status_report(self) -> Dict:
        """Generate comprehensive status report."""
        
        latest_metrics = self.system_metrics[-1] if self.system_metrics else {}
        
        return {
            'observatory_status': {
                'uptime_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                'monitoring_threads': len(self.threads),
                'data_points_collected': len(self.system_metrics),
                'alerts_generated': len(self.alerts)
            },
            'system_resources': latest_metrics,
            'active_processes': {pid: proc.status for pid, proc in self.processes.items()},
            'active_experiments': {name: exp.phase for name, exp in self.experiments.items()},
            'recent_alerts': list(self.alerts)[-5:] if self.alerts else []
        }
    
    def shutdown(self):
        """Shutdown the observatory gracefully."""
        
        logger.info("MASTER OBSERVATORY SHUTDOWN INITIATED")
        self.monitoring_active = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2)
        
        # Save final report
        final_report = self.get_status_report()
        report_path = self.observatory_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Final report saved: {report_path}")
        logger.info("MASTER OBSERVATORY SHUTDOWN COMPLETE")


def main():
    """Run the Master Observatory."""
    
    observatory = MasterObservatory()
    
    try:
        logger.info("MASTER OBSERVATORY ONLINE")
        logger.info("Monitoring all spatial AI operations...")
        logger.info("Press Ctrl+C to shutdown")
        
        # Main monitoring loop
        while True:
            time.sleep(60)  # Main loop runs every minute
            
            # Generate periodic status reports
            if datetime.now().minute % 5 == 0:  # Every 5 minutes
                status = observatory.get_status_report()
                logger.info(f"STATUS: {len(observatory.processes)} processes, {len(observatory.experiments)} experiments")
                
                if observatory.alerts:
                    latest_alert = observatory.alerts[-1]
                    logger.info(f"LATEST ALERT: {latest_alert['message']}")
    
    except KeyboardInterrupt:
        logger.info("Observatory shutdown requested by user")
    
    finally:
        observatory.shutdown()


if __name__ == "__main__":
    main()