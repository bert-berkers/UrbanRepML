#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Performance Monitor for Del Norte Processing
Real-time monitoring of GPU utilization, memory, and processing progress
"""

import torch
import psutil
import GPUtil
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Real-time GPU performance monitoring"""
    
    def __init__(self, log_interval: int = 5, history_size: int = 100):
        """
        Initialize GPU monitor
        
        Args:
            log_interval: Seconds between measurements
            history_size: Number of historical points to keep
        """
        self.log_interval = log_interval
        self.history_size = history_size
        self.monitoring = False
        self.monitor_thread = None
        
        # Performance history
        self.gpu_utilization = deque(maxlen=history_size)
        self.gpu_memory = deque(maxlen=history_size)
        self.gpu_temp = deque(maxlen=history_size)
        self.cpu_utilization = deque(maxlen=history_size)
        self.ram_usage = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
        # Processing metrics
        self.tiles_processed = 0
        self.hexagons_created = 0
        self.start_time = None
        self.checkpoint_path = Path("data/processing_checkpoint.json")
        
        # Output paths
        self.metrics_file = Path("logs/gpu_metrics.json")
        self.metrics_file.parent.mkdir(exist_ok=True)
        
        # Check GPU availability
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Monitoring GPU: {self.gpu_name} ({self.gpu_memory_total:.1f} GB)")
        else:
            logger.warning("No GPU detected, monitoring CPU only")
    
    def get_gpu_metrics(self) -> Dict:
        """Get current GPU metrics"""
        metrics = {}
        
        if self.has_gpu:
            try:
                # Get GPU info using GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics['gpu_utilization'] = gpu.load * 100  # Percentage
                    metrics['gpu_memory_used'] = gpu.memoryUsed / 1024  # GB
                    metrics['gpu_memory_free'] = gpu.memoryFree / 1024  # GB
                    metrics['gpu_memory_percent'] = gpu.memoryUtil * 100  # Percentage
                    metrics['gpu_temperature'] = gpu.temperature  # Celsius
                
                # Additional PyTorch metrics
                if torch.cuda.is_available():
                    metrics['torch_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                    metrics['torch_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
                    
            except Exception as e:
                logger.warning(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def get_system_metrics(self) -> Dict:
        """Get system CPU and RAM metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_count': psutil.cpu_count(),
            'ram_used': psutil.virtual_memory().used / 1024**3,  # GB
            'ram_available': psutil.virtual_memory().available / 1024**3,  # GB
            'ram_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        # Get per-core CPU usage
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        metrics['cpu_per_core'] = cpu_per_core
        
        return metrics
    
    def get_processing_progress(self) -> Dict:
        """Get processing progress from checkpoint"""
        progress = {
            'tiles_processed': 0,
            'processing_rate': 0,
            'estimated_time_remaining': 'Unknown'
        }
        
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                    progress['tiles_processed'] = len(checkpoint.get('processed_tiles', []))
                    
                    # Calculate processing rate
                    if self.start_time:
                        elapsed = time.time() - self.start_time
                        if elapsed > 0 and progress['tiles_processed'] > 0:
                            progress['processing_rate'] = progress['tiles_processed'] / (elapsed / 60)  # tiles/min
                            
                            # Estimate remaining time (assuming we know total tiles)
                            # This would need to be updated with actual total
                            # progress['estimated_time_remaining'] = ...
                            
            except Exception as e:
                logger.warning(f"Error reading checkpoint: {e}")
        
        return progress
    
    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting GPU monitoring...")
        self.start_time = time.time()
        
        while self.monitoring:
            try:
                # Collect metrics
                timestamp = datetime.now()
                gpu_metrics = self.get_gpu_metrics()
                system_metrics = self.get_system_metrics()
                progress = self.get_processing_progress()
                
                # Store in history
                self.timestamps.append(timestamp)
                
                if gpu_metrics:
                    self.gpu_utilization.append(gpu_metrics.get('gpu_utilization', 0))
                    self.gpu_memory.append(gpu_metrics.get('gpu_memory_used', 0))
                    self.gpu_temp.append(gpu_metrics.get('gpu_temperature', 0))
                
                self.cpu_utilization.append(system_metrics['cpu_percent'])
                self.ram_usage.append(system_metrics['ram_used'])
                
                # Create comprehensive metrics
                full_metrics = {
                    'timestamp': timestamp.isoformat(),
                    'gpu': gpu_metrics,
                    'system': system_metrics,
                    'progress': progress,
                    'runtime_minutes': (time.time() - self.start_time) / 60
                }
                
                # Log to file
                self.log_metrics(full_metrics)
                
                # Print summary
                self.print_summary(full_metrics)
                
                # Check for issues
                self.check_thresholds(full_metrics)
                
                # Wait for next interval
                time.sleep(self.log_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.log_interval)
    
    def log_metrics(self, metrics: Dict):
        """Log metrics to file"""
        try:
            # Append to JSON lines file
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except Exception as e:
            logger.warning(f"Error logging metrics: {e}")
    
    def print_summary(self, metrics: Dict):
        """Print monitoring summary to console"""
        summary = []
        summary.append("="*60)
        summary.append(f"GPU MONITOR | Runtime: {metrics['runtime_minutes']:.1f} min")
        summary.append("-"*60)
        
        if metrics['gpu']:
            summary.append(f"GPU Util: {metrics['gpu']['gpu_utilization']:.1f}% | "
                         f"Memory: {metrics['gpu']['gpu_memory_used']:.1f}/{self.gpu_memory_total:.1f} GB | "
                         f"Temp: {metrics['gpu']['gpu_temperature']:.0f}°C")
        
        summary.append(f"CPU: {metrics['system']['cpu_percent']:.1f}% | "
                      f"RAM: {metrics['system']['ram_used']:.1f} GB ({metrics['system']['ram_percent']:.1f}%)")
        
        summary.append(f"Tiles Processed: {metrics['progress']['tiles_processed']} | "
                      f"Rate: {metrics['progress']['processing_rate']:.1f} tiles/min")
        
        summary.append("="*60)
        
        print('\n'.join(summary))
    
    def check_thresholds(self, metrics: Dict):
        """Check for performance issues and warnings"""
        warnings = []
        
        # GPU checks
        if metrics['gpu']:
            if metrics['gpu']['gpu_temperature'] > 83:
                warnings.append(f"⚠️ GPU temperature high: {metrics['gpu']['gpu_temperature']}°C")
            
            if metrics['gpu']['gpu_memory_percent'] > 95:
                warnings.append(f"⚠️ GPU memory nearly full: {metrics['gpu']['gpu_memory_percent']:.1f}%")
            
            if metrics['gpu']['gpu_utilization'] < 50 and metrics['progress']['tiles_processed'] > 0:
                warnings.append(f"⚠️ GPU underutilized: {metrics['gpu']['gpu_utilization']:.1f}%")
        
        # System checks
        if metrics['system']['ram_percent'] > 90:
            warnings.append(f"⚠️ System RAM high: {metrics['system']['ram_percent']:.1f}%")
        
        if metrics['system']['disk_usage'] > 90:
            warnings.append(f"⚠️ Disk space low: {metrics['system']['disk_usage']:.1f}%")
        
        # Print warnings
        for warning in warnings:
            logger.warning(warning)
    
    def start(self):
        """Start monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("GPU monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=10)
            logger.info("GPU monitoring stopped")
    
    def plot_realtime(self):
        """Create real-time performance plot"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('GPU Performance Monitor', fontsize=16)
        
        def update_plots(frame):
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            # GPU Utilization
            if self.gpu_utilization:
                axes[0, 0].plot(list(self.gpu_utilization), 'b-')
                axes[0, 0].set_ylabel('GPU Utilization (%)')
                axes[0, 0].set_ylim(0, 100)
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_title(f'GPU: {self.gpu_name if self.has_gpu else "N/A"}')
            
            # GPU Memory
            if self.gpu_memory:
                axes[0, 1].plot(list(self.gpu_memory), 'g-')
                axes[0, 1].set_ylabel('GPU Memory (GB)')
                axes[0, 1].set_ylim(0, self.gpu_memory_total if self.has_gpu else 24)
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_title('GPU Memory Usage')
            
            # CPU Utilization
            if self.cpu_utilization:
                axes[1, 0].plot(list(self.cpu_utilization), 'r-')
                axes[1, 0].set_ylabel('CPU Utilization (%)')
                axes[1, 0].set_ylim(0, 100)
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_title('CPU Usage')
            
            # RAM Usage
            if self.ram_usage:
                axes[1, 1].plot(list(self.ram_usage), 'm-')
                axes[1, 1].set_ylabel('RAM Usage (GB)')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_title('System Memory')
            
            plt.tight_layout()
        
        # Create animation
        ani = animation.FuncAnimation(fig, update_plots, interval=1000, blit=False)
        plt.show()
        
        return ani
    
    def generate_report(self) -> Dict:
        """Generate final performance report"""
        report = {
            'monitoring_duration_minutes': (time.time() - self.start_time) / 60 if self.start_time else 0,
            'gpu': {
                'name': self.gpu_name if self.has_gpu else 'None',
                'memory_total_gb': self.gpu_memory_total if self.has_gpu else 0,
                'avg_utilization': np.mean(list(self.gpu_utilization)) if self.gpu_utilization else 0,
                'max_utilization': np.max(list(self.gpu_utilization)) if self.gpu_utilization else 0,
                'avg_memory_gb': np.mean(list(self.gpu_memory)) if self.gpu_memory else 0,
                'max_memory_gb': np.max(list(self.gpu_memory)) if self.gpu_memory else 0,
                'avg_temperature': np.mean(list(self.gpu_temp)) if self.gpu_temp else 0,
                'max_temperature': np.max(list(self.gpu_temp)) if self.gpu_temp else 0
            },
            'cpu': {
                'cores': psutil.cpu_count(),
                'avg_utilization': np.mean(list(self.cpu_utilization)) if self.cpu_utilization else 0,
                'max_utilization': np.max(list(self.cpu_utilization)) if self.cpu_utilization else 0
            },
            'ram': {
                'total_gb': psutil.virtual_memory().total / 1024**3,
                'avg_usage_gb': np.mean(list(self.ram_usage)) if self.ram_usage else 0,
                'max_usage_gb': np.max(list(self.ram_usage)) if self.ram_usage else 0
            },
            'processing': self.get_processing_progress()
        }
        
        # Save report
        report_path = Path("logs/performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {report_path}")
        return report


def main():
    """Run standalone monitoring"""
    monitor = GPUMonitor(log_interval=5)
    
    try:
        # Start monitoring
        monitor.start()
        
        # Keep running until interrupted
        logger.info("Monitoring running. Press Ctrl+C to stop...")
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("Stopping monitor...")
        monitor.stop()
        
        # Generate final report
        report = monitor.generate_report()
        print("\nFinal Performance Report:")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()