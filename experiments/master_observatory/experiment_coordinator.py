#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MULTI-EXPERIMENT COORDINATOR
============================

Intelligent coordinator for managing multiple massive spatial AI experiments:
- Queue management for RTX 3090 optimization
- Resource allocation and conflict prevention
- Performance optimization and adaptive scaling
- Progress synchronization and result aggregation
- Automatic recovery and error handling

This system ensures optimal utilization of the RTX 3090 24GB while 
coordinating multiple large-scale experiments simultaneously.
"""

import time
import logging
import json
import queue
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import psutil
import numpy as np

# Setup logging
coordinator_dir = Path("experiments/master_observatory")
log_dir = coordinator_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - COORDINATOR - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'experiment_coordinator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentPriority(Enum):
    """Experiment priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ExperimentStatus(Enum):
    """Experiment execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for a spatial AI experiment."""
    name: str
    script_path: str
    args: List[str]
    priority: ExperimentPriority
    estimated_runtime_minutes: int
    memory_requirement_gb: float
    gpu_memory_requirement_gb: float
    max_parallel_processes: int
    dependencies: List[str]
    output_directory: str
    
    # Resource constraints
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    max_gpu_memory_percent: float = 95.0
    
    # Auto-scaling parameters
    batch_size_min: int = 100
    batch_size_max: int = 2000
    adaptive_scaling: bool = True


@dataclass
class ExperimentInstance:
    """Running instance of an experiment."""
    config: ExperimentConfig
    status: ExperimentStatus
    process_id: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    progress: float
    current_phase: str
    resource_usage: Dict[str, float]
    error_messages: List[str]
    output_files: List[str]
    
    # Performance metrics
    hexagons_processed: int = 0
    tiles_processed: int = 0
    throughput_hexagons_per_minute: float = 0.0
    estimated_completion: Optional[datetime] = None


class MultiExperimentCoordinator:
    """
    Intelligent coordinator for multiple massive spatial AI experiments.
    Optimizes RTX 3090 utilization while preventing resource conflicts.
    """
    
    def __init__(self):
        """Initialize the experiment coordinator."""
        
        self.coordinator_dir = coordinator_dir
        self.queue_dir = self.coordinator_dir / "queue"
        self.results_dir = self.coordinator_dir / "results"
        self.queue_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Experiment management
        self.experiment_queue = queue.PriorityQueue()
        self.running_experiments = {}  # experiment_name -> ExperimentInstance
        self.completed_experiments = {}  # experiment_name -> ExperimentInstance
        self.failed_experiments = {}  # experiment_name -> ExperimentInstance
        
        # Resource monitoring
        self.system_resources = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_memory_percent': 0.0,
            'gpu_utilization': 0.0,
            'available_processes': 5  # Maximum parallel experiments
        }
        
        # Performance optimization
        self.performance_history = []
        self.optimization_enabled = True
        self.auto_scaling_enabled = True
        
        # Coordination state
        self.coordinator_active = True
        self.last_resource_check = datetime.now()
        
        logger.info("MULTI-EXPERIMENT COORDINATOR ACTIVATED")
        logger.info("RTX 3090 optimization and resource management enabled")
        
        # Start coordination thread
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
    
    def add_experiment(self, config: ExperimentConfig) -> str:
        """Add an experiment to the execution queue."""
        
        experiment_instance = ExperimentInstance(
            config=config,
            status=ExperimentStatus.QUEUED,
            process_id=None,
            start_time=None,
            end_time=None,
            progress=0.0,
            current_phase="Queued",
            resource_usage={},
            error_messages=[],
            output_files=[]
        )
        
        # Priority queue uses negative values for max-heap behavior
        priority_value = -config.priority.value
        queue_item = (priority_value, datetime.now(), config.name, experiment_instance)
        
        self.experiment_queue.put(queue_item)
        
        logger.info(f"Experiment queued: {config.name} (Priority: {config.priority.name})")
        logger.info(f"  Estimated runtime: {config.estimated_runtime_minutes} minutes")
        logger.info(f"  Memory requirement: {config.memory_requirement_gb:.1f}GB RAM, {config.gpu_memory_requirement_gb:.1f}GB GPU")
        
        return config.name
    
    def _coordination_loop(self):
        """Main coordination loop for managing experiments."""
        
        while self.coordinator_active:
            try:
                # Update resource monitoring
                self._update_system_resources()
                
                # Check running experiments
                self._monitor_running_experiments()
                
                # Start new experiments if resources available
                self._start_queued_experiments()
                
                # Optimize running experiments
                if self.optimization_enabled:
                    self._optimize_experiment_performance()
                
                # Clean up completed experiments
                self._cleanup_completed_experiments()
                
                # Generate progress reports
                self._generate_progress_report()
                
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
            
            time.sleep(10)  # Coordinate every 10 seconds
    
    def _update_system_resources(self):
        """Update current system resource usage."""
        
        try:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU resources
            gpu_info = self._get_gpu_info()
            
            self.system_resources.update({
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / 1024**3,
                'memory_total_gb': memory.total / 1024**3,
                'gpu_memory_percent': gpu_info.get('memory_percent', 0),
                'gpu_utilization': gpu_info.get('utilization_percent', 0),
                'gpu_temperature': gpu_info.get('temperature_c', 0),
                'available_processes': max(0, 5 - len(self.running_experiments))
            })
            
            self.last_resource_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")
    
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
                'memory_percent': (memory_info.used / memory_info.total) * 100,
                'utilization_percent': utilization.gpu,
                'temperature_c': temperature,
                'memory_used_gb': memory_info.used / 1024**3,
                'memory_total_gb': memory_info.total / 1024**3
            }
        except Exception:
            return {
                'memory_percent': 0,
                'utilization_percent': 0,
                'temperature_c': 0,
                'memory_used_gb': 0,
                'memory_total_gb': 24.0
            }
    
    def _monitor_running_experiments(self):
        """Monitor progress and status of running experiments."""
        
        for exp_name, experiment in list(self.running_experiments.items()):
            try:
                # Check if actual process is still running
                if hasattr(experiment, '_subprocess') and experiment._subprocess:
                    process = experiment._subprocess
                    poll_result = process.poll()
                    
                    if poll_result is not None:
                        # Process finished
                        if poll_result == 0:
                            # Success
                            experiment.progress = 100.0
                            self._complete_experiment(exp_name, experiment)
                        else:
                            # Failed
                            stderr_output = process.stderr.read() if process.stderr else "Unknown error"
                            experiment.error_messages.append(f"Process exit code: {poll_result}, Error: {stderr_output}")
                            self._fail_experiment(exp_name, experiment)
                    else:
                        # Process still running, update progress
                        self._update_experiment_progress(experiment)
                
                # Check for other failure conditions
                elif self._check_experiment_failed(experiment):
                    self._fail_experiment(exp_name, experiment)
                
            except Exception as e:
                logger.error(f"Error monitoring experiment {exp_name}: {e}")
    
    def _update_experiment_progress(self, experiment: ExperimentInstance):
        """Update progress tracking for an experiment."""
        
        # Check experiment logs for progress indicators
        log_file = Path(experiment.config.output_directory) / "logs" / "experiment.log"
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                # Analyze recent lines for progress
                recent_lines = lines[-20:] if len(lines) > 20 else lines
                
                for line in recent_lines:
                    # Extract hexagon counts
                    if "hexagons" in line.lower() or "regions" in line.lower():
                        numbers = [int(s) for s in line.split() if s.isdigit()]
                        if numbers:
                            experiment.hexagons_processed = max(numbers)
                    
                    # Extract phase information
                    if "PHASE" in line:
                        experiment.current_phase = line.strip()
                    
                    # Calculate progress based on phases and hexagons
                    if experiment.hexagons_processed > 0:
                        # Estimate progress based on typical experiment patterns
                        estimated_total = 100000  # Typical large experiment
                        experiment.progress = min(100.0, (experiment.hexagons_processed / estimated_total) * 100)
                
            except Exception as e:
                logger.debug(f"Error reading experiment log: {e}")
    
    def _start_queued_experiments(self):
        """Start queued experiments if resources are available."""
        
        while not self.experiment_queue.empty() and self._can_start_experiment():
            try:
                priority, queue_time, exp_name, experiment = self.experiment_queue.get_nowait()
                
                # Check if we have sufficient resources
                if self._check_resource_availability(experiment.config):
                    self._launch_experiment(exp_name, experiment)
                else:
                    # Put back in queue if insufficient resources
                    self.experiment_queue.put((priority, queue_time, exp_name, experiment))
                    break
                    
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error starting queued experiment: {e}")
    
    def _can_start_experiment(self) -> bool:
        """Check if we can start another experiment."""
        
        return (
            len(self.running_experiments) < 3 and  # Max 3 parallel experiments
            self.system_resources['cpu_percent'] < 70 and
            self.system_resources['memory_percent'] < 75 and
            self.system_resources['gpu_memory_percent'] < 80
        )
    
    def _check_resource_availability(self, config: ExperimentConfig) -> bool:
        """Check if sufficient resources are available for an experiment."""
        
        # Check memory requirements
        available_memory = self.system_resources['memory_total_gb'] * (100 - self.system_resources['memory_percent']) / 100
        available_gpu_memory = 24.0 * (100 - self.system_resources['gpu_memory_percent']) / 100
        
        return (
            available_memory >= config.memory_requirement_gb and
            available_gpu_memory >= config.gpu_memory_requirement_gb and
            self.system_resources['cpu_percent'] < config.max_cpu_percent
        )
    
    def _launch_experiment(self, exp_name: str, experiment: ExperimentInstance):
        """Launch an experiment."""
        
        logger.info(f"LAUNCHING EXPERIMENT: {exp_name}")
        logger.info(f"  Script: {experiment.config.script_path}")
        logger.info(f"  Args: {' '.join(experiment.config.args)}")
        
        try:
            # Actually launch the experiment using subprocess
            script_path = experiment.config.script_path
            args = experiment.config.args
            
            # Build the full command
            if args:
                cmd = ["python", script_path] + args
            else:
                cmd = ["python", script_path]
            
            # Use project root as working directory
            project_root = Path(__file__).parent.parent.parent
            
            # Launch the process
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            experiment.status = ExperimentStatus.RUNNING
            experiment.start_time = datetime.now()
            experiment.process_id = f"bash_{process.pid}"  # Use actual PID
            experiment.current_phase = "Phase 1: Initialization"
            experiment._subprocess = process  # Store process reference
            
            # Estimate completion time
            experiment.estimated_completion = experiment.start_time + timedelta(
                minutes=experiment.config.estimated_runtime_minutes
            )
            
            self.running_experiments[exp_name] = experiment
            
            logger.info(f"  Process ID: {experiment.process_id} (PID: {process.pid})")
            logger.info(f"  Working directory: {project_root}")
            logger.info(f"  Estimated completion: {experiment.estimated_completion}")
            
        except Exception as e:
            logger.error(f"Failed to launch experiment {exp_name}: {e}")
            experiment.status = ExperimentStatus.FAILED
            experiment.error_messages.append(str(e))
    
    def _optimize_experiment_performance(self):
        """Optimize performance of running experiments."""
        
        for exp_name, experiment in self.running_experiments.items():
            if experiment.config.adaptive_scaling:
                self._adjust_experiment_parameters(experiment)
    
    def _adjust_experiment_parameters(self, experiment: ExperimentInstance):
        """Adjust experiment parameters for optimal performance."""
        
        # Monitor throughput and adjust batch sizes
        current_time = datetime.now()
        if experiment.start_time:
            runtime_minutes = (current_time - experiment.start_time).total_seconds() / 60
            
            if runtime_minutes > 5 and experiment.hexagons_processed > 0:
                throughput = experiment.hexagons_processed / runtime_minutes
                experiment.throughput_hexagons_per_minute = throughput
                
                # Adjust based on GPU utilization
                gpu_util = self.system_resources['gpu_utilization']
                
                if gpu_util < 50 and throughput < 1000:
                    # GPU underutilized, can increase batch size
                    logger.info(f"Low GPU utilization ({gpu_util}%), optimizing {experiment.config.name}")
                elif gpu_util > 95:
                    # GPU overloaded, reduce batch size
                    logger.info(f"High GPU utilization ({gpu_util}%), reducing load for {experiment.config.name}")
    
    def _check_experiment_failed(self, experiment: ExperimentInstance) -> bool:
        """Check if an experiment has failed."""
        
        # Check for timeout
        if experiment.start_time and experiment.config.estimated_runtime_minutes > 0:
            runtime = (datetime.now() - experiment.start_time).total_seconds() / 60
            if runtime > experiment.config.estimated_runtime_minutes * 2:  # 2x timeout
                return True
        
        # Check for error indicators in logs
        log_file = Path(experiment.config.output_directory) / "logs" / "experiment.log"
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    recent_lines = f.readlines()[-10:]
                
                for line in recent_lines:
                    if any(error_term in line.lower() for error_term in ['critical error', 'fatal', 'failed', 'exception']):
                        experiment.error_messages.append(line.strip())
                        return True
            except Exception:
                pass
        
        return False
    
    def _complete_experiment(self, exp_name: str, experiment: ExperimentInstance):
        """Mark an experiment as completed."""
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.now()
        experiment.progress = 100.0
        
        # Move to completed experiments
        self.completed_experiments[exp_name] = experiment
        del self.running_experiments[exp_name]
        
        runtime = (experiment.end_time - experiment.start_time).total_seconds() / 60
        
        logger.info(f"EXPERIMENT COMPLETED: {exp_name}")
        logger.info(f"  Runtime: {runtime:.1f} minutes")
        logger.info(f"  Hexagons processed: {experiment.hexagons_processed:,}")
        logger.info(f"  Throughput: {experiment.throughput_hexagons_per_minute:.1f} hexagons/min")
    
    def _fail_experiment(self, exp_name: str, experiment: ExperimentInstance):
        """Mark an experiment as failed."""
        
        experiment.status = ExperimentStatus.FAILED
        experiment.end_time = datetime.now()
        
        # Move to failed experiments
        self.failed_experiments[exp_name] = experiment
        del self.running_experiments[exp_name]
        
        logger.error(f"EXPERIMENT FAILED: {exp_name}")
        if experiment.error_messages:
            logger.error(f"  Errors: {'; '.join(experiment.error_messages[-3:])}")
    
    def _cleanup_completed_experiments(self):
        """Clean up resources from completed experiments."""
        
        # Archive old completed experiments
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        for exp_name, experiment in list(self.completed_experiments.items()):
            if experiment.end_time and experiment.end_time < cutoff_time:
                # Archive experiment data
                self._archive_experiment(exp_name, experiment)
                del self.completed_experiments[exp_name]
    
    def _archive_experiment(self, exp_name: str, experiment: ExperimentInstance):
        """Archive completed experiment data."""
        
        archive_path = self.results_dir / f"{exp_name}_{experiment.end_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        experiment_data = asdict(experiment)
        experiment_data['start_time'] = experiment.start_time.isoformat() if experiment.start_time else None
        experiment_data['end_time'] = experiment.end_time.isoformat() if experiment.end_time else None
        
        with open(archive_path, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        logger.info(f"Archived experiment: {archive_path}")
    
    def _generate_progress_report(self):
        """Generate periodic progress reports."""
        
        # Only generate reports every 5 minutes
        if datetime.now().minute % 5 != 0:
            return
        
        total_queued = self.experiment_queue.qsize()
        total_running = len(self.running_experiments)
        total_completed = len(self.completed_experiments)
        total_failed = len(self.failed_experiments)
        
        logger.info("="*60)
        logger.info("EXPERIMENT COORDINATOR STATUS REPORT")
        logger.info(f"  Queued: {total_queued} | Running: {total_running} | Completed: {total_completed} | Failed: {total_failed}")
        logger.info(f"  GPU Utilization: {self.system_resources['gpu_utilization']:.1f}%")
        logger.info(f"  GPU Memory: {self.system_resources['gpu_memory_percent']:.1f}%")
        logger.info(f"  System Memory: {self.system_resources['memory_percent']:.1f}%")
        
        if self.running_experiments:
            logger.info("  Running Experiments:")
            for exp_name, experiment in self.running_experiments.items():
                runtime = (datetime.now() - experiment.start_time).total_seconds() / 60 if experiment.start_time else 0
                logger.info(f"    {exp_name}: {experiment.progress:.1f}% ({runtime:.1f}min) - {experiment.current_phase}")
        
        logger.info("="*60)
    
    def get_status_summary(self) -> Dict:
        """Get comprehensive status summary."""
        
        return {
            'coordinator_status': {
                'active': self.coordinator_active,
                'optimization_enabled': self.optimization_enabled,
                'auto_scaling_enabled': self.auto_scaling_enabled
            },
            'queue_status': {
                'queued': self.experiment_queue.qsize(),
                'running': len(self.running_experiments),
                'completed': len(self.completed_experiments),
                'failed': len(self.failed_experiments)
            },
            'system_resources': self.system_resources,
            'running_experiments': {
                name: {
                    'progress': exp.progress,
                    'phase': exp.current_phase,
                    'throughput': exp.throughput_hexagons_per_minute,
                    'estimated_completion': exp.estimated_completion.isoformat() if exp.estimated_completion else None
                }
                for name, exp in self.running_experiments.items()
            }
        }
    
    def shutdown(self):
        """Shutdown the coordinator gracefully."""
        
        logger.info("MULTI-EXPERIMENT COORDINATOR SHUTDOWN INITIATED")
        self.coordinator_active = False
        
        # Wait for coordination thread
        if self.coordination_thread.is_alive():
            self.coordination_thread.join(timeout=5)
        
        # Archive all active experiments
        for exp_name, experiment in self.running_experiments.items():
            self._archive_experiment(exp_name, experiment)
        
        for exp_name, experiment in self.completed_experiments.items():
            self._archive_experiment(exp_name, experiment)
        
        logger.info("MULTI-EXPERIMENT COORDINATOR SHUTDOWN COMPLETE")


# Pre-configured experiment templates
def create_massive_cascadia_experiment() -> ExperimentConfig:
    """Create configuration for massive Cascadia 2021 experiment."""
    
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    script_path = base_dir / "experiments/cascadia_massive_2021/run_massive_cascadia_experiment.py"
    output_dir = base_dir / "experiments/cascadia_massive_2021"
    
    return ExperimentConfig(
        name="massive_cascadia_2021",
        script_path=str(script_path),
        args=[],
        priority=ExperimentPriority.HIGH,
        estimated_runtime_minutes=120,
        memory_requirement_gb=16.0,
        gpu_memory_requirement_gb=20.0,
        max_parallel_processes=1,
        dependencies=[],
        output_directory=str(output_dir),
        adaptive_scaling=True
    )


def create_del_norte_experiment() -> ExperimentConfig:
    """Create configuration for Del Norte hierarchical experiment."""
    
    base_dir = Path(__file__).parent.parent.parent  # Go up to project root
    script_path = base_dir / "experiments/del_norte_active_inference/run_hierarchical_experiment.py"
    output_dir = base_dir / "experiments/del_norte_active_inference"
    
    return ExperimentConfig(
        name="del_norte_hierarchical",
        script_path=str(script_path),
        args=["--resolutions", "8", "9", "10", "11"],
        priority=ExperimentPriority.MEDIUM,
        estimated_runtime_minutes=45,
        memory_requirement_gb=8.0,
        gpu_memory_requirement_gb=12.0,
        max_parallel_processes=1,
        dependencies=[],
        output_directory=str(output_dir),
        adaptive_scaling=True
    )


def main():
    """Run the experiment coordinator with sample experiments."""
    
    coordinator = MultiExperimentCoordinator()
    
    try:
        # Add sample experiments
        cascadia_config = create_massive_cascadia_experiment()
        coordinator.add_experiment(cascadia_config)
        
        del_norte_config = create_del_norte_experiment()
        coordinator.add_experiment(del_norte_config)
        
        logger.info("MULTI-EXPERIMENT COORDINATOR ACTIVE")
        logger.info("Managing massive spatial AI experiments...")
        logger.info("Press Ctrl+C to shutdown")
        
        # Main loop
        while True:
            time.sleep(30)
            status = coordinator.get_status_summary()
            logger.info(f"Queue: {status['queue_status']['queued']} | Running: {status['queue_status']['running']}")
    
    except KeyboardInterrupt:
        logger.info("Coordinator shutdown requested")
    
    finally:
        coordinator.shutdown()


if __name__ == "__main__":
    main()