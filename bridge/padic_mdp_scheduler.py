#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P-adic Time Markov Decision Process Scheduler for Distributed TIFF Processing

This implements a continuous decision process operating in p-adic time space,
where scheduling decisions occur across multiple temporal scales simultaneously.
The MDP operates in decision regions within the parameter space of resource allocation.

Mathematical Framework:
- State Space: S = {worker_states, queue_states, resource_states} ⊆ Q_p^n
- Action Space: A = {assign_task, reallocate, pause, resume} ⊆ Q_p^m  
- Transition Function: T: S × A → Δ(S) in p-adic topology
- Reward Function: R: S × A → Q_p (p-adic valued rewards)
- Policy: π: S → A optimized via p-adic reinforcement learning

P-adic Time Structure:
- Simultaneous processing at multiple temporal resolutions
- Ultra-metric decision distances
- Non-Archimedean optimization landscape
"""

import asyncio
import numpy as np
import pandas as pd
import torch
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import websockets
# import aioredis  # Optional dependency for Redis coordination
from fastapi import FastAPI, WebSocket, HTTPException
import uvicorn

# P-adic number implementations
from fractions import Fraction
import sympy as sp

logger = logging.getLogger(__name__)


class PadicTime:
    """
    P-adic time implementation for continuous decision processes.
    
    In p-adic time, events can occur simultaneously at multiple scales,
    enabling ultra-metric decision making where "nearby" decisions in
    p-adic distance can be vastly different in traditional metrics.
    """
    
    def __init__(self, prime: int = 2, precision: int = 10):
        """Initialize p-adic time with given prime and precision"""
        self.prime = prime
        self.precision = precision
        self.current_time = PadicNumber(0, prime, precision)
        self.decision_epochs = []
        
    def advance(self, delta: Union[int, float, 'PadicNumber']) -> 'PadicNumber':
        """Advance p-adic time by delta"""
        if isinstance(delta, (int, float)):
            delta = PadicNumber(delta, self.prime, self.precision)
        
        self.current_time = self.current_time + delta
        self.decision_epochs.append(self.current_time.copy())
        return self.current_time
    
    def get_decision_scale(self, action: str) -> int:
        """Get the p-adic scale for a given decision type"""
        scales = {
            'assign_immediate': 0,    # Scale 0: immediate decisions
            'assign_batch': 1,        # Scale 1: batch-level decisions  
            'reallocate': 2,          # Scale 2: worker reallocation
            'global_optimize': 3,     # Scale 3: global optimization
            'meta_learning': 4        # Scale 4: meta-parameter updates
        }
        return scales.get(action, 0)
    
    def synchronize_scales(self, scales: List[int]) -> List['PadicNumber']:
        """Synchronize decisions across multiple p-adic scales"""
        synchronized_times = []
        for scale in scales:
            # Time at this scale
            scale_time = self.current_time * (self.prime ** scale)
            synchronized_times.append(scale_time)
        return synchronized_times


class PadicNumber:
    """P-adic number implementation for ultra-metric decision spaces"""
    
    def __init__(self, value: Union[int, float, Fraction], prime: int = 2, precision: int = 10):
        self.prime = prime
        self.precision = precision
        
        if isinstance(value, (int, float)):
            self.coefficients = self._to_padic_expansion(Fraction(value).limit_denominator())
        elif isinstance(value, Fraction):
            self.coefficients = self._to_padic_expansion(value)
        else:
            raise ValueError(f"Unsupported type for p-adic number: {type(value)}")
    
    def _to_padic_expansion(self, rational: Fraction) -> List[int]:
        """Convert rational number to p-adic expansion"""
        if rational == 0:
            return [0] * self.precision
        
        # Hensel lifting for p-adic expansion
        num, den = rational.numerator, rational.denominator
        coeffs = []
        
        # Handle negative numbers
        if num < 0:
            num = -num
            # In p-adics, -1 = p^n - 1 for large enough n
            coeffs = [self.prime - 1] * self.precision
            return coeffs
        
        # Standard p-adic expansion
        for i in range(self.precision):
            if den % self.prime != 0:
                # Multiplicative inverse modulo p
                inv_den = pow(den, -1, self.prime)
                coeff = (num * inv_den) % self.prime
                coeffs.append(coeff)
                num = (num - coeff * den) // self.prime
            else:
                coeffs.append(0)
                num //= self.prime
            
            if num == 0:
                break
        
        # Pad with zeros
        while len(coeffs) < self.precision:
            coeffs.append(0)
            
        return coeffs[:self.precision]
    
    def __add__(self, other: 'PadicNumber') -> 'PadicNumber':
        """P-adic addition"""
        result = PadicNumber(0, self.prime, self.precision)
        carry = 0
        
        for i in range(self.precision):
            sum_digit = self.coefficients[i] + other.coefficients[i] + carry
            result.coefficients[i] = sum_digit % self.prime
            carry = sum_digit // self.prime
        
        return result
    
    def __mul__(self, scalar: Union[int, float]) -> 'PadicNumber':
        """Scalar multiplication in p-adic"""
        if isinstance(scalar, (int, float)):
            scalar_padic = PadicNumber(scalar, self.prime, self.precision)
            return self._multiply_padic(scalar_padic)
        return self._multiply_padic(scalar)
    
    def _multiply_padic(self, other: 'PadicNumber') -> 'PadicNumber':
        """P-adic multiplication"""
        result = PadicNumber(0, self.prime, self.precision)
        
        for i in range(self.precision):
            for j in range(self.precision - i):
                if i + j < self.precision:
                    prod = self.coefficients[i] * other.coefficients[j]
                    result.coefficients[i + j] = (result.coefficients[i + j] + prod) % self.prime
        
        return result
    
    def distance(self, other: 'PadicNumber') -> float:
        """Ultra-metric p-adic distance"""
        diff_coeffs = []
        borrow = 0
        
        for i in range(self.precision):
            diff = self.coefficients[i] - other.coefficients[i] - borrow
            if diff < 0:
                diff += self.prime
                borrow = 1
            else:
                borrow = 0
            diff_coeffs.append(diff)
        
        # Find first non-zero coefficient
        for i, coeff in enumerate(diff_coeffs):
            if coeff != 0:
                return self.prime ** (-i)
        
        return 0.0  # Numbers are equal
    
    def copy(self) -> 'PadicNumber':
        """Create a copy of this p-adic number"""
        result = PadicNumber(0, self.prime, self.precision)
        result.coefficients = self.coefficients.copy()
        return result
    
    def to_rational_approx(self) -> float:
        """Convert to rational approximation for display"""
        value = 0.0
        for i, coeff in enumerate(self.coefficients):
            value += coeff * (self.prime ** i)
        return value


@dataclass
class WorkerState:
    """State representation for a worker in p-adic MDP"""
    worker_id: int
    current_task: Optional[str] = None
    queue_size: int = 0
    processing_rate: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    last_update: PadicNumber = field(default_factory=lambda: PadicNumber(0))
    decision_region: str = "idle"
    performance_history: List[float] = field(default_factory=list)


@dataclass 
class MDPState:
    """Complete MDP state in p-adic time"""
    workers: Dict[int, WorkerState]
    global_queue: List[str]
    resource_availability: Dict[str, float]
    padic_time: PadicNumber
    decision_region: str
    reward_accumulated: PadicNumber = field(default_factory=lambda: PadicNumber(0))


class DecisionRegion(Enum):
    """Decision regions in parameter space"""
    EXPLORATION = "exploration"        # High uncertainty, explore actions
    EXPLOITATION = "exploitation"      # Known good regions, exploit
    LOAD_BALANCING = "load_balancing"  # Balance worker loads
    EMERGENCY = "emergency"            # System overload, emergency actions
    OPTIMIZATION = "optimization"      # Fine-tune performance
    LEARNING = "learning"             # Update MDP parameters


class PadicMDPScheduler:
    """
    P-adic time Markov Decision Process Scheduler
    
    Operates in ultra-metric decision space where scheduling decisions
    occur simultaneously across multiple temporal scales.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.padic_time = PadicTime(prime=config.get('padic_prime', 2))
        
        # MDP Components
        self.state = MDPState(
            workers={},
            global_queue=[],
            resource_availability={'cpu': 1.0, 'gpu': 1.0, 'memory': 1.0},
            padic_time=self.padic_time.current_time.copy(),
            decision_region=DecisionRegion.EXPLORATION.value
        )
        
        # Decision regions in parameter space
        self.decision_regions = {
            DecisionRegion.EXPLORATION: self._exploration_policy,
            DecisionRegion.EXPLOITATION: self._exploitation_policy, 
            DecisionRegion.LOAD_BALANCING: self._load_balancing_policy,
            DecisionRegion.EMERGENCY: self._emergency_policy,
            DecisionRegion.OPTIMIZATION: self._optimization_policy,
            DecisionRegion.LEARNING: self._learning_policy
        }
        
        # Reward function parameters (p-adic valued)
        self.reward_params = {
            'throughput_weight': PadicNumber(1.0, self.padic_time.prime),
            'efficiency_weight': PadicNumber(0.8, self.padic_time.prime), 
            'balance_weight': PadicNumber(0.6, self.padic_time.prime),
            'stability_weight': PadicNumber(0.4, self.padic_time.prime)
        }
        
        # Transition probabilities (learned via experience)
        self.transition_matrix = {}
        self.policy_network = self._initialize_policy_network()
        
        # Performance tracking
        self.episode_rewards = []
        self.decision_history = []
        
        # Communication channels
        self.worker_queues = {}
        self.result_queue = mp.Queue()
        self.command_queue = mp.Queue()
        
        # Active learning components
        self.exploration_rate = 0.1
        self.learning_rate = 0.01
        
        logger.info("P-adic MDP Scheduler initialized")
        logger.info(f"Prime: {self.padic_time.prime}, Precision: {self.padic_time.precision}")
    
    def _initialize_policy_network(self) -> torch.nn.Module:
        """Initialize neural network for policy approximation"""
        class PadicPolicyNet(torch.nn.Module):
            def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, action_dim),
                    torch.nn.Softmax(dim=-1)
                )
            
            def forward(self, state):
                return self.net(state)
        
        return PadicPolicyNet(state_dim=20, action_dim=6)  # 6 action types
    
    def encode_state(self, state: MDPState) -> torch.Tensor:
        """Encode MDP state as tensor for neural network"""
        features = []
        
        # Worker features
        for i in range(self.config.get('max_workers', 8)):
            if i in state.workers:
                worker = state.workers[i]
                features.extend([
                    worker.queue_size / 100.0,  # Normalized queue size
                    worker.processing_rate,
                    worker.memory_usage,
                    worker.gpu_usage
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])  # Inactive worker
        
        # Global state features
        features.extend([
            len(state.global_queue) / 1000.0,  # Normalized global queue
            state.resource_availability['cpu'],
            state.resource_availability['gpu'],
            state.resource_availability['memory'],
            state.padic_time.to_rational_approx() / 1000.0  # Normalized p-adic time
        ])
        
        # Pad to fixed length
        while len(features) < 20:
            features.append(0.0)
        
        return torch.tensor(features[:20], dtype=torch.float32)
    
    def select_action(self, state: MDPState) -> str:
        """Select action using policy network with p-adic exploration"""
        state_tensor = self.encode_state(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
        
        # P-adic exploration: perturb probabilities in ultra-metric space
        if np.random.random() < self.exploration_rate:
            # Ultra-metric exploration in p-adic space
            padic_noise = [PadicNumber(np.random.normal(0, 0.1), self.padic_time.prime) 
                          for _ in range(6)]
            noise_values = [pn.to_rational_approx() for pn in padic_noise]
            action_probs += torch.tensor(noise_values, dtype=torch.float32)
            action_probs = torch.softmax(action_probs, dim=-1)
        
        # Sample action
        action_idx = torch.multinomial(action_probs, 1).item()
        actions = ['assign_immediate', 'assign_batch', 'reallocate', 
                  'global_optimize', 'pause_worker', 'meta_learning']
        
        return actions[action_idx]
    
    def compute_reward(self, state: MDPState, action: str, next_state: MDPState) -> PadicNumber:
        """Compute p-adic valued reward for state transition"""
        
        # Throughput component
        throughput_improvement = sum(w.processing_rate for w in next_state.workers.values()) - \
                               sum(w.processing_rate for w in state.workers.values())
        throughput_reward = PadicNumber(throughput_improvement, self.padic_time.prime)
        
        # Load balancing component  
        if len(next_state.workers) > 1:
            rates = [w.processing_rate for w in next_state.workers.values()]
            balance_penalty = np.std(rates) if len(rates) > 1 else 0
            balance_reward = PadicNumber(-balance_penalty, self.padic_time.prime)
        else:
            balance_reward = PadicNumber(0, self.padic_time.prime)
        
        # Resource efficiency
        resource_usage = sum(next_state.resource_availability.values()) / 3
        efficiency_reward = PadicNumber(resource_usage, self.padic_time.prime)
        
        # Queue reduction reward
        queue_improvement = len(state.global_queue) - len(next_state.global_queue)
        queue_reward = PadicNumber(queue_improvement / 10.0, self.padic_time.prime)
        
        # Combine rewards with p-adic weights
        total_reward = (self.reward_params['throughput_weight'] * throughput_reward +
                       self.reward_params['efficiency_weight'] * efficiency_reward +
                       self.reward_params['balance_weight'] * balance_reward +
                       PadicNumber(1.0, self.padic_time.prime) * queue_reward)
        
        return total_reward
    
    def update_decision_region(self, state: MDPState) -> str:
        """Determine current decision region based on system state"""
        
        # Calculate system metrics
        total_queue_size = len(state.global_queue) + sum(w.queue_size for w in state.workers.values())
        avg_cpu_usage = state.resource_availability.get('cpu', 0)
        worker_load_variance = 0
        
        if len(state.workers) > 1:
            loads = [w.queue_size + w.processing_rate for w in state.workers.values()]
            worker_load_variance = np.var(loads)
        
        # Decision region logic with p-adic distances
        if total_queue_size > 500:  # Emergency: system overloaded
            return DecisionRegion.EMERGENCY.value
        elif worker_load_variance > 10:  # Imbalanced load
            return DecisionRegion.LOAD_BALANCING.value
        elif avg_cpu_usage > 0.9:  # High utilization, exploit current strategy
            return DecisionRegion.EXPLOITATION.value
        elif len(self.decision_history) < 100:  # Early phase, explore
            return DecisionRegion.EXPLORATION.value
        elif len(self.decision_history) % 50 == 0:  # Periodic learning
            return DecisionRegion.LEARNING.value
        else:  # Default optimization
            return DecisionRegion.OPTIMIZATION.value
    
    async def decision_loop(self):
        """Main decision loop operating in p-adic time"""
        logger.info("Starting p-adic MDP decision loop")
        
        while True:
            try:
                # Advance p-adic time
                self.padic_time.advance(PadicNumber(1, self.padic_time.prime))
                self.state.padic_time = self.padic_time.current_time.copy()
                
                # Update decision region
                old_region = self.state.decision_region
                self.state.decision_region = self.update_decision_region(self.state)
                
                # Select and execute action based on current policy
                action = self.select_action(self.state)
                next_state = await self.execute_action(action)
                
                # Compute reward and update policy
                reward = self.compute_reward(self.state, action, next_state)
                self.state.reward_accumulated = self.state.reward_accumulated + reward
                
                # Store experience for learning
                experience = {
                    'state': self.encode_state(self.state),
                    'action': action,
                    'reward': reward.to_rational_approx(),
                    'next_state': self.encode_state(next_state),
                    'padic_time': self.state.padic_time.to_rational_approx(),
                    'decision_region': self.state.decision_region
                }
                self.decision_history.append(experience)
                
                # Update state
                self.state = next_state
                
                # Periodic learning updates
                if len(self.decision_history) % 10 == 0:
                    await self.update_policy()
                
                # Log decision
                logger.info(f"P-adic time: {self.state.padic_time.to_rational_approx():.3f}, "
                          f"Action: {action}, Region: {self.state.decision_region}, "
                          f"Reward: {reward.to_rational_approx():.3f}")
                
                # Decision frequency based on p-adic scale
                decision_scale = self.padic_time.get_decision_scale(action)
                sleep_time = 0.1 * (self.padic_time.prime ** decision_scale)  # Scale-dependent timing
                await asyncio.sleep(min(sleep_time, 2.0))  # Cap sleep time
                
            except Exception as e:
                logger.error(f"Error in decision loop: {e}")
                await asyncio.sleep(1.0)
    
    async def execute_action(self, action: str) -> MDPState:
        """Execute selected action and return new state"""
        
        if action == 'assign_immediate':
            return await self._assign_immediate()
        elif action == 'assign_batch':
            return await self._assign_batch()
        elif action == 'reallocate':
            return await self._reallocate_workers()
        elif action == 'global_optimize':
            return await self._global_optimize()
        elif action == 'pause_worker':
            return await self._pause_worker()
        elif action == 'meta_learning':
            return await self._meta_learning_update()
        else:
            return self.state  # No-op
    
    async def _assign_immediate(self) -> MDPState:
        """Assign next task immediately to best available worker"""
        if not self.state.global_queue:
            return self.state
        
        # Find worker with lowest load
        best_worker_id = min(self.state.workers.keys(), 
                           key=lambda w: self.state.workers[w].queue_size,
                           default=None)
        
        if best_worker_id is not None:
            task = self.state.global_queue.pop(0)
            self.state.workers[best_worker_id].queue_size += 1
            self.state.workers[best_worker_id].current_task = task
            
            # Send task to worker
            await self.send_task_to_worker(best_worker_id, task)
        
        return self.state
    
    async def _assign_batch(self) -> MDPState:
        """Assign batch of tasks to workers"""
        batch_size = min(len(self.state.global_queue), 
                        len(self.state.workers) * 2)
        
        if batch_size == 0:
            return self.state
        
        # Distribute tasks evenly
        worker_ids = list(self.state.workers.keys())
        for i in range(batch_size):
            if self.state.global_queue:
                worker_id = worker_ids[i % len(worker_ids)]
                task = self.state.global_queue.pop(0)
                self.state.workers[worker_id].queue_size += 1
                await self.send_task_to_worker(worker_id, task)
        
        return self.state
    
    async def _reallocate_workers(self) -> MDPState:
        """Reallocate tasks between workers for load balancing"""
        if len(self.state.workers) < 2:
            return self.state
        
        # Find most and least loaded workers
        workers_by_load = sorted(self.state.workers.items(), 
                               key=lambda x: x[1].queue_size)
        
        if len(workers_by_load) >= 2:
            least_loaded_id, least_loaded = workers_by_load[0]
            most_loaded_id, most_loaded = workers_by_load[-1]
            
            load_diff = most_loaded.queue_size - least_loaded.queue_size
            if load_diff > 2:  # Significant imbalance
                # Transfer some load
                transfer_amount = load_diff // 2
                most_loaded.queue_size -= transfer_amount
                least_loaded.queue_size += transfer_amount
                
                logger.info(f"Reallocated {transfer_amount} tasks from worker "
                          f"{most_loaded_id} to worker {least_loaded_id}")
        
        return self.state
    
    async def _global_optimize(self) -> MDPState:
        """Global optimization of worker parameters"""
        # Adjust processing parameters based on performance history
        for worker in self.state.workers.values():
            if len(worker.performance_history) > 5:
                recent_perf = worker.performance_history[-5:]
                if np.mean(recent_perf) < 0.5:  # Poor performance
                    # Reduce queue size to allow catch-up
                    worker.queue_size = max(0, worker.queue_size - 1)
        
        return self.state
    
    async def _pause_worker(self) -> MDPState:
        """Pause least efficient worker"""
        if len(self.state.workers) <= 1:
            return self.state
        
        # Find least efficient worker
        worst_worker_id = min(self.state.workers.keys(),
                            key=lambda w: self.state.workers[w].processing_rate)
        
        # Redistribute its tasks
        tasks_to_redistribute = self.state.workers[worst_worker_id].queue_size
        self.state.global_queue.extend([f"redistributed_task_{i}" 
                                       for i in range(tasks_to_redistribute)])
        
        # Pause worker (remove from active workers)
        del self.state.workers[worst_worker_id]
        
        logger.info(f"Paused worker {worst_worker_id}, redistributed {tasks_to_redistribute} tasks")
        return self.state
    
    async def _meta_learning_update(self) -> MDPState:
        """Update meta-parameters of the MDP"""
        if len(self.decision_history) > 50:
            # Analyze recent performance
            recent_rewards = [exp['reward'] for exp in self.decision_history[-50:]]
            avg_reward = np.mean(recent_rewards)
            
            # Adjust exploration rate
            if avg_reward < 0:  # Poor performance, increase exploration
                self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
            else:  # Good performance, reduce exploration
                self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
            
            # Adjust reward weights based on what's working
            region_rewards = {}
            for exp in self.decision_history[-50:]:
                region = exp['decision_region']
                if region not in region_rewards:
                    region_rewards[region] = []
                region_rewards[region].append(exp['reward'])
            
            # Update p-adic reward weights
            for region, rewards in region_rewards.items():
                if len(rewards) > 5:
                    avg_region_reward = np.mean(rewards)
                    if avg_region_reward > 0.5:
                        # Increase weight for successful regions
                        weight_key = f'{region}_weight'
                        if weight_key in self.reward_params:
                            current = self.reward_params[weight_key].to_rational_approx()
                            new_weight = min(2.0, current * 1.05)
                            self.reward_params[weight_key] = PadicNumber(new_weight, self.padic_time.prime)
        
        return self.state
    
    async def update_policy(self):
        """Update policy network using recent experiences"""
        if len(self.decision_history) < 10:
            return
        
        # Prepare training data
        recent_experiences = self.decision_history[-10:]
        states = torch.stack([exp['state'] for exp in recent_experiences])
        rewards = torch.tensor([exp['reward'] for exp in recent_experiences], dtype=torch.float32)
        
        # Simple policy gradient update
        self.policy_network.train()
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        action_probs = self.policy_network(states)
        
        # Convert actions to indices
        action_to_idx = {'assign_immediate': 0, 'assign_batch': 1, 'reallocate': 2,
                        'global_optimize': 3, 'pause_worker': 4, 'meta_learning': 5}
        action_indices = torch.tensor([action_to_idx.get(exp['action'], 0) 
                                     for exp in recent_experiences])
        
        # Policy gradient loss
        log_probs = torch.log(action_probs.gather(1, action_indices.unsqueeze(1)))
        loss = -(log_probs.squeeze() * rewards).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.policy_network.eval()
    
    async def send_task_to_worker(self, worker_id: int, task: str):
        """Send task to specific worker"""
        if worker_id in self.worker_queues:
            try:
                self.worker_queues[worker_id].put_nowait({
                    'task': task,
                    'timestamp': self.padic_time.current_time.to_rational_approx(),
                    'priority': 'normal'
                })
            except:
                logger.warning(f"Failed to send task to worker {worker_id}")
    
    def add_worker(self, worker_id: int):
        """Add new worker to the system"""
        self.state.workers[worker_id] = WorkerState(
            worker_id=worker_id,
            last_update=self.padic_time.current_time.copy()
        )
        self.worker_queues[worker_id] = mp.Queue(maxsize=100)
        logger.info(f"Added worker {worker_id} to MDP system")
    
    def add_tasks(self, tasks: List[str]):
        """Add tasks to global queue"""
        self.state.global_queue.extend(tasks)
        logger.info(f"Added {len(tasks)} tasks to global queue")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for monitoring"""
        return {
            'padic_time': self.state.padic_time.to_rational_approx(),
            'decision_region': self.state.decision_region,
            'num_workers': len(self.state.workers),
            'global_queue_size': len(self.state.global_queue),
            'total_worker_queue': sum(w.queue_size for w in self.state.workers.values()),
            'avg_processing_rate': np.mean([w.processing_rate for w in self.state.workers.values()]) 
                                 if self.state.workers else 0,
            'accumulated_reward': self.state.reward_accumulated.to_rational_approx(),
            'exploration_rate': self.exploration_rate,
            'recent_decisions': len(self.decision_history),
            'resource_availability': self.state.resource_availability
        }


# Policy implementations for different decision regions
def _exploration_policy(self, state: MDPState) -> str:
    """Exploration policy: try diverse actions"""
    actions = ['assign_immediate', 'assign_batch', 'reallocate', 'global_optimize']
    return np.random.choice(actions)

def _exploitation_policy(self, state: MDPState) -> str:
    """Exploitation policy: use known good actions"""
    # Prefer immediate assignment when queue is large
    if len(state.global_queue) > 50:
        return 'assign_batch'
    else:
        return 'assign_immediate'

def _load_balancing_policy(self, state: MDPState) -> str:
    """Load balancing policy: focus on worker balance"""
    return 'reallocate'

def _emergency_policy(self, state: MDPState) -> str:
    """Emergency policy: rapid task assignment"""
    return 'assign_batch'

def _optimization_policy(self, state: MDPState) -> str:
    """Optimization policy: fine-tune performance"""
    return 'global_optimize'

def _learning_policy(self, state: MDPState) -> str:
    """Learning policy: update meta-parameters"""
    return 'meta_learning'


# Bind methods to scheduler class
PadicMDPScheduler._exploration_policy = _exploration_policy
PadicMDPScheduler._exploitation_policy = _exploitation_policy
PadicMDPScheduler._load_balancing_policy = _load_balancing_policy
PadicMDPScheduler._emergency_policy = _emergency_policy
PadicMDPScheduler._optimization_policy = _optimization_policy
PadicMDPScheduler._learning_policy = _learning_policy


async def main():
    """Main entry point for p-adic MDP scheduler"""
    config = {
        'padic_prime': 2,
        'max_workers': 8,
        'decision_frequency': 1.0,
        'learning_rate': 0.01,
        'exploration_decay': 0.99
    }
    
    scheduler = PadicMDPScheduler(config)
    
    # Add some initial workers
    for i in range(4):
        scheduler.add_worker(i)
    
    # Add initial tasks (TIFF files)
    tiff_files = [f"task_{i}.tif" for i in range(100)]
    scheduler.add_tasks(tiff_files)
    
    # Start decision loop
    await scheduler.decision_loop()


if __name__ == "__main__":
    asyncio.run(main())