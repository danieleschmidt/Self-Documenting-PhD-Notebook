"""
Distributed Research Computing Framework
========================================

Advanced distributed computing system for scaling research operations
across multiple nodes, cloud platforms, and heterogeneous environments.

Features:
- Distributed task execution
- Load balancing and auto-scaling
- Fault tolerance and resilience
- Cross-platform deployment
- Resource optimization
- Real-time performance monitoring
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, deque
import pickle
import gzip
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


class NodeType(Enum):
    """Types of computing nodes."""
    LOCAL = "local"
    CLOUD = "cloud"
    HPC = "hpc"
    EDGE = "edge"
    HYBRID = "hybrid"


@dataclass
class ResourceSpecification:
    """Resource requirements specification."""
    cpu_cores: int = 1
    memory_mb: int = 1024
    gpu_count: int = 0
    storage_mb: int = 1024
    network_mbps: int = 100
    max_runtime_minutes: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_cores': self.cpu_cores,
            'memory_mb': self.memory_mb,
            'gpu_count': self.gpu_count,
            'storage_mb': self.storage_mb,
            'network_mbps': self.network_mbps,
            'max_runtime_minutes': self.max_runtime_minutes
        }


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    id: str
    node_type: NodeType
    available_resources: ResourceSpecification
    current_load: float = 0.0
    status: str = "available"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    performance_score: float = 1.0
    reliability_score: float = 1.0
    
    def can_handle_task(self, requirements: ResourceSpecification) -> bool:
        """Check if node can handle a task with given requirements."""
        return (
            self.available_resources.cpu_cores >= requirements.cpu_cores and
            self.available_resources.memory_mb >= requirements.memory_mb and
            self.available_resources.gpu_count >= requirements.gpu_count and
            self.available_resources.storage_mb >= requirements.storage_mb and
            self.current_load < 0.8  # Don't overload nodes
        )
    
    def get_suitability_score(self, requirements: ResourceSpecification) -> float:
        """Calculate how suitable this node is for a task."""
        if not self.can_handle_task(requirements):
            return 0.0
        
        # Base score from performance and reliability
        base_score = (self.performance_score + self.reliability_score) / 2
        
        # Bonus for resource abundance
        cpu_ratio = self.available_resources.cpu_cores / max(requirements.cpu_cores, 1)
        memory_ratio = self.available_resources.memory_mb / max(requirements.memory_mb, 1)
        
        resource_bonus = min((cpu_ratio + memory_ratio) / 2, 2.0)
        
        # Penalty for current load
        load_penalty = 1 - (self.current_load / 2)
        
        return base_score * resource_bonus * load_penalty


@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    id: str
    function_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    requirements: ResourceSpecification = field(default_factory=ResourceSpecification)
    priority: int = 5  # 1-10, higher is more important
    max_retries: int = 3
    timeout_seconds: int = 3600
    
    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    attempt_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            'id': self.id,
            'function_name': self.function_name,
            'args': self.args,
            'kwargs': self.kwargs,
            'requirements': self.requirements.to_dict(),
            'priority': self.priority,
            'max_retries': self.max_retries,
            'timeout_seconds': self.timeout_seconds,
            'status': self.status.value,
            'assigned_node': self.assigned_node,
            'attempt_count': self.attempt_count,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error
        }


class TaskScheduler:
    """Intelligent task scheduler for distributed computing."""
    
    def __init__(self, scheduling_algorithm: str = "priority_first"):
        self.algorithm = scheduling_algorithm
        self.task_queue: deque = deque()
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.nodes: Dict[str, ComputeNode] = {}
        
        self.scheduling_metrics = {
            'tasks_scheduled': 0,
            'successful_completions': 0,
            'failed_tasks': 0,
            'average_scheduling_time': 0.0,
            'average_execution_time': 0.0
        }
    
    def add_node(self, node: ComputeNode):
        """Add a compute node to the scheduler."""
        self.nodes[node.id] = node
        logger.info(f"Added compute node: {node.id} ({node.node_type.value})")
    
    def remove_node(self, node_id: str):
        """Remove a compute node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            # Reschedule any tasks running on this node
            self._reschedule_tasks_from_node(node_id)
            logger.info(f"Removed compute node: {node_id}")
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for execution."""
        task.status = TaskStatus.QUEUED
        self.task_queue.append(task)
        self.scheduling_metrics['tasks_scheduled'] += 1
        logger.info(f"Queued task: {task.id}")
        return task.id
    
    def schedule_next_task(self) -> Optional[DistributedTask]:
        """Schedule the next task based on algorithm and available resources."""
        if not self.task_queue or not self.nodes:
            return None
        
        # Sort tasks by scheduling algorithm
        if self.algorithm == "priority_first":
            sorted_tasks = sorted(self.task_queue, key=lambda t: (-t.priority, t.created_at))
        elif self.algorithm == "shortest_job_first":
            sorted_tasks = sorted(self.task_queue, key=lambda t: t.requirements.cpu_cores)
        elif self.algorithm == "fifo":
            sorted_tasks = list(self.task_queue)
        else:
            sorted_tasks = sorted(self.task_queue, key=lambda t: (-t.priority, t.created_at))
        
        # Find best task-node match
        for task in sorted_tasks:
            best_node = self._find_best_node(task)
            if best_node:
                # Remove from queue and assign
                self.task_queue.remove(task)
                task.assigned_node = best_node.id
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                
                # Update node load
                best_node.current_load += self._calculate_task_load(task)
                
                self.running_tasks[task.id] = task
                logger.info(f"Scheduled task {task.id} on node {best_node.id}")
                return task
        
        return None
    
    def _find_best_node(self, task: DistributedTask) -> Optional[ComputeNode]:
        """Find the best available node for a task."""
        suitable_nodes = [
            node for node in self.nodes.values()
            if node.can_handle_task(task.requirements) and node.status == "available"
        ]
        
        if not suitable_nodes:
            return None
        
        # Sort by suitability score
        suitable_nodes.sort(key=lambda n: n.get_suitability_score(task.requirements), reverse=True)
        return suitable_nodes[0]
    
    def _calculate_task_load(self, task: DistributedTask) -> float:
        """Calculate the load a task will place on a node."""
        # Simple heuristic based on resource requirements
        cpu_load = task.requirements.cpu_cores * 0.3
        memory_load = task.requirements.memory_mb / 1024 * 0.2
        return min(cpu_load + memory_load, 1.0)
    
    def _reschedule_tasks_from_node(self, node_id: str):
        """Reschedule tasks from a failed node."""
        tasks_to_reschedule = [
            task for task in self.running_tasks.values()
            if task.assigned_node == node_id
        ]
        
        for task in tasks_to_reschedule:
            task.status = TaskStatus.QUEUED
            task.assigned_node = None
            task.attempt_count += 1
            
            if task.attempt_count <= task.max_retries:
                self.task_queue.appendleft(task)  # Priority reschedule
                del self.running_tasks[task.id]
                logger.info(f"Rescheduled task {task.id} (attempt {task.attempt_count})")
            else:
                task.status = TaskStatus.FAILED
                task.error = f"Exceeded maximum retries ({task.max_retries})"
                self.completed_tasks[task.id] = task
                del self.running_tasks[task.id]
                self.scheduling_metrics['failed_tasks'] += 1
    
    def complete_task(self, task_id: str, result: Any = None, error: str = None):
        """Mark a task as completed."""
        if task_id not in self.running_tasks:
            return
        
        task = self.running_tasks[task_id]
        task.completed_at = datetime.now()
        task.result = result
        task.error = error
        
        if error:
            task.status = TaskStatus.FAILED
            self.scheduling_metrics['failed_tasks'] += 1
        else:
            task.status = TaskStatus.COMPLETED
            self.scheduling_metrics['successful_completions'] += 1
        
        # Free up node resources
        if task.assigned_node and task.assigned_node in self.nodes:
            node = self.nodes[task.assigned_node]
            node.current_load = max(0, node.current_load - self._calculate_task_load(task))
        
        # Move to completed tasks
        self.completed_tasks[task_id] = task
        del self.running_tasks[task_id]
        
        # Update metrics
        if task.started_at:
            execution_time = (task.completed_at - task.started_at).total_seconds()
            current_avg = self.scheduling_metrics['average_execution_time']
            completed_count = self.scheduling_metrics['successful_completions'] + self.scheduling_metrics['failed_tasks']
            self.scheduling_metrics['average_execution_time'] = (current_avg * (completed_count - 1) + execution_time) / completed_count
        
        logger.info(f"Completed task {task_id}: {task.status.value}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'queued_tasks': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'active_nodes': len([n for n in self.nodes.values() if n.status == "available"]),
            'total_nodes': len(self.nodes),
            'metrics': self.scheduling_metrics.copy(),
            'algorithm': self.algorithm
        }


class WorkerProcess:
    """Worker process for executing distributed tasks."""
    
    def __init__(self, node_id: str, functions: Dict[str, Callable]):
        self.node_id = node_id
        self.functions = functions
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
    def start(self):
        """Start the worker process."""
        self.running = True
        logger.info(f"Worker {self.node_id} started")
    
    def stop(self):
        """Stop the worker process."""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info(f"Worker {self.node_id} stopped")
    
    def execute_task(self, task: DistributedTask) -> Any:
        """Execute a distributed task."""
        if not self.running:
            raise RuntimeError("Worker is not running")
        
        if task.function_name not in self.functions:
            raise ValueError(f"Function '{task.function_name}' not available on this worker")
        
        function = self.functions[task.function_name]
        
        try:
            # Execute with timeout
            future = self.executor.submit(function, *task.args, **task.kwargs)
            result = future.result(timeout=task.timeout_seconds)
            return result
        except Exception as e:
            raise RuntimeError(f"Task execution failed: {str(e)}")


class LoadBalancer:
    """Intelligent load balancer for distributed research computing."""
    
    def __init__(self, balancing_strategy: str = "least_loaded"):
        self.strategy = balancing_strategy
        self.node_weights: Dict[str, float] = {}
        self.node_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def update_node_performance(self, node_id: str, execution_time: float, success: bool):
        """Update node performance metrics."""
        score = 1.0 / max(execution_time, 0.1) if success else 0.1
        self.node_history[node_id].append(score)
        
        # Calculate weighted average (recent performance matters more)
        history = list(self.node_history[node_id])
        if history:
            weights = [1.0 + i * 0.1 for i in range(len(history))]  # Recent scores weighted more
            weighted_sum = sum(score * weight for score, weight in zip(history, weights))
            weight_sum = sum(weights)
            self.node_weights[node_id] = weighted_sum / weight_sum
    
    def select_node(self, nodes: List[ComputeNode], task: DistributedTask) -> Optional[ComputeNode]:
        """Select the best node for a task based on balancing strategy."""
        eligible_nodes = [n for n in nodes if n.can_handle_task(task.requirements)]
        
        if not eligible_nodes:
            return None
        
        if self.strategy == "least_loaded":
            return min(eligible_nodes, key=lambda n: n.current_load)
        
        elif self.strategy == "weighted_round_robin":
            # Use performance-based weights
            weights = [self.node_weights.get(n.id, 1.0) for n in eligible_nodes]
            max_weight_idx = weights.index(max(weights))
            return eligible_nodes[max_weight_idx]
        
        elif self.strategy == "random":
            import random
            return random.choice(eligible_nodes)
        
        else:  # Default to least loaded
            return min(eligible_nodes, key=lambda n: n.current_load)


class AutoScaler:
    """Auto-scaling system for dynamic resource management."""
    
    def __init__(self, min_nodes: int = 1, max_nodes: int = 10):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.scaling_history: deque = deque(maxlen=100)
        self.last_scale_action: datetime = datetime.now()
        self.cooldown_minutes = 5  # Minimum time between scaling actions
        
    def should_scale_up(self, scheduler: TaskScheduler) -> bool:
        """Determine if we should add more nodes."""
        status = scheduler.get_scheduler_status()
        
        # Scale up if queue is backing up and nodes are loaded
        queue_pressure = status['queued_tasks'] > 5
        node_pressure = status['running_tasks'] / max(status['active_nodes'], 1) > 0.8
        
        # Check cooldown
        time_since_last_scale = datetime.now() - self.last_scale_action
        cooldown_expired = time_since_last_scale > timedelta(minutes=self.cooldown_minutes)
        
        can_scale = status['total_nodes'] < self.max_nodes
        
        return queue_pressure and node_pressure and cooldown_expired and can_scale
    
    def should_scale_down(self, scheduler: TaskScheduler) -> bool:
        """Determine if we should remove nodes."""
        status = scheduler.get_scheduler_status()
        
        # Scale down if queue is empty and nodes are underutilized
        no_queue_pressure = status['queued_tasks'] == 0
        low_utilization = status['running_tasks'] / max(status['active_nodes'], 1) < 0.3
        
        # Check cooldown
        time_since_last_scale = datetime.now() - self.last_scale_action
        cooldown_expired = time_since_last_scale > timedelta(minutes=self.cooldown_minutes)
        
        can_scale = status['total_nodes'] > self.min_nodes
        
        return no_queue_pressure and low_utilization and cooldown_expired and can_scale
    
    def execute_scaling_action(self, action: str, scheduler: TaskScheduler) -> bool:
        """Execute a scaling action."""
        if action == "scale_up":
            # Create a new virtual node (in real implementation, this would provision actual resources)
            new_node = ComputeNode(
                id=f"auto_node_{int(time.time())}",
                node_type=NodeType.CLOUD,
                available_resources=ResourceSpecification(
                    cpu_cores=4,
                    memory_mb=8192,
                    storage_mb=50000
                )
            )
            scheduler.add_node(new_node)
            self.last_scale_action = datetime.now()
            self.scaling_history.append(('scale_up', datetime.now()))
            logger.info(f"Auto-scaled up: added node {new_node.id}")
            return True
            
        elif action == "scale_down":
            # Find least utilized node to remove
            nodes_by_load = sorted(scheduler.nodes.values(), key=lambda n: n.current_load)
            for node in nodes_by_load:
                if node.current_load == 0 and node.id.startswith("auto_node_"):
                    scheduler.remove_node(node.id)
                    self.last_scale_action = datetime.now()
                    self.scaling_history.append(('scale_down', datetime.now()))
                    logger.info(f"Auto-scaled down: removed node {node.id}")
                    return True
        
        return False


class DistributedResearchFramework:
    """Main framework for distributed research computing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.scheduler = TaskScheduler(
            self.config.get('scheduling_algorithm', 'priority_first')
        )
        self.load_balancer = LoadBalancer(
            self.config.get('balancing_strategy', 'least_loaded')
        )
        self.auto_scaler = AutoScaler(
            self.config.get('min_nodes', 1),
            self.config.get('max_nodes', 10)
        )
        
        # Worker management
        self.workers: Dict[str, WorkerProcess] = {}
        self.available_functions: Dict[str, Callable] = {}
        
        # Control loop
        self.running = False
        self.control_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.performance_metrics = {
            'tasks_per_second': 0.0,
            'average_task_time': 0.0,
            'resource_utilization': 0.0,
            'scaling_actions': 0,
            'framework_uptime': 0.0
        }
        self.framework_start_time = datetime.now()
        
        logger.info("Distributed Research Framework initialized")
    
    def register_function(self, name: str, function: Callable):
        """Register a function for distributed execution."""
        self.available_functions[name] = function
        logger.info(f"Registered function: {name}")
    
    def add_node(self, node_config: Dict[str, Any]) -> str:
        """Add a compute node to the framework."""
        node = ComputeNode(
            id=node_config['id'],
            node_type=NodeType(node_config.get('type', 'local')),
            available_resources=ResourceSpecification(
                **node_config.get('resources', {})
            )
        )
        
        self.scheduler.add_node(node)
        
        # Create worker for this node
        worker = WorkerProcess(node.id, self.available_functions)
        self.workers[node.id] = worker
        worker.start()
        
        return node.id
    
    def submit_task(self, function_name: str, args: List[Any] = None, 
                   kwargs: Dict[str, Any] = None, 
                   requirements: ResourceSpecification = None,
                   priority: int = 5) -> str:
        """Submit a task for distributed execution."""
        task_id = hashlib.md5(
            f"{function_name}_{time.time()}_{id(self)}".encode()
        ).hexdigest()[:16]
        
        task = DistributedTask(
            id=task_id,
            function_name=function_name,
            args=args or [],
            kwargs=kwargs or {},
            requirements=requirements or ResourceSpecification(),
            priority=priority
        )
        
        return self.scheduler.submit_task(task)
    
    def start_framework(self):
        """Start the distributed framework."""
        if self.running:
            return
        
        self.running = True
        self.framework_start_time = datetime.now()
        
        # Start control loop
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info("Distributed Research Framework started")
    
    def stop_framework(self):
        """Stop the distributed framework."""
        self.running = False
        
        # Stop all workers
        for worker in self.workers.values():
            worker.stop()
        
        # Wait for control thread
        if self.control_thread:
            self.control_thread.join(timeout=5)
        
        logger.info("Distributed Research Framework stopped")
    
    def _control_loop(self):
        """Main control loop for the framework."""
        while self.running:
            try:
                # Schedule tasks
                scheduled_task = self.scheduler.schedule_next_task()
                if scheduled_task:
                    self._execute_task_async(scheduled_task)
                
                # Check auto-scaling
                if self.auto_scaler.should_scale_up(self.scheduler):
                    success = self.auto_scaler.execute_scaling_action("scale_up", self.scheduler)
                    if success:
                        self.performance_metrics['scaling_actions'] += 1
                
                elif self.auto_scaler.should_scale_down(self.scheduler):
                    success = self.auto_scaler.execute_scaling_action("scale_down", self.scheduler)
                    if success:
                        self.performance_metrics['scaling_actions'] += 1
                
                # Update performance metrics
                self._update_performance_metrics()
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
            
            time.sleep(1)  # Run every second
    
    def _execute_task_async(self, task: DistributedTask):
        """Execute a task asynchronously."""
        def execute_wrapper():
            try:
                worker = self.workers.get(task.assigned_node)
                if not worker:
                    raise RuntimeError(f"Worker for node {task.assigned_node} not found")
                
                result = worker.execute_task(task)
                self.scheduler.complete_task(task.id, result=result)
                
                # Update load balancer performance
                execution_time = (datetime.now() - task.started_at).total_seconds()
                self.load_balancer.update_node_performance(task.assigned_node, execution_time, True)
                
            except Exception as e:
                self.scheduler.complete_task(task.id, error=str(e))
                
                # Update load balancer with failure
                if task.assigned_node:
                    execution_time = (datetime.now() - task.started_at).total_seconds()
                    self.load_balancer.update_node_performance(task.assigned_node, execution_time, False)
        
        # Execute in thread pool to avoid blocking control loop
        threading.Thread(target=execute_wrapper, daemon=True).start()
    
    def _update_performance_metrics(self):
        """Update framework performance metrics."""
        status = self.scheduler.get_scheduler_status()
        
        # Calculate tasks per second
        uptime = (datetime.now() - self.framework_start_time).total_seconds()
        if uptime > 0:
            completed_tasks = status['metrics']['successful_completions'] + status['metrics']['failed_tasks']
            self.performance_metrics['tasks_per_second'] = completed_tasks / uptime
        
        # Update other metrics
        self.performance_metrics['average_task_time'] = status['metrics']['average_execution_time']
        self.performance_metrics['framework_uptime'] = uptime
        
        # Resource utilization
        if status['total_nodes'] > 0:
            self.performance_metrics['resource_utilization'] = status['running_tasks'] / status['total_nodes']
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get comprehensive framework status."""
        scheduler_status = self.scheduler.get_scheduler_status()
        
        return {
            'running': self.running,
            'framework_uptime': self.performance_metrics['framework_uptime'],
            'scheduler': scheduler_status,
            'performance_metrics': self.performance_metrics.copy(),
            'auto_scaling': {
                'min_nodes': self.auto_scaler.min_nodes,
                'max_nodes': self.auto_scaler.max_nodes,
                'scaling_actions': len(self.auto_scaler.scaling_history)
            },
            'registered_functions': list(self.available_functions.keys()),
            'active_workers': len([w for w in self.workers.values() if w.running])
        }
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed task."""
        if task_id in self.scheduler.completed_tasks:
            task = self.scheduler.completed_tasks[task_id]
            return {
                'id': task.id,
                'status': task.status.value,
                'result': task.result,
                'error': task.error,
                'execution_time': (task.completed_at - task.started_at).total_seconds() if task.started_at and task.completed_at else None
            }
        return None


# Example research functions for distributed execution
def analyze_large_dataset(data_chunk: List[Any], analysis_type: str = "statistical") -> Dict[str, Any]:
    """Analyze a chunk of research data."""
    import time
    time.sleep(1)  # Simulate processing time
    
    result = {
        'chunk_size': len(data_chunk),
        'analysis_type': analysis_type,
        'processed_at': datetime.now().isoformat(),
        'summary_statistics': {
            'mean': sum(data_chunk) / len(data_chunk) if data_chunk else 0,
            'min': min(data_chunk) if data_chunk else 0,
            'max': max(data_chunk) if data_chunk else 0
        }
    }
    
    return result


def run_simulation(iterations: int, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run a research simulation."""
    import time
    import random
    
    parameters = parameters or {}
    
    # Simulate computation time based on iterations
    time.sleep(iterations * 0.01)
    
    results = []
    for i in range(iterations):
        # Simulate some random result
        result = random.gauss(parameters.get('mean', 0), parameters.get('std', 1))
        results.append(result)
    
    return {
        'iterations': iterations,
        'parameters': parameters,
        'results': results,
        'summary': {
            'mean': sum(results) / len(results),
            'variance': sum((x - sum(results)/len(results))**2 for x in results) / len(results)
        },
        'completed_at': datetime.now().isoformat()
    }


def optimize_hyperparameters(model_config: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize model hyperparameters."""
    import time
    import random
    
    # Simulate hyperparameter optimization
    time.sleep(2)  # Simulate training time
    
    # Generate random "optimal" parameters within search space
    optimal_params = {}
    for param, space in search_space.items():
        if isinstance(space, list):
            optimal_params[param] = random.choice(space)
        elif isinstance(space, dict) and 'min' in space and 'max' in space:
            optimal_params[param] = random.uniform(space['min'], space['max'])
        else:
            optimal_params[param] = space
    
    # Simulate performance score
    performance_score = random.uniform(0.7, 0.95)
    
    return {
        'model_config': model_config,
        'optimal_parameters': optimal_params,
        'performance_score': performance_score,
        'optimization_time': 2.0,
        'completed_at': datetime.now().isoformat()
    }


# Example usage and testing
if __name__ == "__main__":
    async def test_distributed_framework():
        """Test the distributed research framework."""
        print("🚀 Testing Distributed Research Framework")
        
        # Create framework
        framework = DistributedResearchFramework({
            'min_nodes': 1,
            'max_nodes': 5,
            'scheduling_algorithm': 'priority_first'
        })
        
        # Register research functions
        framework.register_function('analyze_dataset', analyze_large_dataset)
        framework.register_function('run_simulation', run_simulation)
        framework.register_function('optimize_hyperparameters', optimize_hyperparameters)
        
        # Add compute nodes
        framework.add_node({
            'id': 'local_node_1',
            'type': 'local',
            'resources': {'cpu_cores': 4, 'memory_mb': 8192}
        })
        
        framework.add_node({
            'id': 'cloud_node_1',
            'type': 'cloud',
            'resources': {'cpu_cores': 8, 'memory_mb': 16384}
        })
        
        # Start framework
        framework.start_framework()
        
        # Submit various research tasks
        tasks = []
        
        # Data analysis tasks
        for i in range(3):
            task_id = framework.submit_task(
                'analyze_dataset',
                args=[[1, 2, 3, 4, 5] * (i + 1)],
                kwargs={'analysis_type': 'statistical'},
                priority=7
            )
            tasks.append(task_id)
        
        # Simulation tasks
        task_id = framework.submit_task(
            'run_simulation',
            args=[100],
            kwargs={'parameters': {'mean': 0.5, 'std': 0.2}},
            requirements=ResourceSpecification(cpu_cores=2, memory_mb=2048),
            priority=9
        )
        tasks.append(task_id)
        
        # Hyperparameter optimization
        task_id = framework.submit_task(
            'optimize_hyperparameters',
            args=[{'model': 'neural_network'}],
            kwargs={'search_space': {'learning_rate': {'min': 0.001, 'max': 0.1}}},
            requirements=ResourceSpecification(cpu_cores=4, memory_mb=4096),
            priority=8
        )
        tasks.append(task_id)
        
        print(f"✅ Submitted {len(tasks)} research tasks")
        
        # Wait for tasks to complete
        print("⏳ Waiting for tasks to complete...")
        completed_count = 0
        
        while completed_count < len(tasks):
            await asyncio.sleep(1)
            
            status = framework.get_framework_status()
            new_completed = status['scheduler']['completed_tasks']
            
            if new_completed > completed_count:
                completed_count = new_completed
                print(f"📊 Progress: {completed_count}/{len(tasks)} tasks completed")
        
        # Get results
        print("\\n📊 Task Results:")
        for i, task_id in enumerate(tasks):
            result = framework.get_task_result(task_id)
            if result:
                print(f"{i+1}. Task {task_id[:8]}: {result['status']} ({result.get('execution_time', 0):.2f}s)")
        
        # Framework statistics
        final_status = framework.get_framework_status()
        print(f"\\n🎯 Framework Performance:")
        print(f"- Tasks per second: {final_status['performance_metrics']['tasks_per_second']:.2f}")
        print(f"- Average task time: {final_status['performance_metrics']['average_task_time']:.2f}s")
        print(f"- Resource utilization: {final_status['performance_metrics']['resource_utilization']:.2%}")
        print(f"- Auto-scaling actions: {final_status['auto_scaling']['scaling_actions']}")
        
        # Stop framework
        framework.stop_framework()
        
        return final_status
    
    # Run test
    result = asyncio.run(test_distributed_framework())
    print("✅ Distributed framework test completed")