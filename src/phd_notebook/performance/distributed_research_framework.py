"""
Distributed Research Framework
Advanced distributed computing system for scaling research workflows across multiple nodes and cloud providers.
"""

import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import queue

from ..core.note import Note, NoteType
from ..utils.logging import setup_logger


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class NodeType(Enum):
    MASTER = "master"
    WORKER = "worker"
    HYBRID = "hybrid"


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    node_type: NodeType
    host: str
    port: int
    
    # Capabilities
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    storage_gb: float = 100.0
    
    # Current state
    active: bool = True
    load_score: float = 0.0  # 0-1, where 1 is fully loaded
    current_tasks: int = 0
    max_concurrent_tasks: int = 4
    
    # Performance metrics
    tasks_completed: int = 0
    total_compute_time: float = 0.0
    avg_task_time: float = 0.0
    success_rate: float = 1.0
    
    # Health monitoring
    last_heartbeat: Optional[datetime] = None
    failure_count: int = 0
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['node_type'] = self.node_type.value
        if self.last_heartbeat:
            result['last_heartbeat'] = self.last_heartbeat.isoformat()
        return result


@dataclass
class DistributedTask:
    """Represents a task in the distributed research system."""
    task_id: str
    name: str
    operation: str  # Function name or operation type
    
    # Task configuration
    priority: TaskPriority
    estimated_duration: float = 60.0  # seconds
    required_resources: Dict[ResourceType, float] = None
    dependencies: List[str] = None  # Task IDs this task depends on
    
    # Task data
    input_data: Dict[str, Any] = None
    parameters: Dict[str, Any] = None
    
    # Execution state
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results and errors
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Progress tracking
    progress: float = 0.0  # 0-1
    progress_message: str = ""
    
    def __post_init__(self):
        if self.required_resources is None:
            self.required_resources = {ResourceType.CPU: 1.0, ResourceType.MEMORY: 1.0}
        if self.dependencies is None:
            self.dependencies = []
        if self.input_data is None:
            self.input_data = {}
        if self.parameters is None:
            self.parameters = {}
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['priority'] = self.priority.value
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        
        if self.started_at:
            result['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            result['completed_at'] = self.completed_at.isoformat()
            
        # Convert ResourceType keys to strings
        if self.required_resources:
            result['required_resources'] = {
                resource.value: amount 
                for resource, amount in self.required_resources.items()
            }
        
        return result


class TaskScheduler:
    """Intelligent task scheduler for distributed research workloads."""
    
    def __init__(self):
        self.pending_tasks: deque = deque()
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.nodes: Dict[str, ComputeNode] = {}
        
        # Scheduling policies
        self.scheduling_policy = "priority_load_balance"  # priority_load_balance, round_robin, least_loaded
        self.max_queue_size = 10000
        
        self.logger = setup_logger("distributed.scheduler")
    
    def add_node(self, node: ComputeNode):
        """Add a compute node to the cluster."""
        self.nodes[node.node_id] = node
        self.logger.info(f"Added node: {node.node_id} ({node.node_type.value})")
    
    def remove_node(self, node_id: str):
        """Remove a compute node from the cluster."""
        if node_id in self.nodes:
            # Reschedule running tasks
            tasks_to_reschedule = [
                task for task in self.running_tasks.values()
                if task.assigned_node == node_id
            ]
            
            for task in tasks_to_reschedule:
                task.status = TaskStatus.PENDING
                task.assigned_node = None
                task.retry_count += 1
                self.pending_tasks.append(task)
                del self.running_tasks[task.task_id]
            
            del self.nodes[node_id]
            self.logger.info(f"Removed node: {node_id}, rescheduled {len(tasks_to_reschedule)} tasks")
    
    def submit_task(self, task: DistributedTask) -> bool:
        """Submit a task for execution."""
        if len(self.pending_tasks) >= self.max_queue_size:
            self.logger.warning(f"Queue full, rejecting task: {task.task_id}")
            return False
        
        # Check dependencies
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                if dep_task_id not in self.running_tasks and dep_task_id not in [t.task_id for t in self.pending_tasks]:
                    self.logger.error(f"Dependency not found: {dep_task_id} for task {task.task_id}")
                    return False
        
        # Insert task based on priority
        self._insert_task_by_priority(task)
        self.logger.info(f"Submitted task: {task.task_id} (priority: {task.priority.value})")
        return True
    
    def _insert_task_by_priority(self, task: DistributedTask):
        """Insert task into queue based on priority."""
        # Find insertion point (higher priority values go first)
        insert_index = 0
        for i, existing_task in enumerate(self.pending_tasks):
            if existing_task.priority.value < task.priority.value:
                insert_index = i
                break
            insert_index = i + 1
        
        # Insert at the appropriate position
        if insert_index >= len(self.pending_tasks):
            self.pending_tasks.append(task)
        else:
            # Convert to list, insert, and convert back
            task_list = list(self.pending_tasks)
            task_list.insert(insert_index, task)
            self.pending_tasks = deque(task_list)
    
    def schedule_tasks(self) -> List[tuple]:
        """Schedule pending tasks to available nodes."""
        scheduled = []
        
        if not self.nodes or not self.pending_tasks:
            return scheduled
        
        # Get available nodes
        available_nodes = [
            node for node in self.nodes.values()
            if node.active and node.current_tasks < node.max_concurrent_tasks
        ]
        
        if not available_nodes:
            return scheduled
        
        # Schedule tasks based on policy
        if self.scheduling_policy == "priority_load_balance":
            scheduled = self._schedule_priority_load_balance(available_nodes)
        elif self.scheduling_policy == "round_robin":
            scheduled = self._schedule_round_robin(available_nodes)
        elif self.scheduling_policy == "least_loaded":
            scheduled = self._schedule_least_loaded(available_nodes)
        
        # Update node and task states
        for task, node in scheduled:
            task.status = TaskStatus.RUNNING
            task.assigned_node = node.node_id
            task.started_at = datetime.now()
            
            self.running_tasks[task.task_id] = task
            node.current_tasks += 1
            node.load_score = node.current_tasks / node.max_concurrent_tasks
        
        return scheduled
    
    def _schedule_priority_load_balance(self, available_nodes: List[ComputeNode]) -> List[tuple]:
        """Schedule tasks using priority and load balancing."""
        scheduled = []
        tasks_to_schedule = list(self.pending_tasks)
        
        for task in tasks_to_schedule:
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(task):
                continue
            
            # Find best node for this task
            best_node = self._find_best_node(task, available_nodes)
            if best_node and best_node.current_tasks < best_node.max_concurrent_tasks:
                scheduled.append((task, best_node))
                self.pending_tasks.remove(task)
                
                # Update node availability for next iteration
                best_node.current_tasks += 1
                if best_node.current_tasks >= best_node.max_concurrent_tasks:
                    available_nodes.remove(best_node)
                
                if not available_nodes:
                    break
        
        return scheduled
    
    def _schedule_round_robin(self, available_nodes: List[ComputeNode]) -> List[tuple]:
        """Schedule tasks using round-robin policy."""
        scheduled = []
        node_index = 0
        tasks_to_schedule = list(self.pending_tasks)
        
        for task in tasks_to_schedule:
            if not self._dependencies_satisfied(task):
                continue
            
            if available_nodes:
                node = available_nodes[node_index % len(available_nodes)]
                if node.current_tasks < node.max_concurrent_tasks:
                    scheduled.append((task, node))
                    self.pending_tasks.remove(task)
                    node.current_tasks += 1
                    node_index += 1
        
        return scheduled
    
    def _schedule_least_loaded(self, available_nodes: List[ComputeNode]) -> List[tuple]:
        """Schedule tasks to least loaded nodes."""
        scheduled = []
        tasks_to_schedule = list(self.pending_tasks)
        
        for task in tasks_to_schedule:
            if not self._dependencies_satisfied(task):
                continue
            
            # Sort nodes by load
            available_nodes.sort(key=lambda n: n.load_score)
            
            if available_nodes and available_nodes[0].current_tasks < available_nodes[0].max_concurrent_tasks:
                node = available_nodes[0]
                scheduled.append((task, node))
                self.pending_tasks.remove(task)
                node.current_tasks += 1
                node.load_score = node.current_tasks / node.max_concurrent_tasks
        
        return scheduled
    
    def _dependencies_satisfied(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
        return True
    
    def _find_best_node(self, task: DistributedTask, available_nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Find the best node for a task based on resources and load."""
        if not available_nodes:
            return None
        
        # Score nodes based on resource availability and current load
        node_scores = []
        
        for node in available_nodes:
            if node.current_tasks >= node.max_concurrent_tasks:
                continue
            
            # Calculate resource match score
            resource_score = self._calculate_resource_score(task, node)
            
            # Calculate load score (lower load is better)
            load_score = 1.0 - node.load_score
            
            # Calculate reliability score
            reliability_score = node.success_rate
            
            # Combined score
            total_score = (resource_score * 0.4 + load_score * 0.4 + reliability_score * 0.2)
            node_scores.append((total_score, node))
        
        if node_scores:
            # Return node with highest score
            node_scores.sort(key=lambda x: x[0], reverse=True)
            return node_scores[0][1]
        
        return None
    
    def _calculate_resource_score(self, task: DistributedTask, node: ComputeNode) -> float:
        """Calculate how well a node matches task resource requirements."""
        score = 0.0
        weight_sum = 0.0
        
        for resource_type, required_amount in task.required_resources.items():
            weight = 1.0  # All resources weighted equally for now
            
            if resource_type == ResourceType.CPU:
                available = node.cpu_cores
                score += min(available / required_amount, 1.0) * weight
            elif resource_type == ResourceType.MEMORY:
                available = node.memory_gb
                score += min(available / required_amount, 1.0) * weight
            elif resource_type == ResourceType.GPU:
                available = node.gpu_count
                if required_amount > 0:
                    score += min(available / required_amount, 1.0) * weight
                else:
                    score += 1.0 * weight  # No GPU required
            elif resource_type == ResourceType.STORAGE:
                available = node.storage_gb
                score += min(available / required_amount, 1.0) * weight
            
            weight_sum += weight
        
        return score / max(weight_sum, 1.0)
    
    def task_completed(self, task_id: str, result: Any = None, error: str = None):
        """Mark task as completed."""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.completed_at = datetime.now()
            task.progress = 1.0
            
            if error:
                task.status = TaskStatus.FAILED
                task.error = error
                
                # Retry if possible
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.RETRY
                    task.assigned_node = None
                    task.started_at = None
                    task.completed_at = None
                    task.progress = 0.0
                    self.pending_tasks.appendleft(task)  # High priority for retry
                    self.logger.info(f"Retrying task: {task_id} (attempt {task.retry_count + 1})")
                else:
                    self.completed_tasks[task_id] = task
                    self.logger.error(f"Task failed permanently: {task_id} - {error}")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                self.completed_tasks[task_id] = task
                self.logger.info(f"Task completed: {task_id}")
            
            # Update node state
            if task.assigned_node and task.assigned_node in self.nodes:
                node = self.nodes[task.assigned_node]
                node.current_tasks -= 1
                node.load_score = node.current_tasks / node.max_concurrent_tasks
                
                # Update performance metrics
                if task.status == TaskStatus.COMPLETED:
                    execution_time = (task.completed_at - task.started_at).total_seconds()
                    node.tasks_completed += 1
                    node.total_compute_time += execution_time
                    node.avg_task_time = node.total_compute_time / node.tasks_completed
                    
                    # Update success rate
                    total_tasks = node.tasks_completed + node.failure_count
                    node.success_rate = node.tasks_completed / max(total_tasks, 1)
                elif task.status == TaskStatus.FAILED and task.retry_count >= task.max_retries:
                    node.failure_count += 1
                    total_tasks = node.tasks_completed + node.failure_count
                    node.success_rate = node.tasks_completed / max(total_tasks, 1)
            
            if task.status != TaskStatus.RETRY:
                del self.running_tasks[task_id]
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "active_nodes": len([n for n in self.nodes.values() if n.active]),
            "total_nodes": len(self.nodes),
            "scheduling_policy": self.scheduling_policy
        }


class WorkerExecutor:
    """Executes tasks on worker nodes."""
    
    def __init__(self, node: ComputeNode, max_workers: int = None):
        self.node = node
        self.max_workers = max_workers or node.max_concurrent_tasks
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(self.max_workers, mp.cpu_count()))
        
        # Task registry
        self.task_handlers: Dict[str, Callable] = {}
        self.active_tasks: Dict[str, Any] = {}  # task_id -> future
        
        self.logger = setup_logger(f"distributed.executor.{node.node_id}")
    
    def register_task_handler(self, operation: str, handler: Callable):
        """Register a task handler function."""
        self.task_handlers[operation] = handler
        self.logger.info(f"Registered task handler: {operation}")
    
    async def execute_task(self, task: DistributedTask) -> Any:
        """Execute a task asynchronously."""
        if task.operation not in self.task_handlers:
            raise ValueError(f"No handler registered for operation: {task.operation}")
        
        handler = self.task_handlers[task.operation]
        
        # Determine execution method based on task type
        if self._is_cpu_intensive(task):
            # Use process pool for CPU-intensive tasks
            future = self.process_pool.submit(self._execute_with_progress, handler, task)
        else:
            # Use thread pool for I/O-bound tasks
            future = self.thread_pool.submit(self._execute_with_progress, handler, task)
        
        self.active_tasks[task.task_id] = future
        
        try:
            # Wait for completion with timeout
            timeout = task.estimated_duration * 2  # Allow 2x estimated time
            result = future.result(timeout=timeout)
            return result
        
        except Exception as e:
            self.logger.error(f"Task execution failed: {task.task_id} - {e}")
            raise e
        
        finally:
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    def _execute_with_progress(self, handler: Callable, task: DistributedTask) -> Any:
        """Execute task with progress tracking."""
        try:
            # Create progress callback
            def update_progress(progress: float, message: str = ""):
                task.progress = progress
                task.progress_message = message
            
            # Execute handler with task data and progress callback
            if asyncio.iscoroutinefunction(handler):
                # Handle async functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        handler(task.input_data, task.parameters, update_progress)
                    )
                finally:
                    loop.close()
            else:
                # Handle sync functions
                return handler(task.input_data, task.parameters, update_progress)
                
        except Exception as e:
            self.logger.error(f"Handler execution failed: {e}")
            raise e
    
    def _is_cpu_intensive(self, task: DistributedTask) -> bool:
        """Determine if task is CPU-intensive."""
        cpu_requirement = task.required_resources.get(ResourceType.CPU, 1.0)
        return cpu_requirement > 2.0  # Threshold for CPU-intensive tasks
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            cancelled = future.cancel()
            if cancelled:
                del self.active_tasks[task_id]
            return cancelled
        return False
    
    def get_task_progress(self, task_id: str) -> tuple:
        """Get task progress."""
        # This would need to be implemented with shared memory or messaging
        # For now, return basic info
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            return (0.5, "In progress")  # Placeholder
        return (0.0, "Unknown")
    
    def shutdown(self):
        """Shutdown executor pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class DistributedResearchFramework:
    """
    Main distributed research framework that coordinates task scheduling,
    execution, and resource management across multiple compute nodes.
    """
    
    def __init__(self, notebook_path: Path, node_id: str = None):
        self.logger = setup_logger("distributed.framework")
        self.notebook_path = notebook_path
        
        # Node identification
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.node_type = NodeType.MASTER  # Can be changed based on configuration
        
        # Core components
        self.scheduler = TaskScheduler()
        self.executor: Optional[WorkerExecutor] = None
        
        # Node management
        self.local_node: Optional[ComputeNode] = None
        self.cluster_nodes: Dict[str, ComputeNode] = {}
        
        # Task management
        self.submitted_tasks: Dict[str, DistributedTask] = {}
        self.task_results: Dict[str, Any] = {}
        
        # Communication and coordination
        self.heartbeat_interval = 30  # seconds
        self.node_timeout = 120  # seconds
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        
        # Create directories
        self.distributed_dir = notebook_path / "distributed"
        self.tasks_dir = self.distributed_dir / "tasks"
        self.nodes_dir = self.distributed_dir / "nodes"
        self.results_dir = self.distributed_dir / "results"
        
        for dir_path in [self.distributed_dir, self.tasks_dir, self.nodes_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize local node
        self._initialize_local_node()
        
        # Start background tasks
        self.running = False
        self._background_tasks: List[asyncio.Task] = []
    
    def _initialize_local_node(self):
        """Initialize local compute node."""
        try:
            import psutil
            
            # Get system specs
            cpu_cores = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Try to detect GPUs (simplified)
            gpu_count = 0
            try:
                import GPUtil
                gpu_count = len(GPUtil.getGPUs())
            except ImportError:
                pass
            
            # Get disk space
            disk_usage = psutil.disk_usage(str(self.notebook_path))
            storage_gb = disk_usage.free / (1024**3)
            
        except ImportError:
            # Fallback values
            cpu_cores = mp.cpu_count()
            memory_gb = 8.0
            gpu_count = 0
            storage_gb = 100.0
        
        self.local_node = ComputeNode(
            node_id=self.node_id,
            node_type=self.node_type,
            host="localhost",
            port=8888,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            storage_gb=storage_gb,
            max_concurrent_tasks=min(cpu_cores, 8),  # Reasonable default
            last_heartbeat=datetime.now()
        )
        
        # Add to scheduler if this is a worker/hybrid node
        if self.node_type in [NodeType.WORKER, NodeType.HYBRID]:
            self.scheduler.add_node(self.local_node)
            
            # Initialize executor
            self.executor = WorkerExecutor(self.local_node)
            self._register_default_task_handlers()
        
        self.logger.info(f"Initialized local node: {self.node_id} ({self.node_type.value})")
        self.logger.info(f"Node specs: {cpu_cores} CPU cores, {memory_gb:.1f}GB RAM, {gpu_count} GPUs")
    
    def _register_default_task_handlers(self):
        """Register default task handlers."""
        if not self.executor:
            return
        
        # Research-specific task handlers
        self.executor.register_task_handler("hypothesis_analysis", self._handle_hypothesis_analysis)
        self.executor.register_task_handler("literature_processing", self._handle_literature_processing)
        self.executor.register_task_handler("data_analysis", self._handle_data_analysis)
        self.executor.register_task_handler("paper_generation", self._handle_paper_generation)
        self.executor.register_task_handler("collaboration_analysis", self._handle_collaboration_analysis)
        
        # Generic computation handlers
        self.executor.register_task_handler("compute_intensive", self._handle_compute_intensive)
        self.executor.register_task_handler("io_intensive", self._handle_io_intensive)
    
    async def start_framework(self):
        """Start the distributed framework."""
        if self.running:
            self.logger.warning("Framework already running")
            return
        
        self.running = True
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._scheduling_loop()),
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._health_monitoring_loop())
        ]
        
        self.logger.info("Distributed research framework started")
    
    async def stop_framework(self):
        """Stop the distributed framework."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown()
        
        self.logger.info("Distributed research framework stopped")
    
    async def _scheduling_loop(self):
        """Main scheduling loop."""
        while self.running:
            try:
                # Schedule pending tasks
                scheduled = self.scheduler.schedule_tasks()
                
                # Execute scheduled tasks
                for task, node in scheduled:
                    if node.node_id == self.node_id and self.executor:
                        # Execute locally
                        asyncio.create_task(self._execute_local_task(task))
                    else:
                        # Send to remote node (not implemented in this basic version)
                        self.logger.info(f"Task {task.task_id} scheduled to remote node {node.node_id}")
                
                # Brief pause
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _execute_local_task(self, task: DistributedTask):
        """Execute task on local node."""
        try:
            result = await self.executor.execute_task(task)
            self.scheduler.task_completed(task.task_id, result=result)
            self.task_results[task.task_id] = result
            
        except Exception as e:
            self.scheduler.task_completed(task.task_id, error=str(e))
    
    async def _heartbeat_loop(self):
        """Send heartbeat signals and monitor node health."""
        while self.running:
            try:
                # Update local node heartbeat
                if self.local_node:
                    self.local_node.last_heartbeat = datetime.now()
                
                # Check for dead nodes
                now = datetime.now()
                dead_nodes = []
                
                for node_id, node in self.cluster_nodes.items():
                    if node.last_heartbeat:
                        time_since_heartbeat = (now - node.last_heartbeat).total_seconds()
                        if time_since_heartbeat > self.node_timeout:
                            dead_nodes.append(node_id)
                
                # Remove dead nodes
                for node_id in dead_nodes:
                    self.remove_node(node_id)
                    self.logger.warning(f"Removed dead node: {node_id}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _health_monitoring_loop(self):
        """Monitor system health and performance."""
        while self.running:
            try:
                # Update local node metrics
                if self.local_node:
                    await self._update_node_metrics(self.local_node)
                
                # Collect cluster-wide metrics
                await self._collect_cluster_metrics()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _update_node_metrics(self, node: ComputeNode):
        """Update node performance metrics."""
        try:
            import psutil
            
            # Update load score based on actual system load
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Combined load score
            system_load = (cpu_percent / 100.0 + memory_percent / 100.0) / 2.0
            task_load = node.current_tasks / node.max_concurrent_tasks
            
            node.load_score = max(system_load, task_load)
            
        except ImportError:
            # Use task-based load only
            node.load_score = node.current_tasks / node.max_concurrent_tasks
    
    async def _collect_cluster_metrics(self):
        """Collect cluster-wide performance metrics."""
        all_nodes = list(self.cluster_nodes.values())
        if self.local_node:
            all_nodes.append(self.local_node)
        
        if not all_nodes:
            return
        
        # Calculate cluster metrics
        total_cpu_cores = sum(node.cpu_cores for node in all_nodes)
        total_memory_gb = sum(node.memory_gb for node in all_nodes)
        total_tasks = sum(node.current_tasks for node in all_nodes)
        max_tasks = sum(node.max_concurrent_tasks for node in all_nodes)
        
        avg_load = sum(node.load_score for node in all_nodes) / len(all_nodes)
        cluster_utilization = total_tasks / max(max_tasks, 1)
        
        self.performance_metrics = {
            "timestamp": datetime.now().isoformat(),
            "cluster_size": len(all_nodes),
            "total_cpu_cores": total_cpu_cores,
            "total_memory_gb": total_memory_gb,
            "total_active_tasks": total_tasks,
            "max_concurrent_tasks": max_tasks,
            "avg_node_load": avg_load,
            "cluster_utilization": cluster_utilization,
            "scheduler_status": self.scheduler.get_status()
        }
    
    def submit_task(
        self,
        operation: str,
        input_data: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        estimated_duration: float = 60.0,
        required_resources: Dict[ResourceType, float] = None,
        dependencies: List[str] = None
    ) -> str:
        """Submit a task for distributed execution."""
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = DistributedTask(
            task_id=task_id,
            name=f"{operation}_task",
            operation=operation,
            priority=priority,
            estimated_duration=estimated_duration,
            required_resources=required_resources or {ResourceType.CPU: 1.0, ResourceType.MEMORY: 1.0},
            dependencies=dependencies or [],
            input_data=input_data or {},
            parameters=parameters or {}
        )
        
        success = self.scheduler.submit_task(task)
        if success:
            self.submitted_tasks[task_id] = task
            self._save_task(task)
            self.logger.info(f"Submitted task: {task_id} ({operation})")
            return task_id
        else:
            raise RuntimeError(f"Failed to submit task: {task_id}")
    
    def add_node(self, node: ComputeNode):
        """Add a node to the cluster."""
        self.cluster_nodes[node.node_id] = node
        self.scheduler.add_node(node)
        self._save_node(node)
    
    def remove_node(self, node_id: str):
        """Remove a node from the cluster."""
        if node_id in self.cluster_nodes:
            del self.cluster_nodes[node_id]
        self.scheduler.remove_node(node_id)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check different task stores
        task = None
        if task_id in self.submitted_tasks:
            task = self.submitted_tasks[task_id]
        elif task_id in self.scheduler.running_tasks:
            task = self.scheduler.running_tasks[task_id]
        elif task_id in self.scheduler.completed_tasks:
            task = self.scheduler.completed_tasks[task_id]
        
        if task:
            status = {
                "task_id": task.task_id,
                "name": task.name,
                "operation": task.operation,
                "status": task.status.value,
                "progress": task.progress,
                "progress_message": task.progress_message,
                "assigned_node": task.assigned_node,
                "created_at": task.created_at.isoformat(),
                "retry_count": task.retry_count
            }
            
            if task.started_at:
                status["started_at"] = task.started_at.isoformat()
            if task.completed_at:
                status["completed_at"] = task.completed_at.isoformat()
                duration = (task.completed_at - task.started_at).total_seconds()
                status["duration"] = duration
            
            if task.error:
                status["error"] = task.error
            
            return status
        
        return None
    
    def get_task_result(self, task_id: str) -> Any:
        """Get result of a completed task."""
        return self.task_results.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        # Check if task is in pending queue
        for task in self.scheduler.pending_tasks:
            if task.task_id == task_id:
                task.status = TaskStatus.CANCELLED
                self.scheduler.pending_tasks.remove(task)
                return True
        
        # Check if task is running
        if task_id in self.scheduler.running_tasks:
            task = self.scheduler.running_tasks[task_id]
            
            # Try to cancel on assigned node
            if task.assigned_node == self.node_id and self.executor:
                cancelled = self.executor.cancel_task(task_id)
                if cancelled:
                    task.status = TaskStatus.CANCELLED
                    self.scheduler.task_completed(task_id, error="Task cancelled")
                    return True
        
        return False
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        all_nodes = list(self.cluster_nodes.values())
        if self.local_node:
            all_nodes.append(self.local_node)
        
        active_nodes = [n for n in all_nodes if n.active]
        
        return {
            "framework_running": self.running,
            "local_node_id": self.node_id,
            "local_node_type": self.node_type.value,
            "total_nodes": len(all_nodes),
            "active_nodes": len(active_nodes),
            "scheduler_status": self.scheduler.get_status(),
            "performance_metrics": self.performance_metrics,
            "node_details": [node.to_dict() for node in all_nodes]
        }
    
    # Task handler implementations
    async def _handle_hypothesis_analysis(self, input_data: Dict, parameters: Dict, progress_callback: Callable):
        """Handle hypothesis analysis tasks."""
        progress_callback(0.1, "Starting hypothesis analysis")
        
        # Simulate hypothesis analysis work
        await asyncio.sleep(1)
        progress_callback(0.5, "Processing evidence")
        
        await asyncio.sleep(1)
        progress_callback(0.9, "Generating insights")
        
        await asyncio.sleep(0.5)
        progress_callback(1.0, "Analysis complete")
        
        return {"analysis": "hypothesis analysis results", "confidence": 0.85}
    
    async def _handle_literature_processing(self, input_data: Dict, parameters: Dict, progress_callback: Callable):
        """Handle literature processing tasks."""
        progress_callback(0.1, "Loading literature data")
        
        await asyncio.sleep(0.5)
        progress_callback(0.3, "Extracting key insights")
        
        await asyncio.sleep(1)
        progress_callback(0.7, "Building citation network")
        
        await asyncio.sleep(0.8)
        progress_callback(1.0, "Processing complete")
        
        return {"processed_papers": 50, "insights": ["insight1", "insight2"]}
    
    async def _handle_data_analysis(self, input_data: Dict, parameters: Dict, progress_callback: Callable):
        """Handle data analysis tasks."""
        progress_callback(0.1, "Loading data")
        
        # Simulate data processing
        data_size = parameters.get("data_size", 1000)
        batch_size = 100
        
        for i in range(0, data_size, batch_size):
            await asyncio.sleep(0.1)  # Simulate processing time
            progress = 0.1 + 0.8 * (i + batch_size) / data_size
            progress_callback(progress, f"Processing batch {i//batch_size + 1}")
        
        progress_callback(1.0, "Analysis complete")
        
        return {"processed_records": data_size, "results": "analysis results"}
    
    async def _handle_paper_generation(self, input_data: Dict, parameters: Dict, progress_callback: Callable):
        """Handle paper generation tasks."""
        progress_callback(0.1, "Analyzing content")
        
        await asyncio.sleep(2)
        progress_callback(0.3, "Generating introduction")
        
        await asyncio.sleep(3)
        progress_callback(0.6, "Generating results section")
        
        await asyncio.sleep(2)
        progress_callback(0.9, "Finalizing paper")
        
        await asyncio.sleep(1)
        progress_callback(1.0, "Paper generation complete")
        
        return {"paper_id": "generated_paper_123", "sections": 5, "word_count": 8500}
    
    async def _handle_collaboration_analysis(self, input_data: Dict, parameters: Dict, progress_callback: Callable):
        """Handle collaboration analysis tasks."""
        progress_callback(0.1, "Loading collaboration data")
        
        await asyncio.sleep(1)
        progress_callback(0.4, "Building network graph")
        
        await asyncio.sleep(1.5)
        progress_callback(0.8, "Calculating metrics")
        
        await asyncio.sleep(0.7)
        progress_callback(1.0, "Analysis complete")
        
        return {"network_nodes": 25, "connections": 45, "insights": "collaboration insights"}
    
    async def _handle_compute_intensive(self, input_data: Dict, parameters: Dict, progress_callback: Callable):
        """Handle compute-intensive tasks."""
        progress_callback(0.1, "Starting computation")
        
        # Simulate CPU-intensive work
        iterations = parameters.get("iterations", 1000000)
        batch_size = iterations // 10
        
        for i in range(0, iterations, batch_size):
            # Simulate computation
            sum(range(batch_size))  # Simple CPU work
            progress = 0.1 + 0.8 * (i + batch_size) / iterations
            progress_callback(progress, f"Computing batch {i//batch_size + 1}/10")
            await asyncio.sleep(0.01)  # Yield control
        
        progress_callback(1.0, "Computation complete")
        
        return {"result": "computation_result", "iterations": iterations}
    
    async def _handle_io_intensive(self, input_data: Dict, parameters: Dict, progress_callback: Callable):
        """Handle I/O-intensive tasks."""
        progress_callback(0.1, "Starting I/O operations")
        
        # Simulate I/O work
        files_to_process = parameters.get("files", 100)
        
        for i in range(files_to_process):
            await asyncio.sleep(0.02)  # Simulate I/O delay
            progress = 0.1 + 0.8 * (i + 1) / files_to_process
            progress_callback(progress, f"Processing file {i + 1}/{files_to_process}")
        
        progress_callback(1.0, "I/O operations complete")
        
        return {"files_processed": files_to_process, "total_size": "1.5GB"}
    
    def _save_task(self, task: DistributedTask):
        """Save task to storage."""
        file_path = self.tasks_dir / f"{task.task_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(task.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_node(self, node: ComputeNode):
        """Save node configuration to storage."""
        file_path = self.nodes_dir / f"{node.node_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(node.to_dict(), f, indent=2, ensure_ascii=False)