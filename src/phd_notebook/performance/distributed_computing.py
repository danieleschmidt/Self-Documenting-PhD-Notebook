"""
Distributed computing and scaling capabilities for the PhD notebook.
Implements workload distribution, parallel processing, and horizontal scaling.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import multiprocessing as mp
import threading
from queue import Queue, Empty
import hashlib

from ..utils.exceptions import DistributedComputingError, ResourceError
from ..monitoring.metrics import MetricsCollector


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerType(Enum):
    """Type of worker for different workloads."""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    GPU_ACCELERATED = "gpu_accelerated"


@dataclass
class ComputeTask:
    """Represents a computational task."""
    task_id: str
    function_name: str
    args: tuple
    kwargs: dict
    priority: int = 0
    worker_type: WorkerType = WorkerType.CPU_INTENSIVE
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None


@dataclass
class WorkerMetrics:
    """Metrics for a worker process/thread."""
    worker_id: str
    worker_type: WorkerType
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_task_time: Optional[datetime] = None
    cpu_percent: float = 0.0
    memory_percent: float = 0.0


class TaskQueue:
    """Priority queue for computational tasks."""
    
    def __init__(self, maxsize: int = 1000):
        self.queue = Queue(maxsize=maxsize)
        self.pending_tasks: Dict[str, ComputeTask] = {}
        self.completed_tasks: Dict[str, ComputeTask] = {}
        self.failed_tasks: Dict[str, ComputeTask] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger("task_queue")
    
    def submit_task(self, task: ComputeTask) -> str:
        """Submit a task to the queue."""
        with self.lock:
            if len(self.pending_tasks) >= 1000:
                raise ResourceError("Task queue is full")
            
            self.pending_tasks[task.task_id] = task
            self.queue.put(task)
            self.logger.info(f"Submitted task {task.task_id}")
            return task.task_id
    
    def get_task(self, timeout: float = 1.0) -> Optional[ComputeTask]:
        """Get the next task from the queue."""
        try:
            task = self.queue.get(timeout=timeout)
            return task
        except Empty:
            return None
    
    def complete_task(self, task: ComputeTask):
        """Mark a task as completed."""
        with self.lock:
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
    
    def fail_task(self, task: ComputeTask, error: str):
        """Mark a task as failed."""
        with self.lock:
            if task.task_id in self.pending_tasks:
                del self.pending_tasks[task.task_id]
            task.error = error
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            if task.retry_count < task.max_retries:
                # Requeue for retry
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                self.pending_tasks[task.task_id] = task
                self.queue.put(task)
                self.logger.warning(f"Retrying task {task.task_id} (attempt {task.retry_count})")
            else:
                self.failed_tasks[task.task_id] = task
                self.logger.error(f"Task {task.task_id} failed after {task.max_retries} attempts")
    
    def get_task_status(self, task_id: str) -> Optional[ComputeTask]:
        """Get the status of a specific task."""
        with self.lock:
            for task_dict in [self.pending_tasks, self.completed_tasks, self.failed_tasks]:
                if task_id in task_dict:
                    return task_dict[task_id]
            return None
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self.lock:
            return {
                "pending": len(self.pending_tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "queue_size": self.queue.qsize()
            }


class ComputeWorker:
    """Worker process/thread for executing computational tasks."""
    
    def __init__(
        self,
        worker_id: str,
        worker_type: WorkerType,
        task_queue: TaskQueue,
        function_registry: Dict[str, Callable]
    ):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.task_queue = task_queue
        self.function_registry = function_registry
        self.metrics = WorkerMetrics(worker_id, worker_type)
        self.is_running = False
        self.logger = logging.getLogger(f"worker.{worker_id}")
    
    async def start(self):
        """Start the worker."""
        self.is_running = True
        self.logger.info(f"Worker {self.worker_id} started")
        
        while self.is_running:
            try:
                task = self.task_queue.get_task(timeout=1.0)
                if task:
                    await self._execute_task(task)
                else:
                    await asyncio.sleep(0.1)  # No tasks available
            except Exception as e:
                self.logger.error(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(1.0)
    
    def stop(self):
        """Stop the worker."""
        self.is_running = False
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    async def _execute_task(self, task: ComputeTask):
        """Execute a computational task."""
        start_time = time.time()
        task.started_at = datetime.now()
        task.status = TaskStatus.RUNNING
        task.worker_id = self.worker_id
        
        try:
            # Get the function to execute
            if task.function_name not in self.function_registry:
                raise DistributedComputingError(f"Function {task.function_name} not found in registry")
            
            func = self.function_registry[task.function_name]
            
            # Execute with timeout if specified
            if task.timeout_seconds:
                result = await asyncio.wait_for(
                    self._run_function(func, task.args, task.kwargs),
                    timeout=task.timeout_seconds
                )
            else:
                result = await self._run_function(func, task.args, task.kwargs)
            
            task.result = result
            self.task_queue.complete_task(task)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, success=True)
            
            self.logger.info(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            error_msg = str(e)
            self.task_queue.fail_task(task, error_msg)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, success=False)
            
            self.logger.error(f"Task {task.task_id} failed: {error_msg}")
    
    async def _run_function(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Run function in appropriate executor based on worker type."""
        if self.worker_type == WorkerType.CPU_INTENSIVE:
            # Use process pool for CPU-intensive tasks
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor() as executor:
                return await loop.run_in_executor(executor, func, *args, **kwargs)
        
        elif self.worker_type == WorkerType.IO_INTENSIVE:
            # Use thread pool for I/O-intensive tasks
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, func, *args, **kwargs)
        
        elif self.worker_type == WorkerType.MEMORY_INTENSIVE:
            # Run in main thread for memory-intensive tasks
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        else:  # GPU_ACCELERATED
            # For GPU tasks, run in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, func, *args, **kwargs)
    
    def _update_metrics(self, execution_time: float, success: bool):
        """Update worker metrics."""
        if success:
            self.metrics.tasks_completed += 1
        else:
            self.metrics.tasks_failed += 1
        
        self.metrics.total_execution_time += execution_time
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        
        if total_tasks > 0:
            self.metrics.average_execution_time = self.metrics.total_execution_time / total_tasks
        
        self.metrics.last_task_time = datetime.now()


class LoadBalancer:
    """Load balancer for distributing tasks across workers."""
    
    def __init__(self):
        self.workers: Dict[str, ComputeWorker] = {}
        self.worker_loads: Dict[str, int] = {}
        self.logger = logging.getLogger("load_balancer")
    
    def register_worker(self, worker: ComputeWorker):
        """Register a worker with the load balancer."""
        self.workers[worker.worker_id] = worker
        self.worker_loads[worker.worker_id] = 0
        self.logger.info(f"Registered worker {worker.worker_id}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            del self.worker_loads[worker_id]
            self.logger.info(f"Unregistered worker {worker_id}")
    
    def get_optimal_worker(self, task: ComputeTask) -> Optional[str]:
        """Get the optimal worker for a task based on load balancing."""
        # Filter workers by type
        compatible_workers = [
            worker_id for worker_id, worker in self.workers.items()
            if worker.worker_type == task.worker_type
        ]
        
        if not compatible_workers:
            return None
        
        # Find worker with minimum load
        optimal_worker = min(
            compatible_workers,
            key=lambda w_id: self.worker_loads[w_id]
        )
        
        return optimal_worker
    
    def update_worker_load(self, worker_id: str, load_delta: int):
        """Update worker load."""
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] + load_delta)


class DistributedComputeManager:
    """
    Central manager for distributed computing operations.
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.task_queue = TaskQueue()
        self.load_balancer = LoadBalancer()
        self.workers: Dict[str, ComputeWorker] = {}
        self.worker_tasks: Dict[str, asyncio.Task] = {}
        self.function_registry: Dict[str, Callable] = {}
        self.metrics_collector = MetricsCollector()
        self.logger = logging.getLogger("distributed_compute")
        
        self.is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def register_function(self, name: str, func: Callable):
        """Register a function for distributed execution."""
        self.function_registry[name] = func
        self.logger.info(f"Registered function: {name}")
    
    async def start(self):
        """Start the distributed computing system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start workers
        await self._start_workers()
        
        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Distributed computing system started")
    
    async def stop(self):
        """Stop the distributed computing system."""
        self.is_running = False
        
        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop workers
        await self._stop_workers()
        
        self.logger.info("Distributed computing system stopped")
    
    async def _start_workers(self):
        """Start worker processes/threads."""
        worker_types = [
            WorkerType.CPU_INTENSIVE,
            WorkerType.IO_INTENSIVE,
            WorkerType.MEMORY_INTENSIVE
        ]
        
        workers_per_type = max(1, self.max_workers // len(worker_types))
        
        for worker_type in worker_types:
            for i in range(workers_per_type):
                worker_id = f"{worker_type.value}_{i}"
                worker = ComputeWorker(
                    worker_id=worker_id,
                    worker_type=worker_type,
                    task_queue=self.task_queue,
                    function_registry=self.function_registry
                )
                
                self.workers[worker_id] = worker
                self.load_balancer.register_worker(worker)
                
                # Start worker as async task
                self.worker_tasks[worker_id] = asyncio.create_task(worker.start())
    
    async def _stop_workers(self):
        """Stop all workers."""
        # Stop all workers
        for worker in self.workers.values():
            worker.stop()
        
        # Wait for worker tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks.values(), return_exceptions=True)
        
        # Clean up
        self.workers.clear()
        self.worker_tasks.clear()
    
    async def submit_task(
        self,
        function_name: str,
        *args,
        priority: int = 0,
        worker_type: WorkerType = WorkerType.CPU_INTENSIVE,
        timeout_seconds: Optional[int] = None,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """Submit a task for distributed execution."""
        
        # Generate unique task ID
        task_content = f"{function_name}_{args}_{kwargs}_{time.time()}"
        task_id = hashlib.md5(task_content.encode()).hexdigest()
        
        task = ComputeTask(
            task_id=task_id,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            worker_type=worker_type,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries
        )
        
        # Submit to queue
        self.task_queue.submit_task(task)
        
        # Update metrics
        self.metrics_collector.record_counter("distributed.tasks.submitted")
        
        return task_id
    
    async def get_task_result(
        self,
        task_id: str,
        timeout_seconds: Optional[int] = None
    ) -> Any:
        """Get the result of a submitted task."""
        start_time = time.time()
        
        while True:
            task = self.task_queue.get_task_status(task_id)
            
            if not task:
                raise DistributedComputingError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            
            elif task.status == TaskStatus.FAILED:
                raise DistributedComputingError(f"Task {task_id} failed: {task.error}")
            
            elif task.status == TaskStatus.CANCELLED:
                raise DistributedComputingError(f"Task {task_id} was cancelled")
            
            # Check timeout
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                raise DistributedComputingError(f"Timeout waiting for task {task_id}")
            
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.task_queue.get_task_status(task_id)
        
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        
        return False
    
    async def execute_parallel(
        self,
        function_name: str,
        task_args_list: List[tuple],
        worker_type: WorkerType = WorkerType.CPU_INTENSIVE,
        max_parallel: int = None
    ) -> List[Any]:
        """Execute multiple tasks in parallel."""
        max_parallel = max_parallel or len(task_args_list)
        
        # Submit all tasks
        task_ids = []
        for args in task_args_list:
            task_id = await self.submit_task(
                function_name=function_name,
                *args,
                worker_type=worker_type
            )
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = await self.get_task_result(task_id)
            results.append(result)
        
        return results
    
    async def _monitoring_loop(self):
        """Monitor system performance and emit metrics."""
        while self.is_running:
            try:
                # Collect queue stats
                queue_stats = self.task_queue.get_queue_stats()
                
                for stat_name, value in queue_stats.items():
                    self.metrics_collector.record_gauge(f"distributed.queue.{stat_name}", value)
                
                # Collect worker metrics
                for worker_id, worker in self.workers.items():
                    metrics = worker.metrics
                    
                    self.metrics_collector.record_gauge(
                        f"distributed.worker.{worker_id}.tasks_completed",
                        metrics.tasks_completed
                    )
                    self.metrics_collector.record_gauge(
                        f"distributed.worker.{worker_id}.tasks_failed",
                        metrics.tasks_failed
                    )
                    self.metrics_collector.record_gauge(
                        f"distributed.worker.{worker_id}.avg_execution_time",
                        metrics.average_execution_time
                    )
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        queue_stats = self.task_queue.get_queue_stats()
        
        worker_status = {}
        for worker_id, worker in self.workers.items():
            worker_status[worker_id] = {
                "type": worker.worker_type.value,
                "is_running": worker.is_running,
                "tasks_completed": worker.metrics.tasks_completed,
                "tasks_failed": worker.metrics.tasks_failed,
                "average_execution_time": worker.metrics.average_execution_time,
                "load": self.load_balancer.worker_loads.get(worker_id, 0)
            }
        
        return {
            "is_running": self.is_running,
            "queue_stats": queue_stats,
            "worker_status": worker_status,
            "registered_functions": list(self.function_registry.keys()),
            "total_workers": len(self.workers)
        }


# Global distributed compute manager
distributed_manager = DistributedComputeManager()


# Convenience decorators and functions
def distributed_function(
    name: str = None,
    worker_type: WorkerType = WorkerType.CPU_INTENSIVE
):
    """Decorator to register a function for distributed execution."""
    def decorator(func: Callable) -> Callable:
        function_name = name or f"{func.__module__}.{func.__name__}"
        distributed_manager.register_function(function_name, func)
        
        async def wrapper(*args, **kwargs):
            task_id = await distributed_manager.submit_task(
                function_name=function_name,
                *args,
                worker_type=worker_type,
                **kwargs
            )
            return await distributed_manager.get_task_result(task_id)
        
        return wrapper
    return decorator


async def parallel_map(
    func_name: str,
    iterable,
    worker_type: WorkerType = WorkerType.CPU_INTENSIVE,
    max_parallel: int = None
) -> List[Any]:
    """Parallel map function using distributed computing."""
    task_args_list = [(item,) for item in iterable]
    return await distributed_manager.execute_parallel(
        function_name=func_name,
        task_args_list=task_args_list,
        worker_type=worker_type,
        max_parallel=max_parallel
    )