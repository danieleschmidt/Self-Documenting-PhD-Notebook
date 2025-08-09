"""
Concurrent processing and resource pooling for high performance.
"""

import asyncio
import concurrent.futures
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable
from queue import Queue, PriorityQueue
import multiprocessing


@dataclass
class Task:
    """Represents a task to be processed."""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
    
    def __lt__(self, other):
        return self.priority < other.priority


class TaskPool:
    """Advanced task pool with priority queuing and resource management."""
    
    def __init__(
        self,
        max_workers: int = None,
        max_queue_size: int = 1000,
        use_processes: bool = False
    ):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.max_queue_size = max_queue_size
        self.use_processes = use_processes
        
        # Task management
        self.task_queue = PriorityQueue(maxsize=max_queue_size)
        self.active_tasks: Dict[str, concurrent.futures.Future] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Exception] = {}
        
        # Executor management
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Worker threads
        self.worker_threads = []
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # Start worker threads
        self._start_workers()
    
    def submit_task(
        self,
        func: Callable,
        *args,
        task_id: str = None,
        priority: int = 0,
        **kwargs
    ) -> str:
        """Submit a task for execution."""
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}_{id(func)}"
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        try:
            self.task_queue.put_nowait(task)
            self.stats['tasks_submitted'] += 1
            return task_id
        except:
            raise RuntimeError("Task queue is full")
    
    def submit_batch(
        self,
        tasks: List[Dict[str, Any]],
        default_priority: int = 0
    ) -> List[str]:
        """Submit multiple tasks as a batch."""
        task_ids = []
        
        for task_spec in tasks:
            task_id = self.submit_task(
                func=task_spec['func'],
                *task_spec.get('args', ()),
                task_id=task_spec.get('task_id'),
                priority=task_spec.get('priority', default_priority),
                **task_spec.get('kwargs', {})
            )
            task_ids.append(task_id)
        
        return task_ids
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of a completed task."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        if task_id in self.failed_tasks:
            raise self.failed_tasks[task_id]
        
        if task_id in self.active_tasks:
            try:
                result = self.active_tasks[task_id].result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            except Exception as e:
                self.failed_tasks[task_id] = e
                raise e
        
        raise ValueError(f"Task {task_id} not found")
    
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for multiple tasks to complete."""
        results = {}
        errors = {}
        
        start_time = time.time()
        
        for task_id in task_ids:
            try:
                remaining_timeout = None
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, timeout - elapsed)
                    
                    if remaining_timeout <= 0:
                        raise TimeoutError(f"Timeout waiting for tasks")
                
                results[task_id] = self.get_result(task_id, remaining_timeout)
            except Exception as e:
                errors[task_id] = e
        
        return {"results": results, "errors": errors}
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or active task."""
        if task_id in self.active_tasks:
            cancelled = self.active_tasks[task_id].cancel()
            if cancelled:
                self.stats['tasks_cancelled'] += 1
            return cancelled
        
        # Try to remove from queue (not efficiently implemented for PriorityQueue)
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the task pool."""
        return {
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'max_workers': self.max_workers,
            'stats': self.stats.copy()
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the task pool."""
        self.shutdown_event.set()
        
        # Wait for worker threads
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
    
    def _start_workers(self) -> None:
        """Start worker threads to process tasks."""
        for i in range(min(4, self.max_workers)):  # Limit dispatcher threads
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"TaskPool-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
    
    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1.0)
                
                # Submit to executor
                future = self.executor.submit(task.func, *task.args, **task.kwargs)
                self.active_tasks[task.id] = future
                
                # Add completion callback
                future.add_done_callback(
                    lambda f, tid=task.id, start_time=time.time(): 
                    self._handle_task_completion(tid, f, start_time)
                )
                
                self.task_queue.task_done()
                
            except:
                # Queue timeout or other error
                continue
    
    def _handle_task_completion(
        self,
        task_id: str,
        future: concurrent.futures.Future,
        start_time: float
    ) -> None:
        """Handle task completion."""
        execution_time = time.time() - start_time
        self.stats['total_execution_time'] += execution_time
        
        # Remove from active tasks
        self.active_tasks.pop(task_id, None)
        
        try:
            result = future.result()
            self.completed_tasks[task_id] = result
            self.stats['tasks_completed'] += 1
        except Exception as e:
            self.failed_tasks[task_id] = e
            self.stats['tasks_failed'] += 1
        
        # Update average execution time
        total_completed = self.stats['tasks_completed'] + self.stats['tasks_failed']
        if total_completed > 0:
            self.stats['average_execution_time'] = (
                self.stats['total_execution_time'] / total_completed
            )


class ConcurrentProcessor:
    """High-level concurrent processing system."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Different pools for different types of tasks
        self.cpu_pool = TaskPool(max_workers=self.max_workers, use_processes=True)
        self.io_pool = TaskPool(max_workers=self.max_workers * 2, use_processes=False)
        
        # Async event loop for coordination
        self._loop = None
        self._loop_thread = None
        self._start_async_loop()
    
    async def process_cpu_intensive(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Process CPU-intensive tasks."""
        loop = asyncio.get_event_loop()
        
        # Submit to process pool
        task_id = self.cpu_pool.submit_task(func, *args, **kwargs)
        
        # Wait for completion asynchronously
        while task_id not in self.cpu_pool.completed_tasks and task_id not in self.cpu_pool.failed_tasks:
            await asyncio.sleep(0.1)
        
        return self.cpu_pool.get_result(task_id)
    
    async def process_io_intensive(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Process I/O-intensive tasks."""
        loop = asyncio.get_event_loop()
        
        # Submit to thread pool
        task_id = self.io_pool.submit_task(func, *args, **kwargs)
        
        # Wait for completion asynchronously
        while task_id not in self.io_pool.completed_tasks and task_id not in self.io_pool.failed_tasks:
            await asyncio.sleep(0.01)  # Shorter wait for I/O
        
        return self.io_pool.get_result(task_id)
    
    async def process_batch_parallel(
        self,
        tasks: List[Dict[str, Any]],
        task_type: str = "io"  # "io" or "cpu"
    ) -> Dict[str, Any]:
        """Process multiple tasks in parallel."""
        pool = self.cpu_pool if task_type == "cpu" else self.io_pool
        
        # Submit all tasks
        task_ids = pool.submit_batch(tasks)
        
        # Wait for all to complete
        results = pool.wait_for_completion(task_ids, timeout=300)  # 5 minute timeout
        
        return results
    
    async def process_with_rate_limit(
        self,
        func: Callable,
        items: List[Any],
        rate_limit: int = 10,  # items per second
        task_type: str = "io"
    ) -> List[Any]:
        """Process items with rate limiting."""
        results = []
        interval = 1.0 / rate_limit
        
        for item in items:
            if task_type == "cpu":
                result = await self.process_cpu_intensive(func, item)
            else:
                result = await self.process_io_intensive(func, item)
            
            results.append(result)
            await asyncio.sleep(interval)
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'cpu_pool': self.cpu_pool.get_status(),
            'io_pool': self.io_pool.get_status(),
            'system_info': {
                'cpu_count': multiprocessing.cpu_count(),
                'max_workers': self.max_workers,
                'python_threads': threading.active_count()
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown all processing pools."""
        self.cpu_pool.shutdown()
        self.io_pool.shutdown()
        
        # Stop async loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._loop_thread:
            self._loop_thread.join(timeout=5.0)
    
    def _start_async_loop(self) -> None:
        """Start background async event loop."""
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        
        # Wait for loop to start
        while self._loop is None:
            time.sleep(0.01)


class ResourcePool:
    """Generic resource pool for managing expensive resources."""
    
    def __init__(
        self,
        resource_factory: Callable,
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: int = 300  # 5 minutes
    ):
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        
        # Resource management
        self.available = Queue()
        self.in_use = set()
        self.created_count = 0
        self.last_used = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize minimum resources
        self._initialize_pool()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire a resource from the pool."""
        with self.lock:
            # Try to get existing resource
            try:
                resource = self.available.get_nowait()
                self.in_use.add(resource)
                self.last_used[id(resource)] = time.time()
                return resource
            except:
                pass  # Queue empty
            
            # Create new resource if under limit
            if self.created_count < self.max_size:
                resource = self._create_resource()
                if resource:
                    self.in_use.add(resource)
                    self.last_used[id(resource)] = time.time()
                    return resource
        
        # Wait for resource to become available
        try:
            resource = self.available.get(timeout=timeout)
            with self.lock:
                self.in_use.add(resource)
                self.last_used[id(resource)] = time.time()
            return resource
        except:
            raise RuntimeError("Could not acquire resource within timeout")
    
    def release(self, resource: Any) -> None:
        """Release a resource back to the pool."""
        with self.lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                self.available.put_nowait(resource)
                self.last_used[id(resource)] = time.time()
    
    def size(self) -> Dict[str, int]:
        """Get pool size statistics."""
        with self.lock:
            return {
                'available': self.available.qsize(),
                'in_use': len(self.in_use),
                'total': self.created_count
            }
    
    def _create_resource(self) -> Optional[Any]:
        """Create a new resource."""
        try:
            resource = self.resource_factory()
            self.created_count += 1
            return resource
        except Exception:
            return None
    
    def _initialize_pool(self) -> None:
        """Initialize pool with minimum resources."""
        for _ in range(self.min_size):
            resource = self._create_resource()
            if resource:
                self.available.put_nowait(resource)
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of idle resources."""
        while True:
            time.sleep(60)  # Check every minute
            
            current_time = time.time()
            resources_to_remove = []
            
            with self.lock:
                if self.created_count <= self.min_size:
                    continue
                
                # Check available resources for idle timeout
                temp_resources = []
                while not self.available.empty():
                    try:
                        resource = self.available.get_nowait()
                        resource_id = id(resource)
                        
                        if (current_time - self.last_used.get(resource_id, current_time)) > self.max_idle_time:
                            resources_to_remove.append(resource)
                        else:
                            temp_resources.append(resource)
                    except:
                        break
                
                # Put back non-idle resources
                for resource in temp_resources:
                    self.available.put_nowait(resource)
                
                # Remove idle resources
                for resource in resources_to_remove:
                    resource_id = id(resource)
                    if resource_id in self.last_used:
                        del self.last_used[resource_id]
                    self.created_count -= 1
                    
                    # Cleanup resource if it has a cleanup method
                    if hasattr(resource, 'close'):
                        try:
                            resource.close()
                        except:
                            pass


# Context manager for resource pools
class ResourceContext:
    """Context manager for automatic resource management."""
    
    def __init__(self, pool: ResourcePool, timeout: Optional[float] = None):
        self.pool = pool
        self.timeout = timeout
        self.resource = None
    
    def __enter__(self):
        self.resource = self.pool.acquire(self.timeout)
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.resource:
            self.pool.release(self.resource)