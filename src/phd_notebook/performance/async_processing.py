"""
Asynchronous processing and concurrency utilities.
"""

import asyncio
import concurrent.futures
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime
import functools
import threading
import queue
import time
from dataclasses import dataclass

from ..utils.exceptions import MetricsError


T = TypeVar('T')


@dataclass
class TaskResult:
    """Result of an async task execution."""
    task_id: str
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    completed_at: datetime = None


class AsyncTaskManager:
    """Manages concurrent async task execution."""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self._task_counter = 0
        self._lock = asyncio.Lock()
    
    async def submit_task(
        self, 
        coro: Awaitable[T], 
        task_id: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """Submit an async task for execution."""
        if task_id is None:
            task_id = f"task_{self._task_counter}"
            self._task_counter += 1
        
        async with self._lock:
            if task_id in self.active_tasks:
                raise ValueError(f"Task {task_id} already active")
            
            # Create task with semaphore for concurrency control
            task = asyncio.create_task(self._execute_with_semaphore(coro, task_id))
            self.active_tasks[task_id] = task
        
        return task_id
    
    async def _execute_with_semaphore(self, coro: Awaitable[T], task_id: str) -> T:
        """Execute coroutine with semaphore control."""
        async with self.semaphore:
            start_time = time.time()
            result = TaskResult(task_id=task_id)
            
            try:
                result.result = await coro
            except Exception as e:
                result.error = e
            finally:
                result.duration = time.time() - start_time
                result.completed_at = datetime.now()
                
                # Move from active to completed
                async with self._lock:
                    self.active_tasks.pop(task_id, None)
                    self.completed_tasks[task_id] = result
                
                return result.result
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Wait for a specific task to complete."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        try:
            await asyncio.wait_for(self.active_tasks[task_id], timeout=timeout)
        except asyncio.TimeoutError:
            # Cancel the task
            self.active_tasks[task_id].cancel()
            async with self._lock:
                self.active_tasks.pop(task_id, None)
            raise
        
        return self.completed_tasks[task_id]
    
    async def wait_for_all(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for all active tasks to complete."""
        if not self.active_tasks:
            return dict(self.completed_tasks)
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.active_tasks.values(), return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            # Cancel all remaining tasks
            for task in self.active_tasks.values():
                task.cancel()
            async with self._lock:
                self.active_tasks.clear()
            raise
        
        return dict(self.completed_tasks)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current task manager status."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'max_concurrent': self.max_concurrent_tasks,
            'available_slots': self.semaphore._value
        }
    
    def cleanup_completed(self, keep_last_n: int = 100) -> None:
        """Clean up old completed task results."""
        if len(self.completed_tasks) <= keep_last_n:
            return
        
        # Keep most recent results
        sorted_tasks = sorted(
            self.completed_tasks.items(),
            key=lambda x: x[1].completed_at,
            reverse=True
        )
        
        self.completed_tasks = dict(sorted_tasks[:keep_last_n])


class BatchProcessor:
    """Processes items in batches with configurable concurrency."""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 3,
        process_timeout: float = 300.0
    ):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.process_timeout = process_timeout
        self.task_manager = AsyncTaskManager(max_concurrent_batches)
    
    async def process_items(
        self,
        items: List[Any],
        processor: Callable[[Any], Awaitable[Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Process items in batches."""
        if not items:
            return []
        
        # Split into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        results = []
        completed = 0
        
        # Process batches concurrently
        batch_tasks = []
        for i, batch in enumerate(batches):
            task_id = await self.task_manager.submit_task(
                self._process_batch(batch, processor),
                f"batch_{i}"
            )
            batch_tasks.append(task_id)
        
        # Collect results as they complete
        for task_id in batch_tasks:
            try:
                task_result = await self.task_manager.wait_for_task(
                    task_id, 
                    timeout=self.process_timeout
                )
                
                if task_result.error:
                    raise task_result.error
                
                results.extend(task_result.result)
                completed += len(task_result.result)
                
                # Report progress
                if progress_callback:
                    progress_callback(completed, len(items))
                    
            except Exception as e:
                print(f"Batch processing error in {task_id}: {e}")
                # Continue processing other batches
        
        return results
    
    async def _process_batch(self, batch: List[Any], processor: Callable) -> List[Any]:
        """Process a single batch of items."""
        batch_results = []
        
        # Process items in batch concurrently
        tasks = [processor(item) for item in batch]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"Item processing error: {result}")
                    batch_results.append(None)  # Placeholder for failed item
                else:
                    batch_results.append(result)
                    
        except Exception as e:
            print(f"Batch processing error: {e}")
            batch_results = [None] * len(batch)
        
        return batch_results


class AsyncQueue:
    """High-performance async queue with priority support."""
    
    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self._queue = asyncio.PriorityQueue(maxsize)
        self._item_counter = 0
        self._processed = 0
        self._errors = 0
    
    async def put(self, item: Any, priority: int = 0) -> None:
        """Put item in queue with priority (lower number = higher priority)."""
        # Use counter to maintain FIFO order for same priority
        await self._queue.put((priority, self._item_counter, item))
        self._item_counter += 1
    
    async def get(self) -> Any:
        """Get next item from queue."""
        priority, counter, item = await self._queue.get()
        return item
    
    def task_done(self) -> None:
        """Mark a task as done."""
        self._queue.task_done()
        self._processed += 1
    
    def task_failed(self) -> None:
        """Mark a task as failed."""
        self._queue.task_done()
        self._errors += 1
    
    async def join(self) -> None:
        """Wait until all tasks are processed."""
        await self._queue.join()
    
    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            'queue_size': self.qsize(),
            'processed': self._processed,
            'errors': self._errors,
            'total_items': self._item_counter
        }


class WorkerPool:
    """Pool of async workers processing from a queue."""
    
    def __init__(
        self,
        queue: AsyncQueue,
        num_workers: int = 5,
        worker_timeout: float = 30.0
    ):
        self.queue = queue
        self.num_workers = num_workers
        self.worker_timeout = worker_timeout
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self._worker_processor: Optional[Callable] = None
    
    async def start(self, processor: Callable[[Any], Awaitable[Any]]) -> None:
        """Start the worker pool."""
        if self.is_running:
            raise RuntimeError("Worker pool already running")
        
        self._worker_processor = processor
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.num_workers):
            worker_task = asyncio.create_task(
                self._worker(f"worker_{i}"),
                name=f"worker_{i}"
            )
            self.workers.append(worker_task)
        
        print(f"Started {self.num_workers} workers")
    
    async def stop(self, timeout: float = 10.0) -> None:
        """Stop the worker pool."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            print("Warning: Some workers didn't stop gracefully")
        
        self.workers.clear()
        print("Worker pool stopped")
    
    async def _worker(self, worker_name: str) -> None:
        """Individual worker coroutine."""
        print(f"Started {worker_name}")
        
        while self.is_running:
            try:
                # Get work item with timeout
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0  # Short timeout to check is_running regularly
                )
                
                # Process the item
                try:
                    await asyncio.wait_for(
                        self._worker_processor(item),
                        timeout=self.worker_timeout
                    )
                    self.queue.task_done()
                    
                except asyncio.TimeoutError:
                    print(f"{worker_name}: Task timeout")
                    self.queue.task_failed()
                    
                except Exception as e:
                    print(f"{worker_name}: Task error: {e}")
                    self.queue.task_failed()
                    
            except asyncio.TimeoutError:
                # Normal timeout - continue to check is_running
                continue
                
            except asyncio.CancelledError:
                print(f"{worker_name} cancelled")
                break
                
            except Exception as e:
                print(f"{worker_name}: Unexpected error: {e}")
        
        print(f"Stopped {worker_name}")


def run_in_thread(func: Callable[..., T], *args, **kwargs) -> Awaitable[T]:
    """Run a blocking function in a thread pool."""
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor()
    
    return loop.run_in_executor(executor, functools.partial(func, **kwargs), *args)


async def gather_with_concurrency(
    n: int, 
    *coros: Awaitable[T],
    return_exceptions: bool = False
) -> List[T]:
    """Run coroutines with limited concurrency."""
    semaphore = asyncio.Semaphore(n)
    
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(
        *(sem_coro(coro) for coro in coros),
        return_exceptions=return_exceptions
    )


class AsyncRateLimiter:
    """Rate limiter for async operations."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a call."""
        async with self._lock:
            now = time.time()
            
            # Remove calls outside time window
            cutoff = now - self.time_window
            self.calls = [call_time for call_time in self.calls if call_time > cutoff]
            
            # Check if we can make a call
            if len(self.calls) >= self.max_calls:
                # Need to wait
                oldest_call = min(self.calls)
                wait_time = oldest_call + self.time_window - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    return await self.acquire()  # Recursive call
            
            # Record this call
            self.calls.append(now)
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# Global instances
_global_task_manager = AsyncTaskManager(max_concurrent_tasks=20)
_global_rate_limiter = AsyncRateLimiter(max_calls=100, time_window=60.0)

def get_task_manager() -> AsyncTaskManager:
    """Get global async task manager."""
    return _global_task_manager

def get_rate_limiter() -> AsyncRateLimiter:
    """Get global rate limiter."""
    return _global_rate_limiter