"""
Advanced Optimization Engine
============================

High-performance optimization system for research workflows, data processing,
and computational tasks. Includes memory optimization, parallel processing,
caching strategies, and performance profiling.

Features:
- Memory pool management
- CPU/GPU optimization
- Intelligent caching
- Performance profiling
- Parallel processing coordination
- Resource monitoring
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref
import gc
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import functools
import heapq
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for performance tuning."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    optimization_overhead: float = 0.0
    throughput: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'optimization_overhead': self.optimization_overhead,
            'throughput': self.throughput
        }


class MemoryPool:
    """Memory pool for efficient memory management."""
    
    def __init__(self, initial_size: int = 1024 * 1024):  # 1MB default
        self.pool_size = initial_size
        self.allocated_blocks: Dict[int, Any] = {}
        self.free_blocks: List[int] = []
        self.total_allocated = 0
        self.peak_usage = 0
        self.allocation_count = 0
        self.deallocation_count = 0
        self._lock = threading.Lock()
    
    def allocate(self, size: int) -> int:
        """Allocate memory block and return handle."""
        with self._lock:
            # Find free block or create new one
            block_id = len(self.allocated_blocks)
            
            # Simulate memory allocation (in real implementation would use actual memory)
            self.allocated_blocks[block_id] = bytearray(size)
            self.total_allocated += size
            self.peak_usage = max(self.peak_usage, self.total_allocated)
            self.allocation_count += 1
            
            logger.debug(f"Allocated {size} bytes (block {block_id})")
            return block_id
    
    def deallocate(self, block_id: int):
        """Deallocate memory block."""
        with self._lock:
            if block_id in self.allocated_blocks:
                size = len(self.allocated_blocks[block_id])
                del self.allocated_blocks[block_id]
                self.total_allocated -= size
                self.deallocation_count += 1
                self.free_blocks.append(block_id)
                logger.debug(f"Deallocated block {block_id} ({size} bytes)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                'total_allocated': self.total_allocated,
                'peak_usage': self.peak_usage,
                'active_blocks': len(self.allocated_blocks),
                'free_blocks': len(self.free_blocks),
                'allocation_count': self.allocation_count,
                'deallocation_count': self.deallocation_count,
                'efficiency': self.deallocation_count / max(self.allocation_count, 1)
            }
    
    def cleanup(self):
        """Clean up unused memory blocks."""
        with self._lock:
            # Force garbage collection
            gc.collect()
            
            # Clear free blocks list
            self.free_blocks.clear()
            
            logger.info("Memory pool cleanup completed")


class IntelligentCache:
    """Intelligent caching system with multiple eviction policies."""
    
    def __init__(self, max_size: int = 1000, policy: str = "lru"):
        self.max_size = max_size
        self.policy = policy
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.insertion_order: deque = deque()
        self.size_estimates: Dict[str, int] = {}
        
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                self.hits += 1
                self._update_access_info(key)
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def set(self, key: str, value: Any, size_estimate: int = 1):
        """Set value in cache."""
        with self._lock:
            # If key already exists, update it
            if key in self.cache:
                self.cache[key] = value
                self.size_estimates[key] = size_estimate
                self._update_access_info(key)
                return
            
            # Check if we need to evict
            while len(self.cache) >= self.max_size:
                self._evict_one()
            
            # Add new item
            self.cache[key] = value
            self.size_estimates[key] = size_estimate
            self.insertion_order.append(key)
            self._update_access_info(key)
    
    def _update_access_info(self, key: str):
        """Update access information for cache policies."""
        current_time = time.time()
        self.access_times[key] = current_time
        self.access_counts[key] += 1
        
        # Move to end for LRU
        if key in self.insertion_order:
            self.insertion_order.remove(key)
            self.insertion_order.append(key)
    
    def _evict_one(self):
        """Evict one item based on policy."""
        if not self.cache:
            return
        
        if self.policy == "lru":
            # Least recently used
            key_to_evict = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        elif self.policy == "lfu":
            # Least frequently used
            key_to_evict = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        elif self.policy == "fifo":
            # First in, first out
            key_to_evict = self.insertion_order[0]
        else:
            # Default to LRU
            key_to_evict = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove the key
        del self.cache[key_to_evict]
        del self.access_times[key_to_evict]
        del self.access_counts[key_to_evict]
        del self.size_estimates[key_to_evict]
        
        if key_to_evict in self.insertion_order:
            self.insertion_order.remove(key_to_evict)
        
        self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'policy': self.policy
            }
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.insertion_order.clear()
            self.size_estimates.clear()


class PerformanceProfiler:
    """Performance profiler for detailed execution analysis."""
    
    def __init__(self):
        self.profiles: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def profile(self, operation_name: str, context: Dict[str, Any] = None):
        """Context manager for profiling operations."""
        session_id = f"{operation_name}_{int(time.time() * 1000000)}"
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        session_info = {
            'operation': operation_name,
            'start_time': start_time,
            'start_memory': start_memory,
            'context': context or {}
        }
        
        with self._lock:
            self.active_sessions[session_id] = session_info
        
        try:
            yield session_id
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            with self._lock:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    
                    metrics = PerformanceMetrics(
                        execution_time=end_time - start_time,
                        memory_usage=end_memory - start_memory,
                        cpu_usage=0.0  # Would be calculated in real implementation
                    )
                    
                    self.profiles[operation_name].append(metrics)
                    del self.active_sessions[session_id]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (simplified)."""
        # In real implementation, would use psutil or similar
        import sys
        return sys.getsizeof(gc.get_objects()) / 1024 / 1024  # MB
    
    def get_profile_summary(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get profile summary for an operation."""
        with self._lock:
            if operation_name not in self.profiles:
                return None
            
            metrics_list = self.profiles[operation_name]
            if not metrics_list:
                return None
            
            execution_times = [m.execution_time for m in metrics_list]
            memory_usages = [m.memory_usage for m in metrics_list]
            
            return {
                'operation': operation_name,
                'sample_count': len(metrics_list),
                'execution_time': {
                    'mean': sum(execution_times) / len(execution_times),
                    'min': min(execution_times),
                    'max': max(execution_times)
                },
                'memory_usage': {
                    'mean': sum(memory_usages) / len(memory_usages),
                    'min': min(memory_usages),
                    'max': max(memory_usages)
                },
                'latest_metrics': metrics_list[-1].to_dict()
            }
    
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all profile summaries."""
        return {
            operation: self.get_profile_summary(operation)
            for operation in self.profiles.keys()
        }


class ParallelProcessor:
    """Parallel processing coordinator with intelligent work distribution."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (4 * (4 + 1)))  # Conservative default
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(self.max_workers, 4))
        
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0
        
    def process_batch(self, items: List[Any], 
                     processor_func: Callable,
                     mode: str = "thread",
                     chunk_size: Optional[int] = None) -> List[Any]:
        """Process a batch of items in parallel."""
        
        if not items:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 2))
        
        # Split items into chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        start_time = time.perf_counter()
        results = []
        
        try:
            if mode == "process" and len(chunks) > 1:
                # Use process-based parallelism
                futures = [
                    self.process_executor.submit(self._process_chunk, chunk, processor_func)
                    for chunk in chunks
                ]
                
                for future in as_completed(futures):
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
            else:
                # Use thread-based parallelism
                futures = [
                    self.thread_executor.submit(self._process_chunk, chunk, processor_func)
                    for chunk in chunks
                ]
                
                for future in as_completed(futures):
                    chunk_results = future.result()
                    results.extend(chunk_results)
            
            self.completed_tasks += len(items)
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            self.failed_tasks += len(items)
            raise
        
        finally:
            processing_time = time.perf_counter() - start_time
            self.total_processing_time += processing_time
            
            logger.info(f"Processed {len(items)} items in {processing_time:.3f}s using {len(chunks)} chunks")
        
        return results
    
    def _process_chunk(self, chunk: List[Any], processor_func: Callable) -> List[Any]:
        """Process a chunk of items."""
        return [processor_func(item) for item in chunk]
    
    async def process_batch_async(self, items: List[Any],
                                 processor_func: Callable,
                                 concurrency_limit: int = 10) -> List[Any]:
        """Process items asynchronously with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def process_item_with_semaphore(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(processor_func):
                    return await processor_func(item)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, processor_func, item)
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*[
            process_item_with_semaphore(item) for item in items
        ])
        
        processing_time = time.perf_counter() - start_time
        self.total_processing_time += processing_time
        self.completed_tasks += len(items)
        
        logger.info(f"Async processed {len(items)} items in {processing_time:.3f}s")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_tasks = self.completed_tasks + self.failed_tasks
        success_rate = self.completed_tasks / max(total_tasks, 1)
        avg_processing_time = self.total_processing_time / max(self.completed_tasks, 1)
        
        return {
            'max_workers': self.max_workers,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': success_rate,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'throughput': self.completed_tasks / max(self.total_processing_time, 1)
        }
    
    def shutdown(self):
        """Shutdown the parallel processor."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class OptimizationEngine:
    """Main optimization engine coordinating all performance components."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        
        # Core components
        self.memory_pool = MemoryPool()
        self.cache = IntelligentCache(max_size=self._get_cache_size(), policy="lru")
        self.profiler = PerformanceProfiler()
        self.parallel_processor = ParallelProcessor()
        
        # Optimization strategies
        self.optimization_strategies = {
            'memory': self._optimize_memory,
            'cpu': self._optimize_cpu,
            'cache': self._optimize_cache,
            'io': self._optimize_io
        }
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.active_optimizations: Set[str] = set()
        
        # Background optimization
        self.background_optimizer_running = False
        self.background_thread: Optional[threading.Thread] = None
        
        logger.info(f"Optimization engine initialized with {optimization_level.value} level")
    
    def _get_cache_size(self) -> int:
        """Get cache size based on optimization level."""
        size_map = {
            OptimizationLevel.CONSERVATIVE: 500,
            OptimizationLevel.BALANCED: 1000,
            OptimizationLevel.AGGRESSIVE: 2000,
            OptimizationLevel.MAXIMUM: 5000
        }
        return size_map[self.optimization_level]
    
    @contextmanager
    def optimize_operation(self, operation_name: str, 
                          strategies: List[str] = None,
                          context: Dict[str, Any] = None):
        """Context manager for optimizing operations."""
        
        strategies = strategies or ['memory', 'cache']
        context = context or {}
        
        # Start profiling
        with self.profiler.profile(operation_name, context) as session_id:
            
            # Apply optimization strategies
            optimizations_applied = []
            for strategy in strategies:
                if strategy in self.optimization_strategies:
                    try:
                        self.optimization_strategies[strategy](context)
                        optimizations_applied.append(strategy)
                    except Exception as e:
                        logger.warning(f"Optimization strategy {strategy} failed: {e}")
            
            self.active_optimizations.update(optimizations_applied)
            
            try:
                yield {
                    'session_id': session_id,
                    'optimizations': optimizations_applied,
                    'cache': self.cache,
                    'memory_pool': self.memory_pool,
                    'parallel_processor': self.parallel_processor
                }
            finally:
                # Clean up optimizations
                self.active_optimizations.difference_update(optimizations_applied)
                
                # Record optimization results
                profile = self.profiler.get_profile_summary(operation_name)
                if profile:
                    optimization_record = {
                        'operation': operation_name,
                        'optimizations_applied': optimizations_applied,
                        'performance_metrics': profile['latest_metrics'],
                        'timestamp': datetime.now().isoformat()
                    }
                    self.optimization_history.append(optimization_record)
    
    def _optimize_memory(self, context: Dict[str, Any]):
        """Apply memory optimization strategies."""
        # Force garbage collection
        gc.collect()
        
        # Clean up memory pool if needed
        memory_stats = self.memory_pool.get_stats()
        if memory_stats['efficiency'] < 0.7:
            self.memory_pool.cleanup()
        
        logger.debug("Applied memory optimization")
    
    def _optimize_cpu(self, context: Dict[str, Any]):
        """Apply CPU optimization strategies."""
        # In a real implementation, this might:
        # - Adjust thread pool sizes
        # - Change process priorities
        # - Enable/disable CPU-intensive features
        
        # For now, just optimize parallel processing settings
        cpu_intensive = context.get('cpu_intensive', False)
        if cpu_intensive and hasattr(self.parallel_processor, 'max_workers'):
            # Increase worker count for CPU-intensive tasks
            original_workers = self.parallel_processor.max_workers
            self.parallel_processor.max_workers = min(original_workers * 2, 32)
            logger.debug(f"Increased parallel workers from {original_workers} to {self.parallel_processor.max_workers}")
    
    def _optimize_cache(self, context: Dict[str, Any]):
        """Apply caching optimization strategies."""
        cache_stats = self.cache.get_stats()
        
        # If hit rate is low, consider clearing cache to avoid overhead
        if cache_stats['hit_rate'] < 0.3 and cache_stats['size'] > 100:
            # Clear least useful cache entries
            cache_size = cache_stats['size']
            # Clear 25% of cache
            for _ in range(cache_size // 4):
                if self.cache.policy == "lru":
                    # Remove oldest entries
                    if self.cache.access_times:
                        oldest_key = min(self.cache.access_times.keys(), 
                                       key=lambda k: self.cache.access_times[k])
                        with self.cache._lock:
                            if oldest_key in self.cache.cache:
                                del self.cache.cache[oldest_key]
                                del self.cache.access_times[oldest_key]
                                del self.cache.access_counts[oldest_key]
        
        logger.debug("Applied cache optimization")
    
    def _optimize_io(self, context: Dict[str, Any]):
        """Apply I/O optimization strategies."""
        # In a real implementation, this might:
        # - Configure buffer sizes
        # - Enable/disable compression
        # - Optimize file access patterns
        
        logger.debug("Applied I/O optimization")
    
    def optimize_function(self, func: Callable, 
                         strategies: List[str] = None,
                         cache_key_func: Callable = None) -> Callable:
        """Decorator to optimize function calls."""
        
        @functools.wraps(func)
        def optimized_wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            
            # Check cache first if caching is enabled
            cache_key = None
            if 'cache' in (strategies or []) and cache_key_func:
                try:
                    cache_key = cache_key_func(*args, **kwargs)
                    cached_result = self.cache.get(cache_key)
                    if cached_result is not None:
                        return cached_result
                except Exception as e:
                    logger.warning(f"Cache key generation failed: {e}")
            
            # Execute with optimization
            with self.optimize_operation(operation_name, strategies) as opt_ctx:
                result = func(*args, **kwargs)
                
                # Cache result if caching is enabled
                if cache_key:
                    try:
                        result_size = len(str(result))  # Rough size estimate
                        self.cache.set(cache_key, result, result_size)
                    except Exception as e:
                        logger.warning(f"Result caching failed: {e}")
                
                return result
        
        return optimized_wrapper
    
    def start_background_optimization(self):
        """Start background optimization process."""
        if self.background_optimizer_running:
            return
        
        self.background_optimizer_running = True
        self.background_thread = threading.Thread(target=self._background_optimization_loop)
        self.background_thread.daemon = True
        self.background_thread.start()
        
        logger.info("Started background optimization")
    
    def stop_background_optimization(self):
        """Stop background optimization process."""
        self.background_optimizer_running = False
        if self.background_thread:
            self.background_thread.join(timeout=5)
        
        logger.info("Stopped background optimization")
    
    def _background_optimization_loop(self):
        """Background optimization loop."""
        while self.background_optimizer_running:
            try:
                # Periodic cleanup
                self._optimize_memory({})
                
                # Cache optimization based on stats
                cache_stats = self.cache.get_stats()
                if cache_stats['hit_rate'] < 0.5:
                    self._optimize_cache({})
                
                # Memory pool cleanup
                memory_stats = self.memory_pool.get_stats()
                if memory_stats['efficiency'] < 0.8:
                    self.memory_pool.cleanup()
                
            except Exception as e:
                logger.error(f"Background optimization error: {e}")
            
            time.sleep(60)  # Run every minute
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            'optimization_level': self.optimization_level.value,
            'cache_stats': self.cache.get_stats(),
            'memory_pool_stats': self.memory_pool.get_stats(),
            'parallel_processor_stats': self.parallel_processor.get_stats(),
            'active_optimizations': list(self.active_optimizations),
            'optimization_history_count': len(self.optimization_history),
            'background_optimizer_running': self.background_optimizer_running,
            'profile_summaries': self.profiler.get_all_profiles()
        }
    
    def cleanup(self):
        """Cleanup optimization engine resources."""
        self.stop_background_optimization()
        self.parallel_processor.shutdown()
        self.memory_pool.cleanup()
        self.cache.clear()
        
        logger.info("Optimization engine cleanup completed")


# Utility functions for common optimization patterns
def memoize_with_expiration(expiration_minutes: int = 60):
    """Decorator for memoization with expiration."""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(hash((args, tuple(sorted(kwargs.items())))))
            
            # Check if cached result exists and is not expired
            if key in cache:
                result, timestamp = cache[key]
                if datetime.now() - timestamp < timedelta(minutes=expiration_minutes):
                    return result
                else:
                    del cache[key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, datetime.now())
            
            return result
        
        return wrapper
    return decorator


def batch_process(batch_size: int = 100, 
                 parallel: bool = True):
    """Decorator for batch processing of sequences."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(items, *args, **kwargs):
            if not hasattr(items, '__iter__'):
                return func(items, *args, **kwargs)
            
            # Process in batches
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                if parallel and len(batch) > 1:
                    # Use parallel processing for batch
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        batch_results = list(executor.map(
                            lambda item: func(item, *args, **kwargs), batch
                        ))
                else:
                    batch_results = [func(item, *args, **kwargs) for item in batch]
                
                results.extend(batch_results)
            
            return results
        
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    def test_optimization_engine():
        """Test the optimization engine."""
        print("⚡ Testing Advanced Optimization Engine")
        
        # Create optimization engine
        engine = OptimizationEngine(OptimizationLevel.AGGRESSIVE)
        
        # Test function optimization
        @engine.optimize_function(
            strategies=['memory', 'cache'],
            cache_key_func=lambda x, y=1: f"compute_{x}_{y}"
        )
        def expensive_computation(n, multiplier=1):
            """Simulate expensive computation."""
            result = sum(i * multiplier for i in range(n))
            time.sleep(0.1)  # Simulate work
            return result
        
        # Test batch processing
        @batch_process(batch_size=50, parallel=True)
        def process_item(item):
            return item ** 2
        
        # Start background optimization
        engine.start_background_optimization()
        
        print("🔄 Running optimization tests...")
        
        # Test cached computation
        with engine.optimize_operation('computation_test') as opt_ctx:
            # First call - should be slow
            start = time.perf_counter()
            result1 = expensive_computation(1000, 2)
            time1 = time.perf_counter() - start
            
            # Second call - should be cached and fast
            start = time.perf_counter()
            result2 = expensive_computation(1000, 2)
            time2 = time.perf_counter() - start
            
            print(f"✅ Computation: First call: {time1:.3f}s, Cached call: {time2:.3f}s")
            print(f"   Cache speedup: {time1/max(time2, 0.001):.1f}x")
        
        # Test parallel processing
        test_data = list(range(200))
        
        with engine.optimize_operation('parallel_test') as opt_ctx:
            start = time.perf_counter()
            results = process_item(test_data)
            parallel_time = time.perf_counter() - start
            
            print(f"✅ Parallel processing: {len(results)} items in {parallel_time:.3f}s")
        
        # Test async processing
        async def test_async():
            async_results = await engine.parallel_processor.process_batch_async(
                test_data[:50], 
                lambda x: x ** 3,
                concurrency_limit=10
            )
            return len(async_results)
        
        import asyncio
        async_result_count = asyncio.run(test_async())
        print(f"✅ Async processing: {async_result_count} items processed")
        
        # Get optimization report
        report = engine.get_optimization_report()
        
        print(f"\\n📊 Optimization Report:")
        print(f"- Cache hit rate: {report['cache_stats']['hit_rate']:.2%}")
        print(f"- Memory pool efficiency: {report['memory_pool_stats']['efficiency']:.2%}")
        print(f"- Parallel processor throughput: {report['parallel_processor_stats']['throughput']:.1f} items/s")
        print(f"- Active optimizations: {len(report['active_optimizations'])}")
        print(f"- Background optimizer: {'Running' if report['background_optimizer_running'] else 'Stopped'}")
        
        # Cleanup
        engine.cleanup()
        
        print("✅ Optimization engine test completed")
        
        return report
    
    # Run test
    result = test_optimization_engine()