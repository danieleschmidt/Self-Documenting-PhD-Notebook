"""
High-performance components for the PhD notebook.
"""

from .caching import (
    LRUCache, AsyncCache, DiskCache, MultiLevelCache,
    cache_result, get_memory_cache, get_disk_cache, get_multi_cache
)

from .async_processing import (
    AsyncTaskManager, BatchProcessor, AsyncQueue, WorkerPool,
    run_in_thread, gather_with_concurrency, AsyncRateLimiter,
    get_task_manager, get_rate_limiter
)

from .indexing import SearchIndex, get_search_index

__all__ = [
    # Caching
    'LRUCache', 'AsyncCache', 'DiskCache', 'MultiLevelCache',
    'cache_result', 'get_memory_cache', 'get_disk_cache', 'get_multi_cache',
    
    # Async processing
    'AsyncTaskManager', 'BatchProcessor', 'AsyncQueue', 'WorkerPool',
    'run_in_thread', 'gather_with_concurrency', 'AsyncRateLimiter',
    'get_task_manager', 'get_rate_limiter',
    
    # Search indexing
    'SearchIndex', 'get_search_index'
]