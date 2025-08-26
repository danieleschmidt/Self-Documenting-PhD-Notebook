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

# Autonomous SDLC Generation 1 components
try:
    from .adaptive_research_optimizer import (
        AdaptiveResearchOptimizer,
        GeneticOptimizer,
        PerformanceMetrics
    )
    _adaptive_optimizer_available = True
except ImportError as e:
    print(f"Warning: Adaptive research optimizer unavailable: {e}")
    _adaptive_optimizer_available = False

# Autonomous SDLC Generation 3 components  
try:
    from .quantum_performance_optimizer import (
        QuantumPerformanceOptimizer,
        ResourceController,
        PerformancePredictor
    )
    _quantum_optimizer_available = True
except ImportError as e:
    print(f"Warning: Quantum performance optimizer unavailable: {e}")
    _quantum_optimizer_available = False

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

# Add autonomous performance components if available
if _adaptive_optimizer_available:
    __all__.extend([
        'AdaptiveResearchOptimizer',
        'GeneticOptimizer',
        'PerformanceMetrics'
    ])

if _quantum_optimizer_available:
    __all__.extend([
        'QuantumPerformanceOptimizer',
        'ResourceController',
        'PerformancePredictor'
    ])