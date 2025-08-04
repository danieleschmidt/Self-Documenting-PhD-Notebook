"""
High-performance memory caching for the PhD notebook system.
"""

import asyncio
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set
from threading import RLock
import weakref
import pickle
import hashlib

from ..utils.logging import get_logger, log_performance


class CacheEntry:
    """Represents a single cache entry."""
    
    def __init__(self, key: str, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.ttl = ttl
        self.size = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of cached value."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback for non-serializable objects
            return len(str(value).encode('utf-8'))
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self) -> Any:
        """Mark as accessed and return value."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value


class MemoryCache:
    """
    High-performance LRU cache with TTL and size limits.
    
    Features:
    - LRU eviction policy
    - TTL (time-to-live) support
    - Memory usage tracking
    - Thread-safe operations
    - Performance metrics
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[float] = None,
        cleanup_interval: float = 300  # 5 minutes
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._total_size = 0
        self._last_cleanup = time.time()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0
        
        self.logger = get_logger(__name__)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return default
            
            if entry.is_expired():
                self._remove_entry(key)
                self._expired += 1
                self._misses += 1
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            
            return entry.touch()
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        if ttl is None:
            ttl = self.default_ttl
        
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(key, value, ttl)
            
            # Check if single entry exceeds memory limit
            if entry.size > self.max_memory_bytes:
                self.logger.warning(f"Cache entry too large: {entry.size} bytes")
                return
            
            # Ensure we have space
            self._ensure_space(entry.size)
            
            # Add to cache
            self._cache[key] = entry
            self._total_size += entry.size
            
            # Periodic cleanup
            if time.time() - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired()
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_size = 0
            self.logger.info("Cache cleared")
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'expired': self._expired,
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._total_size / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            }
    
    def _ensure_space(self, needed_size: int) -> None:
        """Ensure cache has space for new entry."""
        # Remove expired entries first
        self._cleanup_expired()
        
        # Check size limit
        while len(self._cache) >= self.max_size:
            self._evict_lru()
        
        # Check memory limit
        while self._total_size + needed_size > self.max_memory_bytes:
            if not self._cache:
                break
            self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove_entry(key)
            self._evictions += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        entry = self._cache.pop(key, None)
        if entry:
            self._total_size -= entry.size
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
            self._expired += 1
        
        self._last_cleanup = current_time
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class CacheManager:
    """
    Manages multiple specialized caches for different data types.
    """
    
    def __init__(self):
        self.caches: Dict[str, MemoryCache] = {}
        self.logger = get_logger(__name__)
        
        # Create specialized caches
        self._setup_default_caches()
    
    def _setup_default_caches(self) -> None:
        """Setup default caches for different data types."""
        
        # Note content cache - larger, longer TTL
        self.caches['notes'] = MemoryCache(
            max_size=5000,
            max_memory_mb=200,
            default_ttl=3600,  # 1 hour
            cleanup_interval=300
        )
        
        # Search results cache - smaller, shorter TTL
        self.caches['search'] = MemoryCache(
            max_size=1000,
            max_memory_mb=50,
            default_ttl=300,  # 5 minutes
            cleanup_interval=60
        )
        
        # Knowledge graph cache - medium size, medium TTL
        self.caches['graph'] = MemoryCache(
            max_size=2000,
            max_memory_mb=100,
            default_ttl=1800,  # 30 minutes
            cleanup_interval=300
        )
        
        # File system metadata cache - small, short TTL
        self.caches['fs_metadata'] = MemoryCache(
            max_size=10000,
            max_memory_mb=20,
            default_ttl=60,  # 1 minute
            cleanup_interval=30
        )
        
        # Agent results cache - small, medium TTL
        self.caches['agents'] = MemoryCache(
            max_size=500,
            max_memory_mb=50,
            default_ttl=600,  # 10 minutes
            cleanup_interval=120
        )
    
    def get_cache(self, cache_name: str) -> Optional[MemoryCache]:
        """Get a specific cache by name."""
        return self.caches.get(cache_name)
    
    def create_cache(
        self,
        name: str,
        max_size: int = 1000,
        max_memory_mb: int = 50,
        default_ttl: Optional[float] = None
    ) -> MemoryCache:
        """Create a new named cache."""
        cache = MemoryCache(max_size, max_memory_mb, default_ttl)
        self.caches[name] = cache
        return cache
    
    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        self.logger.info("All caches cleared")
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}
        total_memory = 0
        total_entries = 0
        total_hits = 0
        total_requests = 0
        
        for name, cache in self.caches.items():
            cache_stats = cache.stats()
            stats[name] = cache_stats
            
            total_memory += cache_stats['memory_usage_mb']
            total_entries += cache_stats['size']
            total_hits += cache_stats['hits']
            total_requests += cache_stats['hits'] + cache_stats['misses']
        
        stats['global'] = {
            'total_caches': len(self.caches),
            'total_entries': total_entries,
            'total_memory_mb': total_memory,
            'global_hit_rate': total_hits / total_requests if total_requests > 0 else 0,
        }
        
        return stats


# Cache decorators
def cached(cache_name: str = 'default', ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create cache manager
            if not hasattr(wrapper, '_cache_manager'):
                wrapper._cache_manager = CacheManager()
            
            cache = wrapper._cache_manager.get_cache(cache_name) or wrapper._cache_manager.create_cache(cache_name)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            with log_performance(f"cache_miss_{func.__name__}"):
                result = func(*args, **kwargs)
                cache.set(cache_key, result, ttl)
                return result
        
        return wrapper
    return decorator


def cache_invalidate(cache_name: str, pattern: Optional[str] = None):
    """Decorator to invalidate cache entries."""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Get cache manager
            if hasattr(func, '_cache_manager'):
                cache = func._cache_manager.get_cache(cache_name)
                if cache:
                    if pattern:
                        # Invalidate matching keys
                        keys_to_delete = [k for k in cache.keys() if pattern in k]
                        for key in keys_to_delete:
                            cache.delete(key)
                    else:
                        # Clear entire cache
                        cache.clear()
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
_global_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager