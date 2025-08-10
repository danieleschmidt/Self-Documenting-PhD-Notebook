"""
High-performance caching system for PhD notebook.
"""

import asyncio
import json
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import weakref
import threading

from ..utils.exceptions import MetricsError


class CacheStats:
    """Cache statistics tracking."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.size = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def hit(self):
        with self._lock:
            self.hits += 1
    
    def miss(self):
        with self._lock:
            self.misses += 1
    
    def set(self):
        with self._lock:
            self.sets += 1
    
    def evict(self):
        with self._lock:
            self.evictions += 1
    
    def update_size(self, size: int):
        with self._lock:
            self.size = size
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            uptime = time.time() - self.start_time
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'sets': self.sets,
                'evictions': self.evictions,
                'hit_rate': round(hit_rate, 2),
                'size': self.size,
                'uptime_seconds': round(uptime, 2)
            }


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, access_time, create_time)
        self.access_order: List[str] = []
        self._lock = threading.RLock()
        self.stats = CacheStats()
    
    def _make_key(self, key: Any) -> str:
        """Convert key to string representation."""
        if isinstance(key, str):
            return key
        elif isinstance(key, (int, float, bool)):
            return str(key)
        elif isinstance(key, (list, tuple, dict)):
            return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.sha256(str(key).encode()).hexdigest()
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get value from cache."""
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key not in self.cache:
                self.stats.miss()
                return default
            
            value, _, create_time = self.cache[str_key]
            current_time = time.time()
            
            # Check TTL expiration
            if self.ttl_seconds and (current_time - create_time) > self.ttl_seconds:
                del self.cache[str_key]
                self.access_order.remove(str_key)
                self.stats.miss()
                self.stats.evict()
                return default
            
            # Update access time and order
            self.cache[str_key] = (value, current_time, create_time)
            self.access_order.remove(str_key)
            self.access_order.append(str_key)
            
            self.stats.hit()
            return value
    
    def set(self, key: Any, value: Any) -> None:
        """Set value in cache."""
        str_key = self._make_key(key)
        current_time = time.time()
        
        with self._lock:
            # If key already exists, update it
            if str_key in self.cache:
                self.cache[str_key] = (value, current_time, current_time)
                self.access_order.remove(str_key)
                self.access_order.append(str_key)
            else:
                # Add new entry
                self.cache[str_key] = (value, current_time, current_time)
                self.access_order.append(str_key)
                
                # Evict if over capacity
                while len(self.cache) > self.max_size:
                    oldest_key = self.access_order.pop(0)
                    del self.cache[oldest_key]
                    self.stats.evict()
            
            self.stats.set()
            self.stats.update_size(len(self.cache))
    
    def delete(self, key: Any) -> bool:
        """Delete key from cache."""
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key in self.cache:
                del self.cache[str_key]
                self.access_order.remove(str_key)
                self.stats.update_size(len(self.cache))
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats.update_size(0)
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class AsyncCache:
    """Asynchronous cache with background refresh."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = LRUCache(max_size, ttl_seconds)
        self.refresh_tasks: Dict[str, asyncio.Task] = {}
        self.refresh_functions: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: Any, refresh_func: Optional[Callable] = None) -> Any:
        """Get value from cache with optional background refresh."""
        value = self.cache.get(key)
        
        if value is None and refresh_func:
            # Cache miss - fetch value
            value = await self._call_refresh_func(refresh_func, key)
            self.cache.set(key, value)
        elif refresh_func:
            # Cache hit - maybe refresh in background
            str_key = self.cache._make_key(key)
            if str_key not in self.refresh_tasks:
                # Start background refresh
                task = asyncio.create_task(self._background_refresh(key, refresh_func))
                self.refresh_tasks[str_key] = task
        
        return value
    
    async def set(self, key: Any, value: Any) -> None:
        """Set value in cache."""
        self.cache.set(key, value)
    
    async def _call_refresh_func(self, func: Callable, key: Any) -> Any:
        """Call refresh function (sync or async)."""
        if asyncio.iscoroutinefunction(func):
            return await func(key)
        else:
            # Run in thread pool for blocking functions
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, func, key)
    
    async def _background_refresh(self, key: Any, refresh_func: Callable) -> None:
        """Refresh cache value in background."""
        try:
            # Add some jitter to avoid thundering herd
            await asyncio.sleep(0.1 + (hash(key) % 100) / 1000)
            
            new_value = await self._call_refresh_func(refresh_func, key)
            self.cache.set(key, new_value)
            
        except Exception as e:
            # Log error but don't crash
            print(f"Background refresh failed for key {key}: {e}")
        finally:
            # Clean up task reference
            str_key = self.cache._make_key(key)
            self.refresh_tasks.pop(str_key, None)


class DiskCache:
    """Persistent cache using disk storage."""
    
    def __init__(self, cache_dir: Path, max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.stats = CacheStats()
        self._lock = threading.Lock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash[:2]}" / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Any:
        """Get value from disk cache."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            self.stats.miss()
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check expiration
            if 'expires_at' in data and data['expires_at'] is not None and time.time() > data['expires_at']:
                file_path.unlink()
                self.stats.miss()
                self.stats.evict()
                return None
            
            # Update access time
            data['accessed_at'] = time.time()
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.stats.hit()
            return data['value']
            
        except Exception:
            # Remove corrupted file
            file_path.unlink(missing_ok=True)
            self.stats.miss()
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in disk cache."""
        file_path = self._get_file_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'value': value,
            'created_at': time.time(),
            'accessed_at': time.time(),
            'expires_at': time.time() + ttl_seconds if ttl_seconds else None
        }
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.stats.set()
            
            # Check if we need to evict old files
            self._maybe_evict()
            
        except Exception as e:
            raise MetricsError(f"Failed to write to disk cache: {e}")
    
    def _maybe_evict(self) -> None:
        """Evict old files if cache is too large."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*.pkl'))
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return
        
        # Get all cache files with their access times
        files_with_access = []
        for file_path in self.cache_dir.rglob('*.pkl'):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                files_with_access.append((file_path, data.get('accessed_at', 0)))
            except Exception:
                # Remove corrupted files
                file_path.unlink(missing_ok=True)
        
        # Sort by access time (oldest first)
        files_with_access.sort(key=lambda x: x[1])
        
        # Remove oldest files until under limit
        current_size = total_size
        for file_path, _ in files_with_access:
            if current_size <= max_size_bytes:
                break
            
            file_size = file_path.stat().st_size
            file_path.unlink()
            current_size -= file_size
            self.stats.evict()
    
    def clear(self) -> None:
        """Clear all cached files."""
        for file_path in self.cache_dir.rglob('*.pkl'):
            file_path.unlink()
        
        # Remove empty directories
        for dir_path in self.cache_dir.rglob('*'):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                dir_path.rmdir()


class MultiLevelCache:
    """Multi-level cache combining memory and disk."""
    
    def __init__(
        self, 
        memory_size: int = 1000,
        disk_size_mb: int = 100,
        cache_dir: Path = None,
        ttl_seconds: int = 3600
    ):
        self.memory_cache = LRUCache(memory_size, ttl_seconds)
        
        if cache_dir is None:
            cache_dir = Path.home() / '.phd-notebook' / 'cache'
        
        self.disk_cache = DiskCache(cache_dir, disk_size_mb)
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: Any) -> Any:
        """Get value from multi-level cache."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        str_key = self.memory_cache._make_key(key)
        value = self.disk_cache.get(str_key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: Any, value: Any) -> None:
        """Set value in multi-level cache."""
        # Set in both caches
        self.memory_cache.set(key, value)
        str_key = self.memory_cache._make_key(key)
        self.disk_cache.set(str_key, value, self.ttl_seconds)
    
    def delete(self, key: Any) -> bool:
        """Delete from both cache levels."""
        str_key = self.memory_cache._make_key(key)
        
        memory_deleted = self.memory_cache.delete(key)
        
        disk_file = self.disk_cache._get_file_path(str_key)
        disk_deleted = False
        if disk_file.exists():
            disk_file.unlink()
            disk_deleted = True
        
        return memory_deleted or disk_deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            'memory': self.memory_cache.stats.get_stats(),
            'disk': self.disk_cache.stats.get_stats()
        }


def cache_result(ttl_seconds: int = 3600, max_size: int = 100):
    """Decorator to cache function results."""
    cache = LRUCache(max_size, ttl_seconds)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': sorted(kwargs.items())
            }
            
            # Try to get from cache
            result = cache.get(key_data)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(key_data, result)
            
            return result
        
        # Add cache control methods
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = lambda: cache.stats.get_stats()
        
        return wrapper
    
    return decorator


# Global cache instances
_global_memory_cache = LRUCache(max_size=10000, ttl_seconds=3600)
_global_disk_cache = None
_global_multi_cache = None

def get_memory_cache() -> LRUCache:
    """Get global memory cache."""
    return _global_memory_cache

def get_disk_cache(cache_dir: Path = None) -> DiskCache:
    """Get global disk cache."""
    global _global_disk_cache
    
    if _global_disk_cache is None:
        if cache_dir is None:
            cache_dir = Path.home() / '.phd-notebook' / 'cache'
        _global_disk_cache = DiskCache(cache_dir)
    
    return _global_disk_cache

def get_multi_cache(cache_dir: Path = None) -> MultiLevelCache:
    """Get global multi-level cache."""
    global _global_multi_cache
    
    if _global_multi_cache is None:
        _global_multi_cache = MultiLevelCache(cache_dir=cache_dir)
    
    return _global_multi_cache