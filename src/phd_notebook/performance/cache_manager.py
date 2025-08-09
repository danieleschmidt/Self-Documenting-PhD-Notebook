"""
Advanced caching system for performance optimization.
"""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, Callable, Tuple
from pathlib import Path
import threading
import weakref


class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get number of items in cache."""
        pass


class LRUCache(BaseCache):
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value and move to end (mark as recently used)."""
        with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                self._cache[key] = value  # Move to end
                return value
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                # Remove oldest item
                self._cache.popitem(last=False)
            
            self._cache[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            return self._cache.pop(key, None) is not None
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)


class TTLCache(BaseCache):
    """Time-to-Live cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value if not expired."""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check if expired
            if self._is_expired(key):
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with TTL."""
        with self._lock:
            # Clean up expired entries if cache is full
            if len(self._cache) >= self.max_size:
                self._cleanup_expired()
                
                # If still full after cleanup, remove oldest
                if len(self._cache) >= self.max_size:
                    oldest_key = min(self._timestamps.keys(), key=self._timestamps.get)
                    del self._cache[oldest_key]
                    del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time() + (ttl or self.default_ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)
    
    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        return time.time() > self._timestamps.get(key, 0)
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time > timestamp
        ]
        
        for key in expired_keys:
            del self._cache[key]
            del self._timestamps[key]


class PersistentCache(BaseCache):
    """File-based persistent cache."""
    
    def __init__(self, cache_dir: Path, max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._lock = threading.RLock()
        
        # Index file to track cache entries
        self.index_file = self.cache_dir / "cache_index.json"
        self._index = self._load_index()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            if key not in self._index:
                return None
            
            file_path = self._get_file_path(key)
            if not file_path.exists():
                # Clean up stale index entry
                del self._index[key]
                self._save_index()
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # Corrupted file, remove it
                file_path.unlink(missing_ok=True)
                del self._index[key]
                self._save_index()
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in disk cache."""
        with self._lock:
            # Check cache size and clean up if needed
            if len(self._index) >= self.max_size:
                self._cleanup_old_entries()
            
            file_path = self._get_file_path(key)
            
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                self._index[key] = {
                    'timestamp': time.time(),
                    'ttl': ttl,
                    'size': file_path.stat().st_size
                }
                self._save_index()
                
            except Exception as e:
                # Failed to save, clean up
                file_path.unlink(missing_ok=True)
                if key in self._index:
                    del self._index[key]
                raise e
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key not in self._index:
                return False
            
            file_path = self._get_file_path(key)
            file_path.unlink(missing_ok=True)
            del self._index[key]
            self._save_index()
            return True
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            for key in list(self._index.keys()):
                file_path = self._get_file_path(key)
                file_path.unlink(missing_ok=True)
            
            self._index.clear()
            self._save_index()
    
    def size(self) -> int:
        """Get number of items in cache."""
        return len(self._index)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.get('size', 0) for entry in self._index.values())
        
        return {
            'entry_count': len(self._index),
            'total_size_bytes': total_size,
            'cache_dir': str(self.cache_dir),
            'oldest_entry': min(self._index.values(), key=lambda x: x['timestamp'], default={}).get('timestamp'),
            'newest_entry': max(self._index.values(), key=lambda x: x['timestamp'], default={}).get('timestamp')
        }
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from file."""
        if not self.index_file.exists():
            return {}
        
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_index(self) -> None:
        """Save cache index to file."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception:
            pass  # Continue if we can't save index
    
    def _cleanup_old_entries(self) -> None:
        """Remove old cache entries to make space."""
        if not self._index:
            return
        
        # Remove expired entries first
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._index.items():
            if entry.get('ttl') and (entry['timestamp'] + entry['ttl']) < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.delete(key)
        
        # If still too many entries, remove oldest
        if len(self._index) >= self.max_size:
            oldest_keys = sorted(self._index.keys(), key=lambda k: self._index[k]['timestamp'])
            for key in oldest_keys[:len(self._index) - self.max_size + 100]:  # Remove extra to avoid frequent cleanup
                self.delete(key)


class CacheManager:
    """Central cache management system."""
    
    def __init__(self):
        self._caches: Dict[str, BaseCache] = {}
        self._default_cache = LRUCache(max_size=1000)
        
    def get_cache(self, name: str = "default") -> BaseCache:
        """Get named cache instance."""
        if name == "default":
            return self._default_cache
        return self._caches.get(name, self._default_cache)
    
    def create_cache(
        self,
        name: str,
        cache_type: str = "lru",
        **kwargs
    ) -> BaseCache:
        """Create and register a new cache."""
        if cache_type == "lru":
            cache = LRUCache(**kwargs)
        elif cache_type == "ttl":
            cache = TTLCache(**kwargs)
        elif cache_type == "persistent":
            cache = PersistentCache(**kwargs)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
        
        self._caches[name] = cache
        return cache
    
    def remove_cache(self, name: str) -> bool:
        """Remove and clear a named cache."""
        if name in self._caches:
            self._caches[name].clear()
            del self._caches[name]
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()
        self._default_cache.clear()
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        stats = {}
        
        stats["default"] = {
            "type": "lru",
            "size": self._default_cache.size()
        }
        
        for name, cache in self._caches.items():
            cache_stats = {"size": cache.size()}
            
            if hasattr(cache, 'get_cache_stats'):
                cache_stats.update(cache.get_cache_stats())
            
            if isinstance(cache, LRUCache):
                cache_stats["type"] = "lru"
                cache_stats["max_size"] = cache.max_size
            elif isinstance(cache, TTLCache):
                cache_stats["type"] = "ttl"
                cache_stats["max_size"] = cache.max_size
                cache_stats["default_ttl"] = cache.default_ttl
            elif isinstance(cache, PersistentCache):
                cache_stats["type"] = "persistent"
                cache_stats["max_size"] = cache.max_size
            
            stats[name] = cache_stats
        
        return stats


# Global cache manager instance
cache_manager = CacheManager()


def cached(
    cache_name: str = "default",
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cache = cache_manager.get_cache(cache_name)
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


async def cached_async(
    cache_name: str = "default",
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching async function results."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cache = cache_manager.get_cache(cache_name)
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class SmartCache:
    """Intelligent cache that adapts based on usage patterns."""
    
    def __init__(self, name: str = "smart"):
        self.name = name
        self.access_counts = {}
        self.access_times = {}
        self.hit_rate = 0.0
        self.total_requests = 0
        self.cache_hits = 0
        
        # Start with LRU cache, can switch based on patterns
        self.current_cache = LRUCache(max_size=1000)
        self.cache_manager = cache_manager
    
    def get(self, key: str) -> Optional[Any]:
        """Get with intelligence tracking."""
        self.total_requests += 1
        
        result = self.current_cache.get(key)
        
        if result is not None:
            self.cache_hits += 1
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.access_times[key] = time.time()
        
        # Update hit rate
        self.hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0
        
        # Consider cache strategy adjustment
        if self.total_requests % 100 == 0:
            self._optimize_cache_strategy()
        
        return result
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set with pattern tracking."""
        self.current_cache.set(key, value, ttl)
        self.access_times[key] = time.time()
    
    def _optimize_cache_strategy(self) -> None:
        """Optimize cache strategy based on access patterns."""
        if self.hit_rate < 0.5 and isinstance(self.current_cache, LRUCache):
            # Low hit rate with LRU, try TTL cache
            new_cache = TTLCache(max_size=1000, default_ttl=3600)
            self._migrate_cache(new_cache)
            self.current_cache = new_cache
        
        # Additional optimization logic can be added here
    
    def _migrate_cache(self, new_cache: BaseCache) -> None:
        """Migrate data to new cache instance."""
        # This is a simplified migration - in practice, you'd want to
        # preserve the most valuable entries based on access patterns
        old_cache = self.current_cache
        if hasattr(old_cache, '_cache'):
            for key, value in list(old_cache._cache.items()):
                new_cache.set(key, value)