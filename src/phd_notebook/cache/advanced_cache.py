"""
Advanced caching system with TTL, LRU, and intelligent invalidation.
"""

import asyncio
import time
import weakref
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set, Callable, Union, List
from dataclasses import dataclass
from threading import RLock
import pickle
import hashlib
import json

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    size: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at


class AdvancedCache:
    """
    Advanced caching system with multiple eviction policies and features.
    
    Features:
    - LRU (Least Recently Used) eviction
    - TTL (Time To Live) expiration
    - Size-based eviction
    - Access frequency tracking
    - Intelligent cache warming
    - Statistics and monitoring
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        max_memory_mb: float = 100.0,
        cleanup_interval: float = 300.0,  # 5 minutes
        enable_statistics: bool = True
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval
        self.enable_statistics = enable_statistics
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._current_memory = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        
        # Invalidation tracking
        self._tag_to_keys: Dict[str, Set[str]] = {}
        self._key_to_tags: Dict[str, Set[str]] = {}
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1 if self.enable_statistics else 0
                logger.debug(f"Cache miss for key: {key}")
                return default
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self._expirations += 1 if self.enable_statistics else 0
                self._misses += 1 if self.enable_statistics else 0
                logger.debug(f"Cache expired for key: {key}")
                return default
            
            # Update access info
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._hits += 1 if self.enable_statistics else 0
            logger.debug(f"Cache hit for key: {key}")
            return entry.value
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        """Set value in cache."""
        with self._lock:
            # Calculate size
            entry_size = self._calculate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Check memory limit
            if self._current_memory + entry_size > self.max_memory_bytes:
                self._evict_by_memory(entry_size)
            
            # Check size limit
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Create entry
            now = time.time()
            entry = CacheEntry(
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                ttl=ttl or self.default_ttl,
                size=entry_size
            )
            
            # Add to cache
            self._cache[key] = entry
            self._current_memory += entry_size
            
            # Handle tags
            if tags:
                self._key_to_tags[key] = tags
                for tag in tags:
                    if tag not in self._tag_to_keys:
                        self._tag_to_keys[tag] = set()
                    self._tag_to_keys[tag].add(key)
            
            logger.debug(f"Cache set for key: {key}, size: {entry_size} bytes")
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                logger.debug(f"Cache delete for key: {key}")
                return True
            return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag."""
        with self._lock:
            keys_to_remove = self._tag_to_keys.get(tag, set()).copy()
            
            for key in keys_to_remove:
                self._remove_entry(key)
            
            if tag in self._tag_to_keys:
                del self._tag_to_keys[tag]
            
            logger.debug(f"Cache invalidated {len(keys_to_remove)} entries by tag: {tag}")
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._tag_to_keys.clear()
            self._key_to_tags.clear()
            self._current_memory = 0
            logger.debug("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate_percent': hit_rate,
                'evictions': self._evictions,
                'expirations': self._expirations,
                'tags': len(self._tag_to_keys)
            }
    
    def get_hot_keys(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed keys."""
        with self._lock:
            entries = [
                {
                    'key': key,
                    'access_count': entry.access_count,
                    'age': entry.age,
                    'size': entry.size
                }
                for key, entry in self._cache.items()
            ]
            
            return sorted(entries, key=lambda x: x['access_count'], reverse=True)[:limit]
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update tracking."""
        if key in self._cache:
            entry = self._cache[key]
            del self._cache[key]
            self._current_memory -= entry.size
            
            # Remove tag associations
            if key in self._key_to_tags:
                tags = self._key_to_tags[key]
                for tag in tags:
                    if tag in self._tag_to_keys:
                        self._tag_to_keys[tag].discard(key)
                        if not self._tag_to_keys[tag]:
                            del self._tag_to_keys[tag]
                del self._key_to_tags[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            key = next(iter(self._cache))
            self._remove_entry(key)
            self._evictions += 1 if self.enable_statistics else 0
            logger.debug(f"Cache LRU eviction for key: {key}")
    
    def _evict_by_memory(self, needed_bytes: int) -> None:
        """Evict entries to free memory."""
        while self._current_memory + needed_bytes > self.max_memory_bytes and self._cache:
            key = next(iter(self._cache))
            self._remove_entry(key)
            self._evictions += 1 if self.enable_statistics else 0
            logger.debug(f"Cache memory eviction for key: {key}")
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value[:10])  # Sample first 10
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in list(value.items())[:10])  # Sample first 10
            else:
                return 1024  # Default estimate
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_expired(self) -> None:
        """Clean up expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                self._expirations += 1 if self.enable_statistics else 0
            
            if expired_keys:
                logger.debug(f"Cache cleanup removed {len(expired_keys)} expired entries")


class SmartCache(AdvancedCache):
    """
    Smart cache with predictive preloading and adaptive TTL.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._access_patterns: Dict[str, List[float]] = {}
        self._preload_callbacks: Dict[str, Callable] = {}
    
    def register_preload_callback(self, pattern: str, callback: Callable[[str], Any]) -> None:
        """Register callback for predictive preloading."""
        self._preload_callbacks[pattern] = callback
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get with access pattern tracking."""
        # Track access pattern
        now = time.time()
        if key not in self._access_patterns:
            self._access_patterns[key] = []
        self._access_patterns[key].append(now)
        
        # Keep only recent accesses (last hour)
        hour_ago = now - 3600
        self._access_patterns[key] = [
            t for t in self._access_patterns[key] if t > hour_ago
        ]
        
        result = super().get(key, default)
        
        # Trigger predictive loading if pattern detected
        if result is not None:
            self._maybe_preload_related(key)
        
        return result
    
    def _maybe_preload_related(self, accessed_key: str) -> None:
        """Maybe preload related keys based on patterns."""
        # Simple pattern: if key ends with a number, preload next numbers
        import re
        
        for pattern, callback in self._preload_callbacks.items():
            if re.search(pattern, accessed_key):
                try:
                    # Run callback asynchronously
                    asyncio.create_task(self._async_preload(callback, accessed_key))
                except Exception as e:
                    logger.debug(f"Preload failed for {accessed_key}: {e}")
    
    async def _async_preload(self, callback: Callable, key: str) -> None:
        """Async preload operation."""
        try:
            result = await asyncio.to_thread(callback, key)
            if result is not None:
                # Use shorter TTL for preloaded items
                preload_ttl = self.default_ttl / 2 if self.default_ttl else 300
                self.set(f"preload:{key}", result, ttl=preload_ttl)
        except Exception as e:
            logger.debug(f"Async preload failed: {e}")


class CacheManager:
    """
    Global cache manager for different cache types.
    """
    
    def __init__(self):
        self._caches: Dict[str, AdvancedCache] = {}
        self._default_configs = {
            'notes': {'max_size': 1000, 'default_ttl': 3600},  # 1 hour
            'search': {'max_size': 500, 'default_ttl': 1800},   # 30 minutes
            'knowledge_graph': {'max_size': 100, 'default_ttl': 7200},  # 2 hours
            'agents': {'max_size': 200, 'default_ttl': 3600},   # 1 hour
            'connectors': {'max_size': 300, 'default_ttl': 900}  # 15 minutes
        }
    
    def get_cache(self, cache_type: str, smart: bool = False) -> AdvancedCache:
        """Get or create cache of specific type."""
        if cache_type not in self._caches:
            config = self._default_configs.get(cache_type, {})
            
            if smart:
                self._caches[cache_type] = SmartCache(**config)
            else:
                self._caches[cache_type] = AdvancedCache(**config)
            
            logger.info(f"Created {cache_type} cache")
        
        return self._caches[cache_type]
    
    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self._caches.values():
            cache.clear()
        logger.info("Cleared all caches")
    
    def get_global_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches."""
        return {
            cache_type: cache.get_stats()
            for cache_type, cache in self._caches.items()
        }


# Global cache manager instance
cache_manager = CacheManager()


# Decorator for caching function results
def cached(
    cache_type: str = 'default',
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    tags: Optional[Set[str]] = None
):
    """
    Decorator for caching function results.
    
    Args:
        cache_type: Type of cache to use
        ttl: Time to live for cached result
        key_func: Function to generate cache key from args
        tags: Tags for cache invalidation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
                
                # Hash if too long
                if len(cache_key) > 200:
                    cache_key = hashlib.md5(cache_key.encode()).hexdigest()
            
            # Get cache
            cache = cache_manager.get_cache(cache_type)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl, tags=tags)
            
            return result
        
        return wrapper
    return decorator