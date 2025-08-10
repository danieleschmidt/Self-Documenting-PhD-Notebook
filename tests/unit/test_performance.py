"""
Tests for performance components.
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path

from phd_notebook.performance.caching import LRUCache, AsyncCache, DiskCache, MultiLevelCache, cache_result
from phd_notebook.performance.async_processing import AsyncTaskManager, BatchProcessor, AsyncQueue
from phd_notebook.performance.indexing import SearchIndex


class TestLRUCache:
    """Test LRU cache functionality."""
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size=3)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test miss
        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", "default") == "default"
        
        # Test size
        assert cache.size() == 1
    
    def test_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = LRUCache(max_size=10, ttl_seconds=0.1)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("key1") is None
    
    def test_statistics(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10)
        
        # Test misses
        cache.get("nonexistent")
        stats = cache.stats.get_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 0
        
        # Test hits
        cache.set("key1", "value1")
        cache.get("key1")
        stats = cache.stats.get_stats()
        assert stats['hits'] == 1


class TestAsyncCache:
    """Test async cache functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_async_operations(self):
        """Test basic async cache operations."""
        cache = AsyncCache(max_size=10, ttl_seconds=60)
        
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"
    
    @pytest.mark.asyncio
    async def test_refresh_function(self):
        """Test cache with refresh function."""
        cache = AsyncCache(max_size=10, ttl_seconds=60)
        
        async def refresh_func(key):
            return f"refreshed_{key}"
        
        # First call should fetch
        value = await cache.get("key1", refresh_func)
        assert value == "refreshed_key1"
        
        # Second call should return cached value
        value = await cache.get("key1", refresh_func)
        assert value == "refreshed_key1"


class TestDiskCache:
    """Test disk cache functionality."""
    
    def test_disk_cache_basic(self):
        """Test basic disk cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(Path(temp_dir), max_size_mb=1)
            
            cache.set("key1", {"data": "value1"})
            value = cache.get("key1")
            assert value == {"data": "value1"}
            
            # Test miss
            assert cache.get("nonexistent") is None
    
    def test_disk_cache_ttl(self):
        """Test disk cache TTL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(Path(temp_dir), max_size_mb=1)
            
            cache.set("key1", "value1", ttl_seconds=0.1)
            assert cache.get("key1") == "value1"
            
            time.sleep(0.15)
            assert cache.get("key1") is None


class TestMultiLevelCache:
    """Test multi-level cache."""
    
    def test_multi_level_basic(self):
        """Test basic multi-level cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = MultiLevelCache(
                memory_size=10,
                disk_size_mb=1,
                cache_dir=Path(temp_dir)
            )
            
            cache.set("key1", "value1")
            
            # Should be in memory
            assert cache.get("key1") == "value1"
            
            # Clear memory cache, should still get from disk
            cache.memory_cache.clear()
            assert cache.get("key1") == "value1"


class TestCacheDecorator:
    """Test cache decorator."""
    
    def test_cache_decorator(self):
        """Test function caching decorator."""
        call_count = 0
        
        @cache_result(ttl_seconds=60, max_size=10)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args (should be cached)
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not have called function again
        
        # Different args
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2


class TestAsyncTaskManager:
    """Test async task manager."""
    
    @pytest.mark.asyncio
    async def test_task_submission(self):
        """Test task submission and execution."""
        manager = AsyncTaskManager(max_concurrent_tasks=2)
        
        async def test_task(value):
            await asyncio.sleep(0.1)
            return value * 2
        
        # Submit task
        task_id = await manager.submit_task(test_task(5))
        
        # Wait for completion
        result = await manager.wait_for_task(task_id)
        assert result.result == 10
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """Test concurrent task limit."""
        manager = AsyncTaskManager(max_concurrent_tasks=2)
        
        async def slow_task():
            await asyncio.sleep(0.2)
            return "done"
        
        # Submit 3 tasks (only 2 should run concurrently)
        task_ids = []
        for i in range(3):
            task_id = await manager.submit_task(slow_task())
            task_ids.append(task_id)
        
        # Wait for all to complete
        results = await manager.wait_for_all(timeout=1.0)
        
        assert len(results) == 3
        assert all(r.result == "done" for r in results.values())


class TestBatchProcessor:
    """Test batch processor."""
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of items."""
        processor = BatchProcessor(batch_size=2, max_concurrent_batches=2)
        
        async def double_item(item):
            await asyncio.sleep(0.01)  # Simulate work
            return item * 2
        
        items = [1, 2, 3, 4, 5]
        results = await processor.process_items(items, double_item)
        
        assert len(results) == 5
        assert results == [2, 4, 6, 8, 10]


class TestAsyncQueue:
    """Test async queue."""
    
    @pytest.mark.asyncio
    async def test_queue_operations(self):
        """Test basic queue operations."""
        queue = AsyncQueue(maxsize=10)
        
        # Put items
        await queue.put("item1", priority=1)
        await queue.put("item2", priority=0)  # Higher priority
        
        # Get items (higher priority first)
        item1 = await queue.get()
        assert item1 == "item2"  # Higher priority item
        
        item2 = await queue.get()
        assert item2 == "item1"
        
        # Mark tasks done
        queue.task_done()
        queue.task_done()
        
        # Queue should be empty
        assert queue.empty()


class TestSearchIndex:
    """Test search index."""
    
    def test_document_indexing(self):
        """Test document indexing and search."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index = SearchIndex(Path(temp_dir) / "test_index.db")
            
            # Add documents
            index.add_document(
                doc_id="doc1",
                title="Machine Learning Research",
                content="This paper discusses neural networks and deep learning algorithms.",
                tags=["machine-learning", "neural-networks"]
            )
            
            index.add_document(
                doc_id="doc2", 
                title="Data Science Methods",
                content="Statistical analysis and data visualization techniques.",
                tags=["data-science", "statistics"]
            )
            
            # Search
            results = index.search("machine learning")
            assert len(results) >= 1
            assert results[0]['id'] == "doc1"
            
            # Search for partial matches
            results = index.search("neural")
            assert len(results) >= 1
            assert results[0]['id'] == "doc1"
    
    def test_term_suggestions(self):
        """Test term suggestions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index = SearchIndex(Path(temp_dir) / "test_index.db")
            
            index.add_document(
                doc_id="doc1",
                title="Machine Learning",
                content="Machine learning and artificial intelligence",
                tags=["machine-learning"]
            )
            
            suggestions = index.suggest_terms("mach", limit=5)
            assert "machine" in suggestions
    
    def test_related_documents(self):
        """Test finding related documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index = SearchIndex(Path(temp_dir) / "test_index.db")
            
            # Add related documents
            index.add_document("doc1", "Neural Networks", "Deep learning neural networks")
            index.add_document("doc2", "Deep Learning", "Neural networks for deep learning")
            index.add_document("doc3", "Statistics", "Statistical analysis methods")
            
            # Find documents related to doc1
            related = index.get_related_documents("doc1", limit=2)
            
            # doc2 should be related (shares terms), doc3 should not
            related_ids = [doc['id'] for doc in related]
            assert "doc2" in related_ids
            assert "doc3" not in related_ids


if __name__ == "__main__":
    pytest.main([__file__])