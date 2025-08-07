"""
Async processing system for batch operations and concurrent tasks.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Dict, Optional, Union, Coroutine
from dataclasses import dataclass
from datetime import datetime
import threading
from queue import Queue, Empty
from threading import Event

from ..utils.logging import get_logger, log_performance

logger = get_logger(__name__)


@dataclass
class TaskResult:
    """Result of an async task."""
    task_id: str
    result: Any
    error: Optional[Exception] = None
    duration: float = 0.0
    started_at: datetime = None
    completed_at: datetime = None
    
    @property
    def success(self) -> bool:
        """Whether task completed successfully."""
        return self.error is None


class AsyncTaskProcessor:
    """
    High-performance async task processor with batching and concurrency control.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_concurrent_tasks: int = 10,
        batch_size: int = 50,
        batch_timeout: float = 5.0,
        enable_batching: bool = True
    ):
        self.max_workers = max_workers
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.enable_batching = enable_batching
        
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._task_queue = Queue()
        self._batch_processor = None
        self._shutdown_event = Event()
        
        # Statistics
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._total_processing_time = 0.0
        
        # Start batch processor if enabled
        if self.enable_batching:
            self._start_batch_processor()
    
    async def process_single(
        self,
        func: Callable,
        *args,
        task_id: str = None,
        **kwargs
    ) -> TaskResult:
        """Process a single task asynchronously."""
        task_id = task_id or f"task_{int(time.time() * 1000000)}"
        start_time = time.time()
        started_at = datetime.now()
        
        async with self._semaphore:
            try:
                logger.debug(f"Starting task {task_id}")
                
                # Run in thread pool if not coroutine
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self._executor, func, *args, **kwargs
                    )
                
                duration = time.time() - start_time
                completed_at = datetime.now()
                
                task_result = TaskResult(
                    task_id=task_id,
                    result=result,
                    duration=duration,
                    started_at=started_at,
                    completed_at=completed_at
                )
                
                self._tasks_completed += 1
                self._total_processing_time += duration
                
                logger.debug(f"Completed task {task_id} in {duration:.3f}s")
                return task_result
                
            except Exception as e:
                duration = time.time() - start_time
                completed_at = datetime.now()
                
                task_result = TaskResult(
                    task_id=task_id,
                    result=None,
                    error=e,
                    duration=duration,
                    started_at=started_at,
                    completed_at=completed_at
                )
                
                self._tasks_failed += 1
                logger.error(f"Task {task_id} failed: {e}")
                return task_result
    
    async def process_batch(
        self,
        tasks: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TaskResult]:
        """Process multiple tasks concurrently."""
        logger.info(f"Processing batch of {len(tasks)} tasks")
        
        # Create coroutines for all tasks
        coroutines = []
        for i, task in enumerate(tasks):
            func = task['func']
            args = task.get('args', ())
            kwargs = task.get('kwargs', {})
            task_id = task.get('task_id', f"batch_task_{i}")
            
            coro = self.process_single(func, *args, task_id=task_id, **kwargs)
            coroutines.append(coro)
        
        # Process all tasks concurrently
        results = []
        completed = 0
        
        for coro in asyncio.as_completed(coroutines):
            result = await coro
            results.append(result)
            completed += 1
            
            if progress_callback:
                progress_callback(completed, len(tasks))
        
        # Sort results by original order
        task_id_to_index = {task.get('task_id', f"batch_task_{i}"): i 
                           for i, task in enumerate(tasks)}
        results.sort(key=lambda r: task_id_to_index.get(r.task_id, 999999))
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        logger.info(f"Batch completed: {successful} successful, {failed} failed")
        return results
    
    def submit_for_batching(
        self,
        func: Callable,
        *args,
        task_id: str = None,
        priority: int = 0,
        **kwargs
    ) -> str:
        """Submit task for batch processing."""
        if not self.enable_batching:
            raise RuntimeError("Batching is not enabled")
        
        task_id = task_id or f"queued_task_{int(time.time() * 1000000)}"
        
        task = {
            'task_id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'submitted_at': time.time()
        }
        
        self._task_queue.put(task)
        logger.debug(f"Queued task {task_id} for batch processing")
        return task_id
    
    def _start_batch_processor(self):
        """Start background batch processor."""
        self._batch_processor = threading.Thread(
            target=self._batch_processing_loop,
            daemon=True
        )
        self._batch_processor.start()
        logger.info("Started batch processor")
    
    def _batch_processing_loop(self):
        """Background batch processing loop."""
        while not self._shutdown_event.is_set():
            batch = []
            batch_start = time.time()
            
            # Collect tasks for batch
            while (
                len(batch) < self.batch_size and
                time.time() - batch_start < self.batch_timeout
            ):
                try:
                    task = self._task_queue.get(timeout=0.1)
                    batch.append(task)
                except Empty:
                    if batch:  # Process partial batch if timeout reached
                        break
                    continue
            
            if batch:
                # Sort by priority (higher priority first)
                batch.sort(key=lambda t: t['priority'], reverse=True)
                
                # Process batch
                try:
                    asyncio.run(self._process_batch_in_thread(batch))
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
    
    async def _process_batch_in_thread(self, batch: List[Dict[str, Any]]):
        """Process batch in async context."""
        tasks = [
            {
                'func': task['func'],
                'args': task['args'],
                'kwargs': task['kwargs'],
                'task_id': task['task_id']
            }
            for task in batch
        ]
        
        await self.process_batch(tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_tasks = self._tasks_completed + self._tasks_failed
        avg_time = (self._total_processing_time / self._tasks_completed 
                   if self._tasks_completed > 0 else 0)
        success_rate = (self._tasks_completed / total_tasks * 100 
                       if total_tasks > 0 else 0)
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': self._tasks_completed,
            'failed_tasks': self._tasks_failed,
            'success_rate_percent': success_rate,
            'average_task_time': avg_time,
            'total_processing_time': self._total_processing_time,
            'queue_size': self._task_queue.qsize() if self.enable_batching else 0,
            'max_workers': self.max_workers,
            'max_concurrent_tasks': self.max_concurrent_tasks
        }
    
    def shutdown(self):
        """Shutdown the processor."""
        logger.info("Shutting down async processor")
        self._shutdown_event.set()
        
        if self._batch_processor:
            self._batch_processor.join(timeout=5.0)
        
        self._executor.shutdown(wait=True)


class BulkOperationProcessor:
    """
    Specialized processor for bulk operations on research data.
    """
    
    def __init__(self, async_processor: AsyncTaskProcessor = None):
        self.async_processor = async_processor or AsyncTaskProcessor()
    
    async def bulk_create_notes(
        self,
        notebook,
        note_specs: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TaskResult]:
        """Create multiple notes in bulk."""
        logger.info(f"Bulk creating {len(note_specs)} notes")
        
        def create_note_task(spec):
            return notebook.create_note(**spec)
        
        tasks = [
            {
                'func': create_note_task,
                'args': (spec,),
                'task_id': f"create_note_{spec.get('title', i)}"
            }
            for i, spec in enumerate(note_specs)
        ]
        
        return await self.async_processor.process_batch(tasks, progress_callback)
    
    async def bulk_update_notes(
        self,
        notes_and_updates: List[tuple],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TaskResult]:
        """Update multiple notes in bulk."""
        logger.info(f"Bulk updating {len(notes_and_updates)} notes")
        
        def update_note_task(note, updates):
            for key, value in updates.items():
                setattr(note, key, value)
            if hasattr(note, 'save') and note.file_path:
                note.save()
            return note
        
        tasks = [
            {
                'func': update_note_task,
                'args': (note, updates),
                'task_id': f"update_note_{note.title}"
            }
            for note, updates in notes_and_updates
        ]
        
        return await self.async_processor.process_batch(tasks, progress_callback)
    
    async def bulk_analyze_content(
        self,
        content_items: List[str],
        analysis_func: Callable[[str], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TaskResult]:
        """Analyze multiple content items in bulk."""
        logger.info(f"Bulk analyzing {len(content_items)} content items")
        
        tasks = [
            {
                'func': analysis_func,
                'args': (content,),
                'task_id': f"analyze_content_{i}"
            }
            for i, content in enumerate(content_items)
        ]
        
        return await self.async_processor.process_batch(tasks, progress_callback)
    
    async def bulk_import_data(
        self,
        connector,
        data_items: List[Dict[str, Any]],
        notebook,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[TaskResult]:
        """Import multiple data items in bulk."""
        logger.info(f"Bulk importing {len(data_items)} data items")
        
        def import_item_task(data_item):
            return connector._process_data_item(data_item, notebook)
        
        tasks = [
            {
                'func': import_item_task,
                'args': (item,),
                'task_id': f"import_item_{i}"
            }
            for i, item in enumerate(data_items)
        ]
        
        return await self.async_processor.process_batch(tasks, progress_callback)


class SearchIndexBuilder:
    """
    Async search index builder for large vaults.
    """
    
    def __init__(self, async_processor: AsyncTaskProcessor = None):
        self.async_processor = async_processor or AsyncTaskProcessor()
    
    async def build_full_text_index(
        self,
        notes: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Build full-text search index asynchronously."""
        logger.info(f"Building search index for {len(notes)} notes")
        
        def extract_terms_task(note):
            # Simple term extraction (would use proper NLP in production)
            import re
            content = f"{note.title} {note.content}".lower()
            terms = re.findall(r'\b\w+\b', content)
            return {
                'note_id': note.title,
                'terms': list(set(terms)),
                'word_count': len(terms)
            }
        
        tasks = [
            {
                'func': extract_terms_task,
                'args': (note,),
                'task_id': f"extract_terms_{note.title}"
            }
            for note in notes
        ]
        
        results = await self.async_processor.process_batch(tasks, progress_callback)
        
        # Build inverted index
        inverted_index = {}
        note_info = {}
        
        for result in results:
            if result.success and result.result:
                note_data = result.result
                note_id = note_data['note_id']
                note_info[note_id] = {
                    'word_count': note_data['word_count']
                }
                
                for term in note_data['terms']:
                    if term not in inverted_index:
                        inverted_index[term] = []
                    inverted_index[term].append(note_id)
        
        logger.info(f"Built search index with {len(inverted_index)} terms")
        
        return {
            'inverted_index': inverted_index,
            'note_info': note_info,
            'total_notes': len([r for r in results if r.success]),
            'total_terms': len(inverted_index)
        }


# Global instances
default_async_processor = AsyncTaskProcessor()
bulk_processor = BulkOperationProcessor(default_async_processor)
search_builder = SearchIndexBuilder(default_async_processor)


# Utility functions
async def run_concurrent(
    tasks: List[Callable],
    max_concurrent: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """Run multiple tasks concurrently with limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_task(task):
        async with semaphore:
            if asyncio.iscoroutinefunction(task):
                return await task()
            else:
                return await asyncio.get_event_loop().run_in_executor(None, task)
    
    # Create limited tasks
    limited_tasks = [limited_task(task) for task in tasks]
    
    # Process with progress tracking
    results = []
    completed = 0
    
    for task in asyncio.as_completed(limited_tasks):
        result = await task
        results.append(result)
        completed += 1
        
        if progress_callback:
            progress_callback(completed, len(tasks))
    
    return results


def run_async(coro: Coroutine) -> Any:
    """Helper to run async code in sync context."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop running
        return asyncio.run(coro)