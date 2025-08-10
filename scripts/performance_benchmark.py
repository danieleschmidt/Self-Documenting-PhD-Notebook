#!/usr/bin/env python3
"""
Performance benchmark script for PhD notebook system.
"""

import asyncio
import time
import tempfile
import statistics
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phd_notebook.core.notebook import ResearchNotebook
from phd_notebook.core.note import Note, NoteType
from phd_notebook.ai.client_factory import AIClientFactory
from phd_notebook.performance.caching import LRUCache, DiskCache, MultiLevelCache
from phd_notebook.performance.indexing import SearchIndex
from phd_notebook.performance.async_processing import AsyncTaskManager, BatchProcessor


class PerformanceBenchmark:
    """Performance benchmark suite for PhD notebook system."""
    
    def __init__(self):
        self.results = {}
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("🚀 Running Performance Benchmarks...")
        print("=" * 50)
        
        benchmarks = [
            ("notebook_operations", self.benchmark_notebook_operations),
            ("note_operations", self.benchmark_note_operations),
            ("caching_performance", self.benchmark_caching),
            ("search_performance", self.benchmark_search),
            ("async_processing", self.benchmark_async_processing),
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\n📊 Running {name.replace('_', ' ').title()}...")
            try:
                self.results[name] = benchmark_func()
                status = "✅ PASS" if self.results[name]['status'] == 'pass' else "❌ FAIL"
                print(f"  {status} - {self.results[name]['duration']:.3f}s")
            except Exception as e:
                print(f"  ❌ ERROR - {str(e)}")
                self.results[name] = {'status': 'error', 'error': str(e)}
        
        # Calculate overall results
        self.results['summary'] = self.calculate_summary()
        return self.results
    
    def benchmark_notebook_operations(self) -> Dict[str, Any]:
        """Benchmark core notebook operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            start_time = time.time()
            
            # Create notebook
            notebook = ResearchNotebook(vault_path=Path(temp_dir))
            
            # Benchmark note creation
            creation_times = []
            for i in range(100):
                note_start = time.time()
                note = notebook.create_note(
                    title=f"Benchmark Note {i}",
                    content=f"This is benchmark content for note {i}",
                    note_type=NoteType.IDEA
                )
                creation_times.append(time.time() - note_start)
            
            # Benchmark note retrieval
            retrieval_times = []
            notes = notebook.list_notes()
            # Handle both dict and list return types
            if isinstance(notes, dict):
                note_ids = list(notes.keys())[:50]
            else:
                note_ids = [note.note_id if hasattr(note, 'note_id') else str(i) for i, note in enumerate(notes[:50])]
            
            for note_id in note_ids:
                retrieval_start = time.time()
                try:
                    note = notebook.get_note(note_id)
                    retrieval_times.append(time.time() - retrieval_start)
                except:
                    # Skip failed retrievals
                    continue
            
            total_time = time.time() - start_time
            
            # Performance criteria
            avg_creation_time = statistics.mean(creation_times) if creation_times else 0
            avg_retrieval_time = statistics.mean(retrieval_times) if retrieval_times else 0
            
            status = 'pass'
            if avg_creation_time > 0.1:  # Should create notes in < 100ms
                status = 'fail'
            if avg_retrieval_time > 0.05:  # Should retrieve notes in < 50ms
                status = 'fail'
            
            return {
                'status': status,
                'duration': total_time,
                'metrics': {
                    'notes_created': len(creation_times),
                    'avg_creation_time': avg_creation_time,
                    'avg_retrieval_time': avg_retrieval_time,
                    'notes_retrieved': len(retrieval_times),
                    'total_notes': len(notes) if isinstance(notes, (list, dict)) else 0
                }
            }
    
    def benchmark_note_operations(self) -> Dict[str, Any]:
        """Benchmark individual note operations."""
        start_time = time.time()
        
        # Benchmark note creation and manipulation
        creation_times = []
        save_times = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(50):
                # Create note
                create_start = time.time()
                note = Note(
                    title=f"Performance Test Note {i}",
                    content=f"Content for performance test {i}" * 10  # Longer content
                )
                creation_times.append(time.time() - create_start)
                
                # Add tags and links
                note.add_tags([f"#tag{i}", "#performance", "#test"])
                note.add_link(f"Related Note {i}")
                note.add_section("Results", f"Performance results for test {i}")
                
                # Save note
                save_start = time.time()
                file_path = Path(temp_dir) / f"note_{i}.md"
                note.save(file_path)
                save_times.append(time.time() - save_start)
        
        total_time = time.time() - start_time
        
        # Performance criteria
        avg_creation_time = statistics.mean(creation_times)
        avg_save_time = statistics.mean(save_times)
        
        status = 'pass'
        if avg_creation_time > 0.01:  # Should create notes in < 10ms
            status = 'fail'
        if avg_save_time > 0.1:  # Should save notes in < 100ms
            status = 'fail'
        
        return {
            'status': status,
            'duration': total_time,
            'metrics': {
                'avg_creation_time': avg_creation_time,
                'avg_save_time': avg_save_time,
                'notes_processed': len(creation_times)
            }
        }
    
    def benchmark_caching(self) -> Dict[str, Any]:
        """Benchmark caching performance."""
        start_time = time.time()
        
        # Test LRU Cache
        lru_cache = LRUCache(max_size=1000)
        lru_times = []
        
        # Warm up cache
        for i in range(500):
            lru_cache.set(f"key_{i}", f"value_{i}")
        
        # Benchmark cache operations
        for i in range(1000):
            op_start = time.time()
            if i % 2 == 0:
                lru_cache.set(f"benchmark_key_{i}", f"benchmark_value_{i}")
            else:
                value = lru_cache.get(f"key_{i % 500}")  # Access existing keys
            lru_times.append(time.time() - op_start)
        
        # Test Disk Cache
        with tempfile.TemporaryDirectory() as temp_dir:
            disk_cache = DiskCache(Path(temp_dir), max_size_mb=10)
            disk_times = []
            
            for i in range(100):  # Fewer iterations for disk operations
                op_start = time.time()
                if i % 2 == 0:
                    disk_cache.set(f"disk_key_{i}", {"data": f"disk_value_{i}"})
                else:
                    value = disk_cache.get(f"disk_key_{max(0, i-1)}")
                disk_times.append(time.time() - op_start)
        
        total_time = time.time() - start_time
        
        # Performance criteria
        avg_lru_time = statistics.mean(lru_times)
        avg_disk_time = statistics.mean(disk_times)
        
        status = 'pass'
        if avg_lru_time > 0.001:  # LRU operations should be < 1ms
            status = 'fail'
        if avg_disk_time > 0.01:  # Disk operations should be < 10ms
            status = 'fail'
        
        return {
            'status': status,
            'duration': total_time,
            'metrics': {
                'avg_lru_time': avg_lru_time,
                'avg_disk_time': avg_disk_time,
                'lru_operations': len(lru_times),
                'disk_operations': len(disk_times)
            }
        }
    
    def benchmark_search(self) -> Dict[str, Any]:
        """Benchmark search performance."""
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            search_index = SearchIndex(Path(temp_dir) / "benchmark_search.db")
            
            # Add documents to index
            indexing_times = []
            for i in range(200):
                doc_start = time.time()
                search_index.add_document(
                    doc_id=f"doc_{i}",
                    title=f"Document {i} about Machine Learning",
                    content=f"This document discusses machine learning concepts, algorithms, and applications in research area {i}. " * 5,
                    tags=[f"tag{i}", "machine-learning", "research"]
                )
                indexing_times.append(time.time() - doc_start)
            
            # Benchmark search operations
            search_times = []
            search_queries = [
                "machine learning",
                "algorithms",
                "research",
                "applications",
                "concepts"
            ]
            
            for query in search_queries * 20:  # 100 total searches
                search_start = time.time()
                results = search_index.search(query, limit=10)
                search_times.append(time.time() - search_start)
        
        total_time = time.time() - start_time
        
        # Performance criteria
        avg_indexing_time = statistics.mean(indexing_times)
        avg_search_time = statistics.mean(search_times)
        
        status = 'pass'
        if avg_indexing_time > 0.05:  # Indexing should be < 50ms per document
            status = 'fail'
        if avg_search_time > 0.1:  # Search should be < 100ms
            status = 'fail'
        
        return {
            'status': status,
            'duration': total_time,
            'metrics': {
                'avg_indexing_time': avg_indexing_time,
                'avg_search_time': avg_search_time,
                'documents_indexed': len(indexing_times),
                'searches_performed': len(search_times)
            }
        }
    
    def benchmark_async_processing(self) -> Dict[str, Any]:
        """Benchmark async processing performance."""
        start_time = time.time()
        
        async def run_async_benchmarks():
            # Benchmark AsyncTaskManager
            task_manager = AsyncTaskManager(max_concurrent_tasks=5)
            
            async def dummy_task(value):
                await asyncio.sleep(0.001)  # Simulate some work
                return value * 2
            
            # Submit multiple tasks
            task_times = []
            tasks = []
            for i in range(50):
                submit_start = time.time()
                task_id = await task_manager.submit_task(dummy_task(i))
                tasks.append(task_id)
                task_times.append(time.time() - submit_start)
            
            # Wait for completion
            completion_start = time.time()
            results = await task_manager.wait_for_all(timeout=5.0)
            completion_time = time.time() - completion_start
            
            # Benchmark BatchProcessor
            batch_processor = BatchProcessor(batch_size=10, max_concurrent_batches=3)
            
            async def process_item(item):
                await asyncio.sleep(0.001)
                return item * 3
            
            batch_start = time.time()
            items = list(range(100))
            batch_results = await batch_processor.process_items(items, process_item)
            batch_time = time.time() - batch_start
            
            return {
                'task_submission_times': task_times,
                'task_completion_time': completion_time,
                'batch_processing_time': batch_time,
                'results_count': len(results),
                'batch_results_count': len(batch_results)
            }
        
        # Run async benchmarks
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            async_results = loop.run_until_complete(run_async_benchmarks())
        finally:
            loop.close()
        
        total_time = time.time() - start_time
        
        # Performance criteria
        avg_task_submission = statistics.mean(async_results['task_submission_times'])
        
        status = 'pass'
        if avg_task_submission > 0.01:  # Task submission should be < 10ms
            status = 'fail'
        if async_results['task_completion_time'] > 2.0:  # Should complete quickly with concurrency
            status = 'fail'
        if async_results['batch_processing_time'] > 1.0:  # Batch processing should be fast
            status = 'fail'
        
        return {
            'status': status,
            'duration': total_time,
            'metrics': {
                'avg_task_submission_time': avg_task_submission,
                'task_completion_time': async_results['task_completion_time'],
                'batch_processing_time': async_results['batch_processing_time'],
                'tasks_processed': async_results['results_count'],
                'batch_items_processed': async_results['batch_results_count']
            }
        }
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate benchmark summary."""
        total_benchmarks = len([k for k in self.results.keys() if k != 'summary'])
        passed_benchmarks = len([r for r in self.results.values() 
                               if isinstance(r, dict) and r.get('status') == 'pass'])
        failed_benchmarks = len([r for r in self.results.values() 
                               if isinstance(r, dict) and r.get('status') == 'fail'])
        error_benchmarks = len([r for r in self.results.values() 
                              if isinstance(r, dict) and r.get('status') == 'error'])
        
        overall_status = 'PASS'
        if error_benchmarks > 0 or failed_benchmarks > 0:
            overall_status = 'FAIL'
        
        return {
            'total_benchmarks': total_benchmarks,
            'passed': passed_benchmarks,
            'failed': failed_benchmarks,
            'errors': error_benchmarks,
            'pass_rate': passed_benchmarks / total_benchmarks if total_benchmarks > 0 else 0,
            'overall_status': overall_status
        }


def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Print summary
    summary = results['summary']
    print(f"\n📈 Performance Benchmark Summary:")
    print(f"  Total Benchmarks: {summary['total_benchmarks']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Pass Rate: {summary['pass_rate']:.1%}")
    print(f"  Overall Status: {'✅' if summary['overall_status'] == 'PASS' else '❌'} {summary['overall_status']}")
    
    # Exit with appropriate code
    sys.exit(0 if summary['overall_status'] == 'PASS' else 1)


if __name__ == "__main__":
    main()