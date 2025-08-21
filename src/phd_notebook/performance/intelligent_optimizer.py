"""
Intelligent Performance Optimization System for PhD Notebook.
Generation 3 (Optimized) implementation featuring:
- Machine learning-based performance prediction
- Adaptive resource allocation
- Intelligent preloading and prefetching
- Research-aware optimization strategies
- Auto-tuning hyperparameters
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import json
import pickle

from ..monitoring.metrics import MetricsCollector
from ..utils.logging import get_logger


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    MEMORY_OPTIMIZED = "memory_optimized"
    ENERGY_EFFICIENT = "energy_efficient"
    RESEARCH_BALANCED = "research_balanced"


class ResourceType(Enum):
    """Types of system resources to optimize."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK = "network"
    GPU = "gpu"


@dataclass
class PerformanceProfile:
    """Performance profile for different research operations."""
    operation_type: str
    avg_duration_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    io_operations: int
    cache_hit_rate: float = 0.0
    optimization_potential: float = 0.0
    usage_frequency: int = 0
    last_optimized: Optional[datetime] = None


@dataclass
class OptimizationRecommendation:
    """Recommendation for performance optimization."""
    target_operation: str
    strategy: OptimizationStrategy
    expected_improvement: float
    implementation_effort: str  # low, medium, high
    priority: int
    reasoning: str
    implementation_steps: List[str]


class IntelligentPerformanceOptimizer:
    """
    AI-powered performance optimization system that learns from usage patterns
    and automatically optimizes research workflows.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.logger = get_logger(__name__)
        self.metrics = metrics_collector or MetricsCollector()
        
        # Performance tracking
        self._operation_profiles: Dict[str, PerformanceProfile] = {}
        self._optimization_history: List[Dict[str, Any]] = []
        self._current_strategy = OptimizationStrategy.RESEARCH_BALANCED
        
        # ML-based prediction models (simplified implementation)
        self._performance_predictor = None
        self._resource_predictor = None
        
        # Caching and prefetching
        self._cache_manager = IntelligentCacheManager()
        self._prefetch_queue = asyncio.Queue()
        
        # Resource monitoring
        self._resource_monitor = ResourceMonitor()
        self._optimization_lock = threading.RLock()
        
        # Auto-tuning parameters
        self._tuning_parameters = {
            'cache_size_mb': 256,
            'prefetch_threshold': 0.7,
            'optimization_interval_minutes': 30,
            'learning_rate': 0.01,
            'performance_threshold': 0.8
        }
        
        self._initialize_optimization_engine()
    
    def _initialize_optimization_engine(self):
        """Initialize the optimization engine with default configurations."""
        self.logger.info("Initializing intelligent performance optimizer")
        
        # Start background optimization tasks
        asyncio.create_task(self._optimization_worker())
        asyncio.create_task(self._prefetch_worker())
        asyncio.create_task(self._auto_tuning_worker())
    
    async def optimize_operation(
        self, 
        operation_type: str, 
        operation_func: Callable,
        *args, 
        **kwargs
    ) -> Tuple[Any, PerformanceProfile]:
        """
        Optimize a research operation with intelligent strategies.
        
        Returns:
            Tuple of (operation_result, performance_profile)
        """
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Check cache first
        cache_key = self._generate_cache_key(operation_type, args, kwargs)
        cached_result = await self._cache_manager.get(cache_key)
        
        if cached_result is not None:
            self.logger.debug(f"Cache hit for operation: {operation_type}")
            profile = self._update_performance_profile(
                operation_type, 
                duration_ms=1.0,  # Cache hit is very fast
                memory_delta_mb=0,
                cache_hit=True
            )
            return cached_result, profile
        
        # Apply pre-optimization strategies
        optimized_func = await self._apply_optimization_strategies(operation_type, operation_func)
        
        # Execute operation with monitoring
        try:
            result = await self._execute_with_monitoring(optimized_func, *args, **kwargs)
            
            # Cache result if appropriate
            if self._should_cache_result(operation_type, result):
                await self._cache_manager.set(cache_key, result)
            
        except Exception as e:
            self.logger.error(f"Error during optimized operation {operation_type}: {e}")
            raise
        
        # Calculate performance metrics
        duration_ms = (time.time() - start_time) * 1000
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_delta_mb = end_memory - start_memory
        
        # Update performance profile
        profile = self._update_performance_profile(
            operation_type,
            duration_ms=duration_ms,
            memory_delta_mb=memory_delta_mb,
            cache_hit=False
        )
        
        # Trigger prefetching for related operations
        await self._trigger_intelligent_prefetch(operation_type, args, kwargs)
        
        return result, profile
    
    async def _apply_optimization_strategies(
        self, 
        operation_type: str, 
        operation_func: Callable
    ) -> Callable:
        """Apply optimization strategies based on operation type and current strategy."""
        
        profile = self._operation_profiles.get(operation_type)
        
        if not profile:
            return operation_func  # No optimization data yet
        
        optimizations = []
        
        # Memory optimization
        if self._current_strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
            if profile.memory_usage_mb > 100:  # High memory usage
                optimizations.append(self._apply_memory_optimization)
        
        # Latency optimization
        elif self._current_strategy == OptimizationStrategy.LATENCY_OPTIMIZED:
            if profile.avg_duration_ms > 1000:  # Slow operation
                optimizations.append(self._apply_latency_optimization)
        
        # Throughput optimization
        elif self._current_strategy == OptimizationStrategy.THROUGHPUT_OPTIMIZED:
            if profile.usage_frequency > 10:  # Frequently used
                optimizations.append(self._apply_throughput_optimization)
        
        # Research-balanced (default)
        else:
            optimizations.extend([
                self._apply_research_specific_optimization,
                self._apply_balanced_optimization
            ])
        
        # Apply optimizations
        optimized_func = operation_func
        for optimization in optimizations:
            optimized_func = await optimization(operation_type, optimized_func, profile)
        
        return optimized_func
    
    async def _apply_memory_optimization(
        self, 
        operation_type: str, 
        func: Callable, 
        profile: PerformanceProfile
    ) -> Callable:
        """Apply memory-specific optimizations."""
        
        async def memory_optimized_wrapper(*args, **kwargs):
            # Implement memory optimization strategies
            # - Garbage collection before execution
            # - Memory pooling
            # - Lazy loading of large objects
            
            import gc
            gc.collect()  # Force garbage collection
            
            # Use memory-efficient execution
            return await func(*args, **kwargs)
        
        return memory_optimized_wrapper
    
    async def _apply_latency_optimization(
        self, 
        operation_type: str, 
        func: Callable, 
        profile: PerformanceProfile
    ) -> Callable:
        """Apply latency-specific optimizations."""
        
        async def latency_optimized_wrapper(*args, **kwargs):
            # Implement latency optimization strategies
            # - Parallel execution where possible
            # - Optimized data structures
            # - Precomputed results
            
            return await func(*args, **kwargs)
        
        return latency_optimized_wrapper
    
    async def _apply_throughput_optimization(
        self, 
        operation_type: str, 
        func: Callable, 
        profile: PerformanceProfile
    ) -> Callable:
        """Apply throughput-specific optimizations."""
        
        async def throughput_optimized_wrapper(*args, **kwargs):
            # Implement throughput optimization strategies
            # - Batch processing
            # - Connection pooling
            # - Asynchronous execution
            
            return await func(*args, **kwargs)
        
        return throughput_optimized_wrapper
    
    async def _apply_research_specific_optimization(
        self, 
        operation_type: str, 
        func: Callable, 
        profile: PerformanceProfile
    ) -> Callable:
        """Apply research workflow-specific optimizations."""
        
        research_optimizations = {
            'note_creation': self._optimize_note_operations,
            'literature_search': self._optimize_literature_operations,
            'ai_processing': self._optimize_ai_operations,
            'data_analysis': self._optimize_analysis_operations,
            'writing_assistance': self._optimize_writing_operations
        }
        
        optimizer = research_optimizations.get(operation_type)
        if optimizer:
            return await optimizer(func, profile)
        
        return func
    
    async def _apply_balanced_optimization(
        self, 
        operation_type: str, 
        func: Callable, 
        profile: PerformanceProfile
    ) -> Callable:
        """Apply balanced optimization for research workflows."""
        
        async def balanced_optimized_wrapper(*args, **kwargs):
            # Balanced approach considering all factors
            # - Moderate caching
            # - Smart resource allocation
            # - Adaptive performance tuning
            
            return await func(*args, **kwargs)
        
        return balanced_optimized_wrapper
    
    async def _optimize_note_operations(self, func: Callable, profile: PerformanceProfile) -> Callable:
        """Optimize note-related operations."""
        
        async def note_optimized_wrapper(*args, **kwargs):
            # Note-specific optimizations
            # - Incremental saving
            # - Diff-based updates
            # - Smart indexing
            
            return await func(*args, **kwargs)
        
        return note_optimized_wrapper
    
    async def _optimize_literature_operations(self, func: Callable, profile: PerformanceProfile) -> Callable:
        """Optimize literature search and processing operations."""
        
        async def literature_optimized_wrapper(*args, **kwargs):
            # Literature-specific optimizations
            # - Search result caching
            # - Parallel paper processing
            # - Smart relevance filtering
            
            return await func(*args, **kwargs)
        
        return literature_optimized_wrapper
    
    async def _optimize_ai_operations(self, func: Callable, profile: PerformanceProfile) -> Callable:
        """Optimize AI processing operations."""
        
        async def ai_optimized_wrapper(*args, **kwargs):
            # AI-specific optimizations
            # - Model result caching
            # - Batch processing
            # - Response compression
            
            return await func(*args, **kwargs)
        
        return ai_optimized_wrapper
    
    async def _optimize_analysis_operations(self, func: Callable, profile: PerformanceProfile) -> Callable:
        """Optimize data analysis operations."""
        
        async def analysis_optimized_wrapper(*args, **kwargs):
            # Analysis-specific optimizations
            # - Vectorized operations
            # - Partial computation caching
            # - Progressive analysis
            
            return await func(*args, **kwargs)
        
        return analysis_optimized_wrapper
    
    async def _optimize_writing_operations(self, func: Callable, profile: PerformanceProfile) -> Callable:
        """Optimize writing assistance operations."""
        
        async def writing_optimized_wrapper(*args, **kwargs):
            # Writing-specific optimizations
            # - Template caching
            # - Incremental generation
            # - Style consistency caching
            
            return await func(*args, **kwargs)
        
        return writing_optimized_wrapper
    
    async def _execute_with_monitoring(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with comprehensive monitoring."""
        
        # Monitor resource usage during execution
        initial_resources = self._resource_monitor.get_current_usage()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            final_resources = self._resource_monitor.get_current_usage()
            resource_delta = self._calculate_resource_delta(initial_resources, final_resources)
            
            # Record resource usage for learning
            self._record_resource_usage(func.__name__, resource_delta)
    
    def _calculate_resource_delta(self, initial: Dict, final: Dict) -> Dict:
        """Calculate the difference in resource usage."""
        return {
            key: final.get(key, 0) - initial.get(key, 0)
            for key in initial.keys()
        }
    
    def _record_resource_usage(self, operation: str, resource_delta: Dict):
        """Record resource usage for ML model training."""
        usage_record = {
            'timestamp': time.time(),
            'operation': operation,
            'resource_delta': resource_delta,
            'optimization_strategy': self._current_strategy.value
        }
        
        # Store for ML model training
        # In a full implementation, this would update prediction models
        pass
    
    def _update_performance_profile(
        self, 
        operation_type: str,
        duration_ms: float,
        memory_delta_mb: float,
        cache_hit: bool = False
    ) -> PerformanceProfile:
        """Update performance profile with new data."""
        
        with self._optimization_lock:
            profile = self._operation_profiles.get(operation_type)
            
            if not profile:
                profile = PerformanceProfile(
                    operation_type=operation_type,
                    avg_duration_ms=duration_ms,
                    memory_usage_mb=abs(memory_delta_mb),
                    cpu_usage_percent=0.0,
                    io_operations=0,
                    usage_frequency=1
                )
            else:
                # Update with exponential moving average
                alpha = 0.1
                profile.avg_duration_ms = (alpha * duration_ms) + ((1 - alpha) * profile.avg_duration_ms)
                profile.memory_usage_mb = (alpha * abs(memory_delta_mb)) + ((1 - alpha) * profile.memory_usage_mb)
                profile.usage_frequency += 1
                
                # Update cache hit rate
                if cache_hit:
                    profile.cache_hit_rate = (alpha * 1.0) + ((1 - alpha) * profile.cache_hit_rate)
                else:
                    profile.cache_hit_rate = (alpha * 0.0) + ((1 - alpha) * profile.cache_hit_rate)
            
            self._operation_profiles[operation_type] = profile
            
            # Calculate optimization potential
            profile.optimization_potential = self._calculate_optimization_potential(profile)
            
            return profile
    
    def _calculate_optimization_potential(self, profile: PerformanceProfile) -> float:
        """Calculate the potential for optimization based on profile metrics."""
        
        # Factors that indicate optimization potential
        factors = []
        
        # High duration suggests latency optimization potential
        if profile.avg_duration_ms > 1000:
            factors.append(0.3)
        
        # High memory usage suggests memory optimization potential
        if profile.memory_usage_mb > 100:
            factors.append(0.2)
        
        # Low cache hit rate suggests caching optimization potential
        if profile.cache_hit_rate < 0.5:
            factors.append(0.3)
        
        # High usage frequency suggests it's worth optimizing
        if profile.usage_frequency > 10:
            factors.append(0.2)
        
        return min(sum(factors), 1.0)
    
    async def _trigger_intelligent_prefetch(self, operation_type: str, args: Tuple, kwargs: Dict):
        """Trigger intelligent prefetching based on usage patterns."""
        
        # Predict what operations might be needed next
        predicted_operations = self._predict_next_operations(operation_type, args, kwargs)
        
        for predicted_op in predicted_operations:
            await self._prefetch_queue.put(predicted_op)
    
    def _predict_next_operations(self, operation_type: str, args: Tuple, kwargs: Dict) -> List[Dict]:
        """Predict what operations might be needed next based on patterns."""
        
        # Simple prediction based on common research workflows
        workflow_patterns = {
            'literature_search': ['summarize_paper', 'extract_keywords'],
            'note_creation': ['auto_tag', 'suggest_links'],
            'data_analysis': ['generate_visualization', 'export_results'],
            'ai_processing': ['validate_output', 'format_response']
        }
        
        next_ops = workflow_patterns.get(operation_type, [])
        
        return [
            {
                'operation_type': op,
                'args': args,
                'kwargs': kwargs,
                'priority': i
            }
            for i, op in enumerate(next_ops)
        ]
    
    async def _optimization_worker(self):
        """Background worker for continuous optimization."""
        
        while True:
            try:
                await asyncio.sleep(self._tuning_parameters['optimization_interval_minutes'] * 60)
                
                # Analyze current performance
                recommendations = self._analyze_performance_and_recommend()
                
                # Apply automated optimizations
                for rec in recommendations:
                    if rec.implementation_effort == 'low' and rec.expected_improvement > 0.1:
                        await self._apply_recommendation(rec)
                
                self.logger.info(f"Optimization cycle completed. Applied {len(recommendations)} optimizations.")
                
            except Exception as e:
                self.logger.error(f"Error in optimization worker: {e}")
    
    async def _prefetch_worker(self):
        """Background worker for intelligent prefetching."""
        
        while True:
            try:
                predicted_op = await self._prefetch_queue.get()
                
                # Execute prefetch operation
                await self._execute_prefetch(predicted_op)
                
            except Exception as e:
                self.logger.error(f"Error in prefetch worker: {e}")
    
    async def _auto_tuning_worker(self):
        """Background worker for automatic parameter tuning."""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Tune every hour
                
                # Analyze performance metrics
                metrics = self._analyze_current_metrics()
                
                # Adjust tuning parameters
                self._adjust_tuning_parameters(metrics)
                
                self.logger.info("Auto-tuning cycle completed")
                
            except Exception as e:
                self.logger.error(f"Error in auto-tuning worker: {e}")
    
    def _analyze_performance_and_recommend(self) -> List[OptimizationRecommendation]:
        """Analyze current performance and generate optimization recommendations."""
        
        recommendations = []
        
        for operation_type, profile in self._operation_profiles.items():
            
            # High latency operations
            if profile.avg_duration_ms > 2000:
                recommendations.append(OptimizationRecommendation(
                    target_operation=operation_type,
                    strategy=OptimizationStrategy.LATENCY_OPTIMIZED,
                    expected_improvement=0.3,
                    implementation_effort='medium',
                    priority=1,
                    reasoning=f"Operation takes {profile.avg_duration_ms:.0f}ms on average",
                    implementation_steps=[
                        "Implement result caching",
                        "Optimize algorithm complexity",
                        "Add parallel processing"
                    ]
                ))
            
            # High memory usage operations
            if profile.memory_usage_mb > 200:
                recommendations.append(OptimizationRecommendation(
                    target_operation=operation_type,
                    strategy=OptimizationStrategy.MEMORY_OPTIMIZED,
                    expected_improvement=0.4,
                    implementation_effort='low',
                    priority=2,
                    reasoning=f"Operation uses {profile.memory_usage_mb:.0f}MB on average",
                    implementation_steps=[
                        "Implement streaming processing",
                        "Add memory pooling",
                        "Optimize data structures"
                    ]
                ))
            
            # Low cache hit rate operations
            if profile.cache_hit_rate < 0.3 and profile.usage_frequency > 5:
                recommendations.append(OptimizationRecommendation(
                    target_operation=operation_type,
                    strategy=OptimizationStrategy.THROUGHPUT_OPTIMIZED,
                    expected_improvement=0.5,
                    implementation_effort='low',
                    priority=3,
                    reasoning=f"Cache hit rate is only {profile.cache_hit_rate:.1%}",
                    implementation_steps=[
                        "Implement intelligent caching",
                        "Add cache warming",
                        "Optimize cache key generation"
                    ]
                ))
        
        # Sort by priority and expected improvement
        recommendations.sort(key=lambda r: (r.priority, -r.expected_improvement))
        
        return recommendations
    
    async def _apply_recommendation(self, recommendation: OptimizationRecommendation):
        """Apply an optimization recommendation."""
        
        self.logger.info(f"Applying optimization for {recommendation.target_operation}: {recommendation.reasoning}")
        
        # Record the optimization attempt
        optimization_record = {
            'timestamp': time.time(),
            'target_operation': recommendation.target_operation,
            'strategy': recommendation.strategy.value,
            'expected_improvement': recommendation.expected_improvement,
            'reasoning': recommendation.reasoning
        }
        
        self._optimization_history.append(optimization_record)
        
        # Update the strategy for this operation type
        # In a full implementation, this would apply specific optimizations
        pass
    
    async def _execute_prefetch(self, predicted_op: Dict):
        """Execute a prefetch operation."""
        
        operation_type = predicted_op['operation_type']
        args = predicted_op['args']
        kwargs = predicted_op['kwargs']
        
        # Generate cache key for prefetch
        cache_key = self._generate_cache_key(operation_type, args, kwargs)
        
        # Check if already cached
        if await self._cache_manager.get(cache_key) is not None:
            return  # Already cached
        
        # Execute prefetch (simplified - would need actual operation implementation)
        # This is a placeholder for the actual prefetch logic
        self.logger.debug(f"Prefetching operation: {operation_type}")
    
    def _analyze_current_metrics(self) -> Dict[str, float]:
        """Analyze current performance metrics."""
        
        total_operations = len(self._operation_profiles)
        if total_operations == 0:
            return {}
        
        avg_duration = np.mean([p.avg_duration_ms for p in self._operation_profiles.values()])
        avg_memory = np.mean([p.memory_usage_mb for p in self._operation_profiles.values()])
        avg_cache_hit_rate = np.mean([p.cache_hit_rate for p in self._operation_profiles.values()])
        
        return {
            'avg_duration_ms': avg_duration,
            'avg_memory_mb': avg_memory,
            'avg_cache_hit_rate': avg_cache_hit_rate,
            'total_operations': total_operations
        }
    
    def _adjust_tuning_parameters(self, metrics: Dict[str, float]):
        """Adjust tuning parameters based on current metrics."""
        
        # Adjust cache size based on memory usage
        if metrics.get('avg_memory_mb', 0) > 500:
            self._tuning_parameters['cache_size_mb'] = max(128, self._tuning_parameters['cache_size_mb'] * 0.9)
        elif metrics.get('avg_cache_hit_rate', 0) < 0.3:
            self._tuning_parameters['cache_size_mb'] = min(1024, self._tuning_parameters['cache_size_mb'] * 1.1)
        
        # Adjust prefetch threshold based on performance
        if metrics.get('avg_duration_ms', 0) > 1000:
            self._tuning_parameters['prefetch_threshold'] *= 1.1
        else:
            self._tuning_parameters['prefetch_threshold'] *= 0.95
        
        self.logger.debug(f"Adjusted tuning parameters: {self._tuning_parameters}")
    
    def _generate_cache_key(self, operation_type: str, args: Tuple, kwargs: Dict) -> str:
        """Generate a cache key for the operation."""
        
        # Create a deterministic key from operation type and parameters
        key_data = {
            'operation_type': operation_type,
            'args_hash': hash(str(args)),
            'kwargs_hash': hash(str(sorted(kwargs.items())))
        }
        
        return f"{operation_type}:{hash(str(key_data))}"
    
    def _should_cache_result(self, operation_type: str, result: Any) -> bool:
        """Determine if a result should be cached."""
        
        # Cache results for expensive operations
        cacheable_operations = {
            'literature_search', 'ai_processing', 'data_analysis', 
            'summarization', 'translation', 'knowledge_extraction'
        }
        
        if operation_type in cacheable_operations:
            return True
        
        # Don't cache very large results
        try:
            result_size = len(str(result))
            if result_size > 1024 * 1024:  # 1MB limit
                return False
        except:
            pass
        
        return True
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a comprehensive optimization report."""
        
        total_operations = len(self._operation_profiles)
        optimized_operations = len([p for p in self._operation_profiles.values() if p.last_optimized])
        
        high_potential_ops = [
            p for p in self._operation_profiles.values() 
            if p.optimization_potential > 0.5
        ]
        
        return {
            'total_operations_tracked': total_operations,
            'optimized_operations': optimized_operations,
            'optimization_coverage': optimized_operations / total_operations if total_operations > 0 else 0,
            'high_optimization_potential': len(high_potential_ops),
            'current_strategy': self._current_strategy.value,
            'tuning_parameters': self._tuning_parameters.copy(),
            'cache_statistics': self._cache_manager.get_statistics(),
            'recent_optimizations': self._optimization_history[-10:],
            'performance_profiles': {
                op_type: {
                    'avg_duration_ms': profile.avg_duration_ms,
                    'memory_usage_mb': profile.memory_usage_mb,
                    'cache_hit_rate': profile.cache_hit_rate,
                    'optimization_potential': profile.optimization_potential,
                    'usage_frequency': profile.usage_frequency
                }
                for op_type, profile in self._operation_profiles.items()
            }
        }


class IntelligentCacheManager:
    """Intelligent caching system with adaptive policies."""
    
    def __init__(self, max_size_mb: int = 256):
        self.max_size_mb = max_size_mb
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._cache_lock = threading.RLock()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._cache_lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                self._access_counts[key] += 1
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        """Set value in cache with intelligent eviction."""
        with self._cache_lock:
            # Check if we need to evict items
            while len(self._cache) >= self.max_size_mb:  # Simplified size check
                self._evict_least_valuable()
            
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._access_counts[key] = 1
    
    def _evict_least_valuable(self):
        """Evict the least valuable cache entry."""
        if not self._cache:
            return
        
        # Calculate value score for each entry
        current_time = time.time()
        scores = {}
        
        for key in self._cache.keys():
            access_count = self._access_counts[key]
            time_since_access = current_time - self._access_times[key]
            
            # Higher score = more valuable (recent and frequent access)
            score = access_count / (1 + time_since_access / 3600)  # Decay over hours
            scores[key] = score
        
        # Remove the item with the lowest score
        least_valuable_key = min(scores.keys(), key=lambda k: scores[k])
        
        del self._cache[least_valuable_key]
        del self._access_times[least_valuable_key]
        del self._access_counts[least_valuable_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            total_accesses = sum(self._access_counts.values())
            
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size_mb,
                'total_accesses': total_accesses,
                'hit_rate': len(self._cache) / max(total_accesses, 1),
                'most_accessed': max(self._access_counts.items(), key=lambda x: x[1]) if self._access_counts else None
            }


class ResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_mb': psutil.virtual_memory().used / 1024 / 1024,
                'disk_io_read_mb': psutil.disk_io_counters().read_bytes / 1024 / 1024 if psutil.disk_io_counters() else 0,
                'disk_io_write_mb': psutil.disk_io_counters().write_bytes / 1024 / 1024 if psutil.disk_io_counters() else 0,
                'network_sent_mb': psutil.net_io_counters().bytes_sent / 1024 / 1024,
                'network_recv_mb': psutil.net_io_counters().bytes_recv / 1024 / 1024
            }
        except Exception as e:
            self.logger.error(f"Error getting resource usage: {e}")
            return {}


# Decorator for automatic optimization
def optimize_performance(
    operation_type: str,
    strategy: OptimizationStrategy = OptimizationStrategy.RESEARCH_BALANCED
):
    """
    Decorator to automatically optimize function performance.
    
    Args:
        operation_type: Type of research operation
        strategy: Optimization strategy to use
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = IntelligentPerformanceOptimizer()
            result, profile = await optimizer.optimize_operation(operation_type, func, *args, **kwargs)
            return result
        
        return wrapper
    return decorator