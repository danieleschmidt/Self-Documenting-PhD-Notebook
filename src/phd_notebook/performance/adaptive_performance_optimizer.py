"""
Adaptive Performance Optimization Engine
Self-learning system that continuously optimizes research workflows for maximum performance.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import hashlib
import gc
import sys

from ..core.note import Note, NoteType
from ..utils.logging import setup_logger


class OptimizationType(Enum):
    MEMORY_USAGE = "memory_usage"
    CPU_EFFICIENCY = "cpu_efficiency"
    IO_PERFORMANCE = "io_performance"
    CACHE_OPTIMIZATION = "cache_optimization"
    CONCURRENCY = "concurrency"
    DATA_PROCESSING = "data_processing"


class PerformanceMetric(Enum):
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_CONSUMPTION = "memory_consumption"
    CPU_UTILIZATION = "cpu_utilization"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"


@dataclass
class PerformanceMeasurement:
    """Single performance measurement."""
    measurement_id: str
    operation_name: str
    metric_type: PerformanceMetric
    value: float
    timestamp: datetime
    
    # Context
    operation_context: Dict[str, Any]
    system_state: Dict[str, Any]
    optimization_applied: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['metric_type'] = self.metric_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class OptimizationRule:
    """Performance optimization rule."""
    rule_id: str
    name: str
    optimization_type: OptimizationType
    condition: str  # Python expression
    action: str  # Function name to execute
    
    # Configuration
    priority: int = 1
    enabled: bool = True
    min_improvement_threshold: float = 0.05  # 5% minimum improvement
    
    # Statistics
    applications: int = 0
    successful_applications: int = 0
    avg_improvement: float = 0.0
    last_applied: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['optimization_type'] = self.optimization_type.value
        if self.last_applied:
            result['last_applied'] = self.last_applied.isoformat()
        return result


class AdaptiveCache:
    """Self-optimizing cache with intelligent eviction and prefetching."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        
        # Adaptive parameters
        self.eviction_strategy = "lru"  # lru, lfu, adaptive
        self.prefetch_enabled = True
        self.prefetch_patterns: Dict[str, List[str]] = defaultdict(list)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with access tracking."""
        if key in self.cache:
            self.hit_count += 1
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
            # Update prefetch patterns
            if self.prefetch_enabled:
                self._update_prefetch_patterns(key)
            
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with intelligent eviction."""
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_item()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.access_counts[key] += 1
    
    def _evict_item(self):
        """Evict item using adaptive strategy."""
        if not self.cache:
            return
        
        if self.eviction_strategy == "lru":
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        elif self.eviction_strategy == "lfu":
            oldest_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        else:  # adaptive
            # Combine recency and frequency
            current_time = time.time()
            scores = {}
            for key in self.cache:
                recency_score = current_time - self.access_times[key]
                frequency_score = 1.0 / max(self.access_counts[key], 1)
                scores[key] = recency_score * frequency_score
            oldest_key = min(scores.keys(), key=lambda k: scores[k])
        
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        del self.access_counts[oldest_key]
    
    def _update_prefetch_patterns(self, accessed_key: str):
        """Update prefetch patterns based on access patterns."""
        # This is a simplified version - in production would use more sophisticated ML
        if hasattr(self, '_last_accessed'):
            pattern_key = self._last_accessed
            if accessed_key not in self.prefetch_patterns[pattern_key]:
                self.prefetch_patterns[pattern_key].append(accessed_key)
                # Keep only recent patterns
                if len(self.prefetch_patterns[pattern_key]) > 5:
                    self.prefetch_patterns[pattern_key].pop(0)
        
        self._last_accessed = accessed_key
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / max(total, 1)
    
    def optimize_strategy(self):
        """Optimize caching strategy based on performance."""
        hit_rate = self.get_hit_rate()
        
        # Switch strategies based on hit rate
        if hit_rate < 0.6:
            if self.eviction_strategy == "lru":
                self.eviction_strategy = "lfu"
            elif self.eviction_strategy == "lfu":
                self.eviction_strategy = "adaptive"
            else:
                self.eviction_strategy = "lru"


class PerformanceProfiler:
    """Lightweight performance profiler for research operations."""
    
    def __init__(self):
        self.measurements: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        self.operation_stats: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0
        })
    
    def profile_operation(self, operation_name: str, operation_context: Dict = None):
        """Context manager for profiling operations."""
        return OperationProfiler(self, operation_name, operation_context or {})
    
    def record_measurement(
        self,
        operation_name: str,
        duration: float,
        context: Dict,
        error: bool = False
    ):
        """Record a performance measurement."""
        measurement = PerformanceMeasurement(
            measurement_id=f"{operation_name}_{int(time.time()*1000)}",
            operation_name=operation_name,
            metric_type=PerformanceMetric.RESPONSE_TIME,
            value=duration,
            timestamp=datetime.now(),
            operation_context=context,
            system_state=self._get_system_state()
        )
        
        self.measurements.append(measurement)
        
        # Update operation stats
        stats = self.operation_stats[operation_name]
        stats['count'] += 1
        stats['total_time'] += duration
        stats['min_time'] = min(stats['min_time'], duration)
        stats['max_time'] = max(stats['max_time'], duration)
        
        if error:
            stats['error_count'] += 1
    
    def _get_system_state(self) -> Dict:
        """Get current system state."""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'active_threads': threading.active_count()
            }
        except ImportError:
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'active_threads': threading.active_count()
            }
    
    def get_operation_stats(self, operation_name: str) -> Dict:
        """Get statistics for a specific operation."""
        stats = self.operation_stats[operation_name]
        if stats['count'] == 0:
            return {}
        
        return {
            'count': stats['count'],
            'avg_time': stats['total_time'] / stats['count'],
            'min_time': stats['min_time'],
            'max_time': stats['max_time'],
            'error_rate': stats['error_count'] / stats['count'],
            'total_time': stats['total_time']
        }
    
    def get_slowest_operations(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get the slowest operations by average time."""
        operation_times = []
        for op_name, stats in self.operation_stats.items():
            if stats['count'] > 0:
                avg_time = stats['total_time'] / stats['count']
                operation_times.append((op_name, avg_time))
        
        return sorted(operation_times, key=lambda x: x[1], reverse=True)[:limit]


class OperationProfiler:
    """Context manager for profiling individual operations."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str, context: Dict):
        self.profiler = profiler
        self.operation_name = operation_name
        self.context = context
        self.start_time = None
        self.error_occurred = False
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.error_occurred = exc_type is not None
            self.profiler.record_measurement(
                self.operation_name,
                duration,
                self.context,
                self.error_occurred
            )


class AdaptivePerformanceOptimizer:
    """
    Self-learning performance optimization system that continuously monitors
    and optimizes research workflows for maximum efficiency.
    """
    
    def __init__(self, notebook_path: Path):
        self.logger = setup_logger("performance.adaptive_optimizer")
        self.notebook_path = notebook_path
        
        # Core components
        self.profiler = PerformanceProfiler()
        self.cache = AdaptiveCache(max_size=2000)
        
        # Optimization rules and measurements
        self.optimization_rules: Dict[str, OptimizationRule] = {}
        self.performance_history: List[PerformanceMeasurement] = []
        
        # Optimization state
        self.is_optimizing = False
        self.optimization_interval = 60  # seconds
        self._optimization_task: Optional[asyncio.Task] = None
        
        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        # Create directories
        self.perf_dir = notebook_path / "performance"
        self.profiles_dir = self.perf_dir / "profiles"
        self.optimizations_dir = self.perf_dir / "optimizations"
        
        for dir_path in [self.perf_dir, self.profiles_dir, self.optimizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimization rules
        self._initialize_optimization_rules()
        self._load_existing_data()
    
    def _initialize_optimization_rules(self):
        """Initialize default optimization rules."""
        
        # Memory optimization rule
        memory_rule = OptimizationRule(
            rule_id="memory_cleanup",
            name="Automatic Memory Cleanup",
            optimization_type=OptimizationType.MEMORY_USAGE,
            condition="memory_usage > 80",  # More than 80% memory usage
            action="cleanup_memory",
            priority=1
        )
        
        # Cache optimization rule
        cache_rule = OptimizationRule(
            rule_id="cache_optimization",
            name="Adaptive Cache Strategy",
            optimization_type=OptimizationType.CACHE_OPTIMIZATION,
            condition="cache_hit_rate < 0.6",  # Less than 60% hit rate
            action="optimize_cache_strategy",
            priority=2
        )
        
        # I/O optimization rule
        io_rule = OptimizationRule(
            rule_id="io_batching",
            name="I/O Operation Batching",
            optimization_type=OptimizationType.IO_PERFORMANCE,
            condition="avg_io_time > 0.1",  # More than 100ms average I/O
            action="enable_io_batching",
            priority=3
        )
        
        # Concurrency optimization rule
        concurrency_rule = OptimizationRule(
            rule_id="adaptive_concurrency",
            name="Adaptive Concurrency Control",
            optimization_type=OptimizationType.CONCURRENCY,
            condition="cpu_utilization < 50 and active_threads < 4",
            action="increase_concurrency",
            priority=2
        )
        
        # Data processing optimization
        data_rule = OptimizationRule(
            rule_id="data_processing_optimization",
            name="Data Processing Pipeline Optimization",
            optimization_type=OptimizationType.DATA_PROCESSING,
            condition="data_processing_time > 2.0",  # More than 2 seconds
            action="optimize_data_processing",
            priority=1
        )
        
        self.optimization_rules = {
            "memory_cleanup": memory_rule,
            "cache_optimization": cache_rule,
            "io_batching": io_rule,
            "adaptive_concurrency": concurrency_rule,
            "data_processing_optimization": data_rule
        }
    
    def _load_existing_data(self):
        """Load existing performance data."""
        try:
            # Load optimization rules
            rules_file = self.perf_dir / "optimization_rules.json"
            if rules_file.exists():
                with open(rules_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for rule_id, rule_data in data.items():
                        rule_data['optimization_type'] = OptimizationType(rule_data['optimization_type'])
                        if rule_data.get('last_applied'):
                            rule_data['last_applied'] = datetime.fromisoformat(rule_data['last_applied'])
                        self.optimization_rules[rule_id] = OptimizationRule(**rule_data)
            
            # Load performance baselines
            baselines_file = self.perf_dir / "baselines.json"
            if baselines_file.exists():
                with open(baselines_file, 'r', encoding='utf-8') as f:
                    self.baselines = json.load(f)
            
            self.logger.info(f"Loaded {len(self.optimization_rules)} optimization rules, "
                           f"{len(self.baselines)} baselines")
                           
        except Exception as e:
            self.logger.error(f"Error loading performance data: {e}")
    
    def profile_operation(self, operation_name: str, **context):
        """Profile a research operation."""
        return self.profiler.profile_operation(operation_name, context)
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get item from adaptive cache."""
        return self.cache.get(key)
    
    def cache_put(self, key: str, value: Any):
        """Put item in adaptive cache."""
        self.cache.put(key, value)
    
    def cache_key(self, operation: str, **params) -> str:
        """Generate cache key for operation and parameters."""
        key_data = f"{operation}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def start_optimization(self):
        """Start the continuous optimization process."""
        if self.is_optimizing:
            self.logger.warning("Optimization already running")
            return
        
        self.is_optimizing = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self.logger.info("Adaptive performance optimization started")
    
    async def stop_optimization(self):
        """Stop the optimization process."""
        self.is_optimizing = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Adaptive performance optimization stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.is_optimizing:
            try:
                # Collect performance data
                await self._collect_performance_data()
                
                # Apply optimizations
                await self._apply_optimizations()
                
                # Update baselines
                self._update_baselines()
                
                # Save state
                self._save_state()
                
                # Wait for next cycle
                await asyncio.sleep(self.optimization_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30)  # Wait 30 seconds on error
    
    async def _collect_performance_data(self):
        """Collect current performance data."""
        try:
            # Get system metrics
            import psutil
            
            current_metrics = {
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_utilization': psutil.cpu_percent(interval=1),
                'cache_hit_rate': self.cache.get_hit_rate(),
                'active_threads': threading.active_count(),
                'timestamp': datetime.now()
            }
            
            # Get operation performance
            operation_stats = {}
            for op_name in self.profiler.operation_stats:
                stats = self.profiler.get_operation_stats(op_name)
                if stats:
                    operation_stats[op_name] = stats
            
            current_metrics['operation_stats'] = operation_stats
            
            # Store for analysis
            self.performance_history.append(current_metrics)
            
            # Keep only recent history (last 1000 measurements)
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error collecting performance data: {e}")
    
    async def _apply_optimizations(self):
        """Apply optimization rules based on current performance."""
        current_metrics = self._get_current_metrics()
        
        for rule_id, rule in self.optimization_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check if rule condition is met
                if self._evaluate_condition(rule.condition, current_metrics):
                    # Apply optimization
                    success = await self._apply_optimization_action(rule.action, current_metrics)
                    
                    # Update rule statistics
                    rule.applications += 1
                    rule.last_applied = datetime.now()
                    
                    if success:
                        rule.successful_applications += 1
                        self.logger.info(f"Applied optimization: {rule.name}")
                    else:
                        self.logger.warning(f"Failed to apply optimization: {rule.name}")
                        
            except Exception as e:
                self.logger.error(f"Error applying optimization rule {rule_id}: {e}")
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.performance_history:
            return {}
        
        latest = self.performance_history[-1]
        
        # Calculate derived metrics
        metrics = latest.copy()
        
        # Average operation times
        if 'operation_stats' in metrics:
            avg_times = {}
            for op_name, stats in metrics['operation_stats'].items():
                avg_times[f"avg_{op_name}_time"] = stats.get('avg_time', 0)
            metrics.update(avg_times)
        
        # Recent trends (if enough history)
        if len(self.performance_history) >= 5:
            recent = self.performance_history[-5:]
            memory_trend = [m.get('memory_usage', 0) for m in recent]
            cpu_trend = [m.get('cpu_utilization', 0) for m in recent]
            
            metrics['memory_trend'] = sum(memory_trend) / len(memory_trend)
            metrics['cpu_trend'] = sum(cpu_trend) / len(cpu_trend)
        
        return metrics
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Safely evaluate optimization condition."""
        try:
            # Create safe evaluation context
            safe_context = {
                'memory_usage': metrics.get('memory_usage', 0),
                'cpu_utilization': metrics.get('cpu_utilization', 0),
                'cache_hit_rate': metrics.get('cache_hit_rate', 1.0),
                'active_threads': metrics.get('active_threads', 1),
                'memory_trend': metrics.get('memory_trend', 0),
                'cpu_trend': metrics.get('cpu_trend', 0)
            }
            
            # Add operation-specific metrics
            for key, value in metrics.items():
                if key.startswith('avg_') and key.endswith('_time'):
                    safe_context[key] = value
            
            # Safely evaluate condition
            return eval(condition, {"__builtins__": {}}, safe_context)
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    async def _apply_optimization_action(self, action: str, metrics: Dict[str, Any]) -> bool:
        """Apply optimization action."""
        try:
            if action == "cleanup_memory":
                return self._cleanup_memory()
            elif action == "optimize_cache_strategy":
                return self._optimize_cache_strategy()
            elif action == "enable_io_batching":
                return self._enable_io_batching()
            elif action == "increase_concurrency":
                return self._increase_concurrency()
            elif action == "optimize_data_processing":
                return self._optimize_data_processing()
            else:
                self.logger.warning(f"Unknown optimization action: {action}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error applying optimization action '{action}': {e}")
            return False
    
    def _cleanup_memory(self) -> bool:
        """Perform memory cleanup optimization."""
        try:
            # Clear cache of less frequently used items
            if len(self.cache.cache) > self.cache.max_size * 0.8:
                items_to_remove = int(self.cache.max_size * 0.2)
                for _ in range(items_to_remove):
                    self.cache._evict_item()
            
            # Force garbage collection
            collected = gc.collect()
            
            # Clear old performance history
            if len(self.performance_history) > 500:
                self.performance_history = self.performance_history[-500:]
            
            self.logger.debug(f"Memory cleanup: collected {collected} objects")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def _optimize_cache_strategy(self) -> bool:
        """Optimize cache strategy based on performance."""
        try:
            self.cache.optimize_strategy()
            self.logger.debug(f"Cache strategy optimized to: {self.cache.eviction_strategy}")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            return False
    
    def _enable_io_batching(self) -> bool:
        """Enable I/O batching optimization."""
        try:
            # This is a placeholder - in a real implementation, this would
            # configure I/O operations to batch multiple requests together
            self.logger.debug("I/O batching optimization applied")
            return True
            
        except Exception as e:
            self.logger.error(f"I/O batching optimization failed: {e}")
            return False
    
    def _increase_concurrency(self) -> bool:
        """Increase concurrency for better CPU utilization."""
        try:
            # This is a placeholder - in a real implementation, this would
            # adjust thread pool sizes or async concurrency limits
            self.logger.debug("Concurrency optimization applied")
            return True
            
        except Exception as e:
            self.logger.error(f"Concurrency optimization failed: {e}")
            return False
    
    def _optimize_data_processing(self) -> bool:
        """Optimize data processing pipelines."""
        try:
            # This is a placeholder - in a real implementation, this would
            # optimize data processing workflows, enable vectorization, etc.
            self.logger.debug("Data processing optimization applied")
            return True
            
        except Exception as e:
            self.logger.error(f"Data processing optimization failed: {e}")
            return False
    
    def _update_baselines(self):
        """Update performance baselines."""
        if not self.performance_history:
            return
        
        latest = self.performance_history[-1]
        
        # Update system baselines
        for metric in ['memory_usage', 'cpu_utilization', 'cache_hit_rate']:
            if metric in latest:
                if 'system' not in self.baselines:
                    self.baselines['system'] = {}
                
                # Exponential moving average
                alpha = 0.1
                if metric in self.baselines['system']:
                    self.baselines['system'][metric] = (
                        alpha * latest[metric] + 
                        (1 - alpha) * self.baselines['system'][metric]
                    )
                else:
                    self.baselines['system'][metric] = latest[metric]
        
        # Update operation baselines
        if 'operation_stats' in latest:
            for op_name, stats in latest['operation_stats'].items():
                if op_name not in self.baselines:
                    self.baselines[op_name] = {}
                
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        alpha = 0.1
                        if stat_name in self.baselines[op_name]:
                            self.baselines[op_name][stat_name] = (
                                alpha * value + 
                                (1 - alpha) * self.baselines[op_name][stat_name]
                            )
                        else:
                            self.baselines[op_name][stat_name] = value
    
    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Filter recent performance data
        recent_data = [
            data for data in self.performance_history
            if data.get('timestamp', datetime.min) >= cutoff_time
        ]
        
        report = {
            "period": f"Last {days} days",
            "generated_at": datetime.now().isoformat(),
            "summary": self._generate_performance_summary(recent_data),
            "optimization_results": self._analyze_optimization_results(),
            "cache_performance": self._analyze_cache_performance(),
            "operation_performance": self._analyze_operation_performance(),
            "recommendations": self._generate_performance_recommendations(),
            "baselines": self.baselines
        }
        
        return report
    
    def _generate_performance_summary(self, recent_data: List[Dict]) -> Dict[str, Any]:
        """Generate performance summary from recent data."""
        if not recent_data:
            return {"error": "No recent performance data available"}
        
        # Calculate averages
        memory_usage = [d.get('memory_usage', 0) for d in recent_data]
        cpu_usage = [d.get('cpu_utilization', 0) for d in recent_data]
        cache_hit_rates = [d.get('cache_hit_rate', 0) for d in recent_data]
        
        return {
            "avg_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
            "avg_cache_hit_rate": sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else 0,
            "peak_memory_usage": max(memory_usage) if memory_usage else 0,
            "peak_cpu_usage": max(cpu_usage) if cpu_usage else 0,
            "data_points": len(recent_data)
        }
    
    def _analyze_optimization_results(self) -> Dict[str, Any]:
        """Analyze optimization rule effectiveness."""
        results = {}
        
        for rule_id, rule in self.optimization_rules.items():
            if rule.applications > 0:
                success_rate = rule.successful_applications / rule.applications
                results[rule_id] = {
                    "name": rule.name,
                    "applications": rule.applications,
                    "success_rate": success_rate,
                    "last_applied": rule.last_applied.isoformat() if rule.last_applied else None,
                    "enabled": rule.enabled
                }
        
        return results
    
    def _analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance."""
        return {
            "hit_rate": self.cache.get_hit_rate(),
            "cache_size": len(self.cache.cache),
            "max_size": self.cache.max_size,
            "utilization": len(self.cache.cache) / self.cache.max_size,
            "eviction_strategy": self.cache.eviction_strategy,
            "prefetch_enabled": self.cache.prefetch_enabled
        }
    
    def _analyze_operation_performance(self) -> Dict[str, Any]:
        """Analyze operation-level performance."""
        operation_analysis = {}
        
        slowest_ops = self.profiler.get_slowest_operations(10)
        
        for op_name, avg_time in slowest_ops:
            stats = self.profiler.get_operation_stats(op_name)
            operation_analysis[op_name] = {
                "avg_time": avg_time,
                "total_calls": stats.get('count', 0),
                "error_rate": stats.get('error_rate', 0),
                "min_time": stats.get('min_time', 0),
                "max_time": stats.get('max_time', 0)
            }
        
        return operation_analysis
    
    def _generate_performance_recommendations(self) -> List[Dict[str, str]]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Cache recommendations
        if self.cache.get_hit_rate() < 0.5:
            recommendations.append({
                "category": "Cache Performance",
                "priority": "high",
                "recommendation": "Cache hit rate is low - consider caching more frequently used data",
                "action": "Review cache usage patterns and increase cache size if needed"
            })
        
        # Memory recommendations
        if self.performance_history:
            latest = self.performance_history[-1]
            if latest.get('memory_usage', 0) > 80:
                recommendations.append({
                    "category": "Memory Usage",
                    "priority": "high",
                    "recommendation": "High memory usage detected",
                    "action": "Enable more aggressive memory cleanup and consider reducing cache size"
                })
        
        # Operation performance recommendations
        slowest_ops = self.profiler.get_slowest_operations(3)
        for op_name, avg_time in slowest_ops:
            if avg_time > 1.0:  # More than 1 second
                recommendations.append({
                    "category": "Operation Performance",
                    "priority": "medium",
                    "recommendation": f"Operation '{op_name}' is slow (avg: {avg_time:.2f}s)",
                    "action": f"Profile and optimize the '{op_name}' operation"
                })
        
        return recommendations
    
    def _save_state(self):
        """Save optimizer state to storage."""
        try:
            # Save optimization rules
            rules_file = self.perf_dir / "optimization_rules.json"
            rules_data = {}
            for rule_id, rule in self.optimization_rules.items():
                rules_data[rule_id] = rule.to_dict()
            
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            # Save baselines
            baselines_file = self.perf_dir / "baselines.json"
            with open(baselines_file, 'w', encoding='utf-8') as f:
                json.dump(self.baselines, f, indent=2, ensure_ascii=False)
            
            # Save recent performance history (last 100 entries)
            if self.performance_history:
                history_file = self.perf_dir / "performance_history.json"
                recent_history = self.performance_history[-100:]
                
                # Convert datetime objects to strings for JSON serialization
                serializable_history = []
                for entry in recent_history:
                    entry_copy = entry.copy()
                    if 'timestamp' in entry_copy:
                        entry_copy['timestamp'] = entry_copy['timestamp'].isoformat()
                    serializable_history.append(entry_copy)
                
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_history, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            self.logger.error(f"Error saving optimizer state: {e}")
    
    def add_custom_optimization_rule(
        self,
        rule_id: str,
        name: str,
        optimization_type: OptimizationType,
        condition: str,
        action: str,
        priority: int = 1
    ):
        """Add custom optimization rule."""
        rule = OptimizationRule(
            rule_id=rule_id,
            name=name,
            optimization_type=optimization_type,
            condition=condition,
            action=action,
            priority=priority
        )
        
        self.optimization_rules[rule_id] = rule
        self.logger.info(f"Added custom optimization rule: {rule_id}")
    
    def disable_optimization_rule(self, rule_id: str):
        """Disable an optimization rule."""
        if rule_id in self.optimization_rules:
            self.optimization_rules[rule_id].enabled = False
            self.logger.info(f"Disabled optimization rule: {rule_id}")
    
    def enable_optimization_rule(self, rule_id: str):
        """Enable an optimization rule."""
        if rule_id in self.optimization_rules:
            self.optimization_rules[rule_id].enabled = True
            self.logger.info(f"Enabled optimization rule: {rule_id}")