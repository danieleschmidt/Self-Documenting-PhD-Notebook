"""Scaling and performance optimization for pipeline guard."""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
import pickle

from ..utils.logging import get_logger
from ..performance.cache_manager import CacheManager
from ..performance.concurrent_processor import ConcurrentProcessor


@dataclass
class ScalingConfig:
    """Configuration for scaling features."""
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_concurrent_heals: int = 5
    max_concurrent_monitors: int = 10
    batch_size: int = 50
    enable_predictive_scaling: bool = True
    metrics_retention_days: int = 30
    adaptive_intervals: bool = True
    min_check_interval: int = 10  # seconds
    max_check_interval: int = 300  # seconds


class PerformanceOptimizer:
    """Optimizes performance through caching, batching, and smart scheduling."""
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.logger = get_logger(__name__)
        
        # Performance components
        if self.config.enable_caching:
            self.cache = CacheManager(default_ttl=self.config.cache_ttl)
        else:
            self.cache = None
        
        self.concurrent_processor = ConcurrentProcessor(
            max_workers=max(self.config.max_concurrent_heals, self.config.max_concurrent_monitors)
        )
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.adaptive_controller = AdaptiveController(self.config) if self.config.adaptive_intervals else None
    
    async def optimized_pipeline_check(self, 
                                     check_function,
                                     pipeline_ids: List[str]) -> Dict[str, Any]:
        """Perform optimized pipeline checks with caching and batching."""
        start_time = time.time()
        
        # Check cache first
        cached_results = {}
        uncached_ids = []
        
        if self.cache:
            for pipeline_id in pipeline_ids:
                cache_key = f"pipeline_status:{pipeline_id}"
                cached = await self.cache.get(cache_key)
                if cached:
                    cached_results[pipeline_id] = cached
                else:
                    uncached_ids.append(pipeline_id)
        else:
            uncached_ids = pipeline_ids
        
        # Batch process uncached items
        fresh_results = {}
        if uncached_ids:
            # Split into batches
            batches = [
                uncached_ids[i:i + self.config.batch_size]
                for i in range(0, len(uncached_ids), self.config.batch_size)
            ]
            
            # Process batches concurrently
            batch_tasks = [
                self.concurrent_processor.submit(
                    self._check_pipeline_batch,
                    check_function,
                    batch
                )
                for batch in batches
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Combine batch results
            for batch_result in batch_results:
                fresh_results.update(batch_result)
            
            # Cache fresh results
            if self.cache:
                for pipeline_id, status in fresh_results.items():
                    cache_key = f"pipeline_status:{pipeline_id}"
                    await self.cache.set(cache_key, status, ttl=self.config.cache_ttl)
        
        # Combine cached and fresh results
        all_results = {**cached_results, **fresh_results}
        
        # Record performance metrics
        duration = time.time() - start_time
        self.metrics.record_operation("pipeline_check", duration, len(pipeline_ids))
        
        self.logger.debug(f"Pipeline check completed in {duration:.2f}s for {len(pipeline_ids)} pipelines "
                         f"({len(cached_results)} cached, {len(fresh_results)} fresh)")
        
        return all_results
    
    async def _check_pipeline_batch(self, 
                                   check_function,
                                   pipeline_ids: List[str]) -> Dict[str, Any]:
        """Check a batch of pipelines."""
        try:
            # Assume check_function can handle multiple pipelines
            return await check_function(pipeline_ids)
        except Exception as e:
            self.logger.error(f"Error checking pipeline batch: {e}")
            # Fallback to individual checks
            results = {}
            for pipeline_id in pipeline_ids:
                try:
                    result = await check_function([pipeline_id])
                    results.update(result)
                except Exception as inner_e:
                    self.logger.error(f"Error checking individual pipeline {pipeline_id}: {inner_e}")
                    # Return error status
                    results[pipeline_id] = {"error": str(inner_e), "state": "error"}
            return results
    
    async def optimized_healing(self, 
                               healing_function,
                               healing_requests: List[Tuple[str, Any]]) -> Dict[str, bool]:
        """Perform optimized healing with concurrency control."""
        start_time = time.time()
        
        # Deduplicate requests for same pipeline
        unique_requests = {}
        for pipeline_id, failure_analysis in healing_requests:
            if pipeline_id not in unique_requests:
                unique_requests[pipeline_id] = failure_analysis
        
        # Process with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_heals)
        
        async def heal_with_semaphore(pipeline_id: str, analysis: Any) -> Tuple[str, bool]:
            async with semaphore:
                try:
                    result = await healing_function(pipeline_id, analysis)
                    return pipeline_id, result
                except Exception as e:
                    self.logger.error(f"Error healing pipeline {pipeline_id}: {e}")
                    return pipeline_id, False
        
        # Execute all healing operations
        healing_tasks = [
            heal_with_semaphore(pipeline_id, analysis)
            for pipeline_id, analysis in unique_requests.items()
        ]
        
        results = await asyncio.gather(*healing_tasks)
        
        # Convert to dict
        healing_results = dict(results)
        
        # Record metrics
        duration = time.time() - start_time
        self.metrics.record_operation("healing", duration, len(healing_results))
        
        successful_heals = sum(1 for success in healing_results.values() if success)
        self.logger.info(f"Healing completed in {duration:.2f}s: {successful_heals}/{len(healing_results)} successful")
        
        return healing_results
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        # Analyze metrics
        avg_check_time = self.metrics.get_average_duration("pipeline_check")
        avg_heal_time = self.metrics.get_average_duration("healing")
        
        if avg_check_time and avg_check_time > 5.0:
            recommendations.append("Consider increasing cache TTL to reduce pipeline check frequency")
        
        if avg_heal_time and avg_heal_time > 30.0:
            recommendations.append("Consider increasing max_concurrent_heals for better parallelism")
        
        cache_hit_rate = self.metrics.get_cache_hit_rate()
        if cache_hit_rate and cache_hit_rate < 0.5:
            recommendations.append("Cache hit rate is low - consider adjusting cache TTL or cache strategy")
        
        return recommendations
    
    async def cleanup_old_data(self) -> None:
        """Clean up old performance data and cached items."""
        cutoff_date = datetime.now() - timedelta(days=self.config.metrics_retention_days)
        
        # Clean up metrics
        self.metrics.cleanup_old_metrics(cutoff_date)
        
        # Clean up cache
        if self.cache:
            await self.cache.cleanup_expired()
        
        self.logger.info("Cleaned up old performance data")


class PerformanceMetrics:
    """Tracks performance metrics for optimization."""
    
    def __init__(self):
        self.operation_times = defaultdict(deque)
        self.cache_stats = {"hits": 0, "misses": 0}
        self.concurrent_operations = defaultdict(int)
        self.error_counts = defaultdict(int)
        
    def record_operation(self, operation_type: str, duration: float, item_count: int = 1):
        """Record an operation's performance."""
        timestamp = datetime.now()
        self.operation_times[operation_type].append({
            "timestamp": timestamp,
            "duration": duration,
            "item_count": item_count,
            "rate": item_count / duration if duration > 0 else 0
        })
        
        # Keep only recent data (last 1000 operations per type)
        if len(self.operation_times[operation_type]) > 1000:
            self.operation_times[operation_type].popleft()
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_stats["hits"] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_stats["misses"] += 1
    
    def get_average_duration(self, operation_type: str, minutes: int = 60) -> Optional[float]:
        """Get average duration for operation type in recent time window."""
        if operation_type not in self.operation_times:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_ops = [
            op for op in self.operation_times[operation_type]
            if op["timestamp"] > cutoff_time
        ]
        
        if not recent_ops:
            return None
        
        return sum(op["duration"] for op in recent_ops) / len(recent_ops)
    
    def get_cache_hit_rate(self) -> Optional[float]:
        """Get cache hit rate."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total == 0:
            return None
        return self.cache_stats["hits"] / total
    
    def get_throughput(self, operation_type: str, minutes: int = 60) -> Optional[float]:
        """Get throughput (operations per second) for recent time window."""
        if operation_type not in self.operation_times:
            return None
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_ops = [
            op for op in self.operation_times[operation_type]
            if op["timestamp"] > cutoff_time
        ]
        
        if not recent_ops:
            return None
        
        total_items = sum(op["item_count"] for op in recent_ops)
        time_span = (recent_ops[-1]["timestamp"] - recent_ops[0]["timestamp"]).total_seconds()
        
        if time_span <= 0:
            return None
        
        return total_items / time_span
    
    def cleanup_old_metrics(self, cutoff_date: datetime):
        """Remove metrics older than cutoff date."""
        for operation_type in self.operation_times:
            old_count = len(self.operation_times[operation_type])
            self.operation_times[operation_type] = deque([
                op for op in self.operation_times[operation_type]
                if op["timestamp"] > cutoff_date
            ])
            new_count = len(self.operation_times[operation_type])
            
            if old_count > new_count:
                logger = get_logger(__name__)
                logger.debug(f"Cleaned up {old_count - new_count} old {operation_type} metrics")


class AdaptiveController:
    """Controls adaptive scaling based on system load and performance."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
        self.current_check_interval = config.min_check_interval
        self.load_history = deque(maxlen=100)  # Keep last 100 load measurements
        self.performance_history = deque(maxlen=100)
        
        # Adaptive parameters
        self.scale_up_threshold = 0.8  # Scale up if load > 80%
        self.scale_down_threshold = 0.3  # Scale down if load < 30%
        self.adjustment_factor = 1.2  # 20% adjustments
    
    def update_load_metrics(self, 
                           active_pipelines: int,
                           failed_pipelines: int,
                           healing_queue_size: int,
                           avg_response_time: float):
        """Update load metrics for adaptive control."""
        # Calculate composite load score
        load_score = min(1.0, (
            (failed_pipelines / max(1, active_pipelines)) * 0.4 +  # Failure rate
            (healing_queue_size / 10) * 0.3 +  # Queue pressure
            (min(avg_response_time, 30) / 30) * 0.3  # Response time pressure
        ))
        
        self.load_history.append({
            "timestamp": datetime.now(),
            "load_score": load_score,
            "active_pipelines": active_pipelines,
            "failed_pipelines": failed_pipelines,
            "healing_queue_size": healing_queue_size,
            "avg_response_time": avg_response_time
        })
        
        # Update check interval based on load
        self._adjust_check_interval(load_score)
    
    def _adjust_check_interval(self, current_load: float):
        """Adjust check interval based on current load."""
        if current_load > self.scale_up_threshold:
            # High load - check more frequently
            new_interval = max(
                self.config.min_check_interval,
                int(self.current_check_interval / self.adjustment_factor)
            )
            if new_interval != self.current_check_interval:
                self.logger.info(f"Scaling up: reducing check interval from {self.current_check_interval}s to {new_interval}s")
                self.current_check_interval = new_interval
        
        elif current_load < self.scale_down_threshold:
            # Low load - check less frequently
            new_interval = min(
                self.config.max_check_interval,
                int(self.current_check_interval * self.adjustment_factor)
            )
            if new_interval != self.current_check_interval:
                self.logger.info(f"Scaling down: increasing check interval from {self.current_check_interval}s to {new_interval}s")
                self.current_check_interval = new_interval
    
    def get_optimal_check_interval(self) -> int:
        """Get the current optimal check interval."""
        return self.current_check_interval
    
    def predict_scaling_needs(self) -> Dict[str, Any]:
        """Predict future scaling needs based on trends."""
        if len(self.load_history) < 10:
            return {"prediction": "insufficient_data"}
        
        # Analyze trends in recent history
        recent_loads = [entry["load_score"] for entry in list(self.load_history)[-20:]]
        
        # Simple trend analysis
        if len(recent_loads) >= 5:
            early_avg = sum(recent_loads[:5]) / 5
            late_avg = sum(recent_loads[-5:]) / 5
            trend = late_avg - early_avg
        else:
            trend = 0
        
        current_load = recent_loads[-1] if recent_loads else 0
        
        prediction = {
            "current_load": current_load,
            "trend": trend,
            "current_interval": self.current_check_interval,
        }
        
        if trend > 0.1:
            prediction["recommendation"] = "load_increasing"
            prediction["suggested_action"] = "prepare_for_scale_up"
        elif trend < -0.1:
            prediction["recommendation"] = "load_decreasing"
            prediction["suggested_action"] = "consider_scale_down"
        else:
            prediction["recommendation"] = "load_stable"
            prediction["suggested_action"] = "maintain_current_settings"
        
        return prediction
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling history for the past N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            entry for entry in self.load_history
            if entry["timestamp"] > cutoff_time
        ]


class ResourceMonitor:
    """Monitors system resources and suggests optimizations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_free = disk.free / (1024**3)  # GB
        
        # Network stats (basic)
        network = psutil.net_io_counters()
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "percent": memory_percent,
                "available_gb": memory_available,
                "total_gb": memory.total / (1024**3)
            },
            "disk": {
                "percent": disk_percent,
                "free_gb": disk_free,
                "total_gb": disk.total / (1024**3)
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_resource_constraints(self, usage: Dict[str, Any]) -> List[str]:
        """Analyze resource usage and identify constraints."""
        constraints = []
        
        # CPU constraints
        if usage["cpu"]["percent"] > 80:
            constraints.append(f"High CPU usage: {usage['cpu']['percent']:.1f}%")
        
        # Memory constraints
        if usage["memory"]["percent"] > 80:
            constraints.append(f"High memory usage: {usage['memory']['percent']:.1f}%")
        elif usage["memory"]["available_gb"] < 1.0:
            constraints.append(f"Low available memory: {usage['memory']['available_gb']:.1f}GB")
        
        # Disk constraints
        if usage["disk"]["percent"] > 90:
            constraints.append(f"High disk usage: {usage['disk']['percent']:.1f}%")
        elif usage["disk"]["free_gb"] < 5.0:
            constraints.append(f"Low disk space: {usage['disk']['free_gb']:.1f}GB free")
        
        return constraints
    
    def get_scaling_recommendations(self, 
                                  usage: Dict[str, Any],
                                  performance_metrics: PerformanceMetrics) -> List[str]:
        """Get recommendations for scaling based on resource usage and performance."""
        recommendations = []
        
        # CPU-based recommendations
        if usage["cpu"]["percent"] > 70:
            recommendations.append("Consider reducing concurrent operations due to high CPU usage")
        elif usage["cpu"]["percent"] < 30:
            recommendations.append("CPU usage is low - can increase concurrent operations")
        
        # Memory-based recommendations
        if usage["memory"]["percent"] > 75:
            recommendations.append("High memory usage - consider reducing cache size or batch sizes")
        
        # Performance-based recommendations
        avg_check_time = performance_metrics.get_average_duration("pipeline_check")
        if avg_check_time and avg_check_time > 10.0:
            recommendations.append("Slow pipeline checks - consider optimizing or caching")
        
        cache_hit_rate = performance_metrics.get_cache_hit_rate()
        if cache_hit_rate and cache_hit_rate < 0.6:
            recommendations.append("Low cache hit rate - consider increasing cache TTL")
        
        return recommendations