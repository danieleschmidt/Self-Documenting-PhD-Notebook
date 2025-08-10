"""
Metrics collection and monitoring for the PhD notebook system.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from pathlib import Path
from contextlib import contextmanager

from ..utils.exceptions import MetricsError


class MetricsCollector:
    """Collects and tracks system metrics."""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
    def increment(self, metric: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        self.counters[metric] += value
        self._record_metric(metric, value, 'counter', tags)
    
    def gauge(self, metric: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge metric."""
        self.gauges[metric] = value
        self._record_metric(metric, value, 'gauge', tags)
    
    def timing(self, metric: str, duration: float, tags: Dict[str, str] = None) -> None:
        """Record a timing metric."""
        self.timers[metric].append(duration)
        # Keep only last 100 timings per metric
        if len(self.timers[metric]) > 100:
            self.timers[metric] = self.timers[metric][-100:]
        
        self._record_metric(metric, duration, 'timing', tags)
    
    @contextmanager
    def timer(self, metric: str, tags: Dict[str, str] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.timing(metric, duration, tags)
    
    def _record_metric(self, name: str, value: Any, metric_type: str, tags: Dict[str, str] = None) -> None:
        """Record a metric with timestamp."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'value': value,
            'type': metric_type,
            'tags': tags or {}
        }
        self.metrics[name].append(record)
    
    def get_counter(self, metric: str) -> int:
        """Get counter value."""
        return self.counters[metric]
    
    def get_gauge(self, metric: str) -> float:
        """Get gauge value."""
        return self.gauges[metric]
    
    def get_timing_stats(self, metric: str) -> Dict[str, float]:
        """Get timing statistics for a metric."""
        timings = self.timers.get(metric, [])
        if not timings:
            return {}
        
        return {
            'count': len(timings),
            'min': min(timings),
            'max': max(timings),
            'avg': sum(timings) / len(timings),
            'total': sum(timings)
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'timing_stats': {
                name: self.get_timing_stats(name) 
                for name in self.timers.keys()
            },
            'total_metrics': len(self.metrics),
            'collection_time': datetime.now().isoformat()
        }
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export metrics in specified format."""
        if format_type == 'json':
            return json.dumps(self.get_metrics_summary(), indent=2)
        elif format_type == 'prometheus':
            return self._export_prometheus()
        else:
            raise MetricsError(f"Unsupported export format: {format_type}")
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for name, value in self.counters.items():
            lines.append(f"# TYPE {name}_total counter")
            lines.append(f"{name}_total {value}")
        
        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Timing histograms
        for name, timings in self.timers.items():
            if timings:
                stats = self.get_timing_stats(name)
                lines.append(f"# TYPE {name}_duration_seconds histogram")
                lines.append(f"{name}_duration_seconds_count {stats['count']}")
                lines.append(f"{name}_duration_seconds_sum {stats['total']}")
        
        return '\n'.join(lines)
    
    def save_to_file(self, file_path: Path) -> None:
        """Save metrics to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.get_metrics_summary(), f, indent=2)
        except Exception as e:
            raise MetricsError(f"Failed to save metrics to {file_path}: {e}")


class NotebookMetrics:
    """Specific metrics for notebook operations."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    def note_created(self, note_type: str) -> None:
        """Record note creation."""
        self.collector.increment('notes_created_total', tags={'type': note_type})
        self.collector.increment('notes_total')
    
    def note_updated(self, note_type: str) -> None:
        """Record note update."""
        self.collector.increment('notes_updated_total', tags={'type': note_type})
    
    def workflow_executed(self, workflow_name: str, duration: float, success: bool) -> None:
        """Record workflow execution."""
        self.collector.timing('workflow_duration_seconds', duration, 
                            tags={'workflow': workflow_name})
        
        status = 'success' if success else 'failure'
        self.collector.increment('workflows_executed_total', 
                                tags={'workflow': workflow_name, 'status': status})
    
    def ai_request(self, provider: str, model: str, tokens: int, duration: float) -> None:
        """Record AI API request."""
        self.collector.increment('ai_requests_total', 
                                tags={'provider': provider, 'model': model})
        self.collector.increment('ai_tokens_total', value=tokens,
                                tags={'provider': provider, 'model': model})
        self.collector.timing('ai_request_duration_seconds', duration,
                             tags={'provider': provider, 'model': model})
    
    def vault_operation(self, operation: str, success: bool, duration: float = None) -> None:
        """Record vault operation."""
        status = 'success' if success else 'failure'
        self.collector.increment('vault_operations_total', 
                                tags={'operation': operation, 'status': status})
        
        if duration is not None:
            self.collector.timing('vault_operation_duration_seconds', duration,
                                 tags={'operation': operation})
    
    def get_notebook_stats(self) -> Dict[str, Any]:
        """Get notebook-specific statistics."""
        return {
            'notes_created': self.collector.get_counter('notes_created_total'),
            'notes_total': self.collector.get_counter('notes_total'),
            'workflows_executed': self.collector.get_counter('workflows_executed_total'),
            'ai_requests': self.collector.get_counter('ai_requests_total'),
            'ai_tokens': self.collector.get_counter('ai_tokens_total'),
            'vault_operations': self.collector.get_counter('vault_operations_total')
        }


class HealthChecker:
    """Health checking for system components."""
    
    def __init__(self):
        self.health_checks: Dict[str, Dict[str, Any]] = {}
    
    def add_check(self, name: str, check_func, critical: bool = True) -> None:
        """Add a health check."""
        self.health_checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_check': None,
            'last_result': None
        }
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            'healthy': True,
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for name, check_info in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_info['func']()
                duration = time.time() - start_time
                
                check_result = {
                    'healthy': True,
                    'duration': duration,
                    'result': result,
                    'error': None
                }
                
                # Update check info
                check_info['last_check'] = datetime.now()
                check_info['last_result'] = check_result
                
            except Exception as e:
                check_result = {
                    'healthy': False,
                    'duration': time.time() - start_time if 'start_time' in locals() else 0,
                    'result': None,
                    'error': str(e)
                }
                
                # Mark overall health as unhealthy if this is a critical check
                if check_info['critical']:
                    results['healthy'] = False
            
            results['checks'][name] = check_result
        
        return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        health_results = self.run_checks()
        
        critical_failures = sum(
            1 for name, result in health_results['checks'].items()
            if not result['healthy'] and self.health_checks[name]['critical']
        )
        
        warning_failures = sum(
            1 for name, result in health_results['checks'].items()
            if not result['healthy'] and not self.health_checks[name]['critical']
        )
        
        return {
            'overall_healthy': health_results['healthy'],
            'total_checks': len(self.health_checks),
            'critical_failures': critical_failures,
            'warning_failures': warning_failures,
            'last_check': health_results['timestamp']
        }


class PerformanceProfiler:
    """Performance profiling for system operations."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.active_profiles: Dict[str, float] = {}
    
    def start_profile(self, operation: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation}_{int(time.time() * 1000)}"
        self.active_profiles[profile_id] = time.time()
        return profile_id
    
    def end_profile(self, profile_id: str, tags: Dict[str, str] = None) -> float:
        """End profiling and record metrics."""
        if profile_id not in self.active_profiles:
            raise MetricsError(f"Profile {profile_id} not found")
        
        duration = time.time() - self.active_profiles[profile_id]
        del self.active_profiles[profile_id]
        
        # Extract operation name from profile_id
        operation = profile_id.rsplit('_', 1)[0]
        
        self.collector.timing(f'{operation}_duration', duration, tags)
        
        # Record slow operations
        if duration > 5.0:  # 5 seconds threshold
            self.collector.increment('slow_operations_total', 
                                   tags={'operation': operation})
        
        return duration
    
    @contextmanager
    def profile(self, operation: str, tags: Dict[str, str] = None):
        """Context manager for profiling operations."""
        profile_id = self.start_profile(operation)
        try:
            yield profile_id
        finally:
            self.end_profile(profile_id, tags)


# Global metrics instances
_metrics_collector = MetricsCollector()
_notebook_metrics = NotebookMetrics(_metrics_collector)
_health_checker = HealthChecker()
_profiler = PerformanceProfiler(_metrics_collector)

def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics_collector

def get_notebook_metrics() -> NotebookMetrics:
    """Get notebook metrics."""
    return _notebook_metrics

def get_health_checker() -> HealthChecker:
    """Get health checker."""
    return _health_checker

def get_profiler() -> PerformanceProfiler:
    """Get performance profiler."""
    return _profiler