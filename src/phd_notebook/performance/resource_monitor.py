"""
Resource monitoring and performance tracking.
"""

import psutil
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import json


class ResourceMonitor:
    """Monitor system resources and performance metrics."""
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory': dict(psutil.virtual_memory()._asdict()),
            'disk': dict(psutil.disk_usage('/')._asdict()),
            'process': self._get_process_metrics()
        }
    
    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get historical metrics."""
        if limit:
            return list(self.metrics_history)[-limit:]
        return list(self.metrics_history)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.interval)
            except Exception:
                continue
    
    def _get_process_metrics(self) -> Dict[str, Any]:
        """Get current process metrics."""
        process = psutil.Process()
        return {
            'pid': process.pid,
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'threads': process.num_threads()
        }


class PerformanceTracker:
    """Track performance of functions and operations."""
    
    def __init__(self):
        self.metrics = {}
        
    def track_function(self, name: str, duration: float, success: bool = True) -> None:
        """Track function performance."""
        if name not in self.metrics:
            self.metrics[name] = {
                'total_calls': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
                'success_count': 0,
                'error_count': 0
            }
        
        stats = self.metrics[name]
        stats['total_calls'] += 1
        stats['total_duration'] += duration
        stats['avg_duration'] = stats['total_duration'] / stats['total_calls']
        
        if success:
            stats['success_count'] += 1
        else:
            stats['error_count'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.metrics.copy()


# Simplified implementations to avoid import issues
def optimize_notebook():
    """Placeholder optimization function."""
    return {"optimized": True}


class ProfiledFunction:
    """Placeholder profiled function decorator."""
    
    def __init__(self, func):
        self.func = func
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)