"""
Advanced Research Monitoring and Analytics
==========================================

Comprehensive monitoring system for research workflows, performance analytics,
and predictive maintenance of research infrastructure.

Features:
- Real-time performance monitoring
- Research workflow analytics
- Anomaly detection in research patterns
- Predictive maintenance and optimization
- Quality assurance automation
"""

import asyncio
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    USAGE = "usage"
    ERROR = "error"
    SECURITY = "security"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Individual metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    actual_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricCollector:
    """Collects and aggregates metrics from various sources."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.collectors: Dict[str, Callable] = {}
        self.collection_intervals: Dict[str, int] = {}
        self.running = False
        self.collection_thread: Optional[threading.Thread] = None
    
    def register_collector(self, name: str, collector_func: Callable, interval_seconds: int = 60):
        """Register a metric collector function."""
        self.collectors[name] = collector_func
        self.collection_intervals[name] = interval_seconds
        logger.info(f"Registered collector '{name}' with {interval_seconds}s interval")
    
    def add_metric(self, metric: MetricData):
        """Add a metric data point."""
        self.metrics[metric.name].append(metric)
        logger.debug(f"Added metric {metric.name}: {metric.value}")
    
    def get_metrics(self, name: str, since: Optional[datetime] = None, 
                   limit: Optional[int] = None) -> List[MetricData]:
        """Get metrics by name with optional filtering."""
        metric_queue = self.metrics.get(name, deque())
        
        # Convert to list for easier manipulation
        metrics = list(metric_queue)
        
        # Apply time filter
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        # Apply limit
        if limit:
            metrics = metrics[-limit:]
        
        return metrics
    
    def get_aggregated_metrics(self, name: str, period_minutes: int = 60) -> Dict[str, float]:
        """Get aggregated metrics for a time period."""
        since = datetime.now() - timedelta(minutes=period_minutes)
        metrics = self.get_metrics(name, since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest': values[-1] if values else 0.0
        }
    
    def start_collection(self):
        """Start automated metric collection."""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("Metric collection started")
    
    def stop_collection(self):
        """Stop automated metric collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metric collection stopped")
    
    def _collection_loop(self):
        """Main collection loop running in background thread."""
        last_collection = defaultdict(float)
        
        while self.running:
            current_time = time.time()
            
            for name, collector_func in self.collectors.items():
                interval = self.collection_intervals[name]
                
                if current_time - last_collection[name] >= interval:
                    try:
                        # Call collector function
                        result = collector_func()
                        
                        if isinstance(result, MetricData):
                            self.add_metric(result)
                        elif isinstance(result, list):
                            for metric in result:
                                if isinstance(metric, MetricData):
                                    self.add_metric(metric)
                        
                        last_collection[name] = current_time
                        
                    except Exception as e:
                        logger.error(f"Error in collector '{name}': {e}")
            
            time.sleep(1)  # Check every second


class AnomalyDetector:
    """Detects anomalies in research workflow patterns."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baseline_models: Dict[str, Dict[str, float]] = {}
    
    def build_baseline(self, metric_name: str, metrics: List[MetricData]):
        """Build baseline model for a metric."""
        if len(metrics) < 10:
            logger.warning(f"Insufficient data to build baseline for {metric_name}")
            return
        
        values = [m.value for m in metrics]
        
        self.baseline_models[metric_name] = {
            'mean': statistics.mean(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.1,
            'min': min(values),
            'max': max(values),
            'sample_size': len(values),
            'last_updated': datetime.now().isoformat()
        }
        
        logger.info(f"Built baseline for {metric_name} with {len(values)} samples")
    
    def detect_anomaly(self, metric: MetricData) -> Optional[Dict[str, Any]]:
        """Detect if a metric value is anomalous."""
        baseline = self.baseline_models.get(metric.name)
        
        if not baseline:
            return None
        
        mean = baseline['mean']
        std_dev = baseline['std_dev']
        
        # Calculate z-score
        if std_dev == 0:
            z_score = 0
        else:
            z_score = abs(metric.value - mean) / std_dev
        
        # Check if anomalous
        if z_score > self.sensitivity:
            return {
                'metric_name': metric.name,
                'value': metric.value,
                'expected_range': [mean - self.sensitivity * std_dev, 
                                 mean + self.sensitivity * std_dev],
                'z_score': z_score,
                'severity': 'high' if z_score > 3.0 else 'medium',
                'timestamp': metric.timestamp.isoformat()
            }
        
        return None
    
    def update_baseline(self, metric_name: str, new_metrics: List[MetricData]):
        """Update baseline model with new data."""
        if metric_name not in self.baseline_models:
            self.build_baseline(metric_name, new_metrics)
            return
        
        # Exponential moving average update
        values = [m.value for m in new_metrics]
        if not values:
            return
        
        current = self.baseline_models[metric_name]
        new_mean = statistics.mean(values)
        
        # Update with exponential smoothing (alpha = 0.1)
        alpha = 0.1
        current['mean'] = alpha * new_mean + (1 - alpha) * current['mean']
        
        # Update standard deviation
        new_std = statistics.stdev(values) if len(values) > 1 else current['std_dev']
        current['std_dev'] = alpha * new_std + (1 - alpha) * current['std_dev']
        
        current['last_updated'] = datetime.now().isoformat()


class AlertManager:
    """Manages alerts and notifications for research monitoring."""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.notification_handlers: List[Callable] = []
    
    def set_threshold(self, metric_name: str, level: AlertLevel, threshold: float):
        """Set alert threshold for a metric."""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        
        self.thresholds[metric_name][level.value] = threshold
        logger.info(f"Set {level.value} threshold for {metric_name}: {threshold}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler function."""
        self.notification_handlers.append(handler)
    
    def check_thresholds(self, metric: MetricData) -> List[Alert]:
        """Check metric against thresholds and generate alerts."""
        alerts = []
        metric_thresholds = self.thresholds.get(metric.name, {})
        
        for level_name, threshold in metric_thresholds.items():
            level = AlertLevel(level_name)
            
            # Check if threshold is breached
            breached = False
            if level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                # For error levels, trigger if value exceeds threshold
                breached = metric.value > threshold
            else:
                # For info/warning, could be bidirectional
                breached = abs(metric.value) > threshold
            
            if breached:
                alert_id = f"{metric.name}_{level.value}_{int(time.time())}"
                
                alert = Alert(
                    id=alert_id,
                    level=level,
                    message=f"{metric.name} {level.value}: {metric.value} exceeds threshold {threshold}",
                    metric_name=metric.name,
                    threshold=threshold,
                    actual_value=metric.value,
                    timestamp=datetime.now()
                )
                
                alerts.append(alert)
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Send notifications
                for handler in self.notification_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Notification handler failed: {e}")
        
        return alerts
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            del self.active_alerts[alert_id]
            logger.info(f"Resolved alert: {alert_id}")
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by level."""
        alerts = list(self.active_alerts.values())
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_by_level = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_level[alert.level.value] += 1
        
        total_alerts = len(self.alert_history)
        resolved_alerts = len([a for a in self.alert_history if a.resolved])
        
        return {
            'active_alerts': dict(active_by_level),
            'total_active': len(self.active_alerts),
            'total_historical': total_alerts,
            'resolution_rate': resolved_alerts / total_alerts if total_alerts > 0 else 0.0,
            'last_updated': datetime.now().isoformat()
        }


class ResearchWorkflowAnalyzer:
    """Analyzes research workflow patterns and efficiency."""
    
    def __init__(self):
        self.workflow_data: List[Dict[str, Any]] = []
        self.performance_models: Dict[str, Any] = {}
    
    def record_workflow_event(self, event_type: str, duration: float, 
                            metadata: Dict[str, Any] = None):
        """Record a workflow event."""
        event = {
            'type': event_type,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.workflow_data.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.workflow_data) > 1000:
            self.workflow_data = self.workflow_data[-1000:]
    
    def analyze_workflow_efficiency(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """Analyze workflow efficiency over a time period."""
        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
        
        # Filter recent events
        recent_events = [
            event for event in self.workflow_data
            if datetime.fromisoformat(event['timestamp']) >= cutoff_time
        ]
        
        if not recent_events:
            return {'error': 'No workflow data in specified time period'}
        
        # Group by event type
        events_by_type = defaultdict(list)
        for event in recent_events:
            events_by_type[event['type']].append(event['duration'])
        
        # Calculate efficiency metrics
        analysis = {
            'time_period_hours': time_period_hours,
            'total_events': len(recent_events),
            'event_types': {},
            'overall_efficiency': 0.0,
            'bottlenecks': [],
            'recommendations': []
        }
        
        total_duration = 0
        max_avg_duration = 0
        slowest_process = None
        
        for event_type, durations in events_by_type.items():
            avg_duration = statistics.mean(durations)
            total_duration += sum(durations)
            
            analysis['event_types'][event_type] = {
                'count': len(durations),
                'avg_duration': avg_duration,
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations),
                'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0.0
            }
            
            # Track slowest process
            if avg_duration > max_avg_duration:
                max_avg_duration = avg_duration
                slowest_process = event_type
        
        # Calculate overall efficiency (events per hour)
        analysis['overall_efficiency'] = len(recent_events) / time_period_hours
        
        # Identify bottlenecks (processes taking >2x average time)
        avg_duration_all = total_duration / len(recent_events)
        for event_type, stats in analysis['event_types'].items():
            if stats['avg_duration'] > 2 * avg_duration_all:
                analysis['bottlenecks'].append({
                    'process': event_type,
                    'avg_duration': stats['avg_duration'],
                    'expected_duration': avg_duration_all,
                    'slowdown_factor': stats['avg_duration'] / avg_duration_all
                })
        
        # Generate recommendations
        if slowest_process:
            analysis['recommendations'].append(
                f"Focus optimization efforts on '{slowest_process}' process"
            )
        
        if len(analysis['bottlenecks']) > 0:
            analysis['recommendations'].append(
                f"Address {len(analysis['bottlenecks'])} identified bottlenecks"
            )
        
        if analysis['overall_efficiency'] < 1.0:  # Less than 1 event per hour
            analysis['recommendations'].append(
                "Consider automation to improve workflow efficiency"
            )
        
        return analysis
    
    def predict_completion_time(self, workflow_type: str) -> Dict[str, Any]:
        """Predict completion time for a workflow type."""
        # Get historical data for this workflow type
        historical_durations = [
            event['duration'] for event in self.workflow_data
            if event['type'] == workflow_type
        ]
        
        if len(historical_durations) < 5:
            return {'error': f'Insufficient data for {workflow_type}'}
        
        # Simple prediction based on historical average with confidence interval
        mean_duration = statistics.mean(historical_durations)
        std_dev = statistics.stdev(historical_durations)
        
        # 95% confidence interval
        confidence_interval = 1.96 * std_dev / (len(historical_durations) ** 0.5)
        
        return {
            'workflow_type': workflow_type,
            'predicted_duration': mean_duration,
            'confidence_interval': confidence_interval,
            'range': [
                max(0, mean_duration - confidence_interval),
                mean_duration + confidence_interval
            ],
            'based_on_samples': len(historical_durations),
            'prediction_confidence': min(len(historical_durations) / 20, 1.0)  # Max confidence at 20+ samples
        }


class AdvancedResearchMonitor:
    """Main monitoring system coordinating all components."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("monitoring_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metric_collector = MetricCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.workflow_analyzer = ResearchWorkflowAnalyzer()
        
        self.monitoring_active = False
        self._setup_default_collectors()
        self._setup_default_thresholds()
        self._setup_notification_handlers()
    
    def _setup_default_collectors(self):
        """Setup default metric collectors."""
        
        def system_performance_collector() -> List[MetricData]:
            """Collect system performance metrics."""
            import psutil
            
            metrics = []
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricData(
                name="system.cpu_usage",
                value=cpu_percent,
                metric_type=MetricType.PERFORMANCE,
                timestamp=datetime.now()
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append(MetricData(
                name="system.memory_usage",
                value=memory.percent,
                metric_type=MetricType.PERFORMANCE,
                timestamp=datetime.now()
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.append(MetricData(
                name="system.disk_usage",
                value=disk.percent,
                metric_type=MetricType.PERFORMANCE,
                timestamp=datetime.now()
            ))
            
            return metrics
        
        def research_activity_collector() -> MetricData:
            """Collect research activity metrics."""
            # This would be connected to actual research notebook usage
            # For now, return a placeholder metric
            return MetricData(
                name="research.activity_level",
                value=1.0,  # Placeholder
                metric_type=MetricType.USAGE,
                timestamp=datetime.now(),
                metadata={"source": "notebook_usage"}
            )
        
        try:
            import psutil
            self.metric_collector.register_collector("system_performance", system_performance_collector, 30)
        except ImportError:
            logger.warning("psutil not available - system monitoring disabled")
        
        self.metric_collector.register_collector("research_activity", research_activity_collector, 60)
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds."""
        self.alert_manager.set_threshold("system.cpu_usage", AlertLevel.WARNING, 80.0)
        self.alert_manager.set_threshold("system.cpu_usage", AlertLevel.CRITICAL, 95.0)
        
        self.alert_manager.set_threshold("system.memory_usage", AlertLevel.WARNING, 85.0)
        self.alert_manager.set_threshold("system.memory_usage", AlertLevel.CRITICAL, 95.0)
        
        self.alert_manager.set_threshold("system.disk_usage", AlertLevel.WARNING, 90.0)
        self.alert_manager.set_threshold("system.disk_usage", AlertLevel.CRITICAL, 98.0)
    
    def _setup_notification_handlers(self):
        """Setup notification handlers."""
        
        def log_alert_handler(alert: Alert):
            """Log alert to file."""
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.ERROR: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL
            }
            
            logger.log(log_level[alert.level], f"ALERT: {alert.message}")
        
        self.alert_manager.add_notification_handler(log_alert_handler)
    
    def start_monitoring(self):
        """Start comprehensive monitoring."""
        self.monitoring_active = True
        self.metric_collector.start_collection()
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        logger.info("Advanced research monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        self.metric_collector.stop_collection()
        logger.info("Advanced research monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Process recent metrics for anomalies and alerts
                for metric_name in self.metric_collector.metrics:
                    recent_metrics = self.metric_collector.get_metrics(
                        metric_name, 
                        since=datetime.now() - timedelta(minutes=5)
                    )
                    
                    for metric in recent_metrics:
                        # Check for anomalies
                        anomaly = self.anomaly_detector.detect_anomaly(metric)
                        if anomaly:
                            logger.warning(f"Anomaly detected: {anomaly}")
                        
                        # Check thresholds
                        alerts = self.alert_manager.check_thresholds(metric)
                        for alert in alerts:
                            logger.warning(f"Alert triggered: {alert.message}")
                    
                    # Update anomaly baselines
                    if len(recent_metrics) >= 10:
                        self.anomaly_detector.update_baseline(metric_name, recent_metrics)
                
                # Save monitoring state
                await self._save_monitoring_state()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _save_monitoring_state(self):
        """Save current monitoring state to disk."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'active_metrics': list(self.metric_collector.metrics.keys()),
            'anomaly_baselines': self.anomaly_detector.baseline_models
        }
        
        state_file = self.storage_path / "monitoring_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        dashboard = {
            'system_status': 'healthy',
            'last_updated': datetime.now().isoformat(),
            'alerts': self.alert_manager.get_alert_summary(),
            'metrics': {},
            'workflow_analysis': self.workflow_analyzer.analyze_workflow_efficiency(),
            'recommendations': []
        }
        
        # Get aggregated metrics for dashboard
        for metric_name in self.metric_collector.metrics:
            dashboard['metrics'][metric_name] = self.metric_collector.get_aggregated_metrics(metric_name)
        
        # Determine overall system status
        critical_alerts = self.alert_manager.get_active_alerts(AlertLevel.CRITICAL)
        error_alerts = self.alert_manager.get_active_alerts(AlertLevel.ERROR)
        
        if critical_alerts:
            dashboard['system_status'] = 'critical'
        elif error_alerts:
            dashboard['system_status'] = 'degraded'
        elif self.alert_manager.get_active_alerts(AlertLevel.WARNING):
            dashboard['system_status'] = 'warning'
        
        # Generate recommendations
        if critical_alerts or error_alerts:
            dashboard['recommendations'].append("Immediate attention required for system alerts")
        
        if dashboard['workflow_analysis'].get('overall_efficiency', 0) < 0.5:
            dashboard['recommendations'].append("Consider workflow optimization")
        
        return dashboard
    
    def record_research_event(self, event_type: str, duration: float, metadata: Dict[str, Any] = None):
        """Record a research workflow event."""
        self.workflow_analyzer.record_workflow_event(event_type, duration, metadata)
        
        # Also create a metric
        metric = MetricData(
            name=f"workflow.{event_type}",
            value=duration,
            metric_type=MetricType.USAGE,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.metric_collector.add_metric(metric)


# Example usage and testing
if __name__ == "__main__":
    async def test_monitoring_system():
        """Test the monitoring system."""
        print("🔍 Testing Advanced Research Monitoring System")
        
        monitor = AdvancedResearchMonitor()
        monitor.start_monitoring()
        
        # Simulate some research events
        monitor.record_research_event("literature_search", 45.2, {"papers_found": 12})
        monitor.record_research_event("data_analysis", 120.7, {"dataset_size": 1000})
        monitor.record_research_event("writing", 89.3, {"words_written": 500})
        
        # Wait a bit for metrics to be collected
        await asyncio.sleep(2)
        
        # Get dashboard
        dashboard = monitor.get_monitoring_dashboard()
        
        print(f"System Status: {dashboard['system_status']}")
        print(f"Active Alerts: {dashboard['alerts']['total_active']}")
        print(f"Workflow Efficiency: {dashboard['workflow_analysis'].get('overall_efficiency', 'N/A')}")
        print(f"Recommendations: {len(dashboard['recommendations'])}")
        
        monitor.stop_monitoring()
        return dashboard
    
    # Run test
    result = asyncio.run(test_monitoring_system())
    print("✅ Monitoring system test completed")