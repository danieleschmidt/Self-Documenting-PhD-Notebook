"""
Auto-scaling and dynamic resource management for the PhD notebook system.
Implements intelligent scaling based on workload patterns and resource utilization.
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import statistics
import json

from ..monitoring.metrics import MetricsCollector
from ..utils.exceptions import ScalingError, ResourceError


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    COMPUTE_WORKERS = "compute_workers"
    AI_AGENTS = "ai_agents"
    CACHE_SIZE = "cache_size"
    CONNECTION_POOL = "connection_pool"
    MEMORY_BUFFER = "memory_buffer"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_length: int = 0
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    name: str
    resource_type: ResourceType
    metric_name: str
    threshold_up: float
    threshold_down: float
    scale_up_by: int
    scale_down_by: int
    cooldown_minutes: int = 5
    min_instances: int = 1
    max_instances: int = 100
    enabled: bool = True
    last_action_time: Optional[datetime] = None


@dataclass
class ScalingAction:
    """Represents a scaling action."""
    action_id: str
    resource_type: ResourceType
    direction: ScalingDirection
    from_count: int
    to_count: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: Optional[float] = None
    success: bool = False
    error: Optional[str] = None


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, history_window_hours: int = 24):
        self.history_window_hours = history_window_hours
        self.metrics_history: List[ScalingMetrics] = []
        self.logger = logging.getLogger("predictive_scaler")
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to history."""
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=self.history_window_hours)
        self.metrics_history = [
            m for m in self.metrics_history if m.timestamp > cutoff_time
        ]
    
    def predict_load(self, minutes_ahead: int = 15) -> Dict[str, float]:
        """Predict future load based on historical patterns."""
        if len(self.metrics_history) < 10:
            return {"cpu_utilization": 50.0, "memory_utilization": 50.0, "queue_length": 0}
        
        # Simple time-series prediction using moving averages and trends
        recent_metrics = self.metrics_history[-60:]  # Last hour of data points
        
        # Calculate trends
        cpu_values = [m.cpu_utilization for m in recent_metrics]
        memory_values = [m.memory_utilization for m in recent_metrics]
        queue_values = [m.queue_length for m in recent_metrics]
        
        # Linear trend calculation
        def calculate_trend(values: List[float]) -> float:
            if len(values) < 2:
                return 0.0
            
            n = len(values)
            x_mean = (n - 1) / 2
            y_mean = statistics.mean(values)
            
            numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        
        cpu_trend = calculate_trend(cpu_values)
        memory_trend = calculate_trend(memory_values)
        queue_trend = calculate_trend(queue_values)
        
        # Project forward
        current_cpu = cpu_values[-1] if cpu_values else 50.0
        current_memory = memory_values[-1] if memory_values else 50.0
        current_queue = queue_values[-1] if queue_values else 0
        
        predicted_cpu = max(0, min(100, current_cpu + cpu_trend * minutes_ahead))
        predicted_memory = max(0, min(100, current_memory + memory_trend * minutes_ahead))
        predicted_queue = max(0, current_queue + queue_trend * minutes_ahead)
        
        return {
            "cpu_utilization": predicted_cpu,
            "memory_utilization": predicted_memory,
            "queue_length": predicted_queue
        }
    
    def detect_patterns(self) -> Dict[str, Any]:
        """Detect load patterns in historical data."""
        if len(self.metrics_history) < 100:
            return {"patterns": [], "confidence": 0.0}
        
        # Group by hour of day
        hourly_loads: Dict[int, List[float]] = {}
        for metrics in self.metrics_history:
            hour = metrics.timestamp.hour
            if hour not in hourly_loads:
                hourly_loads[hour] = []
            hourly_loads[hour].append(metrics.cpu_utilization)
        
        # Calculate average load by hour
        hourly_averages = {}
        for hour, loads in hourly_loads.items():
            if loads:
                hourly_averages[hour] = statistics.mean(loads)
        
        # Find peak and low hours
        if hourly_averages:
            peak_hour = max(hourly_averages, key=hourly_averages.get)
            low_hour = min(hourly_averages, key=hourly_averages.get)
            
            patterns = [
                {
                    "type": "daily_peak",
                    "hour": peak_hour,
                    "average_load": hourly_averages[peak_hour]
                },
                {
                    "type": "daily_low",
                    "hour": low_hour,
                    "average_load": hourly_averages[low_hour]
                }
            ]
            
            # Calculate confidence based on data consistency
            load_variance = statistics.variance(hourly_averages.values()) if len(hourly_averages) > 1 else 0
            confidence = min(1.0, len(self.metrics_history) / 1000) * (1 - min(1.0, load_variance / 100))
            
            return {"patterns": patterns, "confidence": confidence}
        
        return {"patterns": [], "confidence": 0.0}


class ResourceManager:
    """Manages scalable resources."""
    
    def __init__(self):
        self.resources: Dict[ResourceType, Dict[str, Any]] = {}
        self.scaling_handlers: Dict[ResourceType, Callable] = {}
        self.logger = logging.getLogger("resource_manager")
    
    def register_resource(
        self,
        resource_type: ResourceType,
        current_count: int,
        scaling_handler: Callable[[ResourceType, int, int], bool]
    ):
        """Register a scalable resource."""
        self.resources[resource_type] = {
            "current_count": current_count,
            "target_count": current_count,
            "last_scaled": datetime.now()
        }
        self.scaling_handlers[resource_type] = scaling_handler
        self.logger.info(f"Registered resource {resource_type.value} with {current_count} instances")
    
    async def scale_resource(
        self,
        resource_type: ResourceType,
        target_count: int,
        reason: str = "Manual scaling"
    ) -> ScalingAction:
        """Scale a resource to target count."""
        if resource_type not in self.resources:
            raise ScalingError(f"Resource type {resource_type.value} not registered")
        
        current_count = self.resources[resource_type]["current_count"]
        action_id = f"{resource_type.value}_{int(time.time())}"
        
        if target_count == current_count:
            direction = ScalingDirection.MAINTAIN
        elif target_count > current_count:
            direction = ScalingDirection.SCALE_UP
        else:
            direction = ScalingDirection.SCALE_DOWN
        
        action = ScalingAction(
            action_id=action_id,
            resource_type=resource_type,
            direction=direction,
            from_count=current_count,
            to_count=target_count,
            reason=reason
        )
        
        if direction == ScalingDirection.MAINTAIN:
            action.success = True
            action.duration_seconds = 0.0
            return action
        
        start_time = time.time()
        
        try:
            # Execute scaling through registered handler
            handler = self.scaling_handlers[resource_type]
            success = await self._execute_scaling(handler, resource_type, current_count, target_count)
            
            if success:
                self.resources[resource_type]["current_count"] = target_count
                self.resources[resource_type]["target_count"] = target_count
                self.resources[resource_type]["last_scaled"] = datetime.now()
                action.success = True
                
                self.logger.info(
                    f"Successfully scaled {resource_type.value} from {current_count} to {target_count}"
                )
            else:
                action.error = "Scaling handler returned False"
                self.logger.error(f"Failed to scale {resource_type.value}")
        
        except Exception as e:
            action.error = str(e)
            self.logger.error(f"Error scaling {resource_type.value}: {e}")
        
        action.duration_seconds = time.time() - start_time
        return action
    
    async def _execute_scaling(
        self,
        handler: Callable,
        resource_type: ResourceType,
        current_count: int,
        target_count: int
    ) -> bool:
        """Execute scaling through handler."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(resource_type, current_count, target_count)
        else:
            return handler(resource_type, current_count, target_count)
    
    def get_resource_status(self) -> Dict[ResourceType, Dict[str, Any]]:
        """Get status of all resources."""
        return self.resources.copy()


class AutoScaler:
    """
    Automatic scaling system with predictive capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("auto_scaler")
        
        self.resource_manager = ResourceManager()
        self.predictive_scaler = PredictiveScaler()
        self.metrics_collector = MetricsCollector()
        
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: List[ScalingAction] = []
        
        self.is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        
        # Initialize default scaling rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        
        # CPU-based scaling for compute workers
        self.add_scaling_rule(ScalingRule(
            name="cpu_based_worker_scaling",
            resource_type=ResourceType.COMPUTE_WORKERS,
            metric_name="cpu_utilization",
            threshold_up=80.0,
            threshold_down=30.0,
            scale_up_by=2,
            scale_down_by=1,
            cooldown_minutes=5,
            min_instances=2,
            max_instances=20
        ))
        
        # Queue-based scaling for compute workers
        self.add_scaling_rule(ScalingRule(
            name="queue_based_worker_scaling",
            resource_type=ResourceType.COMPUTE_WORKERS,
            metric_name="queue_length",
            threshold_up=10.0,
            threshold_down=2.0,
            scale_up_by=1,
            scale_down_by=1,
            cooldown_minutes=3,
            min_instances=2,
            max_instances=20
        ))
        
        # Memory-based cache scaling
        self.add_scaling_rule(ScalingRule(
            name="memory_based_cache_scaling",
            resource_type=ResourceType.CACHE_SIZE,
            metric_name="memory_utilization",
            threshold_up=85.0,
            threshold_down=40.0,
            scale_up_by=1,
            scale_down_by=1,
            cooldown_minutes=10,
            min_instances=1,
            max_instances=5
        ))
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a new scaling rule."""
        self.scaling_rules[rule.name] = rule
        self.logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule."""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            self.logger.info(f"Removed scaling rule: {rule_name}")
    
    def register_resource(
        self,
        resource_type: ResourceType,
        current_count: int,
        scaling_handler: Callable[[ResourceType, int, int], bool]
    ):
        """Register a scalable resource."""
        self.resource_manager.register_resource(resource_type, current_count, scaling_handler)
    
    async def start(self):
        """Start the auto-scaling system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring and scaling loops
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        
        self.logger.info("Auto-scaling system started")
    
    async def stop(self):
        """Stop the auto-scaling system."""
        self.is_running = False
        
        # Cancel tasks
        for task in [self._monitoring_task, self._scaling_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Auto-scaling system stopped")
    
    async def _monitoring_loop(self):
        """Monitor system metrics for scaling decisions."""
        while self.is_running:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()
                
                # Add to predictive scaler
                self.predictive_scaler.add_metrics(metrics)
                
                # Emit metrics
                self._emit_metrics(metrics)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.is_running:
            try:
                # Get current metrics
                current_metrics = await self._collect_metrics()
                
                # Get predictions
                predictions = self.predictive_scaler.predict_load(minutes_ahead=15)
                
                # Evaluate scaling rules
                scaling_decisions = self._evaluate_scaling_rules(current_metrics, predictions)
                
                # Execute scaling decisions
                for decision in scaling_decisions:
                    await self._execute_scaling_decision(decision)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(120)
    
    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Get application-specific metrics (placeholder - would integrate with actual metrics)
        queue_length = 0  # Would get from actual queue
        response_time = 0.0  # Would get from actual measurements
        throughput = 0.0  # Would get from actual measurements
        error_rate = 0.0  # Would get from actual measurements
        
        return ScalingMetrics(
            cpu_utilization=cpu_percent,
            memory_utilization=memory.percent,
            queue_length=queue_length,
            response_time=response_time,
            throughput=throughput,
            error_rate=error_rate
        )
    
    def _emit_metrics(self, metrics: ScalingMetrics):
        """Emit metrics to monitoring system."""
        self.metrics_collector.record_gauge("autoscaler.cpu_utilization", metrics.cpu_utilization)
        self.metrics_collector.record_gauge("autoscaler.memory_utilization", metrics.memory_utilization)
        self.metrics_collector.record_gauge("autoscaler.queue_length", metrics.queue_length)
        self.metrics_collector.record_gauge("autoscaler.response_time", metrics.response_time)
        self.metrics_collector.record_gauge("autoscaler.throughput", metrics.throughput)
        self.metrics_collector.record_gauge("autoscaler.error_rate", metrics.error_rate)
    
    def _evaluate_scaling_rules(
        self,
        current_metrics: ScalingMetrics,
        predictions: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Evaluate all scaling rules and return scaling decisions."""
        decisions = []
        
        for rule_name, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if (rule.last_action_time and
                (datetime.now() - rule.last_action_time).total_seconds() < rule.cooldown_minutes * 60):
                continue
            
            # Get current value for the metric
            current_value = getattr(current_metrics, rule.metric_name, 0)
            predicted_value = predictions.get(rule.metric_name, current_value)
            
            # Use the higher of current and predicted values for scaling decisions
            decision_value = max(current_value, predicted_value)
            
            # Get current resource count
            resource_status = self.resource_manager.get_resource_status()
            if rule.resource_type not in resource_status:
                continue
            
            current_count = resource_status[rule.resource_type]["current_count"]
            
            # Determine scaling action
            if decision_value >= rule.threshold_up and current_count < rule.max_instances:
                target_count = min(rule.max_instances, current_count + rule.scale_up_by)
                decisions.append({
                    "rule": rule,
                    "action": ScalingDirection.SCALE_UP,
                    "target_count": target_count,
                    "reason": f"Metric {rule.metric_name} ({decision_value:.1f}) exceeded threshold ({rule.threshold_up})"
                })
            
            elif decision_value <= rule.threshold_down and current_count > rule.min_instances:
                target_count = max(rule.min_instances, current_count - rule.scale_down_by)
                decisions.append({
                    "rule": rule,
                    "action": ScalingDirection.SCALE_DOWN,
                    "target_count": target_count,
                    "reason": f"Metric {rule.metric_name} ({decision_value:.1f}) below threshold ({rule.threshold_down})"
                })
        
        return decisions
    
    async def _execute_scaling_decision(self, decision: Dict[str, Any]):
        """Execute a scaling decision."""
        rule = decision["rule"]
        target_count = decision["target_count"]
        reason = decision["reason"]
        
        try:
            action = await self.resource_manager.scale_resource(
                resource_type=rule.resource_type,
                target_count=target_count,
                reason=reason
            )
            
            # Update rule timestamp
            rule.last_action_time = datetime.now()
            
            # Record action
            self.scaling_history.append(action)
            
            # Keep only recent history
            cutoff_time = datetime.now() - timedelta(days=7)
            self.scaling_history = [
                a for a in self.scaling_history if a.timestamp > cutoff_time
            ]
            
            # Emit metrics
            self.metrics_collector.record_counter(f"autoscaler.actions.{action.direction.value}")
            
            if action.success:
                self.logger.info(f"Scaling decision executed: {reason}")
            else:
                self.logger.error(f"Scaling decision failed: {action.error}")
        
        except Exception as e:
            self.logger.error(f"Error executing scaling decision: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status."""
        resource_status = self.resource_manager.get_resource_status()
        
        # Get recent actions
        recent_actions = [
            {
                "action_id": action.action_id,
                "resource_type": action.resource_type.value,
                "direction": action.direction.value,
                "from_count": action.from_count,
                "to_count": action.to_count,
                "reason": action.reason,
                "timestamp": action.timestamp.isoformat(),
                "success": action.success,
                "duration_seconds": action.duration_seconds
            }
            for action in self.scaling_history[-10:]  # Last 10 actions
        ]
        
        # Get pattern analysis
        patterns = self.predictive_scaler.detect_patterns()
        
        return {
            "is_running": self.is_running,
            "resource_status": {rt.value: status for rt, status in resource_status.items()},
            "active_rules": len([r for r in self.scaling_rules.values() if r.enabled]),
            "recent_actions": recent_actions,
            "load_patterns": patterns,
            "metrics_history_length": len(self.predictive_scaler.metrics_history)
        }


# Global auto-scaler instance
auto_scaler = AutoScaler()


# Example scaling handlers
async def scale_compute_workers(resource_type: ResourceType, current_count: int, target_count: int) -> bool:
    """Example scaling handler for compute workers."""
    # This would integrate with your actual compute worker management
    logging.getLogger("scaling_handler").info(
        f"Scaling {resource_type.value} from {current_count} to {target_count}"
    )
    # Simulate scaling delay
    await asyncio.sleep(2)
    return True


async def scale_cache_size(resource_type: ResourceType, current_count: int, target_count: int) -> bool:
    """Example scaling handler for cache size."""
    # This would integrate with your actual cache management
    logging.getLogger("scaling_handler").info(
        f"Scaling {resource_type.value} from {current_count} to {target_count}"
    )
    return True