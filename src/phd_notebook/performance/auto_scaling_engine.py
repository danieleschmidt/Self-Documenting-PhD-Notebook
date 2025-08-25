"""
Auto-Scaling Engine for Research Platform

Intelligent auto-scaling system that dynamically adjusts system resources
based on workload patterns, predictive analytics, and performance metrics.
"""

import asyncio
import json
import logging
import math
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import uuid
import os

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    STORAGE_GB = "storage_gb"
    NETWORK_BANDWIDTH = "network_bandwidth"
    CONCURRENT_TASKS = "concurrent_tasks"
    CACHE_SIZE = "cache_size"
    WORKER_PROCESSES = "worker_processes"


class ScalingDirection(Enum):
    """Direction of scaling operation."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ScalingStrategy(Enum):
    """Scaling strategies."""
    REACTIVE = "reactive"           # Scale based on current metrics
    PREDICTIVE = "predictive"       # Scale based on predicted load
    SCHEDULED = "scheduled"         # Scale based on schedule
    HYBRID = "hybrid"              # Combination of strategies


@dataclass
class ResourceLimits:
    """Resource scaling limits."""
    min_value: float
    max_value: float
    step_size: float
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period: int  # seconds


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    resource_type: ResourceType
    direction: ScalingDirection
    old_value: float
    new_value: float
    trigger_reason: str
    success: bool
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadPattern:
    """Identified workload pattern."""
    pattern_id: str
    name: str
    description: str
    time_periods: List[Tuple[int, int]]  # (start_hour, end_hour)
    expected_load_multiplier: float
    confidence_score: float
    resource_requirements: Dict[ResourceType, float]


class LoadPredictor:
    """Predicts future load based on historical patterns."""
    
    def __init__(self, history_window_hours: int = 168):  # 1 week
        self.history_window = history_window_hours * 3600  # Convert to seconds
        self.load_history = deque(maxlen=10000)
        self.patterns = []
        
    def record_load_point(self, timestamp: float, load_metrics: Dict[str, float]) -> None:
        """Record a load measurement point."""
        self.load_history.append({
            'timestamp': timestamp,
            'metrics': load_metrics.copy()
        })
        
    def analyze_patterns(self) -> List[WorkloadPattern]:
        """Analyze historical data to identify workload patterns."""
        if len(self.load_history) < 100:  # Need sufficient data
            return []
        
        patterns = []
        
        # Analyze hourly patterns
        hourly_loads = defaultdict(list)
        
        for point in self.load_history:
            hour = datetime.fromtimestamp(point['timestamp']).hour
            cpu_load = point['metrics'].get('cpu_usage', 0)
            memory_load = point['metrics'].get('memory_usage', 0)
            avg_load = (cpu_load + memory_load) / 2
            hourly_loads[hour].append(avg_load)
        
        # Calculate average load for each hour
        hourly_averages = {}
        for hour, loads in hourly_loads.items():
            hourly_averages[hour] = sum(loads) / len(loads)
        
        # Identify high-load periods
        overall_avg = sum(hourly_averages.values()) / len(hourly_averages)
        high_load_hours = [
            hour for hour, avg_load in hourly_averages.items() 
            if avg_load > overall_avg * 1.3
        ]
        
        if high_load_hours:
            # Group consecutive hours
            grouped_hours = self._group_consecutive_hours(high_load_hours)
            
            for group in grouped_hours:
                start_hour = min(group)
                end_hour = max(group)
                avg_load = sum(hourly_averages[h] for h in group) / len(group)
                
                pattern = WorkloadPattern(
                    pattern_id=str(uuid.uuid4()),
                    name=f"High Load Period {start_hour:02d}:00-{end_hour:02d}:00",
                    description=f"Increased workload from {start_hour}:00 to {end_hour}:00",
                    time_periods=[(start_hour, end_hour)],
                    expected_load_multiplier=avg_load / overall_avg,
                    confidence_score=min(len(group) / 4, 1.0),  # Higher confidence for longer periods
                    resource_requirements={
                        ResourceType.CPU_CORES: avg_load / overall_avg,
                        ResourceType.MEMORY_GB: avg_load / overall_avg * 0.8,
                        ResourceType.CONCURRENT_TASKS: avg_load / overall_avg * 1.2
                    }
                )
                patterns.append(pattern)
        
        self.patterns = patterns
        return patterns
    
    def _group_consecutive_hours(self, hours: List[int]) -> List[List[int]]:
        """Group consecutive hours together."""
        if not hours:
            return []
        
        hours = sorted(hours)
        groups = []
        current_group = [hours[0]]
        
        for i in range(1, len(hours)):
            if hours[i] == hours[i-1] + 1 or (hours[i-1] == 23 and hours[i] == 0):
                current_group.append(hours[i])
            else:
                groups.append(current_group)
                current_group = [hours[i]]
        
        groups.append(current_group)
        return groups
    
    def predict_load(self, future_timestamp: float) -> Dict[str, float]:
        """Predict load at a future timestamp."""
        future_dt = datetime.fromtimestamp(future_timestamp)
        future_hour = future_dt.hour
        
        # Base prediction on historical average
        base_load = {'cpu_usage': 0.3, 'memory_usage': 0.4, 'concurrent_tasks': 5}
        
        # Adjust based on patterns
        for pattern in self.patterns:
            for start_hour, end_hour in pattern.time_periods:
                if start_hour <= future_hour <= end_hour:
                    multiplier = pattern.expected_load_multiplier
                    for key in base_load:
                        base_load[key] *= multiplier
                    break
        
        return base_load


class ResourceScaler:
    """Manages scaling of individual resource types."""
    
    def __init__(self, resource_type: ResourceType, limits: ResourceLimits):
        self.resource_type = resource_type
        self.limits = limits
        self.current_value = limits.min_value
        self.last_scaling_time = 0
        self.scaling_history = deque(maxlen=1000)
        
    def can_scale(self, direction: ScalingDirection) -> bool:
        """Check if scaling in the given direction is possible."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.limits.cooldown_period:
            return False
        
        # Check limits
        if direction == ScalingDirection.SCALE_UP:
            return self.current_value < self.limits.max_value
        elif direction == ScalingDirection.SCALE_DOWN:
            return self.current_value > self.limits.min_value
        
        return False
    
    def calculate_target_value(self, current_load: float, predicted_load: float) -> float:
        """Calculate target value based on load metrics."""
        # Use the higher of current or predicted load
        target_load = max(current_load, predicted_load)
        
        # Add buffer for safety
        target_load *= 1.2
        
        # Calculate required capacity
        if self.resource_type == ResourceType.CPU_CORES:
            target_value = max(1, math.ceil(target_load * 4))  # 4 cores per load unit
        elif self.resource_type == ResourceType.MEMORY_GB:
            target_value = max(2, math.ceil(target_load * 8))  # 8GB per load unit
        elif self.resource_type == ResourceType.CONCURRENT_TASKS:
            target_value = max(2, math.ceil(target_load * 10))  # 10 tasks per load unit
        elif self.resource_type == ResourceType.CACHE_SIZE:
            target_value = max(100, math.ceil(target_load * 500))  # 500MB per load unit
        else:
            target_value = self.current_value
        
        # Clamp to limits
        target_value = max(self.limits.min_value, 
                          min(self.limits.max_value, target_value))
        
        return target_value
    
    def scale_to_target(self, target_value: float, reason: str) -> Optional[ScalingEvent]:
        """Scale to target value."""
        if target_value == self.current_value:
            return None
        
        start_time = time.time()
        old_value = self.current_value
        
        direction = (ScalingDirection.SCALE_UP 
                    if target_value > self.current_value 
                    else ScalingDirection.SCALE_DOWN)
        
        if not self.can_scale(direction):
            logger.warning(f"Cannot scale {self.resource_type.value} {direction.value}")
            return None
        
        try:
            # Perform the scaling operation
            success = self._perform_scaling(target_value)
            
            if success:
                self.current_value = target_value
                self.last_scaling_time = time.time()
            
            duration = time.time() - start_time
            
            event = ScalingEvent(
                timestamp=start_time,
                resource_type=self.resource_type,
                direction=direction,
                old_value=old_value,
                new_value=target_value if success else old_value,
                trigger_reason=reason,
                success=success,
                duration=duration,
                metadata={
                    'target_value': target_value,
                    'actual_change': target_value - old_value if success else 0
                }
            )
            
            self.scaling_history.append(event)
            
            if success:
                logger.info(
                    f"Scaled {self.resource_type.value} from {old_value} to {target_value} "
                    f"({direction.value})"
                )
            else:
                logger.error(f"Failed to scale {self.resource_type.value}")
            
            return event
            
        except Exception as e:
            logger.error(f"Scaling error for {self.resource_type.value}: {e}")
            return None
    
    def _perform_scaling(self, target_value: float) -> bool:
        """Perform the actual scaling operation."""
        # This is where actual scaling logic would be implemented
        # For now, we simulate successful scaling
        
        if self.resource_type == ResourceType.CPU_CORES:
            # Simulate CPU core scaling
            os.environ[f'RESEARCH_PLATFORM_CPU_CORES'] = str(int(target_value))
            
        elif self.resource_type == ResourceType.MEMORY_GB:
            # Simulate memory scaling
            os.environ[f'RESEARCH_PLATFORM_MEMORY_GB'] = str(int(target_value))
            
        elif self.resource_type == ResourceType.CONCURRENT_TASKS:
            # Simulate task concurrency scaling
            os.environ[f'RESEARCH_PLATFORM_MAX_TASKS'] = str(int(target_value))
            
        elif self.resource_type == ResourceType.CACHE_SIZE:
            # Simulate cache size scaling
            os.environ[f'RESEARCH_PLATFORM_CACHE_SIZE_MB'] = str(int(target_value))
        
        # Simulate scaling time
        time.sleep(0.1)
        
        return True


class AutoScalingEngine:
    """Main auto-scaling engine that coordinates all scaling activities."""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        self.load_predictor = LoadPredictor()
        self.scalers = {}
        self.is_running = False
        self._scaling_thread = None
        self.scaling_interval = 30  # seconds
        
        # Initialize scalers with default limits
        self._initialize_scalers()
        
    def _initialize_scalers(self) -> None:
        """Initialize resource scalers with default configurations."""
        scaler_configs = {
            ResourceType.CPU_CORES: ResourceLimits(
                min_value=1, max_value=16, step_size=1,
                scale_up_threshold=0.7, scale_down_threshold=0.3,
                cooldown_period=120
            ),
            ResourceType.MEMORY_GB: ResourceLimits(
                min_value=2, max_value=64, step_size=2,
                scale_up_threshold=0.8, scale_down_threshold=0.4,
                cooldown_period=180
            ),
            ResourceType.CONCURRENT_TASKS: ResourceLimits(
                min_value=2, max_value=100, step_size=2,
                scale_up_threshold=0.8, scale_down_threshold=0.3,
                cooldown_period=60
            ),
            ResourceType.CACHE_SIZE: ResourceLimits(
                min_value=100, max_value=5000, step_size=100,
                scale_up_threshold=0.9, scale_down_threshold=0.5,
                cooldown_period=300
            )
        }
        
        for resource_type, limits in scaler_configs.items():
            self.scalers[resource_type] = ResourceScaler(resource_type, limits)
    
    def start(self) -> None:
        """Start the auto-scaling engine."""
        if self.is_running:
            return
        
        self.is_running = True
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop, daemon=True
        )
        self._scaling_thread.start()
        
        logger.info("Auto-scaling engine started")
    
    def stop(self) -> None:
        """Stop the auto-scaling engine."""
        self.is_running = False
        
        if self._scaling_thread:
            self._scaling_thread.join(timeout=10)
        
        logger.info("Auto-scaling engine stopped")
    
    def _scaling_loop(self) -> None:
        """Main scaling loop."""
        while self.is_running:
            try:
                self._perform_scaling_evaluation()
                time.sleep(self.scaling_interval)
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(self.scaling_interval)
    
    def _perform_scaling_evaluation(self) -> None:
        """Evaluate current conditions and perform scaling if needed."""
        current_time = time.time()
        
        # Get current load metrics (simulated)
        current_load = self._get_current_load_metrics()
        
        # Record load point for prediction
        self.load_predictor.record_load_point(current_time, current_load)
        
        # Get predicted load
        future_time = current_time + (self.scaling_interval * 2)  # Look ahead
        predicted_load = self.load_predictor.predict_load(future_time)
        
        # Update patterns periodically
        if len(self.load_predictor.load_history) % 100 == 0:
            self.load_predictor.analyze_patterns()
        
        # Evaluate each resource type
        for resource_type, scaler in self.scalers.items():
            try:
                self._evaluate_resource_scaling(
                    scaler, current_load, predicted_load
                )
            except Exception as e:
                logger.error(f"Error scaling {resource_type.value}: {e}")
    
    def _get_current_load_metrics(self) -> Dict[str, float]:
        """Get current system load metrics."""
        # This would integrate with actual monitoring systems
        # For now, simulate realistic load patterns
        
        current_hour = datetime.now().hour
        base_load = 0.3
        
        # Simulate daily patterns
        if 9 <= current_hour <= 17:  # Business hours
            base_load = 0.6
        elif 18 <= current_hour <= 22:  # Evening
            base_load = 0.4
        else:  # Night/early morning
            base_load = 0.2
        
        # Add some randomness
        import random
        load_factor = base_load + random.uniform(-0.1, 0.1)
        load_factor = max(0.1, min(1.0, load_factor))
        
        return {
            'cpu_usage': load_factor,
            'memory_usage': load_factor * 0.8,
            'concurrent_tasks': load_factor * 10,
            'cache_hit_rate': 1.0 - (load_factor * 0.3)  # Higher load = lower hit rate
        }
    
    def _evaluate_resource_scaling(self, scaler: ResourceScaler, 
                                 current_load: Dict[str, float],
                                 predicted_load: Dict[str, float]) -> None:
        """Evaluate if a specific resource needs scaling."""
        resource_type = scaler.resource_type
        
        # Get relevant load metric
        if resource_type == ResourceType.CPU_CORES:
            current_metric = current_load.get('cpu_usage', 0)
            predicted_metric = predicted_load.get('cpu_usage', 0)
        elif resource_type == ResourceType.MEMORY_GB:
            current_metric = current_load.get('memory_usage', 0)
            predicted_metric = predicted_load.get('memory_usage', 0)
        elif resource_type == ResourceType.CONCURRENT_TASKS:
            current_metric = current_load.get('concurrent_tasks', 0) / 10  # Normalize
            predicted_metric = predicted_load.get('concurrent_tasks', 0) / 10
        elif resource_type == ResourceType.CACHE_SIZE:
            # Scale up cache if hit rate is low
            current_metric = 1.0 - current_load.get('cache_hit_rate', 0.8)
            predicted_metric = 1.0 - predicted_load.get('cache_hit_rate', 0.8)
        else:
            return
        
        # Determine if scaling is needed
        scale_up_needed = (current_metric > scaler.limits.scale_up_threshold or
                          predicted_metric > scaler.limits.scale_up_threshold)
        
        scale_down_needed = (current_metric < scaler.limits.scale_down_threshold and
                            predicted_metric < scaler.limits.scale_down_threshold)
        
        if scale_up_needed:
            target = scaler.calculate_target_value(current_metric, predicted_metric)
            reason = f"High load detected: current={current_metric:.2f}, predicted={predicted_metric:.2f}"
            scaler.scale_to_target(target, reason)
            
        elif scale_down_needed:
            target = scaler.calculate_target_value(current_metric, predicted_metric)
            reason = f"Low load detected: current={current_metric:.2f}, predicted={predicted_metric:.2f}"
            scaler.scale_to_target(target, reason)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and statistics."""
        status = {
            'timestamp': time.time(),
            'is_running': self.is_running,
            'strategy': self.strategy.value,
            'resources': {},
            'recent_events': [],
            'patterns_detected': len(self.load_predictor.patterns),
            'load_history_size': len(self.load_predictor.load_history)
        }
        
        # Resource status
        for resource_type, scaler in self.scalers.items():
            status['resources'][resource_type.value] = {
                'current_value': scaler.current_value,
                'min_value': scaler.limits.min_value,
                'max_value': scaler.limits.max_value,
                'last_scaling_time': scaler.last_scaling_time,
                'can_scale_up': scaler.can_scale(ScalingDirection.SCALE_UP),
                'can_scale_down': scaler.can_scale(ScalingDirection.SCALE_DOWN)
            }
        
        # Recent scaling events
        all_events = []
        for scaler in self.scalers.values():
            all_events.extend(list(scaler.scaling_history))
        
        # Sort by timestamp and take recent events
        all_events.sort(key=lambda x: x.timestamp, reverse=True)
        status['recent_events'] = [
            {
                'timestamp': event.timestamp,
                'resource': event.resource_type.value,
                'direction': event.direction.value,
                'old_value': event.old_value,
                'new_value': event.new_value,
                'reason': event.trigger_reason,
                'success': event.success
            }
            for event in all_events[:20]
        ]
        
        return status
    
    def force_scaling_evaluation(self) -> Dict[str, Any]:
        """Force an immediate scaling evaluation."""
        logger.info("Forcing scaling evaluation")
        
        self._perform_scaling_evaluation()
        
        return self.get_scaling_status()
    
    def set_resource_limits(self, resource_type: ResourceType, 
                           limits: ResourceLimits) -> None:
        """Update resource limits for a specific resource type."""
        if resource_type in self.scalers:
            self.scalers[resource_type].limits = limits
            logger.info(f"Updated limits for {resource_type.value}")
        else:
            # Create new scaler
            self.scalers[resource_type] = ResourceScaler(resource_type, limits)
            logger.info(f"Created new scaler for {resource_type.value}")


# Global auto-scaling engine instance
_global_scaling_engine = None

def get_scaling_engine() -> AutoScalingEngine:
    """Get the global auto-scaling engine instance."""
    global _global_scaling_engine
    if _global_scaling_engine is None:
        _global_scaling_engine = AutoScalingEngine()
    return _global_scaling_engine


def start_auto_scaling(strategy: ScalingStrategy = ScalingStrategy.HYBRID):
    """Start global auto-scaling."""
    engine = get_scaling_engine()
    engine.strategy = strategy
    engine.start()


def stop_auto_scaling():
    """Stop global auto-scaling."""
    engine = get_scaling_engine()
    engine.stop()


def get_scaling_status() -> Dict[str, Any]:
    """Get current auto-scaling status."""
    engine = get_scaling_engine()
    return engine.get_scaling_status()


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Start auto-scaling
    start_auto_scaling(ScalingStrategy.HYBRID)
    
    try:
        # Run for a while to demonstrate scaling
        time.sleep(60)
        
        # Get status
        status = get_scaling_status()
        print(json.dumps(status, indent=2, default=str))
        
    finally:
        stop_auto_scaling()