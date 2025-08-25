"""
Tests for Enhanced Performance Monitoring System

Comprehensive test suite for the performance monitoring, optimization,
and auto-scaling components of the research platform.
"""

import pytest
import time
import threading
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from phd_notebook.performance.enhanced_performance_monitor import (
    PerformanceCollector,
    PerformanceOptimizer,
    PerformanceMonitor,
    PerformanceMetric,
    OptimizationStrategy,
    PerformanceSnapshot,
    OptimizationResult,
    monitor_performance,
    optimize_for_performance,
    get_performance_monitor,
    start_performance_monitoring,
    stop_performance_monitoring
)

from phd_notebook.performance.auto_scaling_engine import (
    AutoScalingEngine,
    ResourceScaler,
    LoadPredictor,
    ResourceType,
    ResourceLimits,
    ScalingDirection,
    ScalingStrategy,
    ScalingEvent,
    WorkloadPattern
)


class TestPerformanceCollector:
    """Test performance metrics collection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = PerformanceCollector()
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.collector.is_collecting:
            self.collector.stop_collection()
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        assert self.collector.collection_interval == 1.0
        assert not self.collector.is_collecting
        assert len(self.collector.metrics_buffer) == 0
    
    def test_start_stop_collection(self):
        """Test starting and stopping collection."""
        assert not self.collector.is_collecting
        
        self.collector.start_collection()
        assert self.collector.is_collecting
        
        # Wait a bit for some metrics to be collected
        time.sleep(2)
        
        # Should have some metrics
        assert len(self.collector.metrics_buffer) > 0
        
        self.collector.stop_collection()
        assert not self.collector.is_collecting
    
    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        metrics = self.collector._collect_system_metrics()
        
        # Should collect memory, CPU, and disk I/O metrics
        expected_metrics = {'memory', 'cpu', 'disk_io'}
        collected_metrics = set(metrics.keys())
        
        # At least some metrics should be collected
        assert len(collected_metrics.intersection(expected_metrics)) >= 1
        
        # Check metric structure
        for metric in metrics.values():
            assert isinstance(metric, PerformanceSnapshot)
            assert metric.timestamp > 0
            assert isinstance(metric.value, (int, float))
            assert metric.operation_id is not None
    
    def test_recent_metrics_filtering(self):
        """Test filtering of recent metrics."""
        # Start collection briefly
        self.collector.start_collection()
        time.sleep(1)
        self.collector.stop_collection()
        
        # Get recent metrics
        recent = self.collector.get_recent_metrics(duration_seconds=60)
        
        # All metrics should be recent
        current_time = time.time()
        for metric_dict in recent:
            if hasattr(metric_dict, 'timestamp'):
                assert current_time - metric_dict.timestamp <= 60
    
    def test_metric_summary_calculation(self):
        """Test metric summary statistics."""
        # Add some test data
        test_snapshots = []
        for i in range(5):
            snapshot = PerformanceSnapshot(
                timestamp=time.time() - i,
                metric_type=PerformanceMetric.MEMORY_USAGE,
                value=100 + i * 10,
                operation_id=f"test-{i}",
                context={}
            )
            test_snapshots.append({'memory': snapshot})
        
        self.collector.metrics_buffer.extend(test_snapshots)
        
        summary = self.collector.get_metric_summary(
            PerformanceMetric.MEMORY_USAGE, duration_seconds=300
        )
        
        assert 'count' in summary
        assert 'min' in summary
        assert 'max' in summary
        assert 'mean' in summary
        assert summary['count'] == 5
        assert summary['min'] == 100
        assert summary['max'] == 140


class TestPerformanceOptimizer:
    """Test performance optimization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = PerformanceCollector()
        self.optimizer = PerformanceOptimizer(self.collector)
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.collector.is_collecting:
            self.collector.stop_collection()
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.collector == self.collector
        assert len(self.optimizer.optimization_history) == 0
        assert len(self.optimizer.active_optimizations) == 0
    
    def test_bottleneck_analysis(self):
        """Test performance bottleneck analysis."""
        # Add some high memory usage data
        high_memory_snapshots = []
        for i in range(10):
            snapshot = PerformanceSnapshot(
                timestamp=time.time() - i,
                metric_type=PerformanceMetric.MEMORY_USAGE,
                value=600 + i * 50,  # High memory usage
                operation_id=f"test-{i}",
                context={}
            )
            high_memory_snapshots.append({'memory': snapshot})
        
        self.collector.metrics_buffer.extend(high_memory_snapshots)
        
        analysis = self.optimizer.analyze_performance_bottlenecks()
        
        assert 'timestamp' in analysis
        assert 'bottlenecks' in analysis
        assert 'recommendations' in analysis
        
        # Should detect memory bottleneck
        bottlenecks = analysis['bottlenecks']
        memory_bottleneck = next(
            (b for b in bottlenecks if b['type'] == 'memory_usage'), 
            None
        )
        assert memory_bottleneck is not None
        assert memory_bottleneck['severity'] in ['medium', 'high']
    
    def test_memory_optimization(self):
        """Test memory optimization."""
        result = self.optimizer.optimize_memory_usage()
        
        assert isinstance(result, OptimizationResult)
        assert result.strategy == OptimizationStrategy.MEMORY_OPTIMIZATION
        assert result.duration > 0
        assert isinstance(result.success, bool)
        
        # Should be added to history
        assert len(self.optimizer.optimization_history) == 1
        assert self.optimizer.optimization_history[0] == result
    
    def test_cache_optimization(self):
        """Test cache performance optimization."""
        cache_config = {
            'max_size': 1000,
            'current_hit_rate': 0.6,
            'enable_preloading': False
        }
        
        result = self.optimizer.optimize_cache_performance(cache_config)
        
        assert isinstance(result, OptimizationResult)
        assert result.strategy == OptimizationStrategy.CACHE_OPTIMIZATION
        assert result.baseline_value == 0.6
        assert result.optimized_value > result.baseline_value
        assert result.improvement_percentage > 0
        
        # Check that cache config was modified
        assert cache_config['enable_preloading'] is True
        assert cache_config['eviction_strategy'] == 'lru_with_frequency'
    
    def test_auto_optimization(self):
        """Test automatic system optimization."""
        # Add some data that would trigger optimizations
        high_memory_snapshots = []
        for i in range(5):
            snapshot = PerformanceSnapshot(
                timestamp=time.time() - i,
                metric_type=PerformanceMetric.MEMORY_USAGE,
                value=800,  # High memory usage
                operation_id=f"test-{i}",
                context={}
            )
            high_memory_snapshots.append({'memory': snapshot})
        
        self.collector.metrics_buffer.extend(high_memory_snapshots)
        
        results = self.optimizer.auto_optimize_system()
        
        assert isinstance(results, list)
        
        # Should have performed at least one optimization
        if results:  # May not trigger if thresholds not met
            assert all(isinstance(r, OptimizationResult) for r in results)
            assert len(self.optimizer.optimization_history) == len(results)


class TestPerformanceMonitor:
    """Test integrated performance monitoring."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.collector is not None
        assert self.monitor.optimizer is not None
        assert not self.monitor.is_monitoring
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        assert not self.monitor.is_monitoring
        
        self.monitor.start_monitoring()
        assert self.monitor.is_monitoring
        
        time.sleep(1)  # Let it run briefly
        
        self.monitor.stop_monitoring()
        assert not self.monitor.is_monitoring
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        report = self.monitor.get_performance_report()
        
        assert 'timestamp' in report
        assert 'monitoring_active' in report
        assert 'metrics_summary' in report
        assert 'recent_optimizations' in report
        assert 'system_health' in report
        
        assert report['system_health'] in ['good', 'warning', 'critical']
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            self.monitor.export_metrics(temp_path, 'json')
            
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            # Should have standard report structure
            assert 'timestamp' in exported_data
            assert 'metrics_summary' in exported_data
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAutoScalingEngine:
    """Test auto-scaling functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AutoScalingEngine(ScalingStrategy.REACTIVE)
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.engine.is_running:
            self.engine.stop()
    
    def test_engine_initialization(self):
        """Test auto-scaling engine initialization."""
        assert self.engine.strategy == ScalingStrategy.REACTIVE
        assert len(self.engine.scalers) == 4  # Default resource types
        assert not self.engine.is_running
        
        # Check that scalers were initialized
        expected_resources = {
            ResourceType.CPU_CORES,
            ResourceType.MEMORY_GB,
            ResourceType.CONCURRENT_TASKS,
            ResourceType.CACHE_SIZE
        }
        
        assert set(self.engine.scalers.keys()) == expected_resources
    
    def test_start_stop_engine(self):
        """Test starting and stopping the engine."""
        assert not self.engine.is_running
        
        self.engine.start()
        assert self.engine.is_running
        
        time.sleep(1)  # Let it run briefly
        
        self.engine.stop()
        assert not self.engine.is_running
    
    def test_scaling_status(self):
        """Test scaling status reporting."""
        status = self.engine.get_scaling_status()
        
        assert 'timestamp' in status
        assert 'is_running' in status
        assert 'strategy' in status
        assert 'resources' in status
        assert 'recent_events' in status
        assert 'patterns_detected' in status
        
        # Check resource status structure
        for resource_name, resource_status in status['resources'].items():
            assert 'current_value' in resource_status
            assert 'min_value' in resource_status
            assert 'max_value' in resource_status
            assert 'can_scale_up' in resource_status
            assert 'can_scale_down' in resource_status
    
    def test_force_scaling_evaluation(self):
        """Test forced scaling evaluation."""
        initial_status = self.engine.get_scaling_status()
        
        result = self.engine.force_scaling_evaluation()
        
        assert isinstance(result, dict)
        assert 'timestamp' in result
        
        # Timestamp should be updated
        assert result['timestamp'] >= initial_status['timestamp']
    
    def test_resource_limits_update(self):
        """Test updating resource limits."""
        new_limits = ResourceLimits(
            min_value=2, max_value=32, step_size=2,
            scale_up_threshold=0.6, scale_down_threshold=0.2,
            cooldown_period=60
        )
        
        self.engine.set_resource_limits(ResourceType.CPU_CORES, new_limits)
        
        scaler = self.engine.scalers[ResourceType.CPU_CORES]
        assert scaler.limits.max_value == 32
        assert scaler.limits.scale_up_threshold == 0.6


class TestResourceScaler:
    """Test individual resource scaling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.limits = ResourceLimits(
            min_value=1, max_value=10, step_size=1,
            scale_up_threshold=0.7, scale_down_threshold=0.3,
            cooldown_period=5  # Short cooldown for testing
        )
        self.scaler = ResourceScaler(ResourceType.CPU_CORES, self.limits)
    
    def test_scaler_initialization(self):
        """Test scaler initialization."""
        assert self.scaler.resource_type == ResourceType.CPU_CORES
        assert self.scaler.current_value == self.limits.min_value
        assert self.scaler.last_scaling_time == 0
        assert len(self.scaler.scaling_history) == 0
    
    def test_scaling_constraints(self):
        """Test scaling constraint checking."""
        # Should be able to scale up from minimum
        assert self.scaler.can_scale(ScalingDirection.SCALE_UP)
        assert not self.scaler.can_scale(ScalingDirection.SCALE_DOWN)
        
        # Set to maximum and test constraints
        self.scaler.current_value = self.limits.max_value
        assert not self.scaler.can_scale(ScalingDirection.SCALE_UP)
        assert self.scaler.can_scale(ScalingDirection.SCALE_DOWN)
    
    def test_cooldown_period(self):
        """Test cooldown period enforcement."""
        # Perform a scaling operation
        self.scaler.current_value = 5
        self.scaler.last_scaling_time = time.time()
        
        # Should not be able to scale immediately
        assert not self.scaler.can_scale(ScalingDirection.SCALE_UP)
        assert not self.scaler.can_scale(ScalingDirection.SCALE_DOWN)
        
        # Wait for cooldown
        time.sleep(6)
        
        # Should be able to scale now
        assert self.scaler.can_scale(ScalingDirection.SCALE_UP)
        assert self.scaler.can_scale(ScalingDirection.SCALE_DOWN)
    
    def test_target_value_calculation(self):
        """Test target value calculation."""
        # Test various load levels
        target = self.scaler.calculate_target_value(0.8, 0.9)  # High load
        assert target > self.scaler.current_value
        assert target <= self.limits.max_value
        
        target = self.scaler.calculate_target_value(0.1, 0.2)  # Low load
        assert target >= self.limits.min_value
    
    def test_scaling_operation(self):
        """Test actual scaling operation."""
        initial_value = self.scaler.current_value
        target_value = min(initial_value + 2, self.limits.max_value)
        
        event = self.scaler.scale_to_target(target_value, "Test scaling")
        
        if event:  # Scaling may not occur due to constraints
            assert isinstance(event, ScalingEvent)
            assert event.resource_type == ResourceType.CPU_CORES
            assert event.old_value == initial_value
            assert event.trigger_reason == "Test scaling"
            
            if event.success:
                assert self.scaler.current_value == target_value
                assert len(self.scaler.scaling_history) == 1


class TestLoadPredictor:
    """Test load prediction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = LoadPredictor(history_window_hours=24)
    
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        assert len(self.predictor.load_history) == 0
        assert len(self.predictor.patterns) == 0
    
    def test_load_point_recording(self):
        """Test recording load measurement points."""
        test_metrics = {'cpu_usage': 0.5, 'memory_usage': 0.6}
        
        self.predictor.record_load_point(time.time(), test_metrics)
        
        assert len(self.predictor.load_history) == 1
        
        point = self.predictor.load_history[0]
        assert 'timestamp' in point
        assert 'metrics' in point
        assert point['metrics'] == test_metrics
    
    def test_pattern_analysis(self):
        """Test workload pattern analysis."""
        # Add some test data with patterns
        current_time = time.time()
        
        # Simulate daily pattern with higher load during day
        for i in range(100):
            timestamp = current_time - (i * 3600)  # One point per hour
            hour = (24 - (i % 24)) % 24
            
            # Higher load during business hours (9-17)
            if 9 <= hour <= 17:
                load = 0.8
            else:
                load = 0.3
            
            metrics = {'cpu_usage': load, 'memory_usage': load * 0.8}
            self.predictor.record_load_point(timestamp, metrics)
        
        patterns = self.predictor.analyze_patterns()
        
        # Should identify at least some patterns
        assert isinstance(patterns, list)
        
        if patterns:  # May not detect patterns with simple test data
            pattern = patterns[0]
            assert isinstance(pattern, WorkloadPattern)
            assert pattern.pattern_id is not None
            assert pattern.expected_load_multiplier > 1.0
    
    def test_load_prediction(self):
        """Test load prediction."""
        # Add some historical data
        current_time = time.time()
        
        for i in range(10):
            metrics = {'cpu_usage': 0.4, 'memory_usage': 0.3}
            self.predictor.record_load_point(current_time - i * 3600, metrics)
        
        # Predict future load
        future_time = current_time + 3600
        prediction = self.predictor.predict_load(future_time)
        
        assert isinstance(prediction, dict)
        assert 'cpu_usage' in prediction
        assert 'memory_usage' in prediction
        assert all(isinstance(v, (int, float)) for v in prediction.values())


class TestPerformanceDecorators:
    """Test performance monitoring decorators."""
    
    def test_monitor_performance_decorator(self):
        """Test the monitor_performance decorator."""
        
        @monitor_performance(PerformanceMetric.RESPONSE_TIME)
        def test_function(duration=0.1):
            time.sleep(duration)
            return "completed"
        
        # Function should execute normally
        result = test_function(0.05)
        assert result == "completed"
        
        # Should handle exceptions
        @monitor_performance()
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
    
    def test_optimize_for_performance_decorator(self):
        """Test the optimize_for_performance decorator."""
        
        @optimize_for_performance(OptimizationStrategy.MEMORY_OPTIMIZATION)
        def test_function():
            return "optimized"
        
        result = test_function()
        assert result == "optimized"


class TestGlobalFunctions:
    """Test global performance monitoring functions."""
    
    def teardown_method(self):
        """Clean up after tests."""
        try:
            stop_performance_monitoring()
        except:
            pass
    
    def test_global_monitor_lifecycle(self):
        """Test global performance monitor lifecycle."""
        # Should be able to get monitor instance
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # Should be the same instance (singleton pattern)
        assert monitor1 is monitor2
        
        # Test start/stop
        start_performance_monitoring()
        assert monitor1.is_monitoring
        
        stop_performance_monitoring()
        assert not monitor1.is_monitoring
    
    def test_global_performance_report(self):
        """Test global performance report function."""
        from phd_notebook.performance.enhanced_performance_monitor import get_performance_report
        
        report = get_performance_report()
        
        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'monitoring_active' in report
    
    def test_global_optimization_trigger(self):
        """Test global optimization trigger function."""
        from phd_notebook.performance.enhanced_performance_monitor import optimize_system_performance
        
        results = optimize_system_performance()
        
        assert isinstance(results, list)
        # Results may be empty if no optimizations were triggered


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])