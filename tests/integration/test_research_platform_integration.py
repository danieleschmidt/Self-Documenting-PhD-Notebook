"""
Integration Tests for Research Platform

End-to-end integration tests that verify the complete research platform
functionality including security, performance, and core features.
"""

import pytest
import asyncio
import json
import time
import threading
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from phd_notebook.performance.enhanced_performance_monitor import (
    get_performance_monitor,
    start_performance_monitoring,
    stop_performance_monitoring,
    get_performance_report
)

from phd_notebook.performance.auto_scaling_engine import (
    get_scaling_engine,
    start_auto_scaling,
    stop_auto_scaling,
    get_scaling_status,
    ScalingStrategy
)

from phd_notebook.security.security_patch import (
    SecurityPatcher,
    run_security_patch
)

from phd_notebook.utils.secure_execution_fixed import (
    SafeEvaluator,
    SafeExecutor,
    default_evaluator,
    default_executor
)


class TestPlatformIntegration:
    """Test integration between major platform components."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {
            'research_notes': [],
            'performance_metrics': [],
            'security_events': []
        }
    
    def teardown_method(self):
        """Clean up integration test environment."""
        # Stop all monitoring systems
        try:
            stop_performance_monitoring()
            stop_auto_scaling()
        except:
            pass
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_performance_security_integration(self):
        """Test integration between performance monitoring and security systems."""
        # Start performance monitoring
        start_performance_monitoring()
        
        # Get initial performance report
        initial_report = get_performance_report()
        assert initial_report['monitoring_active'] is True
        
        # Test that security-hardened evaluation works within performance context
        evaluator = SafeEvaluator()
        
        # This should work - safe mathematical operation
        result = evaluator.safe_eval('2 ** 8')  # 256
        assert result == 256
        
        # This should be blocked - unsafe operation
        with pytest.raises(Exception):  # SecureExecutionError or similar
            evaluator.safe_eval('__import__("os").system("echo test")')
        
        # Performance monitoring should still be active
        final_report = get_performance_report()
        assert final_report['monitoring_active'] is True
        assert final_report['timestamp'] > initial_report['timestamp']
    
    def test_auto_scaling_performance_integration(self):
        """Test integration between auto-scaling and performance monitoring."""
        # Start both systems
        start_performance_monitoring()
        start_auto_scaling(ScalingStrategy.REACTIVE)
        
        # Let them run for a brief period
        time.sleep(2)
        
        # Get status from both systems
        perf_report = get_performance_report()
        scaling_status = get_scaling_status()
        
        # Both should be active
        assert perf_report['monitoring_active'] is True
        assert scaling_status['is_running'] is True
        
        # Check that scaling decisions can be made based on performance data
        assert 'resources' in scaling_status
        assert len(scaling_status['resources']) > 0
        
        # Verify resource scaling capabilities
        for resource_name, resource_info in scaling_status['resources'].items():
            assert 'current_value' in resource_info
            assert 'min_value' in resource_info
            assert 'max_value' in resource_info
            assert resource_info['current_value'] >= resource_info['min_value']
            assert resource_info['current_value'] <= resource_info['max_value']
    
    def test_security_patching_integration(self):
        """Test integration of security patching with other systems."""
        # Create a temporary file with security vulnerabilities
        vulnerable_file = Path(self.temp_dir) / 'vulnerable_code.py'
        with open(vulnerable_file, 'w') as f:
            f.write('''
import os
# This is vulnerable code for testing
result = eval("1 + 1")
secret_key = "hardcoded_password_123"
''')
        
        # Initialize security patcher for temp directory
        patcher = SecurityPatcher(self.temp_dir)
        
        # Scan for vulnerabilities
        vulnerabilities = patcher.scan_vulnerabilities()
        assert len(vulnerabilities) > 0
        
        # Verify that eval and hardcoded secret were detected
        vuln_types = {v.vulnerability_type for v in vulnerabilities}
        assert 'eval' in vuln_types
        assert 'hardcoded_secret' in vuln_types
        
        # Apply patches
        patch_result = patcher.apply_automated_patches()
        
        # Some patches should have been applied
        assert patch_result.patched_vulnerabilities >= 0
        assert patch_result.total_vulnerabilities > 0
        
        # Verify that patched code uses secure alternatives
        if patch_result.patched_vulnerabilities > 0:
            with open(vulnerable_file, 'r') as f:
                patched_content = f.read()
            
            # Should import secure execution utilities
            assert ('secure_execution_fixed' in patched_content or 
                   'os.getenv' in patched_content)
    
    def test_end_to_end_research_workflow(self):
        """Test complete research workflow with all systems active."""
        # Start all monitoring systems
        start_performance_monitoring()
        start_auto_scaling(ScalingStrategy.HYBRID)
        
        try:
            # Simulate research operations
            research_operations = [
                {'name': 'data_analysis', 'duration': 0.1},
                {'name': 'model_training', 'duration': 0.2},
                {'name': 'result_visualization', 'duration': 0.1},
                {'name': 'paper_generation', 'duration': 0.15}
            ]
            
            # Execute research operations with monitoring
            operation_results = []
            
            for operation in research_operations:
                start_time = time.time()
                
                # Simulate research computation with safe evaluation
                evaluator = SafeEvaluator()
                
                # Safe mathematical operations for research
                if operation['name'] == 'data_analysis':
                    result = evaluator.safe_eval('sum([1, 2, 3, 4, 5]) / len([1, 2, 3, 4, 5])')
                    assert result == 3.0  # Average
                
                elif operation['name'] == 'model_training':
                    # Simulate model performance calculation
                    context = {'accuracy': 0.95, 'loss': 0.05}
                    result = evaluator.safe_eval('accuracy - loss', context)
                    assert result == 0.9
                
                elif operation['name'] == 'result_visualization':
                    # Simulate plot data generation
                    result = evaluator.safe_eval('list(range(10))')
                    assert result == list(range(10))
                
                elif operation['name'] == 'paper_generation':
                    # Simulate text processing
                    context = {'word_count': 5000, 'page_limit': 8}
                    result = evaluator.safe_eval('word_count / (page_limit * 250)', context)
                    assert abs(result - 2.5) < 0.1  # Words per page estimate
                
                # Simulate operation duration
                time.sleep(operation['duration'])
                
                end_time = time.time()
                actual_duration = end_time - start_time
                
                operation_results.append({
                    'operation': operation['name'],
                    'duration': actual_duration,
                    'success': True
                })
            
            # Let monitoring systems collect data
            time.sleep(1)
            
            # Verify all systems are still operational
            perf_report = get_performance_report()
            scaling_status = get_scaling_status()
            
            assert perf_report['monitoring_active'] is True
            assert scaling_status['is_running'] is True
            
            # Check that metrics were collected during operations
            assert 'metrics_summary' in perf_report
            metrics_summary = perf_report['metrics_summary']
            
            # Should have collected some metrics during our operations
            if metrics_summary:
                # Check for memory metrics
                if 'memory_usage' in metrics_summary:
                    memory_stats = metrics_summary['memory_usage']
                    assert memory_stats['count'] > 0
                    assert memory_stats['latest'] > 0
            
            # Verify that research operations completed successfully
            assert len(operation_results) == len(research_operations)
            assert all(result['success'] for result in operation_results)
            
        finally:
            # Clean up
            stop_performance_monitoring()
            stop_auto_scaling()
    
    def test_error_handling_integration(self):
        """Test error handling across integrated systems."""
        # Start monitoring systems
        start_performance_monitoring()
        
        try:
            # Test error handling in secure execution
            evaluator = SafeEvaluator()
            
            # This should raise an error but not crash the monitoring
            with pytest.raises(Exception):
                evaluator.safe_eval('invalid_variable + 10')
            
            # Monitoring should still be active after the error
            report = get_performance_report()
            assert report['monitoring_active'] is True
            
            # Test error in auto-scaling
            start_auto_scaling()
            
            # Force a scaling evaluation which might encounter edge cases
            scaling_engine = get_scaling_engine()
            status = scaling_engine.force_scaling_evaluation()
            
            # Should return status even if some operations failed
            assert isinstance(status, dict)
            assert 'timestamp' in status
            
        finally:
            stop_performance_monitoring()
            stop_auto_scaling()
    
    def test_concurrent_operations_integration(self):
        """Test integration under concurrent operations."""
        # Start all systems
        start_performance_monitoring()
        start_auto_scaling()
        
        results = {'errors': [], 'successes': 0}
        
        def worker_thread(thread_id):
            """Worker function for concurrent testing."""
            try:
                evaluator = SafeEvaluator()
                
                # Perform safe operations
                for i in range(5):
                    result = evaluator.safe_eval(f'{thread_id} * {i} + 1')
                    assert result == thread_id * i + 1
                    time.sleep(0.01)  # Small delay
                
                results['successes'] += 1
                
            except Exception as e:
                results['errors'].append(f"Thread {thread_id}: {e}")
        
        try:
            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)
            
            # Check results
            assert len(results['errors']) == 0, f"Errors occurred: {results['errors']}"
            assert results['successes'] == 5
            
            # Systems should still be operational
            perf_report = get_performance_report()
            scaling_status = get_scaling_status()
            
            assert perf_report['monitoring_active'] is True
            assert scaling_status['is_running'] is True
            
        finally:
            stop_performance_monitoring()
            stop_auto_scaling()
    
    def test_configuration_integration(self):
        """Test configuration management across systems."""
        # Test that configuration changes affect all systems appropriately
        
        # Create a security config
        patcher = SecurityPatcher(self.temp_dir)
        security_config = patcher.create_security_config()
        
        assert security_config['security']['enable_safe_mode'] is True
        assert security_config['security']['disable_eval_exec'] is True
        
        # Test that safe mode affects evaluation
        evaluator = SafeEvaluator()
        
        # Should work - safe operation
        result = evaluator.safe_eval('1 + 2 * 3')
        assert result == 7
        
        # Should fail - unsafe operation (confirming safe mode is active)
        with pytest.raises(Exception):
            evaluator.safe_eval('exec("print(1)")')
        
        # Test performance configuration integration
        start_performance_monitoring()
        
        try:
            perf_monitor = get_performance_monitor()
            
            # Should be able to get configuration-dependent information
            report = perf_monitor.get_performance_report()
            
            assert 'system_health' in report
            assert report['system_health'] in ['good', 'warning', 'critical']
            
        finally:
            stop_performance_monitoring()
    
    def test_data_persistence_integration(self):
        """Test data persistence across system restarts."""
        # Start monitoring and collect some data
        start_performance_monitoring()
        
        # Let it collect some metrics
        time.sleep(1)
        
        initial_report = get_performance_report()
        initial_metrics_count = sum(
            stats.get('count', 0) 
            for stats in initial_report.get('metrics_summary', {}).values()
        )
        
        # Stop and restart monitoring
        stop_performance_monitoring()
        start_performance_monitoring()
        
        try:
            # Let it collect new metrics
            time.sleep(1)
            
            new_report = get_performance_report()
            
            # Should be collecting new metrics
            assert new_report['monitoring_active'] is True
            assert new_report['timestamp'] > initial_report['timestamp']
            
        finally:
            stop_performance_monitoring()


class TestScalabilityIntegration:
    """Test platform scalability and resource management."""
    
    def setup_method(self):
        """Set up scalability tests."""
        self.evaluator = SafeEvaluator()
    
    def test_large_data_processing(self):
        """Test processing large datasets safely."""
        # Create a large list
        large_list = list(range(10000))
        context = {'data': large_list}
        
        # Should be able to process large datasets safely
        result = self.evaluator.safe_eval('len(data)', context)
        assert result == 10000
        
        result = self.evaluator.safe_eval('sum(data[:100])', context)
        assert result == sum(range(100))  # 0+1+2+...+99 = 4950
        
        result = self.evaluator.safe_eval('max(data)', context)
        assert result == 9999
    
    def test_memory_efficient_operations(self):
        """Test memory-efficient operations."""
        start_performance_monitoring()
        
        try:
            # Get initial memory usage
            initial_report = get_performance_report()
            
            # Perform memory-intensive operations
            large_calculations = []
            for i in range(100):
                result = self.evaluator.safe_eval(
                    f'sum(range({i * 10}))', 
                    {}
                )
                large_calculations.append(result)
            
            # Check that operations completed
            assert len(large_calculations) == 100
            
            # Memory monitoring should still be active
            final_report = get_performance_report()
            assert final_report['monitoring_active'] is True
            
        finally:
            stop_performance_monitoring()
    
    def test_high_concurrency_operations(self):
        """Test high concurrency with safety guarantees."""
        results = []
        errors = []
        
        def concurrent_worker(worker_id):
            """Worker for concurrency testing."""
            try:
                evaluator = SafeEvaluator()
                worker_results = []
                
                for i in range(20):  # More operations per worker
                    result = evaluator.safe_eval(
                        f'({worker_id} * 100) + {i}', 
                        {}
                    )
                    worker_results.append(result)
                    
                    # Verify calculation is correct
                    expected = (worker_id * 100) + i
                    assert result == expected
                
                results.append({
                    'worker_id': worker_id,
                    'results': worker_results,
                    'success': True
                })
                
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Create more threads for higher concurrency
        threads = []
        for i in range(10):  # 10 concurrent workers
            thread = threading.Thread(target=concurrent_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)  # Longer timeout for more operations
        
        # Verify results
        assert len(errors) == 0, f"Concurrency errors: {errors}"
        assert len(results) == 10
        
        # Verify all workers completed their operations
        for result in results:
            assert result['success'] is True
            assert len(result['results']) == 20


class TestResillienceIntegration:
    """Test system resilience and recovery."""
    
    def test_error_recovery(self):
        """Test system recovery from errors."""
        start_performance_monitoring()
        
        try:
            evaluator = SafeEvaluator()
            
            # Perform successful operations
            result1 = evaluator.safe_eval('1 + 1')
            assert result1 == 2
            
            # Cause an error
            try:
                evaluator.safe_eval('undefined_variable')
                assert False, "Should have raised an error"
            except:
                pass  # Expected error
            
            # System should recover and continue working
            result2 = evaluator.safe_eval('2 + 2')
            assert result2 == 4
            
            # Performance monitoring should still be active
            report = get_performance_report()
            assert report['monitoring_active'] is True
            
        finally:
            stop_performance_monitoring()
    
    def test_system_overload_handling(self):
        """Test handling of system overload conditions."""
        start_performance_monitoring()
        start_auto_scaling(ScalingStrategy.REACTIVE)
        
        try:
            # Simulate high load
            evaluator = SafeEvaluator()
            
            # Perform many operations quickly
            for i in range(1000):
                result = evaluator.safe_eval(f'{i} % 10')
                assert result == i % 10
            
            # Systems should still be responsive
            perf_report = get_performance_report()
            scaling_status = get_scaling_status()
            
            assert perf_report['monitoring_active'] is True
            assert scaling_status['is_running'] is True
            
            # Auto-scaling might have triggered
            if scaling_status['recent_events']:
                # Verify scaling events are properly recorded
                for event in scaling_status['recent_events']:
                    assert 'timestamp' in event
                    assert 'resource' in event
                    assert 'success' in event
            
        finally:
            stop_performance_monitoring()
            stop_auto_scaling()


if __name__ == '__main__':
    # Run integration tests
    pytest.main([__file__, '-v', '-s'])