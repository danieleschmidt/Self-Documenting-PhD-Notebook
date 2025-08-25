"""
Manual Test Runner for Research Platform

Simple test runner that doesn't require pytest or external dependencies.
Provides basic testing functionality for the research platform.
"""

import sys
import time
import traceback
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_test(test_name, test_func):
    """Run a single test function."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print('='*60)
    
    try:
        start_time = time.time()
        result = test_func()
        duration = time.time() - start_time
        
        print(f"✅ PASSED - {test_name} ({duration:.3f}s)")
        return True
        
    except Exception as e:
        print(f"❌ FAILED - {test_name}")
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False


def test_secure_execution():
    """Test secure execution functionality."""
    from phd_notebook.utils.secure_execution_fixed import SafeEvaluator, SecureExecutionError
    
    evaluator = SafeEvaluator()
    
    # Test safe operations
    assert evaluator.safe_eval('1 + 2') == 3
    assert evaluator.safe_eval('max(1, 2, 3)') == 3
    
    context = {'x': 10, 'y': 20}
    assert evaluator.safe_eval('x + y', context) == 30
    
    # Test unsafe operations are blocked
    try:
        evaluator.safe_eval('__import__("os").system("ls")')
        assert False, "Should have blocked unsafe operation"
    except SecureExecutionError:
        pass  # Expected
    
    print("✓ Safe mathematical operations work")
    print("✓ Unsafe operations are blocked")
    return True


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    from phd_notebook.performance.enhanced_performance_monitor import (
        PerformanceCollector, PerformanceMonitor
    )
    
    # Test collector
    collector = PerformanceCollector()
    
    # Test metrics collection
    metrics = collector._collect_system_metrics()
    assert len(metrics) > 0
    
    for metric_name, metric in metrics.items():
        assert hasattr(metric, 'timestamp')
        assert hasattr(metric, 'value')
        assert metric.timestamp > 0
    
    # Test monitor
    monitor = PerformanceMonitor()
    report = monitor.get_performance_report()
    
    assert 'timestamp' in report
    assert 'monitoring_active' in report
    assert 'system_health' in report
    
    print("✓ Performance metrics collection works")
    print("✓ Performance monitoring report generation works")
    return True


def test_auto_scaling():
    """Test auto-scaling functionality."""
    from phd_notebook.performance.auto_scaling_engine import (
        AutoScalingEngine, ResourceType, ResourceLimits, ScalingStrategy
    )
    
    engine = AutoScalingEngine(ScalingStrategy.REACTIVE)
    
    # Test initialization
    assert len(engine.scalers) > 0
    assert not engine.is_running
    
    # Test status reporting
    status = engine.get_scaling_status()
    
    assert 'timestamp' in status
    assert 'is_running' in status
    assert 'resources' in status
    
    # Test resource configuration
    for resource_name, resource_info in status['resources'].items():
        assert 'current_value' in resource_info
        assert 'min_value' in resource_info
        assert 'max_value' in resource_info
        assert resource_info['current_value'] >= resource_info['min_value']
        assert resource_info['current_value'] <= resource_info['max_value']
    
    print("✓ Auto-scaling engine initialization works")
    print("✓ Resource scaling configuration works")
    return True


def test_security_patching():
    """Test security patching functionality."""
    import tempfile
    import os
    from phd_notebook.security.security_patch import SecurityPatcher
    
    # Create temporary test directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test file with vulnerabilities
        test_file = Path(temp_dir) / 'test.py'
        with open(test_file, 'w') as f:
            f.write('''
import os
result = eval("1 + 2")
secret_key = "hardcoded_password_123"
''')
        
        # Test vulnerability scanning
        patcher = SecurityPatcher(temp_dir)
        vulnerabilities = patcher.scan_vulnerabilities()
        
        assert len(vulnerabilities) > 0
        
        vuln_types = {v.vulnerability_type for v in vulnerabilities}
        assert 'eval' in vuln_types
        assert 'hardcoded_secret' in vuln_types
        
        # Test severity assessment
        for vuln in vulnerabilities:
            assert vuln.severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        
        # Test report generation
        report = patcher.generate_security_report()
        assert 'Research Platform Security Report' in report
        assert 'Summary' in report
        
        print("✓ Security vulnerability scanning works")
        print("✓ Security report generation works")
        return True
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_integration():
    """Test basic integration between systems."""
    from phd_notebook.performance.enhanced_performance_monitor import (
        get_performance_monitor, start_performance_monitoring, stop_performance_monitoring
    )
    from phd_notebook.utils.secure_execution_fixed import SafeEvaluator
    
    # Start monitoring
    start_performance_monitoring()
    
    try:
        # Test that secure execution works with monitoring
        evaluator = SafeEvaluator()
        
        # Perform some operations
        results = []
        for i in range(10):
            result = evaluator.safe_eval(f'{i} * 2 + 1')
            results.append(result)
            time.sleep(0.01)  # Small delay
        
        # Check results
        expected_results = [i * 2 + 1 for i in range(10)]
        assert results == expected_results
        
        # Check that monitoring is still active
        monitor = get_performance_monitor()
        report = monitor.get_performance_report()
        
        assert report['monitoring_active'] is True
        
        print("✓ Secure execution works with performance monitoring")
        print("✓ Systems integrate properly")
        return True
        
    finally:
        stop_performance_monitoring()


def test_error_handling():
    """Test error handling across systems."""
    from phd_notebook.utils.secure_execution_fixed import SafeEvaluator, SecureExecutionError
    
    evaluator = SafeEvaluator()
    
    # Test that errors are properly handled
    try:
        evaluator.safe_eval('undefined_variable + 1')
        assert False, "Should have raised an error"
    except SecureExecutionError:
        pass  # Expected
    
    # Test that system continues to work after error
    result = evaluator.safe_eval('1 + 1')
    assert result == 2
    
    # Test invalid syntax handling
    try:
        evaluator.safe_eval('1 + + 2')  # Invalid syntax
        assert False, "Should have raised an error"
    except SecureExecutionError:
        pass  # Expected
    
    # System should still work
    result = evaluator.safe_eval('2 + 2')
    assert result == 4
    
    print("✓ Error handling works properly")
    print("✓ System recovers from errors")
    return True


def test_data_processing():
    """Test data processing capabilities."""
    from phd_notebook.utils.secure_execution_fixed import SafeEvaluator
    
    evaluator = SafeEvaluator()
    
    # Test with different data types
    test_cases = [
        # Numbers
        ('1 + 2 * 3', {}, 7),
        ('10 / 2', {}, 5.0),
        ('2 ** 8', {}, 256),
        
        # Lists
        ('len([1, 2, 3, 4, 5])', {}, 5),
        ('sum([1, 2, 3])', {}, 6),
        ('max([1, 5, 3])', {}, 5),
        
        # Strings
        ('len("hello")', {}, 5),
        ('"hello".upper()', {}, "HELLO"),
        ('"test".startswith("te")', {}, True),
        
        # Context variables
        ('x + y', {'x': 10, 'y': 15}, 25),
        ('name + " " + surname', {'name': 'John', 'surname': 'Doe'}, 'John Doe'),
    ]
    
    for expression, context, expected in test_cases:
        result = evaluator.safe_eval(expression, context)
        assert result == expected, f"Failed: {expression} -> {result} != {expected}"
    
    print("✓ Various data types processed correctly")
    print("✓ Context variables work properly")
    return True


def main():
    """Run all manual tests."""
    print("🧪 Research Platform Manual Test Suite")
    print("=" * 60)
    
    tests = [
        ("Secure Execution", test_secure_execution),
        ("Performance Monitoring", test_performance_monitoring),
        ("Auto-Scaling", test_auto_scaling),
        ("Security Patching", test_security_patching),
        ("Integration", test_integration),
        ("Error Handling", test_error_handling),
        ("Data Processing", test_data_processing),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print("📊 Test Results Summary")
    print('='*60)
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())