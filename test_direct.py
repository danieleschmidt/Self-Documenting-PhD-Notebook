"""
Direct Testing Without Package Dependencies

Test core functionality directly without importing the full package.
"""

import sys
import time
import traceback
from pathlib import Path

# Add specific module paths directly
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_secure_execution_direct():
    """Test secure execution functionality directly."""
    # Import specific modules without going through package init
    sys.path.insert(0, str(Path(__file__).parent / 'src/phd_notebook/utils'))
    
    from secure_execution_fixed import SafeEvaluator, SecureExecutionError
    
    evaluator = SafeEvaluator()
    
    # Test safe operations
    assert evaluator.safe_eval('1 + 2') == 3
    print("✓ Basic arithmetic works")
    
    assert evaluator.safe_eval('max(1, 2, 3)') == 3
    print("✓ Function calls work")
    
    context = {'x': 10, 'y': 20}
    assert evaluator.safe_eval('x + y', context) == 30
    print("✓ Context variables work")
    
    # Test unsafe operations are blocked
    try:
        evaluator.safe_eval('__import__("os").system("ls")')
        assert False, "Should have blocked unsafe operation"
    except SecureExecutionError:
        print("✓ Unsafe operations are blocked")
    
    return True


def test_performance_monitor_direct():
    """Test performance monitoring directly."""
    sys.path.insert(0, str(Path(__file__).parent / 'src/phd_notebook/performance'))
    
    from enhanced_performance_monitor import PerformanceCollector, PerformanceMetric
    
    collector = PerformanceCollector()
    
    # Test metrics collection
    metrics = collector._collect_system_metrics()
    assert len(metrics) > 0
    print(f"✓ Collected {len(metrics)} metric types")
    
    for metric_name, metric in metrics.items():
        assert hasattr(metric, 'timestamp')
        assert hasattr(metric, 'value')
        assert metric.timestamp > 0
        print(f"  - {metric_name}: {metric.value}")
    
    # Test metric summary
    # Add some test data first
    import time
    from collections import deque
    
    # Create some test snapshots
    from enhanced_performance_monitor import PerformanceSnapshot
    
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
    
    collector.metrics_buffer.extend(test_snapshots)
    
    summary = collector.get_metric_summary(PerformanceMetric.MEMORY_USAGE, duration_seconds=300)
    
    assert 'count' in summary
    assert 'min' in summary
    assert 'max' in summary
    assert summary['count'] == 5
    print("✓ Metric summary calculation works")
    
    return True


def test_auto_scaling_direct():
    """Test auto-scaling directly."""
    sys.path.insert(0, str(Path(__file__).parent / 'src/phd_notebook/performance'))
    
    from auto_scaling_engine import AutoScalingEngine, ScalingStrategy, ResourceType
    
    engine = AutoScalingEngine(ScalingStrategy.REACTIVE)
    
    # Test initialization
    assert len(engine.scalers) > 0
    assert not engine.is_running
    print(f"✓ Initialized with {len(engine.scalers)} scalers")
    
    # Test status reporting
    status = engine.get_scaling_status()
    
    assert 'timestamp' in status
    assert 'is_running' in status
    assert 'resources' in status
    print("✓ Status reporting works")
    
    # Test load prediction
    predictor = engine.load_predictor
    
    # Add some test data
    current_time = time.time()
    for i in range(10):
        metrics = {'cpu_usage': 0.5, 'memory_usage': 0.4}
        predictor.record_load_point(current_time - i * 3600, metrics)
    
    # Test prediction
    future_time = current_time + 3600
    prediction = predictor.predict_load(future_time)
    
    assert isinstance(prediction, dict)
    assert 'cpu_usage' in prediction
    print("✓ Load prediction works")
    
    return True


def test_security_patching_direct():
    """Test security patching directly."""
    import tempfile
    import os
    
    sys.path.insert(0, str(Path(__file__).parent / 'src/phd_notebook/security'))
    
    from security_patch import SecurityPatcher
    
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
        print(f"✓ Found {len(vulnerabilities)} vulnerabilities")
        
        vuln_types = {v.vulnerability_type for v in vulnerabilities}
        assert 'eval' in vuln_types
        assert 'hardcoded_secret' in vuln_types
        print(f"✓ Detected vulnerability types: {', '.join(vuln_types)}")
        
        # Test severity assessment
        critical_count = sum(1 for v in vulnerabilities if v.severity == 'CRITICAL')
        high_count = sum(1 for v in vulnerabilities if v.severity == 'HIGH')
        print(f"✓ Severity assessment: {critical_count} critical, {high_count} high")
        
        # Test report generation
        report = patcher.generate_security_report()
        assert 'Research Platform Security Report' in report
        assert 'Summary' in report
        print("✓ Security report generation works")
        
        return True
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_json_operations():
    """Test JSON operations."""
    sys.path.insert(0, str(Path(__file__).parent / 'src/phd_notebook/utils'))
    
    from secure_execution_fixed import safe_json_loads, SecureExecutionError
    
    # Test valid JSON
    valid_json = '{"name": "test", "value": 42, "list": [1, 2, 3]}'
    result = safe_json_loads(valid_json)
    
    assert result['name'] == 'test'
    assert result['value'] == 42
    assert result['list'] == [1, 2, 3]
    print("✓ Valid JSON parsing works")
    
    # Test invalid JSON
    invalid_json = '{"invalid": json}'
    try:
        safe_json_loads(invalid_json)
        assert False, "Should have raised error"
    except SecureExecutionError:
        print("✓ Invalid JSON properly rejected")
    
    return True


def test_mathematical_operations():
    """Test mathematical operations."""
    sys.path.insert(0, str(Path(__file__).parent / 'src/phd_notebook/utils'))
    
    from secure_execution_fixed import safe_mathematical_eval, SecureExecutionError
    
    # Test basic math
    variables = {'x': 5, 'y': 10}
    
    assert safe_mathematical_eval('x + y', variables) == 15.0
    print("✓ Basic addition works")
    
    assert safe_mathematical_eval('x * y + 5', variables) == 55.0
    print("✓ Complex expressions work")
    
    # Test with constants
    result = safe_mathematical_eval('pi * 2', {'pi': 3.14159})
    assert abs(result - 6.28318) < 0.001
    print("✓ Constants work")
    
    # Test error handling
    try:
        safe_mathematical_eval('x + invalid_var', variables)
        assert False, "Should have raised error"
    except SecureExecutionError:
        print("✓ Invalid variables properly rejected")
    
    return True


def run_test(test_name, test_func):
    """Run a single test function."""
    print(f"\n{'='*50}")
    print(f"Running: {test_name}")
    print('='*50)
    
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


def main():
    """Run all direct tests."""
    print("🧪 Research Platform Direct Test Suite")
    print("=" * 50)
    
    tests = [
        ("Secure Execution", test_secure_execution_direct),
        ("Performance Monitoring", test_performance_monitor_direct),
        ("Auto-Scaling", test_auto_scaling_direct),
        ("Security Patching", test_security_patching_direct),
        ("JSON Operations", test_json_operations),
        ("Mathematical Operations", test_mathematical_operations),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*50}")
    print("📊 Test Results Summary")
    print('='*50)
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