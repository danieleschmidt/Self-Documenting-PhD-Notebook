#!/usr/bin/env python3
"""
Direct Autonomous Module Test - Tests modules without package imports
"""

import sys
import os
import importlib.util
from pathlib import Path

def load_module_from_file(module_name, file_path):
    """Load a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading {module_name}: {e}")
        return None

def test_autonomous_modules():
    """Test autonomous modules directly."""
    src_path = Path("src/phd_notebook")
    results = {}
    
    # Test each autonomous module
    modules_to_test = [
        ("research/autonomous_discovery_engine.py", "AutonomousDiscoveryEngine"),
        ("agents/meta_research_agent.py", "MetaResearchAgent"),
        ("performance/adaptive_research_optimizer.py", "AdaptiveResearchOptimizer"),
        ("security/autonomous_security_framework.py", "AutonomousSecurityFramework"),
        ("monitoring/advanced_research_intelligence.py", "AdvancedResearchIntelligence"),
        ("validation/comprehensive_validation_framework.py", "ComprehensiveValidationFramework"),
        ("performance/quantum_performance_optimizer.py", "QuantumPerformanceOptimizer"),
        ("collaboration/global_research_intelligence_network.py", "GlobalResearchIntelligenceNetwork"),
    ]
    
    print("🚀 DIRECT AUTONOMOUS MODULE TEST")
    print("=" * 50)
    
    for module_path, class_name in modules_to_test:
        full_path = src_path / module_path
        print(f"\n📦 Testing {module_path}...")
        
        if not full_path.exists():
            print(f"❌ File not found: {full_path}")
            results[module_path] = False
            continue
            
        # Load module directly
        module = load_module_from_file(class_name, full_path)
        if module is None:
            print(f"❌ Failed to load module: {module_path}")
            results[module_path] = False
            continue
            
        # Check if main class exists
        if hasattr(module, class_name):
            print(f"✅ {class_name} class found")
            
            # Try to instantiate (basic test)
            try:
                cls = getattr(module, class_name)
                instance = cls()
                print(f"✅ {class_name} instantiated successfully")
                results[module_path] = True
            except Exception as e:
                print(f"⚠️  {class_name} class found but instantiation failed: {e}")
                results[module_path] = "partial"
        else:
            print(f"❌ {class_name} class not found in module")
            results[module_path] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r is True)
    partial = sum(1 for r in results.values() if r == "partial")
    failed = sum(1 for r in results.values() if r is False)
    total = len(results)
    
    print(f"✅ Fully Passed: {passed}")
    print(f"⚠️  Partial Pass: {partial}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {((passed + partial * 0.5) / total * 100):.1f}%")
    
    return results

if __name__ == "__main__":
    results = test_autonomous_modules()
    success_rate = sum(1 for r in results.values() if r is True) / len(results)
    sys.exit(0 if success_rate >= 0.8 else 1)