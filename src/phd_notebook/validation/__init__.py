"""
Comprehensive validation framework for research quality assurance.
"""

try:
    from .comprehensive_validation_framework import (
        ComprehensiveValidationFramework,
        ValidationRuleEngine,
        ValidationRule,
        ValidationResult,
        ValidationLevel
    )
    
    __all__ = [
        'ComprehensiveValidationFramework',
        'ValidationRuleEngine',
        'ValidationRule',
        'ValidationResult', 
        'ValidationLevel'
    ]
    
except ImportError as e:
    # Graceful handling of missing dependencies
    __all__ = []
    print(f"Warning: Validation modules unavailable due to missing dependencies: {e}")