"""
Security framework for autonomous research operations.
"""

try:
    from .autonomous_security_framework import (
        AutonomousSecurityFramework,
        ThreatDetector,
        SecurityIncident,
        ThreatLevel,
        SecurityMetrics
    )
    
    __all__ = [
        'AutonomousSecurityFramework',
        'ThreatDetector', 
        'SecurityIncident',
        'ThreatLevel',
        'SecurityMetrics'
    ]
    
except ImportError as e:
    # Graceful handling of missing dependencies
    __all__ = []
    print(f"Warning: Security modules unavailable due to missing dependencies: {e}")