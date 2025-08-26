"""Monitoring and health check systems."""

from .logging_setup import setup_logging, get_logger

# Autonomous SDLC Generation 2 components
try:
    from .advanced_research_intelligence import (
        AdvancedResearchIntelligence,
        PredictiveAnalyticsEngine,
        ResearchInsight,
        IntelligenceMetrics
    )
    _research_intelligence_available = True
except ImportError as e:
    print(f"Warning: Advanced research intelligence unavailable: {e}")
    _research_intelligence_available = False

__all__ = [
    'setup_logging', 'get_logger'
]

# Add autonomous monitoring components if available
if _research_intelligence_available:
    __all__.extend([
        'AdvancedResearchIntelligence',
        'PredictiveAnalyticsEngine',
        'ResearchInsight',
        'IntelligenceMetrics'
    ])