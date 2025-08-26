"""
Global collaboration and knowledge sharing systems.
"""

try:
    from .global_research_intelligence_network import (
        GlobalResearchIntelligenceNetwork,
        ResearchNode,
        CollaborationLink,
        KnowledgePackage,
        CollaborationMatcher
    )
    
    __all__ = [
        'GlobalResearchIntelligenceNetwork',
        'ResearchNode',
        'CollaborationLink', 
        'KnowledgePackage',
        'CollaborationMatcher'
    ]
    
except ImportError as e:
    # Graceful handling of missing dependencies
    __all__ = []
    print(f"Warning: Collaboration modules unavailable due to missing dependencies: {e}")