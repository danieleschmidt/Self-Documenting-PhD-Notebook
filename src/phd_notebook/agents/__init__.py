"""AI agents for research automation."""

from .base import BaseAgent, SimpleAgent
from .literature_agent import LiteratureAgent
from .experiment_agent import ExperimentAgent
from .sparc_agent import SPARCAgent

# Autonomous SDLC Generation 1 agents
try:
    from .meta_research_agent import (
        MetaResearchAgent,
        StatisticalMetaAnalyzer,
        SystematicReview,
        MetaAnalysisResult
    )
    _meta_research_available = True
except ImportError as e:
    print(f"Warning: Meta-research agent unavailable: {e}")
    _meta_research_available = False

__all__ = [
    "BaseAgent", 
    "SimpleAgent", 
    "LiteratureAgent", 
    "ExperimentAgent", 
    "SPARCAgent"
]

# Add autonomous agents if available
if _meta_research_available:
    __all__.extend([
        "MetaResearchAgent",
        "StatisticalMetaAnalyzer", 
        "SystematicReview",
        "MetaAnalysisResult"
    ])