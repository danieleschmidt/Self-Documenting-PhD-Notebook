"""
Advanced Research Components for PhD Notebook
"""

from .hypothesis_engine import HypothesisEngine
from .experiment_designer import ExperimentDesigner
from .research_tracker import ResearchTracker
from .publication_pipeline import PublicationPipeline
from .hypothesis_testing_engine import (
    HypothesisTestingEngine, 
    ResearchDrivenNotebook,
    Hypothesis,
    Evidence,
    HypothesisStatus,
    EvidenceType
)
from .collaboration_engine import (
    CollaborationEngine,
    Collaborator,
    Collaboration,
    CollaborationType,
    InteractionType,
    CollaborationStatus
)
from .intelligent_paper_generator import (
    IntelligentPaperGenerator,
    GeneratedPaper,
    PaperSection,
    ResearchContent,
    PaperTemplate,
    PaperStatus
)

# Autonomous SDLC Generation 1 components
try:
    from .autonomous_discovery_engine import (
        AutonomousDiscoveryEngine,
        HypothesisGenerator,
        DiscoveryMode,
        ResearchOpportunity,
        DiscoveryHypothesis
    )
    _autonomous_discovery_available = True
except ImportError as e:
    print(f"Warning: Autonomous discovery engine unavailable: {e}")
    _autonomous_discovery_available = False

__all__ = [
    'HypothesisEngine',
    'ExperimentDesigner', 
    'ResearchTracker',
    'PublicationPipeline',
    'HypothesisTestingEngine',
    'ResearchDrivenNotebook',
    'CollaborationEngine',
    'IntelligentPaperGenerator',
    'Hypothesis',
    'Evidence',
    'HypothesisStatus',
    'EvidenceType',
    'Collaborator',
    'Collaboration',
    'CollaborationType',
    'InteractionType',
    'CollaborationStatus',
    'GeneratedPaper',
    'PaperSection',
    'ResearchContent',
    'PaperTemplate',
    'PaperStatus'
]

# Add autonomous components if available
if _autonomous_discovery_available:
    __all__.extend([
        'AutonomousDiscoveryEngine',
        'HypothesisGenerator', 
        'DiscoveryMode',
        'ResearchOpportunity',
        'DiscoveryHypothesis'
    ])