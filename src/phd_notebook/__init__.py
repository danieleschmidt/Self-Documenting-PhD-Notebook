"""
Self-Documenting PhD Notebook

An Obsidian-compatible research notebook that automatically ingests lab data,
discussion threads, and LaTeX notes, then generates arXiv-ready drafts using
agentic SPARC pipelines.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .core.notebook import ResearchNotebook
from .core.note import Note, NoteType
from .core.vault_manager import VaultManager
from .connectors.base import DataConnector
from .agents.smart_agent import SmartAgent, LiteratureAgent
from .workflows.automation import WorkflowManager
from .ai.client_factory import AIClientFactory

__all__ = [
    "__version__",
    "ResearchNotebook", 
    "Note",
    "NoteType",
    "VaultManager",
    "DataConnector",
    "SmartAgent",
    "LiteratureAgent", 
    "WorkflowManager",
    "AIClientFactory",
]

# Convenience function for quick setup
def create_phd_workflow(
    field: str,
    subfield: str = None,
    institution: str = None,
    expected_duration: int = 5,
    **kwargs
) -> "ResearchNotebook":
    """Create a pre-configured PhD workflow for a specific field."""
    from .core.notebook import ResearchNotebook
    
    return ResearchNotebook(
        field=field,
        subfield=subfield,
        institution=institution,
        expected_duration=expected_duration,
        **kwargs
    )