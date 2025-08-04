"""Core functionality for the PhD notebook system."""

from .notebook import ResearchNotebook
from .note import Note
from .vault_manager import VaultManager
from .knowledge_graph import KnowledgeGraph

__all__ = [
    "ResearchNotebook",
    "Note", 
    "VaultManager",
    "KnowledgeGraph",
]