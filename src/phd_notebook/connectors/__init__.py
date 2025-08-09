"""Data connectors for external integrations."""

from .base import DataConnector
from .academic_connectors import *
from .simple_connectors import *

__all__ = [
    'DataConnector',
    # Academic connectors
    'ArXivConnector', 'PubMedConnector', 'SemanticScholarConnector', 'GoogleScholarConnector',
    'ZoteroConnector', 'MendeleyConnector', 'CrossRefConnector',
    # Simple connectors for testing
    'SimpleSlackConnector', 'SimpleGoogleDriveConnector'
]