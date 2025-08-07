"""
Custom exceptions for the PhD Notebook system.
"""


class PhDNotebookError(Exception):
    """Base exception for PhD Notebook system."""
    pass


class VaultError(PhDNotebookError):
    """Raised when vault operations fail."""
    pass


class VaultNotFoundError(VaultError):
    """Raised when vault directory is not found."""
    pass


class VaultCorruptedError(VaultError):
    """Raised when vault structure is corrupted."""
    pass


class NoteError(PhDNotebookError):
    """Raised when note operations fail."""
    pass


class NoteNotFoundError(NoteError):
    """Raised when a note cannot be found."""
    pass


class InvalidNoteFormatError(NoteError):
    """Raised when note format is invalid."""
    pass


class NoteFrontmatterError(NoteError):
    """Raised when frontmatter parsing fails."""
    pass


class AgentError(PhDNotebookError):
    """Raised when agent operations fail."""
    pass


class AgentNotFoundError(AgentError):
    """Raised when requested agent is not registered."""
    pass


class AgentProcessingError(AgentError):
    """Raised when agent processing fails."""
    pass


class ConnectorError(PhDNotebookError):
    """Raised when data connector operations fail."""
    pass


class ConnectorConnectionError(ConnectorError):
    """Raised when connector cannot establish connection."""
    pass


class ConnectorDataError(ConnectorError):
    """Raised when connector data processing fails."""
    pass


class ValidationError(PhDNotebookError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(PhDNotebookError):
    """Raised when configuration is invalid."""
    pass


class SecurityError(PhDNotebookError):
    """Raised when security checks fail."""
    pass


class WorkflowError(PhDNotebookError):
    """Raised when workflow execution fails."""
    pass


class ExportError(PhDNotebookError):
    """Raised when export operations fail."""
    pass


class KnowledgeGraphError(PhDNotebookError):
    """Raised when knowledge graph operations fail."""
    pass