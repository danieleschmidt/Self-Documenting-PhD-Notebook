"""Utility functions and classes."""

from .logging import setup_logging, get_logger
from .simple_validation import validate_note_data, ValidationError

try:
    from .security import sanitize_input, SecurityError
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

if SECURITY_AVAILABLE:
    __all__ = [
        "setup_logging",
        "get_logger", 
        "validate_note_data",
        "ValidationError",
        "sanitize_input",
        "SecurityError",
    ]
else:
    __all__ = [
        "setup_logging",
        "get_logger", 
        "validate_note_data",
        "ValidationError",
    ]