"""AI agents for research automation."""

from .base import BaseAgent, SimpleAgent
from .literature_agent import LiteratureAgent
from .experiment_agent import ExperimentAgent
from .sparc_agent import SPARCAgent

__all__ = [
    "BaseAgent", 
    "SimpleAgent", 
    "LiteratureAgent", 
    "ExperimentAgent", 
    "SPARCAgent"
]