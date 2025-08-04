"""
Base class for AI agents in the PhD notebook system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..core.note import Note


class BaseAgent(ABC):
    """
    Base class for all AI agents in the PhD notebook system.
    
    Agents provide specialized functionality for:
    - Data ingestion and processing
    - Content generation and analysis
    - Research workflow automation
    """
    
    def __init__(self, name: str, capabilities: List[str] = None):
        self.name = name
        self.capabilities = capabilities or []
        self.notebook = None  # Will be set when registered
        self.config: Dict[str, Any] = {}
        self.is_active = True
        
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process input data and return results."""
        pass
    
    def setup(self, config: Dict[str, Any]) -> None:
        """Configure the agent with settings."""
        self.config.update(config)
    
    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle a specific task type."""
        return task_type in self.capabilities
    
    def log_activity(self, activity: str, metadata: Dict = None) -> None:
        """Log agent activity."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "activity": activity,
            "metadata": metadata or {}
        }
        
        # In a full implementation, this would write to a log file
        print(f"[{self.name}] {activity}")
    
    def create_note(self, title: str, content: str, **kwargs) -> Note:
        """Create a note through the notebook interface."""
        if not self.notebook:
            raise RuntimeError("Agent must be registered with a notebook")
        
        return self.notebook.create_note(title=title, content=content, **kwargs)
    
    def get_context(self) -> Dict[str, Any]:
        """Get current research context from notebook."""
        if not self.notebook:
            return {}
        
        return self.notebook.get_research_context()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', capabilities={self.capabilities})"


class SimpleAgent(BaseAgent):
    """
    Simple implementation of BaseAgent for basic tasks.
    """
    
    def __init__(self, name: str, process_function=None, **kwargs):
        super().__init__(name, **kwargs)
        self._process_function = process_function
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process using the provided function."""
        if self._process_function:
            return self._process_function(input_data, **kwargs)
        else:
            return input_data