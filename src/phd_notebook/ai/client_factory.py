"""
AI Client Factory - Simple client initialization and management.
"""

import os
from typing import Optional, Dict, Any
from .base_ai import BaseAI
from .anthropic_client import AnthropicClient  
from .openai_client import OpenAIClient


class AIClientFactory:
    """Factory for creating and managing AI clients."""
    
    _clients: Dict[str, BaseAI] = {}
    
    @classmethod
    def get_client(cls, provider: str = "auto", **kwargs) -> Optional[BaseAI]:
        """Get an AI client instance."""
        
        if provider == "auto":
            provider = cls._detect_available_provider()
        
        if provider not in cls._clients:
            cls._clients[provider] = cls._create_client(provider, **kwargs)
        
        return cls._clients[provider]
    
    @classmethod
    def _detect_available_provider(cls) -> str:
        """Detect which AI provider is available based on environment variables."""
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            return "openai" 
        else:
            return "mock"  # Fallback for development
    
    @classmethod
    def _create_client(cls, provider: str, **kwargs) -> BaseAI:
        """Create a client for the specified provider."""
        
        if provider == "anthropic":
            api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key required")
            return AnthropicClient(api_key=api_key)
        
        elif provider == "openai":
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            return OpenAIClient(api_key=api_key)
        
        elif provider == "mock":
            from .base_ai import MockAI
            return MockAI()
        
        else:
            raise ValueError(f"Unknown AI provider: {provider}")


class MockAIClient(BaseAI):
    """Mock AI client for development and testing."""
    
    def __init__(self):
        super().__init__("mock-model")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate mock response."""
        return f"Mock response to: {prompt[:50]}..."
    
    async def analyze_text(self, text: str, task: str = "summarize", **kwargs) -> Dict[str, Any]:
        """Mock text analysis."""
        return {
            "result": f"Mock analysis of {len(text.split())} words for task: {task}",
            "sentiment": "neutral",
            "confidence": 0.75
        }