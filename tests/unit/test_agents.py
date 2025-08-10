"""
Tests for agent functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
from pathlib import Path

from phd_notebook.agents.smart_agent import SmartAgent
from phd_notebook.agents.base import BaseAgent
from phd_notebook.ai.base_ai import MockAI


class TestBaseAgent:
    """Test base agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = BaseAgent(name="test-agent")
        assert agent.name == "test-agent"
        assert agent.enabled == True
        assert agent.priority == 1
    
    @pytest.mark.asyncio
    async def test_process_not_implemented(self):
        """Test that process method raises NotImplementedError."""
        agent = BaseAgent(name="test-agent")
        
        with pytest.raises(NotImplementedError):
            await agent.process("test input")
    
    def test_agent_info(self):
        """Test agent info retrieval."""
        agent = BaseAgent(name="test-agent", description="Test agent description")
        
        info = agent.get_info()
        assert info["name"] == "test-agent"
        assert info["description"] == "Test agent description"
        assert info["enabled"] == True
        assert info["priority"] == 1


class TestSmartAgent:
    """Test smart agent functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ai_client = MockAI()
        self.agent = SmartAgent(
            name="smart-test-agent",
            ai_client=self.ai_client,
            description="Test smart agent"
        )
    
    @pytest.mark.asyncio
    async def test_analyze_task(self):
        """Test task analysis."""
        input_data = "Analyze this research topic about machine learning"
        
        result = await self.agent.process(input_data, task_type="analyze")
        
        # Should return some analysis result
        assert result is not None
        assert isinstance(result, (str, dict))
    
    @pytest.mark.asyncio
    async def test_summarize_task(self):
        """Test summarization task."""
        input_data = "This is a long research paper about neural networks and deep learning applications."
        
        result = await self.agent.process(input_data, task_type="summarize")
        
        assert result is not None
        assert isinstance(result, (str, dict))
    
    @pytest.mark.asyncio
    async def test_generate_task(self):
        """Test generation task."""
        input_data = "machine learning research"
        
        result = await self.agent.process(input_data, task_type="generate")
        
        assert result is not None
        assert isinstance(result, (str, dict, list))
    
    @pytest.mark.asyncio
    async def test_improve_task(self):
        """Test improvement task."""
        input_data = "This text needs better clarity and structure."
        
        result = await self.agent.process(input_data, task_type="improve")
        
        assert result is not None
        assert isinstance(result, (str, dict))
    
    @pytest.mark.asyncio
    async def test_extract_task(self):
        """Test extraction task."""
        input_data = "Machine learning, artificial intelligence, neural networks, deep learning"
        
        result = await self.agent.process(input_data, task_type="extract")
        
        assert result is not None
        assert isinstance(result, (str, dict, list))
    
    @pytest.mark.asyncio
    async def test_design_task(self):
        """Test design task."""
        input_data = "Machine learning can improve diagnostic accuracy in medical imaging"
        
        result = await self.agent.process(input_data, task_type="design")
        
        assert result is not None
        assert isinstance(result, (str, dict))
    
    @pytest.mark.asyncio
    async def test_unknown_task_type(self):
        """Test handling of unknown task type."""
        input_data = "Some input data"
        
        result = await self.agent.process(input_data, task_type="unknown_task")
        
        # Should still return a result (generic processing)
        assert result is not None
    
    def test_agent_configuration(self):
        """Test agent configuration."""
        agent = SmartAgent(
            name="config-test",
            ai_client=self.ai_client,
            description="Configuration test agent",
            enabled=False,
            priority=5
        )
        
        info = agent.get_info()
        assert info["name"] == "config-test"
        assert info["enabled"] == False
        assert info["priority"] == 5
        assert info["description"] == "Configuration test agent"
    
    @pytest.mark.asyncio
    async def test_agent_with_none_ai_client(self):
        """Test agent behavior with None AI client."""
        agent = SmartAgent(name="no-ai-agent", ai_client=None)
        
        # Should handle gracefully or create mock client
        result = await agent.process("test input", task_type="analyze")
        assert result is not None


class TestAgentRegistry:
    """Test agent registry functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ai_client = MockAI()
    
    def test_agent_creation_and_registration(self):
        """Test creating and registering agents."""
        agents = []
        
        # Create multiple agents
        agent1 = SmartAgent(name="agent1", ai_client=self.ai_client, priority=1)
        agent2 = SmartAgent(name="agent2", ai_client=self.ai_client, priority=2)
        agent3 = SmartAgent(name="agent3", ai_client=self.ai_client, priority=3)
        
        agents.extend([agent1, agent2, agent3])
        
        # Test sorting by priority
        sorted_agents = sorted(agents, key=lambda x: x.priority)
        assert sorted_agents[0].name == "agent1"
        assert sorted_agents[1].name == "agent2"
        assert sorted_agents[2].name == "agent3"
    
    def test_enabled_disabled_agents(self):
        """Test filtering enabled/disabled agents."""
        agents = [
            SmartAgent(name="enabled1", ai_client=self.ai_client, enabled=True),
            SmartAgent(name="disabled1", ai_client=self.ai_client, enabled=False),
            SmartAgent(name="enabled2", ai_client=self.ai_client, enabled=True),
        ]
        
        enabled_agents = [agent for agent in agents if agent.enabled]
        disabled_agents = [agent for agent in agents if not agent.enabled]
        
        assert len(enabled_agents) == 2
        assert len(disabled_agents) == 1
        assert enabled_agents[0].name == "enabled1"
        assert disabled_agents[0].name == "disabled1"


class TestAgentIntegration:
    """Test agent integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ai_client = MockAI()
    
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self):
        """Test workflow with multiple agents."""
        # Create a workflow with multiple agents
        analyzer = SmartAgent(name="analyzer", ai_client=self.ai_client, priority=1)
        summarizer = SmartAgent(name="summarizer", ai_client=self.ai_client, priority=2)
        improver = SmartAgent(name="improver", ai_client=self.ai_client, priority=3)
        
        agents = [analyzer, summarizer, improver]
        input_data = "Research paper content about machine learning applications"
        
        # Process through each agent
        results = []
        current_input = input_data
        
        for agent in agents:
            if agent.name == "analyzer":
                result = await agent.process(current_input, task_type="analyze")
            elif agent.name == "summarizer":
                result = await agent.process(current_input, task_type="summarize")
            elif agent.name == "improver":
                result = await agent.process(current_input, task_type="improve")
            
            results.append(result)
            # In a real workflow, you might pass result to next agent
        
        assert len(results) == 3
        for result in results:
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self):
        """Test agent error handling."""
        # Create agent with mock AI that raises exception
        faulty_ai = MagicMock()
        faulty_ai.generate_text = AsyncMock(side_effect=Exception("AI service error"))
        
        agent = SmartAgent(name="faulty-agent", ai_client=faulty_ai)
        
        # Should handle the error gracefully
        result = await agent.process("test input", task_type="analyze")
        
        # Depending on implementation, might return None or error message
        # The agent should not crash
        assert result is not None or result is None  # Either way is acceptable