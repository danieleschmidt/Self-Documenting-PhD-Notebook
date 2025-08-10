"""
Tests for AI client functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from phd_notebook.ai.client_factory import AIClientFactory
from phd_notebook.ai.base_ai import MockAI


class TestAIClientFactory:
    """Test AI client factory."""
    
    def test_mock_client_creation(self):
        """Test creating mock AI client when no providers available."""
        client = AIClientFactory.get_client("mock")
        assert isinstance(client, MockAI)
        assert client.model_name == "mock-model"
    
    def test_auto_client_detection(self):
        """Test automatic client detection falls back to mock."""
        client = AIClientFactory.get_client("auto")
        assert isinstance(client, MockAI)
    
    def test_available_providers(self):
        """Test getting available providers."""
        providers = AIClientFactory.get_available_providers()
        assert isinstance(providers, list)
        assert "mock" in providers


class TestMockAI:
    """Test Mock AI implementation."""
    
    @pytest.mark.asyncio
    async def test_generate_text(self):
        """Test text generation."""
        ai = MockAI()
        
        # Test research questions
        result = await ai.generate_text("Generate research questions about AI")
        assert "research question" in result.lower()
        assert "?" in result
        
        # Test paper summary
        result = await ai.generate_text("Summarize this paper")
        assert "paper" in result.lower()
        
        # Test writing improvement
        result = await ai.generate_text("Improve this writing")
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_text(self):
        """Test text analysis."""
        ai = MockAI()
        
        # Test paper summary
        result = await ai.analyze_text("Paper content", task="paper_summary")
        assert "abstract" in result
        assert "contributions" in result
        assert isinstance(result["contributions"], list)
        
        # Test writing improvement
        result = await ai.analyze_text("Text to improve", task="writing_improvement")
        assert "improved_text" in result
        assert "clarity_score" in result
        assert isinstance(result["clarity_score"], int)
        
        # Test experiment design
        result = await ai.analyze_text("Hypothesis text", task="experiment_design")
        assert "approach" in result
        assert "variables" in result
        assert "methodology" in result
    
    @pytest.mark.asyncio
    async def test_summarize_paper(self):
        """Test paper summarization."""
        ai = MockAI()
        paper_content = "This is a research paper about machine learning applications in healthcare."
        
        result = await ai.summarize_paper(paper_content)
        assert result["summary_type"] == "paper_analysis"
        assert "abstract" in result
        assert "contributions" in result
        assert "generated_at" in result
    
    @pytest.mark.asyncio
    async def test_generate_research_questions(self):
        """Test research question generation."""
        ai = MockAI()
        
        questions = await ai.generate_research_questions(
            topic="machine learning",
            context="healthcare applications",
            num_questions=3
        )
        
        assert isinstance(questions, list)
        assert len(questions) <= 3
        for q in questions:
            assert "?" in q
    
    @pytest.mark.asyncio
    async def test_improve_writing(self):
        """Test writing improvement."""
        ai = MockAI()
        original_text = "This text needs improvement for clarity and style."
        
        result = await ai.improve_writing(original_text, style="academic")
        
        assert result["original"] == original_text
        assert "improved" in result
        assert "changes" in result
        assert "clarity_score" in result
        assert "suggestions" in result
        assert result["style"] == "academic"
    
    @pytest.mark.asyncio
    async def test_extract_keywords(self):
        """Test keyword extraction."""
        ai = MockAI()
        text = "Machine learning and artificial intelligence are transforming healthcare through advanced algorithms."
        
        keywords = await ai.extract_keywords(text, max_keywords=5)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        for keyword in keywords:
            assert isinstance(keyword, str)
            assert len(keyword.strip()) > 0
    
    @pytest.mark.asyncio
    async def test_generate_experiment_design(self):
        """Test experiment design generation."""
        ai = MockAI()
        hypothesis = "Machine learning models can improve diagnostic accuracy"
        
        result = await ai.generate_experiment_design(hypothesis, field="healthcare")
        
        assert result["hypothesis"] == hypothesis
        assert result["field"] == "healthcare"
        assert "approach" in result
        assert "variables" in result
        assert "methodology" in result
        assert "timeline" in result
    
    def test_usage_stats(self):
        """Test usage statistics tracking."""
        ai = MockAI()
        
        # Initial stats
        stats = ai.get_usage_stats()
        assert stats["requests"] == 0
        assert stats["total_tokens"] == 0
        assert stats["model"] == "mock-model"
        
        # Increment usage
        ai._increment_usage(tokens=100)
        stats = ai.get_usage_stats()
        assert stats["requests"] == 1
        assert stats["total_tokens"] == 100