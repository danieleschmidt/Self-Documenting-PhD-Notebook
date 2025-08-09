"""
Base AI client for language model integrations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime


class BaseAI(ABC):
    """Base class for AI model integrations."""
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.request_count = 0
        self.total_tokens = 0
        
    @abstractmethod
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text based on a prompt."""
        pass
    
    @abstractmethod
    async def analyze_text(
        self, 
        text: str, 
        task: str = "summarize",
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze text for various tasks."""
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'model': self.model_name,
            'requests': self.request_count,
            'total_tokens': self.total_tokens,
            'last_request': datetime.now().isoformat()
        }
    
    def _increment_usage(self, tokens: int = 0):
        """Track usage statistics."""
        self.request_count += 1
        self.total_tokens += tokens
    
    async def summarize_paper(self, paper_content: str, **kwargs) -> Dict[str, Any]:
        """Summarize a research paper."""
        prompt = f"""
        Please analyze the following research paper and provide a structured summary:
        
        Paper Content:
        {paper_content[:3000]}  # Limit content to avoid token limits
        
        Please provide:
        1. Abstract summary (2-3 sentences)
        2. Key contributions (3-5 bullet points)
        3. Methodology overview
        4. Main findings
        5. Limitations
        6. Potential impact
        
        Format your response as structured text.
        """
        
        result = await self.analyze_text(paper_content, task="paper_summary")
        
        # Parse structured response
        return {
            'summary_type': 'paper_analysis',
            'abstract': result.get('abstract', ''),
            'contributions': result.get('contributions', []),
            'methodology': result.get('methodology', ''),
            'findings': result.get('findings', []),
            'limitations': result.get('limitations', []),
            'impact': result.get('impact', ''),
            'generated_at': datetime.now().isoformat()
        }
    
    async def generate_research_questions(
        self, 
        topic: str, 
        context: str = "", 
        num_questions: int = 5
    ) -> List[str]:
        """Generate research questions for a topic."""
        prompt = f"""
        Generate {num_questions} novel research questions for the topic: {topic}
        
        Context: {context}
        
        Questions should be:
        - Specific and focused
        - Researchable
        - Significant to the field
        - Novel (not already well-studied)
        
        Format: Return only the questions, one per line.
        """
        
        response = await self.generate_text(prompt, max_tokens=500, temperature=0.8)
        questions = [q.strip() for q in response.split('\n') if q.strip() and '?' in q]
        return questions[:num_questions]
    
    async def improve_writing(self, text: str, style: str = "academic") -> Dict[str, Any]:
        """Improve writing style and clarity."""
        prompt = f"""
        Please improve the following text for {style} writing:
        
        Original text:
        {text}
        
        Provide:
        1. Improved version
        2. Key changes made
        3. Clarity score (1-10)
        4. Suggestions for further improvement
        """
        
        result = await self.analyze_text(text, task="writing_improvement")
        
        return {
            'original': text,
            'improved': result.get('improved_text', text),
            'changes': result.get('changes', []),
            'clarity_score': result.get('clarity_score', 5),
            'suggestions': result.get('suggestions', []),
            'style': style
        }
    
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        prompt = f"""
        Extract the {max_keywords} most important keywords from this text:
        
        {text[:2000]}
        
        Return only the keywords, separated by commas.
        Focus on:
        - Technical terms
        - Key concepts
        - Domain-specific terminology
        - Important entities
        """
        
        response = await self.generate_text(prompt, max_tokens=200, temperature=0.3)
        keywords = [k.strip() for k in response.split(',') if k.strip()]
        return keywords[:max_keywords]
    
    async def generate_experiment_design(
        self, 
        hypothesis: str, 
        field: str = "general"
    ) -> Dict[str, Any]:
        """Generate experimental design suggestions."""
        prompt = f"""
        Design an experiment to test this hypothesis in {field}:
        
        Hypothesis: {hypothesis}
        
        Provide:
        1. Experimental approach
        2. Variables (independent, dependent, control)
        3. Methodology steps
        4. Sample size recommendations
        5. Statistical analysis plan
        6. Potential challenges
        7. Timeline estimate
        """
        
        result = await self.analyze_text(hypothesis, task="experiment_design")
        
        return {
            'hypothesis': hypothesis,
            'field': field,
            'approach': result.get('approach', ''),
            'variables': result.get('variables', {}),
            'methodology': result.get('methodology', []),
            'sample_size': result.get('sample_size', 'TBD'),
            'analysis_plan': result.get('analysis_plan', []),
            'challenges': result.get('challenges', []),
            'timeline': result.get('timeline', 'TBD')
        }


class MockAI(BaseAI):
    """Mock AI implementation for testing without API keys."""
    
    def __init__(self, model_name: str = "mock-model", **kwargs):
        super().__init__(model_name, **kwargs)
        
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate mock text response."""
        self._increment_usage(tokens=max_tokens)
        
        # Simple mock responses based on prompt content
        if "research question" in prompt.lower():
            return "1. How can we improve model performance?\n2. What are the ethical implications?\n3. How does this scale to larger datasets?"
        elif "summarize" in prompt.lower() or "paper" in prompt.lower():
            return "This paper presents novel approaches to the research problem. The methodology involves comprehensive analysis. Key findings show significant improvements. Limitations include sample size constraints."
        elif "improve" in prompt.lower() and "writing" in prompt.lower():
            return "The improved text provides clearer structure and more precise language while maintaining the original meaning and intent."
        else:
            return f"Generated response for: {prompt[:50]}... [Mock AI Response]"
    
    async def analyze_text(
        self, 
        text: str, 
        task: str = "summarize",
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze text with mock responses."""
        self._increment_usage(tokens=500)
        
        if task == "paper_summary":
            return {
                'abstract': 'This paper addresses important research questions in the field.',
                'contributions': ['Novel methodology', 'Improved performance', 'Comprehensive evaluation'],
                'methodology': 'The authors employed a systematic approach with controlled experiments.',
                'findings': ['Significant improvement over baselines', 'Robust results across datasets'],
                'limitations': ['Limited sample size', 'Specific domain focus'],
                'impact': 'This work opens new directions for future research.'
            }
        elif task == "writing_improvement":
            return {
                'improved_text': f"[Improved] {text}",
                'changes': ['Enhanced clarity', 'Better structure', 'Improved flow'],
                'clarity_score': 8,
                'suggestions': ['Consider adding examples', 'Expand on key points']
            }
        elif task == "experiment_design":
            return {
                'approach': 'Controlled experimental design with treatment and control groups',
                'variables': {
                    'independent': ['Treatment condition'],
                    'dependent': ['Primary outcome measure'],
                    'control': ['Baseline condition']
                },
                'methodology': ['Design experiment', 'Recruit participants', 'Collect data', 'Analyze results'],
                'sample_size': 'N=100 (power analysis)',
                'analysis_plan': ['Descriptive statistics', 'Statistical significance testing'],
                'challenges': ['Recruitment', 'Measurement validity'],
                'timeline': '12 weeks'
            }
        
        return {'result': f'Mock analysis result for task: {task}'}