"""
Anthropic Claude API client.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
import httpx

from .base_ai import BaseAI


class AnthropicClient(BaseAI):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.base_url = "https://api.anthropic.com"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        } if api_key else {}
        
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using Claude."""
        if not self.api_key:
            self.logger.warning("No Anthropic API key provided - using fallback")
            return f"[Claude Mock] Generated text for: {prompt[:50]}..."
        
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    **kwargs
                }
                
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers=self.headers,
                    json=data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["content"][0]["text"]
                    tokens_used = result.get("usage", {}).get("output_tokens", 0)
                    
                    self._increment_usage(tokens_used)
                    return text
                else:
                    self.logger.error(f"Anthropic API error: {response.status_code}")
                    return f"[Claude Error {response.status_code}] Could not generate text."
                    
        except Exception as e:
            self.logger.error(f"Anthropic API exception: {e}")
            return f"[Claude Exception] {str(e)}"
    
    async def analyze_text(
        self, 
        text: str, 
        task: str = "summarize",
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze text using Claude."""
        if not self.api_key:
            # Fallback to mock analysis
            from .base_ai import MockAI
            mock_ai = MockAI()
            return await mock_ai.analyze_text(text, task, **kwargs)
        
        # Claude-specific prompts (more detailed than OpenAI)
        prompts = {
            "summarize": f"""
                Please provide a concise summary of the following text:
                
                {text}
                
                Focus on the main points and key insights.
            """,
            "paper_summary": f"""
                Please analyze this research paper and provide a detailed structured analysis.
                
                Return your analysis in the following JSON format:
                {{
                    "abstract": "2-3 sentence summary of the paper's main contribution",
                    "contributions": ["specific contribution 1", "specific contribution 2", ...],
                    "methodology": "description of the research methods used",
                    "findings": ["key finding 1", "key finding 2", ...],
                    "limitations": ["limitation 1", "limitation 2", ...],
                    "impact": "assessment of the paper's potential impact on the field"
                }}
                
                Paper content:
                {text[:4000]}
                
                Please ensure your response is valid JSON.
            """,
            "writing_improvement": f"""
                Please improve the following text for academic writing standards.
                
                Return your analysis in JSON format:
                {{
                    "improved_text": "your improved version of the text",
                    "changes": ["description of change 1", "description of change 2", ...],
                    "clarity_score": numeric_score_from_1_to_10,
                    "suggestions": ["additional suggestion 1", "additional suggestion 2", ...]
                }}
                
                Text to improve:
                {text}
                
                Focus on clarity, conciseness, and academic tone.
            """,
            "experiment_design": f"""
                Based on the following research hypothesis, please design a comprehensive experiment.
                
                Return your design in JSON format:
                {{
                    "approach": "overall experimental approach and design type",
                    "variables": {{
                        "independent": ["variable 1", "variable 2", ...],
                        "dependent": ["outcome measure 1", "outcome measure 2", ...],
                        "control": ["control variable 1", "control variable 2", ...]
                    }},
                    "methodology": ["step 1", "step 2", "step 3", ...],
                    "sample_size": "recommended sample size with justification",
                    "analysis_plan": ["analysis method 1", "analysis method 2", ...],
                    "challenges": ["potential challenge 1", "potential challenge 2", ...],
                    "timeline": "estimated timeline with phases"
                }}
                
                Hypothesis: {text}
                
                Please provide a thorough and practical experimental design.
            """
        }
        
        prompt = prompts.get(task, f"""
            Please analyze the following text for the task: {task}
            
            Text:
            {text}
            
            Please provide a thorough analysis.
        """)
        
        response = await self.generate_text(prompt, **kwargs)
        
        # Try to parse JSON response for structured tasks
        if task in ["paper_summary", "writing_improvement", "experiment_design"]:
            try:
                # Claude often includes explanatory text before/after JSON
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    return json.loads(json_str)
            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse JSON response for {task}")
        
        # Fallback to text response
        return {"result": response, "task": task}
    
    async def critique_writing(self, text: str, focus: str = "academic") -> Dict[str, Any]:
        """Provide detailed writing critique - Claude's strength."""
        prompt = f"""
        Please provide a detailed critique of the following {focus} writing:
        
        {text}
        
        Please analyze:
        1. Clarity and coherence
        2. Argument structure and logic
        3. Writing style and tone
        4. Specific areas for improvement
        5. Strengths to maintain
        
        Provide constructive feedback that will help improve the writing quality.
        """
        
        response = await self.generate_text(prompt, max_tokens=1500)
        
        return {
            "critique": response,
            "focus": focus,
            "text_length": len(text),
            "generated_at": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()