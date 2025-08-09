"""
OpenAI API client for ChatGPT and GPT-4 models.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
import httpx

from .base_ai import BaseAI


class OpenAIClient(BaseAI):
    """OpenAI API client."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model, api_key, **kwargs)
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        } if api_key else {}
        
    async def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using OpenAI's chat completion."""
        if not self.api_key:
            self.logger.warning("No OpenAI API key provided - using fallback")
            return f"[OpenAI Mock] Generated text for: {prompt[:50]}..."
        
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **kwargs
                }
                
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["choices"][0]["message"]["content"]
                    tokens_used = result.get("usage", {}).get("total_tokens", 0)
                    
                    self._increment_usage(tokens_used)
                    return text
                else:
                    self.logger.error(f"OpenAI API error: {response.status_code}")
                    return f"[OpenAI Error {response.status_code}] Could not generate text."
                    
        except Exception as e:
            self.logger.error(f"OpenAI API exception: {e}")
            return f"[OpenAI Exception] {str(e)}"
    
    async def analyze_text(
        self, 
        text: str, 
        task: str = "summarize",
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze text using OpenAI."""
        if not self.api_key:
            # Fallback to mock analysis
            from .base_ai import MockAI
            mock_ai = MockAI()
            return await mock_ai.analyze_text(text, task, **kwargs)
        
        # Create task-specific prompts
        prompts = {
            "summarize": f"Summarize the following text concisely:\n\n{text}",
            "paper_summary": f"""
                Analyze this research paper and provide a structured summary in JSON format:
                {{
                    "abstract": "brief summary",
                    "contributions": ["contribution1", "contribution2"],
                    "methodology": "methodology description", 
                    "findings": ["finding1", "finding2"],
                    "limitations": ["limitation1", "limitation2"],
                    "impact": "potential impact description"
                }}
                
                Paper text:
                {text[:3000]}
            """,
            "writing_improvement": f"""
                Improve the following text for academic writing and provide analysis in JSON format:
                {{
                    "improved_text": "improved version",
                    "changes": ["change1", "change2"],
                    "clarity_score": 8,
                    "suggestions": ["suggestion1", "suggestion2"]
                }}
                
                Text to improve:
                {text}
            """,
            "experiment_design": f"""
                Design an experiment for this hypothesis in JSON format:
                {{
                    "approach": "experimental approach",
                    "variables": {{"independent": [], "dependent": [], "control": []}},
                    "methodology": ["step1", "step2"],
                    "sample_size": "recommended size",
                    "analysis_plan": ["analysis1", "analysis2"],
                    "challenges": ["challenge1", "challenge2"],
                    "timeline": "estimated timeline"
                }}
                
                Hypothesis: {text}
            """
        }
        
        prompt = prompts.get(task, f"Analyze the following text for {task}:\n\n{text}")
        response = await self.generate_text(prompt, **kwargs)
        
        # Try to parse JSON response for structured tasks
        if task in ["paper_summary", "writing_improvement", "experiment_design"]:
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback to text response
        return {"result": response, "task": task}
    
    async def embed_text(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """Generate embeddings for text."""
        if not self.api_key:
            # Return dummy embedding
            return [0.1] * 1536  # ada-002 embedding size
        
        try:
            async with httpx.AsyncClient() as client:
                data = {
                    "model": model,
                    "input": text
                }
                
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result["data"][0]["embedding"]
                    tokens_used = result.get("usage", {}).get("total_tokens", 0)
                    
                    self._increment_usage(tokens_used)
                    return embedding
                else:
                    self.logger.error(f"OpenAI embeddings error: {response.status_code}")
                    return [0.0] * 1536
                    
        except Exception as e:
            self.logger.error(f"OpenAI embeddings exception: {e}")
            return [0.0] * 1536