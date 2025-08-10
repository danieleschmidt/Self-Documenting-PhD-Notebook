"""
Smart Agent - AI-powered agent for research automation.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import BaseAgent
from ..ai.client_factory import AIClientFactory
from ..core.note import Note, NoteType


class SmartAgent(BaseAgent):
    """AI-powered agent for intelligent research tasks."""
    
    def __init__(self, name: str, ai_provider: str = "auto", capabilities: List[str] = None):
        default_capabilities = [
            'summarization', 'analysis', 'content_generation', 
            'tagging', 'linking', 'insight_extraction'
        ]
        super().__init__(name, capabilities or default_capabilities)
        
        self.ai_client = AIClientFactory.get_client(ai_provider)
        
    async def process(self, input_data: Any, **kwargs) -> Any:
        """Process data using AI capabilities."""
        task_type = kwargs.get('task_type', 'analyze')
        
        if task_type == 'summarize':
            return await self._summarize(input_data, **kwargs)
        elif task_type == 'analyze':
            return await self._analyze(input_data, **kwargs)
        elif task_type == 'generate_tags':
            return await self._generate_tags(input_data, **kwargs)
        elif task_type == 'suggest_links':
            return await self._suggest_links(input_data, **kwargs)
        else:
            return await self.ai_client.generate(f"Process: {input_data}", **kwargs)
    
    async def _summarize(self, text: str, max_length: int = 200) -> str:
        """Generate intelligent summary."""
        if not text or len(text) < 50:
            return text
        
        try:
            prompt = f"Summarize this text in {max_length} characters or less: {text}"
            summary = await self.ai_client.generate_text(prompt, max_tokens=max_length//4)
            self.log_activity(f"Summarized {len(text)} chars to {len(summary)} chars")
            return summary
        except Exception as e:
            self.log_activity(f"Summarization failed: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    async def _analyze(self, content: str, analysis_type: str = "research") -> Dict[str, Any]:
        """Perform content analysis."""
        try:
            result = await self.ai_client.analyze_text(content, task=analysis_type)
            
            # Enhance with basic metrics
            result.update({
                "word_count": len(content.split()),
                "estimated_reading_time": len(content.split()) // 200,  # ~200 WPM
            })
            
            self.log_activity(f"Analyzed content ({analysis_type})")
            return result
            
        except Exception as e:
            self.log_activity(f"Analysis failed: {e}")
            return {"error": str(e), "analysis": "Analysis unavailable"}
    
    async def _generate_tags(self, content: str, max_tags: int = 5) -> List[str]:
        """Generate relevant tags for content."""
        try:
            prompt = f"Generate {max_tags} relevant research tags for this content: {content[:500]}"
            response = await self.ai_client.generate_text(prompt, max_tokens=50)
            
            # Extract tags from response (simple parsing)
            tags = []
            for line in response.split('\n'):
                if line.strip().startswith(('#', '-', '*')):
                    tag = line.strip().lstrip('#-*').strip()
                    if tag and len(tags) < max_tags:
                        tags.append(f"#{tag.lower().replace(' ', '_')}")
            
            # Fallback: extract words as tags if no formatted tags found
            if not tags:
                words = response.replace(',', ' ').split()
                for word in words:
                    if word.isalpha() and len(word) > 2 and len(tags) < max_tags:
                        tags.append(f"#{word.lower()}")
            
            self.log_activity(f"Generated {len(tags)} tags")
            return tags[:max_tags] if tags else ["#research", "#auto_tagged"]
            
        except Exception as e:
            self.log_activity(f"Tag generation failed: {e}")
            return ["#research", "#auto_tagged"]
    
    async def _suggest_links(self, note: Note, max_suggestions: int = 3) -> List[Dict[str, str]]:
        """Suggest links to related notes."""
        if not self.notebook:
            return []
        
        try:
            # Get other notes for comparison
            all_notes = self.notebook.list_notes()
            suggestions = []
            
            for other_note in all_notes:
                if other_note.title == note.title:
                    continue
                
                # Simple similarity check (can be enhanced with embeddings)
                common_tags = set(note.frontmatter.tags) & set(other_note.frontmatter.tags)
                if common_tags and len(suggestions) < max_suggestions:
                    suggestions.append({
                        "title": other_note.title,
                        "reason": f"Shares tags: {', '.join(common_tags)}",
                        "similarity": len(common_tags) / len(set(note.frontmatter.tags) | set(other_note.frontmatter.tags))
                    })
            
            self.log_activity(f"Suggested {len(suggestions)} links")
            return sorted(suggestions, key=lambda x: x["similarity"], reverse=True)
            
        except Exception as e:
            self.log_activity(f"Link suggestion failed: {e}")
            return []
    
    def process_note(self, note: Note, tasks: List[str] = None) -> Dict[str, Any]:
        """Process a note with multiple AI tasks."""
        if not tasks:
            tasks = ['summarize', 'analyze', 'generate_tags']
        
        results = {}
        
        # Run tasks asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for task in tasks:
                if task == 'summarize':
                    results['summary'] = loop.run_until_complete(
                        self._summarize(note.content)
                    )
                elif task == 'analyze':
                    results['analysis'] = loop.run_until_complete(
                        self._analyze(note.content)
                    )
                elif task == 'generate_tags':
                    results['suggested_tags'] = loop.run_until_complete(
                        self._generate_tags(note.content)
                    )
                elif task == 'suggest_links':
                    results['suggested_links'] = loop.run_until_complete(
                        self._suggest_links(note)
                    )
        finally:
            loop.close()
        
        return results


class LiteratureAgent(SmartAgent):
    """Specialized agent for literature review and paper processing."""
    
    def __init__(self, ai_provider: str = "auto"):
        super().__init__(
            name="Literature Agent",
            ai_provider=ai_provider,
            capabilities=['paper_summary', 'key_extraction', 'citation_analysis']
        )
    
    async def process_paper(self, paper_content: str, metadata: Dict = None) -> Dict[str, Any]:
        """Process academic paper with specialized analysis."""
        results = {
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Extract key contributions
            contrib_prompt = f"Extract 3-5 key contributions from this paper: {paper_content[:1000]}"
            results['contributions'] = await self.ai_client.generate_text(contrib_prompt, max_tokens=200)
            
            # Extract methodology
            method_prompt = f"Summarize the methodology used in this paper: {paper_content[:1000]}"
            results['methodology'] = await self.ai_client.generate_text(method_prompt, max_tokens=150)
            
            # Generate research questions
            rq_prompt = f"What research questions does this paper address?: {paper_content[:800]}"
            results['research_questions'] = await self.ai_client.generate_text(rq_prompt, max_tokens=100)
            
            self.log_activity(f"Processed paper: {metadata.get('title', 'Unknown')}")
            
        except Exception as e:
            self.log_activity(f"Paper processing failed: {e}")
            results['error'] = str(e)
        
        return results