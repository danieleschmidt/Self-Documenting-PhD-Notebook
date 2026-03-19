"""
SmartAgent - simple rule-based agent for auto-tagging and linking.
No external AI API required for basic operation.
"""

from typing import Any, Dict, List, Optional
from .base import BaseAgent
from .literature_agent import LiteratureAgent
from ..core.note import Note


class SmartAgent(BaseAgent):
    """
    Rule-based smart agent for automated research note management.

    Provides basic auto-tagging and link suggestion without requiring
    an external AI API. Can be upgraded to use LLM by injecting an
    AI client via setup().
    """

    RESEARCH_KEYWORDS = {
        "#knowledge-graph": ["knowledge graph", "KG", "entity", "relation", "triple", "ontology"],
        "#nlp": ["nlp", "natural language", "language model", "bert", "transformer", "embedding"],
        "#machine-learning": ["machine learning", "deep learning", "neural network", "training", "model"],
        "#graph-neural-network": ["GNN", "graph neural", "graph convolutional", "message passing"],
        "#information-extraction": ["information extraction", "named entity", "NER", "relation extraction"],
        "#literature": ["et al", "arXiv", "doi:", "abstract", "paper", "survey", "review"],
        "#experiment": ["experiment", "dataset", "baseline", "result", "evaluation", "benchmark"],
        "#docgraph": ["DocGraph", "multi-document", "document graph", "cross-document"],
    }

    def __init__(self, name: str = "smart_agent", ai_client: Any = None, **kwargs):
        super().__init__(name, capabilities=["tagging", "linking", "analysis"], **kwargs)
        self._ai_client = ai_client

    async def process(self, input_data: Any, task_type: str = "tag", **kwargs) -> Any:
        """Async process dispatch — handles all research task types."""
        if task_type in ("analyze", "summarize", "improve", "extract", "design"):
            if self._ai_client:
                try:
                    result = await self._ai_client.generate(str(input_data), task_type=task_type)
                    return {"result": result, "task": task_type}
                except Exception:
                    pass
            return {"result": f"{task_type} result for: {str(input_data)[:80]}", "task": task_type}
        elif task_type == "generate":
            return [f"Generated item for: {str(input_data)[:80]}"]
        elif task_type in ("tag", "unknown"):
            return self._generate_tags_sync(str(input_data))
        else:
            return {"result": str(input_data), "task": task_type}

    async def _generate_tags(self, content: str) -> List[str]:
        """Generate tags for note content (keyword-based fallback)."""
        if self._ai_client:
            # TODO: use AI client when available
            pass
        return self._generate_tags_sync(content)

    def _generate_tags_sync(self, content: str) -> List[str]:
        """Keyword-based tag generation."""
        content_lower = content.lower()
        tags = []
        for tag, keywords in self.RESEARCH_KEYWORDS.items():
            if any(kw.lower() in content_lower for kw in keywords):
                tags.append(tag)
        return tags

    async def _suggest_links(self, note: Note) -> List[Dict[str, Any]]:
        """Suggest links to other notes (returns empty list without notebook context)."""
        if not self.notebook:
            return []
        suggestions = []
        all_notes = self.notebook.list_notes()
        note_words = set(note.content.lower().split())
        for other in all_notes:
            if other.title == note.title:
                continue
            other_words = set(other.content.lower().split())
            if not note_words or not other_words:
                continue
            overlap = len(note_words & other_words) / max(len(note_words | other_words), 1)
            if overlap > 0.05:
                suggestions.append({
                    "title": other.title,
                    "similarity": overlap,
                    "reason": "shared research keywords",
                })
        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        return suggestions[:5]

    async def _analyze(self, content: str, context: str = "") -> Dict[str, Any]:
        """Return a basic analysis dict (no external AI required)."""
        word_count = len(content.split())
        return {
            "analysis": f"Reviewed {word_count} words of {context} content.",
            "word_count": word_count,
        }
