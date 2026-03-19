"""
SPARCAgent - generates structured arXiv/paper draft outlines from research notes.

SPARC = Situation, Problem, Approach, Results, Conclusion
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from .base import BaseAgent
from ..core.note import Note, NoteType


class SPARCAgent(BaseAgent):
    """
    Agent that generates structured paper drafts from research notes.

    Uses the SPARC writing framework:
      - Situation:    Research context and motivation
      - Problem:      Gap being addressed
      - Approach:     Methodology and system design
      - Results:      Experimental findings
      - Conclusion:   Contributions and future work

    Usage::

        agent = SPARCAgent()
        notebook.register_agent(agent)
        draft = agent.create_paper_draft(
            title="DocGraph: Multi-Document Knowledge Graph Construction",
            topic="knowledge graph construction",
            research_notes=[lit_note, exp_note],
            target_venue="EMNLP 2026",
        )
        draft.save(vault_path / "drafts" / "docgraph.md")
    """

    def __init__(self, name: str = "SPARCAgent", **kwargs):
        super().__init__(
            name=name,
            capabilities=["paper_draft", "outline", "abstract"],
            description="Generates structured paper drafts using SPARC methodology",
            **kwargs,
        )

    def process(self, input_data: Any, **kwargs) -> Any:
        task = kwargs.get("task", "draft")
        if task == "draft" and isinstance(input_data, dict):
            return self.create_paper_draft(**input_data)
        return {}

    def create_paper_draft(
        self,
        title: str,
        topic: str,
        research_notes: Optional[List[Note]] = None,
        target_venue: str = "",
        authors: str = "",
        abstract: str = "",
    ) -> Note:
        """
        Create a structured paper draft note from research notes.

        Args:
            title: Paper title
            topic: Research topic / keywords
            research_notes: List of Note objects to draw from
            target_venue: Target conference/journal
            authors: Author list string
            abstract: Draft abstract (auto-generated placeholder if empty)

        Returns:
            Note object with SPARC-structured content
        """
        research_notes = research_notes or []
        now = datetime.now().strftime("%Y-%m-%d")

        # Collect related literature and experiment summaries
        lit_refs: List[str] = []
        exp_refs: List[str] = []
        for note in research_notes:
            if note.note_type == NoteType.LITERATURE:
                lit_refs.append(f"- [[{note.title}]]")
            elif note.note_type == NoteType.EXPERIMENT:
                exp_refs.append(f"- [[{note.title}]]")

        if not abstract:
            abstract = (
                f"We present {title}, addressing the problem of {topic}. "
                f"This work introduces a novel approach and evaluates it on "
                f"benchmark datasets. Results demonstrate significant improvements "
                f"over existing baselines."
            )

        content_parts = [
            f"# {title}",
            f"\n**Authors:** {authors or 'Daniel Schmidt'}",
            f"**Date:** {now}",
            f"**Target Venue:** {target_venue}" if target_venue else "",
            f"**Topic:** {topic}",
            "",
            "---",
            "",
            "## Abstract",
            "",
            abstract,
            "",
            "---",
            "",
            "## Introduction",
            "",
            f"<!-- Situation: motivate the research area ({topic}) -->",
            "",
            "The rapid growth of [research area] has created demand for ...",
            "",
            "## Problem Statement",
            "",
            "<!-- Problem: clearly state the gap or challenge -->",
            "",
            "Existing approaches suffer from the following limitations:",
            "1. ...",
            "2. ...",
            "",
            "## Related Work",
            "",
            "<!-- Cite relevant literature -->",
        ]

        if lit_refs:
            content_parts += lit_refs
        else:
            content_parts.append("<!-- Add literature review notes here -->")

        content_parts += [
            "",
            "## Methodology",
            "",
            "<!-- Approach: describe system/method design -->",
            "",
            "### System Overview",
            "",
            "```",
            "[ Input ] → [ Processing ] → [ Output ]",
            "```",
            "",
            "### Implementation Details",
            "",
            "...",
            "",
            "## Experiments",
            "",
            "<!-- Results: datasets, baselines, metrics -->",
        ]

        if exp_refs:
            content_parts += ["### Experiment Notes", ""] + exp_refs
        else:
            content_parts.append("<!-- Add experiment notes here -->")

        content_parts += [
            "",
            "### Results",
            "",
            "| Method | Dataset | Metric | Score |",
            "|--------|---------|--------|-------|",
            "| Ours   | -       | -      | -     |",
            "| Baseline| -      | -      | -     |",
            "",
            "## Conclusion",
            "",
            "<!-- Conclusion: contributions and future work -->",
            "",
            "In this paper we presented ...",
            "",
            "### Future Work",
            "",
            "- ...",
            "",
            "## References",
            "",
            "<!-- BibTeX entries or [[wiki links]] to literature notes -->",
        ]

        content = "\n".join(part for part in content_parts if part is not None)

        # If agent is registered with a notebook, create via notebook so it's tracked
        if self.notebook is not None:
            note = self.notebook.create_note(
                title=title,
                content=content,
                note_type=NoteType.PROJECT,
                tags=["#paper", "#sparc", "#draft"],
            )
        else:
            note = Note(
                title=title,
                content=content,
                note_type=NoteType.PROJECT,
            )
            note.frontmatter.tags = ["#paper", "#sparc", "#draft"]

        note.frontmatter.status = "draft"
        note.frontmatter.metadata.update({
            "topic": topic,
            "target_venue": target_venue,
            "authors": authors,
            "sparc_generated": True,
        })

        return note

    def generate_abstract(
        self,
        title: str,
        problem: str,
        method: str,
        results: str,
    ) -> str:
        """
        Generate a structured abstract.

        Args:
            title: Paper title
            problem: One-sentence problem statement
            method: One-sentence method description
            results: One-sentence results summary
        """
        return (
            f"We address the problem of {problem}. "
            f"To this end, we propose {method}. "
            f"Experimental evaluation shows that {results}."
        )
