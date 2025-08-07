"""
SPARC Writing Agent for generating academic papers using the SPARC methodology.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import re

from .base import BaseAgent
from ..core.note import Note, NoteType


class SPARCAgent(BaseAgent):
    """
    AI agent for SPARC (Situation-Problem-Action-Result-Conclusion) writing methodology.
    
    Capabilities:
    - Paper structure generation
    - Academic writing assistance
    - Literature synthesis
    - Draft review and improvement
    """
    
    def __init__(self):
        super().__init__(
            name="SPARCAgent",
            capabilities=[
                'paper_generation',
                'structure_analysis',
                'academic_writing',
                'literature_synthesis',
                'draft_review'
            ]
        )
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process writing-related requests."""
        if isinstance(input_data, dict):
            if 'sparc_request' in input_data:
                return self.generate_sparc_paper(input_data)
            elif 'review_text' in input_data:
                return self.review_draft(input_data['review_text'])
        
        return input_data
    
    def generate_sparc_paper(self, paper_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a paper using SPARC methodology."""
        self.log_activity("Generating SPARC paper structure")
        
        topic = paper_spec.get('topic', 'Research Topic')
        research_notes = paper_spec.get('research_notes', [])
        target_venue = paper_spec.get('target_venue', 'Generic Conference')
        
        paper_structure = {
            'title': self._generate_title(topic, research_notes),
            'abstract': self._generate_abstract_structure(topic),
            'situation': self._analyze_situation(topic, research_notes),
            'problem': self._formulate_problem(topic, research_notes),
            'action': self._describe_actions(research_notes),
            'results': self._compile_results(research_notes),
            'conclusion': self._write_conclusion(topic, research_notes),
            'sections': self._generate_section_outline(target_venue),
            'writing_guidelines': self._get_writing_guidelines(target_venue)
        }
        
        return paper_structure
    
    def create_paper_draft(
        self,
        title: str,
        topic: str,
        research_notes: List[Note],
        target_venue: str = "Generic Conference"
    ) -> Note:
        """Create a paper draft note using SPARC methodology."""
        if not self.notebook:
            raise RuntimeError("Agent must be registered with a notebook")
        
        # Generate SPARC structure
        paper_spec = {
            'topic': topic,
            'research_notes': [note.content for note in research_notes],
            'target_venue': target_venue
        }
        
        structure = self.generate_sparc_paper(paper_spec)
        
        # Build paper content
        content = self._build_paper_content(structure)
        
        # Create note
        note = self.notebook.create_note(
            title=title,
            content=content,
            note_type=NoteType.PROJECT,
            tags=['#paper', '#draft', '#sparc', f'#{target_venue.lower().replace(" ", "_")}']
        )
        
        # Add metadata
        note.frontmatter.metadata.update({
            'paper_type': 'draft',
            'target_venue': target_venue,
            'topic': topic,
            'methodology': 'SPARC',
            'creation_date': datetime.now().isoformat(),
            'status': 'draft',
            'word_count': len(content.split())
        })
        
        if note.file_path:
            note.save()
        
        self.log_activity(f"Created SPARC paper draft: {title}")
        return note
    
    def review_draft(self, draft_text: str) -> Dict[str, Any]:
        """Review a paper draft and provide improvement suggestions."""
        self.log_activity("Reviewing paper draft")
        
        review = {
            'structure_analysis': self._analyze_structure(draft_text),
            'clarity_issues': self._identify_clarity_issues(draft_text),
            'coherence_check': self._check_coherence(draft_text),
            'argument_flow': self._analyze_argument_flow(draft_text),
            'suggestions': self._generate_suggestions(draft_text),
            'sparc_compliance': self._check_sparc_compliance(draft_text),
            'word_count': len(draft_text.split()),
            'readability_score': self._estimate_readability(draft_text)
        }
        
        return review
    
    def improve_draft(self, paper_note: Note) -> Note:
        """Improve a paper draft based on review."""
        review = self.review_draft(paper_note.content)
        
        # Add review as a new section
        review_content = self._build_review_content(review)
        paper_note.add_section("Review and Improvement Suggestions", review_content)
        
        # Update metadata
        paper_note.frontmatter.metadata.update({
            'last_reviewed': datetime.now().isoformat(),
            'review_score': review.get('readability_score', 0),
            'suggestions_count': len(review.get('suggestions', []))
        })
        
        if paper_note.file_path:
            paper_note.save()
        
        self.log_activity(f"Improved draft: {paper_note.title}")
        return paper_note
    
    def _generate_title(self, topic: str, research_notes: List[str]) -> str:
        """Generate a paper title."""
        # Simple title generation based on topic
        if "machine learning" in topic.lower():
            return f"A Novel Approach to {topic}: Methods and Applications"
        elif "analysis" in topic.lower():
            return f"Comprehensive {topic}: Insights and Implications"
        else:
            return f"Advances in {topic}: A Research Perspective"
    
    def _generate_abstract_structure(self, topic: str) -> Dict[str, str]:
        """Generate abstract structure template."""
        return {
            'background': f"Research in {topic} faces several challenges...",
            'objective': "This study aims to investigate...",
            'methods': "We employ a comprehensive approach involving...",
            'results': "Our findings demonstrate...",
            'conclusions': "These results contribute to the field by..."
        }
    
    def _analyze_situation(self, topic: str, research_notes: List[str]) -> Dict[str, Any]:
        """Analyze the research situation (SPARC: Situation)."""
        situation = {
            'field_overview': f"Current state of research in {topic}",
            'recent_advances': "Recent developments have shown...",
            'research_landscape': "The field is characterized by...",
            'key_players': "Major contributors include...",
            'trends': "Emerging trends indicate..."
        }
        
        # Extract situational context from research notes
        if research_notes:
            situation['evidence_from_literature'] = "Based on reviewed literature..."
            situation['research_gaps'] = "Analysis reveals gaps in..."
        
        return situation
    
    def _formulate_problem(self, topic: str, research_notes: List[str]) -> Dict[str, Any]:
        """Formulate the research problem (SPARC: Problem)."""
        problem = {
            'problem_statement': f"The primary challenge in {topic} is...",
            'research_questions': [
                f"How can we improve {topic}?",
                f"What factors influence {topic} outcomes?",
                f"What are the implications for practice?"
            ],
            'significance': "This problem is important because...",
            'scope': "This research focuses on...",
            'limitations': "The study is bounded by..."
        }
        
        return problem
    
    def _describe_actions(self, research_notes: List[str]) -> Dict[str, Any]:
        """Describe research actions (SPARC: Action)."""
        actions = {
            'methodology': "We employed a mixed-methods approach...",
            'data_collection': "Data was collected through...",
            'analysis_methods': "Analysis involved...",
            'tools_used': "Tools and software included...",
            'procedure': [
                "Literature review and gap analysis",
                "Research design and protocol development",
                "Data collection and preprocessing",
                "Statistical analysis and modeling",
                "Results validation and interpretation"
            ]
        }
        
        return actions
    
    def _compile_results(self, research_notes: List[str]) -> Dict[str, Any]:
        """Compile research results (SPARC: Results)."""
        results = {
            'key_findings': "The main findings include...",
            'statistical_results': "Statistical analysis revealed...",
            'qualitative_insights': "Qualitative analysis showed...",
            'data_summary': "The data demonstrates...",
            'validation': "Results were validated through..."
        }
        
        return results
    
    def _write_conclusion(self, topic: str, research_notes: List[str]) -> Dict[str, Any]:
        """Write conclusions (SPARC: Conclusion)."""
        conclusion = {
            'summary': f"This research contributes to {topic} by...",
            'implications': "The findings have implications for...",
            'contributions': "Key contributions include...",
            'limitations': "This study has several limitations...",
            'future_work': "Future research should explore...",
            'practical_applications': "These results can be applied to..."
        }
        
        return conclusion
    
    def _generate_section_outline(self, target_venue: str) -> List[Dict[str, str]]:
        """Generate section outline based on venue."""
        # Standard academic paper structure
        sections = [
            {'title': 'Abstract', 'description': 'Concise summary of the research'},
            {'title': 'Introduction', 'description': 'Background and motivation (Situation)'},
            {'title': 'Problem Statement', 'description': 'Research problem and questions (Problem)'},
            {'title': 'Related Work', 'description': 'Literature review and positioning'},
            {'title': 'Methodology', 'description': 'Research approach and methods (Action)'},
            {'title': 'Results', 'description': 'Findings and analysis (Results)'},
            {'title': 'Discussion', 'description': 'Interpretation and implications'},
            {'title': 'Conclusion', 'description': 'Summary and future work (Conclusion)'},
            {'title': 'References', 'description': 'Citations and bibliography'}
        ]
        
        # Customize based on venue type
        if 'conference' in target_venue.lower():
            sections.insert(4, {'title': 'Experimental Setup', 'description': 'Detailed experimental configuration'})
        elif 'journal' in target_venue.lower():
            sections.insert(-2, {'title': 'Limitations', 'description': 'Study limitations and threats to validity'})
        
        return sections
    
    def _get_writing_guidelines(self, target_venue: str) -> Dict[str, str]:
        """Get writing guidelines for the target venue."""
        return {
            'tone': 'Formal academic writing',
            'person': 'Third person preferred, first person acceptable for contributions',
            'tense': 'Past tense for completed work, present for general facts',
            'structure': 'Follow SPARC methodology throughout',
            'citations': 'Use consistent citation style',
            'figures': 'Include clear, well-labeled figures and tables',
            'length': 'Follow venue-specific word/page limits'
        }
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze paper structure."""
        sections = self._extract_sections(text)
        
        analysis = {
            'section_count': len(sections),
            'has_abstract': 'abstract' in text.lower(),
            'has_introduction': 'introduction' in text.lower(),
            'has_methodology': any(term in text.lower() for term in ['method', 'approach', 'procedure']),
            'has_results': 'results' in text.lower(),
            'has_conclusion': 'conclusion' in text.lower(),
            'sections_present': sections,
            'structure_score': self._calculate_structure_score(sections)
        }
        
        return analysis
    
    def _identify_clarity_issues(self, text: str) -> List[str]:
        """Identify clarity issues in the text."""
        issues = []
        
        # Check for long sentences
        sentences = re.split(r'[.!?]', text)
        long_sentences = [s for s in sentences if len(s.split()) > 30]
        if long_sentences:
            issues.append(f"Found {len(long_sentences)} sentences with >30 words")
        
        # Check for passive voice (simplified)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(text.lower().count(word) for word in passive_indicators)
        if passive_count > len(text.split()) * 0.1:
            issues.append("High use of passive voice detected")
        
        # Check for jargon density
        if len(text.split()) > 0:
            unique_words = set(text.lower().split())
            if len(unique_words) / len(text.split()) < 0.4:
                issues.append("Low lexical diversity may indicate repetitive language")
        
        return issues
    
    def _check_coherence(self, text: str) -> Dict[str, Any]:
        """Check text coherence."""
        sections = self._extract_sections(text)
        
        coherence = {
            'transition_words': self._count_transition_words(text),
            'section_flow': len(sections) > 0,
            'logical_structure': 'introduction' in text.lower() and 'conclusion' in text.lower(),
            'coherence_score': 0.7  # Simplified score
        }
        
        return coherence
    
    def _analyze_argument_flow(self, text: str) -> Dict[str, Any]:
        """Analyze argument flow."""
        return {
            'has_thesis': 'argue' in text.lower() or 'propose' in text.lower(),
            'evidence_support': 'evidence' in text.lower() or 'data' in text.lower(),
            'logical_progression': True,  # Simplified
            'argument_strength': 'moderate'
        }
    
    def _generate_suggestions(self, text: str) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if len(text.split()) < 1000:
            suggestions.append("Consider expanding the content for a more comprehensive treatment")
        
        if 'abstract' not in text.lower():
            suggestions.append("Add an abstract section")
        
        if text.lower().count('figure') + text.lower().count('table') == 0:
            suggestions.append("Consider adding figures or tables to support your arguments")
        
        if text.lower().count('reference') + text.lower().count('citation') < 5:
            suggestions.append("Increase the number of citations to support your claims")
        
        return suggestions
    
    def _check_sparc_compliance(self, text: str) -> Dict[str, Any]:
        """Check compliance with SPARC methodology."""
        compliance = {
            'situation_addressed': 'background' in text.lower() or 'context' in text.lower(),
            'problem_defined': 'problem' in text.lower() or 'challenge' in text.lower(),
            'action_described': 'method' in text.lower() or 'approach' in text.lower(),
            'results_presented': 'results' in text.lower() or 'findings' in text.lower(),
            'conclusion_drawn': 'conclusion' in text.lower() or 'summary' in text.lower(),
            'sparc_score': 0.8  # Simplified score
        }
        
        return compliance
    
    def _estimate_readability(self, text: str) -> float:
        """Estimate readability score (simplified)."""
        if not text.strip():
            return 0.0
        
        words = text.split()
        sentences = re.split(r'[.!?]', text)
        sentences = [s for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability estimate (higher is better, max 10)
        readability = max(0, min(10, 10 - (avg_sentence_length - 15) / 5))
        return round(readability, 1)
    
    def _extract_sections(self, text: str) -> List[str]:
        """Extract section headers from text."""
        # Look for markdown headers
        headers = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
        return [h.strip() for h in headers]
    
    def _calculate_structure_score(self, sections: List[str]) -> float:
        """Calculate structure quality score."""
        required_sections = ['introduction', 'method', 'result', 'conclusion']
        present_sections = [s.lower() for s in sections]
        
        matches = sum(1 for req in required_sections if any(req in sec for sec in present_sections))
        return matches / len(required_sections)
    
    def _count_transition_words(self, text: str) -> int:
        """Count transition words in text."""
        transitions = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'nevertheless', 'thus', 'hence', 'accordingly'
        ]
        
        text_lower = text.lower()
        return sum(text_lower.count(word) for word in transitions)
    
    def _build_paper_content(self, structure: Dict[str, Any]) -> str:
        """Build paper content from SPARC structure."""
        content = f"# {structure['title']}\n\n"
        
        # Abstract
        content += "## Abstract\n\n"
        abstract = structure['abstract']
        for key, value in abstract.items():
            content += f"**{key.title()}**: {value}\n"
        content += "\n"
        
        # Introduction (Situation)
        content += "## Introduction\n\n"
        situation = structure['situation']
        content += f"{situation['field_overview']}\n\n"
        content += f"{situation['recent_advances']}\n\n"
        
        # Problem Statement (Problem)
        content += "## Problem Statement\n\n"
        problem = structure['problem']
        content += f"{problem['problem_statement']}\n\n"
        content += "### Research Questions\n"
        for i, question in enumerate(problem['research_questions'], 1):
            content += f"{i}. {question}\n"
        content += "\n"
        
        # Methodology (Action)
        content += "## Methodology\n\n"
        action = structure['action']
        content += f"{action['methodology']}\n\n"
        content += "### Procedure\n"
        for i, step in enumerate(action['procedure'], 1):
            content += f"{i}. {step}\n"
        content += "\n"
        
        # Results
        content += "## Results\n\n"
        results = structure['results']
        content += f"{results['key_findings']}\n\n"
        content += f"{results['statistical_results']}\n\n"
        
        # Discussion and Conclusion
        content += "## Discussion and Conclusion\n\n"
        conclusion = structure['conclusion']
        content += f"{conclusion['summary']}\n\n"
        content += f"### Implications\n{conclusion['implications']}\n\n"
        content += f"### Limitations\n{conclusion['limitations']}\n\n"
        content += f"### Future Work\n{conclusion['future_work']}\n\n"
        
        # References
        content += "## References\n\n"
        content += "*[References will be added here]*\n\n"
        
        return content
    
    def _build_review_content(self, review: Dict[str, Any]) -> str:
        """Build review content from analysis."""
        content = ""
        
        # Overall Assessment
        content += f"**Word Count**: {review['word_count']}\n"
        content += f"**Readability Score**: {review['readability_score']}/10\n"
        content += f"**SPARC Compliance**: {review['sparc_compliance']['sparc_score']:.1%}\n\n"
        
        # Structure Analysis
        structure = review['structure_analysis']
        content += "### Structure Analysis\n"
        content += f"- Sections present: {structure['section_count']}\n"
        content += f"- Structure score: {structure['structure_score']:.1%}\n\n"
        
        # Issues and Suggestions
        if review['clarity_issues']:
            content += "### Clarity Issues\n"
            for issue in review['clarity_issues']:
                content += f"- {issue}\n"
            content += "\n"
        
        content += "### Improvement Suggestions\n"
        for suggestion in review['suggestions']:
            content += f"- {suggestion}\n"
        content += "\n"
        
        return content