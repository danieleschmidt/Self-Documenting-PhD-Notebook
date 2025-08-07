"""
Literature Agent for processing research papers and academic content.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import re

from .base import BaseAgent
from ..core.note import Note, NoteType


class LiteratureAgent(BaseAgent):
    """
    AI agent for literature review and paper processing.
    
    Capabilities:
    - Paper summarization
    - Key contribution extraction
    - Citation network analysis
    - Research gap identification
    """
    
    def __init__(self):
        super().__init__(
            name="LiteratureAgent",
            capabilities=[
                'paper_summarization', 
                'citation_extraction',
                'keyword_extraction',
                'research_gap_analysis'
            ]
        )
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process literature content."""
        if isinstance(input_data, dict):
            if 'paper_content' in input_data:
                return self.analyze_paper(input_data['paper_content'])
            elif 'citation' in input_data:
                return self.parse_citation(input_data['citation'])
        
        return input_data
    
    def analyze_paper(self, paper_content: str) -> Dict[str, Any]:
        """Analyze a research paper's content."""
        self.log_activity("Analyzing paper content")
        
        analysis = {
            'abstract': self._extract_abstract(paper_content),
            'keywords': self._extract_keywords(paper_content),
            'methodology': self._extract_methodology(paper_content),
            'key_findings': self._extract_key_findings(paper_content),
            'limitations': self._identify_limitations(paper_content),
            'future_work': self._extract_future_work(paper_content),
            'citations': self._extract_citations(paper_content)
        }
        
        return analysis
    
    def create_literature_note(
        self, 
        title: str, 
        authors: str, 
        year: str,
        paper_content: str = "",
        doi: str = "",
        journal: str = ""
    ) -> Note:
        """Create a structured literature review note."""
        if not self.notebook:
            raise RuntimeError("Agent must be registered with a notebook")
        
        # Analyze paper content if provided
        analysis = {}
        if paper_content:
            analysis = self.analyze_paper(paper_content)
        
        # Build note content
        content = self._build_literature_content(title, authors, year, analysis, doi, journal)
        
        # Create note
        note = self.notebook.create_note(
            title=title,
            content=content,
            note_type=NoteType.LITERATURE,
            tags=['#literature', '#paper', f'#{year}']
        )
        
        # Add metadata
        note.frontmatter.metadata.update({
            'authors': authors,
            'year': year,
            'doi': doi,
            'journal': journal,
            'analysis_date': datetime.now().isoformat()
        })
        
        if note.file_path:
            note.save()
        
        self.log_activity(f"Created literature note: {title}")
        return note
    
    def suggest_related_papers(self, note: Note, limit: int = 5) -> List[Dict[str, Any]]:
        """Suggest papers related to a given note."""
        if not self.notebook:
            return []
        
        # Get all literature notes
        literature_notes = self.notebook.list_notes(note_type=NoteType.LITERATURE)
        
        suggestions = []
        current_tags = set(note.frontmatter.tags)
        current_keywords = self._extract_keywords(note.content)
        
        for lit_note in literature_notes:
            if lit_note.title == note.title:
                continue
            
            # Calculate similarity based on tags and keywords
            other_tags = set(lit_note.frontmatter.tags)
            other_keywords = self._extract_keywords(lit_note.content)
            
            tag_overlap = len(current_tags.intersection(other_tags))
            keyword_overlap = len(set(current_keywords).intersection(set(other_keywords)))
            
            if tag_overlap > 0 or keyword_overlap > 0:
                similarity = (tag_overlap * 0.6 + keyword_overlap * 0.4) / max(len(current_tags), 1)
                
                suggestions.append({
                    'title': lit_note.title,
                    'similarity': similarity,
                    'shared_tags': list(current_tags.intersection(other_tags)),
                    'shared_keywords': list(set(current_keywords).intersection(set(other_keywords)))
                })
        
        # Sort by similarity and return top results
        suggestions.sort(key=lambda x: x['similarity'], reverse=True)
        return suggestions[:limit]
    
    def _extract_abstract(self, content: str) -> str:
        """Extract abstract from paper content."""
        # Simple regex-based extraction
        abstract_match = re.search(r'(?i)abstract[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:keywords|introduction|1\.|\d+\.))', 
                                 content, re.DOTALL)
        if abstract_match:
            return abstract_match.group(1).strip()
        return ""
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from paper content."""
        # Simple keyword extraction based on frequency and common patterns
        keywords = []
        
        # Look for explicit keywords section
        keywords_match = re.search(r'(?i)keywords?[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:introduction|1\.|\d+\.))', 
                                 content, re.DOTALL)
        if keywords_match:
            keyword_text = keywords_match.group(1).strip()
            keywords.extend([k.strip() for k in re.split(r'[,;]', keyword_text) if k.strip()])
        
        # Extract common research terms (simplified)
        common_terms = [
            'machine learning', 'deep learning', 'neural networks', 'artificial intelligence',
            'natural language processing', 'computer vision', 'reinforcement learning',
            'algorithm', 'optimization', 'dataset', 'model', 'training', 'evaluation'
        ]
        
        content_lower = content.lower()
        for term in common_terms:
            if term in content_lower and term not in keywords:
                keywords.append(term)
        
        return keywords[:10]  # Return top 10 keywords
    
    def _extract_methodology(self, content: str) -> str:
        """Extract methodology section."""
        method_patterns = [
            r'(?i)methodology[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:results|experiments|evaluation|\d+\.))',
            r'(?i)methods?[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:results|experiments|evaluation|\d+\.))',
            r'(?i)approach[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:results|experiments|evaluation|\d+\.))'
        ]
        
        for pattern in method_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()[:500]  # Limit to 500 chars
        
        return ""
    
    def _extract_key_findings(self, content: str) -> List[str]:
        """Extract key findings from results section."""
        findings = []
        
        # Look for results section
        results_match = re.search(r'(?i)results[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:discussion|conclusion|\d+\.))', 
                                content, re.DOTALL)
        if results_match:
            results_text = results_match.group(1)
            
            # Look for sentences with key result indicators
            result_indicators = ['show', 'demonstrate', 'find', 'achieve', 'obtain', 'improve', 'outperform']
            sentences = re.split(r'[.!?]', results_text)
            
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in result_indicators):
                    findings.append(sentence.strip())
        
        return findings[:5]  # Return top 5 findings
    
    def _identify_limitations(self, content: str) -> List[str]:
        """Identify paper limitations."""
        limitations = []
        
        # Look for limitations section or mentions
        limitation_patterns = [
            r'(?i)limitations?[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:conclusion|future|references|\d+\.))',
            r'(?i)however[,\s]+(.+?)(?=[.!?])',
            r'(?i)despite[,\s]+(.+?)(?=[.!?])',
            r'(?i)although[,\s]+(.+?)(?=[.!?])'
        ]
        
        for pattern in limitation_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                limitation = match.group(1).strip()
                if len(limitation) > 20 and len(limitation) < 200:
                    limitations.append(limitation)
        
        return limitations[:3]  # Return top 3 limitations
    
    def _extract_future_work(self, content: str) -> List[str]:
        """Extract future work suggestions."""
        future_work = []
        
        # Look for future work section
        future_patterns = [
            r'(?i)future\s+work[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:references|acknowledgments|\d+\.))',
            r'(?i)future\s+research[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:references|acknowledgments|\d+\.))',
            r'(?i)next\s+steps[:\s]*(.+?)(?=\n\s*\n|\n\s*(?:references|acknowledgments|\d+\.))'
        ]
        
        for pattern in future_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                future_text = match.group(1).strip()
                sentences = re.split(r'[.!?]', future_text)
                future_work.extend([s.strip() for s in sentences if len(s.strip()) > 20])
        
        return future_work[:3]  # Return top 3 future work items
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations from content."""
        # Simple citation extraction (author, year format)
        citations = []
        
        # Look for in-text citations
        citation_patterns = [
            r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s+\d{4}\)',  # (Author, 2024)
            r'\[[A-Z][a-z]+(?:\s+et\s+al\.?)?,\s+\d{4}\]',  # [Author, 2024]
            r'\([A-Z][a-z]+(?:\s+&\s+[A-Z][a-z]+)?,\s+\d{4}\)'  # (Author & Author, 2024)
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, content)
            citations.extend([match.group(0) for match in matches])
        
        return list(set(citations))[:10]  # Return unique citations, max 10
    
    def parse_citation(self, citation: str) -> Dict[str, str]:
        """Parse a citation string into components."""
        # Simple citation parsing
        parsed = {
            'raw': citation,
            'authors': '',
            'year': '',
            'title': '',
            'journal': ''
        }
        
        # Try to extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', citation)
        if year_match:
            parsed['year'] = year_match.group(0)
        
        # Try to extract authors (before year)
        if parsed['year']:
            author_match = re.search(rf'(.+?)\s*,?\s*{parsed["year"]}', citation)
            if author_match:
                parsed['authors'] = author_match.group(1).strip()
        
        return parsed
    
    def _build_literature_content(
        self, 
        title: str, 
        authors: str, 
        year: str,
        analysis: Dict[str, Any], 
        doi: str, 
        journal: str
    ) -> str:
        """Build structured content for literature note."""
        content = f"# {title}\n\n"
        
        # Citation
        content += "## Citation\n"
        citation = f"{authors} ({year}). {title}."
        if journal:
            citation += f" *{journal}*."
        if doi:
            citation += f" DOI: {doi}"
        content += f"> {citation}\n\n"
        
        # Abstract
        if analysis.get('abstract'):
            content += "## Abstract\n"
            content += f"{analysis['abstract']}\n\n"
        
        # Key Contributions
        content += "## Key Contributions\n"
        if analysis.get('key_findings'):
            for finding in analysis['key_findings']:
                content += f"- {finding}\n"
        else:
            content += "- *[Add key contributions]*\n"
        content += "\n"
        
        # Methodology
        if analysis.get('methodology'):
            content += "## Methodology\n"
            content += f"{analysis['methodology']}\n\n"
        
        # Limitations
        if analysis.get('limitations'):
            content += "## Limitations\n"
            for limitation in analysis['limitations']:
                content += f"- {limitation}\n"
            content += "\n"
        
        # Future Work
        if analysis.get('future_work'):
            content += "## Future Work\n"
            for work in analysis['future_work']:
                content += f"- {work}\n"
            content += "\n"
        
        # Research Notes
        content += "## My Research Notes\n"
        content += "*How does this relate to my research?*\n\n"
        
        # Follow-up Actions
        content += "## Follow-up Actions\n"
        content += "- [ ] Review methodology in detail\n"
        content += "- [ ] Check related work citations\n"
        content += "- [ ] Consider application to my research\n\n"
        
        return content