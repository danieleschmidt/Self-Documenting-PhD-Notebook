"""
Academic-specific translator for research content.
Specialized translation for academic papers, theses, and research documents.
"""

from typing import Dict, List, Optional, Any
from .translator import Translator, Language, TranslationContext


class AcademicTranslator(Translator):
    """Specialized translator for academic content."""
    
    def __init__(self):
        super().__init__()
        self.academic_contexts = self._load_academic_contexts()
    
    def _load_academic_contexts(self) -> Dict[str, TranslationContext]:
        """Load predefined academic contexts."""
        return {
            "research_paper": TranslationContext(
                field="general",
                document_type="paper",
                formality="academic",
                preserve_terms=["PhD", "MSc", "DOI", "ISSN", "ISBN"],
                target_audience="researchers"
            ),
            "thesis": TranslationContext(
                field="general", 
                document_type="thesis",
                formality="academic",
                preserve_terms=["PhD", "MSc", "DOI", "ISBN", "supervisor"],
                target_audience="academics"
            ),
            "conference_abstract": TranslationContext(
                field="general",
                document_type="abstract", 
                formality="academic",
                preserve_terms=["PhD", "MSc", "DOI"],
                target_audience="researchers"
            )
        }
    
    def translate_research_paper(self, paper: Dict[str, Any], target_lang: Language) -> Dict[str, Any]:
        """Translate a complete research paper."""
        context = self.academic_contexts["research_paper"]
        return self.translate_academic_document(paper, target_lang, context)
    
    def translate_thesis_chapter(self, chapter: Dict[str, Any], target_lang: Language) -> Dict[str, Any]:
        """Translate a thesis chapter."""
        context = self.academic_contexts["thesis"]
        return self.translate_academic_document(chapter, target_lang, context)