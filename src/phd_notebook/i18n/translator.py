"""
Translation system for academic content.
Supports multiple languages with academic terminology preservation.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Supported languages for PhD Notebook."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    KOREAN = "ko"


class TranslationError(Exception):
    """Exception raised for translation errors."""
    pass


@dataclass
class TranslationContext:
    """Context for academic translations."""
    field: str = "general"  # Academic field
    document_type: str = "note"  # paper, thesis, note, etc.
    formality: str = "academic"  # academic, informal, technical
    preserve_terms: List[str] = None  # Terms to never translate
    target_audience: str = "researchers"  # researchers, students, general


class Translator:
    """
    Academic-focused translation system.
    
    Provides translation capabilities optimized for academic content,
    preserving technical terminology and research-specific language.
    """
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.translation_cache: Dict[str, str] = {}
        self.academic_terms = self._load_academic_terms()
        self.language_patterns = self._load_language_patterns()
        
    def _load_academic_terms(self) -> Dict[str, Dict[str, str]]:
        """Load academic terminology database."""
        # In a real implementation, this would load from a comprehensive database
        return {
            "general": {
                "hypothesis": {"es": "hipótesis", "fr": "hypothèse", "de": "Hypothese", "ja": "仮説", "zh": "假设"},
                "experiment": {"es": "experimento", "fr": "expérience", "de": "Experiment", "ja": "実験", "zh": "实验"},
                "analysis": {"es": "análisis", "fr": "analyse", "de": "Analyse", "ja": "分析", "zh": "分析"},
                "methodology": {"es": "metodología", "fr": "méthodologie", "de": "Methodologie", "ja": "方法論", "zh": "方法论"},
                "literature review": {"es": "revisión de literatura", "fr": "revue de littérature", "de": "Literaturübersicht", "ja": "文献レビュー", "zh": "文献综述"}
            },
            "computer_science": {
                "algorithm": {"es": "algoritmo", "fr": "algorithme", "de": "Algorithmus", "ja": "アルゴリズム", "zh": "算法"},
                "neural network": {"es": "red neuronal", "fr": "réseau de neurones", "de": "neuronales Netzwerk", "ja": "ニューラルネットワーク", "zh": "神经网络"},
                "machine learning": {"es": "aprendizaje automático", "fr": "apprentissage automatique", "de": "maschinelles Lernen", "ja": "機械学習", "zh": "机器学习"}
            }
        }
    
    def _load_language_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load language-specific patterns for academic writing."""
        return {
            "en": {"formality_markers": ["therefore", "furthermore", "consequently"], "citation_format": "APA"},
            "es": {"formality_markers": ["por tanto", "además", "en consecuencia"], "citation_format": "ISO"},
            "fr": {"formality_markers": ["par conséquent", "de plus", "ainsi"], "citation_format": "French"},
            "de": {"formality_markers": ["daher", "außerdem", "folglich"], "citation_format": "German"},
            "ja": {"formality_markers": ["したがって", "さらに", "その結果"], "citation_format": "Japanese"},
            "zh": {"formality_markers": ["因此", "此外", "因而"], "citation_format": "Chinese"}
        }
    
    def translate_text(self, text: str, source_lang: Language, target_lang: Language, 
                      context: TranslationContext = None) -> str:
        """
        Translate academic text while preserving terminology.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language  
            context: Translation context for optimization
            
        Returns:
            Translated text
        """
        if source_lang == target_lang:
            return text
        
        context = context or TranslationContext()
        
        # Check cache
        cache_key = f"{text[:50]}_{source_lang.value}_{target_lang.value}_{context.field}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Preserve academic terms and references
        preserved_terms = self._extract_preserve_terms(text, context)
        
        # Translate main content
        translated = self._translate_content(text, source_lang, target_lang, context)
        
        # Restore preserved terms
        translated = self._restore_preserved_terms(translated, preserved_terms, target_lang)
        
        # Apply language-specific formatting
        translated = self._apply_language_formatting(translated, target_lang, context)
        
        # Cache result
        if len(self.translation_cache) >= self.cache_size:
            # Simple LRU eviction
            oldest_key = next(iter(self.translation_cache))
            del self.translation_cache[oldest_key]
        
        self.translation_cache[cache_key] = translated
        
        return translated
    
    def _extract_preserve_terms(self, text: str, context: TranslationContext) -> List[Tuple[str, str]]:
        """Extract terms that should be preserved during translation."""
        preserved = []
        
        # Academic citations (e.g., [Author, 2023])
        citation_pattern = r'\[([A-Za-z\s]+,?\s*\d{4})\]'
        citations = re.findall(citation_pattern, text)
        for citation in citations:
            preserved.append((f"[{citation}]", f"[{citation}]"))
        
        # DOIs and URLs
        doi_pattern = r'(doi:|https?://)[^\s\])]+'
        dois = re.findall(doi_pattern, text)
        for doi in dois:
            preserved.append((doi, doi))
        
        # Custom preserve terms from context
        if context.preserve_terms:
            for term in context.preserve_terms:
                if term.lower() in text.lower():
                    preserved.append((term, term))
        
        # Academic abbreviations (e.g., PhD, MSc, etc.)
        abbrev_pattern = r'\b[A-Z]{2,6}\b'
        abbrevs = re.findall(abbrev_pattern, text)
        for abbrev in abbrevs:
            if abbrev in ['PhD', 'MSc', 'BSc', 'MA', 'BA', 'MS', 'BS', 'JD', 'MD']:
                preserved.append((abbrev, abbrev))
        
        return preserved
    
    def _translate_content(self, text: str, source_lang: Language, target_lang: Language, 
                          context: TranslationContext) -> str:
        """Translate main content using academic terminology."""
        
        # Simple academic term substitution (in real implementation, would use proper MT)
        translated = text
        
        # Get academic terms for the field
        field_terms = self.academic_terms.get(context.field, {})
        general_terms = self.academic_terms.get("general", {})
        
        # Combine term dictionaries
        all_terms = {**general_terms, **field_terms}
        
        # Apply term translations
        for term_en, translations in all_terms.items():
            if target_lang.value in translations:
                target_term = translations[target_lang.value]
                
                # Case-insensitive replacement preserving original case
                pattern = re.compile(re.escape(term_en), re.IGNORECASE)
                
                def replace_func(match):
                    original = match.group(0)
                    if original.isupper():
                        return target_term.upper()
                    elif original.istitle():
                        return target_term.capitalize()
                    else:
                        return target_term.lower()
                
                translated = pattern.sub(replace_func, translated)
        
        # Handle common academic phrases
        academic_phrases = {
            "en": {
                "in conclusion": {"es": "en conclusión", "fr": "en conclusion", "de": "zusammenfassend", 
                                "ja": "結論として", "zh": "总之"},
                "on the other hand": {"es": "por otro lado", "fr": "d'autre part", "de": "andererseits", 
                                    "ja": "一方で", "zh": "另一方面"},
                "furthermore": {"es": "además", "fr": "de plus", "de": "außerdem", 
                              "ja": "さらに", "zh": "此外"}
            }
        }
        
        if source_lang.value in academic_phrases:
            for phrase_en, translations in academic_phrases[source_lang.value].items():
                if target_lang.value in translations:
                    target_phrase = translations[target_lang.value]
                    translated = re.sub(re.escape(phrase_en), target_phrase, translated, flags=re.IGNORECASE)
        
        return translated
    
    def _restore_preserved_terms(self, text: str, preserved_terms: List[Tuple[str, str]], 
                                target_lang: Language) -> str:
        """Restore preserved terms in translated text."""
        # In a real implementation, this would use placeholder restoration
        # For now, we assume preserved terms weren't modified
        return text
    
    def _apply_language_formatting(self, text: str, target_lang: Language, 
                                 context: TranslationContext) -> str:
        """Apply language-specific formatting rules."""
        
        # Language-specific punctuation adjustments
        if target_lang == Language.FRENCH:
            # French spacing rules for punctuation
            text = re.sub(r'\s*:\s*', ' : ', text)
            text = re.sub(r'\s*;\s*', ' ; ', text)
            text = re.sub(r'\s*\?\s*', ' ? ', text)
            text = re.sub(r'\s*!\s*', ' ! ', text)
        
        elif target_lang == Language.SPANISH:
            # Spanish inverted question/exclamation marks
            text = re.sub(r'(\?)', r'¿\1', text)
            text = re.sub(r'(\!)', r'¡\1', text)
        
        elif target_lang in [Language.JAPANESE, Language.CHINESE]:
            # Asian typography adjustments
            text = re.sub(r'\s*,\s*', '、', text)
            text = re.sub(r'\s*\.\s*', '。', text)
        
        return text
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages."""
        return list(Language)
    
    def detect_language(self, text: str) -> Language:
        """Detect the language of input text."""
        # Simple heuristic language detection
        # In real implementation, would use proper language detection library
        
        # Character-based detection
        if re.search(r'[ひらがなカタカナ]', text):
            return Language.JAPANESE
        elif re.search(r'[一-龯]', text):
            return Language.CHINESE
        elif re.search(r'[а-яё]', text, re.IGNORECASE):
            return Language.RUSSIAN
        elif re.search(r'[가-힣]', text):
            return Language.KOREAN
        
        # Word-based detection for European languages
        common_words = {
            Language.SPANISH: ['el', 'la', 'de', 'que', 'y', 'es', 'en', 'un', 'para', 'con'],
            Language.FRENCH: ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'],
            Language.GERMAN: ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
            Language.PORTUGUESE: ['o', 'de', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com'],
            Language.ITALIAN: ['il', 'di', 'e', 'la', 'per', 'in', 'un', 'è', 'che', 'con']
        }
        
        words = text.lower().split()
        if len(words) < 3:
            return Language.ENGLISH  # Default
        
        best_match = Language.ENGLISH
        best_score = 0
        
        for lang, common in common_words.items():
            score = sum(1 for word in words[:20] if word in common)  # Check first 20 words
            if score > best_score:
                best_score = score
                best_match = lang
        
        return best_match
    
    def translate_academic_document(self, document: Dict[str, Any], target_lang: Language, 
                                  context: TranslationContext = None) -> Dict[str, Any]:
        """
        Translate an entire academic document while preserving structure.
        
        Args:
            document: Document with title, content, metadata, etc.
            target_lang: Target language
            context: Translation context
            
        Returns:
            Translated document
        """
        context = context or TranslationContext()
        
        # Detect source language
        content_text = document.get('content', '') + ' ' + document.get('title', '')
        source_lang = self.detect_language(content_text)
        
        translated_doc = document.copy()
        
        # Translate title
        if 'title' in document:
            translated_doc['title'] = self.translate_text(
                document['title'], source_lang, target_lang, context
            )
        
        # Translate content
        if 'content' in document:
            translated_doc['content'] = self.translate_text(
                document['content'], source_lang, target_lang, context
            )
        
        # Translate abstract if present
        if 'abstract' in document:
            translated_doc['abstract'] = self.translate_text(
                document['abstract'], source_lang, target_lang, context
            )
        
        # Update metadata
        if 'metadata' not in translated_doc:
            translated_doc['metadata'] = {}
        
        translated_doc['metadata'].update({
            'source_language': source_lang.value,
            'target_language': target_lang.value,
            'translation_timestamp': str(datetime.now()),
            'translator_version': '1.0'
        })
        
        # Translate tags while preserving academic hashtags
        if 'tags' in document:
            translated_tags = []
            for tag in document['tags']:
                if tag.startswith('#'):
                    # Preserve hashtag format
                    tag_content = tag[1:]
                    translated_content = self.translate_text(tag_content, source_lang, target_lang, context)
                    translated_tags.append(f"#{translated_content}")
                else:
                    translated_tags.append(self.translate_text(tag, source_lang, target_lang, context))
            translated_doc['tags'] = translated_tags
        
        return translated_doc
    
    def get_translation_quality_score(self, original: str, translated: str, 
                                    target_lang: Language) -> float:
        """
        Estimate translation quality score.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not original or not translated:
            return 0.0
        
        # Simple quality metrics
        score = 0.0
        
        # Length similarity (academic translations should be similar length)
        length_ratio = min(len(translated), len(original)) / max(len(translated), len(original))
        score += length_ratio * 0.3
        
        # Preserve numbers and academic references
        orig_numbers = re.findall(r'\d+', original)
        trans_numbers = re.findall(r'\d+', translated)
        if orig_numbers == trans_numbers:
            score += 0.2
        
        # Preserve citation patterns
        orig_citations = len(re.findall(r'\[[^\]]+\]', original))
        trans_citations = len(re.findall(r'\[[^\]]+\]', translated))
        if orig_citations == trans_citations:
            score += 0.2
        
        # Basic completeness check
        if len(translated) > len(original) * 0.5:
            score += 0.3
        
        return min(score, 1.0)


# Import datetime here to avoid circular imports
from datetime import datetime