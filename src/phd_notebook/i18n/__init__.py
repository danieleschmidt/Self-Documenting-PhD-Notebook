"""
Internationalization (i18n) module for PhD Notebook.
Supports multiple languages and regions for global research collaboration.
"""

from .translator import Translator, TranslationError
from .localization import LocalizationManager, get_current_locale
from .academic_translator import AcademicTranslator
from .compliance import ComplianceManager

__all__ = [
    "Translator",
    "TranslationError", 
    "LocalizationManager",
    "get_current_locale",
    "AcademicTranslator",
    "ComplianceManager"
]