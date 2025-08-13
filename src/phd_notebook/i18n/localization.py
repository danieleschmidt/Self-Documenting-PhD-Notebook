"""
Localization management for PhD Notebook.
Handles locale-specific formatting, dates, numbers, and cultural adaptations.
"""

import locale
import os
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass
from .translator import Language


@dataclass
class LocaleConfig:
    """Configuration for a specific locale."""
    language: Language
    country: str
    date_format: str
    time_format: str
    number_format: str
    currency_symbol: str
    academic_year_start: int  # Month (1-12)
    citation_style: str


class LocalizationManager:
    """Manages localization for different regions."""
    
    def __init__(self):
        self.current_locale = "en_US"
        self.locale_configs = self._load_locale_configs()
    
    def _load_locale_configs(self) -> Dict[str, LocaleConfig]:
        """Load locale configurations for supported regions."""
        return {
            "en_US": LocaleConfig(
                language=Language.ENGLISH,
                country="US",
                date_format="%m/%d/%Y",
                time_format="%I:%M %p",
                number_format="1,234.56",
                currency_symbol="$",
                academic_year_start=9,  # September
                citation_style="APA"
            ),
            "en_GB": LocaleConfig(
                language=Language.ENGLISH,
                country="GB", 
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1,234.56",
                currency_symbol="£",
                academic_year_start=9,
                citation_style="Harvard"
            ),
            "es_ES": LocaleConfig(
                language=Language.SPANISH,
                country="ES",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                number_format="1.234,56",
                currency_symbol="€",
                academic_year_start=9,
                citation_style="ISO"
            ),
            "fr_FR": LocaleConfig(
                language=Language.FRENCH,
                country="FR",
                date_format="%d/%m/%Y", 
                time_format="%H:%M",
                number_format="1 234,56",
                currency_symbol="€",
                academic_year_start=9,
                citation_style="French"
            ),
            "de_DE": LocaleConfig(
                language=Language.GERMAN,
                country="DE",
                date_format="%d.%m.%Y",
                time_format="%H:%M",
                number_format="1.234,56",
                currency_symbol="€",
                academic_year_start=10,  # October
                citation_style="German"
            ),
            "ja_JP": LocaleConfig(
                language=Language.JAPANESE,
                country="JP",
                date_format="%Y/%m/%d",
                time_format="%H:%M",
                number_format="1,234.56",
                currency_symbol="¥",
                academic_year_start=4,  # April
                citation_style="Japanese"
            ),
            "zh_CN": LocaleConfig(
                language=Language.CHINESE,
                country="CN",
                date_format="%Y-%m-%d",
                time_format="%H:%M",
                number_format="1,234.56",
                currency_symbol="¥",
                academic_year_start=9,
                citation_style="Chinese"
            )
        }
    
    def set_locale(self, locale_code: str) -> bool:
        """Set current locale."""
        if locale_code in self.locale_configs:
            self.current_locale = locale_code
            return True
        return False
    
    def get_current_config(self) -> LocaleConfig:
        """Get current locale configuration."""
        return self.locale_configs.get(self.current_locale, self.locale_configs["en_US"])


def get_current_locale() -> str:
    """Get current system locale."""
    try:
        return locale.getlocale()[0] or "en_US"
    except:
        return "en_US"