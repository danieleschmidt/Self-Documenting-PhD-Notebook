"""
Comprehensive localization system for global research collaboration.
"""

import json
import locale
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedLocale(Enum):
    """Supported locales for global research."""
    EN_US = "en_US"  # English (United States)
    EN_GB = "en_GB"  # English (United Kingdom)
    ES_ES = "es_ES"  # Spanish (Spain)
    FR_FR = "fr_FR"  # French (France)
    DE_DE = "de_DE"  # German (Germany)
    JA_JP = "ja_JP"  # Japanese (Japan)
    ZH_CN = "zh_CN"  # Chinese (Simplified, China)
    PT_BR = "pt_BR"  # Portuguese (Brazil)
    RU_RU = "ru_RU"  # Russian (Russia)
    KO_KR = "ko_KR"  # Korean (South Korea)


@dataclass
class LocaleInfo:
    """Information about a locale configuration."""
    code: str
    name: str
    native_name: str
    date_format: str
    time_format: str
    currency: str
    number_format: str
    rtl: bool = False  # Right-to-left text direction
    academic_conventions: Dict[str, str] = field(default_factory=dict)


class LocalizationManager:
    """Manages localization for global academic research environments."""
    
    def __init__(self, default_locale: SupportedLocale = SupportedLocale.EN_US):
        self.current_locale = default_locale
        self._translations: Dict[str, Dict[str, str]] = {}
        self._locale_info: Dict[str, LocaleInfo] = self._init_locale_info()
        self.load_translations()
        
    def _init_locale_info(self) -> Dict[str, LocaleInfo]:
        """Initialize locale configuration data."""
        return {
            "en_US": LocaleInfo(
                code="en_US",
                name="English (US)",
                native_name="English",
                date_format="%m/%d/%Y",
                time_format="%I:%M %p",
                currency="USD",
                number_format="1,234.56",
                academic_conventions={
                    "citation_style": "APA",
                    "reference_format": "Author, A. A. (Year). Title. Journal, Volume(Issue), pages.",
                    "date_style": "American",
                    "measurement_system": "Imperial"
                }
            ),
            "en_GB": LocaleInfo(
                code="en_GB",
                name="English (UK)",
                native_name="English",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                currency="GBP",
                number_format="1,234.56",
                academic_conventions={
                    "citation_style": "Oxford",
                    "reference_format": "Author, A. A., 'Title', Journal, Vol. Volume, No. Issue (Year), pages.",
                    "date_style": "British",
                    "measurement_system": "Metric"
                }
            ),
            "es_ES": LocaleInfo(
                code="es_ES",
                name="Spanish (Spain)",
                native_name="Español",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                currency="EUR",
                number_format="1.234,56",
                academic_conventions={
                    "citation_style": "ISO",
                    "reference_format": "Autor, A. A. Título. Revista, Volumen(Número), páginas (Año).",
                    "date_style": "European",
                    "measurement_system": "Metric"
                }
            ),
            "fr_FR": LocaleInfo(
                code="fr_FR",
                name="French (France)",
                native_name="Français",
                date_format="%d/%m/%Y",
                time_format="%H:%M",
                currency="EUR",
                number_format="1 234,56",
                academic_conventions={
                    "citation_style": "French",
                    "reference_format": "Auteur, A. A., « Titre », Revue, vol. Volume, n° Numéro, Année, p. pages.",
                    "date_style": "European",
                    "measurement_system": "Metric"
                }
            ),
            "de_DE": LocaleInfo(
                code="de_DE",
                name="German (Germany)",
                native_name="Deutsch",
                date_format="%d.%m.%Y",
                time_format="%H:%M",
                currency="EUR",
                number_format="1.234,56",
                academic_conventions={
                    "citation_style": "German",
                    "reference_format": "Autor, A. A.: Titel. In: Zeitschrift, Jahrgang (Jahr), Nr. Nummer, S. Seiten.",
                    "date_style": "German",
                    "measurement_system": "Metric"
                }
            ),
            "ja_JP": LocaleInfo(
                code="ja_JP",
                name="Japanese (Japan)",
                native_name="日本語",
                date_format="%Y/%m/%d",
                time_format="%H:%M",
                currency="JPY",
                number_format="1,234",
                academic_conventions={
                    "citation_style": "Japanese",
                    "reference_format": "著者名「タイトル」『雑誌名』巻号、頁数、発行年。",
                    "date_style": "Japanese",
                    "measurement_system": "Metric"
                }
            ),
            "zh_CN": LocaleInfo(
                code="zh_CN",
                name="Chinese (Simplified)",
                native_name="中文",
                date_format="%Y-%m-%d",
                time_format="%H:%M",
                currency="CNY",
                number_format="1,234.56",
                academic_conventions={
                    "citation_style": "Chinese",
                    "reference_format": "作者姓名.《论文标题》.期刊名,卷期号,页码,年份.",
                    "date_style": "Chinese",
                    "measurement_system": "Metric"
                }
            )
        }
    
    def load_translations(self):
        """Load translation strings for supported languages."""
        # Core UI translations
        self._translations = {
            "en_US": {
                "welcome": "Welcome to Your Research Vault",
                "create_note": "Create Note",
                "search_placeholder": "Search notes...",
                "note_created": "Note created successfully",
                "experiment_started": "Experiment started",
                "literature_review": "Literature Review",
                "methodology": "Methodology",
                "results": "Results",
                "conclusion": "Conclusion",
                "references": "References",
                "export_success": "Export completed successfully",
                "security_notice": "Your research data is encrypted and secure"
            },
            "es_ES": {
                "welcome": "Bienvenido a tu Archivo de Investigación",
                "create_note": "Crear Nota",
                "search_placeholder": "Buscar notas...",
                "note_created": "Nota creada exitosamente",
                "experiment_started": "Experimento iniciado",
                "literature_review": "Revisión de Literatura",
                "methodology": "Metodología",
                "results": "Resultados",
                "conclusion": "Conclusión",
                "references": "Referencias",
                "export_success": "Exportación completada exitosamente",
                "security_notice": "Tus datos de investigación están cifrados y seguros"
            },
            "fr_FR": {
                "welcome": "Bienvenue dans votre Coffre de Recherche",
                "create_note": "Créer une Note",
                "search_placeholder": "Rechercher des notes...",
                "note_created": "Note créée avec succès",
                "experiment_started": "Expérience commencée",
                "literature_review": "Revue de Littérature",
                "methodology": "Méthodologie",
                "results": "Résultats",
                "conclusion": "Conclusion",
                "references": "Références",
                "export_success": "Exportation terminée avec succès",
                "security_notice": "Vos données de recherche sont chiffrées et sécurisées"
            },
            "de_DE": {
                "welcome": "Willkommen in Ihrem Forschungsarchiv",
                "create_note": "Notiz Erstellen",
                "search_placeholder": "Notizen suchen...",
                "note_created": "Notiz erfolgreich erstellt",
                "experiment_started": "Experiment begonnen",
                "literature_review": "Literaturrecherche",
                "methodology": "Methodik",
                "results": "Ergebnisse",
                "conclusion": "Fazit",
                "references": "Literaturverzeichnis",
                "export_success": "Export erfolgreich abgeschlossen",
                "security_notice": "Ihre Forschungsdaten sind verschlüsselt und sicher"
            },
            "ja_JP": {
                "welcome": "研究資料庫へようこそ",
                "create_note": "ノート作成",
                "search_placeholder": "ノートを検索...",
                "note_created": "ノートが正常に作成されました",
                "experiment_started": "実験を開始しました",
                "literature_review": "文献レビュー",
                "methodology": "方法論",
                "results": "結果",
                "conclusion": "結論",
                "references": "参考文献",
                "export_success": "エクスポートが正常に完了しました",
                "security_notice": "あなたの研究データは暗号化され安全です"
            },
            "zh_CN": {
                "welcome": "欢迎使用您的研究资料库",
                "create_note": "创建笔记",
                "search_placeholder": "搜索笔记...",
                "note_created": "笔记创建成功",
                "experiment_started": "实验已开始",
                "literature_review": "文献综述",
                "methodology": "研究方法",
                "results": "结果",
                "conclusion": "结论",
                "references": "参考文献",
                "export_success": "导出成功完成",
                "security_notice": "您的研究数据已加密并安全保护"
            }
        }
    
    def set_locale(self, locale_code: SupportedLocale) -> bool:
        """Set the current locale."""
        try:
            self.current_locale = locale_code
            logger.info(f"Locale changed to {locale_code.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to set locale to {locale_code}: {e}")
            return False
    
    def get_text(self, key: str, fallback: Optional[str] = None) -> str:
        """Get localized text for a given key."""
        locale_code = self.current_locale.value
        
        if locale_code in self._translations:
            if key in self._translations[locale_code]:
                return self._translations[locale_code][key]
        
        # Fallback to English
        if "en_US" in self._translations and key in self._translations["en_US"]:
            return self._translations["en_US"][key]
        
        # Final fallback
        return fallback or f"[{key}]"
    
    def get_locale_info(self) -> LocaleInfo:
        """Get information about the current locale."""
        return self._locale_info.get(self.current_locale.value, self._locale_info["en_US"])
    
    def format_date(self, date: datetime) -> str:
        """Format date according to current locale."""
        locale_info = self.get_locale_info()
        return date.strftime(locale_info.date_format)
    
    def format_time(self, time: datetime) -> str:
        """Format time according to current locale."""
        locale_info = self.get_locale_info()
        return time.strftime(locale_info.time_format)
    
    def get_academic_style(self, style_type: str) -> str:
        """Get academic convention for current locale."""
        locale_info = self.get_locale_info()
        return locale_info.academic_conventions.get(style_type, "")


# Global instance
_localization_manager: Optional[LocalizationManager] = None

def get_localization_manager() -> LocalizationManager:
    """Get the global localization manager instance."""
    global _localization_manager
    if _localization_manager is None:
        _localization_manager = LocalizationManager()
    return _localization_manager

def get_text(key: str, fallback: Optional[str] = None) -> str:
    """Convenience function to get localized text."""
    return get_localization_manager().get_text(key, fallback)

def set_locale(locale_code: SupportedLocale) -> bool:
    """Convenience function to set locale."""
    return get_localization_manager().set_locale(locale_code)