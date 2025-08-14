"""
Global-first internationalization and localization support.
"""

from .localization import LocalizationManager, get_text, set_locale
from .compliance import ComplianceManager, PrivacyMode
from .regional_adapters import RegionalAdapter

__all__ = ['LocalizationManager', 'get_text', 'set_locale', 'ComplianceManager', 'PrivacyMode', 'RegionalAdapter']