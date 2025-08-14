"""
Tests for global-first implementation: internationalization and compliance.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from phd_notebook.internationalization.localization import (
    LocalizationManager, SupportedLocale, get_text, set_locale
)
from phd_notebook.internationalization.compliance import (
    ComplianceManager, PrivacyRegulation, PrivacyMode, ConsentType
)
from phd_notebook.internationalization.regional_adapters import (
    RegionalAdapter, DeploymentRegion, PlatformType
)


class TestLocalization:
    """Test localization functionality."""
    
    def test_default_locale(self):
        """Test default locale initialization."""
        manager = LocalizationManager()
        assert manager.current_locale == SupportedLocale.EN_US
        
    def test_locale_switching(self):
        """Test switching between locales."""
        manager = LocalizationManager()
        
        # Test Spanish
        assert manager.set_locale(SupportedLocale.ES_ES)
        assert manager.current_locale == SupportedLocale.ES_ES
        
        # Test French
        assert manager.set_locale(SupportedLocale.FR_FR)
        assert manager.current_locale == SupportedLocale.FR_FR
        
    def test_text_translation(self):
        """Test text translation functionality."""
        manager = LocalizationManager()
        
        # English
        manager.set_locale(SupportedLocale.EN_US)
        assert "Welcome" in manager.get_text("welcome")
        
        # Spanish
        manager.set_locale(SupportedLocale.ES_ES)
        assert "Bienvenido" in manager.get_text("welcome")
        
        # French
        manager.set_locale(SupportedLocale.FR_FR)
        assert "Bienvenue" in manager.get_text("welcome")
        
        # German
        manager.set_locale(SupportedLocale.DE_DE)
        assert "Willkommen" in manager.get_text("welcome")
        
        # Japanese
        manager.set_locale(SupportedLocale.JA_JP)
        assert "研究資料庫" in manager.get_text("welcome")
        
        # Chinese
        manager.set_locale(SupportedLocale.ZH_CN)
        assert "研究资料库" in manager.get_text("welcome")
    
    def test_fallback_text(self):
        """Test fallback behavior for missing translations."""
        manager = LocalizationManager()
        
        # Test with non-existent key
        result = manager.get_text("nonexistent_key", "fallback_value")
        assert result == "fallback_value"
        
        # Test with no fallback
        result = manager.get_text("nonexistent_key")
        assert result == "[nonexistent_key]"
    
    def test_date_formatting(self):
        """Test locale-specific date formatting."""
        manager = LocalizationManager()
        test_date = datetime(2023, 12, 25, 14, 30, 0)
        
        # US format (MM/DD/YYYY)
        manager.set_locale(SupportedLocale.EN_US)
        us_date = manager.format_date(test_date)
        assert us_date == "12/25/2023"
        
        # UK format (DD/MM/YYYY)
        manager.set_locale(SupportedLocale.EN_GB)
        uk_date = manager.format_date(test_date)
        assert uk_date == "25/12/2023"
        
        # German format (DD.MM.YYYY)
        manager.set_locale(SupportedLocale.DE_DE)
        de_date = manager.format_date(test_date)
        assert de_date == "25.12.2023"
    
    def test_academic_conventions(self):
        """Test locale-specific academic conventions."""
        manager = LocalizationManager()
        
        # US academic style (APA)
        manager.set_locale(SupportedLocale.EN_US)
        citation_style = manager.get_academic_style("citation_style")
        assert citation_style == "APA"
        
        # UK academic style (Oxford)
        manager.set_locale(SupportedLocale.EN_GB)
        citation_style = manager.get_academic_style("citation_style")
        assert citation_style == "Oxford"
        
        # German academic style
        manager.set_locale(SupportedLocale.DE_DE)
        citation_style = manager.get_academic_style("citation_style")
        assert citation_style == "German"
    
    def test_global_functions(self):
        """Test global convenience functions."""
        # Test global text function
        text = get_text("welcome")
        assert len(text) > 0
        
        # Test global locale setting
        assert set_locale(SupportedLocale.ES_ES)
        spanish_text = get_text("welcome")
        assert "Bienvenido" in spanish_text


class TestCompliance:
    """Test compliance management functionality."""
    
    def test_gdpr_compliance(self):
        """Test GDPR compliance features."""
        manager = ComplianceManager(PrivacyRegulation.GDPR)
        
        # Record consent
        consent = manager.record_consent(
            user_id="user123",
            consent_type=ConsentType.EXPLICIT,
            purpose="research_data_processing"
        )
        assert consent.user_id == "user123"
        assert consent.consent_type == ConsentType.EXPLICIT
        
        # Check compliance
        compliance = manager.check_compliance()
        assert compliance["regulation"] == "gdpr"
        assert "overall_compliant" in compliance
        
        # Generate privacy notice
        notice = manager.generate_privacy_notice()
        assert "GDPR" in notice
        assert "right to erasure" in notice.lower()
    
    def test_ccpa_compliance(self):
        """Test CCPA compliance features."""
        manager = ComplianceManager(PrivacyRegulation.CCPA)
        
        # Check compliance
        compliance = manager.check_compliance()
        assert compliance["regulation"] == "ccpa"
        
        # Generate privacy notice
        notice = manager.generate_privacy_notice()
        assert "CCPA" in notice
        assert "right to opt-out" in notice.lower()
    
    def test_privacy_modes(self):
        """Test different privacy protection modes."""
        manager = ComplianceManager()
        
        # Test minimal privacy mode
        assert manager.set_privacy_mode(PrivacyMode.MINIMAL)
        assert manager.privacy_mode == PrivacyMode.MINIMAL
        
        # Test anonymization in minimal mode
        anon_data = manager.anonymize_data("sensitive_info", "identifier1")
        assert len(anon_data) == 16  # Hash length for minimal mode
        
        # Test enhanced privacy mode
        assert manager.set_privacy_mode(PrivacyMode.ENHANCED)
        anon_data_enhanced = manager.anonymize_data("sensitive_info", "identifier2")
        assert len(anon_data_enhanced) == 16  # PBKDF2 hash length
        
        # Test maximum privacy mode
        assert manager.set_privacy_mode(PrivacyMode.MAXIMUM)
        anon_data_max = manager.anonymize_data("sensitive_info", "identifier3")
        assert anon_data_max.startswith("anon_")
    
    def test_consent_withdrawal(self):
        """Test consent withdrawal functionality."""
        manager = ComplianceManager()
        
        # Record consent
        manager.record_consent(
            user_id="user456",
            consent_type=ConsentType.EXPLICIT,
            purpose="data_analysis"
        )
        
        # Withdraw consent
        assert manager.withdraw_consent("user456", "data_analysis")
        
        # Try to withdraw non-existent consent
        assert not manager.withdraw_consent("user456", "nonexistent_purpose")
    
    def test_data_processing_records(self):
        """Test data processing activity recording."""
        manager = ComplianceManager()
        
        activity_id = manager.record_data_processing(
            data_type="research_data",
            purpose="academic_research",
            legal_basis="legitimate_interest",
            controller="University Research Dept"
        )
        
        assert activity_id.startswith("proc_")
        assert len(manager._processing_records) == 1
    
    def test_compliance_report_export(self):
        """Test compliance report export."""
        manager = ComplianceManager()
        
        # Add some data
        manager.record_consent("user789", ConsentType.EXPLICIT, "research")
        manager.record_data_processing("notes", "research", "consent", "researcher")
        
        report = manager.export_compliance_report()
        
        assert "report_generated" in report
        assert report["consent_records"] == 1
        assert report["processing_records"] == 1
        assert "compliance_checks" in report
        assert "gdpr" in report["compliance_checks"]


class TestRegionalAdapters:
    """Test regional adaptation functionality."""
    
    def test_platform_detection(self):
        """Test platform detection."""
        adapter = RegionalAdapter()
        assert isinstance(adapter.current_platform, PlatformType)
        assert adapter.current_platform != PlatformType.UNKNOWN
    
    def test_region_detection(self):
        """Test region detection."""
        adapter = RegionalAdapter()
        assert isinstance(adapter.target_region, DeploymentRegion)
    
    def test_storage_paths(self):
        """Test platform-specific storage paths."""
        adapter = RegionalAdapter()
        
        storage_path = adapter.get_storage_path()
        assert isinstance(storage_path, Path)
        assert storage_path.exists()  # Should create if doesn't exist
        
        temp_path = adapter.get_temp_path()
        assert isinstance(temp_path, Path)
        assert temp_path.exists()
    
    def test_regional_configurations(self):
        """Test regional configuration settings."""
        adapter = RegionalAdapter(DeploymentRegion.EU_WEST)
        config = adapter.get_regional_config()
        
        assert config.region == DeploymentRegion.EU_WEST
        assert config.timezone == "Europe/Dublin"
        assert "GDPR" in config.compliance_frameworks
        assert "EU" in config.data_residency_requirements
    
    def test_data_residency_validation(self):
        """Test data residency validation."""
        # EU region adapter
        eu_adapter = RegionalAdapter(DeploymentRegion.EU_WEST)
        
        # Valid EU location
        assert eu_adapter.validate_data_residency("eu-west-1")
        assert eu_adapter.validate_data_residency("European Union")
        
        # Invalid location for EU
        assert not eu_adapter.validate_data_residency("us-east-1")
    
    def test_encryption_configuration(self):
        """Test encryption configuration."""
        adapter = RegionalAdapter(DeploymentRegion.EU_WEST)
        encryption_config = adapter.get_encryption_config()
        
        assert "data_at_rest" in encryption_config
        assert "data_in_transit" in encryption_config
        assert encryption_config["data_at_rest"] == "AES-256-GCM"
        assert encryption_config["data_in_transit"] == "TLS 1.3"
    
    def test_platform_capabilities(self):
        """Test platform capability detection."""
        adapter = RegionalAdapter()
        capabilities = adapter.get_platform_capabilities()
        
        assert isinstance(capabilities, dict)
        assert "file_watching" in capabilities
        assert "background_processing" in capabilities
        assert "native_encryption" in capabilities
    
    def test_path_adaptation(self):
        """Test cross-platform path adaptation."""
        adapter = RegionalAdapter()
        
        test_paths = {
            "config": "/etc/phd-notebook/config.json",
            "data": "~/.phd_notebook/data",
            "logs": "/var/log/phd-notebook"
        }
        
        adapted_paths = adapter.adapt_file_paths(test_paths)
        
        assert isinstance(adapted_paths, dict)
        assert len(adapted_paths) == len(test_paths)
        
        # Paths should be adapted for current platform
        for key in test_paths:
            assert key in adapted_paths
            assert isinstance(adapted_paths[key], str)
    
    def test_regional_optimization(self):
        """Test configuration optimization for regions."""
        adapter = RegionalAdapter(DeploymentRegion.ASIA_PACIFIC)
        
        base_config = {
            "timeout": 30,
            "max_connections": 100
        }
        
        optimized_config = adapter.optimize_for_region(base_config)
        
        assert "timezone" in optimized_config
        assert "encryption" in optimized_config
        assert optimized_config["timezone"] == "Asia/Singapore"
    
    def test_deployment_info_generation(self):
        """Test comprehensive deployment information generation."""
        adapter = RegionalAdapter()
        deployment_info = adapter.generate_deployment_info()
        
        assert "deployment_info" in deployment_info
        info = deployment_info["deployment_info"]
        
        assert "platform" in info
        assert "region" in info
        assert "timezone" in info
        assert "storage_path" in info
        assert "compliance_frameworks" in info
        assert "generated_at" in info
    
    def test_compliance_frameworks_by_region(self):
        """Test region-specific compliance frameworks."""
        # Test EU region
        eu_adapter = RegionalAdapter(DeploymentRegion.EU_WEST)
        eu_frameworks = eu_adapter.get_compliance_frameworks()
        assert "GDPR" in eu_frameworks
        
        # Test US region
        us_adapter = RegionalAdapter(DeploymentRegion.US_EAST)
        us_frameworks = us_adapter.get_compliance_frameworks()
        assert "CCPA" in us_frameworks or "SOC2" in us_frameworks
        
        # Test APAC region
        apac_adapter = RegionalAdapter(DeploymentRegion.ASIA_PACIFIC)
        apac_frameworks = apac_adapter.get_compliance_frameworks()
        assert "PDPA" in apac_frameworks