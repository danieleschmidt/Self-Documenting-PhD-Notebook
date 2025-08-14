"""
Global compliance management for research data privacy and regulations.
Supports GDPR, CCPA, PDPA, and other international privacy frameworks.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import uuid

logger = logging.getLogger(__name__)


class PrivacyRegulation(Enum):
    """Supported privacy regulations."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore/Thailand)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    PRIVACY_ACT = "privacy_act"  # Privacy Act (Australia)


class PrivacyMode(Enum):
    """Privacy protection modes."""
    MINIMAL = "minimal"  # Basic anonymization
    ENHANCED = "enhanced"  # Strong anonymization with pseudonymization
    MAXIMUM = "maximum"  # Full anonymization with data isolation
    RESEARCH_EXEMPT = "research_exempt"  # Research exemption mode


class ConsentType(Enum):
    """Types of data processing consent."""
    EXPLICIT = "explicit"  # Explicit consent required
    IMPLICIT = "implicit"  # Implicit consent (legitimate interest)
    OPT_OUT = "opt_out"  # Opt-out based consent
    RESEARCH = "research"  # Research-specific consent


@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""
    user_id: str
    consent_type: ConsentType
    purpose: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    regulation: PrivacyRegulation = PrivacyRegulation.GDPR
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    activity_id: str
    data_type: str
    processing_purpose: str
    legal_basis: str
    data_controller: str
    data_processor: Optional[str]
    retention_period: timedelta
    created_at: datetime
    regulation: PrivacyRegulation
    automated_decision_making: bool = False
    third_party_sharing: List[str] = field(default_factory=list)


class ComplianceManager:
    """Manages global privacy compliance for research data."""
    
    def __init__(self, default_regulation: PrivacyRegulation = PrivacyRegulation.GDPR):
        self.default_regulation = default_regulation
        self.privacy_mode = PrivacyMode.ENHANCED
        self._consent_records: Dict[str, ConsentRecord] = {}
        self._processing_records: Dict[str, DataProcessingRecord] = {}
        self._anonymized_data: Dict[str, str] = {}
        self._data_lineage: Dict[str, List[str]] = {}
        
        # Initialize compliance rules for different regulations
        self._compliance_rules = self._init_compliance_rules()
    
    def _init_compliance_rules(self) -> Dict[PrivacyRegulation, Dict[str, Any]]:
        """Initialize compliance rules for different regulations."""
        return {
            PrivacyRegulation.GDPR: {
                "consent_required": True,
                "right_to_erasure": True,
                "right_to_portability": True,
                "right_to_rectification": True,
                "data_protection_officer_required": False,  # Depends on organization size
                "breach_notification_hours": 72,
                "retention_limits": {
                    "research_data": timedelta(days=365*10),  # 10 years typical
                    "personal_data": timedelta(days=365*5),   # 5 years typical
                    "metadata": timedelta(days=365*2)        # 2 years typical
                },
                "lawful_basis": [
                    "consent", "contract", "legal_obligation", 
                    "vital_interests", "public_task", "legitimate_interests"
                ],
                "special_categories_protection": True,
                "automated_decision_making_restrictions": True
            },
            PrivacyRegulation.CCPA: {
                "consent_required": False,  # Opt-out model
                "right_to_know": True,
                "right_to_delete": True,
                "right_to_non_discrimination": True,
                "right_to_opt_out": True,
                "breach_notification_hours": None,  # No specific requirement
                "retention_limits": {
                    "research_data": timedelta(days=365*7),   # 7 years
                    "personal_data": timedelta(days=365*3),   # 3 years
                    "metadata": timedelta(days=365*1)        # 1 year
                },
                "sale_opt_out_required": True,
                "third_party_disclosure_limits": True
            },
            PrivacyRegulation.PDPA: {
                "consent_required": True,
                "purpose_limitation": True,
                "data_minimization": True,
                "retention_limits": {
                    "research_data": timedelta(days=365*5),   # 5 years
                    "personal_data": timedelta(days=365*3),   # 3 years
                    "metadata": timedelta(days=365*1)        # 1 year
                },
                "cross_border_restrictions": True,
                "breach_notification_hours": 72,
                "dpo_appointment_threshold": True
            }
        }
    
    def set_privacy_mode(self, mode: PrivacyMode) -> bool:
        """Set the privacy protection mode."""
        try:
            self.privacy_mode = mode
            logger.info(f"Privacy mode set to {mode.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to set privacy mode: {e}")
            return False
    
    def record_consent(self, 
                      user_id: str, 
                      consent_type: ConsentType,
                      purpose: str,
                      duration: Optional[timedelta] = None) -> ConsentRecord:
        """Record user consent for data processing."""
        consent_id = f"consent_{uuid.uuid4().hex[:8]}"
        expires_at = None
        if duration:
            expires_at = datetime.now() + duration
        
        consent = ConsentRecord(
            user_id=user_id,
            consent_type=consent_type,
            purpose=purpose,
            granted_at=datetime.now(),
            expires_at=expires_at,
            regulation=self.default_regulation
        )
        
        self._consent_records[consent_id] = consent
        logger.info(f"Consent recorded for user {user_id}, purpose: {purpose}")
        return consent
    
    def withdraw_consent(self, user_id: str, purpose: str) -> bool:
        """Allow user to withdraw consent."""
        for consent_id, record in self._consent_records.items():
            if record.user_id == user_id and record.purpose == purpose:
                record.withdrawn_at = datetime.now()
                logger.info(f"Consent withdrawn for user {user_id}, purpose: {purpose}")
                return True
        
        logger.warning(f"No consent found to withdraw for user {user_id}, purpose: {purpose}")
        return False
    
    def anonymize_data(self, data: Any, identifier: str) -> str:
        """Anonymize sensitive data according to privacy mode."""
        if self.privacy_mode == PrivacyMode.MINIMAL:
            # Simple hash-based anonymization
            anonymous_id = hashlib.sha256(str(data).encode()).hexdigest()[:16]
        elif self.privacy_mode == PrivacyMode.ENHANCED:
            # Pseudonymization with salt
            salt = str(uuid.uuid4())
            anonymous_id = hashlib.pbkdf2_hmac('sha256', 
                                             str(data).encode(), 
                                             salt.encode(), 
                                             100000).hex()[:16]
        else:  # MAXIMUM
            # Full anonymization with random ID
            anonymous_id = f"anon_{uuid.uuid4().hex[:12]}"
        
        # Store mapping for potential re-identification (if legally required)
        if self.privacy_mode != PrivacyMode.MAXIMUM:
            self._anonymized_data[anonymous_id] = identifier
        
        return anonymous_id
    
    def record_data_processing(self,
                             data_type: str,
                             purpose: str,
                             legal_basis: str,
                             controller: str,
                             processor: Optional[str] = None,
                             retention_days: int = 365) -> str:
        """Record data processing activity for compliance."""
        activity_id = f"proc_{uuid.uuid4().hex[:8]}"
        
        processing_record = DataProcessingRecord(
            activity_id=activity_id,
            data_type=data_type,
            processing_purpose=purpose,
            legal_basis=legal_basis,
            data_controller=controller,
            data_processor=processor,
            retention_period=timedelta(days=retention_days),
            created_at=datetime.now(),
            regulation=self.default_regulation
        )
        
        self._processing_records[activity_id] = processing_record
        logger.info(f"Data processing recorded: {activity_id} for {data_type}")
        return activity_id
    
    def check_compliance(self, regulation: Optional[PrivacyRegulation] = None) -> Dict[str, Any]:
        """Check compliance status against specified regulation."""
        reg = regulation or self.default_regulation
        rules = self._compliance_rules.get(reg, {})
        
        compliance_status = {
            "regulation": reg.value,
            "overall_compliant": True,
            "issues": [],
            "recommendations": [],
            "last_checked": datetime.now().isoformat()
        }
        
        # Check consent requirements
        if rules.get("consent_required", False):
            expired_consents = [
                c for c in self._consent_records.values()
                if c.expires_at and c.expires_at < datetime.now()
            ]
            if expired_consents:
                compliance_status["issues"].append(f"Found {len(expired_consents)} expired consents")
                compliance_status["overall_compliant"] = False
        
        # Check retention periods
        retention_limits = rules.get("retention_limits", {})
        for record in self._processing_records.values():
            max_retention = retention_limits.get(record.data_type, timedelta(days=365))
            if record.created_at + record.retention_period > datetime.now() + max_retention:
                compliance_status["issues"].append(
                    f"Retention period for {record.activity_id} exceeds regulatory limits"
                )
                compliance_status["overall_compliant"] = False
        
        # General recommendations
        if reg == PrivacyRegulation.GDPR:
            compliance_status["recommendations"].extend([
                "Regularly review and update consent records",
                "Implement automated data deletion for expired retention periods",
                "Conduct periodic privacy impact assessments",
                "Ensure data processing records are complete and up-to-date"
            ])
        
        return compliance_status
    
    def generate_privacy_notice(self, regulation: Optional[PrivacyRegulation] = None) -> str:
        """Generate privacy notice text for the specified regulation."""
        reg = regulation or self.default_regulation
        
        notices = {
            PrivacyRegulation.GDPR: """
Privacy Notice - GDPR Compliance

Your personal data is processed in accordance with the General Data Protection Regulation (GDPR).

Data Controller: PhD Research Notebook System
Purpose: Academic research data management and analysis
Legal Basis: Legitimate interest for research purposes

Your Rights:
- Right to access your personal data
- Right to rectification of inaccurate data  
- Right to erasure ("right to be forgotten")
- Right to restrict processing
- Right to data portability
- Right to object to processing

Data Retention: Research data is retained for up to 10 years as required by academic standards.

Contact: For questions about data processing, contact your institution's Data Protection Officer.
""",
            PrivacyRegulation.CCPA: """
Privacy Notice - CCPA Compliance

California Consumer Privacy Act (CCPA) Rights

We collect and process personal information for research purposes.

Your California Rights:
- Right to know what personal information is collected
- Right to delete personal information
- Right to opt-out of the sale of personal information
- Right to non-discrimination for exercising privacy rights

Categories of Information: Research notes, academic data, usage analytics
Business Purpose: Academic research and educational activities

Contact: privacy@research-institution.edu
""",
            PrivacyRegulation.PDPA: """
Privacy Notice - PDPA Compliance

Personal Data Protection Notice

Your personal data is collected and processed in accordance with the Personal Data Protection Act.

Purpose: Academic research data management
Consent: By using this system, you consent to data processing for research purposes

Your Rights:
- Right to withdraw consent
- Right to access your personal data
- Right to correct inaccurate data

Data Protection: Your data is protected with industry-standard security measures.
Retention: Data is retained only as long as necessary for research purposes.
"""
        }
        
        return notices.get(reg, notices[PrivacyRegulation.GDPR])
    
    def export_compliance_report(self) -> Dict[str, Any]:
        """Export comprehensive compliance report."""
        return {
            "report_generated": datetime.now().isoformat(),
            "privacy_mode": self.privacy_mode.value,
            "default_regulation": self.default_regulation.value,
            "consent_records": len(self._consent_records),
            "processing_records": len(self._processing_records),
            "anonymized_entries": len(self._anonymized_data),
            "compliance_checks": {
                reg.value: self.check_compliance(reg) 
                for reg in [PrivacyRegulation.GDPR, PrivacyRegulation.CCPA, PrivacyRegulation.PDPA]
            }
        }