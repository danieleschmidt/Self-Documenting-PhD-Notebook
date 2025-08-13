"""
Multi-region compliance module for PhD Notebook.
Handles GDPR, CCPA, PDPA and other data protection regulations.
"""

import json
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ComplianceRegulation(Enum):
    """Supported data protection regulations."""
    GDPR = "gdpr"  # European Union
    CCPA = "ccpa"  # California, USA
    PDPA_SG = "pdpa_sg"  # Singapore
    PDPA_TH = "pdpa_th"  # Thailand
    LGPD = "lgpd"  # Brazil
    PIPEDA = "pipeda"  # Canada
    DPA = "dpa"  # UK


class DataCategory(Enum):
    """Categories of personal data."""
    BASIC_PERSONAL = "basic_personal"  # Name, email, address
    SENSITIVE_PERSONAL = "sensitive_personal"  # Health, biometrics, etc.
    RESEARCH_DATA = "research_data"  # Research participants, surveys
    ACADEMIC_RECORDS = "academic_records"  # Grades, transcripts
    COLLABORATION_DATA = "collaboration_data"  # Communication, shared documents
    TECHNICAL_DATA = "technical_data"  # IP addresses, device info


@dataclass
class DataProcessingPurpose:
    """Purpose for processing personal data."""
    purpose: str
    legal_basis: str
    retention_period: int  # days
    categories: List[DataCategory]
    description: str = ""


@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""
    user_id: str
    purposes: List[str]
    granted_at: datetime
    withdrawn_at: Optional[datetime] = None
    consent_version: str = "1.0"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class DataBreachIncident:
    """Record of data breach incident."""
    incident_id: str
    detected_at: datetime
    affected_records: int
    data_categories: List[DataCategory]
    breach_type: str  # unauthorized_access, data_loss, etc.
    severity: str  # low, medium, high, critical
    status: str  # detected, investigating, contained, resolved
    reported_at: Optional[datetime] = None
    notification_required: bool = True


class ComplianceManager:
    """
    Multi-region compliance manager for academic research data.
    
    Ensures compliance with major data protection regulations
    while supporting international research collaboration.
    """
    
    def __init__(self, base_regulation: ComplianceRegulation = ComplianceRegulation.GDPR):
        self.base_regulation = base_regulation
        self.enabled_regulations: Set[ComplianceRegulation] = {base_regulation}
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self.processing_purposes: Dict[str, DataProcessingPurpose] = {}
        self.breach_incidents: List[DataBreachIncident] = []
        self.audit_log: List[Dict[str, Any]] = []
        
        # Initialize default processing purposes for academic research
        self._initialize_default_purposes()
        
        # PII detection patterns
        self.pii_patterns = self._load_pii_patterns()
        
    def _initialize_default_purposes(self):
        """Initialize default data processing purposes for academic research."""
        
        # Academic research purpose
        self.processing_purposes["academic_research"] = DataProcessingPurpose(
            purpose="academic_research",
            legal_basis="legitimate_interest",  # GDPR Article 6(1)(f)
            retention_period=2555,  # 7 years
            categories=[DataCategory.RESEARCH_DATA, DataCategory.COLLABORATION_DATA],
            description="Processing personal data for legitimate academic research purposes"
        )
        
        # System administration
        self.processing_purposes["system_admin"] = DataProcessingPurpose(
            purpose="system_administration",
            legal_basis="legitimate_interest",
            retention_period=1095,  # 3 years
            categories=[DataCategory.TECHNICAL_DATA],
            description="Processing technical data for system security and administration"
        )
        
        # User account management
        self.processing_purposes["account_management"] = DataProcessingPurpose(
            purpose="account_management",
            legal_basis="contract",  # GDPR Article 6(1)(b)
            retention_period=2555,  # 7 years
            categories=[DataCategory.BASIC_PERSONAL],
            description="Processing personal data for user account management"
        )
        
        # Research collaboration
        self.processing_purposes["research_collaboration"] = DataProcessingPurpose(
            purpose="research_collaboration",
            legal_basis="consent",  # GDPR Article 6(1)(a)
            retention_period=1825,  # 5 years
            categories=[DataCategory.BASIC_PERSONAL, DataCategory.COLLABORATION_DATA],
            description="Processing personal data to facilitate research collaboration"
        )
    
    def _load_pii_patterns(self) -> Dict[str, List[str]]:
        """Load PII detection patterns for different regions."""
        return {
            "email": [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            "phone": [
                r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',  # US/Canada
                r'\+44[-.\s]?[0-9]{4}[-.\s]?[0-9]{6}',  # UK
                r'\+33[-.\s]?[0-9][-.\s]?[0-9]{8}',  # France
                r'\+49[-.\s]?[0-9]{3}[-.\s]?[0-9]{7,8}',  # Germany
            ],
            "ssn": [r'\b\d{3}-\d{2}-\d{4}\b'],  # US SSN
            "credit_card": [r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'],
            "ip_address": [r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'],
            "passport": [r'\b[A-Z]{1,2}[0-9]{6,9}\b'],  # Common passport format
        }
    
    def enable_regulation(self, regulation: ComplianceRegulation):
        """Enable compliance with additional regulation."""
        self.enabled_regulations.add(regulation)
        self._log_audit_event("regulation_enabled", {"regulation": regulation.value})
    
    def record_consent(self, user_id: str, purposes: List[str], 
                      metadata: Dict[str, str] = None) -> str:
        """Record user consent for data processing."""
        metadata = metadata or {}
        
        consent = ConsentRecord(
            user_id=user_id,
            purposes=purposes,
            granted_at=datetime.now(),
            ip_address=metadata.get("ip_address"),
            user_agent=metadata.get("user_agent")
        )
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = []
        
        self.consent_records[user_id].append(consent)
        
        self._log_audit_event("consent_granted", {
            "user_id": user_id,
            "purposes": purposes,
            "consent_id": f"{user_id}_{len(self.consent_records[user_id])}"
        })
        
        return f"{user_id}_{len(self.consent_records[user_id])}"
    
    def withdraw_consent(self, user_id: str, purposes: List[str] = None) -> bool:
        """Withdraw user consent for specified purposes."""
        if user_id not in self.consent_records:
            return False
        
        withdrawn_any = False
        
        for consent in self.consent_records[user_id]:
            if consent.withdrawn_at is None:
                if purposes is None:
                    # Withdraw all consent
                    consent.withdrawn_at = datetime.now()
                    withdrawn_any = True
                else:
                    # Only withdraw if this consent record contains ANY of the specified purposes
                    if any(p in consent.purposes for p in purposes):
                        # Remove only the specified purposes, not the whole consent
                        for purpose in purposes:
                            if purpose in consent.purposes:
                                consent.purposes.remove(purpose)
                                withdrawn_any = True
                        
                        # If no purposes left, mark consent as withdrawn
                        if not consent.purposes:
                            consent.withdrawn_at = datetime.now()
        
        if withdrawn_any:
            self._log_audit_event("consent_withdrawn", {
                "user_id": user_id,
                "purposes": purposes or "all"
            })
        
        return withdrawn_any
    
    def check_consent_valid(self, user_id: str, purpose: str) -> bool:
        """Check if user has valid consent for a specific purpose."""
        if user_id not in self.consent_records:
            return False
        
        for consent in self.consent_records[user_id]:
            if (consent.withdrawn_at is None and 
                purpose in consent.purposes):
                
                # Check if consent has expired (GDPR: consent should be refreshed periodically)
                if ComplianceRegulation.GDPR in self.enabled_regulations:
                    consent_age = datetime.now() - consent.granted_at
                    if consent_age > timedelta(days=730):  # 2 years
                        return False
                
                return True
        
        return False
    
    def scan_for_pii(self, text: str) -> List[Dict[str, Any]]:
        """Scan text for personally identifiable information."""
        findings = []
        
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    findings.append({
                        "type": pii_type,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.8  # Simple confidence score
                    })
        
        return findings
    
    def anonymize_text(self, text: str, method: str = "replacement") -> Tuple[str, List[Dict[str, Any]]]:
        """
        Anonymize PII in text while preserving research value.
        
        Args:
            text: Input text to anonymize
            method: Anonymization method (replacement, hashing, removal)
            
        Returns:
            Tuple of (anonymized_text, anonymization_log)
        """
        anonymized = text
        log = []
        
        pii_findings = self.scan_for_pii(text)
        
        # Sort by position in reverse order to maintain indices
        pii_findings.sort(key=lambda x: x["start"], reverse=True)
        
        for finding in pii_findings:
            original_value = finding["value"]
            start, end = finding["start"], finding["end"]
            pii_type = finding["type"]
            
            if method == "replacement":
                if pii_type == "email":
                    replacement = "[EMAIL_REDACTED]"
                elif pii_type == "phone":
                    replacement = "[PHONE_REDACTED]"
                elif pii_type == "ssn":
                    replacement = "[SSN_REDACTED]"
                elif pii_type == "credit_card":
                    replacement = "[CARD_REDACTED]"
                elif pii_type == "ip_address":
                    replacement = "[IP_REDACTED]"
                else:
                    replacement = f"[{pii_type.upper()}_REDACTED]"
                    
            elif method == "hashing":
                # Create consistent hash for the same value
                hash_value = hashlib.sha256(original_value.encode()).hexdigest()[:8]
                replacement = f"[{pii_type.upper()}_{hash_value}]"
                
            elif method == "removal":
                replacement = ""
                
            else:
                replacement = "[REDACTED]"
            
            anonymized = anonymized[:start] + replacement + anonymized[end:]
            
            log.append({
                "original_value": original_value,
                "replacement": replacement,
                "type": pii_type,
                "method": method,
                "position": (start, end)
            })
        
        return anonymized, log
    
    def report_data_breach(self, affected_records: int, data_categories: List[DataCategory], 
                          breach_type: str, severity: str = "medium") -> str:
        """Report a data breach incident."""
        
        incident_id = f"BREACH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        incident = DataBreachIncident(
            incident_id=incident_id,
            detected_at=datetime.now(),
            affected_records=affected_records,
            data_categories=data_categories,
            breach_type=breach_type,
            severity=severity,
            status="detected"
        )
        
        # Determine notification requirements based on regulations
        if ComplianceRegulation.GDPR in self.enabled_regulations:
            # GDPR requires notification within 72 hours for high-risk breaches
            if severity in ["high", "critical"] or affected_records > 100:
                incident.notification_required = True
        
        self.breach_incidents.append(incident)
        
        self._log_audit_event("data_breach_reported", {
            "incident_id": incident_id,
            "affected_records": affected_records,
            "severity": severity,
            "breach_type": breach_type
        })
        
        return incident_id
    
    def get_data_retention_period(self, purpose: str, regulation: ComplianceRegulation = None) -> int:
        """Get data retention period for a specific purpose and regulation."""
        regulation = regulation or self.base_regulation
        
        if purpose in self.processing_purposes:
            base_period = self.processing_purposes[purpose].retention_period
            
            # Apply regulation-specific adjustments
            if regulation == ComplianceRegulation.GDPR:
                # GDPR emphasizes data minimization
                return min(base_period, 2555)  # Max 7 years for most academic data
            elif regulation == ComplianceRegulation.CCPA:
                # CCPA allows business purposes
                return base_period
            elif regulation in [ComplianceRegulation.PDPA_SG, ComplianceRegulation.PDPA_TH]:
                # PDPA emphasizes purpose limitation
                return min(base_period, 1825)  # Max 5 years
            
        return 1095  # Default 3 years
    
    def generate_privacy_notice(self, regulation: ComplianceRegulation = None) -> str:
        """Generate privacy notice compliant with specified regulation."""
        regulation = regulation or self.base_regulation
        
        notice_parts = []
        
        # Header
        notice_parts.append("# PRIVACY NOTICE - PhD Notebook System")
        notice_parts.append(f"Compliant with {regulation.value.upper()}")
        notice_parts.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")
        
        # Data controller information
        notice_parts.append("\\n## Data Controller")
        notice_parts.append("PhD Notebook Research Platform")
        notice_parts.append("Contact: privacy@phd-notebook.org")
        
        # Data processing purposes
        notice_parts.append("\\n## How We Use Your Data")
        for purpose_id, purpose in self.processing_purposes.items():
            notice_parts.append(f"\\n### {purpose.purpose.replace('_', ' ').title()}")
            notice_parts.append(f"**Purpose:** {purpose.description}")
            notice_parts.append(f"**Legal Basis:** {purpose.legal_basis}")
            notice_parts.append(f"**Retention Period:** {purpose.retention_period // 365} years")
            notice_parts.append(f"**Data Categories:** {', '.join([cat.value for cat in purpose.categories])}")
        
        # User rights
        notice_parts.append("\\n## Your Rights")
        if regulation == ComplianceRegulation.GDPR:
            rights = [
                "Right to access your data",
                "Right to rectify inaccurate data", 
                "Right to erasure ('right to be forgotten')",
                "Right to restrict processing",
                "Right to data portability",
                "Right to object to processing",
                "Right to withdraw consent"
            ]
        elif regulation == ComplianceRegulation.CCPA:
            rights = [
                "Right to know what personal information is collected",
                "Right to delete personal information",
                "Right to opt-out of sale of personal information",
                "Right to non-discrimination"
            ]
        else:
            rights = [
                "Right to access your data",
                "Right to correct your data",
                "Right to delete your data",
                "Right to withdraw consent"
            ]
        
        for right in rights:
            notice_parts.append(f"- {right}")
        
        # Contact information
        notice_parts.append("\\n## Contact Us")
        notice_parts.append("For privacy-related questions: privacy@phd-notebook.org")
        notice_parts.append("For data protection officer: dpo@phd-notebook.org")
        
        return "\\n".join(notice_parts)
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for GDPR data portability requests."""
        export_data = {
            "user_id": user_id,
            "export_timestamp": datetime.now().isoformat(),
            "consent_records": [],
            "processing_purposes": list(self.processing_purposes.keys()),
            "data_categories": []
        }
        
        # Include consent records
        if user_id in self.consent_records:
            for consent in self.consent_records[user_id]:
                export_data["consent_records"].append({
                    "purposes": consent.purposes,
                    "granted_at": consent.granted_at.isoformat(),
                    "withdrawn_at": consent.withdrawn_at.isoformat() if consent.withdrawn_at else None,
                    "consent_version": consent.consent_version
                })
        
        self._log_audit_event("data_exported", {"user_id": user_id})
        
        return export_data
    
    def delete_user_data(self, user_id: str, verify_consent_withdrawal: bool = True) -> bool:
        """Delete all user data for GDPR erasure requests."""
        
        if verify_consent_withdrawal:
            # Check if user has withdrawn all consents
            has_active_consent = any(
                consent.withdrawn_at is None 
                for consent in self.consent_records.get(user_id, [])
            )
            if has_active_consent:
                return False  # Cannot delete data with active consent
        
        # Remove user data
        if user_id in self.consent_records:
            del self.consent_records[user_id]
        
        self._log_audit_event("data_deleted", {
            "user_id": user_id,
            "verified_consent_withdrawal": verify_consent_withdrawal
        })
        
        return True
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event for compliance tracking."""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "regulation_context": [reg.value for reg in self.enabled_regulations]
        })
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        # Calculate consent statistics
        total_users = len(self.consent_records)
        total_consents = sum(len(records) for records in self.consent_records.values())
        withdrawn_consents = sum(
            sum(1 for consent in records if consent.withdrawn_at is not None)
            for records in self.consent_records.values()
        )
        
        # Breach statistics
        total_breaches = len(self.breach_incidents)
        high_severity_breaches = sum(
            1 for incident in self.breach_incidents 
            if incident.severity in ["high", "critical"]
        )
        
        # Audit statistics
        recent_audit_events = [
            event for event in self.audit_log
            if datetime.fromisoformat(event["timestamp"]) > datetime.now() - timedelta(days=30)
        ]
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "enabled_regulations": [reg.value for reg in self.enabled_regulations],
            "consent_statistics": {
                "total_users": total_users,
                "total_consents": total_consents,
                "withdrawn_consents": withdrawn_consents,
                "consent_withdrawal_rate": withdrawn_consents / max(total_consents, 1)
            },
            "breach_statistics": {
                "total_incidents": total_breaches,
                "high_severity_incidents": high_severity_breaches,
                "incident_rate": total_breaches / max(total_users, 1)
            },
            "audit_statistics": {
                "total_audit_events": len(self.audit_log),
                "recent_events_30d": len(recent_audit_events),
                "event_types": list(set(event["event_type"] for event in self.audit_log))
            },
            "processing_purposes": {
                purpose_id: {
                    "description": purpose.description,
                    "legal_basis": purpose.legal_basis,
                    "retention_days": purpose.retention_period,
                    "data_categories": [cat.value for cat in purpose.categories]
                }
                for purpose_id, purpose in self.processing_purposes.items()
            }
        }
    
    def validate_cross_border_transfer(self, source_country: str, target_country: str) -> Dict[str, Any]:
        """Validate international data transfer compliance."""
        
        # EU adequacy decisions (simplified)
        eu_adequate_countries = [
            "andorra", "argentina", "canada", "faroe_islands", "guernsey", "israel", 
            "isle_of_man", "japan", "jersey", "new_zealand", "republic_of_korea", 
            "switzerland", "united_kingdom", "uruguay"
        ]
        
        # Safe Harbor / Privacy Shield countries
        safe_harbor_countries = ["united_states"]  # With appropriate frameworks
        
        validation_result = {
            "transfer_allowed": False,
            "mechanism_required": None,
            "additional_safeguards": [],
            "risk_level": "high"
        }
        
        source_lower = source_country.lower()
        target_lower = target_country.lower()
        
        # Same country transfer
        if source_lower == target_lower:
            validation_result.update({
                "transfer_allowed": True,
                "risk_level": "low",
                "mechanism_required": "none"
            })
            return validation_result
        
        # EU-based transfers
        if source_lower in ["eu", "european_union"] or source_lower.endswith("_eu"):
            if target_lower in eu_adequate_countries:
                validation_result.update({
                    "transfer_allowed": True,
                    "mechanism_required": "adequacy_decision",
                    "risk_level": "low"
                })
            elif target_lower in safe_harbor_countries:
                validation_result.update({
                    "transfer_allowed": True,
                    "mechanism_required": "privacy_shield_or_scc",
                    "risk_level": "medium",
                    "additional_safeguards": ["standard_contractual_clauses", "impact_assessment"]
                })
            else:
                validation_result.update({
                    "transfer_allowed": True,
                    "mechanism_required": "standard_contractual_clauses",
                    "risk_level": "high",
                    "additional_safeguards": ["impact_assessment", "additional_measures", "regular_review"]
                })
        
        # Academic research exemptions
        if all(country.lower() in ["academic", "research", "university"] for country in [source_country, target_country]):
            validation_result.update({
                "transfer_allowed": True,
                "mechanism_required": "academic_collaboration_agreement",
                "risk_level": "medium",
                "additional_safeguards": ["data_sharing_agreement", "ethical_approval"]
            })
        
        return validation_result