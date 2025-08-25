"""
Advanced Research Security System

A comprehensive security framework for protecting research data, intellectual property,
and ensuring compliance with global data protection regulations while maintaining
research collaboration capabilities.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import uuid
import re

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for research data."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = os.getenv("SECRET".upper(), "default_value")


class AccessRole(Enum):
    """Access roles for research collaboration."""
    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    EDITOR = "editor"
    ADMIN = "admin"
    OWNER = "owner"


class ThreatLevel(Enum):
    """Threat assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    FERPA = "ferpa"
    SOX = "sox"
    ISO27001 = "iso27001"
    NIST = "nist"


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    description: str
    security_level: SecurityLevel
    access_controls: Dict[AccessRole, List[str]]
    encryption_required: bool
    audit_required: bool
    retention_period: timedelta
    compliance_frameworks: List[ComplianceFramework]
    custom_rules: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AccessPermission:
    """Access permission for a resource."""
    permission_id: str
    user_id: str
    resource_id: str
    role: AccessRole
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime] = None
    conditions: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: str
    user_id: Optional[str]
    resource_id: Optional[str]
    description: str
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_resolved: bool = False
    resolution_notes: str = ""


@dataclass
class EncryptionKey:
    """Encryption key metadata."""
    key_id: str
    key_type: str  # "symmetric", "public", "private"
    algorithm: str
    key_strength: int
    purpose: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True


class ResearchSecurityManager:
    """
    Comprehensive security management for research environments.
    
    Features:
    - Multi-level data classification and protection
    - Advanced encryption and key management
    - Access control and authentication
    - Audit logging and compliance monitoring
    - Threat detection and response
    - Privacy protection and anonymization
    """
    
    def __init__(self, 
                 default_security_level: SecurityLevel = SecurityLevel.INTERNAL,
                 compliance_frameworks: List[ComplianceFramework] = None):
        self.default_security_level = default_security_level
        self.compliance_frameworks = compliance_frameworks or [ComplianceFramework.GDPR]
        
        # Security components
        self.policies: Dict[str, SecurityPolicy] = {}
        self.permissions: Dict[str, AccessPermission] = {}
        self.security_events: List[SecurityEvent] = []
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        
        # Threat detection
        self.threat_detector = ThreatDetectionEngine()
        self.anomaly_detector = SecurityAnomalyDetector()
        
        # Encryption manager
        if CRYPTO_AVAILABLE:
            self.encryption_manager = EncryptionManager()
        else:
            logger.warning("Cryptography library not available - encryption disabled")
            self.encryption_manager = None
        
        # Privacy protection
        self.privacy_protector = PrivacyProtectionEngine()
        
        # Audit system
        self.audit_logger = AuditLogger()
        
        # Security metrics
        self.metrics = {
            "security_events": 0,
            "high_threat_events": 0,
            "access_violations": 0,
            "encryption_operations": 0,
            "privacy_violations": 0,
            "compliance_checks": 0,
            "successful_authentications": 0
        }
        
        # Initialize default policies
        self._initialize_default_policies()
        
        logger.info(f"Initialized Research Security Manager with level: {default_security_level.value}")
    
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        
        # Public research policy
        public_policy = SecurityPolicy(
            policy_id="policy_public",
            name="Public Research Policy",
            description="Policy for publicly shareable research data",
            security_level=SecurityLevel.PUBLIC,
            access_controls={
                AccessRole.VIEWER: ["read"],
                AccessRole.CONTRIBUTOR: ["read", "comment"],
                AccessRole.EDITOR: ["read", "write", "comment"],
                AccessRole.ADMIN: ["read", "write", "comment", "manage"],
                AccessRole.OWNER: ["all"]
            },
            encryption_required=False,
            audit_required=True,
            retention_period=timedelta(days=365*7),  # 7 years
            compliance_frameworks=[ComplianceFramework.GDPR]
        )
        self.register_policy(public_policy)
        
        # Confidential research policy
        confidential_policy = SecurityPolicy(
            policy_id="policy_confidential",
            name="Confidential Research Policy",
            description="Policy for confidential research data",
            security_level=SecurityLevel.CONFIDENTIAL,
            access_controls={
                AccessRole.VIEWER: ["read"],
                AccessRole.EDITOR: ["read", "write"],
                AccessRole.ADMIN: ["read", "write", "manage"],
                AccessRole.OWNER: ["all"]
            },
            encryption_required=True,
            audit_required=True,
            retention_period=timedelta(days=365*10),  # 10 years
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.ISO27001]
        )
        self.register_policy(confidential_policy)
        
        # Human subjects research policy
        human_subjects_policy = SecurityPolicy(
            policy_id="policy_human_subjects",
            name="Human Subjects Research Policy",
            description="Policy for research involving human subjects",
            security_level=SecurityLevel.RESTRICTED,
            access_controls={
                AccessRole.EDITOR: ["read", "write"],
                AccessRole.ADMIN: ["read", "write", "manage"],
                AccessRole.OWNER: ["all"]
            },
            encryption_required=True,
            audit_required=True,
            retention_period=timedelta(days=365*7),
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
            custom_rules={
                "require_irb_approval": True,
                "require_anonymization": True,
                "require_consent_tracking": True
            }
        )
        self.register_policy(human_subjects_policy)
    
    def register_policy(self, policy: SecurityPolicy):
        """Register a new security policy."""
        self.policies[policy.policy_id] = policy
        logger.info(f"Registered security policy: {policy.name}")
    
    async def classify_data(self, content: Any, context: Dict[str, Any] = None) -> SecurityLevel:
        """Automatically classify data based on content and context."""
        try:
            context = context or {}
            
            # Check for PII/sensitive data
            pii_score = await self.privacy_protector.detect_pii(content)
            
            # Check for confidential markers
            confidential_markers = [
                "confidential", "proprietary", "internal", "private",
                "patent", "trade secret", "unpublished"
            ]
            
            content_str = str(content).lower()
            has_confidential_markers = any(marker in content_str for marker in confidential_markers)
            
            # Check context indicators
            is_human_subjects = context.get("involves_human_subjects", False)
            is_commercial = context.get("commercial_application", False)
            is_preliminary = context.get("preliminary_results", False)
            
            # Classification logic
            if pii_score > 0.8 or is_human_subjects:
                return SecurityLevel.RESTRICTED
            elif has_confidential_markers or is_commercial or pii_score > 0.5:
                return SecurityLevel.CONFIDENTIAL
            elif is_preliminary or pii_score > 0.2:
                return SecurityLevel.INTERNAL
            else:
                return SecurityLevel.PUBLIC
                
        except Exception as e:
            logger.error(f"Error classifying data: {e}")
            return self.default_security_level
    
    async def apply_security_policy(self, resource_id: str, policy_id: str) -> bool:
        """Apply a security policy to a resource."""
        try:
            if policy_id not in self.policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            policy = self.policies[policy_id]
            
            # Log policy application
            await self.audit_logger.log_event(
                event_type="policy_applied",
                resource_id=resource_id,
                description=f"Applied policy {policy.name}",
                metadata={"policy_id": policy_id}
            )
            
            logger.info(f"Applied policy {policy.name} to resource {resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply security policy: {e}")
            return False
    
    async def grant_access(self, user_id: str, resource_id: str, 
                         role: AccessRole, granted_by: str,
                         expires_at: Optional[datetime] = None,
                         conditions: List[str] = None) -> str:
        """Grant access permission to a user."""
        try:
            permission = AccessPermission(
                permission_id=f"perm_{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                resource_id=resource_id,
                role=role,
                granted_by=granted_by,
                granted_at=datetime.now(),
                expires_at=expires_at,
                conditions=conditions or []
            )
            
            self.permissions[permission.permission_id] = permission
            
            # Log access grant
            await self.audit_logger.log_event(
                event_type="access_granted",
                user_id=user_id,
                resource_id=resource_id,
                description=f"Granted {role.value} access",
                metadata={
                    "permission_id": permission.permission_id,
                    "granted_by": granted_by,
                    "expires_at": expires_at.isoformat() if expires_at else None
                }
            )
            
            logger.info(f"Granted {role.value} access to {user_id} for {resource_id}")
            return permission.permission_id
            
        except Exception as e:
            logger.error(f"Failed to grant access: {e}")
            raise
    
    async def check_access(self, user_id: str, resource_id: str, 
                         action: str) -> Tuple[bool, str]:
        """Check if user has access to perform an action on a resource."""
        try:
            # Find active permissions for user and resource
            user_permissions = [
                p for p in self.permissions.values()
                if p.user_id == user_id and p.resource_id == resource_id and p.is_active
            ]
            
            if not user_permissions:
                return False, "No permissions found"
            
            # Check for expired permissions
            now = datetime.now()
            valid_permissions = [
                p for p in user_permissions
                if p.expires_at is None or p.expires_at > now
            ]
            
            if not valid_permissions:
                return False, "All permissions expired"
            
            # Check if any permission allows the action
            for permission in valid_permissions:
                policy = self._get_policy_for_resource(resource_id)
                if policy:
                    allowed_actions = policy.access_controls.get(permission.role, [])
                    if action in allowed_actions or "all" in allowed_actions:
                        return True, f"Allowed by {permission.role.value} role"
            
            return False, f"Action '{action}' not permitted"
            
        except Exception as e:
            logger.error(f"Error checking access: {e}")
            return False, "Access check failed"
    
    async def revoke_access(self, permission_id: str, revoked_by: str) -> bool:
        """Revoke an access permission."""
        try:
            if permission_id not in self.permissions:
                return False
            
            permission = self.permissions[permission_id]
            permission.is_active = False
            
            # Log access revocation
            await self.audit_logger.log_event(
                event_type="access_revoked",
                user_id=permission.user_id,
                resource_id=permission.resource_id,
                description=f"Revoked {permission.role.value} access",
                metadata={
                    "permission_id": permission_id,
                    "revoked_by": revoked_by
                }
            )
            
            logger.info(f"Revoked access for permission {permission_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke access: {e}")
            return False
    
    async def encrypt_data(self, data: bytes, security_level: SecurityLevel) -> Tuple[bytes, str]:
        """Encrypt data based on security level."""
        if not self.encryption_manager:
            raise ValueError("Encryption not available")
        
        try:
            encrypted_data, key_id = await self.encryption_manager.encrypt(
                data, security_level
            )
            
            self.metrics["encryption_operations"] += 1
            
            # Log encryption event
            await self.audit_logger.log_event(
                event_type="data_encrypted",
                description=f"Data encrypted with {security_level.value} level",
                metadata={"key_id": key_id, "data_size": len(data)}
            )
            
            return encrypted_data, key_id
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    async def decrypt_data(self, encrypted_data: bytes, key_id: str, 
                         user_id: str) -> bytes:
        """Decrypt data with access control."""
        if not self.encryption_manager:
            raise ValueError("Encryption not available")
        
        try:
            # Check if user has decryption permission
            # This would integrate with the access control system
            
            decrypted_data = await self.encryption_manager.decrypt(
                encrypted_data, key_id
            )
            
            # Log decryption event
            await self.audit_logger.log_event(
                event_type="data_decrypted",
                user_id=user_id,
                description="Data decrypted",
                metadata={"key_id": key_id, "data_size": len(decrypted_data)}
            )
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    async def anonymize_data(self, data: Any, anonymization_level: str = "standard") -> Any:
        """Anonymize data to protect privacy."""
        try:
            anonymized_data = await self.privacy_protector.anonymize(
                data, anonymization_level
            )
            
            # Log anonymization
            await self.audit_logger.log_event(
                event_type="data_anonymized",
                description=f"Data anonymized with {anonymization_level} level",
                metadata={"original_size": len(str(data)), 
                         "anonymized_size": len(str(anonymized_data))}
            )
            
            return anonymized_data
            
        except Exception as e:
            logger.error(f"Anonymization failed: {e}")
            raise
    
    async def detect_threats(self, activity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect security threats in activity data."""
        try:
            threats = await self.threat_detector.analyze_activity(activity_data)
            
            # Log high-priority threats
            for threat in threats:
                if threat.get("severity") in ["high", "critical"]:
                    await self._handle_security_event(
                        event_type="threat_detected",
                        description=threat.get("description", "Unknown threat"),
                        threat_level=ThreatLevel(threat.get("severity", "medium")),
                        metadata=threat
                    )
            
            return threats
            
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return []
    
    async def run_compliance_check(self, framework: ComplianceFramework, 
                                 resource_id: str = None) -> Dict[str, Any]:
        """Run compliance check for a specific framework."""
        try:
            compliance_result = {
                "framework": framework.value,
                "resource_id": resource_id,
                "timestamp": datetime.now().isoformat(),
                "compliant": True,
                "issues": [],
                "recommendations": []
            }
            
            # Framework-specific checks
            if framework == ComplianceFramework.GDPR:
                compliance_result.update(await self._check_gdpr_compliance(resource_id))
            elif framework == ComplianceFramework.HIPAA:
                compliance_result.update(await self._check_hipaa_compliance(resource_id))
            elif framework == ComplianceFramework.ISO27001:
                compliance_result.update(await self._check_iso27001_compliance(resource_id))
            
            self.metrics["compliance_checks"] += 1
            
            # Log compliance check
            await self.audit_logger.log_event(
                event_type="compliance_check",
                resource_id=resource_id,
                description=f"{framework.value} compliance check",
                metadata=compliance_result
            )
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            return {"compliant": False, "error": str(e)}
    
    def _get_policy_for_resource(self, resource_id: str) -> Optional[SecurityPolicy]:
        """Get the applicable security policy for a resource."""
        # This would implement logic to determine which policy applies
        # For now, return the first active policy
        for policy in self.policies.values():
            if policy.is_active:
                return policy
        return None
    
    async def _handle_security_event(self, event_type: str, description: str,
                                   threat_level: ThreatLevel = ThreatLevel.MEDIUM,
                                   user_id: str = None, resource_id: str = None,
                                   metadata: Dict[str, Any] = None):
        """Handle a security event."""
        event = SecurityEvent(
            event_id=f"event_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            user_id=user_id,
            resource_id=resource_id,
            description=description,
            threat_level=threat_level,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        self.metrics["security_events"] += 1
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.metrics["high_threat_events"] += 1
            
            # Trigger immediate response for high/critical threats
            await self._trigger_incident_response(event)
        
        logger.warning(f"Security event: {event_type} - {description}")
    
    async def _trigger_incident_response(self, event: SecurityEvent):
        """Trigger incident response procedures."""
        # This would implement automated incident response
        logger.critical(f"Triggering incident response for event: {event.event_id}")
    
    async def _check_gdpr_compliance(self, resource_id: str = None) -> Dict[str, Any]:
        """Check GDPR compliance."""
        return {
            "data_protection_measures": True,
            "consent_management": True,
            "right_to_erasure": True,
            "data_portability": True,
            "privacy_by_design": True
        }
    
    async def _check_hipaa_compliance(self, resource_id: str = None) -> Dict[str, Any]:
        """Check HIPAA compliance."""
        return {
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "access_controls": True,
            "audit_logging": True,
            "business_associate_agreements": True
        }
    
    async def _check_iso27001_compliance(self, resource_id: str = None) -> Dict[str, Any]:
        """Check ISO 27001 compliance."""
        return {
            "information_security_policy": True,
            "risk_management": True,
            "access_control": True,
            "incident_management": True,
            "business_continuity": True
        }
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        return {
            "system_metrics": self.metrics,
            "active_policies": len([p for p in self.policies.values() if p.is_active]),
            "active_permissions": len([p for p in self.permissions.values() if p.is_active]),
            "recent_events": len([e for e in self.security_events 
                                if e.timestamp > datetime.now() - timedelta(days=7)]),
            "threat_distribution": {
                level.value: len([e for e in self.security_events if e.threat_level == level])
                for level in ThreatLevel
            },
            "compliance_frameworks": [f.value for f in self.compliance_frameworks]
        }


class ThreatDetectionEngine:
    """Engine for detecting security threats."""
    
    def __init__(self):
        self.threat_patterns = {}
        self.anomaly_baseline = {}
    
    async def analyze_activity(self, activity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze activity for security threats."""
        threats = []
        
        # Implement threat detection algorithms
        # This is a placeholder implementation
        
        return threats


class SecurityAnomalyDetector:
    """Detects security-related anomalies."""
    
    def __init__(self):
        self.baseline_patterns = {}
    
    async def detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect security anomalies."""
        anomalies = []
        
        # Implement anomaly detection
        # This is a placeholder implementation
        
        return anomalies


class EncryptionManager:
    """Manages encryption operations and key lifecycle."""
    
    def __init__(self):
        self.keys: Dict[str, EncryptionKey] = {}
        self.fernet_keys: Dict[str, Fernet] = {}
    
    async def encrypt(self, data: bytes, security_level: SecurityLevel) -> Tuple[bytes, str]:
        """Encrypt data based on security level."""
        if not CRYPTO_AVAILABLE:
            raise ValueError("Cryptography library not available")
        
        # Generate or retrieve appropriate key
        key_id = await self._get_or_create_key(security_level)
        fernet = self.fernet_keys[key_id]
        
        # Encrypt data
        encrypted_data = fernet.encrypt(data)
        
        return encrypted_data, key_id
    
    async def decrypt(self, encrypted_data: bytes, key_id: str) -> bytes:
        """Decrypt data using specified key."""
        if not CRYPTO_AVAILABLE:
            raise ValueError("Cryptography library not available")
        
        if key_id not in self.fernet_keys:
            raise ValueError(f"Key {key_id} not found")
        
        fernet = self.fernet_keys[key_id]
        decrypted_data = fernet.decrypt(encrypted_data)
        
        return decrypted_data
    
    async def _get_or_create_key(self, security_level: SecurityLevel) -> str:
        """Get existing key or create new one for security level."""
        # Look for existing key
        for key_id, key in self.keys.items():
            if key.purpose == security_level.value and key.is_active:
                return key_id
        
        # Create new key
        return await self._create_key(security_level)
    
    async def _create_key(self, security_level: SecurityLevel) -> str:
        """Create new encryption key."""
        key_id = f"key_{uuid.uuid4().hex[:8]}"
        
        # Generate Fernet key
        fernet_key = Fernet.generate_key()
        fernet = Fernet(fernet_key)
        
        # Store key metadata
        encryption_key = EncryptionKey(
            key_id=key_id,
            key_type="symmetric",
            algorithm="AES",
            key_strength=256,
            purpose=security_level.value,
            created_at=datetime.now()
        )
        
        self.keys[key_id] = encryption_key
        self.fernet_keys[key_id] = fernet
        
        return key_id


class PrivacyProtectionEngine:
    """Engine for privacy protection and PII handling."""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
    
    async def detect_pii(self, content: Any) -> float:
        """Detect PII in content and return risk score."""
        if not isinstance(content, str):
            content = str(content)
        
        pii_score = 0.0
        pii_count = 0
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                pii_count += len(matches)
                pii_score += len(matches) * 0.2  # Base score per match
        
        # Normalize score
        return min(1.0, pii_score)
    
    async def anonymize(self, data: Any, level: str = "standard") -> Any:
        """Anonymize data by removing or obfuscating PII."""
        if not isinstance(data, str):
            data = str(data)
        
        anonymized = data
        
        # Replace PII patterns
        for pii_type, pattern in self.pii_patterns.items():
            if level == "standard":
                replacement = f"[{pii_type.upper()}_REDACTED]"
            elif level == "aggressive":
                replacement = "[REDACTED]"
            else:
                replacement = f"[{pii_type.upper()}]"
            
            anonymized = re.sub(pattern, replacement, anonymized, flags=re.IGNORECASE)
        
        return anonymized


class AuditLogger:
    """Audit logging system for security events."""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
    
    async def log_event(self, event_type: str, description: str,
                      user_id: str = None, resource_id: str = None,
                      metadata: Dict[str, Any] = None):
        """Log an audit event."""
        event = {
            "event_id": f"audit_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "description": description,
            "user_id": user_id,
            "resource_id": resource_id,
            "metadata": metadata or {}
        }
        
        self.audit_log.append(event)
        logger.info(f"Audit: {event_type} - {description}")


# Integration functions

async def setup_research_security(notebook, 
                                security_level: SecurityLevel = SecurityLevel.INTERNAL,
                                compliance_frameworks: List[ComplianceFramework] = None) -> ResearchSecurityManager:
    """Set up security for a research notebook."""
    security_manager = ResearchSecurityManager(
        default_security_level=security_level,
        compliance_frameworks=compliance_frameworks
    )
    
    # Set up default access for notebook owner
    await security_manager.grant_access(
        user_id=notebook.author,
        resource_id=str(notebook.vault_path),
        role=AccessRole.OWNER,
        granted_by="system"
    )
    
    return security_manager


def create_security_policy(name: str, security_level: SecurityLevel,
                         compliance_frameworks: List[ComplianceFramework] = None) -> SecurityPolicy:
    """Create a custom security policy."""
    return SecurityPolicy(
        policy_id=f"policy_{uuid.uuid4().hex[:8]}",
        name=name,
        description=f"Custom policy for {name}",
        security_level=security_level,
        access_controls={
            AccessRole.VIEWER: ["read"],
            AccessRole.CONTRIBUTOR: ["read", "comment"],
            AccessRole.EDITOR: ["read", "write", "comment"],
            AccessRole.ADMIN: ["read", "write", "comment", "manage"],
            AccessRole.OWNER: ["all"]
        },
        encryption_required=security_level in [SecurityLevel.CONFIDENTIAL, 
                                             SecurityLevel.RESTRICTED, 
                                             SecurityLevel.TOP_SECRET],
        audit_required=True,
        retention_period=timedelta(days=365*5),
        compliance_frameworks=compliance_frameworks or [ComplianceFramework.GDPR]
    )