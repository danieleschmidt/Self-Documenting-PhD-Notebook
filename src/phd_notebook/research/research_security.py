"""
Research Security Module
========================

Comprehensive security measures for research data protection,
intellectual property safeguarding, and compliance with academic integrity standards.

Features:
- Data encryption and secure storage
- Intellectual property protection
- Research integrity validation
- Compliance monitoring (GDPR, FERPA, etc.)
- Secure collaboration protocols
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security classification levels for research data."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class IntegrityLevel(Enum):
    """Research integrity classification."""
    PRELIMINARY = "preliminary"
    PEER_REVIEWED = "peer_reviewed"
    VALIDATED = "validated"
    PUBLISHED = "published"


@dataclass
class SecurityContext:
    """Security context for research operations."""
    user_id: str
    project_id: str
    security_level: SecurityLevel
    integrity_level: IntegrityLevel
    permissions: Set[str]
    session_token: str
    expires_at: datetime
    ip_address: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if security context is still valid."""
        return datetime.now() < self.expires_at
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions


class ResearchSecurityValidator:
    """Validates research data integrity and authenticity."""
    
    def __init__(self):
        self.known_patterns = {
            'plagiarism_indicators': [
                'copy', 'paste', 'duplicate', 'identical', 'verbatim'
            ],
            'data_fabrication_patterns': [
                'too_perfect_correlation', 'unrealistic_precision', 
                'missing_outliers', 'statistical_impossibilities'
            ],
            'integrity_violations': [
                'data_manipulation', 'selective_reporting', 
                'hypothesis_switching', 'cherry_picking'
            ]
        }
    
    def validate_research_integrity(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive research integrity validation."""
        validation_result = {
            'status': 'PASS',
            'score': 1.0,
            'violations': [],
            'warnings': [],
            'recommendations': [],
            'validated_at': datetime.now().isoformat()
        }
        
        try:
            # Check for plagiarism indicators
            plagiarism_score = self._check_plagiarism_indicators(research_data)
            if plagiarism_score > 0.3:
                validation_result['violations'].append({
                    'type': 'potential_plagiarism',
                    'severity': 'high',
                    'score': plagiarism_score,
                    'description': 'Text similarity patterns suggest potential plagiarism'
                })
                validation_result['status'] = 'WARNING'
                validation_result['score'] -= 0.3
            
            # Validate statistical integrity
            stats_issues = self._validate_statistical_integrity(research_data)
            if stats_issues:
                validation_result['violations'].extend(stats_issues)
                validation_result['status'] = 'WARNING'
                validation_result['score'] -= 0.2
            
            # Check data consistency
            consistency_issues = self._check_data_consistency(research_data)
            if consistency_issues:
                validation_result['warnings'].extend(consistency_issues)
                validation_result['score'] -= 0.1
            
            # Generate recommendations
            validation_result['recommendations'] = self._generate_integrity_recommendations(
                validation_result
            )
            
        except Exception as e:
            logger.error(f"Integrity validation failed: {e}")
            validation_result.update({
                'status': 'ERROR',
                'score': 0.0,
                'violations': [{'type': 'validation_error', 'description': str(e)}]
            })
        
        return validation_result
    
    def _check_plagiarism_indicators(self, data: Dict[str, Any]) -> float:
        """Check for plagiarism indicators in research content."""
        content = str(data.get('content', '')) + str(data.get('title', ''))
        if not content:
            return 0.0
        
        indicator_count = 0
        total_words = len(content.split())
        
        for pattern in self.known_patterns['plagiarism_indicators']:
            indicator_count += content.lower().count(pattern)
        
        # Simple heuristic: excessive use of copy/paste indicators
        return min(indicator_count / max(total_words, 1), 1.0)
    
    def _validate_statistical_integrity(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate statistical aspects of research data."""
        issues = []
        
        # Check for statistical results
        results = data.get('results', {})
        if not results:
            return issues
        
        # Check for suspiciously perfect correlations
        correlations = results.get('correlations', [])
        for corr in correlations:
            if isinstance(corr, (int, float)) and abs(corr) > 0.99:
                issues.append({
                    'type': 'suspicious_correlation',
                    'severity': 'medium',
                    'value': corr,
                    'description': f'Correlation of {corr:.3f} is suspiciously high'
                })
        
        # Check for unrealistic p-values
        p_values = results.get('p_values', [])
        if p_values:
            perfect_pvals = [p for p in p_values if p == 0.05]
            if len(perfect_pvals) > len(p_values) * 0.5:
                issues.append({
                    'type': 'suspicious_pvalues',
                    'severity': 'high',
                    'description': 'Too many p-values exactly at 0.05 threshold'
                })
        
        return issues
    
    def _check_data_consistency(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for data consistency issues."""
        warnings = []
        
        # Check metadata consistency
        metadata = data.get('metadata', {})
        if metadata:
            # Check date consistency
            created = metadata.get('created')
            modified = metadata.get('modified')
            
            if created and modified:
                try:
                    created_dt = datetime.fromisoformat(created)
                    modified_dt = datetime.fromisoformat(modified)
                    
                    if created_dt > modified_dt:
                        warnings.append({
                            'type': 'date_inconsistency',
                            'description': 'Created date is after modified date'
                        })
                except ValueError:
                    warnings.append({
                        'type': 'invalid_dates',
                        'description': 'Invalid date format in metadata'
                    })
        
        return warnings
    
    def _generate_integrity_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        violations = validation_result.get('violations', [])
        
        for violation in violations:
            if violation['type'] == 'potential_plagiarism':
                recommendations.append(
                    "Review text for originality and add proper citations"
                )
            elif violation['type'] == 'suspicious_correlation':
                recommendations.append(
                    "Verify statistical calculations and check for data errors"
                )
            elif violation['type'] == 'suspicious_pvalues':
                recommendations.append(
                    "Review statistical analysis methodology and significance testing"
                )
        
        if validation_result['score'] < 0.8:
            recommendations.append(
                "Consider peer review before publication"
            )
        
        return recommendations


class SecureResearchStorage:
    """Secure storage system for sensitive research data."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.encryption_key = self._generate_encryption_key()
        
    def _generate_encryption_key(self) -> str:
        """Generate a secure encryption key."""
        return secrets.token_hex(32)
    
    def _encrypt_data(self, data: str) -> str:
        """Simple XOR encryption (for demonstration - use proper encryption in production)."""
        key_bytes = self.encryption_key.encode()
        data_bytes = data.encode()
        
        encrypted = []
        for i, byte in enumerate(data_bytes):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return ''.join(format(b, '02x') for b in encrypted)
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Simple XOR decryption."""
        try:
            # Convert hex string back to bytes
            encrypted_bytes = [int(encrypted_data[i:i+2], 16) 
                             for i in range(0, len(encrypted_data), 2)]
            
            key_bytes = self.encryption_key.encode()
            
            decrypted = []
            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
            
            return bytes(decrypted).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return ""
    
    def store_secure_data(self, data_id: str, data: Dict[str, Any], 
                         security_level: SecurityLevel) -> bool:
        """Store data securely based on security level."""
        try:
            # Create secure filename
            safe_filename = hashlib.sha256(data_id.encode()).hexdigest()
            file_path = self.storage_path / f"{safe_filename}.enc"
            
            # Prepare data with metadata
            secure_data = {
                'id': data_id,
                'data': data,
                'security_level': security_level.value,
                'created_at': datetime.now().isoformat(),
                'checksum': self._calculate_checksum(data)
            }
            
            # Encrypt and store
            json_data = json.dumps(secure_data)
            if security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]:
                encrypted_data = self._encrypt_data(json_data)
                with open(file_path, 'w') as f:
                    f.write(encrypted_data)
            else:
                with open(file_path, 'w') as f:
                    f.write(json_data)
            
            logger.info(f"Data {data_id} stored securely at level {security_level.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secure data: {e}")
            return False
    
    def retrieve_secure_data(self, data_id: str, context: SecurityContext) -> Optional[Dict[str, Any]]:
        """Retrieve data with security validation."""
        try:
            safe_filename = hashlib.sha256(data_id.encode()).hexdigest()
            file_path = self.storage_path / f"{safe_filename}.enc"
            
            if not file_path.exists():
                return None
            
            # Read and decrypt if necessary
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Try to parse as JSON first (unencrypted)
            try:
                secure_data = json.loads(content)
            except json.JSONDecodeError:
                # Must be encrypted
                if not context.has_permission('decrypt_confidential'):
                    logger.warning(f"Access denied for encrypted data {data_id}")
                    return None
                
                decrypted_content = self._decrypt_data(content)
                secure_data = json.loads(decrypted_content)
            
            # Validate security level access
            data_security_level = SecurityLevel(secure_data['security_level'])
            if not self._can_access_level(context, data_security_level):
                logger.warning(f"Insufficient permissions for {data_id}")
                return None
            
            # Verify data integrity
            if not self._verify_checksum(secure_data):
                logger.error(f"Checksum validation failed for {data_id}")
                return None
            
            return secure_data['data']
            
        except Exception as e:
            logger.error(f"Failed to retrieve secure data: {e}")
            return None
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _verify_checksum(self, secure_data: Dict[str, Any]) -> bool:
        """Verify data integrity using checksum."""
        stored_checksum = secure_data.get('checksum')
        calculated_checksum = self._calculate_checksum(secure_data['data'])
        return stored_checksum == calculated_checksum
    
    def _can_access_level(self, context: SecurityContext, data_level: SecurityLevel) -> bool:
        """Check if user can access data at given security level."""
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3
        }
        
        user_level = context.security_level
        required_level = data_level
        
        return level_hierarchy[user_level] >= level_hierarchy[required_level]


class ComplianceMonitor:
    """Monitor compliance with research regulations and standards."""
    
    def __init__(self):
        self.regulations = {
            'GDPR': {
                'data_retention_months': 36,
                'requires_consent': True,
                'anonymization_required': True,
                'geographic_restriction': 'EU'
            },
            'FERPA': {
                'data_retention_months': 60,
                'requires_consent': True,
                'educational_records_only': True,
                'geographic_restriction': 'US'
            },
            'HIPAA': {
                'data_retention_months': 72,
                'requires_consent': True,
                'health_records_only': True,
                'encryption_required': True
            }
        }
    
    def check_compliance(self, project_data: Dict[str, Any], 
                        applicable_regulations: List[str]) -> Dict[str, Any]:
        """Check compliance with specified regulations."""
        compliance_report = {
            'status': 'COMPLIANT',
            'violations': [],
            'warnings': [],
            'recommendations': [],
            'checked_regulations': applicable_regulations,
            'check_timestamp': datetime.now().isoformat()
        }
        
        for regulation in applicable_regulations:
            if regulation not in self.regulations:
                compliance_report['warnings'].append(f"Unknown regulation: {regulation}")
                continue
            
            reg_config = self.regulations[regulation]
            violations = self._check_regulation_compliance(project_data, regulation, reg_config)
            
            if violations:
                compliance_report['violations'].extend(violations)
                compliance_report['status'] = 'NON_COMPLIANT'
        
        # Generate recommendations
        compliance_report['recommendations'] = self._generate_compliance_recommendations(
            compliance_report
        )
        
        return compliance_report
    
    def _check_regulation_compliance(self, project_data: Dict[str, Any], 
                                   regulation: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check compliance for a specific regulation."""
        violations = []
        
        # Check data retention
        if 'created_at' in project_data:
            try:
                created_date = datetime.fromisoformat(project_data['created_at'])
                retention_period = timedelta(days=config['data_retention_months'] * 30)
                
                if datetime.now() - created_date > retention_period:
                    violations.append({
                        'regulation': regulation,
                        'type': 'data_retention_exceeded',
                        'description': f"Data older than {config['data_retention_months']} months",
                        'severity': 'high'
                    })
            except ValueError:
                violations.append({
                    'regulation': regulation,
                    'type': 'invalid_date',
                    'description': 'Invalid creation date format',
                    'severity': 'medium'
                })
        
        # Check consent requirements
        if config.get('requires_consent') and not project_data.get('consent_obtained'):
            violations.append({
                'regulation': regulation,
                'type': 'missing_consent',
                'description': 'Required consent not documented',
                'severity': 'high'
            })
        
        # Check encryption requirements
        if config.get('encryption_required') and not project_data.get('encrypted'):
            violations.append({
                'regulation': regulation,
                'type': 'missing_encryption',
                'description': 'Data encryption required but not implemented',
                'severity': 'high'
            })
        
        # Check anonymization
        if config.get('anonymization_required') and not project_data.get('anonymized'):
            violations.append({
                'regulation': regulation,
                'type': 'missing_anonymization',
                'description': 'Data anonymization required',
                'severity': 'medium'
            })
        
        return violations
    
    def _generate_compliance_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on compliance violations."""
        recommendations = []
        
        violations = report.get('violations', [])
        violation_types = {v['type'] for v in violations}
        
        if 'data_retention_exceeded' in violation_types:
            recommendations.append("Archive or delete data that exceeds retention periods")
        
        if 'missing_consent' in violation_types:
            recommendations.append("Document participant consent for data use")
        
        if 'missing_encryption' in violation_types:
            recommendations.append("Implement data encryption for sensitive information")
        
        if 'missing_anonymization' in violation_types:
            recommendations.append("Anonymize personal data before analysis")
        
        if report['status'] == 'NON_COMPLIANT':
            recommendations.append("Consult with legal/compliance team before proceeding")
        
        return recommendations


class ResearchSecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self, storage_path: Path):
        self.validator = ResearchSecurityValidator()
        self.storage = SecureResearchStorage(storage_path)
        self.compliance = ComplianceMonitor()
        self.active_sessions: Dict[str, SecurityContext] = {}
    
    def create_security_context(self, user_id: str, project_id: str, 
                              permissions: Set[str]) -> SecurityContext:
        """Create and register a new security context."""
        session_token = secrets.token_urlsafe(32)
        
        context = SecurityContext(
            user_id=user_id,
            project_id=project_id,
            security_level=SecurityLevel.INTERNAL,  # Default level
            integrity_level=IntegrityLevel.PRELIMINARY,
            permissions=permissions,
            session_token=session_token,
            expires_at=datetime.now() + timedelta(hours=8)
        )
        
        self.active_sessions[session_token] = context
        return context
    
    def validate_context(self, session_token: str) -> Optional[SecurityContext]:
        """Validate and return security context."""
        context = self.active_sessions.get(session_token)
        
        if not context or not context.is_valid():
            if context:
                del self.active_sessions[session_token]
            return None
        
        return context
    
    def secure_research_operation(self, operation_name: str, data: Dict[str, Any], 
                                context: SecurityContext) -> Dict[str, Any]:
        """Perform a secure research operation with full validation."""
        operation_result = {
            'operation': operation_name,
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'user_id': context.user_id,
            'project_id': context.project_id
        }
        
        try:
            # Validate security context
            if not context.is_valid():
                raise SecurityError("Invalid or expired security context")
            
            # Validate research integrity
            integrity_result = self.validator.validate_research_integrity(data)
            operation_result['integrity_validation'] = integrity_result
            
            if integrity_result['status'] == 'ERROR':
                raise SecurityError("Research integrity validation failed")
            
            # Check compliance if applicable
            regulations = data.get('applicable_regulations', [])
            if regulations:
                compliance_result = self.compliance.check_compliance(data, regulations)
                operation_result['compliance_check'] = compliance_result
                
                if compliance_result['status'] == 'NON_COMPLIANT':
                    logger.warning(f"Compliance violations detected: {compliance_result['violations']}")
            
            # Store data securely
            data_id = f"{context.project_id}_{operation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            security_level = SecurityLevel(data.get('security_level', 'internal'))
            
            storage_success = self.storage.store_secure_data(data_id, data, security_level)
            
            if not storage_success:
                raise SecurityError("Failed to store data securely")
            
            operation_result['data_id'] = data_id
            operation_result['security_level'] = security_level.value
            
        except Exception as e:
            logger.error(f"Secure operation failed: {e}")
            operation_result.update({
                'status': 'ERROR',
                'error': str(e)
            })
        
        return operation_result


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


# Security utility functions
def generate_research_audit_log(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive audit log for research operations."""
    audit_log = {
        'log_created': datetime.now().isoformat(),
        'total_operations': len(operations),
        'security_summary': {
            'high_security_operations': 0,
            'integrity_violations': 0,
            'compliance_issues': 0,
            'successful_operations': 0
        },
        'operations': operations,
        'recommendations': []
    }
    
    for operation in operations:
        if operation.get('status') == 'SUCCESS':
            audit_log['security_summary']['successful_operations'] += 1
        
        security_level = operation.get('security_level')
        if security_level in ['confidential', 'restricted']:
            audit_log['security_summary']['high_security_operations'] += 1
        
        integrity = operation.get('integrity_validation', {})
        if integrity.get('violations'):
            audit_log['security_summary']['integrity_violations'] += len(integrity['violations'])
        
        compliance = operation.get('compliance_check', {})
        if compliance.get('violations'):
            audit_log['security_summary']['compliance_issues'] += len(compliance['violations'])
    
    # Generate recommendations
    if audit_log['security_summary']['integrity_violations'] > 0:
        audit_log['recommendations'].append("Review research integrity practices")
    
    if audit_log['security_summary']['compliance_issues'] > 0:
        audit_log['recommendations'].append("Address compliance violations immediately")
    
    return audit_log


# Example usage and testing
if __name__ == "__main__":
    # Example security workflow
    storage_path = Path("secure_research_data")
    security_manager = ResearchSecurityManager(storage_path)
    
    # Create security context
    permissions = {'read_data', 'write_data', 'decrypt_confidential'}
    context = security_manager.create_security_context("researcher_001", "project_ml", permissions)
    
    # Sample research data
    research_data = {
        'title': 'Machine Learning Model Performance',
        'content': 'Novel approach to transformer architecture optimization',
        'results': {
            'correlations': [0.85, 0.92, 0.78],
            'p_values': [0.03, 0.01, 0.045]
        },
        'security_level': 'confidential',
        'applicable_regulations': ['GDPR'],
        'created_at': datetime.now().isoformat(),
        'consent_obtained': True,
        'encrypted': True,
        'anonymized': True
    }
    
    # Perform secure operation
    result = security_manager.secure_research_operation("data_analysis", research_data, context)
    
    print("🔒 Security Operation Result:")
    print(f"Status: {result['status']}")
    print(f"Data ID: {result.get('data_id', 'N/A')}")
    print(f"Integrity Score: {result.get('integrity_validation', {}).get('score', 'N/A')}")
    
    compliance = result.get('compliance_check')
    if compliance:
        print(f"Compliance Status: {compliance['status']}")