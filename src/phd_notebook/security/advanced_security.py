"""
Advanced security framework for the PhD notebook system.
Implements comprehensive security measures including encryption, access control,
audit logging, and security scanning.
"""

import hashlib
import secrets
import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..utils.exceptions import SecurityError, AuthenticationError, AuthorizationError
from ..monitoring.metrics import MetricsCollector


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class PermissionType(Enum):
    """Permission types for access control."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    permissions: List[PermissionType]
    security_level: SecurityLevel
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class AuditEvent:
    """Audit log event."""
    event_id: str
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, denied
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class EncryptionManager:
    """Handles encryption and decryption operations."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.logger = logging.getLogger("encryption_manager")
        
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = self._generate_master_key()
        
        self.cipher_suite = Fernet(self.master_key)
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master encryption key."""
        return Fernet.generate_key()
    
    def derive_key_from_password(self, password: str, salt: bytes = None) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data using the master key."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.cipher_suite.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using the master key."""
        return self.cipher_suite.decrypt(encrypted_data)
    
    def encrypt_sensitive_fields(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Encrypt specific fields in a dictionary."""
        result = data.copy()
        
        for field in sensitive_fields:
            if field in result:
                value = result[field]
                if value is not None:
                    encrypted_value = self.encrypt_data(str(value))
                    result[field] = base64.b64encode(encrypted_value).decode('utf-8')
        
        return result
    
    def decrypt_sensitive_fields(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Decrypt specific fields in a dictionary."""
        result = data.copy()
        
        for field in sensitive_fields:
            if field in result:
                encrypted_value = result[field]
                if encrypted_value is not None:
                    try:
                        decoded_value = base64.b64decode(encrypted_value.encode('utf-8'))
                        decrypted_value = self.decrypt_data(decoded_value)
                        result[field] = decrypted_value.decode('utf-8')
                    except Exception as e:
                        self.logger.error(f"Failed to decrypt field {field}: {e}")
                        result[field] = None
        
        return result


class AccessControlManager:
    """Manages access control and permissions."""
    
    def __init__(self):
        self.logger = logging.getLogger("access_control")
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def create_session(
        self,
        user_id: str,
        permissions: List[PermissionType],
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        session_duration: timedelta = timedelta(hours=8),
        ip_address: str = None,
        user_agent: str = None
    ) -> str:
        """Create a new authenticated session."""
        if self._is_user_locked_out(user_id):
            raise AuthenticationError(f"User {user_id} is temporarily locked out")
        
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + session_duration
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            permissions=permissions,
            security_level=security_level,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
        
        self.active_sessions[session_id] = context
        self.logger.info(f"Created session {session_id} for user {user_id}")
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate and return security context for a session."""
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        
        # Check if session has expired
        if context.expires_at and datetime.now() > context.expires_at:
            self.revoke_session(session_id)
            return None
        
        return context
    
    def revoke_session(self, session_id: str):
        """Revoke an active session."""
        if session_id in self.active_sessions:
            context = self.active_sessions.pop(session_id)
            self.logger.info(f"Revoked session {session_id} for user {context.user_id}")
    
    def check_permission(
        self,
        context: SecurityContext,
        required_permission: PermissionType,
        resource_security_level: SecurityLevel = SecurityLevel.INTERNAL
    ) -> bool:
        """Check if a security context has the required permission."""
        # Check if user has the required permission
        if required_permission not in context.permissions and PermissionType.ADMIN not in context.permissions:
            return False
        
        # Check security level access
        security_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3
        }
        
        user_level = security_hierarchy[context.security_level]
        resource_level = security_hierarchy[resource_security_level]
        
        return user_level >= resource_level
    
    def record_failed_attempt(self, user_id: str):
        """Record a failed authentication attempt."""
        now = datetime.now()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Clean old attempts (older than lockout duration)
        cutoff_time = now - self.lockout_duration
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > cutoff_time
        ]
        
        self.failed_attempts[user_id].append(now)
        self.logger.warning(f"Failed authentication attempt for user {user_id}")
    
    def _is_user_locked_out(self, user_id: str) -> bool:
        """Check if a user is currently locked out."""
        if user_id not in self.failed_attempts:
            return False
        
        recent_attempts = len(self.failed_attempts[user_id])
        return recent_attempts >= self.max_failed_attempts


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, log_file: Path = None):
        self.logger = logging.getLogger("audit_logger")
        self.log_file = log_file or Path("audit.log")
        self.metrics_collector = MetricsCollector()
    
    async def log_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        ip_address: str = None,
        user_agent: str = None,
        details: Dict[str, Any] = None
    ):
        """Log an audit event."""
        event = AuditEvent(
            event_id=secrets.token_hex(16),
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        
        # Log to file
        await self._write_audit_log(event)
        
        # Emit metrics
        self.metrics_collector.record_counter(f"audit.{action}.{result}")
        
        # Log security-relevant events
        if result == "denied" or action in ["login_failed", "permission_denied"]:
            self.logger.warning(
                f"Security event: {action} by {user_id} on {resource} - {result}"
            )
    
    async def _write_audit_log(self, event: AuditEvent):
        """Write audit event to log file."""
        log_entry = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "action": event.action,
            "resource": event.resource,
            "result": event.result,
            "ip_address": event.ip_address,
            "user_agent": event.user_agent,
            "details": event.details
        }
        
        try:
            async with aiofiles.open(self.log_file, "a") as f:
                await f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
    
    async def search_audit_logs(
        self,
        user_id: str = None,
        action: str = None,
        resource: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Search audit logs with filters."""
        events = []
        
        try:
            async with aiofiles.open(self.log_file, "r") as f:
                async for line in f:
                    try:
                        entry = json.loads(line.strip())
                        event = AuditEvent(
                            event_id=entry["event_id"],
                            user_id=entry["user_id"],
                            action=entry["action"],
                            resource=entry["resource"],
                            result=entry["result"],
                            timestamp=datetime.fromisoformat(entry["timestamp"]),
                            ip_address=entry.get("ip_address"),
                            user_agent=entry.get("user_agent"),
                            details=entry.get("details", {})
                        )
                        
                        # Apply filters
                        if user_id and event.user_id != user_id:
                            continue
                        if action and event.action != action:
                            continue
                        if resource and event.resource != resource:
                            continue
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue
                        
                        events.append(event)
                        
                        if len(events) >= limit:
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.error(f"Error searching audit logs: {e}")
        
        return events[::-1]  # Return most recent first


class SecurityScanner:
    """Security vulnerability scanner."""
    
    def __init__(self):
        self.logger = logging.getLogger("security_scanner")
        self.vulnerability_checks = [
            self._check_weak_passwords,
            self._check_exposed_secrets,
            self._check_insecure_configurations,
            self._check_outdated_dependencies,
            self._check_file_permissions
        ]
    
    async def run_security_scan(self, scan_path: Path = None) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        scan_path = scan_path or Path.cwd()
        results = {
            "scan_timestamp": datetime.now().isoformat(),
            "scan_path": str(scan_path),
            "vulnerabilities": [],
            "warnings": [],
            "passed_checks": []
        }
        
        for check in self.vulnerability_checks:
            try:
                check_result = await check(scan_path)
                
                if check_result["status"] == "vulnerability":
                    results["vulnerabilities"].extend(check_result["issues"])
                elif check_result["status"] == "warning":
                    results["warnings"].extend(check_result["issues"])
                else:
                    results["passed_checks"].append(check_result["check_name"])
                    
            except Exception as e:
                self.logger.error(f"Security check failed: {e}")
                results["warnings"].append({
                    "type": "scan_error",
                    "description": f"Security check failed: {e}"
                })
        
        # Calculate overall security score
        total_checks = len(self.vulnerability_checks)
        passed_checks = len(results["passed_checks"])
        vulnerability_count = len(results["vulnerabilities"])
        
        security_score = max(0, (passed_checks - vulnerability_count * 2) / total_checks * 100)
        results["security_score"] = security_score
        
        return results
    
    async def _check_weak_passwords(self, scan_path: Path) -> Dict[str, Any]:
        """Check for weak password patterns in configuration files."""
        issues = []
        
        # Common weak password patterns
        weak_patterns = [
            "password", "123456", "admin", "default", "changeme",
            "secret", "password123", "admin123"
        ]
        
        config_files = list(scan_path.glob("**/*.conf")) + list(scan_path.glob("**/*.config")) + list(scan_path.glob("**/*.ini"))
        
        for file_path in config_files:
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    
                for pattern in weak_patterns:
                    if pattern.lower() in content.lower():
                        issues.append({
                            "type": "weak_password",
                            "file": str(file_path),
                            "description": f"Potential weak password pattern '{pattern}' found"
                        })
                        
            except Exception:
                continue
        
        return {
            "check_name": "weak_passwords",
            "status": "vulnerability" if issues else "passed",
            "issues": issues
        }
    
    async def _check_exposed_secrets(self, scan_path: Path) -> Dict[str, Any]:
        """Check for exposed secrets in code and configuration."""
        issues = []
        
        # Common secret patterns
        secret_patterns = [
            r'api[_-]?key',
            r'secret[_-]?key',
            r'access[_-]?token',
            r'private[_-]?key',
            r'password',
            r'credentials'
        ]
        
        code_files = (
            list(scan_path.glob("**/*.py")) + 
            list(scan_path.glob("**/*.js")) + 
            list(scan_path.glob("**/*.yaml")) + 
            list(scan_path.glob("**/*.yml")) +
            list(scan_path.glob("**/*.json"))
        )
        
        for file_path in code_files:
            # Skip test files and virtual environments
            if "test" in str(file_path) or "venv" in str(file_path):
                continue
                
            try:
                async with aiofiles.open(file_path, "r") as f:
                    lines = await f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    for pattern in secret_patterns:
                        if pattern in line.lower() and "=" in line:
                            # Check if it looks like a real secret (not a variable name)
                            if len(line.split("=")[1].strip().strip('"\'')) > 8:
                                issues.append({
                                    "type": "exposed_secret",
                                    "file": str(file_path),
                                    "line": line_num,
                                    "description": f"Potential exposed secret matching pattern '{pattern}'"
                                })
                                
            except Exception:
                continue
        
        return {
            "check_name": "exposed_secrets",
            "status": "vulnerability" if issues else "passed",
            "issues": issues
        }
    
    async def _check_insecure_configurations(self, scan_path: Path) -> Dict[str, Any]:
        """Check for insecure configuration settings."""
        issues = []
        
        # Look for insecure settings
        insecure_settings = [
            ("debug = true", "Debug mode enabled in production"),
            ("ssl = false", "SSL/TLS disabled"),
            ("verify_ssl = false", "SSL verification disabled"),
            ("allow_anonymous = true", "Anonymous access allowed")
        ]
        
        config_files = list(scan_path.glob("**/*.conf")) + list(scan_path.glob("**/*.ini")) + list(scan_path.glob("**/*.yaml"))
        
        for file_path in config_files:
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    
                for setting, description in insecure_settings:
                    if setting.lower() in content.lower():
                        issues.append({
                            "type": "insecure_configuration",
                            "file": str(file_path),
                            "description": description
                        })
                        
            except Exception:
                continue
        
        return {
            "check_name": "insecure_configurations",
            "status": "warning" if issues else "passed",
            "issues": issues
        }
    
    async def _check_outdated_dependencies(self, scan_path: Path) -> Dict[str, Any]:
        """Check for potentially outdated dependencies."""
        issues = []
        
        # This is a simplified check - in production you'd integrate with vulnerability databases
        requirements_files = list(scan_path.glob("**/requirements*.txt")) + list(scan_path.glob("**/Pipfile"))
        
        for file_path in requirements_files:
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                    
                # Look for dependencies without version pinning
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and '==' not in line and '>=' not in line:
                        issues.append({
                            "type": "unpinned_dependency",
                            "file": str(file_path),
                            "description": f"Dependency '{line}' is not version-pinned"
                        })
                        
            except Exception:
                continue
        
        return {
            "check_name": "outdated_dependencies",
            "status": "warning" if issues else "passed",
            "issues": issues
        }
    
    async def _check_file_permissions(self, scan_path: Path) -> Dict[str, Any]:
        """Check for insecure file permissions."""
        issues = []
        
        # Check for world-writable files
        for file_path in scan_path.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    # Check if file is world-writable (mode 002)
                    if stat.st_mode & 0o002:
                        issues.append({
                            "type": "insecure_permissions",
                            "file": str(file_path),
                            "description": "File is world-writable"
                        })
                except Exception:
                    continue
        
        return {
            "check_name": "file_permissions",
            "status": "vulnerability" if issues else "passed",
            "issues": issues
        }


class SecurityManager:
    """Central security management system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("security_manager")
        
        self.encryption_manager = EncryptionManager()
        self.access_control = AccessControlManager()
        self.audit_logger = AuditLogger()
        self.security_scanner = SecurityScanner()
        
        self.metrics_collector = MetricsCollector()
    
    async def authenticate_and_authorize(
        self,
        user_id: str,
        required_permission: PermissionType,
        resource: str,
        resource_security_level: SecurityLevel = SecurityLevel.INTERNAL,
        session_id: str = None,
        ip_address: str = None,
        user_agent: str = None
    ) -> SecurityContext:
        """Authenticate user and check authorization for an operation."""
        
        # Validate session if provided
        if session_id:
            context = self.access_control.validate_session(session_id)
            if not context:
                await self.audit_logger.log_event(
                    user_id=user_id,
                    action="access_denied",
                    resource=resource,
                    result="invalid_session",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                raise AuthenticationError("Invalid or expired session")
        else:
            raise AuthenticationError("No session provided")
        
        # Check permissions
        if not self.access_control.check_permission(context, required_permission, resource_security_level):
            await self.audit_logger.log_event(
                user_id=user_id,
                action="access_denied",
                resource=resource,
                result="insufficient_permissions",
                ip_address=ip_address,
                user_agent=user_agent
            )
            raise AuthorizationError("Insufficient permissions")
        
        # Log successful access
        await self.audit_logger.log_event(
            user_id=user_id,
            action="access_granted",
            resource=resource,
            result="success",
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return context
    
    async def secure_operation(
        self,
        operation: Callable,
        user_id: str,
        required_permission: PermissionType,
        resource: str,
        session_id: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with full security controls."""
        
        # Authenticate and authorize
        context = await self.authenticate_and_authorize(
            user_id=user_id,
            required_permission=required_permission,
            resource=resource,
            session_id=session_id
        )
        
        try:
            # Execute operation
            result = await operation(*args, **kwargs)
            
            # Log successful operation
            await self.audit_logger.log_event(
                user_id=user_id,
                action=operation.__name__,
                resource=resource,
                result="success"
            )
            
            return result
            
        except Exception as e:
            # Log failed operation
            await self.audit_logger.log_event(
                user_id=user_id,
                action=operation.__name__,
                resource=resource,
                result="failure",
                details={"error": str(e)}
            )
            raise
    
    async def run_security_health_check(self) -> Dict[str, Any]:
        """Run comprehensive security health check."""
        
        # Run security scan
        scan_results = await self.security_scanner.run_security_scan()
        
        # Check active sessions
        active_sessions = len(self.access_control.active_sessions)
        
        # Check recent audit events
        recent_events = await self.audit_logger.search_audit_logs(
            start_time=datetime.now() - timedelta(hours=24),
            limit=100
        )
        
        failed_events = [e for e in recent_events if e.result in ["failure", "denied"]]
        
        return {
            "security_scan": scan_results,
            "active_sessions": active_sessions,
            "recent_failed_events": len(failed_events),
            "security_status": "healthy" if scan_results["security_score"] > 80 else "needs_attention"
        }