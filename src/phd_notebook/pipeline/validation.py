"""Validation and security features for pipeline operations."""

import re
import hashlib
import hmac
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.security import InputSanitizer
from ..utils.validation import ValidationError


@dataclass
class ValidationRule:
    """Defines a validation rule."""
    name: str
    pattern: str
    error_message: str
    severity: str = "error"  # error, warning, info
    applies_to: Set[str] = None  # file extensions, command types, etc.


class InputValidator:
    """Validates inputs for security and correctness."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.sanitizer = InputSanitizer()
        self.validation_rules = self._build_validation_rules()
        self.command_whitelist = self._build_command_whitelist()
    
    def validate_command(self, command: str, context: str = "general") -> Dict[str, Any]:
        """Validate a shell command for security."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "sanitized_command": command
        }
        
        # Check against command whitelist
        if not self._is_command_whitelisted(command):
            validation_result["errors"].append(f"Command not in whitelist: {command.split()[0] if command.split() else ''}")
            validation_result["is_valid"] = False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'rm\s+-rf\s+/',  # Dangerous rm commands
            r'rm\s+-rf\s+\*',
            r':\(\)\{\s*:\|\:&\s*\};:',  # Fork bomb
            r'curl.*\|\s*sh',  # Dangerous curl pipe to shell
            r'wget.*\|\s*sh',
            r'>\s*/dev/sd[a-z]',  # Writing to disk devices
            r'dd\s+.*of=/dev/',
            r'mkfs\.',  # Filesystem creation
            r'fdisk',
            r'crontab',  # Cron modification
            r'chmod\s+777',  # Dangerous permissions
            r'chown\s+.*:.*/',
            r'sudo\s+su',  # Privilege escalation
            r'nc\s+.*-e',  # Netcat shell
            r'/etc/passwd',  # System file access
            r'/etc/shadow',
            r';\s*reboot',  # System control
            r';\s*shutdown',
            r'>\s*/etc/',  # Writing to system directories
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                validation_result["errors"].append(f"Dangerous pattern detected: {pattern}")
                validation_result["is_valid"] = False
        
        # Check for injection attempts
        injection_patterns = [
            r';.*rm',  # Command injection
            r'&&.*rm',
            r'\|\|.*rm',
            r'`.*`',  # Command substitution
            r'\$\(',  # Command substitution
            r'>\s*&',  # Output redirection
            r'<\s*&',
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, command):
                validation_result["warnings"].append(f"Potential injection pattern: {pattern}")
        
        # Sanitize command if possible
        if validation_result["is_valid"]:
            validation_result["sanitized_command"] = self._sanitize_command(command)
        
        return validation_result
    
    def validate_file_path(self, file_path: str, operation: str = "read") -> Dict[str, Any]:
        """Validate file path for security."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "normalized_path": file_path
        }
        
        try:
            # Normalize path
            path = Path(file_path).resolve()
            validation_result["normalized_path"] = str(path)
            
            # Check for path traversal
            if ".." in file_path or "/.." in file_path:
                validation_result["warnings"].append("Path traversal detected")
            
            # Check for system directories
            system_dirs = ["/etc", "/sys", "/proc", "/dev", "/boot", "/root"]
            if any(str(path).startswith(sys_dir) for sys_dir in system_dirs):
                if operation in ["write", "delete"]:
                    validation_result["errors"].append(f"Cannot {operation} in system directory")
                    validation_result["is_valid"] = False
                else:
                    validation_result["warnings"].append("Accessing system directory")
            
            # Check file permissions for write operations
            if operation in ["write", "delete"] and path.exists():
                if not path.is_file():
                    validation_result["errors"].append("Path is not a regular file")
                    validation_result["is_valid"] = False
                
                # Check if file is executable
                if path.stat().st_mode & 0o111:
                    validation_result["warnings"].append("Modifying executable file")
            
        except Exception as e:
            validation_result["errors"].append(f"Path validation error: {e}")
            validation_result["is_valid"] = False
        
        return validation_result
    
    def validate_webhook_url(self, url: str) -> Dict[str, Any]:
        """Validate webhook URL for security."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Basic URL validation
        url_pattern = r'^https?://[a-zA-Z0-9.-]+(?:\:[0-9]+)?(?:/[^\s]*)?$'
        if not re.match(url_pattern, url):
            validation_result["errors"].append("Invalid URL format")
            validation_result["is_valid"] = False
            return validation_result
        
        # Check for HTTPS
        if not url.startswith('https://'):
            validation_result["warnings"].append("Webhook URL is not HTTPS")
        
        # Check for internal/local addresses
        internal_patterns = [
            r'://localhost',
            r'://127\.',
            r'://10\.',
            r'://192\.168\.',
            r'://172\.(1[6-9]|2[0-9]|3[01])\.',
            r'://169\.254\.',  # Link-local
        ]
        
        for pattern in internal_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                validation_result["warnings"].append("Webhook points to internal address")
                break
        
        return validation_result
    
    def validate_config_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration data."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "sanitized_config": config.copy()
        }
        
        # Check for sensitive data in config
        sensitive_keys = ["password", "secret", "token", "key", "credential"]
        for key, value in config.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 0:
                    # Mask sensitive values in logs
                    validation_result["sanitized_config"][key] = "***MASKED***"
                    validation_result["warnings"].append(f"Sensitive data detected in key: {key}")
        
        # Validate numeric ranges
        numeric_validations = {
            "check_interval": (1, 3600),  # 1 second to 1 hour
            "heal_timeout": (1, 7200),    # 1 second to 2 hours
            "max_heal_attempts": (1, 100),
        }
        
        for key, (min_val, max_val) in numeric_validations.items():
            if key in config:
                try:
                    value = int(config[key])
                    if not min_val <= value <= max_val:
                        validation_result["errors"].append(
                            f"{key} must be between {min_val} and {max_val}, got {value}"
                        )
                        validation_result["is_valid"] = False
                except (ValueError, TypeError):
                    validation_result["errors"].append(f"{key} must be a valid integer")
                    validation_result["is_valid"] = False
        
        return validation_result
    
    def _build_validation_rules(self) -> List[ValidationRule]:
        """Build list of validation rules."""
        return [
            ValidationRule(
                name="no_dangerous_rm",
                pattern=r'rm\s+-rf\s+/',
                error_message="Dangerous rm command detected",
                severity="error",
                applies_to={"shell_command"}
            ),
            ValidationRule(
                name="no_system_dirs",
                pattern=r'(/etc|/sys|/proc|/dev|/boot)',
                error_message="Access to system directories",
                severity="warning",
                applies_to={"file_path"}
            ),
            ValidationRule(
                name="https_required",
                pattern=r'^http://',
                error_message="HTTP URLs should use HTTPS",
                severity="warning",
                applies_to={"webhook_url"}
            ),
        ]
    
    def _build_command_whitelist(self) -> Set[str]:
        """Build whitelist of allowed commands."""
        return {
            # Version control
            "git", "gh", "glab",
            # Package managers
            "npm", "pip", "cargo", "yarn", "composer",
            # Build tools
            "make", "cmake", "ninja",
            # Testing
            "pytest", "jest", "mocha", "cargo test",
            # Linting/formatting
            "eslint", "black", "isort", "flake8", "mypy",
            # File operations (safe subset)
            "ls", "cat", "grep", "find", "head", "tail",
            # System info
            "ps", "top", "df", "free", "uname",
            # Docker (safe subset)
            "docker build", "docker run", "docker ps", "docker logs",
            # Other safe commands
            "echo", "date", "sleep", "which", "whoami"
        }
    
    def _is_command_whitelisted(self, command: str) -> bool:
        """Check if command is in whitelist."""
        if not command.strip():
            return False
        
        # Extract the base command
        parts = command.strip().split()
        if not parts:
            return False
        
        base_command = parts[0]
        
        # Check exact matches
        if base_command in self.command_whitelist:
            return True
        
        # Check compound commands
        for allowed in self.command_whitelist:
            if " " in allowed and command.startswith(allowed):
                return True
        
        return False
    
    def _sanitize_command(self, command: str) -> str:
        """Sanitize command by removing/escaping dangerous elements."""
        # Remove null bytes
        sanitized = command.replace('\x00', '')
        
        # Escape shell metacharacters in arguments (basic sanitization)
        # This is a simple approach - more sophisticated sanitization might be needed
        dangerous_chars = ['|', '&', ';', '(', ')', '<', '>', '`', '$']
        for char in dangerous_chars:
            if char in sanitized and not self._is_char_context_safe(sanitized, char):
                sanitized = sanitized.replace(char, f'\\{char}')
        
        return sanitized
    
    def _is_char_context_safe(self, command: str, char: str) -> bool:
        """Check if a character is safe in its context."""
        # This is a simplified check - real implementation would be more sophisticated
        # For now, assume characters are safe if they're in quoted strings
        in_quotes = False
        escape_next = False
        
        for c in command:
            if escape_next:
                escape_next = False
                continue
            
            if c == '\\':
                escape_next = True
                continue
            
            if c == '"' or c == "'":
                in_quotes = not in_quotes
            
            if c == char and in_quotes:
                return True
        
        return False


class SecurityAuditor:
    """Performs security audits on pipeline operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.sanitizer = InputSanitizer()
        self.audit_log = []
    
    def audit_pipeline_execution(self, 
                                pipeline_id: str,
                                commands: List[str],
                                files_accessed: List[str],
                                network_calls: List[str]) -> Dict[str, Any]:
        """Audit a pipeline execution for security issues."""
        audit_result = {
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "security_score": 100,  # Start with perfect score
            "issues": [],
            "recommendations": []
        }
        
        validator = InputValidator()
        
        # Audit commands
        for command in commands:
            cmd_validation = validator.validate_command(command)
            if not cmd_validation["is_valid"]:
                audit_result["issues"].extend([
                    f"Command security issue: {error}" for error in cmd_validation["errors"]
                ])
                audit_result["security_score"] -= 20
            
            audit_result["issues"].extend([
                f"Command warning: {warning}" for warning in cmd_validation["warnings"]
            ])
            if cmd_validation["warnings"]:
                audit_result["security_score"] -= 5
        
        # Audit file access
        for file_path in files_accessed:
            file_validation = validator.validate_file_path(file_path)
            if not file_validation["is_valid"]:
                audit_result["issues"].extend([
                    f"File access issue: {error}" for error in file_validation["errors"]
                ])
                audit_result["security_score"] -= 15
            
            audit_result["issues"].extend([
                f"File access warning: {warning}" for warning in file_validation["warnings"]
            ])
            if file_validation["warnings"]:
                audit_result["security_score"] -= 3
        
        # Audit network calls
        for url in network_calls:
            url_validation = validator.validate_webhook_url(url)
            if not url_validation["is_valid"]:
                audit_result["issues"].extend([
                    f"Network call issue: {error}" for error in url_validation["errors"]
                ])
                audit_result["security_score"] -= 10
            
            audit_result["issues"].extend([
                f"Network call warning: {warning}" for warning in url_validation["warnings"]
            ])
            if url_validation["warnings"]:
                audit_result["security_score"] -= 2
        
        # Generate recommendations
        if audit_result["security_score"] < 80:
            audit_result["recommendations"].append("Review and restrict command permissions")
        if audit_result["security_score"] < 60:
            audit_result["recommendations"].append("Implement additional security controls")
        if audit_result["security_score"] < 40:
            audit_result["recommendations"].append("Consider blocking this pipeline until security issues are resolved")
        
        # Ensure score doesn't go below 0
        audit_result["security_score"] = max(0, audit_result["security_score"])
        
        # Log audit
        self.audit_log.append(audit_result)
        
        # Keep only recent audits (last 1000)
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
        
        return audit_result
    
    def get_security_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get security summary for the past N days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_audits = [
            audit for audit in self.audit_log
            if datetime.fromisoformat(audit["timestamp"]) > cutoff_date
        ]
        
        if not recent_audits:
            return {"message": "No recent audit data"}
        
        avg_score = sum(audit["security_score"] for audit in recent_audits) / len(recent_audits)
        
        total_issues = sum(len(audit["issues"]) for audit in recent_audits)
        
        # Find most common issues
        all_issues = [issue for audit in recent_audits for issue in audit["issues"]]
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "period_days": days,
            "total_audits": len(recent_audits),
            "average_security_score": avg_score,
            "total_issues": total_issues,
            "most_common_issues": common_issues,
            "trend": self._calculate_trend(recent_audits)
        }
    
    def _calculate_trend(self, audits: List[Dict[str, Any]]) -> str:
        """Calculate security trend."""
        if len(audits) < 2:
            return "insufficient_data"
        
        # Compare first and second half
        mid_point = len(audits) // 2
        first_half_avg = sum(audit["security_score"] for audit in audits[:mid_point]) / mid_point
        second_half_avg = sum(audit["security_score"] for audit in audits[mid_point:]) / (len(audits) - mid_point)
        
        difference = second_half_avg - first_half_avg
        
        if difference > 5:
            return "improving"
        elif difference < -5:
            return "degrading"
        else:
            return "stable"