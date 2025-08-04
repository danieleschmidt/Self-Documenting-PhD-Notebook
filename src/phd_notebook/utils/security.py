"""
Security utilities for the PhD notebook system.
"""

import hashlib
import hmac
import re
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import html
import bleach
from urllib.parse import urlparse


class SecurityError(Exception):
    """Custom security error."""
    pass


class InputSanitizer:
    """Sanitize user inputs to prevent security issues."""
    
    # Allowed HTML tags for rich content
    ALLOWED_TAGS = [
        'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'blockquote', 'code', 'pre', 'a', 'img'
    ]
    
    ALLOWED_ATTRIBUTES = {
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title', 'width', 'height'],
        '*': ['class', 'id']
    }
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'document\.',
        r'window\.',
        r'alert\s*\(',
        r'confirm\s*\(',
        r'prompt\s*\(',
    ]
    
    @classmethod
    def sanitize_html(cls, content: str) -> str:
        """Sanitize HTML content."""
        if not content:
            return ""
        
        # First check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                raise SecurityError(f"Dangerous content detected: {pattern}")
        
        # Use bleach to clean HTML
        cleaned = bleach.clean(
            content,
            tags=cls.ALLOWED_TAGS,
            attributes=cls.ALLOWED_ATTRIBUTES,
            strip=True
        )
        
        return cleaned
    
    @classmethod
    def sanitize_markdown(cls, content: str) -> str:
        """Sanitize markdown content."""
        if not content:
            return ""
        
        # Check for dangerous patterns in raw markdown
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                raise SecurityError(f"Dangerous content detected in markdown: {pattern}")
        
        # Check for dangerous link patterns
        link_pattern = r'\[([^\]]*)\]\(([^)]*)\)'
        for match in re.finditer(link_pattern, content):
            link_url = match.group(2)
            if not cls._is_safe_url(link_url):
                raise SecurityError(f"Unsafe URL detected: {link_url}")
        
        return content
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        if not filename:
            raise SecurityError("Filename cannot be empty")
        
        # Remove or replace dangerous characters
        safe_filename = re.sub(r'[<>:"|?*\\]', '', filename)
        safe_filename = re.sub(r'\.\.+', '.', safe_filename)  # Prevent path traversal
        safe_filename = safe_filename.strip('. ')  # Remove leading/trailing dots and spaces
        
        if not safe_filename:
            raise SecurityError("Filename becomes empty after sanitization")
        
        if len(safe_filename) > 255:
            safe_filename = safe_filename[:255]
        
        # Check for reserved names (Windows)
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5',
            'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4',
            'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        if safe_filename.upper() in reserved_names:
            safe_filename = f"safe_{safe_filename}"
        
        return safe_filename
    
    @classmethod
    def sanitize_search_query(cls, query: str) -> str:
        """Sanitize search query."""
        if not query:
            return ""
        
        # Remove potentially dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        # Limit length
        if len(query) > 500:
            query = query[:500]
        
        return query.strip()
    
    @classmethod
    def _is_safe_url(cls, url: str) -> bool:
        """Check if URL is safe."""
        if not url:
            return True
        
        # Allow relative URLs and common safe protocols
        if url.startswith(('#', '/', './')) or not url.startswith(('http://', 'https://')):
            return True
        
        try:
            parsed = urlparse(url)
            
            # Block dangerous protocols
            dangerous_protocols = ['javascript', 'data', 'vbscript', 'file']
            if parsed.scheme.lower() in dangerous_protocols:
                return False
            
            # Allow http/https
            if parsed.scheme.lower() in ['http', 'https']:
                return True
            
        except Exception:
            pass
        
        return False


class PathSecurityChecker:
    """Check paths for security issues."""
    
    @staticmethod
    def is_safe_path(path: Path, base_path: Path) -> bool:
        """Check if path is safe relative to base path."""
        try:
            # Resolve paths to handle symlinks and relative references
            safe_path = path.resolve()
            safe_base = base_path.resolve()
            
            # Check if path is within base directory
            try:
                safe_path.relative_to(safe_base)
                return True
            except ValueError:
                return False
                
        except Exception:
            return False
    
    @staticmethod
    def validate_vault_path(vault_path: Path) -> None:
        """Validate vault path for security."""
        if not vault_path.exists():
            return  # Will be created later
        
        # Check permissions
        if not vault_path.is_dir():
            raise SecurityError("Vault path must be a directory")
        
        # Check for world-writable permissions (Unix)
        try:
            import stat
            vault_stat = vault_path.stat()
            if vault_stat.st_mode & stat.S_IWOTH:
                raise SecurityError("Vault directory is world-writable")
        except (ImportError, AttributeError):
            pass  # Not on Unix system


class DataEncryption:
    """Handle sensitive data encryption."""
    
    @staticmethod
    def generate_key() -> str:
        """Generate a secure random key."""
        return secrets.token_hex(32)
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        import hashlib
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return key.hex(), salt
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash."""
        key, _ = DataEncryption.hash_password(password, salt)
        return hmac.compare_digest(key, hashed)
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data for comparison."""
        return hashlib.sha256(data.encode()).hexdigest()


class AccessController:
    """Control access to research data."""
    
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b',  # Phone
        ]
        
        self.pii_detected: Set[str] = set()
    
    def scan_for_pii(self, content: str, source: str = "") -> List[Dict[str, Any]]:
        """Scan content for personally identifiable information."""
        detections = []
        
        for pattern in self.sensitive_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                detection = {
                    'pattern': pattern,
                    'match': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'source': source,
                    'confidence': 0.8  # Basic confidence score
                }
                detections.append(detection)
                
                # Log detection
                detection_hash = DataEncryption.hash_sensitive_data(match.group())
                self.pii_detected.add(detection_hash)
        
        return detections
    
    def anonymize_content(self, content: str) -> str:
        """Anonymize sensitive content."""
        anonymized = content
        
        for pattern in self.sensitive_patterns:
            if pattern == r'\b\d{3}-\d{2}-\d{4}\b':  # SSN
                anonymized = re.sub(pattern, 'XXX-XX-XXXX', anonymized)
            elif pattern == r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b':  # Credit card
                anonymized = re.sub(pattern, 'XXXX-XXXX-XXXX-XXXX', anonymized)
            elif r'@' in pattern:  # Email
                anonymized = re.sub(pattern, 'user@example.com', anonymized)
            elif r'\d{3}[- ]?\d{3}[- ]?\d{4}' in pattern:  # Phone
                anonymized = re.sub(pattern, 'XXX-XXX-XXXX', anonymized)
        
        return anonymized
    
    def check_export_safety(self, content: str, destination: str = "public") -> Dict[str, Any]:
        """Check if content is safe for export."""
        pii_detections = self.scan_for_pii(content)
        
        safety_check = {
            'safe_for_export': len(pii_detections) == 0,
            'pii_found': len(pii_detections),
            'detections': pii_detections,
            'destination': destination,
            'recommendation': 'approve' if len(pii_detections) == 0 else 'review_required'
        }
        
        if pii_detections:
            safety_check['anonymized_content'] = self.anonymize_content(content)
        
        return safety_check


def sanitize_input(
    data: Any,
    input_type: str = "text",
    max_length: Optional[int] = None
) -> Any:
    """
    Main function to sanitize various types of input.
    
    Args:
        data: Input data to sanitize
        input_type: Type of input (text, html, markdown, filename, etc.)
        max_length: Maximum allowed length
        
    Returns:
        Sanitized data
        
    Raises:
        SecurityError: If input is unsafe
    """
    if data is None:
        return None
    
    if isinstance(data, str):
        # Apply length limit
        if max_length and len(data) > max_length:
            raise SecurityError(f"Input too long (max {max_length} characters)")
        
        # Apply type-specific sanitization
        if input_type == "html":
            return InputSanitizer.sanitize_html(data)
        elif input_type == "markdown":
            return InputSanitizer.sanitize_markdown(data)
        elif input_type == "filename":
            return InputSanitizer.sanitize_filename(data)
        elif input_type == "search":
            return InputSanitizer.sanitize_search_query(data)
        else:  # text
            return html.escape(data)
    
    elif isinstance(data, dict):
        # Recursively sanitize dictionary values
        return {
            key: sanitize_input(value, input_type, max_length)
            for key, value in data.items()
        }
    
    elif isinstance(data, list):
        # Recursively sanitize list items
        return [
            sanitize_input(item, input_type, max_length)
            for item in data
        ]
    
    else:
        # For other types, return as-is
        return data


# Security audit decorator
def security_audit(operation_type: str = "unknown"):
    """Decorator to audit security-sensitive operations."""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            from .logging import get_logger
            
            logger = get_logger()
            
            # Log operation start
            logger.info(
                f"Security audit: {operation_type} started",
                extra={
                    "operation": operation_type,
                    "function": func.__name__,
                    "category": "security_audit"
                }
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                logger.info(
                    f"Security audit: {operation_type} completed successfully",
                    extra={
                        "operation": operation_type,
                        "function": func.__name__,
                        "category": "security_audit"
                    }
                )
                
                return result
                
            except SecurityError as e:
                # Log security violation
                logger.error(
                    f"Security violation in {operation_type}: {str(e)}",
                    extra={
                        "operation": operation_type,
                        "function": func.__name__,
                        "error": str(e),
                        "category": "security_violation"
                    }
                )
                raise
                
            except Exception as e:
                # Log other errors
                logger.error(
                    f"Error in {operation_type}: {str(e)}",
                    extra={
                        "operation": operation_type,
                        "function": func.__name__,
                        "error": str(e),
                        "category": "security_audit"
                    }
                )
                raise
        
        return wrapper
    return decorator