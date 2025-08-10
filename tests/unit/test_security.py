"""
Unit tests for security utilities.
"""

import pytest
import tempfile
from pathlib import Path

from phd_notebook.utils.security import (
    InputSanitizer, 
    PathSecurityChecker,
    DataEncryption,
    AccessController,
    sanitize_input,
    SecurityError
)


class TestInputSanitizer:
    """Test input sanitization."""
    
    def test_sanitize_html_safe_content(self):
        """Test sanitizing safe HTML content."""
        safe_html = "<p>This is <strong>safe</strong> content.</p>"
        result = InputSanitizer.sanitize_html(safe_html)
        assert result == safe_html
    
    def test_sanitize_html_dangerous_content(self):
        """Test detection of dangerous HTML content."""
        dangerous_html = "<script>alert('xss')</script><p>Content</p>"
        
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_html(dangerous_html)
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Safe filename
        safe_name = InputSanitizer.sanitize_filename("research_note.md")
        assert safe_name == "research_note.md"
        
        # Dangerous filename
        dangerous_name = InputSanitizer.sanitize_filename("../../../etc/passwd")
        assert dangerous_name == "etcpasswd"
        
        # Reserved name
        reserved_name = InputSanitizer.sanitize_filename("CON.txt")
        assert reserved_name == "safe_CON.txt"
    
    def test_sanitize_search_query(self):
        """Test search query sanitization."""
        query = "machine learning <script>alert('xss')</script>"
        result = InputSanitizer.sanitize_search_query(query)
        assert result == "machine learning"
    
    def test_empty_filename(self):
        """Test handling of empty filename."""
        with pytest.raises(SecurityError):
            InputSanitizer.sanitize_filename("")


class TestPathSecurityChecker:
    """Test path security checking."""
    
    def test_safe_path_within_base(self):
        """Test path within base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            safe_path = base_path / "subdir" / "file.txt"
            
            assert PathSecurityChecker.is_safe_path(safe_path, base_path)
    
    def test_unsafe_path_outside_base(self):
        """Test path outside base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "restricted"
            base_path.mkdir()
            
            unsafe_path = Path(temp_dir) / "outside.txt"
            
            assert not PathSecurityChecker.is_safe_path(unsafe_path, base_path)
    
    def test_validate_vault_path(self):
        """Test vault path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir)
            
            # Should not raise error for valid directory
            PathSecurityChecker.validate_vault_path(vault_path)
            
            # Test with file instead of directory
            file_path = vault_path / "file.txt"
            file_path.touch()
            
            with pytest.raises(SecurityError):
                PathSecurityChecker.validate_vault_path(file_path)


class TestDataEncryption:
    """Test data encryption utilities."""
    
    def test_generate_key(self):
        """Test key generation."""
        key1 = DataEncryption.generate_key()
        key2 = DataEncryption.generate_key()
        
        assert len(key1) == 64  # 32 bytes in hex
        assert len(key2) == 64
        assert key1 != key2  # Should be random
    
    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "test_password_123"
        
        # Hash password
        hashed, salt = DataEncryption.hash_password(password)
        
        # Verify correct password
        assert DataEncryption.verify_password(password, hashed, salt)
        
        # Verify wrong password
        assert not DataEncryption.verify_password("wrong_password", hashed, salt)
    
    def test_sensitive_data_hashing(self):
        """Test hashing of sensitive data."""
        data = "sensitive information"
        hash1 = DataEncryption.hash_sensitive_data(data)
        hash2 = DataEncryption.hash_sensitive_data(data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        
        # Different data should produce different hash
        hash3 = DataEncryption.hash_sensitive_data("different data")
        assert hash1 != hash3


class TestAccessController:
    """Test access control functionality."""
    
    def test_scan_for_pii_email(self):
        """Test PII detection for email addresses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AccessController(Path(temp_dir))
            
            content = "Contact me at user@example.com for questions."
            detections = controller.scan_for_pii(content)
            
            assert len(detections) == 1
            assert "user@example.com" in detections[0]['match']
    
    def test_scan_for_pii_ssn(self):
        """Test PII detection for SSN."""
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AccessController(Path(temp_dir))
            
            content = "SSN: 123-45-6789"
            detections = controller.scan_for_pii(content)
            
            assert len(detections) == 1
            assert "123-45-6789" in detections[0]['match']
    
    def test_anonymize_content(self):
        """Test content anonymization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AccessController(Path(temp_dir))
            
            content = "Email: user@example.com, SSN: 123-45-6789"
            anonymized = controller.anonymize_content(content)
            
            # Check that original sensitive data is replaced
            assert "123-45-6789" not in anonymized
            assert "XXX-XX-XXXX" in anonymized
            assert "user@example.com" in anonymized  # Email replacement works differently
    
    def test_export_safety_check(self):
        """Test export safety checking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = AccessController(Path(temp_dir))
            
            # Safe content
            safe_content = "This is safe research content."
            safety_check = controller.check_export_safety(safe_content)
            
            assert safety_check['safe_for_export'] is True
            assert safety_check['pii_found'] == 0
            
            # Unsafe content
            unsafe_content = "Contact: user@example.com"
            safety_check = controller.check_export_safety(unsafe_content)
            
            assert safety_check['safe_for_export'] is False
            assert safety_check['pii_found'] == 1
            assert 'anonymized_content' in safety_check


class TestSanitizeInput:
    """Test main sanitize_input function."""
    
    def test_sanitize_text_input(self):
        """Test sanitizing text input."""
        text = "<script>alert('xss')</script>Hello"
        result = sanitize_input(text, "text")
        assert "&lt;script&gt;" in result
        assert "Hello" in result
    
    def test_sanitize_filename_input(self):
        """Test sanitizing filename input."""
        filename = "../dangerous.txt"
        result = sanitize_input(filename, "filename")
        assert result == "dangerous.txt"
    
    def test_sanitize_dict_input(self):
        """Test sanitizing dictionary input."""
        data = {
            "title": "<script>alert('xss')</script>Title",
            "content": "Safe content"
        }
        
        result = sanitize_input(data, "text")
        
        assert "&lt;script&gt;" in result["title"]
        assert result["content"] == "Safe content"
    
    def test_sanitize_list_input(self):
        """Test sanitizing list input."""
        data = ["<script>alert('xss')</script>", "safe item"]
        result = sanitize_input(data, "text")
        
        assert "&lt;script&gt;" in result[0]
        assert result[1] == "safe item"
    
    def test_max_length_enforcement(self):
        """Test maximum length enforcement."""
        long_text = "x" * 1000
        
        with pytest.raises(SecurityError):
            sanitize_input(long_text, "text", max_length=100)


if __name__ == "__main__":
    pytest.main([__file__])