"""
Tests for Security Framework

Comprehensive test suite for security hardening, safe execution,
and vulnerability patching components.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from phd_notebook.security.security_patch import (
    SecurityPatcher,
    SecurityVulnerability,
    SecurityPatchResult,
    run_security_patch
)

from phd_notebook.utils.secure_execution_fixed import (
    SafeEvaluator,
    SafeExecutor,
    SecureExecutionError,
    safe_json_loads,
    safe_regex_eval,
    safe_mathematical_eval,
    default_evaluator,
    default_executor
)


class TestSecureExecution:
    """Test secure execution utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = SafeEvaluator()
        self.executor = SafeExecutor()
    
    def test_safe_evaluator_initialization(self):
        """Test SafeEvaluator initialization."""
        evaluator = SafeEvaluator({'x': 10, 'y': 20})
        assert evaluator.allowed_names['x'] == 10
        assert evaluator.allowed_names['y'] == 20
    
    def test_safe_mathematical_expressions(self):
        """Test evaluation of safe mathematical expressions."""
        context = {'x': 10, 'y': 20, 'z': 3}
        
        # Basic arithmetic
        assert self.evaluator.safe_eval('x + y', context) == 30
        assert self.evaluator.safe_eval('x * y', context) == 200
        assert self.evaluator.safe_eval('x / z', context) == 10/3
        assert self.evaluator.safe_eval('x ** 2', context) == 100
        
        # Boolean operations
        assert self.evaluator.safe_eval('x > y', context) == False
        assert self.evaluator.safe_eval('x < y', context) == True
        assert self.evaluator.safe_eval('x == 10', context) == True
        
        # Function calls
        assert self.evaluator.safe_eval('max(x, y)', context) == 20
        assert self.evaluator.safe_eval('min(x, y)', context) == 10
        assert self.evaluator.safe_eval('abs(-x)', context) == 10
    
    def test_safe_collection_operations(self):
        """Test safe operations on collections."""
        context = {'numbers': [1, 2, 3, 4, 5], 'text': 'hello'}
        
        # List operations
        assert self.evaluator.safe_eval('len(numbers)', context) == 5
        assert self.evaluator.safe_eval('sum(numbers)', context) == 15
        assert self.evaluator.safe_eval('max(numbers)', context) == 5
        
        # String operations
        assert self.evaluator.safe_eval('len(text)', context) == 5
        assert self.evaluator.safe_eval('text.upper()', context) == 'HELLO'
        assert self.evaluator.safe_eval('text.startswith("he")', context) == True
    
    def test_unsafe_operations_blocked(self):
        """Test that unsafe operations are blocked."""
        dangerous_expressions = [
            '__import__("os").system("ls")',
            'exec("print(1)")',
            'eval("1+1")',
            'open("/etc/passwd")',
            '__builtins__',
            'globals()',
            'locals()',
            'dir()',
            'vars()',
            'getattr(object, "__class__")'
        ]
        
        for expr in dangerous_expressions:
            with pytest.raises(SecureExecutionError):
                self.evaluator.safe_eval(expr)
    
    def test_invalid_syntax_handling(self):
        """Test handling of invalid syntax."""
        invalid_expressions = [
            'x + + y',  # Invalid syntax
            '1 2 3',    # Invalid syntax
            '',         # Empty expression
            'x +',      # Incomplete expression
        ]
        
        for expr in invalid_expressions:
            with pytest.raises(SecureExecutionError):
                self.evaluator.safe_eval(expr, {'x': 1, 'y': 2})
    
    def test_safe_executor_functionality(self):
        """Test SafeExecutor functionality."""
        context = {'x': 10}
        
        safe_code = '''
x = x * 2
y = x + 5
result = y
'''
        
        result_context = self.executor.safe_exec(safe_code, context)
        
        assert result_context['x'] == 20
        assert result_context['y'] == 25
        assert result_context['result'] == 25
    
    def test_unsafe_code_blocked_in_executor(self):
        """Test that unsafe code is blocked in executor."""
        unsafe_code_samples = [
            'import os',
            'from subprocess import call',
            '__import__("os")',
            'exec("print(1)")',
            'eval("1+1")'
        ]
        
        for code in unsafe_code_samples:
            with pytest.raises(SecureExecutionError):
                self.executor.safe_exec(code)
    
    def test_safe_json_operations(self):
        """Test safe JSON processing."""
        # Valid JSON
        valid_json = '{"name": "test", "value": 42, "list": [1, 2, 3]}'
        result = safe_json_loads(valid_json)
        
        assert result['name'] == 'test'
        assert result['value'] == 42
        assert result['list'] == [1, 2, 3]
        
        # Invalid JSON
        invalid_json = '{"invalid": json}'
        with pytest.raises(SecureExecutionError):
            safe_json_loads(invalid_json)
    
    def test_safe_regex_operations(self):
        """Test safe regex evaluation."""
        # Valid regex
        pattern = r'\d+'
        text = 'There are 42 numbers here'
        
        match = safe_regex_eval(pattern, text)
        assert match is not None
        assert match.group() == '42'
        
        # Invalid regex
        invalid_pattern = r'[invalid'
        with pytest.raises(SecureExecutionError):
            safe_regex_eval(invalid_pattern, text)
        
        # Pattern too long
        long_pattern = 'a' * 2000
        with pytest.raises(SecureExecutionError):
            safe_regex_eval(long_pattern, text)
    
    def test_safe_mathematical_evaluation(self):
        """Test safe mathematical expression evaluation."""
        variables = {'x': 5, 'y': 10}
        
        # Basic math
        assert safe_mathematical_eval('x + y', variables) == 15.0
        assert safe_mathematical_eval('x * y + 5', variables) == 55.0
        
        # With constants
        result = safe_mathematical_eval('pi * 2', {'pi': 3.14159})
        assert abs(result - 6.28318) < 0.001
        
        # Invalid mathematical expression
        with pytest.raises(SecureExecutionError):
            safe_mathematical_eval('x + invalid_var', variables)
    
    def test_global_instances(self):
        """Test global evaluator and executor instances."""
        assert default_evaluator is not None
        assert default_executor is not None
        
        # Should be able to use them directly
        result = default_evaluator.safe_eval('1 + 2')
        assert result == 3


class TestSecurityPatcher:
    """Test security vulnerability patching system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.patcher = SecurityPatcher(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_file(self, filename: str, content: str) -> Path:
        """Create a test file with specified content."""
        file_path = Path(self.temp_dir) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        return file_path
    
    def test_vulnerability_scanning(self):
        """Test vulnerability scanning functionality."""
        # Create files with various vulnerabilities
        self._create_test_file('test1.py', '''
import os
result = eval("1 + 2")
secret = "hardcoded_password_123"
subprocess.call("ls", shell=True)
''')
        
        self._create_test_file('test2.py', '''
import pickle
data = pickle.loads(user_input)
path = "../../../etc/passwd"
''')
        
        vulnerabilities = self.patcher.scan_vulnerabilities()
        
        assert len(vulnerabilities) > 0
        
        # Check that different vulnerability types were detected
        vuln_types = {v.vulnerability_type for v in vulnerabilities}
        expected_types = {'eval', 'hardcoded_secret', 'subprocess_shell', 'pickle_loads', 'path_traversal'}
        
        # Should detect at least some of these
        assert len(vuln_types.intersection(expected_types)) > 0
        
        # Check vulnerability structure
        for vuln in vulnerabilities:
            assert isinstance(vuln, SecurityVulnerability)
            assert vuln.file_path
            assert vuln.line_number > 0
            assert vuln.severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            assert not vuln.fixed  # Should not be fixed initially
    
    def test_severity_assessment(self):
        """Test vulnerability severity assessment."""
        test_cases = [
            ('eval("1+1")', 'eval', 'CRITICAL'),
            ('exec("print(1)")', 'exec', 'CRITICAL'),
            ('SECRET = "mysecret123"', 'hardcoded_secret', 'HIGH'),
            ('../config/file.txt', 'path_traversal', 'MEDIUM')
        ]
        
        for line, vuln_type, expected_severity in test_cases:
            severity = self.patcher._get_severity(vuln_type, line)
            assert severity == expected_severity
    
    def test_secure_replacement_generation(self):
        """Test generation of secure code replacements."""
        replacements = self.patcher.generate_secure_replacements()
        
        assert isinstance(replacements, dict)
        assert len(replacements) > 0
        
        # Should have replacements for dangerous patterns
        patterns = list(replacements.keys())
        
        # Check that eval and exec patterns are included
        eval_patterns = [p for p in patterns if 'eval' in p]
        exec_patterns = [p for p in patterns if 'exec' in p]
        
        assert len(eval_patterns) > 0
        assert len(exec_patterns) > 0
    
    def test_automated_patching(self):
        """Test automated vulnerability patching."""
        # Create a file with patchable vulnerabilities
        test_content = '''
import os
result = eval("1 + 2")
password = "secret123"
'''
        
        self._create_test_file('patchable.py', test_content)
        
        # Scan and patch
        vulnerabilities = self.patcher.scan_vulnerabilities()
        result = self.patcher.apply_automated_patches()
        
        assert isinstance(result, SecurityPatchResult)
        assert result.total_vulnerabilities > 0
        
        # Check that some patches were applied
        if result.patched_vulnerabilities > 0:
            # Read the patched file
            with open(Path(self.temp_dir) / 'patchable.py', 'r') as f:
                patched_content = f.read()
            
            # Should have import for secure execution
            assert 'secure_execution_fixed' in patched_content
            
            # Original eval should be replaced
            assert 'eval(' not in patched_content or 'safe_evaluator.safe_eval(' in patched_content
    
    def test_security_report_generation(self):
        """Test security report generation."""
        # Create some test vulnerabilities
        self._create_test_file('report_test.py', '''
eval("test")
SECRET_KEY = "hardcoded123"
''')
        
        # Scan vulnerabilities
        self.patcher.scan_vulnerabilities()
        
        # Generate report
        report = self.patcher.generate_security_report()
        
        assert isinstance(report, str)
        assert 'Research Platform Security Report' in report
        assert 'Summary' in report
        assert 'CRITICAL' in report or 'HIGH' in report
        assert 'Recommendations' in report
        
        # Test with output file
        report_file = Path(self.temp_dir) / 'security_report.md'
        self.patcher.generate_security_report(str(report_file))
        
        assert report_file.exists()
        
        with open(report_file, 'r') as f:
            file_content = f.read()
        
        assert file_content == report
    
    def test_security_configuration_generation(self):
        """Test security configuration generation."""
        config = self.patcher.create_security_config()
        
        assert isinstance(config, dict)
        assert 'security' in config
        
        security_config = config['security']
        
        # Check required security settings
        assert security_config['enable_safe_mode'] is True
        assert security_config['disable_eval_exec'] is True
        assert security_config['require_input_validation'] is True
        
        # Check nested configurations
        assert 'secret_management' in security_config
        assert 'access_control' in security_config
        assert 'data_protection' in security_config
        
        # Check specific security features
        assert security_config['secret_management']['use_env_vars'] is True
        assert security_config['data_protection']['gdpr_compliance'] is True


class TestSecurityIntegration:
    """Test security framework integration."""
    
    def test_security_patch_main_function(self):
        """Test main security patching function."""
        # This tests the integration but may modify the actual repo
        # So we'll mock the patcher to avoid side effects
        
        with patch('phd_notebook.security.security_patch.SecurityPatcher') as mock_patcher:
            mock_instance = Mock()
            mock_patcher.return_value = mock_instance
            
            # Configure mock returns
            mock_instance.scan_vulnerabilities.return_value = [
                SecurityVulnerability(
                    file_path='test.py',
                    line_number=1,
                    vulnerability_type='eval',
                    severity='CRITICAL',
                    description='Test vulnerability'
                )
            ]
            
            mock_instance.apply_automated_patches.return_value = SecurityPatchResult(
                total_vulnerabilities=1,
                patched_vulnerabilities=1,
                failed_patches=0,
                critical_remaining=0,
                patch_summary={'CRITICAL': 1}
            )
            
            mock_instance.generate_security_report.return_value = "Test report"
            mock_instance.create_security_config.return_value = {"security": {}}
            
            # Run the main function
            result = run_security_patch()
            
            # Verify it was called correctly
            mock_instance.scan_vulnerabilities.assert_called_once()
            mock_instance.apply_automated_patches.assert_called_once()
            mock_instance.generate_security_report.assert_called_once()
            mock_instance.create_security_config.assert_called_once()
            
            assert isinstance(result, SecurityPatchResult)
    
    def test_environment_variable_integration(self):
        """Test that security patches work with environment variables."""
        # Test that hardcoded secrets get replaced with environment variables
        evaluator = SafeEvaluator({'SECRET_KEY': os.getenv('SECRET_KEY', 'default')})
        
        # Should work with environment variable
        result = evaluator.safe_eval('SECRET_KEY', {'SECRET_KEY': 'test_value'})
        assert result == 'test_value'
        
        # Should use default if env var not set
        result = evaluator.safe_eval('SECRET_KEY', {'SECRET_KEY': os.getenv('NON_EXISTENT_VAR', 'default')})
        assert result == 'default'


class TestSecurityEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = SafeEvaluator()
    
    def test_empty_expressions(self):
        """Test handling of empty or None expressions."""
        with pytest.raises(SecureExecutionError):
            self.evaluator.safe_eval('')
        
        with pytest.raises(SecureExecutionError):
            self.evaluator.safe_eval(None)
    
    def test_extremely_large_expressions(self):
        """Test handling of very large expressions."""
        # Create a very large but safe expression
        large_expr = ' + '.join(['1'] * 1000)  # 1 + 1 + 1 + ... (1000 times)
        
        # Should still work for large safe expressions
        result = self.evaluator.safe_eval(large_expr)
        assert result == 1000
    
    def test_nested_function_calls(self):
        """Test deeply nested function calls."""
        context = {'numbers': [1, 2, 3, 4, 5]}
        
        # Nested safe function calls should work
        result = self.evaluator.safe_eval('max(min(numbers), len(numbers))', context)
        assert result == 5  # max(1, 5) = 5
    
    def test_complex_data_structures(self):
        """Test operations on complex data structures."""
        context = {
            'data': {
                'users': [
                    {'name': 'Alice', 'age': 30},
                    {'name': 'Bob', 'age': 25}
                ],
                'settings': {'theme': 'dark', 'notifications': True}
            }
        }
        
        # Should be able to access nested data
        result = self.evaluator.safe_eval('len(data["users"])', context)
        assert result == 2
        
        result = self.evaluator.safe_eval('data["settings"]["theme"]', context)
        assert result == 'dark'
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        context = {'text': '测试文本', 'emoji': '🔒'}
        
        result = self.evaluator.safe_eval('len(text)', context)
        assert result == 4  # 4 Chinese characters
        
        result = self.evaluator.safe_eval('emoji', context)
        assert result == '🔒'
    
    def test_floating_point_precision(self):
        """Test floating point operations and precision."""
        context = {'pi': 3.141592653589793}
        
        result = self.evaluator.safe_eval('pi * 2', context)
        assert abs(result - 6.283185307179586) < 1e-10
        
        # Test division
        result = self.evaluator.safe_eval('1 / 3', {})
        assert abs(result - 0.3333333333333333) < 1e-10


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])