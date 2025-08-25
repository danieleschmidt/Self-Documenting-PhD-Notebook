"""
Security Patch System for Research Platform

This module provides comprehensive security patches for the identified
vulnerabilities in the research platform codebase.
"""

import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability in the codebase."""
    file_path: str
    line_number: int
    vulnerability_type: str
    severity: str
    description: str
    fixed: bool = False


@dataclass
class SecurityPatchResult:
    """Result of applying security patches."""
    total_vulnerabilities: int
    patched_vulnerabilities: int
    failed_patches: int
    critical_remaining: int
    patch_summary: Dict[str, int]


class SecurityPatcher:
    """Automated security vulnerability patcher."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.vulnerabilities: List[SecurityVulnerability] = []
        
    def scan_vulnerabilities(self) -> List[SecurityVulnerability]:
        """Scan codebase for security vulnerabilities."""
        vulnerabilities = []
        
        # Patterns for dangerous functions
        dangerous_patterns = {
            'eval': r'\beval\s*\(',
            'exec': r'\bexec\s*\(',  
            'compile': r'\bcompile\s*\(',
            'hardcoded_secret': r'(?i)(secret|password|key|token)\s*=\s*["\'][^"\']{8,}["\']',
            'path_traversal': r'\.\./',
            'subprocess_shell': r'subprocess\.(call|run|Popen).*shell\s*=\s*True',
            'pickle_loads': r'\bpickle\.loads\s*\(',
            'yaml_unsafe': r'\byaml\.(load|unsafe_load)\s*\(',
        }
        
        # Scan Python files
        for py_file in self.project_root.rglob('*.py'):
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for line_num, line in enumerate(lines, 1):
                    for vuln_type, pattern in dangerous_patterns.items():
                        if re.search(pattern, line):
                            severity = self._get_severity(vuln_type, line)
                            vulnerability = SecurityVulnerability(
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=line_num,
                                vulnerability_type=vuln_type,
                                severity=severity,
                                description=f"Found {vuln_type} on line {line_num}: {line.strip()}"
                            )
                            vulnerabilities.append(vulnerability)
                            
            except Exception as e:
                logger.warning(f"Error scanning {py_file}: {e}")
                
        self.vulnerabilities = vulnerabilities
        return vulnerabilities
    
    def _get_severity(self, vuln_type: str, line: str) -> str:
        """Determine severity level of vulnerability."""
        critical_types = {'eval', 'exec', 'subprocess_shell', 'pickle_loads'}
        high_types = {'hardcoded_secret', 'compile', 'yaml_unsafe'}
        
        if vuln_type in critical_types:
            return 'CRITICAL'
        elif vuln_type in high_types:
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def generate_secure_replacements(self) -> Dict[str, str]:
        """Generate secure code replacements for common patterns."""
        return {
            # Eval replacements
            r'\beval\s*\(([^)]+)\)': r'safe_evaluator.safe_eval(\1)',
            
            # Exec replacements
            r'\bexec\s*\(([^)]+)\)': r'safe_executor.safe_exec(\1)',
            
            # Subprocess with shell=True
            r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True': 
            r'subprocess.\1(  # SECURITY: shell=False for safety',
            
            # Pickle loads
            r'\bpickle\.loads\s*\(': r'safe_pickle_loads(',
            
            # YAML unsafe load
            r'\byaml\.(load|unsafe_load)\s*\(': r'yaml.safe_load(',
        }
    
    def apply_automated_patches(self) -> SecurityPatchResult:
        """Apply automated security patches where possible."""
        patched = 0
        failed = 0
        
        replacements = self.generate_secure_replacements()
        
        for vulnerability in self.vulnerabilities:
            file_path = self.project_root / vulnerability.file_path
            
            # Skip test files for some patches to avoid breaking tests
            if 'test_' in vulnerability.file_path and vulnerability.vulnerability_type == 'path_traversal':
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply replacements based on vulnerability type
                if vulnerability.vulnerability_type in ['eval', 'exec']:
                    # Add import for secure execution
                    if 'from phd_notebook.utils.secure_execution_fixed import' not in content:
                        content = 'from phd_notebook.utils.secure_execution_fixed import default_evaluator as safe_evaluator, default_executor as safe_executor\n' + content
                    
                    # Apply specific replacements
                    for pattern, replacement in replacements.items():
                        content = re.sub(pattern, replacement, content)
                
                elif vulnerability.vulnerability_type == 'hardcoded_secret':
                    # Replace hardcoded secrets with environment variables
                    content = re.sub(
                        r'(?i)(secret|password|key|token)\s*=\s*["\'][^"\']+["\']',
                        r'\1 = os.getenv("\1".upper(), "default_value")',
                        content
                    )
                    
                    # Add os import if not present
                    if 'import os' not in content:
                        content = 'import os\n' + content
                
                # Write back if changed
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    vulnerability.fixed = True
                    patched += 1
                    logger.info(f"Patched vulnerability in {vulnerability.file_path}:{vulnerability.line_number}")
                    
            except Exception as e:
                logger.error(f"Failed to patch {vulnerability.file_path}: {e}")
                failed += 1
        
        # Generate summary
        severity_counts = {}
        critical_remaining = 0
        
        for vuln in self.vulnerabilities:
            severity_counts[vuln.severity] = severity_counts.get(vuln.severity, 0) + 1
            if vuln.severity == 'CRITICAL' and not vuln.fixed:
                critical_remaining += 1
        
        return SecurityPatchResult(
            total_vulnerabilities=len(self.vulnerabilities),
            patched_vulnerabilities=patched,
            failed_patches=failed,
            critical_remaining=critical_remaining,
            patch_summary=severity_counts
        )
    
    def generate_security_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive security report."""
        report_lines = [
            "# Research Platform Security Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            f"Total vulnerabilities found: {len(self.vulnerabilities)}",
            ""
        ]
        
        # Group by severity
        by_severity = {}
        for vuln in self.vulnerabilities:
            if vuln.severity not in by_severity:
                by_severity[vuln.severity] = []
            by_severity[vuln.severity].append(vuln)
        
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            if severity in by_severity:
                vulns = by_severity[severity]
                report_lines.append(f"### {severity} ({len(vulns)} issues)")
                
                for vuln in vulns[:10]:  # Limit to first 10
                    status = "✅ FIXED" if vuln.fixed else "❌ OPEN"
                    report_lines.append(f"- {vuln.file_path}:{vuln.line_number} - {vuln.vulnerability_type} {status}")
                
                if len(vulns) > 10:
                    report_lines.append(f"... and {len(vulns) - 10} more")
                report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "1. Replace all eval/exec calls with safe alternatives",
            "2. Use environment variables for secrets",
            "3. Implement input validation for user data",
            "4. Enable security linting in CI/CD pipeline",
            "5. Regular security audits and penetration testing",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def create_security_config(self) -> Dict[str, any]:
        """Create security configuration for the platform."""
        return {
            "security": {
                "enable_safe_mode": True,
                "disable_eval_exec": True,
                "require_input_validation": True,
                "enable_audit_logging": True,
                "secret_management": {
                    "use_env_vars": True,
                    "encrypt_at_rest": True,
                    "rotate_keys": True
                },
                "access_control": {
                    "enable_rbac": True,
                    "require_2fa": True,
                    "session_timeout": 3600
                },
                "data_protection": {
                    "encrypt_pii": True,
                    "anonymize_exports": True,
                    "gdpr_compliance": True
                }
            }
        }


def run_security_patch():
    """Main function to run security patching."""
    try:
        # Initialize patcher
        patcher = SecurityPatcher('/root/repo')
        
        # Scan vulnerabilities
        logger.info("Scanning for security vulnerabilities...")
        vulnerabilities = patcher.scan_vulnerabilities()
        
        logger.info(f"Found {len(vulnerabilities)} security vulnerabilities")
        
        # Apply patches
        logger.info("Applying automated security patches...")
        result = patcher.apply_automated_patches()
        
        # Generate report
        report = patcher.generate_security_report('/root/repo/SECURITY_REPORT.md')
        
        # Create security config
        import json
        config = patcher.create_security_config()
        with open('/root/repo/security_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Security patching complete:")
        logger.info(f"  - Total vulnerabilities: {result.total_vulnerabilities}")
        logger.info(f"  - Patched: {result.patched_vulnerabilities}")
        logger.info(f"  - Failed: {result.failed_patches}")
        logger.info(f"  - Critical remaining: {result.critical_remaining}")
        
        return result
        
    except Exception as e:
        logger.error(f"Security patching failed: {e}")
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_security_patch()