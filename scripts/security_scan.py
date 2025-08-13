#!/usr/bin/env python3
"""
Security scan script for PhD notebook system.
"""

import os
import sys
import re
import ast
from pathlib import Path
from typing import List, Dict, Any
import subprocess


class SecurityScanner:
    """Basic security scanner for the PhD notebook codebase."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.issues = []
        
    def scan_all(self) -> Dict[str, Any]:
        """Run all security scans."""
        results = {
            'file_permissions': self.check_file_permissions(),
            'sensitive_data': self.scan_for_sensitive_data(),
            'imports': self.check_dangerous_imports(),
            'code_injection': self.check_code_injection_risks(),
            'path_traversal': self.check_path_traversal(),
            'summary': {}
        }
        
        # Calculate summary
        total_issues = sum(len(issues) for issues in results.values() if isinstance(issues, list))
        results['summary'] = {
            'total_issues': total_issues,
            'critical_issues': self.count_critical_issues(results),
            'status': 'PASS' if total_issues == 0 else 'WARNING' if self.count_critical_issues(results) == 0 else 'FAIL'
        }
        
        return results
    
    def check_file_permissions(self) -> List[Dict]:
        """Check for overly permissive file permissions."""
        issues = []
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                stat = py_file.stat()
                mode = oct(stat.st_mode)[-3:]
                
                # Check for world-writable files
                if mode[2] in ['2', '3', '6', '7']:
                    issues.append({
                        'type': 'file_permission',
                        'severity': 'medium',
                        'file': str(py_file),
                        'issue': f'File is world-writable (permissions: {mode})'
                    })
            except (OSError, IndexError):
                continue
                
        return issues
    
    def scan_for_sensitive_data(self) -> List[Dict]:
        """Scan for potential sensitive data in code."""
        issues = []
        
        # Patterns that might indicate sensitive data
        sensitive_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'hardcoded secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'hardcoded token'),
            (r'["\'][A-Za-z0-9]{32,}["\']', 'potential secret/key'),
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            if py_file.name.startswith('.'):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, description in sensitive_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip test files and mock data
                        if 'test' in py_file.name.lower() or 'mock' in content.lower():
                            continue
                            
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'type': 'sensitive_data',
                            'severity': 'high',
                            'file': str(py_file),
                            'line': line_num,
                            'issue': f'Potential {description}: {match.group()[:50]}...'
                        })
                        
            except (UnicodeDecodeError, OSError):
                continue
                
        return issues
    
    def check_dangerous_imports(self) -> List[Dict]:
        """Check for potentially dangerous imports."""
        issues = []
        
        dangerous_modules = [
            'eval', 'exec', 'compile', 'execfile',
            'subprocess.call', 'os.system', 'commands',
            'pickle.loads', 'marshal.loads', 'cPickle.loads'
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for dangerous in dangerous_modules:
                    if dangerous in content:
                        line_num = None
                        for i, line in enumerate(content.split('\n'), 1):
                            if dangerous in line and not line.strip().startswith('#'):
                                line_num = i
                                break
                        
                        if line_num:
                            issues.append({
                                'type': 'dangerous_import',
                                'severity': 'medium',
                                'file': str(py_file),
                                'line': line_num,
                                'issue': f'Potentially dangerous function: {dangerous}'
                            })
                            
            except (UnicodeDecodeError, OSError):
                continue
                
        return issues
    
    def check_code_injection_risks(self) -> List[Dict]:
        """Check for code injection risks."""
        issues = []
        
        injection_patterns = [
            (r'eval\s*\(', 'eval() usage'),
            (r'exec\s*\(', 'exec() usage'),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'shell injection risk'),
            (r'os\.system\s*\(', 'os.system() usage'),
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            # Skip security scan files themselves
            if 'security_scan' in str(py_file) or 'test' in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, description in injection_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'type': 'code_injection',
                            'severity': 'high',
                            'file': str(py_file),
                            'line': line_num,
                            'issue': f'{description}: {match.group()}'
                        })
                        
            except (UnicodeDecodeError, OSError):
                continue
                
        return issues
    
    def check_path_traversal(self) -> List[Dict]:
        """Check for path traversal vulnerabilities."""
        issues = []
        
        traversal_patterns = [
            (r'open\s*\([^)]*\.\./[^)]*\)', 'potential path traversal in file open'),
            (r'Path\s*\([^)]*\.\./[^)]*\)', 'potential path traversal in Path'),
            (r'["\'][^"\']*\.\./[^"\']*["\']', 'hardcoded path traversal'),
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, description in traversal_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        # Skip if in comments
                        line_start = content.rfind('\n', 0, match.start()) + 1
                        line_end = content.find('\n', match.end())
                        if line_end == -1:
                            line_end = len(content)
                        line = content[line_start:line_end]
                        
                        if line.strip().startswith('#'):
                            continue
                            
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'type': 'path_traversal',
                            'severity': 'medium',
                            'file': str(py_file),
                            'line': line_num,
                            'issue': f'{description}: {match.group()[:50]}...'
                        })
                        
            except (UnicodeDecodeError, OSError):
                continue
                
        return issues
    
    def count_critical_issues(self, results: Dict) -> int:
        """Count critical/high severity issues."""
        count = 0
        for category, issues in results.items():
            if isinstance(issues, list):
                count += len([issue for issue in issues if issue.get('severity') == 'high'])
        return count


def main():
    """Run security scan."""
    project_root = Path(__file__).parent.parent
    scanner = SecurityScanner(project_root)
    
    print("🔒 Running Security Scan...")
    print("=" * 50)
    
    results = scanner.scan_all()
    
    # Print results
    for category, issues in results.items():
        if category == 'summary':
            continue
            
        print(f"\n📋 {category.replace('_', ' ').title()}:")
        if isinstance(issues, list):
            if issues:
                for issue in issues:
                    severity_emoji = "🔴" if issue['severity'] == 'high' else "🟡"
                    print(f"  {severity_emoji} {issue['file']}:{issue.get('line', '?')} - {issue['issue']}")
            else:
                print("  ✅ No issues found")
    
    # Print summary
    summary = results['summary']
    print(f"\n📊 Security Scan Summary:")
    print(f"  Total Issues: {summary['total_issues']}")
    print(f"  Critical Issues: {summary['critical_issues']}")
    print(f"  Status: {'✅' if summary['status'] == 'PASS' else '⚠️' if summary['status'] == 'WARNING' else '❌'} {summary['status']}")
    
    # Exit with appropriate code
    if summary['status'] == 'FAIL':
        sys.exit(1)
    elif summary['status'] == 'WARNING':
        sys.exit(0)  # Warnings are acceptable
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()