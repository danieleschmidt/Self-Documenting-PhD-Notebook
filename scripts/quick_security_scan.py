#!/usr/bin/env python3
"""
Quick security scan for PhD notebook project files only.
"""

import os
import re
from pathlib import Path


def scan_project_security():
    """Scan only project files for security issues."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    
    issues = []
    
    # Check src directory files only
    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # Check for dangerous patterns in our code (excluding security scan files)
            if 'security_scan' in str(py_file):
                continue
                
            dangerous_patterns = [
                (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'shell injection risk'),
                (r'password\s*=\s*["\'][^"\']{3,}["\']', 'hardcoded password'),
                (r'api_key\s*=\s*["\'][^"\']{10,}["\']', 'hardcoded API key'),
            ]
            
            for pattern, description in dangerous_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Skip comments and test files
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_end = content.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(content)
                    line = content[line_start:line_end]
                    
                    if line.strip().startswith('#') or 'mock' in line.lower():
                        continue
                        
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'file': str(py_file.relative_to(project_root)),
                        'line': line_num,
                        'issue': description,
                        'code': match.group()
                    })
                    
        except (UnicodeDecodeError, OSError):
            continue
    
    return issues


def main():
    """Run quick security scan."""
    print("🔒 Running Quick Security Scan...")
    print("=" * 40)
    
    issues = scan_project_security()
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} potential security issues:")
        for issue in issues:
            print(f"  📄 {issue['file']}:{issue['line']} - {issue['issue']}")
            print(f"     Code: {issue['code']}")
        print(f"\n❌ Security Scan: FAIL")
        return 1
    else:
        print("✅ No security issues found in project files")
        print("✅ Security Scan: PASS")
        return 0


if __name__ == "__main__":
    exit(main())