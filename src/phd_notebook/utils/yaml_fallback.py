"""
YAML fallback for systems without PyYAML.
Provides basic YAML functionality for frontmatter.
"""

import json
import re
from typing import Any, Dict


class BasicYAMLError(Exception):
    """Basic YAML parsing error."""
    pass


def safe_load(yaml_str: str) -> Dict[str, Any]:
    """
    Basic YAML parser for simple frontmatter.
    Only supports basic key-value pairs and lists.
    """
    if not yaml_str or not yaml_str.strip():
        return {}
    
    result = {}
    lines = yaml_str.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Handle key-value pairs
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Handle different value types
            if not value:
                result[key] = ""
            elif value.lower() in ('true', 'false'):
                result[key] = value.lower() == 'true'
            elif value.startswith('[') and value.endswith(']'):
                # Simple list parsing
                list_str = value[1:-1]
                if not list_str.strip():
                    result[key] = []
                else:
                    items = [item.strip().strip('"\'') for item in list_str.split(',')]
                    result[key] = items
            elif value.startswith('"') and value.endswith('"'):
                result[key] = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                result[key] = value[1:-1]
            else:
                # Try to parse as number
                try:
                    if '.' in value:
                        result[key] = float(value)
                    else:
                        result[key] = int(value)
                except ValueError:
                    result[key] = value
    
    return result


def safe_dump(data: Dict[str, Any], **kwargs) -> str:
    """
    Basic YAML dumper for simple data structures.
    """
    if not data:
        return ""
    
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            # Format list
            if not value:
                lines.append(f"{key}: []")
            else:
                list_items = ', '.join(f'"{item}"' if isinstance(item, str) else str(item) for item in value)
                lines.append(f"{key}: [{list_items}]")
        elif isinstance(value, str):
            lines.append(f'{key}: "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        else:
            lines.append(f"{key}: {value}")
    
    return '\n'.join(lines)


# Try to import real YAML, fall back to basic implementation
try:
    import yaml
    safe_load = yaml.safe_load
    safe_dump = yaml.safe_dump
    YAMLError = yaml.YAMLError
except ImportError:
    YAMLError = BasicYAMLError