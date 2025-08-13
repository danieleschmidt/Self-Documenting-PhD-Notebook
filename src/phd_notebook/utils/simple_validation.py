"""
Simple validation system without external dependencies.
"""

import re
from typing import Any, Dict, List, Optional, Union


class ValidationError(Exception):
    """Custom validation error for research data."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


def validate_note_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate basic note data."""
    validated = {}
    
    # Title validation
    title = data.get('title', '')
    if not title or not title.strip():
        raise ValidationError('Title cannot be empty', field='title')
    
    if len(title.strip()) > 200:
        raise ValidationError('Title cannot exceed 200 characters', field='title')
    
    if re.search(r'[<>:"|?*\\]', title):
        raise ValidationError('Title contains invalid characters', field='title')
    
    validated['title'] = title.strip()
    
    # Content validation
    content = data.get('content', '')
    if '<script' in content.lower() or 'javascript:' in content.lower():
        raise ValidationError('Content contains potentially unsafe elements', field='content')
    
    validated['content'] = content
    
    # Tags validation
    tags = data.get('tags', [])
    if not isinstance(tags, list):
        tags = []
    
    valid_tags = []
    for tag in tags:
        if isinstance(tag, str):
            clean_tag = tag.strip()
            if clean_tag and not clean_tag.startswith('#'):
                clean_tag = f"#{clean_tag}"
            
            if clean_tag and re.match(r'^#[a-zA-Z0-9_-]+$', clean_tag):
                if clean_tag not in valid_tags:
                    valid_tags.append(clean_tag)
    
    validated['tags'] = valid_tags
    
    # Note type validation
    note_type = data.get('type', 'idea')
    valid_types = ['experiment', 'literature', 'idea', 'meeting', 'project', 'analysis']
    if note_type not in valid_types:
        note_type = 'idea'
    
    validated['type'] = note_type
    
    # Status validation
    status = data.get('status', 'draft')
    valid_statuses = ['draft', 'active', 'completed', 'archived']
    if status not in valid_statuses:
        status = 'draft'
    
    validated['status'] = status
    
    # Priority validation
    priority = data.get('priority', 3)
    if not isinstance(priority, int) or priority < 1 or priority > 5:
        priority = 3
    
    validated['priority'] = priority
    
    return validated


def validate_file_path(path: str) -> str:
    """Validate file path for security."""
    if '..' in path or path.startswith('/'):
        raise ValidationError('Invalid file path', field='path')
    
    return path


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for cross-platform compatibility."""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"|?*\\]', '_', filename)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip('. ')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = 'untitled'
    
    return sanitized