"""
Comprehensive validation system for research data.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ValidationError as PydanticValidationError, validator
import yaml


class ValidationError(Exception):
    """Custom validation error for research data."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class NoteValidator(BaseModel):
    """Pydantic model for note validation."""
    
    title: str
    content: str = ""
    tags: List[str] = []
    note_type: str = "idea"
    status: str = "draft"
    priority: int = 3
    
    @validator('title')
    def title_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        
        if len(v.strip()) > 200:
            raise ValueError('Title cannot exceed 200 characters')
        
        # Check for problematic characters
        if re.search(r'[<>:"|?*\\]', v):
            raise ValueError('Title contains invalid characters')
        
        return v.strip()
    
    @validator('tags')
    def tags_must_be_valid(cls, v):
        if not isinstance(v, list):
            raise ValueError('Tags must be a list')
        
        valid_tags = []
        for tag in v:
            if not isinstance(tag, str):
                raise ValueError('All tags must be strings')
            
            # Normalize tag format
            clean_tag = tag.strip()
            if clean_tag and not clean_tag.startswith('#'):
                clean_tag = f"#{clean_tag}"
            
            # Validate tag format
            if not re.match(r'^#[a-zA-Z0-9_-]+$', clean_tag):
                raise ValueError(f'Invalid tag format: {clean_tag}')
            
            if clean_tag not in valid_tags:
                valid_tags.append(clean_tag)
        
        return valid_tags
    
    @validator('note_type')
    def note_type_must_be_valid(cls, v):
        valid_types = [
            'experiment', 'literature', 'idea', 'meeting', 
            'project', 'analysis', 'methodology', 'observation'
        ]
        if v not in valid_types:
            raise ValueError(f'Note type must be one of: {valid_types}')
        return v
    
    @validator('priority')
    def priority_must_be_valid(cls, v):
        if not isinstance(v, int) or v < 1 or v > 5:
            raise ValueError('Priority must be an integer between 1 and 5')
        return v
    
    @validator('content')
    def content_must_be_safe(cls, v):
        # Check for potential security issues
        if '<script' in v.lower() or 'javascript:' in v.lower():
            raise ValueError('Content contains potentially unsafe elements')
        
        return v


class ExperimentValidator(BaseModel):
    """Validator for experiment data."""
    
    title: str
    hypothesis: str = ""
    methodology: str = ""
    status: str = "planning"
    expected_duration: Optional[int] = None  # days
    
    @validator('status')
    def status_must_be_valid(cls, v):
        valid_statuses = ['planning', 'setup', 'running', 'analysis', 'completed', 'cancelled']
        if v not in valid_statuses:
            raise ValueError(f'Status must be one of: {valid_statuses}')
        return v
    
    @validator('expected_duration')
    def duration_must_be_reasonable(cls, v):
        if v is not None and (v <= 0 or v > 365):
            raise ValueError('Expected duration must be between 1 and 365 days')
        return v


class ConfigValidator(BaseModel):
    """Validator for configuration data."""
    
    vault_path: str
    author: str
    institution: str = ""
    field: str = ""
    backup_enabled: bool = True
    max_file_size_mb: int = 100
    
    @validator('vault_path')
    def path_must_be_valid(cls, v):
        try:
            path = Path(v).expanduser().resolve()
            if not path.parent.exists():
                raise ValueError(f'Parent directory does not exist: {path.parent}')
        except Exception as e:
            raise ValueError(f'Invalid path: {e}')
        return str(path)
    
    @validator('author')
    def author_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError('Author name is required')
        
        if len(v.strip()) > 100:
            raise ValueError('Author name cannot exceed 100 characters')
        
        return v.strip()
    
    @validator('max_file_size_mb')
    def file_size_must_be_reasonable(cls, v):
        if v <= 0 or v > 1000:
            raise ValueError('Max file size must be between 1 and 1000 MB')
        return v


def validate_note_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate note data against schema.
    
    Args:
        data: Dictionary containing note data
        
    Returns:
        Validated and normalized data
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        validator = NoteValidator(**data)
        return validator.dict()
        
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(x) for x in error['loc'])
            message = error['msg']
            errors.append(f"{field}: {message}")
        
        raise ValidationError(f"Validation failed: {'; '.join(errors)}")


def validate_experiment_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate experiment data."""
    try:
        validator = ExperimentValidator(**data)
        return validator.dict()
        
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(x) for x in error['loc'])
            message = error['msg']
            errors.append(f"{field}: {message}")
        
        raise ValidationError(f"Experiment validation failed: {'; '.join(errors)}")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration data."""
    try:
        validator = ConfigValidator(**config)
        return validator.dict()
        
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(x) for x in error['loc'])
            message = error['msg']
            errors.append(f"{field}: {message}")
        
        raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")


def validate_file_path(file_path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate and normalize file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the path must already exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    try:
        path = Path(file_path).expanduser().resolve()
        
        if must_exist and not path.exists():
            raise ValidationError(f"Path does not exist: {path}")
        
        # Check for suspicious paths
        if '..' in str(path) or str(path).startswith('/'):
            # More sophisticated path traversal detection would go here
            pass
        
        return path
        
    except Exception as e:
        raise ValidationError(f"Invalid file path: {e}")


def validate_yaml_frontmatter(content: str) -> Dict[str, Any]:
    """
    Validate YAML frontmatter in markdown content.
    
    Args:
        content: Markdown content with potential frontmatter
        
    Returns:
        Parsed frontmatter dictionary
        
    Raises:
        ValidationError: If YAML is invalid
    """
    if not content.startswith('---\n'):
        return {}
    
    try:
        parts = content.split('---\n', 2)
        if len(parts) < 3:
            return {}
        
        yaml_content = parts[1]
        frontmatter = yaml.safe_load(yaml_content)
        
        if not isinstance(frontmatter, dict):
            raise ValidationError("Frontmatter must be a dictionary")
        
        return frontmatter
        
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML frontmatter: {e}")


def validate_search_query(query: str) -> str:
    """
    Validate and sanitize search query.
    
    Args:
        query: Search query string
        
    Returns:
        Sanitized query
        
    Raises:
        ValidationError: If query is invalid
    """
    if not query or not query.strip():
        raise ValidationError("Search query cannot be empty")
    
    query = query.strip()
    
    if len(query) > 500:
        raise ValidationError("Search query too long (max 500 characters)")
    
    # Remove potentially problematic patterns
    dangerous_patterns = [
        r'<script.*?>',
        r'javascript:',
        r'on\w+\s*=',
        r'eval\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValidationError("Search query contains unsafe content")
    
    return query


def validate_tag_name(tag: str) -> str:
    """
    Validate and normalize tag name.
    
    Args:
        tag: Tag name to validate
        
    Returns:
        Normalized tag name
        
    Raises:
        ValidationError: If tag is invalid
    """
    if not tag or not tag.strip():
        raise ValidationError("Tag cannot be empty")
    
    tag = tag.strip()
    
    # Ensure tag starts with #
    if not tag.startswith('#'):
        tag = f"#{tag}"
    
    # Validate format
    if not re.match(r'^#[a-zA-Z0-9_-]{1,50}$', tag):
        raise ValidationError(
            "Tag must start with # and contain only letters, numbers, underscores, "
            "and hyphens (max 50 characters)"
        )
    
    return tag


class DataIntegrityChecker:
    """Check data integrity across the research vault."""
    
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
    
    def check_vault_structure(self) -> bool:
        """Check if vault has proper structure."""
        required_dirs = ['daily', 'projects', 'experiments', 'literature', 'ideas', 'templates']
        
        for dir_name in required_dirs:
            dir_path = self.vault_path / dir_name
            if not dir_path.exists():
                self.warnings.append({
                    'type': 'missing_directory',
                    'path': dir_path,
                    'message': f'Recommended directory missing: {dir_name}'
                })
        
        return len(self.errors) == 0
    
    def check_note_files(self) -> bool:
        """Check all note files for validity."""
        for md_file in self.vault_path.rglob("*.md"):
            if "templates" in md_file.parts:
                continue
            
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Validate frontmatter
                validate_yaml_frontmatter(content)
                
                # Check file size
                size_mb = md_file.stat().st_size / (1024 * 1024)
                if size_mb > 10:  # 10MB warning threshold
                    self.warnings.append({
                        'type': 'large_file',
                        'path': md_file,
                        'size_mb': size_mb,
                        'message': f'Large file: {size_mb:.1f}MB'
                    })
                
            except Exception as e:
                self.errors.append({
                    'type': 'invalid_file',
                    'path': md_file,
                    'error': str(e),
                    'message': f'Cannot read file: {e}'
                })
        
        return len(self.errors) == 0
    
    def check_broken_links(self) -> bool:
        """Check for broken internal links."""
        # This would be implemented to check [[wiki-links]]
        # For now, just placeholder
        return True
    
    def run_full_check(self) -> Dict[str, Any]:
        """Run comprehensive integrity check."""
        self.errors.clear()
        self.warnings.clear()
        
        results = {
            'vault_structure': self.check_vault_structure(),
            'note_files': self.check_note_files(),
            'broken_links': self.check_broken_links(),
            'errors': self.errors,
            'warnings': self.warnings,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
        }
        
        results['overall_health'] = (
            results['vault_structure'] and 
            results['note_files'] and 
            results['broken_links']
        )
        
        return results