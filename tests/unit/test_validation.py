"""
Unit tests for validation utilities.
"""

import pytest
from datetime import datetime

from phd_notebook.utils.validation import (
    validate_note_data,
    validate_experiment_data,
    validate_config,
    validate_file_path,
    validate_yaml_frontmatter,
    validate_search_query,
    validate_tag_name,
    ValidationError,
    DataIntegrityChecker
)


class TestNoteValidation:
    """Test note data validation."""
    
    def test_valid_note_data(self):
        """Test validation of valid note data."""
        data = {
            'title': 'Test Note',
            'content': 'This is test content.',
            'tags': ['research', 'test'],
            'note_type': 'idea',
            'priority': 3
        }
        
        validated = validate_note_data(data)
        
        assert validated['title'] == 'Test Note'
        assert validated['content'] == 'This is test content.'
        assert '#research' in validated['tags']
        assert '#test' in validated['tags']
        assert validated['note_type'] == 'idea'
        assert validated['priority'] == 3
    
    def test_empty_title_validation(self):
        """Test that empty titles are rejected."""
        data = {
            'title': '',
            'content': 'Content',
            'note_type': 'idea'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_note_data(data)
        
        assert 'Title cannot be empty' in str(exc_info.value)
    
    def test_long_title_validation(self):
        """Test that overly long titles are rejected."""
        data = {
            'title': 'A' * 250,  # 250 characters
            'content': 'Content',
            'note_type': 'idea'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_note_data(data)
        
        assert 'Title cannot exceed 200 characters' in str(exc_info.value)
    
    def test_invalid_characters_in_title(self):
        """Test that invalid characters in titles are rejected."""
        invalid_titles = [
            'Title with <script>',
            'Title with |',
            'Title with *',
            'Title with ?'
        ]
        
        for title in invalid_titles:
            data = {
                'title': title,
                'content': 'Content',
                'note_type': 'idea'
            }
            
            with pytest.raises(ValidationError):
                validate_note_data(data)
    
    def test_tag_normalization(self):
        """Test tag normalization and validation."""
        data = {
            'title': 'Test Note',
            'tags': ['research', '#experiment', 'machine-learning', 'ai_research'],
            'note_type': 'idea'
        }
        
        validated = validate_note_data(data)
        
        assert '#research' in validated['tags']
        assert '#experiment' in validated['tags']
        assert '#machine-learning' in validated['tags']
        assert '#ai_research' in validated['tags']
    
    def test_invalid_tag_format(self):
        """Test that invalid tag formats are rejected."""
        data = {
            'title': 'Test Note',
            'tags': ['valid-tag', 'invalid tag with spaces', 'invalid@tag'],
            'note_type': 'idea'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_note_data(data)
        
        assert 'Invalid tag format' in str(exc_info.value)
    
    def test_invalid_note_type(self):
        """Test that invalid note types are rejected."""
        data = {
            'title': 'Test Note',
            'note_type': 'invalid_type'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_note_data(data)
        
        assert 'Note type must be one of' in str(exc_info.value)
    
    def test_priority_validation(self):
        """Test priority validation."""
        # Valid priorities
        for priority in [1, 2, 3, 4, 5]:
            data = {
                'title': 'Test Note',
                'priority': priority,
                'note_type': 'idea'
            }
            validated = validate_note_data(data)
            assert validated['priority'] == priority
        
        # Invalid priorities
        for priority in [0, 6, -1, 'high']:
            data = {
                'title': 'Test Note',
                'priority': priority,
                'note_type': 'idea'
            }
            
            with pytest.raises(ValidationError):
                validate_note_data(data)
    
    def test_unsafe_content_detection(self):
        """Test detection of unsafe content."""
        unsafe_contents = [
            '<script>alert("xss")</script>',
            'javascript:alert("xss")',
            'Some content with <script> tags'
        ]
        
        for content in unsafe_contents:
            data = {
                'title': 'Test Note',
                'content': content,
                'note_type': 'idea'
            }
            
            with pytest.raises(ValidationError) as exc_info:
                validate_note_data(data)
            
            assert 'unsafe elements' in str(exc_info.value)


class TestExperimentValidation:
    """Test experiment data validation."""
    
    def test_valid_experiment_data(self):
        """Test validation of valid experiment data."""
        data = {
            'title': 'Test Experiment',
            'hypothesis': 'Testing will pass',
            'methodology': 'Run unit tests',
            'status': 'planning',
            'expected_duration': 30
        }
        
        validated = validate_experiment_data(data)
        
        assert validated['title'] == 'Test Experiment'
        assert validated['hypothesis'] == 'Testing will pass'
        assert validated['status'] == 'planning'
        assert validated['expected_duration'] == 30
    
    def test_invalid_experiment_status(self):
        """Test that invalid experiment statuses are rejected."""
        data = {
            'title': 'Test Experiment',
            'status': 'invalid_status'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_experiment_data(data)
        
        assert 'Status must be one of' in str(exc_info.value)
    
    def test_invalid_duration(self):
        """Test that invalid durations are rejected."""
        invalid_durations = [0, -1, 500]
        
        for duration in invalid_durations:
            data = {
                'title': 'Test Experiment',
                'expected_duration': duration
            }
            
            with pytest.raises(ValidationError):
                validate_experiment_data(data)


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test validation of valid configuration."""
        data = {
            'vault_path': '/tmp/test_vault',
            'author': 'Test Author',
            'institution': 'Test University',
            'field': 'Computer Science',
            'backup_enabled': True,
            'max_file_size_mb': 50
        }
        
        validated = validate_config(data)
        
        assert 'vault_path' in validated
        assert validated['author'] == 'Test Author'
        assert validated['backup_enabled'] is True
        assert validated['max_file_size_mb'] == 50
    
    def test_empty_author_validation(self):
        """Test that empty author names are rejected."""
        data = {
            'vault_path': '/tmp/test_vault',
            'author': ''
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_config(data)
        
        assert 'Author name is required' in str(exc_info.value)
    
    def test_invalid_file_size(self):
        """Test that invalid file sizes are rejected."""
        invalid_sizes = [0, -1, 2000]
        
        for size in invalid_sizes:
            data = {
                'vault_path': '/tmp/test_vault',
                'author': 'Test Author',
                'max_file_size_mb': size
            }
            
            with pytest.raises(ValidationError):
                validate_config(data)


class TestUtilityValidation:
    """Test utility validation functions."""
    
    def test_search_query_validation(self):
        """Test search query validation."""
        # Valid queries
        valid_queries = [
            'simple search',
            'research experiment',
            'machine learning AI'
        ]
        
        for query in valid_queries:
            validated = validate_search_query(query)
            assert validated == query
        
        # Invalid queries
        with pytest.raises(ValidationError):
            validate_search_query('')
        
        with pytest.raises(ValidationError):
            validate_search_query('A' * 600)  # Too long
        
        with pytest.raises(ValidationError):
            validate_search_query('<script>alert("xss")</script>')
    
    def test_tag_name_validation(self):
        """Test tag name validation."""
        # Valid tags
        valid_tags = [
            'research',
            'machine-learning',
            'ai_research',
            '#already-prefixed'
        ]
        
        for tag in valid_tags:
            validated = validate_tag_name(tag)
            assert validated.startswith('#')
            assert len(validated) <= 51  # including #
        
        # Invalid tags
        invalid_tags = [
            '',
            '   ',
            'tag with spaces',
            'tag@with@symbols',
            'A' * 60  # Too long
        ]
        
        for tag in invalid_tags:
            with pytest.raises(ValidationError):
                validate_tag_name(tag)
    
    def test_yaml_frontmatter_validation(self):
        """Test YAML frontmatter validation."""
        # Valid YAML
        valid_yaml = """---
title: "Test Note"
tags: ["#research", "#test"]
priority: 5
---

Content here.
"""
        
        frontmatter = validate_yaml_frontmatter(valid_yaml)
        assert frontmatter['title'] == 'Test Note'
        assert frontmatter['priority'] == 5
        assert '#research' in frontmatter['tags']
        
        # Invalid YAML
        invalid_yaml = """---
title: "Test Note"
invalid: yaml: structure
---

Content here.
"""
        
        with pytest.raises(ValidationError):
            validate_yaml_frontmatter(invalid_yaml)
        
        # No frontmatter
        no_frontmatter = "Just content without frontmatter."
        result = validate_yaml_frontmatter(no_frontmatter)
        assert result == {}


class TestDataIntegrityChecker:
    """Test data integrity checking."""
    
    def test_vault_structure_check(self, tmp_path):
        """Test vault structure validation."""
        # Create minimal vault structure
        vault_path = tmp_path / "test_vault"
        vault_path.mkdir()
        
        checker = DataIntegrityChecker(vault_path)
        result = checker.check_vault_structure()
        
        # Should have warnings about missing directories
        assert len(checker.warnings) > 0
        assert any('missing_directory' in warning['type'] for warning in checker.warnings)
        
        # Create required directories
        for dir_name in ['daily', 'projects', 'experiments', 'literature', 'ideas', 'templates']:
            (vault_path / dir_name).mkdir()
        
        checker = DataIntegrityChecker(vault_path)
        result = checker.check_vault_structure()
        
        # Should have no warnings now
        assert len(checker.warnings) == 0
    
    def test_note_files_check(self, tmp_path):
        """Test note files validation."""
        vault_path = tmp_path / "test_vault"
        vault_path.mkdir()
        
        # Create valid note file
        valid_note = vault_path / "valid_note.md"
        valid_note.write_text("""---
title: "Valid Note"
tags: ["#test"]
---

This is valid content.
""")
        
        # Create invalid note file
        invalid_note = vault_path / "invalid_note.md"
        invalid_note.write_text("""---
title: "Invalid Note"
invalid: yaml: structure
---

This has invalid YAML.
""")
        
        checker = DataIntegrityChecker(vault_path)
        result = checker.check_note_files()
        
        # Should detect the invalid file
        assert len(checker.errors) == 1
        assert 'invalid_file' in checker.errors[0]['type']
    
    def test_full_integrity_check(self, tmp_path):
        """Test comprehensive integrity check."""
        vault_path = tmp_path / "test_vault"
        vault_path.mkdir()
        
        # Create some structure
        (vault_path / "daily").mkdir()
        (vault_path / "projects").mkdir()
        
        # Create a valid note
        note_file = vault_path / "daily" / "test_note.md"
        note_file.write_text("""---
title: "Test Note"
tags: ["#daily"]
---

Test content.
""")
        
        checker = DataIntegrityChecker(vault_path)
        results = checker.run_full_check()
        
        assert 'vault_structure' in results
        assert 'note_files' in results
        assert 'broken_links' in results
        assert 'errors' in results
        assert 'warnings' in results
        assert 'overall_health' in results