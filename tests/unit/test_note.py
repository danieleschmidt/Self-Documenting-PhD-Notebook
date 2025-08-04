"""
Unit tests for the Note class.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from phd_notebook.core.note import Note, NoteType, Link, NoteFrontmatter


class TestNote:
    """Test cases for the Note class."""
    
    def test_note_creation(self):
        """Test basic note creation."""
        note = Note(
            title="Test Note",
            content="This is a test note.",
            note_type=NoteType.IDEA
        )
        
        assert note.title == "Test Note"
        assert note.content == "This is a test note."
        assert note.note_type == NoteType.IDEA
        assert note.frontmatter.title == "Test Note"
        assert note.frontmatter.type == NoteType.IDEA
    
    def test_note_with_frontmatter(self):
        """Test note creation with custom frontmatter."""
        frontmatter = {
            "title": "Custom Note",
            "tags": ["#test", "#unit"],
            "status": "active",
            "priority": 4
        }
        
        note = Note(
            title="Custom Note",
            frontmatter=frontmatter,
            note_type=NoteType.PROJECT
        )
        
        assert note.frontmatter.tags == ["#test", "#unit"]
        assert note.frontmatter.status == "active"
        assert note.frontmatter.priority == 4
    
    def test_add_tags(self):
        """Test adding tags to a note."""
        note = Note("Test Note")
        
        note.add_tag("experiment")
        note.add_tag("#analysis")
        
        assert "#experiment" in note.frontmatter.tags
        assert "#analysis" in note.frontmatter.tags
    
    def test_add_multiple_tags(self):
        """Test adding multiple tags at once."""
        note = Note("Test Note")
        
        note.add_tags(["experiment", "analysis", "#important"])
        
        assert "#experiment" in note.frontmatter.tags
        assert "#analysis" in note.frontmatter.tags
        assert "#important" in note.frontmatter.tags
    
    def test_add_section(self):
        """Test adding sections to a note."""
        note = Note("Test Note", "Initial content")
        
        note.add_section("Results", "The experiment was successful.")
        note.add_section("Conclusion", "Further research needed.", level=3)
        
        assert "## Results" in note.content
        assert "The experiment was successful." in note.content
        assert "### Conclusion" in note.content
        assert "Further research needed." in note.content
    
    def test_add_link(self):
        """Test adding links between notes."""
        note = Note("Test Note")
        
        note.add_link("Related Note", "references")
        
        links = note.get_links()
        assert len(links) == 1
        assert links[0].target == "Related Note"
        assert links[0].relationship == "references"
        assert "[[Related Note]]" in note.content
    
    def test_extract_links_from_content(self):
        """Test extracting existing links from content."""
        content = """
        This note references [[Another Note]] and also [[Third Note|with display text]].
        There's also a link to [[Fourth Note]].
        """
        
        note = Note("Test Note", content)
        note._extract_links_from_content()
        
        links = note.get_links()
        targets = [link.target for link in links]
        
        assert "Another Note" in targets
        assert "Third Note" in targets
        assert "Fourth Note" in targets
    
    def test_add_figure(self):
        """Test adding figures to notes."""
        note = Note("Test Note")
        
        note.add_figure("figure.png", "This is a test figure")
        
        assert "![[figure.png]]" in note.content
        assert "*This is a test figure*" in note.content
    
    def test_create_version(self):
        """Test version creation and tracking."""
        note = Note("Test Note", "Original content")
        
        note.create_version("initial")
        note.update_content("Updated content")
        note.create_version("updated")
        
        assert len(note._versions) == 2
        assert note._versions[0]["label"] == "initial"
        assert note._versions[0]["content"] == "Original content"
        assert note._versions[1]["label"] == "updated"
        assert note.content == "Updated content"
    
    def test_to_markdown(self):
        """Test markdown export."""
        note = Note(
            title="Test Note",
            content="This is the content.",
            note_type=NoteType.EXPERIMENT
        )
        note.add_tags(["#test", "#unit"])
        
        markdown = note.to_markdown()
        
        assert "---" in markdown
        assert "title: Test Note" in markdown
        assert "type: experiment" in markdown
        assert "tags:" in markdown
        assert "#test" in markdown
        assert "#unit" in markdown
        assert "This is the content." in markdown
    
    def test_save_and_load(self):
        """Test saving and loading notes from files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "test_note.md"
            
            # Create and save note
            original_note = Note(
                title="Test Note",
                content="Test content",
                note_type=NoteType.LITERATURE
            )
            original_note.add_tags(["#test", "#literature"])
            original_note.save(file_path)
            
            # Load note from file
            loaded_note = Note.from_file(file_path)
            
            assert loaded_note.title == "Test Note"
            assert loaded_note.content.strip() == "Test content"
            assert loaded_note.note_type == NoteType.LITERATURE
            assert "#test" in loaded_note.frontmatter.tags
            assert "#literature" in loaded_note.frontmatter.tags
    
    def test_parse_markdown_with_frontmatter(self):
        """Test parsing markdown with YAML frontmatter."""
        markdown_content = """---
title: "Sample Note"
created: 2024-01-01T10:00:00
tags: ["#research", "#important"]
type: experiment
status: active
priority: 5
---

# Sample Note

This is the main content of the note.

## Results

Some experimental results here.
"""
        
        frontmatter, body = Note._parse_markdown(markdown_content)
        
        assert frontmatter["title"] == "Sample Note"
        assert frontmatter["type"] == "experiment"
        assert frontmatter["status"] == "active"
        assert frontmatter["priority"] == 5
        assert "#research" in frontmatter["tags"]
        assert "#important" in frontmatter["tags"]
        assert "# Sample Note" in body
        assert "## Results" in body
    
    def test_parse_markdown_without_frontmatter(self):
        """Test parsing markdown without frontmatter."""
        markdown_content = """# Simple Note

This note has no frontmatter.

Just plain markdown content.
"""
        
        frontmatter, body = Note._parse_markdown(markdown_content)
        
        assert frontmatter == {}
        assert body == markdown_content
    
    def test_update_timestamp(self):
        """Test that timestamps are updated correctly."""
        note = Note("Test Note")
        original_updated = note.frontmatter.updated
        
        # Add small delay to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        note.add_tag("test")
        
        assert note.frontmatter.updated > original_updated
    
    def test_get_related_notes(self):
        """Test getting related notes from links and backlinks."""
        note = Note("Main Note")
        
        # Add outgoing links
        note.add_link("Note A")
        note.add_link("Note B")
        
        # Add backlinks
        note.add_backlink("Note C")
        note.add_backlink("Note D")
        
        related = note.get_related_notes()
        
        assert "Note A" in related
        assert "Note B" in related
        assert "Note C" in related
        assert "Note D" in related
        assert len(related) == 4


class TestLink:
    """Test cases for the Link class."""
    
    def test_link_creation(self):
        """Test basic link creation."""
        link = Link("Target Note", "references", 0.8)
        
        assert link.target == "Target Note"
        assert link.relationship == "references"
        assert link.confidence == 0.8
        assert isinstance(link.created_at, datetime)
    
    def test_link_default_values(self):
        """Test link with default values."""
        link = Link("Target Note")
        
        assert link.target == "Target Note"
        assert link.relationship == "relates_to"
        assert link.confidence == 1.0


class TestNoteFrontmatter:
    """Test cases for the NoteFrontmatter class."""
    
    def test_frontmatter_creation(self):
        """Test frontmatter creation with defaults."""
        fm = NoteFrontmatter(title="Test Note")
        
        assert fm.title == "Test Note"
        assert fm.type == NoteType.IDEA
        assert fm.status == "draft"
        assert fm.priority == 3
        assert isinstance(fm.created, datetime)
        assert isinstance(fm.updated, datetime)
        assert fm.tags == []
    
    def test_frontmatter_with_custom_values(self):
        """Test frontmatter with custom values."""
        custom_date = datetime(2024, 1, 1, 12, 0, 0)
        
        fm = NoteFrontmatter(
            title="Custom Note",
            created=custom_date,
            tags=["#test", "#custom"],
            type=NoteType.EXPERIMENT,
            status="active",
            priority=5,
            author="Test Author",
            project="Test Project"
        )
        
        assert fm.title == "Custom Note"
        assert fm.created == custom_date
        assert fm.tags == ["#test", "#custom"]
        assert fm.type == NoteType.EXPERIMENT
        assert fm.status == "active"
        assert fm.priority == 5
        assert fm.author == "Test Author"
        assert fm.project == "Test Project"