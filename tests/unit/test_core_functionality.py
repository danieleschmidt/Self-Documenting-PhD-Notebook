"""
Comprehensive tests for core PhD Notebook functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.phd_notebook.core.notebook import ResearchNotebook
from src.phd_notebook.core.note import Note, NoteType
from src.phd_notebook.core.vault_manager import VaultManager
from src.phd_notebook.agents import LiteratureAgent, ExperimentAgent, SPARCAgent


class TestResearchNotebook:
    """Test ResearchNotebook functionality."""
    
    @pytest.fixture
    def temp_vault(self):
        """Create temporary vault for testing."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "test_vault"
        yield vault_path
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def notebook(self, temp_vault):
        """Create test notebook."""
        return ResearchNotebook(
            vault_path=temp_vault,
            author="Test Author",
            institution="Test University",
            field="Computer Science"
        )
    
    def test_notebook_initialization(self, notebook, temp_vault):
        """Test notebook creates properly."""
        assert notebook.vault_path == temp_vault
        assert notebook.author == "Test Author"
        assert notebook.field == "Computer Science"
        assert temp_vault.exists()
        
        # Check required directories
        assert (temp_vault / "daily").exists()
        assert (temp_vault / "experiments").exists()
        assert (temp_vault / "literature").exists()
        assert (temp_vault / "templates").exists()
    
    def test_note_creation(self, notebook):
        """Test note creation."""
        note = notebook.create_note(
            title="Test Note",
            content="This is test content",
            note_type=NoteType.IDEA,
            tags=["test", "example"]
        )
        
        assert note.title == "Test Note"
        assert note.content == "This is test content"
        assert note.note_type == NoteType.IDEA
        assert "#test" in note.frontmatter.tags
        assert "#example" in note.frontmatter.tags
        assert note.file_path.exists()
    
    def test_note_retrieval(self, notebook):
        """Test note retrieval."""
        # Create note
        original = notebook.create_note(
            title="Retrievable Note",
            content="Content to retrieve"
        )
        
        # Retrieve by title
        retrieved = notebook.get_note("Retrievable Note")
        assert retrieved is not None
        assert retrieved.title == "Retrievable Note"
        assert retrieved.content == "Content to retrieve"
    
    def test_note_search(self, notebook):
        """Test note searching."""
        # Create test notes
        notebook.create_note(
            title="Machine Learning Basics",
            content="Deep learning and neural networks",
            tags=["ml", "ai"]
        )
        notebook.create_note(
            title="Python Programming",
            content="Scripting and automation",
            tags=["programming"]
        )
        
        # Search by title
        results = notebook.search_notes("Machine Learning")
        assert len(results) == 1
        assert results[0].title == "Machine Learning Basics"
        
        # Search by content
        results = notebook.search_notes("neural", in_content=True)
        assert len(results) == 1
        
        # Search by tag
        results = notebook.search_notes("ml")
        assert len(results) == 1
    
    def test_agent_registration(self, notebook):
        """Test AI agent registration."""
        lit_agent = LiteratureAgent()
        notebook.register_agent(lit_agent)
        
        assert "LiteratureAgent" in notebook.list_agents()
        assert notebook.get_agent("LiteratureAgent") == lit_agent
        assert lit_agent.notebook == notebook
    
    def test_experiment_context_manager(self, notebook):
        """Test experiment context manager."""
        with notebook.new_experiment("Test Experiment", "Test hypothesis") as exp:
            assert exp.title == "Test Experiment"
            assert "#active" in exp.frontmatter.tags
            assert exp.frontmatter.metadata["hypothesis"] == "Test hypothesis"
        
        # After context exit, should be completed
        assert "#completed" in exp.frontmatter.tags
        assert "#active" not in exp.frontmatter.tags


class TestNote:
    """Test Note functionality."""
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary file for testing."""
        temp_dir = tempfile.mkdtemp()
        file_path = Path(temp_dir) / "test_note.md"
        yield file_path
        shutil.rmtree(temp_dir)
    
    def test_note_creation(self):
        """Test basic note creation."""
        note = Note(
            title="Test Note",
            content="Test content",
            note_type=NoteType.IDEA
        )
        
        assert note.title == "Test Note"
        assert note.content == "Test content"
        assert note.note_type == NoteType.IDEA
        assert note.frontmatter.title == "Test Note"
    
    def test_note_save_and_load(self, temp_file):
        """Test saving and loading notes."""
        # Create and save note
        note = Note(
            title="Saveable Note",
            content="Content to save",
            note_type=NoteType.LITERATURE
        )
        note.add_tags(["test", "save"])
        note.save(temp_file)
        
        assert temp_file.exists()
        
        # Load note from file
        loaded = Note.from_file(temp_file)
        assert loaded.title == "Saveable Note"
        assert loaded.content == "Content to save"
        assert loaded.note_type == NoteType.LITERATURE
        assert "#test" in loaded.frontmatter.tags
    
    def test_note_links(self):
        """Test note linking functionality."""
        note = Note(title="Test Note", content="Initial content")
        
        # Add link
        note.add_link("Related Note", "relates_to", 0.8)
        
        links = note.get_links()
        assert len(links) == 1
        assert links[0].target == "Related Note"
        assert links[0].relationship == "relates_to"
        assert links[0].confidence == 0.8
        
        # Content should be updated
        assert "[[Related Note]]" in note.content
    
    def test_note_sections(self):
        """Test adding sections to notes."""
        note = Note(title="Test Note", content="Initial content")
        
        note.add_section("New Section", "Section content", level=2)
        
        assert "## New Section" in note.content
        assert "Section content" in note.content
    
    def test_note_versioning(self):
        """Test note version tracking."""
        note = Note(title="Test Note", content="Original content")
        
        # Create version
        note.create_version("initial")
        
        # Update content
        note.update_content("Updated content")
        
        # Check versions
        assert len(note._versions) == 2
        assert note._versions[0]["label"] == "initial"
        assert note._versions[1]["label"] == "content_update"


class TestAgents:
    """Test AI agents functionality."""
    
    @pytest.fixture
    def notebook(self):
        """Create test notebook."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "test_vault"
        notebook = ResearchNotebook(vault_path=vault_path)
        yield notebook
        shutil.rmtree(temp_dir)
    
    def test_literature_agent(self, notebook):
        """Test literature agent functionality."""
        lit_agent = LiteratureAgent()
        notebook.register_agent(lit_agent)
        
        # Test note creation
        note = lit_agent.create_literature_note(
            title="Test Paper",
            authors="Smith et al.",
            year="2024",
            paper_content="This paper presents novel approaches to machine learning."
        )
        
        assert note.title == "Test Paper"
        assert note.note_type == NoteType.LITERATURE
        assert "#literature" in note.frontmatter.tags
        assert note.frontmatter.metadata["authors"] == "Smith et al."
        assert note.frontmatter.metadata["year"] == "2024"
    
    def test_experiment_agent(self, notebook):
        """Test experiment agent functionality."""
        exp_agent = ExperimentAgent()
        notebook.register_agent(exp_agent)
        
        # Test experiment note creation
        note = exp_agent.create_experiment_note(
            title="Test Experiment",
            hypothesis="This will work",
            experiment_type="controlled"
        )
        
        assert note.title == "Test Experiment"
        assert note.note_type == NoteType.EXPERIMENT
        assert "#experiment" in note.frontmatter.tags
        assert note.frontmatter.metadata["hypothesis"] == "This will work"
        assert note.frontmatter.metadata["experiment_type"] == "controlled"
        
        # Check content structure
        assert "## Hypothesis" in note.content
        assert "## Methodology" in note.content
        assert "## Timeline" in note.content
    
    def test_sparc_agent(self, notebook):
        """Test SPARC agent functionality."""
        sparc_agent = SPARCAgent()
        notebook.register_agent(sparc_agent)
        
        # Test paper draft creation
        note = sparc_agent.create_paper_draft(
            title="Test Paper Draft",
            topic="Machine Learning",
            research_notes=[],
            target_venue="Test Conference"
        )
        
        assert note.title == "Test Paper Draft"
        assert note.note_type == NoteType.PROJECT
        assert "#paper" in note.frontmatter.tags
        assert "#sparc" in note.frontmatter.tags
        assert note.frontmatter.metadata["target_venue"] == "Test Conference"
        
        # Check SPARC structure in content
        assert "## Introduction" in note.content
        assert "## Problem Statement" in note.content
        assert "## Methodology" in note.content
        assert "## Results" in note.content


class TestValidation:
    """Test validation functionality."""
    
    def test_note_validation(self):
        """Test note data validation."""
        from src.phd_notebook.utils.validation import validate_note_data
        
        # Valid data
        valid_data = {
            'title': 'Valid Title',
            'content': 'Valid content',
            'tags': ['test', 'validation'],
            'note_type': 'idea',
            'priority': 3
        }
        
        result = validate_note_data(valid_data)
        assert result['title'] == 'Valid Title'
        assert '#test' in result['tags']
        assert '#validation' in result['tags']
        
        # Invalid data
        invalid_data = {
            'title': '',  # Empty title
            'tags': 'not a list',  # Wrong type
            'priority': 10  # Out of range
        }
        
        with pytest.raises(Exception):  # ValidationError
            validate_note_data(invalid_data)
    
    def test_path_validation(self):
        """Test path validation."""
        from src.phd_notebook.utils.validation import validate_file_path
        
        # Valid path
        valid_path = validate_file_path("/tmp/test.md")
        assert isinstance(valid_path, Path)
        
        # Test with existing file requirement
        temp_file = Path(tempfile.mktemp())
        temp_file.touch()
        
        validated = validate_file_path(temp_file, must_exist=True)
        assert validated.exists()
        
        temp_file.unlink()


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def full_notebook(self):
        """Create notebook with all agents registered."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "integration_vault"
        
        notebook = ResearchNotebook(
            vault_path=vault_path,
            author="Integration Test",
            field="Computer Science"
        )
        
        # Register all agents
        notebook.register_agent(LiteratureAgent())
        notebook.register_agent(ExperimentAgent())
        notebook.register_agent(SPARCAgent())
        
        yield notebook
        shutil.rmtree(temp_dir)
    
    def test_complete_research_workflow(self, full_notebook):
        """Test complete research workflow."""
        # 1. Create literature notes
        lit_agent = full_notebook.get_agent("LiteratureAgent")
        paper1 = lit_agent.create_literature_note(
            "Deep Learning Survey",
            "Goodfellow et al.",
            "2016"
        )
        
        # 2. Create experiment based on literature
        exp_agent = full_notebook.get_agent("ExperimentAgent")
        experiment = exp_agent.create_experiment_note(
            "CNN Performance Test",
            "CNNs will outperform traditional ML on image tasks",
            "comparative"
        )
        
        # 3. Generate paper draft
        sparc_agent = full_notebook.get_agent("SPARCAgent")
        paper_draft = sparc_agent.create_paper_draft(
            "CNN Performance Analysis",
            "Deep Learning",
            [paper1, experiment],
            "AI Conference"
        )
        
        # Verify complete workflow
        assert len(full_notebook.list_notes()) >= 4  # Welcome + 3 created
        assert paper1.note_type == NoteType.LITERATURE
        assert experiment.note_type == NoteType.EXPERIMENT
        assert paper_draft.note_type == NoteType.PROJECT
        
        # Test knowledge graph connections
        kg = full_notebook.knowledge_graph
        assert len(kg.nodes) >= 3
        
        # Test statistics
        stats = full_notebook.get_stats()
        assert stats['total_notes'] >= 4
        assert stats['agents'] == 3
        assert 'literature' in stats['types']
        assert 'experiment' in stats['types']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])