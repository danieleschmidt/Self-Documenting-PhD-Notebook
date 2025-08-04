"""
Integration tests for complete notebook workflows.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from phd_notebook import ResearchNotebook, create_phd_workflow
from phd_notebook.core.note import NoteType
from phd_notebook.agents.base import SimpleAgent
from phd_notebook.connectors.base import FileSystemConnector


class TestCompleteWorkflow:
    """Test complete research workflow scenarios."""
    
    @pytest.fixture
    def temp_vault(self):
        """Create a temporary vault for testing."""
        temp_dir = tempfile.mkdtemp()
        vault_path = Path(temp_dir) / "test_vault"
        
        yield vault_path
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_full_notebook_lifecycle(self, temp_vault):
        """Test complete notebook lifecycle from creation to analysis."""
        # Initialize notebook
        notebook = ResearchNotebook(
            vault_path=temp_vault,
            author="Test Researcher",
            institution="Test University",
            field="Computer Science",
            subfield="Machine Learning"
        )
        
        # Verify initialization
        assert temp_vault.exists()
        assert (temp_vault / "daily").exists()
        assert (temp_vault / "experiments").exists()
        assert (temp_vault / "literature").exists()
        
        # Create various types of notes
        idea_note = notebook.create_note(
            "Novel AI Architecture",
            "Exploring transformer variants for improved efficiency.",
            note_type=NoteType.IDEA,
            tags=["ai", "transformers", "efficiency"]
        )
        
        literature_note = notebook.create_note(
            "Attention Is All You Need",
            "Seminal paper on transformer architecture.",
            note_type=NoteType.LITERATURE,
            tags=["literature", "transformers", "attention"]
        )
        
        # Create experiment
        with notebook.new_experiment("Transformer Efficiency Test", "New architecture will be 20% faster") as exp:
            exp.add_section("Methodology", "Compare training time and accuracy metrics.")
            exp.add_section("Results", "Achieved 18% improvement in training speed.")
            exp.add_link("Novel AI Architecture", "tests_idea")
        
        # Verify notes were created
        notes = notebook.list_notes()
        assert len(notes) >= 3
        
        note_titles = [note.title for note in notes]
        assert "Novel AI Architecture" in note_titles
        assert "Attention Is All You Need" in note_titles
        assert "Transformer Efficiency Test" in note_titles
        
        # Test search functionality
        search_results = notebook.search_notes("transformer")
        assert len(search_results) >= 2
        
        # Test filtering
        experiments = notebook.list_notes(note_type=NoteType.EXPERIMENT)
        assert len(experiments) == 1
        assert experiments[0].title == "Transformer Efficiency Test"
        
        # Test knowledge graph
        kg_stats = notebook.knowledge_graph.find_clusters()
        assert len(kg_stats) > 0
        
        # Test statistics
        stats = notebook.get_stats()
        assert stats["total_notes"] >= 3
        assert "Computer Science" in stats["research_context"]["field"]
        
        # Test related notes
        related = notebook.knowledge_graph.get_connected_notes("Novel AI Architecture")
        assert "Transformer Efficiency Test" in related
    
    def test_experiment_workflow(self, temp_vault):
        """Test complete experiment workflow."""
        notebook = ResearchNotebook(vault_path=temp_vault, author="Test Researcher")
        
        # Create experiment with context manager
        experiment_title = "Learning Rate Optimization"
        hypothesis = "Adaptive learning rates will improve convergence"
        
        with notebook.new_experiment(experiment_title, hypothesis) as exp:
            # Add methodology
            exp.add_section("Materials", "PyTorch, CUDA, Adam optimizer")
            exp.add_section("Procedure", "1. Set up baseline\n2. Test adaptive rates\n3. Compare results")
            
            # Add results
            exp.add_section("Results", "Adaptive rate achieved 15% faster convergence")
            
            # Add figure
            exp.add_figure("convergence_plot.png", "Convergence comparison between methods")
            
            # Link to related work
            exp.add_link("Previous Learning Rate Study", "builds_on")
        
        # Verify experiment was created properly
        experiments = notebook.list_notes(note_type=NoteType.EXPERIMENT)
        assert len(experiments) == 1
        
        exp_note = experiments[0]
        assert exp_note.title == experiment_title
        assert hypothesis in exp_note.content
        assert "Materials" in exp_note.content
        assert "Results" in exp_note.content
        assert "![[convergence_plot.png]]" in exp_note.content
        assert "[[Previous Learning Rate Study]]" in exp_note.content
        
        # Check that experiment status was updated
        assert "#completed" in exp_note.frontmatter.tags
        assert "#active" not in exp_note.frontmatter.tags
        assert exp_note.frontmatter.status == "completed"
    
    def test_agent_integration(self, temp_vault):
        """Test AI agent integration workflow."""
        notebook = ResearchNotebook(vault_path=temp_vault, author="Test Researcher")
        
        # Create a simple test agent
        def process_note(note_content, **kwargs):
            return f"Processed: {note_content[:50]}..."
        
        test_agent = SimpleAgent(
            name="test_processor",
            process_function=process_note,
            capabilities=["text_processing"]
        )
        
        # Register agent
        notebook.register_agent(test_agent)
        
        # Verify agent registration
        agents = notebook.list_agents()
        assert "test_processor" in agents
        
        agent = notebook.get_agent("test_processor")
        assert agent is not None
        assert agent.can_handle("text_processing")
        
        # Create note and process with agent
        note = notebook.create_note(
            "Test Note for Processing",
            "This is a long note that will be processed by our test agent to demonstrate the integration workflow."
        )
        
        # Process with agent
        result = agent.process(note.content)
        assert result.startswith("Processed:")
        assert "This is a long note that will be processed" in result
    
    def test_data_connector_workflow(self, temp_vault):
        """Test data connector integration workflow."""
        notebook = ResearchNotebook(vault_path=temp_vault, author="Test Researcher")
        
        # Create test data directory
        data_dir = temp_vault.parent / "test_data"
        data_dir.mkdir()
        
        # Create test files
        (data_dir / "data1.txt").write_text("Research finding 1: Important discovery")
        (data_dir / "data2.txt").write_text("Research finding 2: Another insight")
        
        # Create and connect file system connector
        fs_connector = FileSystemConnector(str(data_dir), "*.txt")
        
        # Test connection
        assert fs_connector.test_connection()
        
        # Connect to notebook
        notebook.connect_sources({
            'local_files': {
                'research_data': fs_connector
            }
        })
        
        # Sync data
        synced_count = fs_connector.sync(notebook)
        assert synced_count == 2
        
        # Verify notes were created from synced data
        notes = notebook.search_notes("research finding")
        assert len(notes) >= 2
        
        # Check that notes have auto-import tags
        auto_imported = notebook.search_notes("auto-imported")
        assert len(auto_imported) >= 2
    
    def test_knowledge_graph_analysis(self, temp_vault):
        """Test knowledge graph analysis workflow."""
        notebook = ResearchNotebook(vault_path=temp_vault, author="Test Researcher")
        
        # Create interconnected notes
        ai_note = notebook.create_note(
            "Artificial Intelligence Overview",
            "AI is a broad field encompassing machine learning and deep learning.",
            tags=["ai", "overview"]
        )
        
        ml_note = notebook.create_note(
            "Machine Learning Fundamentals",
            "ML is a subset of AI focused on learning from data.",
            tags=["ml", "fundamentals"]
        )
        ml_note.add_link("Artificial Intelligence Overview", "subset_of")
        
        dl_note = notebook.create_note(
            "Deep Learning Methods",
            "Deep learning uses neural networks with multiple layers.",
            tags=["dl", "neural-networks"]
        )
        dl_note.add_link("Machine Learning Fundamentals", "subset_of")
        dl_note.add_link("Artificial Intelligence Overview", "part_of")
        
        # Create an experiment that connects to the concepts
        with notebook.new_experiment("CNN Image Classification", "CNNs will achieve >90% accuracy") as exp:
            exp.add_link("Deep Learning Methods", "uses")
            exp.add_link("Machine Learning Fundamentals", "applies")
        
        # Analyze knowledge graph
        kg = notebook.knowledge_graph
        
        # Test clustering
        clusters = kg.find_clusters()
        assert len(clusters) > 0
        
        # Test centrality analysis
        centrality = kg.calculate_centrality()
        assert len(centrality) > 0
        
        # AI Overview should be fairly central
        assert "Artificial Intelligence Overview" in centrality
        
        # Test connected notes
        connected = kg.get_connected_notes("Deep Learning Methods")
        assert "Machine Learning Fundamentals" in connected
        assert "Artificial Intelligence Overview" in connected
        
        # Test research gap identification
        gaps = kg.identify_research_gaps()
        # Should find some gaps or bridge opportunities
        assert isinstance(gaps, list)
        
        # Test connection suggestions
        suggestions = kg.suggest_connections("Machine Learning Fundamentals")
        assert isinstance(suggestions, list)
    
    def test_search_and_discovery(self, temp_vault):
        """Test search and discovery workflow."""
        notebook = ResearchNotebook(vault_path=temp_vault, author="Test Researcher")
        
        # Create diverse content
        notebook.create_note(
            "Transformer Architecture Study",
            "Analysis of attention mechanisms in transformer models for NLP tasks.",
            note_type=NoteType.LITERATURE,
            tags=["transformers", "nlp", "attention"]
        )
        
        notebook.create_note(
            "BERT Implementation",
            "Implementing BERT for text classification using transformers library.",
            note_type=NoteType.PROJECT,
            tags=["bert", "transformers", "classification"]
        )
        
        notebook.create_note(
            "Attention Visualization",
            "Visualizing attention weights in transformer models to understand focus patterns.",
            note_type=NoteType.IDEA,
            tags=["attention", "visualization", "interpretability"]
        )
        
        # Test basic search
        transformer_results = notebook.search_notes("transformer")
        assert len(transformer_results) >= 2
        
        # Test tag-based filtering
        attention_notes = notebook.search_notes("attention", tags=["attention"])
        assert len(attention_notes) >= 2
        
        # Test type-based filtering
        literature_notes = notebook.search_notes("transformer", note_type="literature")
        assert len(literature_notes) >= 1
        assert literature_notes[0].note_type == NoteType.LITERATURE
        
        # Test combined filtering
        project_transformers = notebook.list_notes(
            note_type=NoteType.PROJECT,
            tags=["transformers"]
        )
        assert len(project_transformers) >= 1
        assert project_transformers[0].title == "BERT Implementation"
    
    def test_create_phd_workflow_helper(self, temp_vault):
        """Test the convenience function for PhD workflow creation."""
        notebook = create_phd_workflow(
            field="Computer Science",
            subfield="Artificial Intelligence", 
            institution="Test University",
            expected_duration=4,
            vault_path=temp_vault,
            author="PhD Student"
        )
        
        assert notebook.field == "Computer Science"
        assert notebook.subfield == "Artificial Intelligence"
        assert notebook.institution == "Test University"
        assert notebook.expected_duration == 4
        assert notebook.author == "PhD Student"
        
        # Verify vault was created
        assert temp_vault.exists()
        assert (temp_vault / "Welcome.md").exists()
        
        # Test that we can use all functionality
        notebook.create_note("Test Research Idea", tags=["ai", "research"])
        notes = notebook.list_notes()
        assert len(notes) > 0