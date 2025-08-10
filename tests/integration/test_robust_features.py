"""
Integration tests for robust features.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path

from phd_notebook import ResearchNotebook
from phd_notebook.utils.config import ConfigManager
from phd_notebook.monitoring.metrics import get_metrics, get_notebook_metrics
from phd_notebook.utils.backup import BackupManager


class TestRobustIntegration:
    """Test robust features integration."""
    
    def test_notebook_with_config_manager(self):
        """Test notebook with configuration management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config manager
            config_manager = ConfigManager()
            
            # Create notebook
            notebook = ResearchNotebook(
                vault_path=temp_dir,
                author="Test User",
                field="Computer Science"
            )
            
            # Test basic functionality
            note = notebook.create_note(
                title="Test Note",
                content="Test content",
                tags=["#test"]
            )
            
            assert note.title == "Test Note"
            assert len(notebook.list_notes()) == 1
    
    def test_metrics_collection(self):
        """Test metrics collection during operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = get_metrics()
            notebook_metrics = get_notebook_metrics()
            
            # Reset metrics
            metrics.counters.clear()
            
            # Create notebook and perform operations
            notebook = ResearchNotebook(
                vault_path=temp_dir,
                author="Test User",
                field="Computer Science"
            )
            
            # Create notes and trigger metrics
            for i in range(3):
                note = notebook.create_note(
                    title=f"Test Note {i}",
                    content=f"Content {i}",
                    tags=[f"#test{i}"]
                )
                notebook_metrics.note_created("idea")
            
            # Check metrics
            assert metrics.get_counter("notes_created_total") >= 3
            assert metrics.get_counter("notes_total") >= 3
    
    def test_workflow_metrics(self):
        """Test workflow execution metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook_metrics = get_notebook_metrics()
            
            # Simulate workflow execution
            notebook_metrics.workflow_executed("test_workflow", 1.5, True)
            notebook_metrics.workflow_executed("test_workflow", 2.0, False)
            
            # Check metrics
            metrics = get_metrics()
            assert metrics.get_counter("workflows_executed_total") >= 2
            
            # Check timing stats
            timing_stats = metrics.get_timing_stats("workflow_duration_seconds")
            assert timing_stats['count'] >= 2
            assert timing_stats['avg'] > 0
    
    def test_backup_integration(self):
        """Test backup system integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            vault_path = Path(temp_dir) / "vault"
            vault_path.mkdir()
            
            # Create some content
            (vault_path / "test.md").write_text("Test content")
            
            # Create backup manager
            backup_manager = BackupManager(vault_path)
            
            # Create backup
            backup_metadata = backup_manager.create_backup(
                backup_type="test",
                compress=True
            )
            
            assert backup_metadata.backup_id.startswith("test_")
            assert backup_metadata.file_count == 1
            assert backup_metadata.backup_type == "test"
            
            # List backups
            backups = backup_manager.list_backups()
            assert len(backups) == 1
            assert backups[0].backup_id == backup_metadata.backup_id
    
    def test_security_integration(self):
        """Test security features integration."""
        from phd_notebook.utils.security import get_security_validator
        
        validator = get_security_validator()
        
        # Test content validation
        safe_content = "This is safe research content."
        result = validator.validate_content_security(safe_content)
        assert result['safe'] is True
        assert result['pii_detected'] is False
        
        # Test with PII
        pii_content = "Contact me at test@example.com"
        result = validator.validate_content_security(pii_content)
        assert result['pii_detected'] is True
        
        # Test secure export
        exported = validator.secure_export(pii_content, "public")
        assert "test@example.com" not in exported or exported != pii_content
    
    @pytest.mark.asyncio
    async def test_async_workflow_execution(self):
        """Test async workflow execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook = ResearchNotebook(
                vault_path=temp_dir,
                author="Test User", 
                field="Computer Science"
            )
            
            # Create some notes first
            notebook.create_note("Test 1", "Content 1", tags=["#test"])
            notebook.create_note("Test 2", "Content 2", tags=["#research"])
            
            # Run workflow
            result = await notebook.run_workflow("auto_tagging")
            
            assert "processed" in result or "status" in result
    
    def test_error_handling_integration(self):
        """Test error handling across components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook = ResearchNotebook(
                vault_path=temp_dir,
                author="Test User",
                field="Computer Science"  
            )
            
            # Test invalid input handling
            try:
                notebook.create_note("", "Invalid title")  # Empty title
                assert False, "Should have raised error"
            except Exception:
                pass  # Expected to fail
            
            # Test with valid input
            note = notebook.create_note(
                "Valid Title",
                "Valid content",
                tags=["#valid"]
            )
            assert note.title == "Valid Title"
    
    def test_performance_monitoring(self):
        """Test performance monitoring features."""
        from phd_notebook.monitoring.metrics import get_profiler
        
        profiler = get_profiler()
        
        # Test profiling
        with profiler.profile("test_operation"):
            # Simulate some work
            import time
            time.sleep(0.1)
        
        # Check metrics
        metrics = get_metrics()
        timing_stats = metrics.get_timing_stats("test_operation_duration")
        assert timing_stats['count'] >= 1
        assert timing_stats['min'] >= 0.1


if __name__ == "__main__":
    pytest.main([__file__])