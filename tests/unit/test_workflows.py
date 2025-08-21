"""
Tests for workflow functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from phd_notebook.workflows.base_workflow import BaseWorkflow, WorkflowScheduler
from phd_notebook.workflows.automation import (
    WorkflowManager, AutoTaggingWorkflow, SmartLinkingWorkflow, 
    DailyReviewWorkflow, LiteratureProcessingWorkflow
)
from phd_notebook.core.notebook import ResearchNotebook
from phd_notebook.ai.base_ai import MockAI


class TestBaseWorkflow:
    """Test base workflow functionality."""
    
    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = AutoTaggingWorkflow()
        
        assert workflow.name == "auto_tagging"
        assert workflow.enabled == True
        assert workflow.priority == 1
        assert workflow.description == "Automatically generate and apply tags to notes"
        assert workflow.last_run is None
    
    def test_workflow_configuration(self):
        """Test workflow configuration."""
        workflow = SmartLinkingWorkflow()
        workflow.enabled = False
        workflow.priority = 5
        
        assert workflow.name == "smart_linking"
        assert workflow.description == "Create intelligent links between related notes"
        assert workflow.enabled == False
        assert workflow.priority == 5
    
    @pytest.mark.asyncio
    async def test_execute_implementation(self):
        """Test that execute method works with concrete workflow."""
        workflow = AutoTaggingWorkflow()
        
        # Should execute without errors (though may not do much without a notebook)
        result = await workflow.execute()
        assert isinstance(result, dict)
    
    def test_should_run_enabled_check(self):
        """Test should_run method with enabled/disabled workflows."""
        enabled_workflow = AutoTaggingWorkflow()
        disabled_workflow = SmartLinkingWorkflow()
        disabled_workflow.enabled = False
        
        assert enabled_workflow.should_run() == True
        assert disabled_workflow.should_run() == False
    
    def test_workflow_info(self):
        """Test workflow info retrieval."""
        workflow = DailyReviewWorkflow()
        workflow.priority = 3
        
        info = workflow.get_info()
        assert info["name"] == "daily_review"
        assert info["description"] == "Daily review and organization of notes"
        assert info["enabled"] == True
        assert info["priority"] == 3
        assert "last_run" in info


class TestWorkflowScheduler:
    """Test workflow scheduler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = WorkflowScheduler()
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        assert self.scheduler.workflows == []
        assert self.scheduler.running == False
    
    def test_add_workflow(self):
        """Test adding workflows to scheduler."""
        workflow1 = BaseWorkflow(name="workflow1", priority=2)
        workflow2 = BaseWorkflow(name="workflow2", priority=1)
        
        self.scheduler.add_workflow(workflow1)
        self.scheduler.add_workflow(workflow2)
        
        assert len(self.scheduler.workflows) == 2
        # Should be sorted by priority
        assert self.scheduler.workflows[0].name == "workflow2"  # Priority 1
        assert self.scheduler.workflows[1].name == "workflow1"  # Priority 2
    
    def test_remove_workflow(self):
        """Test removing workflows from scheduler."""
        workflow = BaseWorkflow(name="to-remove")
        self.scheduler.add_workflow(workflow)
        
        assert len(self.scheduler.workflows) == 1
        
        self.scheduler.remove_workflow("to-remove")
        assert len(self.scheduler.workflows) == 0
    
    def test_get_workflow(self):
        """Test getting workflow by name."""
        workflow = BaseWorkflow(name="findme", description="Find this workflow")
        self.scheduler.add_workflow(workflow)
        
        found = self.scheduler.get_workflow("findme")
        assert found is not None
        assert found.name == "findme"
        assert found.description == "Find this workflow"
        
        not_found = self.scheduler.get_workflow("nonexistent")
        assert not_found is None
    
    def test_list_workflows(self):
        """Test listing all workflows."""
        workflow1 = BaseWorkflow(name="list1", enabled=True)
        workflow2 = BaseWorkflow(name="list2", enabled=False)
        workflow3 = BaseWorkflow(name="list3", enabled=True)
        
        for wf in [workflow1, workflow2, workflow3]:
            self.scheduler.add_workflow(wf)
        
        all_workflows = self.scheduler.list_workflows()
        assert len(all_workflows) == 3
        
        enabled_workflows = self.scheduler.list_workflows(enabled_only=True)
        assert len(enabled_workflows) == 2
        assert all(wf.enabled for wf in enabled_workflows)


class TestWorkflowManager:
    """Test workflow manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_path = Path(temp_dir)
            self.notebook = ResearchNotebook(vault_path=self.temp_path)
            self.manager = WorkflowManager(self.notebook)
    
    def test_manager_initialization(self):
        """Test workflow manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            notebook = ResearchNotebook(vault_path=temp_path)
            manager = WorkflowManager(notebook)
            
            assert manager.notebook is notebook
            assert isinstance(manager.workflows, dict)
    
    def test_register_default_workflows(self):
        """Test registering default workflows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            notebook = ResearchNotebook(vault_path=temp_path)
            manager = WorkflowManager(notebook)
            
            initial_count = len(manager.workflows)
            manager.register_default_workflows()
            
            # Should have registered some default workflows
            assert len(manager.workflows) > initial_count


class MockWorkflow(BaseWorkflow):
    """Mock workflow for testing."""
    
    def __init__(self, name: str, should_fail: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.should_fail = should_fail
        self.execution_count = 0
    
    async def execute(self, **kwargs):
        """Mock execute method."""
        self.execution_count += 1
        
        if self.should_fail:
            raise Exception(f"Mock workflow {self.name} failed")
        
        return {"status": "success", "message": f"Mock workflow {self.name} executed"}


class TestWorkflowExecution:
    """Test workflow execution scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = WorkflowScheduler()
    
    @pytest.mark.asyncio
    async def test_successful_workflow_execution(self):
        """Test successful workflow execution."""
        workflow = MockWorkflow(name="success-workflow")
        
        # Execute directly
        result = await workflow.execute()
        
        assert result["status"] == "success"
        assert workflow.execution_count == 1
        assert workflow.last_run is not None
    
    @pytest.mark.asyncio
    async def test_failed_workflow_execution(self):
        """Test failed workflow execution."""
        workflow = MockWorkflow(name="fail-workflow", should_fail=True)
        
        # Should handle failure gracefully
        with pytest.raises(Exception):
            await workflow.execute()
        
        assert workflow.execution_count == 1
    
    def test_workflow_priority_sorting(self):
        """Test workflow priority-based sorting."""
        workflows = [
            MockWorkflow(name="low", priority=5),
            MockWorkflow(name="high", priority=1),
            MockWorkflow(name="medium", priority=3),
        ]
        
        for wf in workflows:
            self.scheduler.add_workflow(wf)
        
        # Should be sorted by priority (lowest first)
        assert self.scheduler.workflows[0].name == "high"    # Priority 1
        assert self.scheduler.workflows[1].name == "medium"  # Priority 3  
        assert self.scheduler.workflows[2].name == "low"     # Priority 5
    
    def test_workflow_enabling_disabling(self):
        """Test enabling and disabling workflows."""
        workflow = MockWorkflow(name="toggle-workflow")
        self.scheduler.add_workflow(workflow)
        
        # Initially enabled
        assert workflow.enabled == True
        assert workflow.should_run() == True
        
        # Disable
        workflow.enabled = False
        assert workflow.should_run() == False
        
        # Re-enable
        workflow.enabled = True
        assert workflow.should_run() == True


class TestWorkflowIntegration:
    """Test workflow integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_path = Path(temp_dir)
            self.notebook = ResearchNotebook(vault_path=self.temp_path)
    
    def test_workflow_with_notebook(self):
        """Test workflow integration with notebook."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            notebook = ResearchNotebook(vault_path=temp_path)
            manager = WorkflowManager(notebook)
            
            # Test that manager has reference to notebook
            assert manager.notebook is notebook
            
            # Test workflow registration
            workflow = MockWorkflow(name="integration-test")
            manager.register_workflow(workflow)
            
            assert "integration-test" in manager.workflows
            assert manager.workflows["integration-test"] is workflow
    
    @pytest.mark.asyncio
    async def test_multiple_workflow_execution(self):
        """Test executing multiple workflows in sequence."""
        workflows = [
            MockWorkflow(name="first", priority=1),
            MockWorkflow(name="second", priority=2),
            MockWorkflow(name="third", priority=3),
        ]
        
        # Execute in priority order
        results = []
        for workflow in sorted(workflows, key=lambda x: x.priority):
            if workflow.should_run():
                result = await workflow.execute()
                results.append(result)
        
        assert len(results) == 3
        for result in results:
            assert result["status"] == "success"
    
    def test_workflow_state_tracking(self):
        """Test workflow state and execution tracking."""
        workflow = MockWorkflow(name="state-tracker")
        
        # Initial state
        assert workflow.last_run is None
        assert workflow.execution_count == 0
        
        # After execution (simulated)
        workflow.execution_count = 1
        workflow.last_run = datetime.now()
        
        info = workflow.get_info()
        assert info["last_run"] is not None
        
        # Test should_run logic (could be extended for scheduling)
        assert workflow.should_run() == True  # Still enabled