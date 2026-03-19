"""
Base workflow system for automation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
# import schedule  # Optional dependency - simplified for now
import threading

from ..utils.error_handling import handle_async_errors, error_handler
from ..utils.exceptions import WorkflowError


class BaseWorkflow:
    """Base class for automation workflows."""

    def __init__(self, name: str, notebook=None, description: str = "", enabled: bool = True, priority: int = 1, **config):
        self.name = name
        self.notebook = notebook
        self.config = config
        self.description = description
        self.enabled = enabled
        self.priority = priority
        self.is_active = enabled
        self.last_run = None
        self.run_count = 0
        self.execution_count = 0
        self.logger = logging.getLogger(f"Workflow.{name}")

    async def execute(self, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Execute the workflow. Override in subclasses."""
        self.last_run = datetime.now()
        self.execution_count += 1
        return {"status": "success", "message": "No-op base workflow"}
    
    @handle_async_errors(severity="error")
    async def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the workflow with error handling and logging."""
        if not self.is_active:
            self.logger.warning(f"Workflow {self.name} is not active")
            return {"status": "inactive"}
        
        start_time = datetime.now()
        self.logger.info(f"Starting workflow: {self.name}")
        
        try:
            result = await self.execute(context or {})
            
            # Update run statistics
            self.last_run = start_time
            self.run_count += 1
            self.execution_count += 1
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Workflow {self.name} completed in {duration:.2f}s")
            
            return {
                "status": "success",
                "duration": duration,
                "result": result,
                "run_count": self.run_count
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Workflow {self.name} failed after {duration:.2f}s: {e}")
            
            return {
                "status": "error",
                "duration": duration,
                "error": str(e),
                "run_count": self.run_count
            }
    
    def start(self) -> None:
        """Start the workflow."""
        self.is_active = True
        self.logger.info(f"Workflow {self.name} started")
    
    def stop(self) -> None:
        """Stop the workflow."""
        self.is_active = False
        self.logger.info(f"Workflow {self.name} stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get workflow status information."""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "config": self.config
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get workflow information."""
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "priority": self.priority,
            "is_active": self.is_active,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "config": self.config
        }
    
    def should_run(self) -> bool:
        """Check if workflow should run (enabled check)."""
        return self.enabled and self.is_active


class WorkflowScheduler:
    """Scheduler for managing and running workflows."""
    
    def __init__(self):
        self.workflows: List[BaseWorkflow] = []
        self.scheduled_workflows: Dict[str, Any] = {}
        self.scheduler_thread = None
        self.running = False
        self.is_running = False  # alias
        self.logger = logging.getLogger("WorkflowScheduler")
        
    def add_workflow(self, workflow: BaseWorkflow) -> None:
        """Add a workflow to the scheduler (sorted by priority)."""
        self.workflows.append(workflow)
        self.workflows.sort(key=lambda w: w.priority)
        self.logger.info(f"Added workflow: {workflow.name}")

    def register_workflow(self, workflow: BaseWorkflow) -> None:
        """Alias for add_workflow (backwards compat)."""
        self.add_workflow(workflow)

    def remove_workflow(self, name: str) -> None:
        """Remove a workflow by name."""
        self.workflows = [w for w in self.workflows if w.name != name]
        self.scheduled_workflows.pop(name, None)
        self.logger.info(f"Removed workflow: {name}")

    def get_workflow(self, name: str) -> Optional[BaseWorkflow]:
        """Get a workflow by name."""
        for w in self.workflows:
            if w.name == name:
                return w
        return None

    def list_workflows(self, enabled_only: bool = False) -> List[BaseWorkflow]:
        """List all registered workflows, optionally filtered by enabled state."""
        if enabled_only:
            return [w for w in self.workflows if getattr(w, 'enabled', True)]
        return list(self.workflows)

    def schedule_workflow(
        self,
        workflow_name: str,
        schedule_type: str,
        **schedule_kwargs
    ) -> None:
        """Schedule a workflow to run automatically (simplified version)."""
        workflow = self.get_workflow(workflow_name)
        if workflow is None:
            raise WorkflowError(f"Workflow {workflow_name} not registered")

        # Store schedule info (but don't actually schedule with external library)
        self.scheduled_workflows[workflow_name] = {
            "schedule_type": schedule_type,
            "schedule_kwargs": schedule_kwargs,
            "job": None  # Simplified for now
        }

        workflow.start()
        self.logger.info(f"Registered workflow {workflow_name} for {schedule_type} scheduling")

    def unschedule_workflow(self, workflow_name: str) -> None:
        """Remove a workflow from the schedule."""
        if workflow_name in self.scheduled_workflows:
            del self.scheduled_workflows[workflow_name]
            wf = self.get_workflow(workflow_name)
            if wf:
                wf.stop()
            self.logger.info(f"Unscheduled workflow: {workflow_name}")

    def _run_workflow_sync(self, workflow_name: str) -> None:
        """Run workflow synchronously (for scheduler)."""
        workflow = self.get_workflow(workflow_name)
        if workflow:
            def run_async():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(workflow.run())
                    self.logger.info(f"Scheduled run of {workflow_name}: {result['status']}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_async)
            thread.start()
            thread.join(timeout=300)  # 5 minute timeout

    async def run_workflow_once(
        self,
        workflow_name: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run a workflow once manually."""
        workflow = self.get_workflow(workflow_name)
        if workflow is None:
            raise WorkflowError(f"Workflow {workflow_name} not registered")

        was_active = workflow.is_active
        
        # Temporarily activate if needed
        if not was_active:
            workflow.start()
        
        try:
            result = await workflow.run(context)
            return result
        finally:
            # Restore original state
            if not was_active:
                workflow.stop()
    
    def start_scheduler(self) -> None:
        """Start the background scheduler (simplified)."""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Workflow scheduler started (manual trigger mode)")
    
    def stop_scheduler(self) -> None:
        """Stop the background scheduler."""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Workflow scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler and workflow status."""
        return {
            "scheduler_running": self.is_running,
            "registered_workflows": len(self.workflows),
            "scheduled_workflows": len(self.scheduled_workflows),
            "workflows": {
                w.name: w.get_status()
                for w in self.workflows
            },
            "schedules": {
                name: {
                    "type": schedule_info["schedule_type"],
                    "config": schedule_info["schedule_kwargs"]
                }
                for name, schedule_info in self.scheduled_workflows.items()
            }
        }


class ConditionalWorkflow(BaseWorkflow):
    """Workflow that runs based on conditions."""
    
    def __init__(self, name: str, condition_func: Callable, **kwargs):
        super().__init__(name, **kwargs)
        self.condition_func = condition_func
    
    async def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run workflow only if condition is met."""
        if not self.condition_func(context or {}):
            self.logger.info(f"Condition not met for workflow: {self.name}")
            return {"status": "skipped", "reason": "condition_not_met"}
        
        return await super().run(context)


class ChainedWorkflow(BaseWorkflow):
    """Workflow that chains multiple other workflows."""
    
    def __init__(self, name: str, workflow_names: List[str], **kwargs):
        super().__init__(name, **kwargs)
        self.workflow_names = workflow_names
    
    async def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute chained workflows in sequence."""
        results = {}
        current_context = context or {}
        
        for workflow_name in self.workflow_names:
            if workflow_name not in self.notebook.scheduler.workflows:
                self.logger.warning(f"Workflow {workflow_name} not found, skipping")
                continue
            
            workflow = self.notebook.scheduler.workflows[workflow_name]
            result = await workflow.run(current_context)
            
            results[workflow_name] = result
            
            # Pass result as context to next workflow
            if result.get("status") == "success" and "result" in result:
                current_context.update(result["result"])
        
        return {
            "chained_results": results,
            "final_context": current_context
        }