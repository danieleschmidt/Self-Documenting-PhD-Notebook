"""Automation workflows for research tasks."""

from .base_workflow import BaseWorkflow, WorkflowScheduler
from .daily_workflows import DailyOrganizer, DailyNoteProcessor  
from .literature_workflows import LiteratureReviewWorkflow, PaperAlertWorkflow
from .experiment_workflows import ExperimentTracker, ResultsProcessor
from .writing_workflows import DraftAnalyzer, CitationValidator

__all__ = [
    'BaseWorkflow', 'WorkflowScheduler',
    'DailyOrganizer', 'DailyNoteProcessor',
    'LiteratureReviewWorkflow', 'PaperAlertWorkflow', 
    'ExperimentTracker', 'ResultsProcessor',
    'DraftAnalyzer', 'CitationValidator'
]