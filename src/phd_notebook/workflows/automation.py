"""
Automation workflows for the PhD notebook system.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional
from pathlib import Path

from .base_workflow import BaseWorkflow
from ..agents.smart_agent import SmartAgent, LiteratureAgent
from ..core.note import Note, NoteType


class AutoTaggingWorkflow(BaseWorkflow):
    """Automatically tag notes based on content."""
    
    def __init__(self, notebook=None):
        super().__init__(
            name="auto_tagging",
            description="Automatically generate and apply tags to notes"
        )
        self.notebook = notebook
        self.smart_agent = SmartAgent("auto_tagger")
        
    async def execute(self, note: Note = None, **kwargs) -> Dict[str, Any]:
        """Execute auto-tagging workflow."""
        if not note:
            # Process all untagged notes
            notes = self.notebook.list_notes() if self.notebook else []
            notes = [n for n in notes if len(n.frontmatter.tags) < 2]  # Minimally tagged
        else:
            notes = [note]
        
        results = {"processed": 0, "tagged": 0, "errors": 0}
        
        for note in notes:
            try:
                suggested_tags = await self.smart_agent._generate_tags(note.content)
                
                # Add new tags (avoid duplicates)
                existing_tags = set(note.frontmatter.tags)
                new_tags = [tag for tag in suggested_tags if tag not in existing_tags]
                
                if new_tags:
                    note.frontmatter.tags.extend(new_tags[:3])  # Add up to 3 new tags
                    if note.file_path:
                        note.save()
                    results["tagged"] += 1
                
                results["processed"] += 1
                
            except Exception as e:
                self.log(f"Error tagging note {note.title}: {e}")
                results["errors"] += 1
        
        return results


class SmartLinkingWorkflow(BaseWorkflow):
    """Create intelligent links between related notes."""
    
    def __init__(self, notebook=None):
        super().__init__(
            name="smart_linking",
            description="Create intelligent links between related notes"
        )
        self.notebook = notebook
        self.smart_agent = SmartAgent("smart_linker")
        
    async def execute(self, note: Note = None, **kwargs) -> Dict[str, Any]:
        """Execute smart linking workflow."""
        if not self.notebook:
            return {"error": "No notebook available"}
        
        if note:
            notes_to_process = [note]
        else:
            # Process recently created/modified notes
            notes_to_process = self.notebook.list_notes()[:10]  # Last 10 notes
        
        results = {"processed": 0, "links_created": 0, "errors": 0}
        
        for current_note in notes_to_process:
            try:
                suggestions = await self.smart_agent._suggest_links(current_note)
                
                for suggestion in suggestions[:2]:  # Add top 2 suggestions
                    if suggestion["similarity"] > 0.3:  # Minimum similarity threshold
                        # Add link to note content
                        link_text = f"\n\n**Related:** [[{suggestion['title']}]] - {suggestion['reason']}"
                        if link_text not in current_note.content:
                            current_note.content += link_text
                            results["links_created"] += 1
                
                if note.file_path:
                    current_note.save()
                results["processed"] += 1
                
            except Exception as e:
                self.log(f"Error linking note {current_note.title}: {e}")
                results["errors"] += 1
        
        return results


class DailyReviewWorkflow(BaseWorkflow):
    """Daily review and organization workflow."""
    
    def __init__(self, notebook=None):
        super().__init__(
            name="daily_review",
            description="Daily review and organization of notes"
        )
        self.notebook = notebook
        self.auto_tagger = AutoTaggingWorkflow(notebook)
        self.smart_linker = SmartLinkingWorkflow(notebook)
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute daily review workflow."""
        if not self.notebook:
            return {"error": "No notebook available"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "steps_completed": [],
            "summary": {}
        }
        
        try:
            # Step 1: Auto-tag recent notes
            self.log("Starting auto-tagging...")
            tag_results = await self.auto_tagger.execute()
            results["summary"]["tagging"] = tag_results
            results["steps_completed"].append("auto_tagging")
            
            # Step 2: Create smart links
            self.log("Creating smart links...")
            link_results = await self.smart_linker.execute()
            results["summary"]["linking"] = link_results
            results["steps_completed"].append("smart_linking")
            
            # Step 3: Generate daily summary
            self.log("Generating daily summary...")
            today_notes = self._get_todays_notes()
            if today_notes:
                summary_note = await self._create_daily_summary(today_notes)
                results["summary"]["daily_note"] = {
                    "created": summary_note.title if summary_note else False,
                    "notes_reviewed": len(today_notes)
                }
                results["steps_completed"].append("daily_summary")
            
            self.log(f"Daily review completed: {len(results['steps_completed'])} steps")
            
        except Exception as e:
            self.log(f"Daily review error: {e}")
            results["error"] = str(e)
        
        return results
    
    def _get_todays_notes(self) -> List[Note]:
        """Get notes created or modified today."""
        today = datetime.now().date()
        notes = self.notebook.list_notes() if self.notebook else []
        
        todays_notes = []
        for note in notes:
            # Check creation or modification date
            if (note.frontmatter.created.date() == today or 
                note.frontmatter.updated.date() == today):
                todays_notes.append(note)
        
        return todays_notes
    
    async def _create_daily_summary(self, notes: List[Note]) -> Optional[Note]:
        """Create a daily summary note."""
        if not notes or not self.notebook:
            return None
        
        try:
            # Generate summary content
            content_parts = []
            content_parts.append(f"# Daily Review - {datetime.now().strftime('%Y-%m-%d')}")
            content_parts.append(f"\n## Notes Created/Modified Today ({len(notes)})")
            
            for note in notes:
                content_parts.append(f"- [[{note.title}]] ({note.note_type.value})")
            
            # Add insights if available
            if len(notes) > 0:
                combined_content = "\n".join([n.content[:200] for n in notes])
                smart_agent = SmartAgent("daily_summarizer")
                insights = await smart_agent._analyze(combined_content, "research")
                
                content_parts.append("\n## Key Insights")
                content_parts.append(insights.get("analysis", "No insights generated"))
            
            # Create summary note
            summary_content = "\n".join(content_parts)
            summary_note = self.notebook.create_note(
                title=f"Daily Review {datetime.now().strftime('%Y-%m-%d')}",
                content=summary_content,
                note_type=NoteType.DAILY,
                tags=["#daily", "#review", "#auto_generated"]
            )
            
            return summary_note
            
        except Exception as e:
            self.log(f"Failed to create daily summary: {e}")
            return None


class LiteratureProcessingWorkflow(BaseWorkflow):
    """Process new literature and papers."""
    
    def __init__(self, notebook=None):
        super().__init__(
            name="literature_processing",
            description="Process and analyze new literature"
        )
        self.notebook = notebook
        self.lit_agent = LiteratureAgent()
        
    async def execute(self, paper_content: str = None, metadata: Dict = None, **kwargs) -> Dict[str, Any]:
        """Process literature content."""
        if not paper_content or not self.notebook:
            return {"error": "No content or notebook available"}
        
        try:
            # Process paper with literature agent
            analysis = await self.lit_agent.process_paper(paper_content, metadata)
            
            # Create literature note
            title = metadata.get("title", f"Paper {datetime.now().strftime('%Y%m%d_%H%M')}")
            
            content_parts = [
                f"# {title}",
                f"\n**Authors:** {metadata.get('authors', 'Unknown')}",
                f"**DOI:** {metadata.get('doi', 'Unknown')}",
                f"\n## Key Contributions",
                analysis.get('contributions', 'Not extracted'),
                f"\n## Methodology", 
                analysis.get('methodology', 'Not extracted'),
                f"\n## Research Questions",
                analysis.get('research_questions', 'Not extracted'),
                f"\n## Notes",
                "<!-- Add your notes and insights here -->"
            ]
            
            literature_note = self.notebook.create_note(
                title=title,
                content="\n".join(content_parts),
                note_type=NoteType.LITERATURE,
                tags=["#literature", "#paper", "#auto_processed"]
            )
            
            self.log(f"Processed literature: {title}")
            
            return {
                "note_created": literature_note.title,
                "analysis": analysis,
                "success": True
            }
            
        except Exception as e:
            self.log(f"Literature processing error: {e}")
            return {"error": str(e), "success": False}


class WorkflowManager:
    """Manage and coordinate multiple workflows."""
    
    def __init__(self, notebook=None):
        self.notebook = notebook
        self.workflows: Dict[str, BaseWorkflow] = {}
        self.scheduled_workflows: Dict[str, Dict] = {}
        
        # Register default workflows
        self.register_default_workflows()
    
    def register_default_workflows(self):
        """Register default workflows."""
        workflows = [
            AutoTaggingWorkflow(self.notebook),
            SmartLinkingWorkflow(self.notebook), 
            DailyReviewWorkflow(self.notebook),
            LiteratureProcessingWorkflow(self.notebook)
        ]
        
        for workflow in workflows:
            self.register_workflow(workflow)
    
    def register_workflow(self, workflow: BaseWorkflow):
        """Register a workflow."""
        self.workflows[workflow.name] = workflow
        print(f"📋 Registered workflow: {workflow.name}")
    
    async def run_workflow(self, name: str, **kwargs) -> Dict[str, Any]:
        """Run a specific workflow."""
        if name not in self.workflows:
            return {"error": f"Workflow '{name}' not found"}
        
        workflow = self.workflows[name]
        print(f"▶️ Running workflow: {name}")
        
        try:
            result = await workflow.execute(**kwargs)
            print(f"✅ Workflow '{name}' completed")
            return result
        except Exception as e:
            print(f"❌ Workflow '{name}' failed: {e}")
            return {"error": str(e)}
    
    def schedule_workflow(self, name: str, frequency: str, **schedule_config):
        """Schedule a workflow to run automatically."""
        if name not in self.workflows:
            print(f"❌ Cannot schedule unknown workflow: {name}")
            return
        
        self.scheduled_workflows[name] = {
            "frequency": frequency,
            "config": schedule_config,
            "last_run": None
        }
        
        print(f"⏰ Scheduled workflow '{name}' to run {frequency}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get status of all workflows."""
        return {
            "registered": list(self.workflows.keys()),
            "scheduled": list(self.scheduled_workflows.keys()),
            "total_workflows": len(self.workflows)
        }