"""
Daily automation workflows for research organization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_workflow import BaseWorkflow
from ..core.note import Note, NoteType
from ..utils.error_handling import handle_async_errors


class DailyOrganizer(BaseWorkflow):
    """Daily workflow for organizing and processing research content."""
    
    def __init__(self, notebook=None, **config):
        super().__init__("daily_organizer", notebook, **config)
        self.max_notes_per_run = config.get("max_notes_per_run", 50)
        
    async def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute daily organization tasks."""
        tasks_completed = []
        stats = {
            "notes_processed": 0,
            "tags_added": 0,
            "links_created": 0,
            "orphaned_notes": 0
        }
        
        if not self.notebook:
            return {"error": "No notebook instance available"}
        
        # 1. Process untagged notes
        untagged_result = await self._process_untagged_notes()
        tasks_completed.append("process_untagged_notes")
        stats["notes_processed"] += untagged_result.get("processed", 0)
        stats["tags_added"] += untagged_result.get("tags_added", 0)
        
        # 2. Find and create automatic links
        linking_result = await self._create_automatic_links()
        tasks_completed.append("create_automatic_links")
        stats["links_created"] += linking_result.get("links_created", 0)
        
        # 3. Identify orphaned notes
        orphan_result = await self._identify_orphaned_notes()
        tasks_completed.append("identify_orphaned_notes")
        stats["orphaned_notes"] = orphan_result.get("orphaned_count", 0)
        
        # 4. Update daily note with summary
        daily_note_result = await self._update_daily_note(stats)
        tasks_completed.append("update_daily_note")
        
        return {
            "tasks_completed": tasks_completed,
            "stats": stats,
            "daily_note_created": daily_note_result.get("created", False)
        }
    
    @handle_async_errors(return_value={"processed": 0, "tags_added": 0})
    async def _process_untagged_notes(self) -> Dict[str, Any]:
        """Process notes that don't have tags."""
        notes = self.notebook.list_notes()
        processed = 0
        tags_added = 0
        
        for note in notes:
            if processed >= self.max_notes_per_run:
                break
                
            if not note.frontmatter.tags:
                # Auto-suggest tags based on content and type
                suggested_tags = self._suggest_tags_for_note(note)
                
                if suggested_tags:
                    for tag in suggested_tags:
                        note.add_tag(tag)
                        tags_added += 1
                    
                    if note.file_path:
                        note.save()
                    
                    processed += 1
                    self.logger.info(f"Added {len(suggested_tags)} tags to note: {note.title}")
        
        return {"processed": processed, "tags_added": tags_added}
    
    @handle_async_errors(return_value={"links_created": 0})
    async def _create_automatic_links(self) -> Dict[str, Any]:
        """Create automatic links between related notes."""
        notes = self.notebook.list_notes()
        links_created = 0
        
        # Simple algorithm: link notes with common tags or keywords
        for i, note1 in enumerate(notes):
            if i >= self.max_notes_per_run:
                break
                
            for j, note2 in enumerate(notes[i+1:], i+1):
                if self._notes_should_be_linked(note1, note2):
                    # Check if link already exists
                    existing_links = [link.target for link in note1.get_links()]
                    
                    if note2.title not in existing_links:
                        note1.add_link(note2.title, "related_to", 0.7)
                        
                        if note1.file_path:
                            note1.save()
                        
                        links_created += 1
                        self.logger.info(f"Linked {note1.title} -> {note2.title}")
                        
                        # Limit links per note
                        if len(note1.get_links()) >= 5:
                            break
        
        return {"links_created": links_created}
    
    @handle_async_errors(return_value={"orphaned_count": 0})
    async def _identify_orphaned_notes(self) -> Dict[str, Any]:
        """Identify notes with no links or references."""
        notes = self.notebook.list_notes()
        orphaned_notes = []
        
        for note in notes:
            # Check if note has any links (in or out)
            incoming_links = self._count_incoming_links(note, notes)
            outgoing_links = len(note.get_links())
            
            if incoming_links == 0 and outgoing_links == 0:
                orphaned_notes.append(note.title)
        
        if orphaned_notes:
            self.logger.warning(f"Found {len(orphaned_notes)} orphaned notes")
            
            # Create a note tracking orphaned notes
            orphan_note = self.notebook.create_note(
                title=f"Orphaned Notes - {datetime.now().strftime('%Y-%m-%d')}",
                content=self._build_orphan_report(orphaned_notes),
                note_type=NoteType.PROJECT,
                tags=["#maintenance", "#orphaned", "#daily-report"]
            )
        
        return {"orphaned_count": len(orphaned_notes), "orphaned_notes": orphaned_notes}
    
    @handle_async_errors(return_value={"created": False})
    async def _update_daily_note(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update today's daily note with organization stats."""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_note_title = f"Daily Notes - {today}"
        
        # Check if daily note already exists
        existing_note = self.notebook.get_note(daily_note_title)
        
        organization_summary = f"""
## Daily Organization Summary ({today})

### Statistics
- **Notes Processed**: {stats['notes_processed']}
- **Tags Added**: {stats['tags_added']}
- **Links Created**: {stats['links_created']}
- **Orphaned Notes**: {stats['orphaned_notes']}

### Completed Tasks
- Processed untagged notes
- Created automatic links
- Identified orphaned content
- Updated daily summary

*Generated automatically by DailyOrganizer workflow*
"""
        
        if existing_note:
            # Append to existing note
            existing_note.add_section("Organization Summary", organization_summary)
            if existing_note.file_path:
                existing_note.save()
        else:
            # Create new daily note
            self.notebook.create_note(
                title=daily_note_title,
                content=f"# {daily_note_title}\n\n{organization_summary}",
                note_type=NoteType.DAILY,
                tags=["#daily", "#organization", "#automated"]
            )
        
        return {"created": True}
    
    def _suggest_tags_for_note(self, note: Note) -> List[str]:
        """Suggest tags for a note based on content and type."""
        suggested_tags = []
        
        # Type-based tags
        type_tags = {
            NoteType.LITERATURE: ["#literature", "#paper"],
            NoteType.EXPERIMENT: ["#experiment", "#research"],
            NoteType.IDEA: ["#idea", "#brainstorm"],
            NoteType.PROJECT: ["#project", "#work"],
            NoteType.MEETING: ["#meeting", "#collaboration"]
        }
        
        suggested_tags.extend(type_tags.get(note.note_type, []))
        
        # Content-based tags
        content_lower = note.content.lower()
        
        content_keywords = {
            "#machine-learning": ["machine learning", "ml", "neural", "model", "algorithm"],
            "#data-analysis": ["data", "analysis", "statistics", "visualization", "dataset"],
            "#programming": ["code", "programming", "python", "javascript", "software"],
            "#methodology": ["methodology", "method", "approach", "procedure", "protocol"],
            "#results": ["results", "findings", "outcome", "conclusion", "evaluation"],
            "#todo": ["todo", "task", "action", "follow-up", "next steps"]
        }
        
        for tag, keywords in content_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                suggested_tags.append(tag)
        
        return suggested_tags[:5]  # Limit to 5 suggestions
    
    def _notes_should_be_linked(self, note1: Note, note2: Note) -> bool:
        """Determine if two notes should be linked."""
        # Don't link same types of notes too aggressively
        if note1.note_type == note2.note_type == NoteType.DAILY:
            return False
        
        # Check for common tags (at least 2)
        common_tags = set(note1.frontmatter.tags).intersection(set(note2.frontmatter.tags))
        if len(common_tags) >= 2:
            return True
        
        # Check for keyword overlap in titles
        title1_words = set(note1.title.lower().split())
        title2_words = set(note2.title.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        title1_words -= common_words
        title2_words -= common_words
        
        if len(title1_words.intersection(title2_words)) >= 2:
            return True
        
        return False
    
    def _count_incoming_links(self, target_note: Note, all_notes: List[Note]) -> int:
        """Count how many notes link to the target note."""
        count = 0
        target_title = target_note.title
        
        for note in all_notes:
            if note.title == target_title:
                continue
            
            # Check if note content mentions target note
            if f"[[{target_title}]]" in note.content:
                count += 1
            
            # Check explicit links
            for link in note.get_links():
                if link.target == target_title:
                    count += 1
                    break
        
        return count
    
    def _build_orphan_report(self, orphaned_notes: List[str]) -> str:
        """Build a report of orphaned notes."""
        report = f"""# Orphaned Notes Report - {datetime.now().strftime('%Y-%m-%d')}

## Summary
Found **{len(orphaned_notes)}** notes with no incoming or outgoing links.

## Orphaned Notes
"""
        
        for note_title in orphaned_notes:
            report += f"- [[{note_title}]]\n"
        
        report += f"""
## Recommendations
1. Review each orphaned note for relevance
2. Add links to related notes
3. Consider adding tags for better discoverability
4. Archive or delete notes that are no longer needed

*This report was generated automatically. Consider reviewing these notes during your next research session.*
"""
        
        return report


class DailyNoteProcessor(BaseWorkflow):
    """Process and enhance daily notes."""
    
    def __init__(self, notebook=None, **config):
        super().__init__("daily_note_processor", notebook, **config)
        
    async def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process today's daily note."""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_note_title = f"Daily Notes - {today}"
        
        # Get or create today's daily note
        daily_note = self.notebook.get_note(daily_note_title)
        
        if not daily_note:
            daily_note = self._create_daily_note_template(today)
        
        # Process the daily note
        processing_result = await self._process_daily_note(daily_note)
        
        return {
            "daily_note_title": daily_note_title,
            "note_exists": daily_note is not None,
            "processing_result": processing_result
        }
    
    def _create_daily_note_template(self, date_str: str) -> Note:
        """Create a daily note template."""
        template_content = f"""# Daily Notes - {date_str}

## Goals for Today
- [ ] 

## Research Progress
### Literature Review
- 

### Experiments
- 

### Writing
- 

## Ideas and Thoughts
- 

## Tasks Completed
- [ ] 

## Tomorrow's Plan
- [ ] 

## Reflection
*What went well today? What could be improved?*

---
*Daily note created automatically*
"""
        
        return self.notebook.create_note(
            title=f"Daily Notes - {date_str}",
            content=template_content,
            note_type=NoteType.DAILY,
            tags=["#daily", "#template", "#planning"]
        )
    
    async def _process_daily_note(self, daily_note: Note) -> Dict[str, Any]:
        """Process and enhance a daily note."""
        stats = {
            "tasks_found": 0,
            "completed_tasks": 0,
            "links_added": 0
        }
        
        content = daily_note.content
        
        # Count tasks
        total_tasks = content.count("- [ ]") + content.count("- [x]")
        completed_tasks = content.count("- [x]")
        
        stats["tasks_found"] = total_tasks
        stats["completed_tasks"] = completed_tasks
        
        # Add progress summary if tasks exist
        if total_tasks > 0:
            progress_percentage = (completed_tasks / total_tasks) * 100
            progress_section = f"""
## Daily Progress Summary
- **Total Tasks**: {total_tasks}
- **Completed**: {completed_tasks}
- **Progress**: {progress_percentage:.1f}%

"""
            
            if progress_section not in daily_note.content:
                daily_note.content += progress_section
                if daily_note.file_path:
                    daily_note.save()
        
        return stats