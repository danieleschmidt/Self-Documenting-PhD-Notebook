"""
Vault Manager for handling Obsidian vault operations.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Iterator
from dataclasses import dataclass

from ..core.note import Note, NoteType


@dataclass
class VaultConfig:
    """Configuration for an Obsidian vault."""
    name: str
    path: Path
    author: str
    institution: str = ""
    field: str = ""
    auto_sync: bool = True
    backup_enabled: bool = True
    templates_path: str = "templates"
    attachments_path: str = "attachments"


class VaultManager:
    """
    Manages an Obsidian vault for research notes.
    
    Handles:
    - Vault initialization and configuration
    - Note creation, reading, updating, deletion
    - Template management
    - File organization
    - Search and indexing
    """
    
    def __init__(self, vault_path: Union[str, Path], config: Optional[VaultConfig] = None):
        self.vault_path = Path(vault_path)
        
        if config:
            self.config = config
        else:
            self.config = VaultConfig(
                name=self.vault_path.name,
                path=self.vault_path,
                author="Researcher"
            )
        
        self._notes_cache: Dict[str, Note] = {}
        self._last_scan: Optional[datetime] = None
        
        # Initialize vault if it doesn't exist
        if not self.vault_path.exists():
            self.initialize_vault()
    
    def initialize_vault(self) -> None:
        """Initialize a new Obsidian vault with research structure."""
        print(f"Initializing research vault at {self.vault_path}")
        
        # Create main directories
        directories = [
            "daily",
            "projects", 
            "experiments",
            "literature",
            "ideas",
            "meetings",
            "templates",
            "attachments",
            "archive",
            ".obsidian",
        ]
        
        for directory in directories:
            (self.vault_path / directory).mkdir(parents=True, exist_ok=True)
        
        # Create Obsidian configuration
        self._create_obsidian_config()
        
        # Create default templates
        self._create_default_templates()
        
        # Create welcome note
        self._create_welcome_note()
        
        print("✅ Vault initialized successfully!")
    
    def _create_obsidian_config(self) -> None:
        """Create Obsidian-specific configuration files."""
        obsidian_dir = self.vault_path / ".obsidian"
        
        # Main app config
        app_config = {
            "legacyEditor": False,
            "livePreview": True,
            "showLineNumber": True,
            "spellcheck": True,
            "useMarkdownLinks": True,
            "newLinkFormat": "relative",
            "attachmentFolderPath": "attachments",
        }
        
        with open(obsidian_dir / "app.json", "w") as f:
            json.dump(app_config, f, indent=2)
        
        # Core plugins
        core_plugins = [
            "file-explorer",
            "search", 
            "quick-switcher",
            "graph",
            "backlink",
            "tag-pane",
            "page-preview",
            "templates",
            "note-composer",
            "command-palette",
            "markdown-importer",
            "outline",
            "word-count",
        ]
        
        with open(obsidian_dir / "core-plugins.json", "w") as f:
            json.dump(core_plugins, f, indent=2)
        
        # Workspace layout
        workspace = {
            "main": {
                "id": "main",
                "type": "split",
                "children": [
                    {
                        "id": "sidebar-left",
                        "type": "split",
                        "children": [
                            {
                                "id": "file-explorer",
                                "type": "leaf",
                                "state": {"type": "file-explorer"}
                            }
                        ]
                    },
                    {
                        "id": "editor",
                        "type": "leaf", 
                        "state": {"type": "empty"}
                    },
                    {
                        "id": "sidebar-right",
                        "type": "split",
                        "children": [
                            {
                                "id": "backlink",
                                "type": "leaf",
                                "state": {"type": "backlink"}
                            }
                        ]
                    }
                ]
            }
        }
        
        with open(obsidian_dir / "workspace.json", "w") as f:
            json.dump(workspace, f, indent=2)
    
    def _create_default_templates(self) -> None:
        """Create default note templates."""
        templates_dir = self.vault_path / "templates"
        
        # Experiment template
        experiment_template = """---
title: "{{title}}"
created: {{date}}
updated: {{date}}
tags: [#experiment, #{{project}}]
type: experiment
status: planning
priority: 3
experiment_id: "{{date:YYYY-MM-DD}}-{{title}}"
hypothesis: ""
methodology: ""
results: ""
---

# {{title}}

## Hypothesis
*What do you expect to find?*

## Methodology
*How will you test your hypothesis?*

### Materials
- 

### Procedure
1. 

### Data Collection
- 

## Results
*What did you observe?*

## Analysis
*What do the results mean?*

## Conclusions
*What did you learn?*

## Next Steps
- 

## Related Work
- 

"""
        
        (templates_dir / "experiment.md").write_text(experiment_template)
        
        # Literature review template
        literature_template = """---
title: "{{title}}"
created: {{date}}
updated: {{date}}
tags: [#literature, #paper]
type: literature
status: unread
priority: 3
authors: ""
year: ""
journal: ""
doi: ""
---

# {{title}}

## Citation
> Author, A. (Year). Title. *Journal*, Volume(Issue), pages. DOI

## Key Contributions
- 

## Methodology
*How did they approach the problem?*

## Results
*What did they find?*

## Critical Analysis
*Strengths and weaknesses*

### Strengths
- 

### Weaknesses
- 

## Relevance to My Research
*How does this relate to your work?*

## Follow-up Actions
- [ ] 

## Related Papers
- 

"""
        
        (templates_dir / "literature.md").write_text(literature_template)
        
        # Meeting template
        meeting_template = """---
title: "{{title}}"
created: {{date}}
updated: {{date}}
tags: [#meeting]
type: meeting
status: completed
priority: 3
attendees: []
---

# {{title}}

**Date**: {{date}}
**Attendees**: 

## Agenda
1. 

## Discussion Points

### Key Points
- 

### Decisions Made
- 

### Action Items
- [ ] 

## Next Meeting
**Date**: 
**Agenda**: 

"""
        
        (templates_dir / "meeting.md").write_text(meeting_template)
    
    def _create_welcome_note(self) -> None:
        """Create a welcome note with vault overview."""
        welcome_content = f"""---
title: "Welcome to Your Research Vault"
created: {datetime.now().isoformat()}
updated: {datetime.now().isoformat()}
tags: [#index, #welcome]
type: project
status: active
priority: 5
---

# Welcome to Your Research Vault 🎓

This is your self-documenting PhD notebook! Here's how to get started:

## Quick Start
1. Create daily notes in the `daily/` folder
2. Document experiments in `experiments/`
3. Save literature reviews in `literature/`
4. Track project progress in `projects/`

## Folder Structure
- **daily/**: Daily research notes and logs
- **projects/**: Major research projects
- **experiments/**: Experimental procedures and results
- **literature/**: Paper reviews and summaries
- **ideas/**: Research ideas and hypotheses
- **meetings/**: Meeting notes and discussions
- **templates/**: Note templates for different types
- **attachments/**: Images, PDFs, and other files

## Templates Available
- [[templates/experiment]]: For documenting experiments
- [[templates/literature]]: For paper reviews
- [[templates/meeting]]: For meeting notes

## Tips
- Use `[[double brackets]]` to link between notes
- Add `#tags` to categorize your notes
- Use the graph view to visualize connections
- Daily notes help track progress over time

## Getting Help
- Press `Ctrl+P` to open command palette
- Use `Ctrl+O` to quickly open any note
- Press `Ctrl+G` to open graph view

Happy researching! 🔬✨
"""
        
        welcome_note = Note(
            title="Welcome to Your Research Vault",
            content=welcome_content,
            note_type=NoteType.PROJECT
        )
        
        welcome_note.save(self.vault_path / "Welcome.md")
    
    def create_note(
        self,
        title: str,
        content: str = "",
        note_type: NoteType = NoteType.IDEA,
        tags: List[str] = None,
        folder: str = "",
        template: str = None
    ) -> Note:
        """Create a new note in the vault."""
        
        # Apply template if specified
        if template:
            template_content = self._load_template(template)
            if template_content:
                content = template_content + "\n\n" + content
        
        # Create note object
        note = Note(
            title=title,
            content=content,
            note_type=note_type
        )
        
        # Add tags
        if tags:
            note.add_tags(tags)
        
        # Determine file path
        if folder:
            folder_path = self.vault_path / folder
        else:
            folder_path = self._get_default_folder(note_type)
        
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        safe_title = self._sanitize_filename(title)
        file_path = folder_path / f"{safe_title}.md"
        
        # Handle duplicate names
        counter = 1
        while file_path.exists():
            file_path = folder_path / f"{safe_title}_{counter}.md"
            counter += 1
        
        # Save note
        note.save(file_path)
        
        # Cache the note
        self._notes_cache[title] = note
        
        return note
    
    def get_note(self, title: str) -> Optional[Note]:
        """Get a note by title."""
        # Check cache first
        if title in self._notes_cache:
            return self._notes_cache[title]
        
        # Search for file
        note_file = self._find_note_file(title)
        if note_file:
            note = Note.from_file(note_file)
            self._notes_cache[title] = note
            return note
        
        return None
    
    def list_notes(self, note_type: Optional[NoteType] = None, tags: List[str] = None) -> List[Note]:
        """List all notes, optionally filtered by type or tags."""
        self._refresh_cache()
        
        notes = list(self._notes_cache.values())
        
        # Filter by type
        if note_type:
            notes = [n for n in notes if n.note_type == note_type]
        
        # Filter by tags
        if tags:
            tag_set = set(f"#{tag}" if not tag.startswith('#') else tag for tag in tags)
            notes = [n for n in notes if tag_set.intersection(set(n.frontmatter.tags))]
        
        return notes
    
    def search_notes(self, query: str, in_content: bool = True) -> List[Note]:
        """Search notes by title and optionally content."""
        self._refresh_cache()
        
        results = []
        query_lower = query.lower()
        
        for note in self._notes_cache.values():
            # Search in title
            if query_lower in note.title.lower():
                results.append(note)
                continue
            
            # Search in content if requested
            if in_content and query_lower in note.content.lower():
                results.append(note)
                continue
            
            # Search in tags
            tag_text = " ".join(note.frontmatter.tags).lower()
            if query_lower in tag_text:
                results.append(note)
        
        return results
    
    def delete_note(self, title: str) -> bool:
        """Delete a note from the vault."""
        note = self.get_note(title)
        if not note or not note.file_path:
            return False
        
        # Move to archive instead of deleting
        archive_dir = self.vault_path / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        archive_path = archive_dir / note.file_path.name
        shutil.move(str(note.file_path), str(archive_path))
        
        # Remove from cache
        if title in self._notes_cache:
            del self._notes_cache[title]
        
        return True
    
    def _load_template(self, template_name: str) -> Optional[str]:
        """Load a note template."""
        template_path = self.vault_path / "templates" / f"{template_name}.md"
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")
        return None
    
    def _get_default_folder(self, note_type: NoteType) -> Path:
        """Get the default folder for a note type."""
        folder_map = {
            NoteType.EXPERIMENT: "experiments",
            NoteType.LITERATURE: "literature", 
            NoteType.IDEA: "ideas",
            NoteType.MEETING: "meetings",
            NoteType.PROJECT: "projects",
            NoteType.ANALYSIS: "experiments",
            NoteType.METHODOLOGY: "experiments",
            NoteType.OBSERVATION: "daily",
        }
        
        folder = folder_map.get(note_type, "ideas")
        return self.vault_path / folder
    
    def _sanitize_filename(self, title: str) -> str:
        """Create a safe filename from a title."""
        # Remove or replace problematic characters
        safe_title = title.replace("/", "-").replace("\\", "-")
        safe_title = re.sub(r'[<>:"|?*]', '', safe_title)
        safe_title = safe_title.strip()
        
        # Limit length
        if len(safe_title) > 100:
            safe_title = safe_title[:100]
        
        return safe_title
    
    def _find_note_file(self, title: str) -> Optional[Path]:
        """Find a note file by title."""
        safe_title = self._sanitize_filename(title)
        
        # Look in all directories
        for md_file in self.vault_path.rglob("*.md"):
            if md_file.stem == safe_title or md_file.stem.startswith(f"{safe_title}_"):
                return md_file
        
        return None
    
    def _refresh_cache(self) -> None:
        """Refresh the notes cache from filesystem."""
        current_time = datetime.now()
        
        # Only refresh if it's been more than 30 seconds
        if (self._last_scan and 
            (current_time - self._last_scan).seconds < 30):
            return
        
        self._notes_cache.clear()
        
        # Scan all markdown files
        for md_file in self.vault_path.rglob("*.md"):
            # Skip template files
            if "templates" in md_file.parts:
                continue
                
            try:
                note = Note.from_file(md_file)
                self._notes_cache[note.title] = note
            except Exception as e:
                print(f"Warning: Could not load note {md_file}: {e}")
        
        self._last_scan = current_time
    
    def get_vault_stats(self) -> Dict:
        """Get statistics about the vault."""
        self._refresh_cache()
        
        notes = list(self._notes_cache.values())
        total_notes = len(notes)
        
        # Count by type
        type_counts = {}
        for note_type in NoteType:
            count = sum(1 for n in notes if n.note_type == note_type)
            if count > 0:
                type_counts[note_type.value] = count
        
        # Count by status
        status_counts = {}
        for note in notes:
            status = getattr(note.frontmatter, 'status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get all tags
        all_tags = set()
        for note in notes:
            all_tags.update(note.frontmatter.tags)
        
        return {
            "total_notes": total_notes,
            "types": type_counts,
            "statuses": status_counts,
            "total_tags": len(all_tags),
            "vault_path": str(self.vault_path),
            "last_updated": self._last_scan.isoformat() if self._last_scan else None,
        }