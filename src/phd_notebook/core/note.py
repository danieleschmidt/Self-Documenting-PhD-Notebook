"""
Core Note class for managing individual research notes.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

from ..utils.yaml_fallback import safe_load, safe_dump, BasicYAMLError as YAMLError
from ..utils.simple_validation import validate_note_data, ValidationError
from ..utils.exceptions import NoteError, InvalidNoteFormatError, NoteFrontmatterError
from ..monitoring.logging_setup import setup_logger


class NoteType(str, Enum):
    """Types of research notes."""
    EXPERIMENT = "experiment"
    LITERATURE = "literature"
    IDEA = "idea"
    MEETING = "meeting"
    PROJECT = "project"
    ANALYSIS = "analysis"
    METHODOLOGY = "methodology"
    OBSERVATION = "observation"


@dataclass(frozen=True)
class Link:
    """Represents a link between notes."""
    target: str
    relationship: str = "relates_to"
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


class NoteFrontmatter(BaseModel):
    """Structured frontmatter for research notes."""
    title: str
    created: datetime = Field(default_factory=datetime.now)
    updated: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    type: NoteType = NoteType.IDEA
    status: str = "draft"
    priority: int = 3  # 1-5, 5 being highest
    author: Optional[str] = None
    project: Optional[str] = None
    experiment_id: Optional[str] = None
    related_papers: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Note:
    """
    Core Note class for managing individual research notes in Obsidian format.
    
    Supports:
    - YAML frontmatter
    - Markdown content
    - Bidirectional linking
    - Automatic tagging
    - Version tracking
    """
    
    def __init__(
        self,
        title: str,
        content: str = "",
        file_path: Optional[Path] = None,
        frontmatter: Optional[Dict] = None,
        note_type: NoteType = NoteType.IDEA,
    ):
        self.title = title
        self.content = content
        self.file_path = file_path
        self.note_type = note_type
        
        # Initialize frontmatter
        if frontmatter:
            # Handle dict to dataclass conversion
            if isinstance(frontmatter, dict):
                # Convert string dates to datetime if present
                if 'created' in frontmatter and isinstance(frontmatter['created'], str):
                    try:
                        frontmatter['created'] = datetime.fromisoformat(frontmatter['created'])
                    except ValueError:
                        frontmatter['created'] = datetime.now()
                
                if 'updated' in frontmatter and isinstance(frontmatter['updated'], str):
                    try:
                        frontmatter['updated'] = datetime.fromisoformat(frontmatter['updated'])
                    except ValueError:
                        frontmatter['updated'] = datetime.now()
                
                # Convert type string to enum if needed
                if 'type' in frontmatter and isinstance(frontmatter['type'], str):
                    try:
                        frontmatter['type'] = NoteType(frontmatter['type'])
                    except ValueError:
                        frontmatter['type'] = note_type
                
                # Handle metadata field - convert string to dict if needed
                if 'metadata' in frontmatter and isinstance(frontmatter['metadata'], str):
                    try:
                        import json
                        frontmatter['metadata'] = json.loads(frontmatter['metadata']) if frontmatter['metadata'] else {}
                    except (json.JSONDecodeError, TypeError):
                        frontmatter['metadata'] = {}
                
                # Filter out unknown keys and create dataclass
                valid_keys = {'title', 'created', 'updated', 'tags', 'type', 'status', 'priority', 'author', 'project', 'experiment_id', 'related_papers', 'metadata'}
                filtered_frontmatter = {k: v for k, v in frontmatter.items() if k in valid_keys}
                
                # Don't override title/type if they're already in frontmatter
                if 'title' not in filtered_frontmatter:
                    filtered_frontmatter['title'] = title
                if 'type' not in filtered_frontmatter:
                    filtered_frontmatter['type'] = note_type
                
                self.frontmatter = NoteFrontmatter(**filtered_frontmatter)
            else:
                self.frontmatter = frontmatter
        else:
            self.frontmatter = NoteFrontmatter(
                title=title,
                type=note_type
            )
        
        # Track links and backlinks
        self._links: Set[Link] = set()
        self._backlinks: Set[Link] = set()
        
        # Version tracking
        self._versions: List[Dict] = []
        
    @classmethod
    def from_file(cls, file_path: Path) -> "Note":
        """Load a note from an Obsidian markdown file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Note file not found: {file_path}")
            
        content = file_path.read_text(encoding="utf-8")
        
        # Parse frontmatter and content
        frontmatter, body = cls._parse_markdown(content)
        
        # Extract title from frontmatter or filename
        title = frontmatter.get("title", file_path.stem)
        
        note = cls(
            title=title,
            content=body,
            file_path=file_path,
            frontmatter=frontmatter,
            note_type=NoteType(frontmatter.get("type", "idea"))
        )
        
        # Parse existing links
        note._extract_links_from_content()
        
        return note
    
    @staticmethod
    def _parse_markdown(content: str) -> tuple[Dict, str]:
        """Parse YAML frontmatter and markdown content."""
        if content.startswith("---\n"):
            try:
                _, yaml_content, body = content.split("---\n", 2)
                frontmatter = safe_load(yaml_content) or {}
                return frontmatter, body.strip()
            except ValueError:
                pass
        
        return {}, content
    
    def _extract_links_from_content(self) -> None:
        """Extract Obsidian-style links from content."""
        # Find [[wiki-style links]]
        wiki_links = re.findall(r'\[\[([^\]]+)\]\]', self.content)
        
        for link_text in wiki_links:
            # Handle link with display text: [[target|display]]
            if '|' in link_text:
                target, _ = link_text.split('|', 1)
            else:
                target = link_text
                
            self._links.add(Link(target=target.strip()))
    
    def add_link(self, target: str, relationship: str = "relates_to", confidence: float = 1.0) -> None:
        """Add a link to another note."""
        link = Link(target=target, relationship=relationship, confidence=confidence)
        self._links.add(link)
        
        # Add to content if not already present
        if f"[[{target}]]" not in self.content:
            self.content += f"\n\n**Related**: [[{target}]]"
            
        self._update_timestamp()
        
        # Notify callback if set (for knowledge graph updates)
        if hasattr(self, '_link_added_callback') and self._link_added_callback:
            self._link_added_callback(self, link)
    
    def add_backlink(self, source: str, relationship: str = "relates_to") -> None:
        """Add a backlink from another note."""
        backlink = Link(target=source, relationship=relationship)
        self._backlinks.add(backlink)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the note."""
        if not tag.startswith('#'):
            tag = f"#{tag}"
            
        if tag not in self.frontmatter.tags:
            self.frontmatter.tags.append(tag)
            self._update_timestamp()
    
    def add_tags(self, tags: List[str]) -> None:
        """Add multiple tags to the note."""
        for tag in tags:
            self.add_tag(tag)
    
    def add_section(self, heading: str, content: str, level: int = 2) -> None:
        """Add a new section to the note."""
        heading_prefix = "#" * level
        section = f"\n\n{heading_prefix} {heading}\n\n{content}"
        self.content += section
        self._update_timestamp()
    
    def add_figure(self, figure_path: str, caption: str = "") -> None:
        """Add a figure to the note."""
        figure_embed = f"![[{figure_path}]]"
        if caption:
            figure_embed += f"\n*{caption}*"
            
        self.content += f"\n\n{figure_embed}"
        self._update_timestamp()
    
    def update_content(self, new_content: str) -> None:
        """Update the note content and track the change."""
        self.create_version("content_update")
        self.content = new_content
        self._update_timestamp()
    
    def create_version(self, label: str = "") -> None:
        """Create a version snapshot of the current note."""
        version = {
            "timestamp": datetime.now(),
            "label": label,
            "content": self.content,
            "frontmatter": self.frontmatter.model_dump(),
        }
        self._versions.append(version)
    
    def get_links(self) -> List[Link]:
        """Get all outgoing links."""
        return list(self._links)
    
    def get_backlinks(self) -> List[Link]:
        """Get all incoming links."""
        return list(self._backlinks)
    
    def get_related_notes(self) -> List[str]:
        """Get titles of all related notes."""
        related = set()
        for link in self._links:
            related.add(link.target)
        for link in self._backlinks:
            related.add(link.target)
        return list(related)
    
    def to_markdown(self) -> str:
        """Convert note to Obsidian-compatible markdown format."""
        # Prepare frontmatter
        try:
            # Try pydantic v2 method first
            frontmatter_dict = self.frontmatter.model_dump()
        except AttributeError:
            # Fall back to pydantic v1 method
            frontmatter_dict = self.frontmatter.dict()
        
        # Convert datetime objects to strings and enums to values
        for key, value in frontmatter_dict.items():
            if isinstance(value, datetime):
                frontmatter_dict[key] = value.isoformat()
            elif hasattr(value, 'value'):  # Handle enums
                frontmatter_dict[key] = value.value
        
        # Build markdown content
        frontmatter_yaml = safe_dump(frontmatter_dict)
        
        markdown = f"---\n{frontmatter_yaml}---\n\n{self.content}"
        
        return markdown
    
    def save(self, file_path: Optional[Path] = None) -> None:
        """Save the note to a file."""
        logger = setup_logger(f"note.{self.title}")
        
        try:
            if file_path:
                self.file_path = file_path
            elif not self.file_path:
                error_msg = "No file path specified for saving note"
                logger.error(error_msg, extra={'note_title': self.title})
                raise NoteError(error_msg)
            
            logger.info(f"Saving note to {self.file_path}", extra={
                'note_title': self.title,
                'file_path': str(self.file_path),
                'note_type': self.note_type.value
            })
            
            # Ensure directory exists
            try:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create directory {self.file_path.parent}: {e}")
                raise NoteError(f"Cannot create directory for note: {e}") from e
            
            # Write to file
            markdown_content = self.to_markdown()
            self.file_path.write_text(markdown_content, encoding="utf-8")
            
            # Update timestamp
            self.frontmatter.updated = datetime.now()
            
            logger.info(f"Note saved successfully", extra={
                'note_title': self.title,
                'file_size': len(markdown_content)
            })
            
        except Exception as e:
            logger.error(f"Failed to save note: {e}", exc_info=True, extra={
                'note_title': self.title,
                'file_path': str(self.file_path) if self.file_path else None
            })
            raise NoteError(f"Failed to save note '{self.title}': {e}") from e
    
    def _update_timestamp(self) -> None:
        """Update the modified timestamp."""
        self.frontmatter.updated = datetime.now()
    
    def __str__(self) -> str:
        return f"Note(title='{self.title}', type={self.note_type.value})"
    
    def __repr__(self) -> str:
        return f"Note(title='{self.title}', type={self.note_type.value}, tags={self.frontmatter.tags})"