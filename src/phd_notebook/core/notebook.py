"""
Main ResearchNotebook class - the primary interface for the PhD notebook system.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from .vault_manager import VaultManager, VaultConfig
from .note import Note, NoteType
from .knowledge_graph import KnowledgeGraph
from ..agents.base import BaseAgent
from ..utils.exceptions import NotebookError, VaultError, ConfigurationError
from ..monitoring.logging_setup import setup_logger


class ResearchNotebook:
    """
    Main interface for the Self-Documenting PhD Notebook.
    
    Integrates:
    - Vault management
    - AI agents
    - Data connectors
    - Automation workflows
    """
    
    def __init__(
        self,
        vault_path: Union[str, Path] = "~/Documents/PhD_Research",
        author: str = "Researcher",
        institution: str = "",
        field: str = "",
        subfield: str = "",
        expected_duration: int = 5,
        config: Optional[Dict] = None,
    ):
        # Set up logging
        self.logger = setup_logger(f"notebook.{self.__class__.__name__}")
        self.logger.info("Initializing ResearchNotebook", extra={
            'vault_path': str(vault_path),
            'author': author,
            'field': field
        })
        
        try:
            # Validate inputs
            if not author or not author.strip():
                raise ConfigurationError("Author name is required")
            
            # Expand and validate vault path
            self.vault_path = Path(vault_path).expanduser()
            if not self.vault_path.parent.exists():
                self.logger.error(f"Parent directory does not exist: {self.vault_path.parent}")
                raise VaultError(f"Cannot create vault - parent directory does not exist: {self.vault_path.parent}")
            
            # Create vault configuration
            vault_config = VaultConfig(
                name=self.vault_path.name,
                path=self.vault_path,
                author=author.strip(),
                institution=institution.strip() if institution else "",
                field=field.strip() if field else "",
            )
            
            # Initialize core components with error handling
            try:
                self.vault = VaultManager(self.vault_path, vault_config)
                self.logger.info("Vault manager initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize vault manager: {e}", exc_info=True)
                raise VaultError(f"Failed to initialize vault: {e}") from e
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ResearchNotebook: {e}", exc_info=True)
            raise NotebookError(f"Notebook initialization failed: {e}") from e
        self.knowledge_graph = KnowledgeGraph(self.vault)
        
        # Research context
        self.author = author
        self.institution = institution
        self.field = field
        self.subfield = subfield
        self.expected_duration = expected_duration
        
        # AI agents registry
        self._agents: Dict[str, BaseAgent] = {}
        
        # Workflow manager
        from ..workflows.automation import WorkflowManager
        self.workflow_manager = WorkflowManager(self)
        
        # Legacy workflow support
        self._workflows: Dict[str, Any] = {}
        self._active_workflows: List[str] = []
        
        # Data connectors
        self._connectors: Dict[str, Any] = {}
        
        # Configuration
        self.config = config or {}
        
        # Performance optimizations
        try:
            from ..performance.cache_manager import CacheManager
            from ..performance.resource_monitor import ResourceMonitor
            from ..performance.async_processing import AsyncTaskManager
            
            self.cache_manager = CacheManager()
            # Create custom cache with desired settings if needed
            if self.config.get('cache_size', 1000) != 1000:
                self.cache_manager.create_cache(
                    'main',
                    'lru',
                    max_size=self.config.get('cache_size', 1000),
                    ttl=self.config.get('cache_ttl', 3600)
                )
            self.resource_monitor = ResourceMonitor(interval=self.config.get('monitor_interval', 30))
            self.task_manager = AsyncTaskManager(
                max_concurrent_tasks=self.config.get('max_concurrent_tasks', 10)
            )
            
            # Start performance monitoring if enabled
            if self.config.get('enable_monitoring', True):
                self.resource_monitor.start_monitoring()
                self.logger.info("Performance monitoring enabled")
                
        except ImportError as e:
            self.logger.warning(f"Performance modules not available: {e}")
            self.cache_manager = None
            self.resource_monitor = None
            self.task_manager = None
        
        # Global-first features: internationalization and compliance
        try:
            from ..internationalization.localization import get_localization_manager, SupportedLocale
            from ..internationalization.compliance import ComplianceManager, PrivacyRegulation
            from ..internationalization.regional_adapters import RegionalAdapter
            
            # Initialize localization based on config
            self.localization = get_localization_manager()
            locale_config = self.config.get('locale', 'en_US')
            if locale_config in [locale.value for locale in SupportedLocale]:
                self.localization.set_locale(SupportedLocale(locale_config))
            
            # Initialize compliance management
            regulation_config = self.config.get('privacy_regulation', 'gdpr')
            privacy_regulation = PrivacyRegulation.GDPR
            if regulation_config in [reg.value for reg in PrivacyRegulation]:
                privacy_regulation = PrivacyRegulation(regulation_config)
            
            self.compliance = ComplianceManager(privacy_regulation)
            
            # Initialize regional adapter
            self.regional_adapter = RegionalAdapter()
            
            self.logger.info("Global-first features enabled", extra={
                'locale': self.localization.current_locale.value,
                'privacy_regulation': privacy_regulation.value,
                'deployment_region': self.regional_adapter.target_region.value
            })
            
        except ImportError as e:
            self.logger.warning(f"Global features not available: {e}")
            self.localization = None
            self.compliance = None
            self.regional_adapter = None
        
        print(f"✅ Research Notebook initialized at {self.vault_path}")
        print(f"📚 Field: {field} {f'({subfield})' if subfield else ''}")
        print(f"🏛️ Institution: {institution}")
    
    # Note Management
    def create_note(
        self,
        title: str,
        content: str = "",
        note_type: NoteType = NoteType.IDEA,
        tags: List[str] = None,
        template: str = None,
        **kwargs
    ) -> Note:
        """Create a new research note."""
        note = self.vault.create_note(
            title=title,
            content=content,
            note_type=note_type,
            tags=tags,
            template=template
        )
        
        # Update knowledge graph
        self.knowledge_graph.add_note(note)
        
        # Set up callback for link updates
        note._link_added_callback = self._on_link_added
        
        # Trigger any automation workflows
        self._trigger_workflows("note_created", note=note)
        
        return note
    
    def _on_link_added(self, note: 'Note', link: 'Link') -> None:
        """Callback when a link is added to a note - update knowledge graph."""
        # Re-add the note to the knowledge graph to capture new links
        self.knowledge_graph.add_note(note)
    
    def get_note(self, title: str) -> Optional[Note]:
        """Get a note by title."""
        return self.vault.get_note(title)
    
    def search_notes(self, query: str, **filters) -> List[Note]:
        """Search notes with various filters."""
        # Generate cache key based on query and filters
        cache_key = f"search:{hash((query, tuple(sorted(filters.items()))))}"
        
        # Check cache if available
        if hasattr(self, 'cache_manager') and self.cache_manager:
            cached_result = self.cache_manager.get_cache().get(cache_key)
            if cached_result:
                self.logger.debug(f"Cache hit for search query: {query}")
                return cached_result
        
        # Perform search
        results = self.vault.search_notes(query)
        
        # Apply additional filters
        if "note_type" in filters:
            note_type = filters["note_type"]
            if isinstance(note_type, str):
                note_type = NoteType(note_type)
            results = [n for n in results if n.note_type == note_type]
        
        if "tags" in filters:
            tag_filter = set(filters["tags"])
            results = [n for n in results 
                      if tag_filter.intersection(set(n.frontmatter.tags))]
        
        if "date_range" in filters:
            # TODO: Implement date filtering
            pass
        
        # Cache the results
        if hasattr(self, 'cache_manager') and self.cache_manager:
            self.cache_manager.get_cache().set(cache_key, results, ttl=300)  # Cache for 5 minutes
            self.logger.debug(f"Cached search results for query: {query}")
        
        return results
    
    def list_notes(self, note_type: Optional[NoteType] = None, tags: List[str] = None) -> List[Note]:
        """List notes with optional filtering."""
        return self.vault.list_notes(note_type=note_type, tags=tags)
    
    # Experiment Management
    def new_experiment(self, title: str, hypothesis: str = "", **kwargs):
        """Context manager for creating and tracking experiments."""
        return ExperimentContext(self, title, hypothesis, **kwargs)
    
    def get_experiments(self, topic: str = None) -> List[Note]:
        """Get experiment notes, optionally filtered by topic."""
        experiments = self.list_notes(note_type=NoteType.EXPERIMENT)
        
        if topic:
            experiments = [e for e in experiments if topic.lower() in e.content.lower()]
        
        return experiments
    
    # Literature Management
    def add_paper(self, title: str, authors: str = "", doi: str = "", **metadata) -> Note:
        """Add a paper to the literature collection."""
        paper_note = self.create_note(
            title=title,
            note_type=NoteType.LITERATURE,
            tags=["#literature", "#paper"]
        )
        
        # Add metadata to frontmatter
        paper_note.frontmatter.metadata.update({
            "authors": authors,
            "doi": doi,
            **metadata
        })
        
        return paper_note
    
    def get_recent_papers(self, topic: str, days: int = 30) -> List[Note]:
        """Get recent papers related to a topic."""
        # TODO: Implement date-based filtering
        papers = self.list_notes(note_type=NoteType.LITERATURE)
        return [p for p in papers if topic.lower() in p.content.lower()]
    
    # Agent Management
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an AI agent."""
        self._agents[agent.name] = agent
        agent.notebook = self
        print(f"🤖 Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get a registered agent by name."""
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self._agents.keys())
    
    # Data Connector Management
    def connect_sources(self, sources: Dict[str, Any]) -> None:
        """Connect to various data sources."""
        for category, connectors in sources.items():
            self._connectors[category] = connectors
            print(f"🔌 Connected {category}: {list(connectors.keys())}")
    
    # Workflow Management
    def register_workflow(self, workflow: Any) -> None:
        """Register an automation workflow."""
        self._workflows[workflow.name] = workflow
        print(f"⚙️ Registered workflow: {workflow.name}")
    
    def start_workflow(self, name: str) -> None:
        """Start an automation workflow."""
        if name in self._workflows and name not in self._active_workflows:
            self._active_workflows.append(name)
            print(f"▶️ Started workflow: {name}")
    
    def stop_workflow(self, name: str) -> None:
        """Stop an automation workflow."""
        if name in self._active_workflows:
            self._active_workflows.remove(name)
            print(f"⏹️ Stopped workflow: {name}")
    
    def _trigger_workflows(self, event: str, **context) -> None:
        """Trigger workflows based on events."""
        for workflow_name in self._active_workflows:
            workflow = self._workflows.get(workflow_name)
            if workflow and hasattr(workflow, 'on_event'):
                try:
                    workflow.on_event(event, **context)
                except Exception as e:
                    print(f"⚠️ Workflow {workflow_name} error: {e}")
    
    # Research Context Methods
    def get_research_context(self) -> Dict[str, Any]:
        """Get current research context."""
        stats = self.vault.get_vault_stats()
        
        return {
            "field": self.field,
            "subfield": self.subfield,
            "institution": self.institution,
            "vault_stats": stats,
            "active_projects": self.get_active_projects(),
            "recent_experiments": self.get_recent_experiments(),
        }
    
    def get_active_projects(self) -> List[Note]:
        """Get currently active projects."""
        projects = self.list_notes(note_type=NoteType.PROJECT)
        return [p for p in projects if 
                getattr(p.frontmatter, 'status', '') == 'active']
    
    def get_recent_experiments(self, days: int = 30) -> List[Note]:
        """Get recent experiments."""
        # TODO: Implement date filtering
        return self.list_notes(note_type=NoteType.EXPERIMENT)[:10]
    
    def get_field_overview(self, topic: str) -> str:
        """Get an overview of the field related to a topic."""
        # Placeholder - would use AI agents in full implementation
        return f"Overview of {topic} in {self.field}"
    
    def get_research_keywords(self) -> List[str]:
        """Get current research keywords based on notes."""
        # Extract keywords from tags and content
        notes = self.list_notes()
        keywords = set()
        
        for note in notes:
            # Add tags (remove # prefix)
            for tag in note.frontmatter.tags:
                if tag.startswith('#'):
                    keywords.add(tag[1:])
                else:
                    keywords.add(tag)
        
        return list(keywords)
    
    # Utility Methods
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive notebook statistics."""
        vault_stats = self.vault.get_vault_stats()
        
        return {
            **vault_stats,
            "agents": len(self._agents),
            "active_workflows": len(self._active_workflows),
            "connectors": len(self._connectors),
            "research_context": {
                "field": self.field,
                "subfield": self.subfield,
                "institution": self.institution,
            }
        }
    
    def start_auto_documentation(self) -> None:
        """Start automatic documentation processes."""
        print("🚀 Starting auto-documentation...")
        
        # Schedule default workflows
        self.workflow_manager.schedule_workflow("auto_tagging", "daily")
        self.workflow_manager.schedule_workflow("smart_linking", "daily") 
        self.workflow_manager.schedule_workflow("daily_review", "daily")
        
        print("✅ Auto-documentation workflows scheduled")
    
    async def run_workflow(self, name: str, **kwargs):
        """Run a workflow by name."""
        return await self.workflow_manager.run_workflow(name, **kwargs)
    
    def get_workflow_status(self):
        """Get workflow status."""
        return self.workflow_manager.get_workflow_status()
    
    def __repr__(self) -> str:
        return f"ResearchNotebook(field='{self.field}', vault='{self.vault_path}')"


class ExperimentContext:
    """Context manager for experiment tracking."""
    
    def __init__(self, notebook: ResearchNotebook, title: str, hypothesis: str = "", **kwargs):
        self.notebook = notebook
        self.title = title
        self.hypothesis = hypothesis
        self.kwargs = kwargs
        self.experiment_note: Optional[Note] = None
    
    def __enter__(self) -> Note:
        """Start experiment tracking."""
        self.experiment_note = self.notebook.create_note(
            title=self.title,
            note_type=NoteType.EXPERIMENT,
            tags=["#experiment", "#active"],
            template="experiment"
        )
        
        # Add hypothesis if provided
        if self.hypothesis:
            self.experiment_note.frontmatter.metadata["hypothesis"] = self.hypothesis
            self.experiment_note.add_section("Hypothesis", self.hypothesis)
        
        print(f"🧪 Started experiment: {self.title}")
        return self.experiment_note
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End experiment tracking."""
        if self.experiment_note:
            # Remove active tag and add completed tag
            tags = self.experiment_note.frontmatter.tags
            if "#active" in tags:
                tags.remove("#active")
            if "#completed" not in tags:
                tags.append("#completed")
            
            # Update status
            self.experiment_note.frontmatter.status = "completed"
            
            # Save changes
            if self.experiment_note.file_path:
                self.experiment_note.save()
            
            print(f"✅ Completed experiment: {self.title}")