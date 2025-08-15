"""
Main CLI interface for the Self-Documenting PhD Notebook.
"""

import click
from pathlib import Path
from typing import Optional

from ..core.notebook import ResearchNotebook
from ..core.note import NoteType
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from .pipeline_cli import pipeline

console = Console()


@click.group()
@click.version_option()
def main():
    """Self-Documenting PhD Notebook - AI-powered research automation."""
    pass


# Add pipeline commands
main.add_command(pipeline)


@main.command()
@click.argument('name')
@click.option('--path', '-p', default='~/Documents/PhD_Research',
              help='Path for the research vault')
@click.option('--author', '-a', prompt='Your name', help='Author name')
@click.option('--institution', '-i', default='', help='Institution name')
@click.option('--field', '-f', prompt='Research field', help='Research field')
@click.option('--subfield', '-s', default='', help='Research subfield')
def init(name: str, path: str, author: str, institution: str, field: str, subfield: str):
    """Initialize a new PhD research notebook."""
    vault_path = Path(path).expanduser() / name
    
    rprint(f"🚀 Initializing PhD notebook: [bold]{name}[/bold]")
    rprint(f"📍 Location: {vault_path}")
    
    try:
        notebook = ResearchNotebook(
            vault_path=vault_path,
            author=author,
            institution=institution,
            field=field,
            subfield=subfield
        )
        
        rprint("✅ [green]Notebook initialized successfully![/green]")
        rprint(f"📚 Start by opening {vault_path} in Obsidian")
        
    except Exception as e:
        rprint(f"❌ [red]Error initializing notebook: {e}[/red]")


@main.command()
@click.option('--vault', '-v', default='~/Documents/PhD_Research',
              help='Path to the research vault')
@click.argument('title')
@click.option('--type', '-t', type=click.Choice([t.value for t in NoteType]),
              default='idea', help='Type of note')
@click.option('--tags', '-T', help='Tags for the note (comma-separated)')
@click.option('--template', help='Template to use')
def create(vault: str, title: str, type: str, tags: str, template: str):
    """Create a new research note."""
    vault_path = Path(vault).expanduser()
    
    if not vault_path.exists():
        rprint(f"❌ [red]Vault not found at {vault_path}[/red]")
        rprint("Run 'sdpn init' to create a new notebook first.")
        return
    
    try:
        notebook = ResearchNotebook(vault_path=vault_path)
        
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',')]
        
        note = notebook.create_note(
            title=title,
            note_type=NoteType(type),
            tags=tag_list,
            template=template
        )
        
        rprint(f"✅ [green]Created note: {title}[/green]")
        rprint(f"📄 File: {note.file_path}")
        
    except Exception as e:
        rprint(f"❌ [red]Error creating note: {e}[/red]")


@main.command()
@click.option('--vault', '-v', default='~/Documents/PhD_Research',
              help='Path to the research vault')
@click.option('--type', '-t', type=click.Choice([t.value for t in NoteType]),
              help='Filter by note type')
@click.option('--tags', '-T', help='Filter by tags (comma-separated)')
@click.option('--limit', '-l', default=20, help='Maximum number of notes to show')
def list(vault: str, type: str, tags: str, limit: int):
    """List research notes."""
    vault_path = Path(vault).expanduser()
    
    if not vault_path.exists():
        rprint(f"❌ [red]Vault not found at {vault_path}[/red]")
        return
    
    try:
        notebook = ResearchNotebook(vault_path=vault_path)
        
        # Apply filters
        note_type = NoteType(type) if type else None
        tag_list = [t.strip() for t in tags.split(',')] if tags else None
        
        notes = notebook.list_notes(note_type=note_type, tags=tag_list)
        
        if not notes:
            rprint("📝 No notes found matching criteria")
            return
        
        # Create table
        table = Table(title="Research Notes")
        table.add_column("Title", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Tags", style="yellow")
        table.add_column("Updated", style="green")
        
        for note in notes[:limit]:
            tags_str = ', '.join(note.frontmatter.tags[:3])  # Show first 3 tags
            if len(note.frontmatter.tags) > 3:
                tags_str += f" (+{len(note.frontmatter.tags) - 3})"
            
            table.add_row(
                note.title,
                note.note_type.value,
                tags_str,
                note.frontmatter.updated.strftime("%Y-%m-%d")
            )
        
        console.print(table)
        
        if len(notes) > limit:
            rprint(f"📝 Showing {limit} of {len(notes)} notes. Use --limit to see more.")
    
    except Exception as e:
        rprint(f"❌ [red]Error listing notes: {e}[/red]")


@main.command()
@click.option('--vault', '-v', default='~/Documents/PhD_Research',
              help='Path to the research vault')
@click.argument('query')
@click.option('--content', '-c', is_flag=True, help='Search in content too')
def search(vault: str, query: str, content: bool):
    """Search research notes."""
    vault_path = Path(vault).expanduser()
    
    if not vault_path.exists():
        rprint(f"❌ [red]Vault not found at {vault_path}[/red]")
        return
    
    try:
        notebook = ResearchNotebook(vault_path=vault_path)
        results = notebook.search_notes(query, in_content=content)
        
        if not results:
            rprint(f"🔍 No results found for '{query}'")
            return
        
        rprint(f"🔍 Found {len(results)} results for '[bold]{query}[/bold]':")
        
        for note in results:
            rprint(f"📄 [cyan]{note.title}[/cyan] ({note.note_type.value})")
            
            # Show preview if searching in content
            if content and query.lower() in note.content.lower():
                # Find the line containing the query
                lines = note.content.split('\n')
                for line in lines:
                    if query.lower() in line.lower():
                        preview = line.strip()[:100]
                        if len(line) > 100:
                            preview += "..."
                        rprint(f"   📝 {preview}")
                        break
    
    except Exception as e:
        rprint(f"❌ [red]Error searching notes: {e}[/red]")


@main.command()
@click.option('--vault', '-v', default='~/Documents/PhD_Research',
              help='Path to the research vault')
def stats(vault: str):
    """Show notebook statistics."""
    vault_path = Path(vault).expanduser()
    
    if not vault_path.exists():
        rprint(f"❌ [red]Vault not found at {vault_path}[/red]")
        return
    
    try:
        notebook = ResearchNotebook(vault_path=vault_path)
        stats = notebook.get_stats()
        
        # Create statistics table
        table = Table(title="Notebook Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Notes", str(stats["total_notes"]))
        table.add_row("Total Tags", str(stats["total_tags"]))
        table.add_row("Research Field", stats["research_context"]["field"])
        
        if stats["research_context"]["subfield"]:
            table.add_row("Subfield", stats["research_context"]["subfield"])
        
        table.add_row("Institution", stats["research_context"]["institution"])
        table.add_row("Registered Agents", str(stats["agents"]))
        table.add_row("Active Workflows", str(stats["active_workflows"]))
        
        console.print(table)
        
        # Show note types breakdown
        if stats["types"]:
            rprint("\n📊 [bold]Notes by Type:[/bold]")
            for note_type, count in stats["types"].items():
                rprint(f"   {note_type}: {count}")
        
        # Show status breakdown
        if stats["statuses"]:
            rprint("\n📈 [bold]Notes by Status:[/bold]")
            for status, count in stats["statuses"].items():
                rprint(f"   {status}: {count}")
    
    except Exception as e:
        rprint(f"❌ [red]Error getting stats: {e}[/red]")


@main.command()
@click.option('--vault', '-v', default='~/Documents/PhD_Research',
              help='Path to the research vault')
@click.argument('title')
@click.argument('hypothesis', default='')
def experiment(vault: str, title: str, hypothesis: str):
    """Start a new experiment."""
    vault_path = Path(vault).expanduser()
    
    if not vault_path.exists():
        rprint(f"❌ [red]Vault not found at {vault_path}[/red]")
        return
    
    try:
        notebook = ResearchNotebook(vault_path=vault_path)
        
        # Create experiment note
        with notebook.new_experiment(title, hypothesis) as exp:
            rprint(f"🧪 [green]Started experiment: {title}[/green]")
            rprint(f"📄 File: {exp.file_path}")
            
            if hypothesis:
                rprint(f"🔬 Hypothesis: {hypothesis}")
            
            rprint("\n💡 [yellow]Open the note in Obsidian to continue documenting your experiment![/yellow]")
    
    except Exception as e:
        rprint(f"❌ [red]Error creating experiment: {e}[/red]")


@main.command()
@click.option('--vault', '-v', default='~/Documents/PhD_Research',
              help='Path to the research vault')
@click.option('--workflow', '-w', help='Specific workflow to run')
def workflow(vault: str, workflow: str):
    """Run automation workflows."""
    vault_path = Path(vault).expanduser()
    
    if not vault_path.exists():
        rprint(f"❌ [red]Vault not found at {vault_path}[/red]")
        return
    
    try:
        notebook = ResearchNotebook(vault_path=vault_path)
        
        if workflow:
            # Run specific workflow
            import asyncio
            result = asyncio.run(notebook.run_workflow(workflow))
            rprint(f"✅ [green]Workflow '{workflow}' completed[/green]")
            rprint(f"📊 Result: {result}")
        else:
            # Show workflow status
            status = notebook.get_workflow_status()
            rprint("⚙️ [bold]Workflow Status[/bold]")
            rprint(f"📋 Registered: {', '.join(status['registered'])}")
            rprint(f"⏰ Scheduled: {', '.join(status['scheduled'])}")
            rprint(f"📊 Total: {status['total_workflows']} workflows")
    
    except Exception as e:
        rprint(f"❌ [red]Workflow error: {e}[/red]")


@main.command()
@click.option('--vault', '-v', default='~/Documents/PhD_Research',
              help='Path to the research vault')
def graph(vault: str):
    """Show knowledge graph information."""
    vault_path = Path(vault).expanduser()
    
    if not vault_path.exists():
        rprint(f"❌ [red]Vault not found at {vault_path}[/red]")
        return
    
    try:
        notebook = ResearchNotebook(vault_path=vault_path)
        kg = notebook.knowledge_graph
        
        # Get graph statistics
        clusters = kg.find_clusters()
        centrality = kg.calculate_centrality()
        gaps = kg.identify_research_gaps()
        
        rprint(f"📊 [bold]Knowledge Graph Analysis[/bold]")
        rprint(f"🔗 Nodes: {len(kg.nodes)}")
        rprint(f"➡️ Edges: {len(kg.edges)}")
        rprint(f"🎯 Clusters: {len(clusters)}")
        rprint(f"🕳️ Research Gaps: {len(gaps)}")
        
        # Show top central notes
        if centrality:
            rprint(f"\n⭐ [bold]Most Connected Notes:[/bold]")
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            for note_title, score in top_central:
                rprint(f"   📄 {note_title} ({score:.2f})")
        
        # Show research gaps
        if gaps:
            rprint(f"\n🕳️ [bold]Research Gaps Found:[/bold]")
            for gap in gaps[:3]:  # Show first 3 gaps
                rprint(f"   ⚠️ {gap['description']}")
    
    except Exception as e:
        rprint(f"❌ [red]Error analyzing graph: {e}[/red]")


if __name__ == '__main__':
    main()