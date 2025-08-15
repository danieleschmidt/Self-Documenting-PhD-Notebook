"""CLI commands for pipeline guard functionality."""

import asyncio
import json
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from ..pipeline.guard import PipelineGuard, GuardConfig
from ..pipeline.monitor import PipelineMonitor
from ..utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@click.group()
def pipeline():
    """Pipeline guard commands."""
    pass


@pipeline.command()
@click.option("--interval", default=30, help="Check interval in seconds")
@click.option("--max-attempts", default=3, help="Maximum heal attempts per hour")
@click.option("--webhook", multiple=True, help="Notification webhook URLs")
@click.option("--background", is_flag=True, help="Run in background")
def start(interval, max_attempts, webhook, background):
    """Start the pipeline guard."""
    config = GuardConfig(
        check_interval=interval,
        max_heal_attempts=max_attempts,
        notification_webhooks=list(webhook)
    )
    
    guard = PipelineGuard(config)
    
    if background:
        console.print("🚀 Starting pipeline guard in background...")
        # In a real implementation, this would use a proper daemon process
        console.print("❌ Background mode not implemented yet. Use a process manager like systemd or supervisor.")
        return
    
    console.print("🚀 Starting pipeline guard...")
    console.print(f"📊 Check interval: {interval}s")
    console.print(f"🔄 Max heal attempts: {max_attempts}")
    console.print(f"📢 Webhooks: {len(webhook)}")
    
    try:
        asyncio.run(guard.start_monitoring())
    except KeyboardInterrupt:
        console.print("\n🛑 Pipeline guard stopped")


@pipeline.command()
def status():
    """Show pipeline status."""
    console.print("📊 Pipeline Status")
    
    async def get_status():
        monitor = PipelineMonitor()
        pipelines = await monitor.get_pipeline_status()
        return pipelines
    
    pipelines = asyncio.run(get_status())
    
    if not pipelines:
        console.print("No pipelines found")
        return
    
    table = Table(title="Pipeline Status")
    table.add_column("Pipeline ID")
    table.add_column("System")
    table.add_column("State")
    table.add_column("Started")
    table.add_column("Error")
    
    for pipeline_id, status in pipelines.items():
        state_color = {
            "success": "green",
            "running": "yellow", 
            "failed": "red",
            "pending": "blue"
        }.get(status.state, "white")
        
        system = status.metadata.get("system", "unknown")
        error = status.error[:50] + "..." if status.error and len(status.error) > 50 else (status.error or "")
        
        table.add_row(
            pipeline_id,
            system,
            f"[{state_color}]{status.state}[/{state_color}]",
            status.started_at.strftime("%H:%M:%S"),
            error
        )
    
    console.print(table)


@pipeline.command()
@click.argument("pipeline_id")
def logs(pipeline_id):
    """Show logs for a specific pipeline."""
    console.print(f"📋 Logs for {pipeline_id}")
    
    async def get_logs():
        monitor = PipelineMonitor()
        return await monitor.get_pipeline_logs(pipeline_id)
    
    logs = asyncio.run(get_logs())
    
    if not logs:
        console.print("No logs found")
        return
    
    for line in logs[-50:]:  # Show last 50 lines
        console.print(line)


@pipeline.command()
@click.argument("pipeline_id")
def heal(pipeline_id):
    """Manually trigger healing for a pipeline."""
    console.print(f"🔧 Attempting to heal pipeline {pipeline_id}")
    
    async def manual_heal():
        from ..pipeline.detector import FailureDetector, FailureType, FailureAnalysis
        from ..pipeline.healer import SelfHealer
        
        # Create a dummy failure analysis for manual healing
        analysis = FailureAnalysis(
            failure_type=FailureType.UNKNOWN,
            confidence=0.5,
            description="Manual healing attempt",
            suggested_fixes=[],
            error_patterns=[],
            metadata={"pipeline_id": pipeline_id}
        )
        
        healer = SelfHealer()
        success = await healer.heal_pipeline(pipeline_id, analysis)
        return success
    
    success = asyncio.run(manual_heal())
    
    if success:
        console.print("✅ Healing successful")
    else:
        console.print("❌ Healing failed")


@pipeline.command()
def monitor():
    """Show real-time pipeline monitoring."""
    console.print("🔍 Real-time Pipeline Monitor")
    console.print("Press Ctrl+C to stop")
    
    async def monitor_loop():
        monitor = PipelineMonitor()
        
        while True:
            try:
                pipelines = await monitor.get_pipeline_status()
                
                # Clear screen and show status
                console.clear()
                console.print("🔍 Real-time Pipeline Monitor")
                console.print(f"Last updated: {asyncio.get_event_loop().time()}")
                
                if pipelines:
                    table = Table()
                    table.add_column("Pipeline")
                    table.add_column("State")
                    table.add_column("System")
                    
                    for pipeline_id, status in pipelines.items():
                        state_emoji = {
                            "success": "✅",
                            "running": "🔄",
                            "failed": "❌", 
                            "pending": "⏳"
                        }.get(status.state, "❓")
                        
                        table.add_row(
                            pipeline_id,
                            f"{state_emoji} {status.state}",
                            status.metadata.get("system", "unknown")
                        )
                    
                    console.print(table)
                else:
                    console.print("No pipelines found")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                console.print(f"Error: {e}")
                await asyncio.sleep(5)
    
    try:
        asyncio.run(monitor_loop())
    except KeyboardInterrupt:
        console.print("\n🛑 Monitoring stopped")


@pipeline.command()
def config():
    """Show and manage pipeline guard configuration."""
    console.print("⚙️ Pipeline Guard Configuration")
    
    # Show current configuration
    config = GuardConfig()
    
    config_panel = Panel.fit(
        f"""
Check Interval: {config.check_interval}s
Heal Timeout: {config.heal_timeout}s  
Max Heal Attempts: {config.max_heal_attempts}
Notification Webhooks: {len(config.notification_webhooks)}
        """.strip(),
        title="Current Configuration"
    )
    
    console.print(config_panel)


@pipeline.command()
def stats():
    """Show pipeline guard statistics."""
    console.print("📈 Pipeline Guard Statistics")
    
    # This would show actual statistics from a running guard
    console.print("Statistics would be shown here when guard is running")


if __name__ == "__main__":
    pipeline()