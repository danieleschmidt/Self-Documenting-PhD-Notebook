"""Pipeline monitoring and status collection."""

import asyncio
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import get_logger


@dataclass
class PipelineStatus:
    """Pipeline status information."""
    pipeline_id: str
    state: str  # running, success, failed, pending
    started_at: datetime
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    logs: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.metadata is None:
            self.metadata = {}


class PipelineMonitor:
    """Monitors pipeline status across different CI/CD systems."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._supported_systems = {
            "github_actions": self._monitor_github_actions,
            "gitlab_ci": self._monitor_gitlab_ci,
            "jenkins": self._monitor_jenkins,
            "local_git": self._monitor_local_git,
        }
    
    async def get_pipeline_status(self) -> Dict[str, PipelineStatus]:
        """Get status of all monitored pipelines."""
        all_status = {}
        
        for system_name, monitor_func in self._supported_systems.items():
            try:
                status = await monitor_func()
                all_status.update(status)
            except Exception as e:
                self.logger.error(f"Error monitoring {system_name}: {e}")
        
        return all_status
    
    async def _monitor_github_actions(self) -> Dict[str, PipelineStatus]:
        """Monitor GitHub Actions workflows."""
        try:
            # Check if we're in a GitHub repository
            result = await self._run_command("gh workflow list")
            if result.returncode != 0:
                return {}
            
            workflows = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    workflows.append(line.split('\t')[0])
            
            status = {}
            for workflow in workflows:
                runs = await self._run_command(f"gh run list --workflow='{workflow}' --limit=1 --json=status,conclusion,createdAt,id")
                if runs.returncode == 0:
                    run_data = json.loads(runs.stdout)
                    if run_data:
                        run = run_data[0]
                        pipeline_id = f"gh_{workflow}_{run['id']}"
                        state = self._map_github_state(run['status'], run.get('conclusion'))
                        
                        status[pipeline_id] = PipelineStatus(
                            pipeline_id=pipeline_id,
                            state=state,
                            started_at=datetime.fromisoformat(run['createdAt'].replace('Z', '+00:00')),
                            metadata={"system": "github_actions", "workflow": workflow}
                        )
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error monitoring GitHub Actions: {e}")
            return {}
    
    async def _monitor_gitlab_ci(self) -> Dict[str, PipelineStatus]:
        """Monitor GitLab CI pipelines."""
        try:
            # Check if we're in a GitLab repository
            result = await self._run_command("git remote get-url origin")
            if result.returncode != 0 or "gitlab" not in result.stdout.lower():
                return {}
            
            # Get latest pipeline info
            result = await self._run_command("glab ci list --limit=5")
            if result.returncode != 0:
                return {}
            
            status = {}
            # Parse glab output and create status entries
            # This is a simplified implementation
            return status
            
        except Exception as e:
            self.logger.error(f"Error monitoring GitLab CI: {e}")
            return {}
    
    async def _monitor_jenkins(self) -> Dict[str, PipelineStatus]:
        """Monitor Jenkins builds."""
        # Jenkins monitoring would require Jenkins API access
        # This is a placeholder implementation
        return {}
    
    async def _monitor_local_git(self) -> Dict[str, PipelineStatus]:
        """Monitor local git repository for commit validation."""
        try:
            # Check git status
            result = await self._run_command("git status --porcelain")
            has_changes = bool(result.stdout.strip())
            
            # Get latest commit
            commit_result = await self._run_command("git rev-parse HEAD")
            if commit_result.returncode != 0:
                return {}
            
            commit_hash = commit_result.stdout.strip()[:8]
            
            # Simple local validation
            pipeline_id = f"local_git_{commit_hash}"
            state = "pending" if has_changes else "success"
            
            return {
                pipeline_id: PipelineStatus(
                    pipeline_id=pipeline_id,
                    state=state,
                    started_at=datetime.now(),
                    metadata={"system": "local_git", "commit": commit_hash}
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring local git: {e}")
            return {}
    
    async def _run_command(self, command: str) -> subprocess.CompletedProcess:
        """Run a shell command asynchronously."""
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout.decode() if stdout else "",
            stderr.decode() if stderr else ""
        )
    
    def _map_github_state(self, status: str, conclusion: Optional[str]) -> str:
        """Map GitHub Actions status to our standard states."""
        if status == "completed":
            if conclusion == "success":
                return "success"
            else:
                return "failed"
        elif status in ["in_progress", "queued"]:
            return "running"
        else:
            return "pending"
    
    async def get_pipeline_logs(self, pipeline_id: str) -> List[str]:
        """Get logs for a specific pipeline."""
        try:
            if pipeline_id.startswith("gh_"):
                # Extract run ID from pipeline_id
                run_id = pipeline_id.split("_")[-1]
                result = await self._run_command(f"gh run view {run_id} --log")
                if result.returncode == 0:
                    return result.stdout.split('\n')
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting logs for {pipeline_id}: {e}")
            return []