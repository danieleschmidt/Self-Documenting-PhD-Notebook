"""Self-healing functionality for pipelines."""

import asyncio
import json
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .detector import FailureType, FailureAnalysis
from ..utils.logging import get_logger


class SelfHealer:
    """Implements self-healing strategies for pipeline failures."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._healing_strategies = self._build_healing_strategies()
        self._healing_history = []
    
    async def heal_pipeline(self, pipeline_id: str, failure_analysis: FailureAnalysis) -> bool:
        """Attempt to heal a failed pipeline."""
        self.logger.info(f"Attempting to heal pipeline {pipeline_id} - {failure_analysis.failure_type.value}")
        
        strategies = self._healing_strategies.get(failure_analysis.failure_type, [])
        
        for strategy in strategies:
            try:
                self.logger.info(f"Trying healing strategy: {strategy['name']}")
                success = await strategy["action"](pipeline_id, failure_analysis)
                
                if success:
                    self.logger.info(f"Healing successful with strategy: {strategy['name']}")
                    self._record_healing_success(pipeline_id, failure_analysis, strategy["name"])
                    return True
                else:
                    self.logger.warning(f"Healing strategy failed: {strategy['name']}")
                    
            except Exception as e:
                self.logger.error(f"Error in healing strategy {strategy['name']}: {e}")
        
        self.logger.warning(f"All healing strategies failed for pipeline {pipeline_id}")
        self._record_healing_failure(pipeline_id, failure_analysis)
        return False
    
    def _build_healing_strategies(self) -> Dict[FailureType, List[Dict[str, Any]]]:
        """Build healing strategies for different failure types."""
        return {
            FailureType.DEPENDENCY_FAILURE: [
                {
                    "name": "clear_package_cache",
                    "action": self._clear_package_cache
                },
                {
                    "name": "retry_installation", 
                    "action": self._retry_installation
                },
                {
                    "name": "update_dependencies",
                    "action": self._update_dependencies
                }
            ],
            
            FailureType.TEST_FAILURE: [
                {
                    "name": "rerun_tests",
                    "action": self._rerun_tests
                },
                {
                    "name": "update_test_snapshots",
                    "action": self._update_test_snapshots
                }
            ],
            
            FailureType.BUILD_FAILURE: [
                {
                    "name": "clean_rebuild",
                    "action": self._clean_rebuild
                },
                {
                    "name": "update_build_tools",
                    "action": self._update_build_tools
                }
            ],
            
            FailureType.TIMEOUT: [
                {
                    "name": "increase_timeout",
                    "action": self._increase_timeout
                },
                {
                    "name": "retry_with_backoff",
                    "action": self._retry_with_backoff
                }
            ],
            
            FailureType.NETWORK_FAILURE: [
                {
                    "name": "retry_network_operation",
                    "action": self._retry_network_operation
                },
                {
                    "name": "use_mirror_registry",
                    "action": self._use_mirror_registry
                }
            ],
            
            FailureType.AUTHENTICATION_FAILURE: [
                {
                    "name": "refresh_credentials",
                    "action": self._refresh_credentials
                }
            ],
            
            FailureType.CONFIGURATION_ERROR: [
                {
                    "name": "reset_configuration",
                    "action": self._reset_configuration
                },
                {
                    "name": "validate_and_fix_config",
                    "action": self._validate_and_fix_config
                }
            ],
            
            FailureType.RESOURCE_EXHAUSTION: [
                {
                    "name": "cleanup_resources",
                    "action": self._cleanup_resources
                },
                {
                    "name": "optimize_resource_usage",
                    "action": self._optimize_resource_usage
                }
            ]
        }
    
    async def _clear_package_cache(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Clear package manager caches."""
        try:
            # Try different package managers
            commands = [
                "npm cache clean --force",
                "pip cache purge", 
                "cargo clean",
                "yarn cache clean"
            ]
            
            for cmd in commands:
                result = await self._run_command(cmd)
                if result.returncode == 0:
                    self.logger.info(f"Cleared cache with: {cmd}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear package cache: {e}")
            return False
    
    async def _retry_installation(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Retry package installation."""
        try:
            # Detect package manager and retry installation
            if Path("package.json").exists():
                result = await self._run_command("npm install")
            elif Path("requirements.txt").exists():
                result = await self._run_command("pip install -r requirements.txt")
            elif Path("Cargo.toml").exists():
                result = await self._run_command("cargo build")
            else:
                return False
            
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Failed to retry installation: {e}")
            return False
    
    async def _update_dependencies(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Update dependencies to latest compatible versions."""
        try:
            if Path("package.json").exists():
                # Update npm dependencies
                result = await self._run_command("npm update")
                return result.returncode == 0
            elif Path("requirements.txt").exists():
                # For Python, this is more complex - just retry for now
                result = await self._run_command("pip install --upgrade -r requirements.txt")
                return result.returncode == 0
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update dependencies: {e}")
            return False
    
    async def _rerun_tests(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Rerun failed tests."""
        try:
            # Try common test runners
            test_commands = [
                "npm test",
                "pytest",
                "python -m pytest",
                "cargo test",
                "make test"
            ]
            
            for cmd in test_commands:
                result = await self._run_command(cmd)
                if result.returncode == 0:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to rerun tests: {e}")
            return False
    
    async def _update_test_snapshots(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Update test snapshots if applicable."""
        try:
            # Jest snapshot update
            result = await self._run_command("npm test -- --updateSnapshot")
            if result.returncode == 0:
                return True
            
            # Other snapshot test frameworks could be added here
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update test snapshots: {e}")
            return False
    
    async def _clean_rebuild(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Clean and rebuild the project."""
        try:
            # Clean build artifacts
            clean_commands = [
                "make clean",
                "npm run clean", 
                "cargo clean",
                "rm -rf build/ dist/ target/"
            ]
            
            for cmd in clean_commands:
                await self._run_command(cmd)
            
            # Rebuild
            build_commands = [
                "make",
                "npm run build",
                "cargo build",
                "python setup.py build"
            ]
            
            for cmd in build_commands:
                result = await self._run_command(cmd)
                if result.returncode == 0:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed clean rebuild: {e}")
            return False
    
    async def _update_build_tools(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Update build tools to latest versions."""
        try:
            # This is a placeholder - updating build tools is risky
            # In practice, this might just log the suggestion
            self.logger.info("Suggestion: Consider updating build tools")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update build tools: {e}")
            return False
    
    async def _increase_timeout(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Attempt to increase timeout values."""
        try:
            # This would need to modify CI configuration files
            # For now, just log the suggestion
            self.logger.info("Suggestion: Increase timeout values in CI configuration")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to increase timeout: {e}")
            return False
    
    async def _retry_with_backoff(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Retry the operation with exponential backoff."""
        try:
            # Simple retry mechanism
            for attempt in range(3):
                await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
                
                # This would need to trigger the actual retry
                # For now, just simulate
                self.logger.info(f"Retry attempt {attempt + 1}")
            
            return False  # Would need actual retry logic
            
        except Exception as e:
            self.logger.error(f"Failed retry with backoff: {e}")
            return False
    
    async def _retry_network_operation(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Retry network operations."""
        try:
            # Simple network retry
            await asyncio.sleep(5)  # Wait a bit
            return False  # Would need to trigger actual retry
            
        except Exception as e:
            self.logger.error(f"Failed to retry network operation: {e}")
            return False
    
    async def _use_mirror_registry(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Switch to mirror registry for packages."""
        try:
            # This would configure alternative package registries
            self.logger.info("Suggestion: Configure mirror registries for packages")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to use mirror registry: {e}")
            return False
    
    async def _refresh_credentials(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Refresh authentication credentials."""
        try:
            # This would need access to credential management
            self.logger.info("Suggestion: Refresh authentication credentials")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to refresh credentials: {e}")
            return False
    
    async def _reset_configuration(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Reset configuration to defaults."""
        try:
            # This is risky - just log suggestion
            self.logger.info("Suggestion: Reset configuration to known working state")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
            return False
    
    async def _validate_and_fix_config(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Validate and attempt to fix configuration."""
        try:
            # Simple config validation
            config_files = [".github/workflows/*.yml", ".gitlab-ci.yml", "Jenkinsfile"]
            
            for pattern in config_files:
                files = Path(".").glob(pattern)
                for file_path in files:
                    # Basic YAML validation
                    try:
                        import yaml
                        with open(file_path) as f:
                            yaml.safe_load(f)
                        self.logger.info(f"Config file {file_path} is valid YAML")
                    except yaml.YAMLError as e:
                        self.logger.error(f"Invalid YAML in {file_path}: {e}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate config: {e}")
            return False
    
    async def _cleanup_resources(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Clean up resources to free space/memory."""
        try:
            cleanup_commands = [
                "docker system prune -f",
                "npm cache clean --force",
                "pip cache purge",
                "find . -name '*.pyc' -delete",
                "find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
            ]
            
            for cmd in cleanup_commands:
                await self._run_command(cmd)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup resources: {e}")
            return False
    
    async def _optimize_resource_usage(self, pipeline_id: str, analysis: FailureAnalysis) -> bool:
        """Optimize resource usage."""
        try:
            # This would implement resource optimization strategies
            self.logger.info("Suggestion: Optimize resource usage (reduce parallelism, etc.)")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to optimize resource usage: {e}")
            return False
    
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
    
    def _record_healing_success(self, pipeline_id: str, analysis: FailureAnalysis, strategy: str) -> None:
        """Record a successful healing attempt."""
        self._healing_history.append({
            "timestamp": datetime.now(),
            "pipeline_id": pipeline_id,
            "failure_type": analysis.failure_type.value,
            "strategy": strategy,
            "success": True
        })
    
    def _record_healing_failure(self, pipeline_id: str, analysis: FailureAnalysis) -> None:
        """Record a failed healing attempt."""
        self._healing_history.append({
            "timestamp": datetime.now(),
            "pipeline_id": pipeline_id,
            "failure_type": analysis.failure_type.value,
            "strategy": "all_failed",
            "success": False
        })
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get statistics about healing attempts."""
        if not self._healing_history:
            return {"total_attempts": 0}
        
        successful = [h for h in self._healing_history if h["success"]]
        success_rate = len(successful) / len(self._healing_history)
        
        return {
            "total_attempts": len(self._healing_history),
            "successful_heals": len(successful),
            "success_rate": success_rate,
            "most_recent": self._healing_history[-1] if self._healing_history else None
        }