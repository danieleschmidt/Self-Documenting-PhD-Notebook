"""Failure detection and analysis."""

import re
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .monitor import PipelineStatus
from ..utils.logging import get_logger


class FailureType(Enum):
    """Types of pipeline failures."""
    DEPENDENCY_FAILURE = "dependency_failure"
    TEST_FAILURE = "test_failure"
    BUILD_FAILURE = "build_failure"
    DEPLOYMENT_FAILURE = "deployment_failure"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    AUTHENTICATION_FAILURE = "auth_failure"
    CONFIGURATION_ERROR = "config_error"
    UNKNOWN = "unknown"


@dataclass
class FailureAnalysis:
    """Analysis of a pipeline failure."""
    failure_type: FailureType
    confidence: float  # 0.0 to 1.0
    description: str
    suggested_fixes: List[str]
    error_patterns: List[str]
    metadata: Dict[str, Any]


class FailureDetector:
    """Detects and analyzes pipeline failures."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._failure_patterns = self._build_failure_patterns()
    
    async def analyze_failure(self, pipeline_id: str, status: PipelineStatus) -> FailureAnalysis:
        """Analyze a pipeline failure and determine its type."""
        from .monitor import PipelineMonitor
        
        monitor = PipelineMonitor()
        logs = await monitor.get_pipeline_logs(pipeline_id)
        
        # Combine error message and logs for analysis
        all_text = (status.error or "") + "\n" + "\n".join(logs)
        
        return self._analyze_error_text(all_text, status)
    
    def _analyze_error_text(self, error_text: str, status: PipelineStatus) -> FailureAnalysis:
        """Analyze error text to determine failure type."""
        error_text_lower = error_text.lower()
        
        # Check each failure pattern
        for failure_type, patterns in self._failure_patterns.items():
            for pattern_info in patterns:
                if re.search(pattern_info["pattern"], error_text_lower):
                    return FailureAnalysis(
                        failure_type=failure_type,
                        confidence=pattern_info["confidence"],
                        description=pattern_info["description"],
                        suggested_fixes=pattern_info["fixes"],
                        error_patterns=[pattern_info["pattern"]],
                        metadata={"pipeline_id": status.pipeline_id}
                    )
        
        # Default to unknown failure
        return FailureAnalysis(
            failure_type=FailureType.UNKNOWN,
            confidence=0.5,
            description="Unknown failure type",
            suggested_fixes=["Check logs manually", "Retry pipeline"],
            error_patterns=[],
            metadata={"pipeline_id": status.pipeline_id}
        )
    
    def _build_failure_patterns(self) -> Dict[FailureType, List[Dict[str, Any]]]:
        """Build patterns for detecting different failure types."""
        return {
            FailureType.DEPENDENCY_FAILURE: [
                {
                    "pattern": r"(npm|pip|cargo|gem) install.*failed",
                    "confidence": 0.9,
                    "description": "Package installation failed",
                    "fixes": ["Clear package cache", "Update package versions", "Check package availability"]
                },
                {
                    "pattern": r"could not resolve dependencies",
                    "confidence": 0.85,
                    "description": "Dependency resolution failed",
                    "fixes": ["Update dependency versions", "Check for conflicting dependencies"]
                },
                {
                    "pattern": r"package not found",
                    "confidence": 0.8,
                    "description": "Required package not found",
                    "fixes": ["Check package name spelling", "Verify package exists", "Update package source"]
                }
            ],
            
            FailureType.TEST_FAILURE: [
                {
                    "pattern": r"(\d+) (tests? failed|failing)",
                    "confidence": 0.95,
                    "description": "Unit tests failed",
                    "fixes": ["Review test failures", "Update test expectations", "Fix code bugs"]
                },
                {
                    "pattern": r"assertion error|test.*failed",
                    "confidence": 0.9,
                    "description": "Test assertion failed",
                    "fixes": ["Check test logic", "Update expected values", "Fix implementation"]
                }
            ],
            
            FailureType.BUILD_FAILURE: [
                {
                    "pattern": r"compilation failed|build failed",
                    "confidence": 0.9,
                    "description": "Build/compilation failed",
                    "fixes": ["Check syntax errors", "Verify build configuration", "Update build tools"]
                },
                {
                    "pattern": r"syntax error|parse error",
                    "confidence": 0.85,
                    "description": "Syntax or parse error",
                    "fixes": ["Fix syntax errors", "Check file encoding", "Validate configuration files"]
                }
            ],
            
            FailureType.DEPLOYMENT_FAILURE: [
                {
                    "pattern": r"deployment failed|deploy.*error",
                    "confidence": 0.9,
                    "description": "Deployment failed",
                    "fixes": ["Check deployment configuration", "Verify target environment", "Review permissions"]
                }
            ],
            
            FailureType.TIMEOUT: [
                {
                    "pattern": r"timeout|timed out|exceeded.*time",
                    "confidence": 0.9,
                    "description": "Operation timed out",
                    "fixes": ["Increase timeout values", "Optimize slow operations", "Check for hanging processes"]
                }
            ],
            
            FailureType.RESOURCE_EXHAUSTION: [
                {
                    "pattern": r"out of memory|memory.*exceeded",
                    "confidence": 0.9,
                    "description": "Out of memory",
                    "fixes": ["Increase memory allocation", "Optimize memory usage", "Use larger instance"]
                },
                {
                    "pattern": r"no space left|disk.*full",
                    "confidence": 0.9,
                    "description": "Disk space exhausted",
                    "fixes": ["Clean up disk space", "Increase storage", "Remove temporary files"]
                }
            ],
            
            FailureType.NETWORK_FAILURE: [
                {
                    "pattern": r"connection refused|network.*error|dns.*error",
                    "confidence": 0.85,
                    "description": "Network connectivity issue",
                    "fixes": ["Check network connectivity", "Verify DNS resolution", "Retry with backoff"]
                },
                {
                    "pattern": r"ssl.*error|certificate.*error",
                    "confidence": 0.8,
                    "description": "SSL/Certificate error",
                    "fixes": ["Update certificates", "Check SSL configuration", "Verify certificate validity"]
                }
            ],
            
            FailureType.AUTHENTICATION_FAILURE: [
                {
                    "pattern": r"authentication failed|unauthorized|access denied",
                    "confidence": 0.9,
                    "description": "Authentication or authorization failed",
                    "fixes": ["Check credentials", "Verify permissions", "Update access tokens"]
                }
            ],
            
            FailureType.CONFIGURATION_ERROR: [
                {
                    "pattern": r"configuration.*error|invalid.*config",
                    "confidence": 0.8,
                    "description": "Configuration error",
                    "fixes": ["Validate configuration files", "Check environment variables", "Review settings"]
                }
            ]
        }
    
    def get_failure_statistics(self, failures: List[FailureAnalysis]) -> Dict[str, Any]:
        """Get statistics about detected failures."""
        if not failures:
            return {}
        
        type_counts = {}
        for failure in failures:
            failure_type = failure.failure_type.value
            type_counts[failure_type] = type_counts.get(failure_type, 0) + 1
        
        avg_confidence = sum(f.confidence for f in failures) / len(failures)
        
        return {
            "total_failures": len(failures),
            "failure_types": type_counts,
            "average_confidence": avg_confidence,
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }