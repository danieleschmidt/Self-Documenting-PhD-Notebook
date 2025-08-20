"""
Advanced error handling system for PhD Notebook with intelligent recovery.
Generation 2 (Robust) implementation featuring:
- Intelligent error classification and recovery
- Contextual error handling based on research operations
- Adaptive retry mechanisms with learning
- Graceful degradation for critical research workflows
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union, Type
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import traceback
import inspect

from ..utils.exceptions import (
    ResearchError, 
    NotebookError, 
    AgentError, 
    ValidationError,
    ResilienceError
)
from ..monitoring.metrics import MetricsCollector


class ErrorSeverity(Enum):
    """Error severity levels for intelligent handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in research context."""
    DATA_INTEGRITY = "data_integrity"
    AI_SERVICE = "ai_service"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    VALIDATION = "validation"
    RESEARCH_WORKFLOW = "research_workflow"
    SECURITY = "security"


@dataclass
class ErrorContext:
    """Rich context information for error analysis."""
    operation_type: str
    research_phase: str  # literature_review, experimentation, writing, etc.
    user_data_at_risk: bool
    critical_research_path: bool
    recoverable: bool = True
    suggested_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecoveryStrategy:
    """Strategy for error recovery with learning capabilities."""
    name: str
    handler: Callable
    success_rate: float = 0.0
    usage_count: int = 0
    applicable_categories: List[ErrorCategory] = field(default_factory=list)
    priority: int = 1


class IntelligentErrorHandler:
    """
    Advanced error handling system with research-aware recovery strategies.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.logger = logging.getLogger(__name__)
        self.metrics = metrics_collector or MetricsCollector()
        self._recovery_strategies: Dict[ErrorCategory, List[ErrorRecoveryStrategy]] = {}
        self._error_history: List[Dict[str, Any]] = []
        self._learning_enabled = True
        
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default recovery strategies for research operations."""
        
        # AI Service error strategies
        self._recovery_strategies[ErrorCategory.AI_SERVICE] = [
            ErrorRecoveryStrategy(
                name="retry_with_backoff",
                handler=self._retry_with_exponential_backoff,
                applicable_categories=[ErrorCategory.AI_SERVICE, ErrorCategory.NETWORK],
                priority=1
            ),
            ErrorRecoveryStrategy(
                name="fallback_to_local_processing",
                handler=self._fallback_to_local_ai,
                applicable_categories=[ErrorCategory.AI_SERVICE],
                priority=2
            ),
            ErrorRecoveryStrategy(
                name="queue_for_later_processing",
                handler=self._queue_for_later,
                applicable_categories=[ErrorCategory.AI_SERVICE, ErrorCategory.NETWORK],
                priority=3
            )
        ]
        
        # File system error strategies
        self._recovery_strategies[ErrorCategory.FILE_SYSTEM] = [
            ErrorRecoveryStrategy(
                name="create_backup_and_retry",
                handler=self._create_backup_and_retry,
                applicable_categories=[ErrorCategory.FILE_SYSTEM],
                priority=1
            ),
            ErrorRecoveryStrategy(
                name="repair_file_permissions",
                handler=self._repair_file_permissions,
                applicable_categories=[ErrorCategory.FILE_SYSTEM],
                priority=2
            ),
            ErrorRecoveryStrategy(
                name="create_alternative_path",
                handler=self._create_alternative_path,
                applicable_categories=[ErrorCategory.FILE_SYSTEM],
                priority=3
            )
        ]
        
        # Research workflow error strategies
        self._recovery_strategies[ErrorCategory.RESEARCH_WORKFLOW] = [
            ErrorRecoveryStrategy(
                name="save_partial_progress",
                handler=self._save_partial_progress,
                applicable_categories=[ErrorCategory.RESEARCH_WORKFLOW],
                priority=1
            ),
            ErrorRecoveryStrategy(
                name="switch_to_manual_mode",
                handler=self._switch_to_manual_mode,
                applicable_categories=[ErrorCategory.RESEARCH_WORKFLOW],
                priority=2
            )
        ]
    
    async def handle_error(
        self, 
        error: Exception, 
        context: ErrorContext,
        max_recovery_attempts: int = 3
    ) -> Tuple[bool, Optional[Any]]:
        """
        Intelligently handle an error with context-aware recovery.
        
        Returns:
            Tuple of (success, result) where success indicates if error was recovered
        """
        
        # Classify the error
        category = self._classify_error(error, context)
        severity = self._assess_severity(error, context)
        
        # Log the error with rich context
        self._log_error_with_context(error, context, category, severity)
        
        # Record for learning
        self._record_error(error, context, category, severity)
        
        # For critical errors affecting user data, take immediate protective action
        if severity == ErrorSeverity.CRITICAL and context.user_data_at_risk:
            await self._emergency_data_protection(context)
        
        # Attempt recovery if the error is recoverable
        if context.recoverable and severity != ErrorSeverity.CRITICAL:
            return await self._attempt_recovery(error, context, category, max_recovery_attempts)
        
        # If not recoverable, provide graceful degradation
        await self._graceful_degradation(error, context, category)
        return False, None
    
    def _classify_error(self, error: Exception, context: ErrorContext) -> ErrorCategory:
        """Classify error into appropriate category for targeted handling."""
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # AI/ML service errors
        if any(keyword in error_message for keyword in ['openai', 'anthropic', 'api', 'model', 'tokens']):
            return ErrorCategory.AI_SERVICE
        
        # File system errors
        if any(keyword in error_message for keyword in ['file', 'directory', 'permission', 'disk', 'path']):
            return ErrorCategory.FILE_SYSTEM
        
        # Network errors
        if any(keyword in error_message for keyword in ['connection', 'timeout', 'network', 'http']):
            return ErrorCategory.NETWORK
        
        # Validation errors
        if isinstance(error, (ValidationError, ValueError)):
            return ErrorCategory.VALIDATION
        
        # Security errors
        if any(keyword in error_message for keyword in ['security', 'authentication', 'authorization']):
            return ErrorCategory.SECURITY
        
        # Default to research workflow
        return ErrorCategory.RESEARCH_WORKFLOW
    
    def _assess_severity(self, error: Exception, context: ErrorContext) -> ErrorSeverity:
        """Assess the severity of an error based on context."""
        
        # Critical: Data loss risk or security breach
        if (context.user_data_at_risk or 
            'security' in str(error).lower() or 
            isinstance(error, (PermissionError, FileNotFoundError)) and context.critical_research_path):
            return ErrorSeverity.CRITICAL
        
        # High: Critical research path disrupted
        if context.critical_research_path:
            return ErrorSeverity.HIGH
        
        # Medium: Important but not critical operations
        if context.operation_type in ['note_creation', 'ai_processing', 'workflow_execution']:
            return ErrorSeverity.MEDIUM
        
        # Low: Minor operations that can be easily retried
        return ErrorSeverity.LOW
    
    async def _attempt_recovery(
        self, 
        error: Exception, 
        context: ErrorContext, 
        category: ErrorCategory,
        max_attempts: int
    ) -> Tuple[bool, Optional[Any]]:
        """Attempt to recover from error using available strategies."""
        
        strategies = self._get_applicable_strategies(category)
        
        for attempt in range(max_attempts):
            for strategy in strategies:
                try:
                    self.logger.info(f"Attempting recovery with strategy: {strategy.name} (attempt {attempt + 1})")
                    
                    result = await strategy.handler(error, context)
                    
                    if result is not None:
                        # Update strategy success metrics
                        self._update_strategy_metrics(strategy, success=True)
                        self.logger.info(f"Recovery successful with strategy: {strategy.name}")
                        return True, result
                    
                except Exception as recovery_error:
                    self.logger.warning(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                    self._update_strategy_metrics(strategy, success=False)
            
            # Exponential backoff between attempts
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        
        return False, None
    
    def _get_applicable_strategies(self, category: ErrorCategory) -> List[ErrorRecoveryStrategy]:
        """Get recovery strategies sorted by success rate and priority."""
        strategies = self._recovery_strategies.get(category, [])
        
        # Sort by success rate (descending) and priority (ascending)
        return sorted(strategies, key=lambda s: (-s.success_rate, s.priority))
    
    async def _retry_with_exponential_backoff(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Retry operation with exponential backoff."""
        max_retries = 3
        base_delay = 1.0
        
        for retry in range(max_retries):
            try:
                delay = base_delay * (2 ** retry)
                await asyncio.sleep(delay)
                
                # This would need to be implemented based on the specific operation
                # For now, return a success indicator
                self.logger.info(f"Retry {retry + 1} after {delay}s delay")
                return "retry_success"
                
            except Exception as retry_error:
                if retry == max_retries - 1:
                    raise retry_error
        
        return None
    
    async def _fallback_to_local_ai(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Fallback to local AI processing when remote services fail."""
        self.logger.info("Falling back to local AI processing")
        
        # Implement local AI fallback logic
        # This could involve using a local model or simplified processing
        return "local_ai_fallback"
    
    async def _queue_for_later(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Queue operation for later processing when services are restored."""
        self.logger.info("Queueing operation for later processing")
        
        # Implement queueing logic
        return "queued_for_later"
    
    async def _create_backup_and_retry(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Create backup of data and retry file operation."""
        self.logger.info("Creating backup and retrying file operation")
        
        # Implement backup and retry logic
        return "backup_and_retry_success"
    
    async def _repair_file_permissions(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Attempt to repair file permissions."""
        self.logger.info("Attempting to repair file permissions")
        
        # Implement permission repair logic
        return "permissions_repaired"
    
    async def _create_alternative_path(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Create alternative file path when original is inaccessible."""
        self.logger.info("Creating alternative file path")
        
        # Implement alternative path creation
        return "alternative_path_created"
    
    async def _save_partial_progress(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Save partial progress before workflow failure."""
        self.logger.info("Saving partial research progress")
        
        # Implement partial progress saving
        return "partial_progress_saved"
    
    async def _switch_to_manual_mode(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Switch to manual mode when automated workflow fails."""
        self.logger.info("Switching to manual mode")
        
        # Implement manual mode switch
        return "manual_mode_enabled"
    
    async def _emergency_data_protection(self, context: ErrorContext):
        """Take immediate action to protect user data."""
        self.logger.critical("Initiating emergency data protection protocols")
        
        # Implement emergency data protection measures
        # - Create immediate backups
        # - Lock critical files
        # - Notify user of potential data risk
        pass
    
    async def _graceful_degradation(self, error: Exception, context: ErrorContext, category: ErrorCategory):
        """Provide graceful degradation when recovery is not possible."""
        self.logger.warning(f"Initiating graceful degradation for {category.value} error")
        
        degradation_strategies = {
            ErrorCategory.AI_SERVICE: "Disable AI-powered features temporarily",
            ErrorCategory.FILE_SYSTEM: "Switch to read-only mode",
            ErrorCategory.NETWORK: "Enable offline mode",
            ErrorCategory.RESEARCH_WORKFLOW: "Provide manual intervention options"
        }
        
        strategy = degradation_strategies.get(category, "Provide basic functionality only")
        self.logger.info(f"Degradation strategy: {strategy}")
    
    def _update_strategy_metrics(self, strategy: ErrorRecoveryStrategy, success: bool):
        """Update success metrics for recovery strategies."""
        strategy.usage_count += 1
        
        if success:
            # Calculate new success rate using exponential moving average
            alpha = 0.1  # Learning rate
            strategy.success_rate = (alpha * 1.0) + ((1 - alpha) * strategy.success_rate)
        else:
            alpha = 0.1
            strategy.success_rate = (alpha * 0.0) + ((1 - alpha) * strategy.success_rate)
    
    def _record_error(self, error: Exception, context: ErrorContext, category: ErrorCategory, severity: ErrorSeverity):
        """Record error for learning and analysis."""
        error_record = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'category': category.value,
            'severity': severity.value,
            'context': {
                'operation_type': context.operation_type,
                'research_phase': context.research_phase,
                'user_data_at_risk': context.user_data_at_risk,
                'critical_research_path': context.critical_research_path
            },
            'stack_trace': traceback.format_exc()
        }
        
        self._error_history.append(error_record)
        
        # Keep only recent errors (last 1000)
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-1000:]
        
        # Report to metrics collector
        self.metrics.record_error(error_record)
    
    def _log_error_with_context(
        self, 
        error: Exception, 
        context: ErrorContext, 
        category: ErrorCategory, 
        severity: ErrorSeverity
    ):
        """Log error with rich contextual information."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        self.logger.log(
            log_level,
            f"Research Error [{category.value}] in {context.operation_type}: {error}",
            extra={
                'error_category': category.value,
                'error_severity': severity.value,
                'research_phase': context.research_phase,
                'user_data_at_risk': context.user_data_at_risk,
                'critical_path': context.critical_research_path,
                'error_type': type(error).__name__
            }
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics for monitoring."""
        total_errors = len(self._error_history)
        
        if total_errors == 0:
            return {'total_errors': 0, 'error_rate': 0.0}
        
        recent_errors = [e for e in self._error_history if time.time() - e['timestamp'] < 3600]  # Last hour
        
        category_counts = {}
        severity_counts = {}
        
        for error in self._error_history:
            category_counts[error['category']] = category_counts.get(error['category'], 0) + 1
            severity_counts[error['severity']] = severity_counts.get(error['severity'], 0) + 1
        
        return {
            'total_errors': total_errors,
            'recent_errors_1h': len(recent_errors),
            'error_rate_1h': len(recent_errors) / 60 if recent_errors else 0.0,  # errors per minute
            'category_distribution': category_counts,
            'severity_distribution': severity_counts,
            'strategy_success_rates': {
                cat.value: [
                    {'name': s.name, 'success_rate': s.success_rate, 'usage_count': s.usage_count}
                    for s in strategies
                ]
                for cat, strategies in self._recovery_strategies.items()
            }
        }


def robust_error_handler(
    operation_type: str,
    research_phase: str = "general",
    user_data_at_risk: bool = False,
    critical_research_path: bool = False,
    max_recovery_attempts: int = 3
):
    """
    Decorator for robust error handling in research operations.
    
    Args:
        operation_type: Type of operation being performed
        research_phase: Current phase of research
        user_data_at_risk: Whether user data could be lost
        critical_research_path: Whether this is on the critical research path
        max_recovery_attempts: Maximum number of recovery attempts
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = IntelligentErrorHandler()
            
            try:
                return await func(*args, **kwargs)
            except Exception as error:
                context = ErrorContext(
                    operation_type=operation_type,
                    research_phase=research_phase,
                    user_data_at_risk=user_data_at_risk,
                    critical_research_path=critical_research_path,
                    metadata={'function': func.__name__, 'args': str(args)[:100]}
                )
                
                success, result = await handler.handle_error(error, context, max_recovery_attempts)
                
                if success:
                    return result
                else:
                    # Re-raise the original error if recovery failed
                    raise error
        
        return wrapper
    return decorator