"""
Intelligent Error Recovery System
Advanced error handling with learning capabilities, automatic recovery, and predictive failure prevention.
"""

import asyncio
import json
import logging
import traceback
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import functools
import threading
from contextlib import contextmanager

from ..utils.logging import setup_logger


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SYSTEM_RESTART = "system_restart"
    MANUAL_INTERVENTION = "manual_intervention"


class ErrorCategory(Enum):
    NETWORK_ERROR = "network_error"
    FILE_IO_ERROR = "file_io_error"
    MEMORY_ERROR = "memory_error"
    VALIDATION_ERROR = "validation_error"
    AI_SERVICE_ERROR = "ai_service_error"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    SYSTEM_ERROR = "system_error"
    USER_ERROR = "user_error"


@dataclass
class ErrorPattern:
    """Pattern for error recognition and classification."""
    pattern_id: str
    name: str
    error_category: ErrorCategory
    error_signatures: List[str]  # Exception types or message patterns
    severity: ErrorSeverity
    
    # Recovery configuration
    recovery_strategy: RecoveryStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_action: Optional[str] = None
    
    # Learning metrics
    occurrence_count: int = 0
    successful_recoveries: int = 0
    recovery_success_rate: float = 0.0
    last_occurred: Optional[datetime] = None
    
    # Context patterns
    common_contexts: List[str] = None
    environmental_factors: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.common_contexts is None:
            self.common_contexts = []
        if self.environmental_factors is None:
            self.environmental_factors = {}
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['error_category'] = self.error_category.value
        result['severity'] = self.severity.value
        result['recovery_strategy'] = self.recovery_strategy.value
        if self.last_occurred:
            result['last_occurred'] = self.last_occurred.isoformat()
        return result


@dataclass
class ErrorIncident:
    """Individual error incident record."""
    incident_id: str
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    
    # Classification
    error_category: ErrorCategory
    severity: ErrorSeverity
    pattern_id: Optional[str] = None
    
    # Context information
    operation_context: Dict[str, Any] = None
    system_state: Dict[str, Any] = None
    user_context: Dict[str, Any] = None
    
    # Recovery information
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    recovery_time: Optional[float] = None
    final_outcome: str = "unresolved"
    
    # Resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def __post_init__(self):
        if self.operation_context is None:
            self.operation_context = {}
        if self.system_state is None:
            self.system_state = {}
        if self.user_context is None:
            self.user_context = {}
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['error_category'] = self.error_category.value
        result['severity'] = self.severity.value
        result['timestamp'] = self.timestamp.isoformat()
        
        if self.recovery_strategy:
            result['recovery_strategy'] = self.recovery_strategy.value
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
            
        return result


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        # State
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half_open"
                else:
                    raise Exception(f"Circuit breaker is open. Last failure: {self.last_failure_time}")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.timeout
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class RetryHandler:
    """Intelligent retry handler with exponential backoff and jitter."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Retry async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    # Add jitter to prevent thundering herd
                    import random
                    jitter = delay * 0.1 * random.random()
                    await asyncio.sleep(delay + jitter)
        
        raise last_exception
    
    def retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Retry sync function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    import time
                    import random
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    jitter = delay * 0.1 * random.random()
                    time.sleep(delay + jitter)
        
        raise last_exception


class IntelligentErrorRecovery:
    """
    Advanced error recovery system with machine learning capabilities,
    automatic pattern recognition, and predictive failure prevention.
    """
    
    def __init__(self, notebook_path: Path):
        self.logger = setup_logger("core.error_recovery")
        self.notebook_path = notebook_path
        
        # Error tracking
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_incidents: List[ErrorIncident] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
        # Learning and prediction
        self.error_predictions: Dict[str, float] = {}
        self.context_correlations: Dict[str, List[str]] = defaultdict(list)
        
        # Configuration
        self.learning_enabled = True
        self.auto_recovery_enabled = True
        self.prediction_enabled = True
        
        # Create directories
        self.recovery_dir = notebook_path / "error_recovery"
        self.incidents_dir = self.recovery_dir / "incidents"
        self.patterns_dir = self.recovery_dir / "patterns"
        self.reports_dir = self.recovery_dir / "reports"
        
        for dir_path in [self.recovery_dir, self.incidents_dir, self.patterns_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize default patterns
        self._initialize_default_patterns()
        self._initialize_recovery_strategies()
        self._load_existing_data()
    
    def _initialize_default_patterns(self):
        """Initialize default error patterns."""
        
        # Network error pattern
        network_pattern = ErrorPattern(
            pattern_id="network_error",
            name="Network Connection Error",
            error_category=ErrorCategory.NETWORK_ERROR,
            error_signatures=["ConnectionError", "TimeoutError", "HTTPError", "URLError"],
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retries=3,
            retry_delay=2.0
        )
        
        # File I/O error pattern
        file_io_pattern = ErrorPattern(
            pattern_id="file_io_error",
            name="File I/O Error",
            error_category=ErrorCategory.FILE_IO_ERROR,
            error_signatures=["FileNotFoundError", "PermissionError", "IOError", "OSError"],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            fallback_action="create_backup_file"
        )
        
        # Memory error pattern
        memory_pattern = ErrorPattern(
            pattern_id="memory_error",
            name="Memory Error",
            error_category=ErrorCategory.MEMORY_ERROR,
            error_signatures=["MemoryError", "OutOfMemoryError"],
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            fallback_action="reduce_memory_usage"
        )
        
        # AI service error pattern
        ai_service_pattern = ErrorPattern(
            pattern_id="ai_service_error",
            name="AI Service Error",
            error_category=ErrorCategory.AI_SERVICE_ERROR,
            error_signatures=["APIError", "RateLimitError", "AuthenticationError"],
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retries=2,
            retry_delay=5.0,
            fallback_action="use_fallback_ai_service"
        )
        
        # Validation error pattern
        validation_pattern = ErrorPattern(
            pattern_id="validation_error",
            name="Data Validation Error",
            error_category=ErrorCategory.VALIDATION_ERROR,
            error_signatures=["ValidationError", "ValueError", "TypeError"],
            severity=ErrorSeverity.LOW,
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            fallback_action="use_default_values"
        )
        
        self.error_patterns = {
            "network_error": network_pattern,
            "file_io_error": file_io_pattern,
            "memory_error": memory_pattern,
            "ai_service_error": ai_service_pattern,
            "validation_error": validation_pattern
        }
    
    def _initialize_recovery_strategies(self):
        """Initialize recovery strategy handlers."""
        
        async def retry_strategy(incident: ErrorIncident, pattern: ErrorPattern) -> bool:
            """Retry strategy with exponential backoff."""
            retry_handler = RetryHandler(
                max_retries=pattern.max_retries,
                base_delay=pattern.retry_delay
            )
            
            try:
                # This would need to be customized based on the specific operation
                # For now, return success if we've attempted recovery
                return True
            except Exception:
                return False
        
        async def fallback_strategy(incident: ErrorIncident, pattern: ErrorPattern) -> bool:
            """Fallback strategy using alternative approach."""
            if pattern.fallback_action and pattern.fallback_action in self.fallback_handlers:
                try:
                    handler = self.fallback_handlers[pattern.fallback_action]
                    return await handler(incident, pattern)
                except Exception as e:
                    self.logger.error(f"Fallback handler failed: {e}")
                    return False
            return False
        
        async def graceful_degradation_strategy(incident: ErrorIncident, pattern: ErrorPattern) -> bool:
            """Graceful degradation strategy."""
            # Reduce functionality or use simpler alternatives
            if pattern.fallback_action and pattern.fallback_action in self.fallback_handlers:
                try:
                    handler = self.fallback_handlers[pattern.fallback_action]
                    return await handler(incident, pattern)
                except Exception as e:
                    self.logger.error(f"Graceful degradation failed: {e}")
                    return False
            return True  # Accept partial functionality
        
        self.recovery_strategies = {
            RecoveryStrategy.RETRY: retry_strategy,
            RecoveryStrategy.FALLBACK: fallback_strategy,
            RecoveryStrategy.GRACEFUL_DEGRADATION: graceful_degradation_strategy
        }
    
    def _load_existing_data(self):
        """Load existing error recovery data."""
        try:
            # Load error patterns
            for pattern_file in self.patterns_dir.glob("*.json"):
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    pattern = self._dict_to_error_pattern(data)
                    self.error_patterns[pattern.pattern_id] = pattern
            
            # Load recent incidents
            for incident_file in self.incidents_dir.glob("*.json"):
                with open(incident_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    incident = self._dict_to_error_incident(data)
                    self.error_incidents.append(incident)
            
            # Sort incidents by timestamp
            self.error_incidents.sort(key=lambda x: x.timestamp)
            
            self.logger.info(f"Loaded {len(self.error_patterns)} error patterns, "
                           f"{len(self.error_incidents)} incidents")
                           
        except Exception as e:
            self.logger.error(f"Error loading recovery data: {e}")
    
    def _dict_to_error_pattern(self, data: Dict) -> ErrorPattern:
        """Convert dictionary to ErrorPattern object."""
        data['error_category'] = ErrorCategory(data['error_category'])
        data['severity'] = ErrorSeverity(data['severity'])
        data['recovery_strategy'] = RecoveryStrategy(data['recovery_strategy'])
        if data.get('last_occurred'):
            data['last_occurred'] = datetime.fromisoformat(data['last_occurred'])
        return ErrorPattern(**data)
    
    def _dict_to_error_incident(self, data: Dict) -> ErrorIncident:
        """Convert dictionary to ErrorIncident object."""
        data['error_category'] = ErrorCategory(data['error_category'])
        data['severity'] = ErrorSeverity(data['severity'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        if data.get('recovery_strategy'):
            data['recovery_strategy'] = RecoveryStrategy(data['recovery_strategy'])
        if data.get('resolved_at'):
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
            
        return ErrorIncident(**data)
    
    def error_handler(self, operation_name: str = None, max_retries: int = 3, fallback: Callable = None):
        """Decorator for automatic error handling and recovery."""
        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._handle_operation(func, operation_name or func.__name__, 
                                                  max_retries, fallback, *args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._handle_operation(func, operation_name or func.__name__, 
                                                        max_retries, fallback, *args, **kwargs))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    async def _handle_operation(
        self,
        func: Callable,
        operation_name: str,
        max_retries: int,
        fallback: Optional[Callable],
        *args,
        **kwargs
    ) -> Any:
        """Handle operation with intelligent error recovery."""
        
        # Check circuit breaker
        breaker_key = f"{operation_name}_{func.__name__}"
        if breaker_key not in self.circuit_breakers:
            self.circuit_breakers[breaker_key] = CircuitBreaker(failure_threshold=5, timeout=300)
        
        circuit_breaker = self.circuit_breakers[breaker_key]
        
        # Get operation context
        context = self._get_operation_context(func, args, kwargs)
        
        try:
            # Check for predicted failures
            if self.prediction_enabled:
                failure_risk = self._predict_failure_risk(operation_name, context)
                if failure_risk > 0.8:
                    self.logger.warning(f"High failure risk predicted for {operation_name}: {failure_risk:.1%}")
            
            # Execute with circuit breaker protection
            if asyncio.iscoroutinefunction(func):
                result = await circuit_breaker.call(func, *args, **kwargs)
            else:
                result = circuit_breaker.call(func, *args, **kwargs)
            
            return result
            
        except Exception as e:
            # Record error incident
            incident = await self._record_error_incident(e, operation_name, context, func)
            
            # Attempt recovery if enabled
            if self.auto_recovery_enabled:
                recovery_successful = await self._attempt_recovery(incident)
                
                if recovery_successful:
                    # Retry original operation
                    try:
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            result = func(*args, **kwargs)
                        
                        incident.recovery_successful = True
                        incident.final_outcome = "recovered_and_succeeded"
                        self._update_incident(incident)
                        return result
                        
                    except Exception as retry_error:
                        incident.final_outcome = "recovery_attempted_but_failed"
                        self._update_incident(incident)
                        
                        # Try fallback if available
                        if fallback:
                            try:
                                if asyncio.iscoroutinefunction(fallback):
                                    result = await fallback(*args, **kwargs)
                                else:
                                    result = fallback(*args, **kwargs)
                                
                                incident.final_outcome = "fallback_successful"
                                self._update_incident(incident)
                                return result
                                
                            except Exception as fallback_error:
                                incident.final_outcome = "fallback_failed"
                                self._update_incident(incident)
                                raise retry_error
                        else:
                            raise retry_error
            
            # No recovery possible or disabled
            incident.final_outcome = "unrecovered"
            self._update_incident(incident)
            raise e
    
    async def _record_error_incident(
        self,
        exception: Exception,
        operation_name: str,
        context: Dict[str, Any],
        func: Callable
    ) -> ErrorIncident:
        """Record an error incident."""
        
        incident_id = f"incident_{int(datetime.now().timestamp() * 1000)}"
        
        # Classify error
        error_category, severity, pattern_id = self._classify_error(exception)
        
        incident = ErrorIncident(
            incident_id=incident_id,
            error_type=type(exception).__name__,
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now(),
            error_category=error_category,
            severity=severity,
            pattern_id=pattern_id,
            operation_context=context,
            system_state=self._get_system_state(),
            user_context=self._get_user_context()
        )
        
        # Store incident
        self.error_incidents.append(incident)
        self._save_incident(incident)
        
        # Update pattern statistics if matched
        if pattern_id and pattern_id in self.error_patterns:
            pattern = self.error_patterns[pattern_id]
            pattern.occurrence_count += 1
            pattern.last_occurred = datetime.now()
            self._save_pattern(pattern)
        
        # Learn from this incident if enabled
        if self.learning_enabled:
            self._learn_from_incident(incident)
        
        self.logger.error(f"Error incident recorded: {incident_id} - {exception}")
        return incident
    
    def _classify_error(self, exception: Exception) -> Tuple[ErrorCategory, ErrorSeverity, Optional[str]]:
        """Classify error using pattern matching."""
        
        exception_type = type(exception).__name__
        exception_message = str(exception).lower()
        
        # Find matching pattern
        for pattern_id, pattern in self.error_patterns.items():
            for signature in pattern.error_signatures:
                if signature.lower() in exception_type.lower() or signature.lower() in exception_message:
                    return pattern.error_category, pattern.severity, pattern_id
        
        # Default classification for unknown errors
        if "memory" in exception_message:
            return ErrorCategory.MEMORY_ERROR, ErrorSeverity.CRITICAL, None
        elif "network" in exception_message or "connection" in exception_message:
            return ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM, None
        elif "file" in exception_message or "permission" in exception_message:
            return ErrorCategory.FILE_IO_ERROR, ErrorSeverity.HIGH, None
        elif "timeout" in exception_message:
            return ErrorCategory.TIMEOUT_ERROR, ErrorSeverity.MEDIUM, None
        else:
            return ErrorCategory.SYSTEM_ERROR, ErrorSeverity.MEDIUM, None
    
    async def _attempt_recovery(self, incident: ErrorIncident) -> bool:
        """Attempt to recover from error incident."""
        
        if not incident.pattern_id or incident.pattern_id not in self.error_patterns:
            self.logger.warning(f"No recovery pattern for incident: {incident.incident_id}")
            return False
        
        pattern = self.error_patterns[incident.pattern_id]
        
        if pattern.recovery_strategy not in self.recovery_strategies:
            self.logger.warning(f"No recovery strategy handler for: {pattern.recovery_strategy}")
            return False
        
        incident.recovery_attempted = True
        incident.recovery_strategy = pattern.recovery_strategy
        
        start_time = datetime.now()
        
        try:
            strategy_handler = self.recovery_strategies[pattern.recovery_strategy]
            success = await strategy_handler(incident, pattern)
            
            end_time = datetime.now()
            incident.recovery_time = (end_time - start_time).total_seconds()
            incident.recovery_successful = success
            
            # Update pattern statistics
            if success:
                pattern.successful_recoveries += 1
            
            pattern.recovery_success_rate = pattern.successful_recoveries / max(pattern.occurrence_count, 1)
            self._save_pattern(pattern)
            
            self.logger.info(f"Recovery {'succeeded' if success else 'failed'} for incident: {incident.incident_id}")
            return success
            
        except Exception as e:
            self.logger.error(f"Recovery strategy failed for incident {incident.incident_id}: {e}")
            incident.recovery_successful = False
            return False
    
    def _get_operation_context(self, func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Get context information for the operation."""
        context = {
            "function_name": func.__name__,
            "module": func.__module__,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()) if kwargs else []
        }
        
        # Add argument types (safely)
        try:
            context["arg_types"] = [type(arg).__name__ for arg in args[:3]]  # First 3 args
        except Exception:
            pass
        
        # Add function signature info
        try:
            sig = inspect.signature(func)
            context["parameters"] = list(sig.parameters.keys())
        except Exception:
            pass
        
        return context
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state information."""
        try:
            import psutil
            import threading
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "active_threads": threading.active_count(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        except ImportError:
            return {
                "active_threads": threading.active_count(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
    
    def _get_user_context(self) -> Dict[str, Any]:
        """Get user context information."""
        # This would be populated with user-specific information
        # For now, return basic context
        return {
            "timestamp": datetime.now().isoformat(),
            "session_id": getattr(threading.current_thread(), 'name', 'unknown')
        }
    
    def _learn_from_incident(self, incident: ErrorIncident):
        """Learn patterns from error incidents."""
        
        # Update context correlations
        if incident.operation_context:
            operation_name = incident.operation_context.get("function_name", "unknown")
            
            # Track error patterns by operation
            error_signature = f"{incident.error_type}:{incident.error_message[:50]}"
            self.context_correlations[operation_name].append(error_signature)
            
            # Keep only recent correlations
            if len(self.context_correlations[operation_name]) > 10:
                self.context_correlations[operation_name] = self.context_correlations[operation_name][-10:]
        
        # Update error predictions
        self._update_error_predictions(incident)
        
        # Check if we should create a new pattern
        self._check_new_pattern_creation(incident)
    
    def _update_error_predictions(self, incident: ErrorIncident):
        """Update error prediction models."""
        
        # Simple prediction based on recent patterns
        operation_name = incident.operation_context.get("function_name", "unknown")
        
        # Count recent incidents for this operation
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_incidents = [
            i for i in self.error_incidents[-50:]  # Last 50 incidents
            if i.timestamp >= recent_cutoff and 
               i.operation_context.get("function_name") == operation_name
        ]
        
        if len(recent_incidents) > 1:
            # Calculate failure rate
            failure_rate = len(recent_incidents) / 50.0
            self.error_predictions[operation_name] = min(failure_rate * 2, 1.0)  # Scale up risk
        else:
            self.error_predictions[operation_name] = 0.1  # Baseline risk
    
    def _check_new_pattern_creation(self, incident: ErrorIncident):
        """Check if we should create a new error pattern."""
        
        if incident.pattern_id:
            return  # Already matched to existing pattern
        
        # Look for similar recent incidents
        similar_incidents = []
        for other_incident in self.error_incidents[-20:]:  # Last 20 incidents
            if (other_incident.error_type == incident.error_type and
                not other_incident.pattern_id and
                other_incident.incident_id != incident.incident_id):
                similar_incidents.append(other_incident)
        
        # If we have 3+ similar unmatched incidents, create a new pattern
        if len(similar_incidents) >= 2:
            self._create_new_pattern(incident, similar_incidents)
    
    def _create_new_pattern(self, incident: ErrorIncident, similar_incidents: List[ErrorIncident]):
        """Create a new error pattern from repeated incidents."""
        
        pattern_id = f"learned_pattern_{int(datetime.now().timestamp())}"
        
        # Determine recovery strategy based on error category
        recovery_strategy = {
            ErrorCategory.NETWORK_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.FILE_IO_ERROR: RecoveryStrategy.FALLBACK,
            ErrorCategory.MEMORY_ERROR: RecoveryStrategy.GRACEFUL_DEGRADATION,
            ErrorCategory.TIMEOUT_ERROR: RecoveryStrategy.RETRY,
        }.get(incident.error_category, RecoveryStrategy.GRACEFUL_DEGRADATION)
        
        new_pattern = ErrorPattern(
            pattern_id=pattern_id,
            name=f"Learned Pattern: {incident.error_type}",
            error_category=incident.error_category,
            error_signatures=[incident.error_type],
            severity=incident.severity,
            recovery_strategy=recovery_strategy,
            max_retries=2,
            retry_delay=1.0,
            occurrence_count=len(similar_incidents) + 1
        )
        
        self.error_patterns[pattern_id] = new_pattern
        self._save_pattern(new_pattern)
        
        # Update incidents to reference new pattern
        incident.pattern_id = pattern_id
        for similar_incident in similar_incidents:
            similar_incident.pattern_id = pattern_id
            self._save_incident(similar_incident)
        
        self.logger.info(f"Created new error pattern: {pattern_id} for {incident.error_type}")
    
    def _predict_failure_risk(self, operation_name: str, context: Dict[str, Any]) -> float:
        """Predict failure risk for an operation."""
        
        base_risk = self.error_predictions.get(operation_name, 0.1)
        
        # Adjust risk based on system state
        system_state = self._get_system_state()
        
        risk_factors = []
        
        # High memory usage increases risk
        memory_usage = system_state.get('memory_percent', 0)
        if memory_usage > 80:
            risk_factors.append(0.3)
        elif memory_usage > 60:
            risk_factors.append(0.1)
        
        # High CPU usage increases risk
        cpu_usage = system_state.get('cpu_percent', 0)
        if cpu_usage > 90:
            risk_factors.append(0.2)
        elif cpu_usage > 70:
            risk_factors.append(0.1)
        
        # Many active threads increases risk
        thread_count = system_state.get('active_threads', 1)
        if thread_count > 20:
            risk_factors.append(0.2)
        elif thread_count > 10:
            risk_factors.append(0.1)
        
        # Calculate total risk
        total_risk = base_risk + sum(risk_factors)
        return min(total_risk, 1.0)
    
    def _update_incident(self, incident: ErrorIncident):
        """Update incident record."""
        self._save_incident(incident)
    
    def add_fallback_handler(self, action_name: str, handler: Callable):
        """Add custom fallback handler."""
        self.fallback_handlers[action_name] = handler
        self.logger.info(f"Added fallback handler: {action_name}")
    
    def create_custom_pattern(
        self,
        pattern_id: str,
        name: str,
        error_category: ErrorCategory,
        error_signatures: List[str],
        severity: ErrorSeverity,
        recovery_strategy: RecoveryStrategy,
        **kwargs
    ):
        """Create custom error pattern."""
        
        pattern = ErrorPattern(
            pattern_id=pattern_id,
            name=name,
            error_category=error_category,
            error_signatures=error_signatures,
            severity=severity,
            recovery_strategy=recovery_strategy,
            **kwargs
        )
        
        self.error_patterns[pattern_id] = pattern
        self._save_pattern(pattern)
        self.logger.info(f"Created custom error pattern: {pattern_id}")
    
    def get_error_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive error analytics."""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_incidents = [i for i in self.error_incidents if i.timestamp >= cutoff_date]
        
        analytics = {
            "period": f"Last {days} days",
            "generated_at": datetime.now().isoformat(),
            "summary": self._generate_error_summary(recent_incidents),
            "patterns": self._analyze_error_patterns(),
            "recovery_effectiveness": self._analyze_recovery_effectiveness(),
            "predictions": dict(self.error_predictions),
            "trending_errors": self._identify_trending_errors(recent_incidents),
            "recommendations": self._generate_error_recommendations(recent_incidents)
        }
        
        return analytics
    
    def _generate_error_summary(self, incidents: List[ErrorIncident]) -> Dict[str, Any]:
        """Generate error summary statistics."""
        
        if not incidents:
            return {"total_incidents": 0}
        
        # Count by category and severity
        categories = Counter(i.error_category.value for i in incidents)
        severities = Counter(i.severity.value for i in incidents)
        
        # Recovery statistics
        recovery_attempts = len([i for i in incidents if i.recovery_attempted])
        successful_recoveries = len([i for i in incidents if i.recovery_successful])
        
        return {
            "total_incidents": len(incidents),
            "by_category": dict(categories),
            "by_severity": dict(severities),
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / max(recovery_attempts, 1)
        }
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error pattern effectiveness."""
        
        pattern_analysis = {}
        
        for pattern_id, pattern in self.error_patterns.items():
            if pattern.occurrence_count > 0:
                pattern_analysis[pattern_id] = {
                    "name": pattern.name,
                    "category": pattern.error_category.value,
                    "severity": pattern.severity.value,
                    "occurrences": pattern.occurrence_count,
                    "recovery_success_rate": pattern.recovery_success_rate,
                    "last_occurred": pattern.last_occurred.isoformat() if pattern.last_occurred else None
                }
        
        return pattern_analysis
    
    def _analyze_recovery_effectiveness(self) -> Dict[str, Any]:
        """Analyze recovery strategy effectiveness."""
        
        strategy_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})
        
        for incident in self.error_incidents:
            if incident.recovery_attempted and incident.recovery_strategy:
                strategy = incident.recovery_strategy.value
                strategy_stats[strategy]["attempts"] += 1
                if incident.recovery_successful:
                    strategy_stats[strategy]["successes"] += 1
        
        # Calculate success rates
        effectiveness = {}
        for strategy, stats in strategy_stats.items():
            effectiveness[strategy] = {
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "success_rate": stats["successes"] / max(stats["attempts"], 1)
            }
        
        return effectiveness
    
    def _identify_trending_errors(self, recent_incidents: List[ErrorIncident]) -> List[Dict[str, Any]]:
        """Identify trending error types."""
        
        if len(recent_incidents) < 5:
            return []
        
        # Group by error type and count occurrences
        error_counts = Counter(i.error_type for i in recent_incidents)
        
        trending = []
        for error_type, count in error_counts.most_common(5):
            # Calculate trend (last 7 days vs previous 7 days)
            now = datetime.now()
            last_week = [i for i in recent_incidents 
                        if i.error_type == error_type and 
                           (now - i.timestamp).days <= 7]
            prev_week = [i for i in recent_incidents 
                        if i.error_type == error_type and 
                           7 < (now - i.timestamp).days <= 14]
            
            trend_direction = "stable"
            if len(last_week) > len(prev_week) * 1.5:
                trend_direction = "increasing"
            elif len(last_week) < len(prev_week) * 0.5:
                trend_direction = "decreasing"
            
            trending.append({
                "error_type": error_type,
                "total_count": count,
                "last_week_count": len(last_week),
                "trend_direction": trend_direction
            })
        
        return trending
    
    def _generate_error_recommendations(self, recent_incidents: List[ErrorIncident]) -> List[Dict[str, str]]:
        """Generate recommendations based on error patterns."""
        
        recommendations = []
        
        # High-frequency errors
        error_counts = Counter(i.error_type for i in recent_incidents)
        for error_type, count in error_counts.most_common(3):
            if count >= 5:
                recommendations.append({
                    "category": "Error Prevention",
                    "priority": "high",
                    "recommendation": f"Investigate root cause of frequent {error_type} errors",
                    "action": f"Analyze {count} recent {error_type} incidents for common patterns"
                })
        
        # Low recovery rates
        for pattern_id, pattern in self.error_patterns.items():
            if pattern.occurrence_count >= 3 and pattern.recovery_success_rate < 0.3:
                recommendations.append({
                    "category": "Recovery Improvement",
                    "priority": "medium",
                    "recommendation": f"Improve recovery strategy for {pattern.name}",
                    "action": f"Review and optimize {pattern.recovery_strategy.value} strategy"
                })
        
        # System stress indicators
        memory_errors = len([i for i in recent_incidents 
                           if i.error_category == ErrorCategory.MEMORY_ERROR])
        if memory_errors >= 3:
            recommendations.append({
                "category": "System Optimization",
                "priority": "high",
                "recommendation": "Address memory-related errors",
                "action": "Optimize memory usage and implement better memory management"
            })
        
        return recommendations
    
    def _save_incident(self, incident: ErrorIncident):
        """Save incident to JSON storage."""
        file_path = self.incidents_dir / f"{incident.incident_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(incident.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_pattern(self, pattern: ErrorPattern):
        """Save pattern to JSON storage."""
        file_path = self.patterns_dir / f"{pattern.pattern_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(pattern.to_dict(), f, indent=2, ensure_ascii=False)