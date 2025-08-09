"""
Comprehensive error handling and recovery mechanisms.
"""

import logging
import traceback
import functools
import asyncio
from datetime import datetime
from typing import Any, Callable, Optional, Dict, Type, List
from pathlib import Path
import json

from .exceptions import (
    PhDNotebookError, ValidationError, ConnectorError, 
    AgentError, WorkflowError
)


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.logger = logging.getLogger("ErrorHandler")
        self.error_log: List[Dict[str, Any]] = []
        self.log_file = log_file
        
        # Setup file logging if specified
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: str = "error"
    ) -> str:
        """Log an error with context information."""
        error_id = f"err_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(error)}"
        
        error_info = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
            "severity": severity
        }
        
        self.error_log.append(error_info)
        
        # Log to file/console
        log_message = f"[{error_id}] {type(error).__name__}: {error}"
        if context:
            log_message += f" (Context: {context})"
        
        if severity == "critical":
            self.logger.critical(log_message)
        elif severity == "error":
            self.logger.error(log_message)
        elif severity == "warning":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        return error_id
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_log:
            return {"total_errors": 0, "recent_errors": []}
        
        return {
            "total_errors": len(self.error_log),
            "recent_errors": self.error_log[-10:],  # Last 10 errors
            "error_types": self._count_error_types(),
            "critical_count": len([e for e in self.error_log if e["severity"] == "critical"])
        }
    
    def _count_error_types(self) -> Dict[str, int]:
        """Count occurrences of each error type."""
        type_counts = {}
        for error in self.error_log:
            error_type = error["error_type"]
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
        return type_counts
    
    def save_error_log(self, file_path: Path) -> None:
        """Save error log to file."""
        try:
            with open(file_path, 'w') as f:
                json.dump({
                    "error_log": self.error_log,
                    "summary": self.get_error_summary()
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save error log: {e}")


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(
    exceptions: tuple = (Exception,),
    return_value: Any = None,
    log_context: Dict[str, Any] = None,
    severity: str = "error"
):
    """Decorator for handling exceptions gracefully."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                context = (log_context or {}).copy()
                context.update({
                    "function": func.__name__,
                    "args": str(args)[:200],  # Limit arg length
                    "kwargs": str(kwargs)[:200]
                })
                
                error_id = error_handler.log_error(e, context, severity)
                
                # Return default value or re-raise based on severity
                if severity == "critical":
                    raise
                return return_value
        return wrapper
    return decorator


def handle_async_errors(
    exceptions: tuple = (Exception,),
    return_value: Any = None,
    log_context: Dict[str, Any] = None,
    severity: str = "error"
):
    """Decorator for handling async function exceptions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                context = (log_context or {}).copy()
                context.update({
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                })
                
                error_id = error_handler.log_error(e, context, severity)
                
                if severity == "critical":
                    raise
                return return_value
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for external API calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).seconds >= self.timeout
    
    def _on_success(self):
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RetryHandler:
    """Retry mechanism with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        exceptions: tuple = (Exception,)
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        break
                    
                    # Exponential backoff
                    wait_time = self.backoff_factor * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    
                    error_handler.log_error(
                        e,
                        {"attempt": attempt + 1, "function": func.__name__},
                        "warning"
                    )
            
            # All retries exhausted
            error_handler.log_error(
                last_exception,
                {"max_retries_exhausted": True, "function": func.__name__},
                "error"
            )
            raise last_exception
        
        return wrapper


class ValidationHandler:
    """Data validation and sanitization."""
    
    @staticmethod
    def validate_file_path(path: Path, must_exist: bool = False) -> Path:
        """Validate and sanitize file path."""
        if not isinstance(path, Path):
            path = Path(path)
        
        # Resolve path and check for directory traversal
        try:
            resolved_path = path.resolve()
        except Exception as e:
            raise ValidationError(f"Invalid path: {path}") from e
        
        # Check if path exists when required
        if must_exist and not resolved_path.exists():
            raise ValidationError(f"Path does not exist: {resolved_path}")
        
        return resolved_path
    
    @staticmethod
    def validate_text_input(text: str, max_length: int = 10000) -> str:
        """Validate and sanitize text input."""
        if not isinstance(text, str):
            raise ValidationError("Input must be a string")
        
        if len(text) > max_length:
            raise ValidationError(f"Text too long: {len(text)} > {max_length}")
        
        # Basic sanitization
        sanitized = text.strip()
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized 
                          if ord(char) >= 32 or char in '\n\r\t')
        
        return sanitized
    
    @staticmethod
    def validate_tags(tags: List[str]) -> List[str]:
        """Validate and clean tags."""
        if not isinstance(tags, list):
            raise ValidationError("Tags must be a list")
        
        cleaned_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
                
            # Clean and normalize tag
            clean_tag = tag.strip().lower()
            
            # Add # prefix if not present
            if clean_tag and not clean_tag.startswith('#'):
                clean_tag = f'#{clean_tag}'
            
            # Validate tag format
            if clean_tag and len(clean_tag) > 1 and clean_tag[1:].replace('-', '').replace('_', '').isalnum():
                cleaned_tags.append(clean_tag)
        
        return list(set(cleaned_tags))  # Remove duplicates


# Convenience functions
def safe_execute(func: Callable, *args, **kwargs) -> tuple[Any, Optional[Exception]]:
    """Safely execute a function and return result and any exception."""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_handler.log_error(e, {"function": func.__name__})
        return None, e


async def safe_execute_async(func: Callable, *args, **kwargs) -> tuple[Any, Optional[Exception]]:
    """Safely execute an async function and return result and any exception."""
    try:
        result = await func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_handler.log_error(e, {"function": func.__name__})
        return None, e