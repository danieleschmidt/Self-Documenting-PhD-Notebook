"""
Comprehensive logging system for the PhD notebook.
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class ResearchLogger:
    """Specialized logger for research activities."""
    
    def __init__(self, name: str, vault_path: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.vault_path = vault_path
        
        # Don't add handlers if already configured
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        self.logger.setLevel(logging.INFO)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # File handler if vault path is provided
        if self.vault_path:
            log_dir = self.vault_path / ".obsidian" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "research.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(JSONFormatter())
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)
            
            # Error log
            error_handler = logging.handlers.RotatingFileHandler(
                log_dir / "errors.log",
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            error_handler.setFormatter(JSONFormatter())
            error_handler.setLevel(logging.ERROR)
            self.logger.addHandler(error_handler)
    
    def log_experiment(self, experiment_id: str, action: str, **metadata) -> None:
        """Log experiment-related activities."""
        self.logger.info(
            f"Experiment {action}: {experiment_id}",
            extra={
                "experiment_id": experiment_id,
                "action": action,
                "category": "experiment",
                **metadata
            }
        )
    
    def log_note_operation(self, note_title: str, operation: str, **metadata) -> None:
        """Log note operations."""
        self.logger.info(
            f"Note {operation}: {note_title}",
            extra={
                "note_title": note_title,
                "operation": operation,
                "category": "note",
                **metadata
            }
        )
    
    def log_agent_activity(self, agent_name: str, activity: str, **metadata) -> None:
        """Log AI agent activities."""
        self.logger.info(
            f"Agent {agent_name}: {activity}",
            extra={
                "agent_name": agent_name,
                "activity": activity,
                "category": "agent",
                **metadata
            }
        )
    
    def log_connector_sync(self, connector_name: str, items_synced: int, **metadata) -> None:
        """Log data connector synchronization."""
        self.logger.info(
            f"Connector {connector_name} synced {items_synced} items",
            extra={
                "connector_name": connector_name,
                "items_synced": items_synced,
                "category": "sync",
                **metadata
            }
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log errors with context."""
        self.logger.error(
            f"Error: {str(error)}",
            exc_info=error,
            extra={
                "error_type": type(error).__name__,
                "context": context or {},
                "category": "error"
            }
        )
    
    def log_security_event(self, event_type: str, details: str, **metadata) -> None:
        """Log security-related events."""
        self.logger.warning(
            f"Security event [{event_type}]: {details}",
            extra={
                "event_type": event_type,
                "details": details,
                "category": "security",
                **metadata
            }
        )
    
    def log_performance(self, operation: str, duration: float, **metadata) -> None:
        """Log performance metrics."""
        self.logger.info(
            f"Performance [{operation}]: {duration:.3f}s",
            extra={
                "operation": operation,
                "duration_seconds": duration,
                "category": "performance",
                **metadata
            }
        )


def setup_logging(
    vault_path: Optional[Path] = None,
    level: str = "INFO",
    enable_json: bool = True
) -> ResearchLogger:
    """
    Setup comprehensive logging for the PhD notebook system.
    
    Args:
        vault_path: Path to the research vault for log storage
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_json: Whether to use JSON formatting for file logs
    
    Returns:
        Configured ResearchLogger instance
    """
    # Configure root logger
    root_logger = logging.getLogger("phd_notebook")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Create research logger
    research_logger = ResearchLogger("phd_notebook", vault_path)
    
    return research_logger


def get_logger(name: str = "phd_notebook") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Performance monitoring decorator
def log_performance(operation_name: str = None):
    """Decorator to log function performance."""
    def decorator(func):
        import time
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger = get_logger()
                logger.info(
                    f"Performance [{operation}]: {duration:.3f}s",
                    extra={
                        "operation": operation,
                        "duration_seconds": duration,
                        "category": "performance",
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger = get_logger()
                logger.error(
                    f"Performance [{operation}]: {duration:.3f}s (FAILED)",
                    exc_info=e,
                    extra={
                        "operation": operation,
                        "duration_seconds": duration,
                        "category": "performance",
                        "success": False,
                        "error": str(e)
                    }
                )
                
                raise
        
        return wrapper
    return decorator